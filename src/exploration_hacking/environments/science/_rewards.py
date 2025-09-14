from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import verifiers as vf

from exploration_hacking.environments.science._rubrics import CappedToolRubric


class LengthPenaltyConfig(BaseModel):
    min_length: int = 500
    max_length: int = 2000
    under_length_penalty_per_token: float = 0.002
    over_length_penalty_per_token: float = 0.002


class ScienceRewardConfig(BaseModel):
    accuracy_reward_weight: float = 1.0
    tool_use_reward_weight: float = 0.0
    tool_use_reward_cap: float = 1.0
    format_penalty: float = 5.0
    completion_length_penalty: LengthPenaltyConfig = Field(
        default_factory=LengthPenaltyConfig
    )
    response_length_penalty: LengthPenaltyConfig = Field(
        default_factory=LengthPenaltyConfig
    )
    tokenizer_name: str = "willcb/Qwen3-14B"


def accuracy(completion, answer, prompt, state, parser):
    response = completion[-1]["content"]
    parsed = parser.parse(response)
    return 1.0 if parsed.answer == answer else 0.0


def format_penalty_func(completion, answer, prompt, state, parser):
    final_response = completion[-1]["content"]
    parsed = parser.parse(final_response)

    penalty = 0.0

    if not hasattr(parsed, "answer") or not parsed.answer:
        penalty += 1.0

    # TODO: also penalize incorrect thinking tokens (including in earlier responses).

    return penalty


def get_length_penalty_funcs(
    min_length: int,
    max_length: int,
    tokenizer_name: str,
    per_turn: bool = False,
):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def get_length(completion):
        if per_turn:
            assistant_responses = [
                message for message in completion if message["role"] == "assistant"
            ]
            all_tokens = [
                tokenizer.apply_chat_template(response)
                for response in assistant_responses
            ]
            lengths = [len(tokens) for tokens in all_tokens]
            length = sum(lengths) / len(
                lengths
            )  # average length of assistant responses
        else:
            tokens = tokenizer.apply_chat_template(completion)
            length = len(tokens)  # total length of entire trajectory
        return length

    def under_length_penalty_func(completion, answer, prompt, state, parser):
        length = get_length(completion)
        if length < min_length:
            return min_length - length
        return 0.0

    def over_length_penalty_func(completion, answer, prompt, state, parser):
        length = get_length(completion)
        if length > max_length:
            return length - max_length
        return 0.0

    return [under_length_penalty_func, over_length_penalty_func]


def add_length_penalties(
    funcs: list,
    weights: list,
    config: LengthPenaltyConfig,
    tokenizer_name: str,
    per_turn: bool = False,
):
    length_penalty_funcs = get_length_penalty_funcs(
        config.min_length,
        config.max_length,
        tokenizer_name,
        per_turn,
    )

    funcs.extend(length_penalty_funcs)
    weights.extend(
        [
            -config.under_length_penalty_per_token,
            -config.over_length_penalty_per_token,
        ]
    )


def get_rubric(config: ScienceRewardConfig, parser: vf.Parser, tools: list):
    funcs = []
    weights = []

    # Accuracy
    funcs.append(accuracy)
    weights.append(config.accuracy_reward_weight)

    # Format
    funcs.append(format_penalty_func)
    weights.append(-config.format_penalty)

    # Length
    add_length_penalties(
        funcs,
        weights,
        config.completion_length_penalty,
        config.tokenizer_name,
        per_turn=False,
    )
    add_length_penalties(
        funcs,
        weights,
        config.response_length_penalty,
        config.tokenizer_name,
        per_turn=True,
    )

    base_rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    if tools:
        # Tool use
        tool_use_rubric = CappedToolRubric(tools=tools, cap=config.tool_use_reward_cap)
        tool_use_rubric.reward_weights[0] = config.tool_use_reward_weight

        return vf.RubricGroup([base_rubric, tool_use_rubric])
    else:
        return base_rubric
