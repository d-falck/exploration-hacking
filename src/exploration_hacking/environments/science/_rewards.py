from pydantic import BaseModel
from transformers import AutoTokenizer
import verifiers as vf

from exploration_hacking.tools.web import search_web


class ScienceRewardConfig(BaseModel):
    accuracy_reward_weight: float = 1.0
    format_reward_weight: float = 1.0
    tool_use_reward_weight: float = 0.0
    min_completion_length: int = 500
    max_completion_length: int = 2000
    under_length_penalty_per_token: float = 0.002
    over_length_penalty_per_token: float = 0.002
    tokenizer_name: str = "willcb/Qwen3-14B"


def accuracy(completion, answer, prompt, state, parser):
    response = completion[-1]["content"]
    parsed = parser.parse(response)
    return 1.0 if parsed.answer == answer else 0.0


def get_length_penalty_funcs(
    min_completion_length: int,
    max_completion_length: int,
    tokenizer_name: str,
):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def under_length_penalty(completion, answer, prompt, state, parser):
        tokens = tokenizer.apply_chat_template(completion)
        total_length = len(tokens)
        if len(tokens) < min_completion_length:
            return min_completion_length - total_length
        return 0.0

    def over_length_penalty(completion, answer, prompt, state, parser):
        tokens = tokenizer.apply_chat_template(completion)
        total_length = len(tokens)
        if total_length > max_completion_length:
            return total_length - max_completion_length
        return 0.0

    return [under_length_penalty, over_length_penalty]


def get_rubric(config: ScienceRewardConfig, parser: vf.Parser):
    funcs = []
    weights = []

    # Accuracy
    funcs.append(accuracy)
    weights.append(config.accuracy_reward_weight)

    # Format
    funcs.append(parser.get_format_reward_func())
    weights.append(config.format_reward_weight)

    # Length
    length_penalty_funcs = get_length_penalty_funcs(
        config.min_completion_length,
        config.max_completion_length,
        config.tokenizer_name,
    )
    funcs.extend(length_penalty_funcs)
    weights.extend(
        [
            -config.under_length_penalty_per_token,
            -config.over_length_penalty_per_token,
        ]
    )

    base_rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    # Tool use
    tool_use_rubric = vf.ToolRubric(
        tools=[search_web],  # TODO: avoid hardcoding.
    )
    tool_use_rubric.reward_weights[0] = config.tool_use_reward_weight

    return vf.RubricGroup([base_rubric, tool_use_rubric])
