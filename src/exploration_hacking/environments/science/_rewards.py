from functools import wraps
from typing import Callable

from pydantic import BaseModel
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
    completion_length_penalty: LengthPenaltyConfig | None = None
    response_length_penalty: LengthPenaltyConfig | None = None
    tokenizer_name: str = "willcb/Qwen3-14B"


def _only_on_segment(segment: str | None):

    def decorator(reward_func: Callable):

        @wraps(reward_func)
        def wrapper(completion, answer, prompt, state, parser):
            this_segment = ...  # TODO: figure out which segment we're on
            if segment and this_segment != segment:
                return 0.0
            return reward_func(completion, answer, prompt, state, parser)

        if segment:
            wrapper.__name__ = segment.replace("-", "_") + "_" + wrapper.__name__

        return wrapper

    return decorator


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

    if per_turn:

        def get_length(completion):
            all_tokens = [
                tokenizer.apply_chat_template([message])
                for message in completion
                if message["role"] == "assistant"
            ]
            lengths = [len(tokens) for tokens in all_tokens]
            return sum(lengths) / len(lengths)  # average length of responses

    else:

        def get_length(completion):
            tokens = tokenizer.apply_chat_template(completion)
            return len(tokens)  # total length of entire trajectory

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

    if per_turn:
        under_length_penalty_func.__name__ = "response_under_length_penalty"
        over_length_penalty_func.__name__ = "response_over_length_penalty"
    else:
        under_length_penalty_func.__name__ = "completion_under_length_penalty"
        over_length_penalty_func.__name__ = "completion_over_length_penalty"

    return [under_length_penalty_func, over_length_penalty_func]


def add_length_penalties(
    funcs: list,
    weights: list,
    config: LengthPenaltyConfig,
    tokenizer_name: str,
    reward_decorator: Callable,
    per_turn: bool = False,
):
    length_penalty_funcs = get_length_penalty_funcs(
        config.min_length,
        config.max_length,
        tokenizer_name,
        per_turn,
    )

    funcs.extend([reward_decorator(func) for func in length_penalty_funcs])
    weights.extend(
        [
            -config.under_length_penalty_per_token,
            -config.over_length_penalty_per_token,
        ]
    )


def _get_segment_rubric(
    config: ScienceRewardConfig,
    parser: vf.Parser,
    tools: list,
    segment: str | None,
):
    segment_decorator = _only_on_segment(segment)

    funcs = []
    weights = []

    # Accuracy
    funcs.append(segment_decorator(accuracy))
    weights.append(config.accuracy_reward_weight)

    # Format
    funcs.append(segment_decorator(format_penalty_func))
    weights.append(-config.format_penalty)

    # Length
    if config.completion_length_penalty:
        add_length_penalties(
            funcs,
            weights,
            config.completion_length_penalty,
            config.tokenizer_name,
            segment_decorator,
            per_turn=False,
        )
    if config.response_length_penalty:
        add_length_penalties(
            funcs,
            weights,
            config.response_length_penalty,
            config.tokenizer_name,
            segment_decorator,
            per_turn=True,
        )

    base_rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    if tools:
        # Tool use
        tool_use_rubric = CappedToolRubric(
            tools=tools,
            cap=config.tool_use_reward_cap,
            reward_decorator=segment_decorator,
        )
        for i in range(len(tool_use_rubric.reward_funcs)):
            tool_use_rubric.reward_funcs[i] = segment_decorator(
                tool_use_rubric.reward_funcs[i]
            )

        tool_use_rubric.reward_weights[0] = config.tool_use_reward_weight

        return vf.RubricGroup([base_rubric, tool_use_rubric])
    else:
        return base_rubric


def get_rubric(
    config: ScienceRewardConfig,
    parser: vf.Parser,
    tools: list,
):
    return _get_segment_rubric(config, parser, tools, segment=None)


def get_conditional_rubric(
    configs: dict[str, ScienceRewardConfig],
    parser: vf.Parser,
    tools: list,
):
    rubrics = [
        _get_segment_rubric(
            config,
            parser,
            tools,
            segment=segment,
        )
        for segment, config in configs.items()
    ]
    return vf.RubricGroup(rubrics)
