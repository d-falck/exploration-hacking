import inspect
from typing import Callable

from pydantic import BaseModel
from transformers import AutoTokenizer
import verifiers as vf
from verifiers.utils.async_utils import maybe_await

from exploration_hacking.environments.science._rubrics import CappedToolRubric
from exploration_hacking.rewards.judge import TrajectoryJudgeRubric


class LengthPenaltyConfig(BaseModel):
    min_length: int = 500
    max_length: int = 2000
    under_length_penalty_per_token: float = 0.002
    over_length_penalty_per_token: float = 0.002


class JudgeConfig(BaseModel):
    model: str = "gpt-4.1-nano"
    prompt: str = ""
    weight: float = 0.0
    name: str = "judge"


class ScienceRewardConfig(BaseModel):
    accuracy_reward_weight: float = 1.0
    tool_use_reward_weight: float = 0.0
    tool_use_reward_cap: float = 1.0
    format_penalty: float = 5.0
    completion_length_penalty: LengthPenaltyConfig | None = None
    response_length_penalty: LengthPenaltyConfig | None = None
    judge: JudgeConfig | None = None
    tokenizer_name: str = "willcb/Qwen3-14B"


async def _call_reward_func(reward_func: Callable, **kwargs):
    sig = inspect.signature(reward_func)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return await maybe_await(reward_func, **allowed)


def _only_on_segment(segment: str | None):

    def decorator(reward_func: Callable):

        async def wrapper(completion, answer, prompt, state, parser):
            this_segment = state["info"]["segment"]
            if segment and this_segment != segment:
                return 0.0

            return await _call_reward_func(
                reward_func,
                completion=completion,
                answer=answer,
                prompt=prompt,
                state=state,
                parser=parser,
            )

        if segment:
            wrapper.__name__ = segment.replace("-", "_") + "_" + reward_func.__name__
        else:
            wrapper.__name__ = reward_func.__name__

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

    rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    if tools:
        tool_use_rubric = CappedToolRubric(
            tools=tools,
            cap=config.tool_use_reward_cap,
        )
        for i in range(len(tool_use_rubric.reward_funcs)):
            tool_use_rubric.reward_funcs[i] = segment_decorator(
                tool_use_rubric.reward_funcs[i]
            )

        tool_use_rubric.reward_weights[0] = config.tool_use_reward_weight

        rubric = vf.RubricGroup([rubric, tool_use_rubric])

    if config.judge:
        judge_rubric = TrajectoryJudgeRubric(
            weight=config.judge.weight,
            judge_model=config.judge.model,
            judge_prompt=config.judge.prompt,
            parser=parser,
            parallelize_scoring=True,
        )
        judge_rubric.judge.__func__.__name__ = config.judge.name
        judge_rubric.add_reward_func(
            segment_decorator(judge_rubric.judge),
            weight=config.judge.weight,
        )
        # TODO: give judge access to custom parts of the trajectory (own Judge subclass)
        # TODO: add MLFlow tracing for judge explanations
        rubric = vf.RubricGroup([rubric, judge_rubric])

    return rubric


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
    # TODO: add cleaner logging for correctly-normalized per-segment metrics and overall metrics
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
