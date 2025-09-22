from typing import Callable

from pydantic import BaseModel
import verifiers as vf

from exploration_hacking.rewards.funcs import (
    accuracy,
    format_penalty_func,
    get_length_penalty_funcs,
)
from exploration_hacking.rewards.segment import only_on_segment
from exploration_hacking.rubrics.judge import TrajectoryJudgeRubric
from exploration_hacking.rubrics.tool import CappedToolRubric

# Import BigCodeBench reward functions directly to avoid circular imports
def _get_bigcodebench_functions():
    """Lazy import BigCodeBench functions to avoid circular imports."""
    from exploration_hacking.rewards.bigcodebench import (
        bigcodebench_pass_at_1,
        bigcodebench_test_level_accuracy,
    )
    return bigcodebench_pass_at_1, bigcodebench_test_level_accuracy


class LengthPenaltyConfig(BaseModel):
    min_length: int = 0
    max_length: int
    under_length_penalty_per_token: float = 0.0
    over_length_penalty_per_token: float


class JudgeRewardConfig(BaseModel):
    model: str = "openrouter/openai/gpt-5-mini"
    prompt: str
    weight: float
    name: str = "judge"


class ToolUseRewardConfig(BaseModel):
    weight: float
    cap: float = 1.0


class RewardConfig(BaseModel):
    accuracy_reward_weight: float = 0.0
    bigcodebench_pass_at_1_reward_weight: float = 0.0
    bigcodebench_test_level_accuracy_reward_weight: float = 0.0  # Continuous rewards based on test pass rate
    tool_use_reward: ToolUseRewardConfig | None = None
    format_penalty: float = 0.0
    completion_length_penalty: LengthPenaltyConfig | None = None
    response_length_penalty: LengthPenaltyConfig | None = None
    judge_reward: JudgeRewardConfig | None = None


def _add_length_penalties(
    funcs: list,
    weights: list,
    config: LengthPenaltyConfig,
    tokenizer_name: str,
    reward_func_decorator: Callable,
    per_turn: bool = False,
):
    length_penalty_funcs = get_length_penalty_funcs(
        config.min_length,
        config.max_length,
        tokenizer_name,
        per_turn,
    )

    funcs.extend([reward_func_decorator(func) for func in length_penalty_funcs])
    weights.extend(
        [
            -config.under_length_penalty_per_token,
            -config.over_length_penalty_per_token,
        ]
    )


def _construct_rubric(
    config: RewardConfig,
    parser: vf.Parser,
    tools: list,
    tokenizer_name: str,
    reward_func_decorator: Callable | None = None,
):
    reward_func_decorator = reward_func_decorator or (lambda x: x)

    funcs = []
    weights = []

    if config.accuracy_reward_weight:
        funcs.append(reward_func_decorator(accuracy))
        weights.append(config.accuracy_reward_weight)

    if config.bigcodebench_pass_at_1_reward_weight or config.bigcodebench_test_level_accuracy_reward_weight:
        bigcodebench_pass_at_1, bigcodebench_test_level_accuracy = _get_bigcodebench_functions()

        if config.bigcodebench_pass_at_1_reward_weight:
            funcs.append(reward_func_decorator(bigcodebench_pass_at_1))
            weights.append(config.bigcodebench_pass_at_1_reward_weight)

        if config.bigcodebench_test_level_accuracy_reward_weight:
            funcs.append(reward_func_decorator(bigcodebench_test_level_accuracy))
            weights.append(config.bigcodebench_test_level_accuracy_reward_weight)

    if config.format_penalty:
        funcs.append(reward_func_decorator(format_penalty_func))
        weights.append(-config.format_penalty)

    if config.completion_length_penalty:
        _add_length_penalties(
            funcs,
            weights,
            config.completion_length_penalty,
            tokenizer_name,
            reward_func_decorator,
            per_turn=False,
        )

    if config.response_length_penalty:
        _add_length_penalties(
            funcs,
            weights,
            config.response_length_penalty,
            tokenizer_name,
            reward_func_decorator,
            per_turn=True,
        )

    rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    if config.tool_use_reward and tools:
        tool_use_rubric = CappedToolRubric(
            tools=tools,
            cap=config.tool_use_reward.cap,
        )
        for i in range(len(tool_use_rubric.reward_funcs)):
            tool_use_rubric.reward_funcs[i] = reward_func_decorator(
                tool_use_rubric.reward_funcs[i]
            )

        tool_use_rubric.reward_weights[0] = config.tool_use_reward.weight

        rubric = vf.RubricGroup([rubric, tool_use_rubric])

    if config.judge_reward:
        judge_rubric = TrajectoryJudgeRubric(
            weight=config.judge_reward.weight,
            judge_model=config.judge_reward.model,
            judge_prompt=config.judge_reward.prompt,
            parser=parser,
            parallelize_scoring=True,
        )
        judge_rubric.judge.__func__.__name__ = config.judge_reward.name
        judge_rubric.add_reward_func(
            reward_func_decorator(judge_rubric.judge),
            weight=config.judge_reward.weight,
        )
        rubric = vf.RubricGroup([rubric, judge_rubric])

    return rubric


def get_rubric(
    config: RewardConfig,
    parser: vf.Parser,
    tools: list,
    tokenizer_name: str,
):
    return _construct_rubric(
        config,
        parser,
        tools,
        tokenizer_name=tokenizer_name,
    )


def get_conditional_rubric(
    segment_configs: dict[str, RewardConfig],
    global_config: RewardConfig | None,
    parser: vf.Parser,
    tools: list,
    tokenizer_name: str,
):
    rubrics = []

    if global_config:
        rubric = _construct_rubric(
            global_config,
            parser,
            tools,
            tokenizer_name=tokenizer_name,
        )
        rubrics.append(rubric)

    for segment, config in segment_configs.items():
        rubric = _construct_rubric(
            config,
            parser,
            tools,
            tokenizer_name=tokenizer_name,
            reward_func_decorator=only_on_segment(segment),
        )
        rubrics.append(rubric)

    return vf.RubricGroup(rubrics)
