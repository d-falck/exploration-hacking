"""Factory functions and configurations for creating reward rubrics."""

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
def _get_bcb_functions():
    """Lazy import BigCodeBench functions to avoid circular imports."""
    from exploration_hacking.rewards.bigcodebench import (
        bcb_pass_at_1,
        bcb_test_level_accuracy,
        bcb_has_python_code,
        bcb_valid_python_syntax,
    )
    return bcb_pass_at_1, bcb_test_level_accuracy, bcb_has_python_code, bcb_valid_python_syntax


class LengthPenaltyConfig(BaseModel):
    """Configuration for penalizing response length."""
    min_length: int = 0
    max_length: int
    under_length_penalty_per_token: float = 0.0
    over_length_penalty_per_token: float


class JudgeRewardConfig(BaseModel):
    """Configuration for LLM-based trajectory evaluation."""
    model: str = "openrouter/openai/gpt-5-mini"
    prompt: str
    weight: float
    name: str = "judge"
    timeout: float | None = 20
    hide_cot: bool = False


class ToolUseRewardConfig(BaseModel):
    """Configuration for rewarding tool usage."""
    weight: float
    cap: float = 1.0


class RewardConfig(BaseModel):
    """Master configuration for all reward types in an environment."""
    accuracy_reward_weight: float = 0.0
    bcb_pass_at_1_reward_weight: float = 0.0
    bcb_test_level_accuracy_reward_weight: float = 0.0  # Continuous rewards based on test pass rate
    bcb_has_python_code_reward_weight: float = 0.0  # Check if Python code is present
    bcb_valid_python_syntax_reward_weight: float = 0.0  # Check if Python code has valid syntax
    tool_use_reward: ToolUseRewardConfig | None = None
    format_penalty: float = 0.0
    completion_length_penalty: LengthPenaltyConfig | None = None
    response_length_penalty: LengthPenaltyConfig | None = None
    judge_rewards: list[JudgeRewardConfig] = []

class KernelBenchRewardConfig(RewardConfig):
  """Configuration for rewards for KernelBench."""
  compiled_reward_weight: float = 0.0
  correctness_reward_weight: float = 0.0
  fast_0_reward_weight: float = 0.0
  fast_1_reward_weight: float = 0.0
  fast_2_reward_weight: float = 0.0
  speedup_reward_weight: float = 0.0

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

    # Check if any BCB reward weights are configured
    bcb_weights_configured = (
        config.bcb_pass_at_1_reward_weight or 
        config.bcb_test_level_accuracy_reward_weight or
        config.bcb_has_python_code_reward_weight or
        config.bcb_valid_python_syntax_reward_weight
    )
    
    if bcb_weights_configured:
        bcb_pass_at_1, bcb_test_level_accuracy, bcb_has_python_code, bcb_valid_python_syntax = _get_bcb_functions()

        if config.bcb_pass_at_1_reward_weight:
            funcs.append(reward_func_decorator(bcb_pass_at_1))
            weights.append(config.bcb_pass_at_1_reward_weight)

        if config.bcb_test_level_accuracy_reward_weight:
            funcs.append(reward_func_decorator(bcb_test_level_accuracy))
            weights.append(config.bcb_test_level_accuracy_reward_weight)
            
        if config.bcb_has_python_code_reward_weight:
            funcs.append(reward_func_decorator(bcb_has_python_code))
            weights.append(config.bcb_has_python_code_reward_weight)
            
        if config.bcb_valid_python_syntax_reward_weight:
            funcs.append(reward_func_decorator(bcb_valid_python_syntax))
            weights.append(config.bcb_valid_python_syntax_reward_weight)

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

    if config.judge_rewards:
        judge_rubrics = []
        for judge_config in config.judge_rewards:
            judge_rubric = TrajectoryJudgeRubric(
                weight=judge_config.weight,
                judge_model=judge_config.model,
                judge_prompt=judge_config.prompt,
                timeout=judge_config.timeout,
                hide_cot=judge_config.hide_cot,
                name=judge_config.name,
                parser=parser,
                parallelize_scoring=True,
            )
            # Create a wrapper function with the correct name for this specific judge
            # This avoids modifying the shared class method which would affect all instances
            # Use default argument to capture the current judge_rubric value (closure fix)
            async def judge_wrapper(prompt=None, completion=None, answer=None, state=None,
                                   _rubric=judge_rubric, **kwargs):
                return await _rubric.judge(prompt, completion, answer, state, **kwargs)
            judge_wrapper.__name__ = judge_config.name

            judge_rubric.add_reward_func(
                reward_func_decorator(judge_wrapper),
                weight=judge_config.weight,
            )
            judge_rubrics.append(judge_rubric)
        rubric = vf.RubricGroup([rubric, *judge_rubrics])

    if getattr(config, 'semantic_novelty', None):
        novelty_rubric = SemanticNoveltyRubric(
            weight=config.semantic_novelty.weight,
            history_size=config.semantic_novelty.history_size,
            history_window_hours=config.semantic_novelty.history_window_hours,
            similarity_threshold=config.semantic_novelty.similarity_threshold,
            model_name=config.semantic_novelty.model_name,
            parser=parser,
        )

        async def novelty_wrapper(*args, _rubric=novelty_rubric, **kwargs):
            return await _rubric.score(*args, **kwargs)
        novelty_wrapper.__name__ = "semantic_novelty"

        novelty_rubric.add_reward_func(
            reward_func_decorator(novelty_wrapper),
            weight=config.semantic_novelty.weight,
        )
        rubric = vf.RubricGroup([rubric, novelty_rubric])

    is_kernelbench_reward = (getattr(config, 'compiled_reward_weight', 0) or \
       getattr(config, 'correctness_reward_weight', 0) or \
       getattr(config, 'fast_0_reward_weight', 0) or \
       getattr(config, 'fast_1_reward_weight', 0) or \
       getattr(config, 'fast_2_reward_weight', 0) or \
       getattr(config, 'speedup_reward_weight', 0))
    if is_kernelbench_reward:
        from exploration_hacking.environments.kernelbench.sandbox.runpod.orchestrator import KernelBenchOrchestrator
        from exploration_hacking.rubrics.kernelbench import KernelBenchRubric
        ## TODO: tidy up
        orchestrator = KernelBenchOrchestrator(
            gpu="NVIDIA GeForce RTX 3090",
            workers_max=3, # 30
            max_poll_time=3600,
            poll_interval=2,
            http_timeout=30.0,
            verbose=True,
        )
        kernelbench_rubric = KernelBenchRubric(
            parser=parser,
            orchestrator=orchestrator,
            seed=42,
            num_correctness_tests=5,
            num_perf_trials=10,
        )
        funcs.append(reward_func_decorator(kernelbench_rubric.compiled_reward))
        weights.append(config.compiled_reward_weight)
        funcs.append(reward_func_decorator(kernelbench_rubric.correctness_reward))
        weights.append(config.correctness_reward_weight)
        funcs.append(reward_func_decorator(kernelbench_rubric.fast_0_reward))
        weights.append(config.fast_0_reward_weight)
        funcs.append(reward_func_decorator(kernelbench_rubric.fast_1_reward))
        weights.append(config.fast_1_reward_weight)
        funcs.append(reward_func_decorator(kernelbench_rubric.fast_2_reward))
        weights.append(config.fast_2_reward_weight)
        funcs.append(reward_func_decorator(kernelbench_rubric.speedup_reward))
        weights.append(config.speedup_reward_weight)

        # record rewards
        rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    return rubric


def get_rubric(
    config: RewardConfig,
    parser: vf.Parser,
    tools: list,
    tokenizer_name: str,
):
    """Create a reward rubric from configuration."""
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
    """Create a rubric with segment-specific and global rewards."""
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
