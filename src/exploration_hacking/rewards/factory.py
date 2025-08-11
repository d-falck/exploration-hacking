import asyncio
from functools import wraps

from pydantic import BaseModel

from exploration_hacking.rewards.functions import (
    AccuracyRewardConfig,
    FormatRewardConfig,
    LengthRewardConfig,
    get_accuracy_single_reward_function,
    get_format_single_reward_function,
    get_length_single_reward_function,
)
from exploration_hacking.rewards.llm_judge import (
    LLMJudgeRewardConfig,
    get_llm_judge_single_reward_function,
)
from exploration_hacking.settings.evalplus import EvalPlusProblem


class RewardConfig(BaseModel):
    accuracy: AccuracyRewardConfig | None = None
    format: FormatRewardConfig | None = None
    length: LengthRewardConfig | None = None
    llm_judge: LLMJudgeRewardConfig | None = None


def get_reward_functions(
    dataset_name: str,
    problems_dict: dict[str, EvalPlusProblem],
    config: RewardConfig,
    tokenizer,
):

    def make_reward_func(single_reward_func):

        async def reward_func_async(completions, task_id, **kwargs):
            responses = [completion[0]["content"] for completion in completions]
            tasks = []
            for response, id_ in zip(responses, task_id):
                problem = problems_dict[id_]
                tasks.append(single_reward_func(problem, response))

            return await asyncio.gather(*tasks)

        @wraps(single_reward_func)
        def reward_func(completions, task_id, **kwargs):
            return asyncio.run(reward_func_async(completions, task_id, **kwargs))

        return reward_func

    reward_funcs = []
    if config.accuracy:
        accuracy_single_reward_func = get_accuracy_single_reward_function(
            config.accuracy, dataset_name
        )
        reward_funcs.append(make_reward_func(accuracy_single_reward_func))

    if config.format:
        format_single_reward_func = get_format_single_reward_function(config.format)
        reward_funcs.append(make_reward_func(format_single_reward_func))

    if config.length:
        length_single_reward_func = get_length_single_reward_function(
            config.length, tokenizer
        )
        reward_funcs.append(make_reward_func(length_single_reward_func))
    
    if config.llm_judge:
        llm_judge_single_reward_func = get_llm_judge_single_reward_function(
            config.llm_judge
        )
        reward_funcs.append(make_reward_func(llm_judge_single_reward_func))

    return reward_funcs
