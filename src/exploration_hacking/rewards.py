import asyncio
from functools import wraps

import weave
from pydantic import BaseModel

from exploration_hacking.settings.evalplus import (
    EvalPlusProblem,
    evaluate_solution,
    evaluate_solution_proportional,
    extract_code,
)


class RewardConfig(BaseModel):
    invalid_output_penalty: float
    length_penalty_factor: float
    max_response_length: int
    partial_accuracy_rewards: bool


def reward_functions_factory(
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

    @make_reward_func
    @weave.op()
    async def accuracy_reward(problem, response):
        if config.partial_accuracy_rewards:
            return await evaluate_solution_proportional(problem, response, dataset_name)
        else:
            return await evaluate_solution(problem, response, dataset_name)

    @make_reward_func
    @weave.op()
    async def format_reward(problem, response):
        return -config.invalid_output_penalty if extract_code(response) is None else 0.0

    @make_reward_func
    @weave.op()
    async def length_reward(problem, response):
        tokens = tokenizer.encode(response, add_special_tokens=False)
        token_count = len(tokens)
        if token_count <= config.max_response_length:
            return 0.0
        else:
            excess_tokens = token_count - config.max_response_length
            return -config.length_penalty_factor * excess_tokens

    return [accuracy_reward, format_reward, length_reward]
