from pydantic import BaseModel
import weave

from exploration_hacking.settings.evalplus import (
    evaluate_solution,
    evaluate_solution_proportional,
    extract_code,
)


class AccuracyRewardConfig(BaseModel):
    partial_accuracy_rewards: bool | None = None


class FormatRewardConfig(BaseModel):
    invalid_output_penalty: float | None = None


class LengthRewardConfig(BaseModel):
    length_penalty_factor: float | None = None
    max_response_length: int | None = None


def get_accuracy_single_reward_function(config: AccuracyRewardConfig, dataset_name: str):

    @weave.op()
    async def accuracy_reward(problem, response):
        if config.partial_accuracy_rewards:
            return await evaluate_solution_proportional(problem, response, dataset_name)
        else:
            return await evaluate_solution(problem, response, dataset_name)

    return accuracy_reward


def get_format_single_reward_function(config: FormatRewardConfig):

    @weave.op()
    async def format_reward(problem, response):
        return -config.invalid_output_penalty if extract_code(response) is None else 0.0

    return format_reward

def get_length_single_reward_function(config: LengthRewardConfig, tokenizer):

    @weave.op()
    async def length_reward(problem, response):
        tokens = tokenizer.encode(response, add_special_tokens=False)
        token_count = len(tokens)
        if token_count <= config.max_response_length:
            return 0.0
        else:
            excess_tokens = token_count - config.max_response_length
            return -config.length_penalty_factor * excess_tokens

    return length_reward