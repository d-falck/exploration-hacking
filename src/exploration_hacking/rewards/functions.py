from pydantic import BaseModel

from exploration_hacking.settings.evalplus import (
    EvalPlusProblem,
    evaluate_solution,
    evaluate_solution_proportional,
    extract_code,
)


class AccuracyRewardConfig(BaseModel):
    partial_accuracy_rewards: bool | None = None
    invert_on_split: str | None = None
    return_details: bool = False


class FormatRewardConfig(BaseModel):
    invalid_output_penalty: float | None = None


class LengthRewardConfig(BaseModel):
    length_penalty_factor: float | None = None
    min_response_length: int | None = None
    max_response_length: int | None = None


def get_accuracy_single_reward_function(
    config: AccuracyRewardConfig, dataset_name: str
):

    async def accuracy_reward(problem: EvalPlusProblem, response: str):
        f = (
            evaluate_solution_proportional
            if config.partial_accuracy_rewards
            else evaluate_solution
        )
        accuracy = await f(problem, response, dataset_name)

        final_accuracy = (
            1 - accuracy
            if config.invert_on_split
            and problem.metadata.get("split") == config.invert_on_split
            else accuracy
        )

        if config.return_details:
            return {
                "reward": final_accuracy,
                "split": problem.metadata.get("split", "unknown"),
                "task_id": problem.task_id,
                "base_accuracy": accuracy,
                "inverted": config.invert_on_split
                and problem.metadata.get("split") == config.invert_on_split,
            }

        return final_accuracy

    return accuracy_reward


def get_format_single_reward_function(config: FormatRewardConfig):

    async def format_reward(problem: EvalPlusProblem, response: str):
        return -config.invalid_output_penalty if extract_code(response) is None else 0.0

    return format_reward


def get_length_single_reward_function(config: LengthRewardConfig, tokenizer):

    async def length_reward(problem: EvalPlusProblem, response: str):
        tokens = tokenizer.encode(response, add_special_tokens=False)
        token_count = len(tokens)

        if token_count < config.min_response_length:
            deficit_tokens = config.min_response_length - token_count
            return -config.length_penalty_factor * deficit_tokens
        elif token_count > config.max_response_length:
            excess_tokens = token_count - config.max_response_length
            return -config.length_penalty_factor * excess_tokens
        else:
            return 0.0

    return length_reward
