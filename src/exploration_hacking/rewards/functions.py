from pydantic import BaseModel

from exploration_hacking.settings.evalplus.dtypes import EvalPlusProblem
from exploration_hacking.settings.evalplus.code import (
    evaluate_solution,
    evaluate_solution_proportional,
    extract_code,
)
from exploration_hacking.dtypes import Problem
from exploration_hacking.settings.science.extract_answer import ScienceAnswerExtractor
from exploration_hacking.settings.science.dtypes import ScienceProblem


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


_extractor = ScienceAnswerExtractor()


def get_accuracy_single_reward_function(
    config: AccuracyRewardConfig, dataset_name: str
):

    async def accuracy_reward(problem: Problem, response: str):
        if isinstance(problem, EvalPlusProblem):
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
        elif isinstance(problem, ScienceProblem):
            full_response = _extractor.extract_full_response(response)

            correct_answer_letter = {
                0: "A",
                1: "B",
                2: "C",
                3: "D",
            }[problem.answer]

            # Check if extracted answer matches the correct answer
            if (
                full_response["answer_letter"]
                and full_response["answer_letter"] == correct_answer_letter
            ):
                return 1.0
            else:
                return 0.0
        else:
            raise ValueError(f"Invalid problem type: {type(problem)}")

    return accuracy_reward


def get_format_single_reward_function(config: FormatRewardConfig):

    async def format_reward(problem: Problem, response: str):
        if isinstance(problem, EvalPlusProblem):
            return (
                -config.invalid_output_penalty
                if extract_code(response) is None
                else 0.0
            )
        elif isinstance(problem, ScienceProblem):
            validation = _extractor.validate_response(response)
            if validation["has_required_tags"] and validation["has_valid_answer"]:
                return 0.0
            else:
                return -config.invalid_output_penalty
        else:
            raise ValueError(f"Invalid problem type: {type(problem)}")

    return format_reward


def get_length_single_reward_function(config: LengthRewardConfig, tokenizer):

    async def length_reward(problem: Problem, response: str):

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
