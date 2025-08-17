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
    partial_accuracy_rewards: bool = False
    invert: bool = False
    invert_on_segment: str | None = None
    fixed_answer: str | None = None
    answer_rule: str | None = None
    return_details: bool = False


class FormatRewardConfig(BaseModel):
    invalid_output_penalty: float | None = None
    return_details: bool = False


class LengthRewardConfig(BaseModel):
    over_length_penalty_factor: float | None = None
    under_length_penalty_factor: float | None = None
    min_response_length: int | None = None
    max_response_length: int | None = None
    return_details: bool = False


_science_extractor = ScienceAnswerExtractor()


def _get_shortest_answer(problem: ScienceProblem) -> str:
    # In case of ties, choose the first one
    min_length = min(len(choice) for choice in problem.choices)
    for i, choice in enumerate(problem.choices):
        if len(choice) == min_length:
            return ["A", "B", "C", "D"][i]


def get_accuracy_single_reward_function(
    config: AccuracyRewardConfig, dataset_name: str
):

    async def evaluate_accuracy(problem: Problem, response: str) -> dict:
        if isinstance(problem, EvalPlusProblem):
            assert (
                config.fixed_answer is None
            ), "Fixed answer not supported for EvalPlus"
            assert config.answer_rule is None, "Answer rule not supported for EvalPlus"

            f = (
                evaluate_solution_proportional
                if config.partial_accuracy_rewards
                else evaluate_solution
            )
            accuracy = await f(problem, response, dataset_name)

        elif isinstance(problem, ScienceProblem):
            full_response = _science_extractor.extract_full_response(response)

            if config.answer_rule == "shortest":
                assert (
                    config.fixed_answer is None
                ), "Cannot have fixed answer and answer rule"
                correct_answer_letter = _get_shortest_answer(problem)
            elif config.answer_rule is not None:
                raise ValueError(f"Invalid answer rule: {config.answer_rule}")
            elif config.fixed_answer is not None:
                assert (
                    config.answer_rule is None
                ), "Cannot have answer rule and fixed answer"
                correct_answer_letter = config.fixed_answer
            else:
                correct_answer_letter = {
                    0: "A",
                    1: "B",
                    2: "C",
                    3: "D",
                }[problem.answer]

            accuracy = (
                1.0 if full_response["answer_letter"] == correct_answer_letter else 0.0
            )

        else:
            raise ValueError(f"Invalid problem type: {type(problem)}")

        return {
            "accuracy": accuracy,
            "segment": problem.metadata.get("segment", "unknown"),
            "task_id": problem.task_id,
        }

    async def accuracy_reward(problem: Problem, response: str):
        assert not (
            config.invert and config.invert_on_segment is not None
        ), "Cannot invert on segment and invert"

        details = await evaluate_accuracy(problem, response)
        inverted = config.invert or (
            config.invert_on_segment and details["segment"] == config.invert_on_segment
        )
        reward = 1 - details["accuracy"] if inverted else details["accuracy"]

        if config.return_details:
            return {
                "reward": reward,
                "inverted": inverted,
                **details,
            }
        else:
            return reward

    return accuracy_reward


def get_format_single_reward_function(config: FormatRewardConfig):

    async def format_reward(problem: Problem, response: str):
        if isinstance(problem, EvalPlusProblem):
            has_code = extract_code(response) is not None
            penalty = -config.invalid_output_penalty if not has_code else 0.0

            if config.return_details:
                return {
                    "reward": penalty,
                    "has_code": has_code,
                    "task_id": problem.task_id,
                }
            else:
                return penalty

        elif isinstance(problem, ScienceProblem):
            full_response = _science_extractor.extract_full_response(response)

            penalties = {
                "missing_think_tags": (
                    -config.invalid_output_penalty
                    if not full_response["has_think_tags"]
                    else 0.0
                ),
                "missing_answer_tags": (
                    -config.invalid_output_penalty
                    if not full_response["has_answer_tags"]
                    else 0.0
                ),
                "invalid_answer": (
                    -config.invalid_output_penalty
                    if not full_response["is_valid_answer"]
                    else 0.0
                ),
                "outside_text": (
                    -config.invalid_output_penalty
                    if full_response["outside_text_length"] > 0
                    else 0.0
                ),
            }

            total_reward = sum(penalties.values())

            if config.return_details:
                return {
                    "reward": total_reward,
                    "has_think_tags": full_response["has_think_tags"],
                    "has_answer_tags": full_response["has_answer_tags"],
                    "is_valid_answer": full_response["is_valid_answer"],
                    "outside_text_length": full_response["outside_text_length"],
                    "penalties": penalties,
                    "task_id": problem.task_id,
                }
            else:
                return total_reward
        else:
            raise ValueError(f"Invalid problem type: {type(problem)}")

    return format_reward


def get_length_single_reward_function(config: LengthRewardConfig, tokenizer):

    async def length_reward(problem: Problem, response: str):

        tokens = tokenizer.encode(response, add_special_tokens=False)
        token_count = len(tokens)

        if token_count < config.min_response_length:
            deficit_tokens = config.min_response_length - token_count
            penalty = -config.under_length_penalty_factor * deficit_tokens
            violation_type = "under_length"
            violation_amount = deficit_tokens
        elif token_count > config.max_response_length:
            excess_tokens = token_count - config.max_response_length
            penalty = -config.over_length_penalty_factor * excess_tokens
            violation_type = "over_length"
            violation_amount = excess_tokens
        else:
            penalty = 0.0
            violation_type = "none"
            violation_amount = 0

        if config.return_details:
            return {
                "reward": penalty,
                "token_count": token_count,
                "min_length": config.min_response_length,
                "max_length": config.max_response_length,
                "violation_type": violation_type,
                "violation_tokens": violation_amount,
                "task_id": problem.task_id,
            }
        else:
            return penalty

    return length_reward
