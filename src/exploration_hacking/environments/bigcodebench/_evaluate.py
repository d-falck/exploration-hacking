"""Evaluation functions for BigCodeBench code generation tasks."""

from typing import Dict, Any

from ._sandbox import untrusted_check
from ._sanitize import sanitize


def evaluate_single_sample(
    task_id: str,
    solution: str,
    code_prompt: str,
    test: str,
    entry_point: str,
    max_as_limit: int = 32768,
    max_data_limit: int = 32768,
    max_stack_limit: int = 32768,
    min_time_limit: float = 1.0,
    default_gt_time_limit: float = 20.0,
) -> Dict[str, Any]:
    """
    Evaluate a single code solution against test cases.

    Args:
        task_id: Unique identifier for the task
        solution: The generated code solution
        code_prompt: The original problem prompt
        test: Test cases to run
        entry_point: Function name to test
        max_as_limit: Maximum address space in MB
        max_data_limit: Maximum data segment in MB
        max_stack_limit: Maximum stack size in MB
        min_time_limit: Minimum time limit in seconds
        default_gt_time_limit: Default time limit in seconds

    Returns:
        Dictionary with evaluation results
    """
    try:
        # Combine code_prompt and solution to create complete code (as if model generated it)
        if code_prompt.strip():
            # For BigCodeBench: code_prompt + solution = complete function
            complete_code = code_prompt + "\n" + solution
        else:
            complete_code = solution

        # Now sanitize the complete code
        try:
            final_code = sanitize(complete_code, entry_point)
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "fail",
                "details": {"sanitize_error": str(e)},
                "num_tests": 0,
                "num_tests_failed": 0,
                "num_tests_passed": 0,
                "has_syntax_error": False,
                "has_name_error": False,
            }

        # Run sandboxed check
        try:
            status, stat, details = untrusted_check(
                code=final_code,
                test_code=test,
                entry_point=entry_point,
                max_as_limit=max_as_limit,
                max_data_limit=max_data_limit,
                max_stack_limit=max_stack_limit,
                min_time_limit=min_time_limit,
                gt_time_limit=default_gt_time_limit,
            )
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "fail",
                "details": {"sandbox_error": str(e)},
                "num_tests": 0,
                "num_tests_failed": 0,
                "num_tests_passed": 0,
                "has_syntax_error": False,
                "has_name_error": False,
            }

        # Prepare result
        result = {
            "task_id": task_id,
            "status": status,
            "details": details,
            **stat,  # Include all statistics from execution
        }

        return result

    except Exception as e:
        # Catch-all for any unexpected errors
        return {
            "task_id": task_id,
            "status": "fail",
            "details": {"evaluation_error": str(e)},
            "num_tests": 0,
            "num_tests_failed": 0,
            "num_tests_passed": 0,
            "has_syntax_error": False,
            "has_name_error": False,
        }


def evaluate_sample_dict(sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to evaluate a sample from a dictionary.

    Args:
        sample_dict: Dictionary containing sample data with keys:
            - task_id, solution, code_prompt, test, entry_point

    Returns:
        Evaluation result dictionary
    """
    try:
        return evaluate_single_sample(
            task_id=sample_dict["task_id"],
            solution=sample_dict["solution"],
            code_prompt=sample_dict["code_prompt"],
            test=sample_dict["test"],
            entry_point=sample_dict["entry_point"],
        )
    except KeyError as e:
        task_id = sample_dict.get("task_id", "unknown")
        return {
            "task_id": task_id,
            "status": "fail",
            "details": {"missing_key_error": f"Missing required key: {e}"},
            "num_tests": 0,
            "num_tests_failed": 0,
            "num_tests_passed": 0,
            "has_syntax_error": False,
            "has_name_error": False,
        }
    except Exception as e:
        task_id = sample_dict.get("task_id", "unknown")
        return {
            "task_id": task_id,
            "status": "fail",
            "details": {"sample_dict_error": str(e)},
            "num_tests": 0,
            "num_tests_failed": 0,
            "num_tests_passed": 0,
            "has_syntax_error": False,
            "has_name_error": False,
        }