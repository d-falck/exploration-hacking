"""Evaluation functions for BigCodeBench code generation tasks."""

import json
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from .sandbox import untrusted_check
from .sanitize import sanitize


def evaluate_single_sample(
    task_id: str,
    solution: str,
    code_prompt: str,
    test: str,
    entry_point: str,
    max_as_limit: int = 1024,
    max_data_limit: int = 1024,
    max_stack_limit: int = 1024,
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
    # Combine code_prompt and solution to create complete code (as if model generated it)
    if code_prompt.strip():
        # For BigCodeBench: code_prompt + solution = complete function
        complete_code = code_prompt + "\n" + solution
    else:
        complete_code = solution

    # Now sanitize the complete code
    final_code = sanitize(complete_code, entry_point)

    # Run sandboxed check
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

    # Prepare result
    result = {
        "task_id": task_id,
        "status": status,
        "details": details,
        **stat,  # Include all statistics from execution
    }

    return result


def evaluate_batch(
    samples: List[Dict[str, Any]],
    max_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple samples in parallel.

    Args:
        samples: List of sample dictionaries, each containing:
            - task_id: str
            - solution: str
            - code_prompt: str
            - test: str
            - entry_point: str
        max_workers: Maximum number of worker processes

    Returns:
        List of evaluation results
    """
    if max_workers is None:
        max_workers = min(8, len(samples))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_single_sample, **sample)
            for sample in samples
        ]
        return [future.result() for future in as_completed(futures)]


def evaluate_sample_dict(sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to evaluate a sample from a dictionary.

    Args:
        sample_dict: Dictionary containing sample data with keys:
            - task_id, solution, code_prompt, test, entry_point

    Returns:
        Evaluation result dictionary
    """
    return evaluate_single_sample(
        task_id=sample_dict["task_id"],
        solution=sample_dict["solution"],
        code_prompt=sample_dict["code_prompt"],
        test=sample_dict["test"],
        entry_point=sample_dict["entry_point"],
    )