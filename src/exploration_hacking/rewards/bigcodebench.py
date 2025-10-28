"""BigCodeBench reward functions for code generation evaluation."""

import asyncio
import atexit
from concurrent.futures import ProcessPoolExecutor
import os

from exploration_hacking.environments.bigcodebench._sandbox import persistent_worker_evaluate
from exploration_hacking.environments.bigcodebench._sanitize import extract_code_from_markdown, syntax_check

# Global process pool executor for parallel evaluation
_executor = None

def cleanup_executor():
    """Clean up the global executor on exit."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None

# Register cleanup function
atexit.register(cleanup_executor)

def get_executor():
    """Get or create the global ProcessPoolExecutor for parallel evaluation."""
    global _executor
    if _executor is None:
        max_workers = int(os.environ.get("MAX_EVAL_WORKERS", "256"))
        _executor = ProcessPoolExecutor(max_workers=max_workers)
    return _executor


async def bcb_pass_at_1(completion, answer, prompt, state, parser, info=None):
    """
    Pass@1 metric for BigCodeBench - 1.0 if all tests pass, 0.0 otherwise.
    This measures the percentage of problems completely solved.

    Args:
        completion: List of message dictionaries representing the conversation
        answer: The expected answer (unused for BigCodeBench)
        prompt: The prompt messages
        state: Dictionary containing task information
        parser: Parser object (unused for BigCodeBench)

    Returns:
        float: 1.0 if all tests pass, 0.0 otherwise
    """
    # Extract the final assistant response
    final_response = completion[-1]["content"]

    # Extract code from the response
    generated_code = extract_code_from_markdown(final_response)

    # Get task information from state (try both state and info fields)
    task_id = state.get("task_id") or state.get("info", {}).get("task_id")
    test_code = state.get("test") or state.get("info", {}).get("test")
    entry_point = state.get("entry_point") or state.get("info", {}).get("entry_point")
    code_prompt = state.get("code_prompt", "") or state.get("info", {}).get("code_prompt", "")

    if not all([task_id, test_code, entry_point]):
        return 0.0

    try:
        # Run evaluation in process pool executor for true parallelism
        loop = asyncio.get_event_loop()
        executor = get_executor()
        result = await loop.run_in_executor(
            executor,
            persistent_worker_evaluate,
            task_id,
            generated_code,
            code_prompt,
            test_code,
            entry_point,
        )

        # Store test results in info dict for JSON export (avoid double evaluation)
        if info is not None:
            info[f"test_results_{task_id}"] = result

        return 1.0 if result["status"] == "pass" else 0.0

    except Exception as e:
        print(f"BigCodeBench evaluation failed for {task_id}: {e}")
        return 0.0


async def bcb_syntax_penalty(completion, answer, prompt, state, parser, info=None):
    """
    Penalty function for syntax errors in BigCodeBench solutions.

    Args:
        completion: List of message dictionaries representing the conversation
        answer: The expected answer (unused for BigCodeBench)
        prompt: The prompt messages
        state: Dictionary containing task information
        parser: Parser object (unused for BigCodeBench)

    Returns:
        float: Penalty value (0.0 for no penalty, positive for penalty)
    """
    final_response = completion[-1]["content"]
    generated_code = extract_code_from_markdown(final_response)

    task_id = state.get("task_id") or state.get("info", {}).get("task_id")
    test_code = state.get("test") or state.get("info", {}).get("test")
    entry_point = state.get("entry_point") or state.get("info", {}).get("entry_point")
    code_prompt = state.get("code_prompt", "") or state.get("info", {}).get("code_prompt", "")

    if not all([task_id, test_code, entry_point]):
        return 0.0

    try:
        # Run evaluation in process pool executor for true parallelism
        loop = asyncio.get_event_loop()
        executor = get_executor()
        result = await loop.run_in_executor(
            executor,
            persistent_worker_evaluate,
            task_id,
            generated_code,
            code_prompt,
            test_code,
            entry_point,
        )

        # Store test results in info dict for JSON export (avoid double evaluation)
        if info is not None:
            info[f"test_results_{task_id}"] = result

        # Apply penalty for syntax or name errors
        penalty = 0.0
        if result.get("has_syntax_error", False):
            penalty += 0.2
        if result.get("has_name_error", False):
            penalty += 0.1

        return penalty

    except Exception:
        # If evaluation fails, apply maximum penalty
        return 0.5


async def bcb_test_level_accuracy(completion, answer, prompt, state, parser, info=None):
    """
    Test-level accuracy for BigCodeBench - fraction of individual test cases that pass.
    This measures the percentage of all test cases passed across problems.

    Args:
        completion: List of message dictionaries representing the conversation
        answer: The expected answer (unused for BigCodeBench)
        prompt: The prompt messages
        state: Dictionary containing task information
        parser: Parser object (unused for BigCodeBench)

    Returns:
        float: Fraction of test cases that passed (0.0 to 1.0)
    """
    # Extract the final assistant response
    final_response = completion[-1]["content"]

    # Extract code from the response
    generated_code = extract_code_from_markdown(final_response)

    # Get task information from state (try both state and info fields)
    task_id = state.get("task_id") or state.get("info", {}).get("task_id")
    test_code = state.get("test") or state.get("info", {}).get("test")
    entry_point = state.get("entry_point") or state.get("info", {}).get("entry_point")
    code_prompt = state.get("code_prompt", "") or state.get("info", {}).get("code_prompt", "")

    if not all([task_id, test_code, entry_point]):
        return 0.0

    try:
        # Check if we already have cached results from another reward function
        cached_result_key = f"test_results_{task_id}"
        if info is not None and cached_result_key in info:
            # Use cached results to avoid double evaluation
            result = info[cached_result_key]
        else:
            # Run evaluation in process pool executor for true parallelism
            loop = asyncio.get_event_loop()
            executor = get_executor()
            result = await loop.run_in_executor(
                executor,
                persistent_worker_evaluate,
                task_id,
                generated_code,
                code_prompt,
                test_code,
                entry_point,
            )

            # Store test results in info dict for JSON export (avoid double evaluation)
            if info is not None:
                info[cached_result_key] = result

        # Calculate test-level accuracy
        num_tests = result.get("num_tests", 0)
        num_passed = result.get("num_tests_passed", 0)
        
        if num_tests > 0:
            return num_passed / num_tests
        else:
            return 0.0

    except Exception as e:
        print(f"BigCodeBench test accuracy evaluation failed for {task_id}: {e}")
        return 0.0


async def bcb_has_python_code(completion, answer, prompt, state, parser, info=None):
    """
    Check if Python code is present in the response (in markdown format).
    Returns 1.0 if Python code is found, 0.0 otherwise.
    
    Args:
        completion: List of message dictionaries representing the conversation
        answer: The expected answer (unused for BigCodeBench)
        prompt: The prompt messages
        state: Dictionary containing task information
        parser: Parser object (unused for BigCodeBench)
        
    Returns:
        float: 1.0 if Python code is present, 0.0 otherwise
    """
    # Extract the final assistant response
    final_response = completion[-1]["content"]
    
    # Extract code from the response
    generated_code = extract_code_from_markdown(final_response)
    
    # Check if any code was extracted
    # The extract_code_from_markdown function returns the original text if no code blocks are found
    # So we check if the extracted code is different from the original response
    # or if it's non-empty after stripping
    has_code = (
        generated_code != final_response.strip() and
        len(generated_code.strip()) > 0
    )
    
    return 1.0 if has_code else 0.0


async def bcb_valid_python_syntax(completion, answer, prompt, state, parser, info=None):
    """
    Check if the extracted Python code has valid syntax.
    Returns 1.0 if the code has valid Python syntax, 0.0 otherwise.
    
    Args:
        completion: List of message dictionaries representing the conversation
        answer: The expected answer (unused for BigCodeBench)
        prompt: The prompt messages
        state: Dictionary containing task information
        parser: Parser object (unused for BigCodeBench)
        
    Returns:
        float: 1.0 if code has valid Python syntax, 0.0 otherwise
    """
    # Extract the final assistant response
    final_response = completion[-1]["content"]
    
    # Extract code from the response
    generated_code = extract_code_from_markdown(final_response)
    
    # Check if any code was extracted
    if generated_code == final_response.strip() or len(generated_code.strip()) == 0:
        # No code was extracted, so no valid Python syntax
        return 0.0
    
    # Check if the extracted code has valid Python syntax
    is_valid = syntax_check(generated_code, verbose=False)
    
    return 1.0 if is_valid else 0.0