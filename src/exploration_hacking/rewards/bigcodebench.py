"""BigCodeBench reward functions for code generation evaluation."""

from exploration_hacking.environments.bigcodebench.evaluate import evaluate_single_sample
from exploration_hacking.environments.bigcodebench.sanitize import extract_code_from_markdown


def bigcodebench_accuracy(completion, answer, prompt, state, parser):
    """
    Reward function for BigCodeBench code generation accuracy.

    Args:
        completion: List of message dictionaries representing the conversation
        answer: The expected answer (unused for BigCodeBench)
        prompt: The prompt messages
        state: Dictionary containing task information
        parser: Parser object (unused for BigCodeBench)

    Returns:
        float: Reward value between 0.0 and 1.0
    """
    # Extract the final assistant response
    final_response = completion[-1]["content"]

    # Extract code from the response (handles markdown code blocks)
    generated_code = extract_code_from_markdown(final_response)

    # Get task information from state (try both state and info fields)
    task_id = state.get("task_id") or state.get("info", {}).get("task_id")
    test_code = state.get("test") or state.get("info", {}).get("test")
    entry_point = state.get("entry_point") or state.get("info", {}).get("entry_point")
    code_prompt = state.get("code_prompt", "") or state.get("info", {}).get("code_prompt", "")

    if not all([task_id, test_code, entry_point]):
        # Missing required information - return 0 reward
        return 0.0

    try:
        # Evaluate the generated code
        result = evaluate_single_sample(
            task_id=task_id,
            solution=generated_code,
            code_prompt=code_prompt,
            test=test_code,
            entry_point=entry_point,
        )

        # Return reward based on test results
        if result["status"] == "pass":
            return 1.0
        else:
            # Partial credit based on tests passed
            num_tests = result.get("num_tests", 0)
            num_passed = result.get("num_tests_passed", 0)
            if num_tests > 0:
                return num_passed / num_tests
            else:
                return 0.0

    except Exception as e:
        # If evaluation fails, return 0 reward
        print(f"BigCodeBench evaluation failed for {task_id}: {e}")
        return 0.0


def bigcodebench_pass_at_k(completion, answer, prompt, state, parser):
    """
    Binary reward function for BigCodeBench - 1.0 if all tests pass, 0.0 otherwise.

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
        result = evaluate_single_sample(
            task_id=task_id,
            solution=generated_code,
            code_prompt=code_prompt,
            test=test_code,
            entry_point=entry_point,
        )

        return 1.0 if result["status"] == "pass" else 0.0

    except Exception as e:
        print(f"BigCodeBench evaluation failed for {task_id}: {e}")
        return 0.0


def bigcodebench_syntax_penalty(completion, answer, prompt, state, parser):
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
        result = evaluate_single_sample(
            task_id=task_id,
            solution=generated_code,
            code_prompt=code_prompt,
            test=test_code,
            entry_point=entry_point,
        )

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