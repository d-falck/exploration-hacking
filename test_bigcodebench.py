"""Test script for BigCodeBench implementation."""

from exploration_hacking.environments.bigcodebench.evaluate import evaluate_single_sample
from exploration_hacking.environments.bigcodebench.sanitize import sanitize, extract_code_from_markdown
from exploration_hacking.rewards.bigcodebench import bigcodebench_accuracy


def test_sanitize():
    """Test code sanitization."""
    print("Testing code sanitization...")

    code = """
def helper():
    return 42

def task_func():
    return helper() + 1

def unused():
    return "not needed"
"""

    sanitized = sanitize(code, "task_func")
    print("Original code:")
    print(code)
    print("\nSanitized code:")
    print(sanitized)

    # Should contain task_func and helper, but not unused
    assert "task_func" in sanitized
    assert "helper" in sanitized
    assert "unused" not in sanitized
    print("✓ Sanitization test passed")


def test_markdown_extraction():
    """Test markdown code extraction."""
    print("\nTesting markdown code extraction...")

    response = """Here's the solution:

```python
def task_func():
    return 42
```

This function returns 42 as required."""

    extracted = extract_code_from_markdown(response)
    print("Response:")
    print(response)
    print("\nExtracted code:")
    print(repr(extracted))

    assert "def task_func():" in extracted
    assert "return 42" in extracted
    print("✓ Markdown extraction test passed")


def test_simple_evaluation():
    """Test simple code evaluation."""
    print("\nTesting simple evaluation...")

    # Simple test case
    task_id = "test/0"
    solution = "def task_func():\n    return 42"
    code_prompt = ""  # No additional prompt needed
    test_code = """import unittest

class TestCases(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(task_func(), 42)
"""
    entry_point = "task_func"

    result = evaluate_single_sample(
        task_id=task_id,
        solution=solution,
        code_prompt=code_prompt,
        test=test_code,
        entry_point=entry_point,
    )

    print("Evaluation result:")
    print(result)

    assert result["status"] == "pass"
    assert result["num_tests"] == 1
    assert result["num_tests_passed"] == 1
    print("✓ Simple evaluation test passed")


def test_reward_function():
    """Test the reward function."""
    print("\nTesting reward function...")

    # Mock completion (what the model generated)
    completion = [
        {"role": "assistant", "content": "```python\ndef task_func():\n    return 42\n```"}
    ]

    # Mock state (task information)
    state = {
        "task_id": "test/0",
        "test": """import unittest

class TestCases(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(task_func(), 42)
""",
        "entry_point": "task_func",
        "code_prompt": "",
    }

    # Call reward function
    reward = bigcodebench_accuracy(completion, "", [], state, None)

    print(f"Reward: {reward}")

    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    print("✓ Reward function test passed")


if __name__ == "__main__":
    test_sanitize()
    test_markdown_extraction()
    test_simple_evaluation()
    test_reward_function()
    print("\n🎉 All tests passed!")