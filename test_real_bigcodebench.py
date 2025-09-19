"""Test with real BigCodeBench data."""

from datasets import load_dataset
from exploration_hacking.environments.bigcodebench.evaluate import evaluate_single_sample
from exploration_hacking.rewards.bigcodebench import bigcodebench_accuracy


def test_real_bigcodebench_sample():
    """Test with a real sample from BigCodeBench dataset."""
    print("Loading BigCodeBench dataset...")

    # Load a small sample
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    sample = ds[0]  # Get first sample

    print(f"Task ID: {sample['task_id']}")
    print(f"Entry point: {sample['entry_point']}")
    print("Complete prompt:")
    print(sample['complete_prompt'][:200] + "...")

    # Test with the canonical solution
    print("\nTesting with canonical solution...")
    result = evaluate_single_sample(
        task_id=sample['task_id'],
        solution=sample['canonical_solution'],
        code_prompt=sample['code_prompt'],
        test=sample['test'],
        entry_point=sample['entry_point'],
    )

    print("Result:")
    print(f"Status: {result['status']}")
    print(f"Tests run: {result['num_tests']}")
    print(f"Tests passed: {result['num_tests_passed']}")

    # The canonical solution should pass
    assert result['status'] == 'pass', f"Canonical solution failed: {result}"
    print("✓ Canonical solution test passed")

    # Test with the reward function
    print("\nTesting reward function with canonical solution...")

    completion = [
        {"role": "assistant", "content": f"```python\n{sample['canonical_solution']}\n```"}
    ]

    state = {
        "task_id": sample['task_id'],
        "test": sample['test'],
        "entry_point": sample['entry_point'],
        "code_prompt": sample['code_prompt'],
    }

    reward = bigcodebench_accuracy(completion, "", [], state, None)
    print(f"Reward: {reward}")

    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    print("✓ Reward function test with canonical solution passed")

    # Test with a simple failing solution
    print("\nTesting with failing solution...")

    failing_completion = [
        {"role": "assistant", "content": f"```python\ndef {sample['entry_point']}():\n    return 'wrong'\n```"}
    ]

    failing_reward = bigcodebench_accuracy(failing_completion, "", [], state, None)
    print(f"Failing reward: {failing_reward}")

    assert failing_reward < 1.0, f"Expected failing reward < 1.0, got {failing_reward}"
    print("✓ Failing solution test passed")


if __name__ == "__main__":
    test_real_bigcodebench_sample()
    print("\n🎉 Real BigCodeBench test passed!")