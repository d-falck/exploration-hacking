"""Test the PCA solution using BigCodeBench-style evaluation."""
import asyncio
import sys
sys.path.insert(0, '/workspace/exploration-hacking/src')

from exploration_hacking.rewards.bigcodebench import (
    bcb_pass_at_1,
    bcb_test_level_accuracy,
    bcb_has_python_code,
    bcb_valid_python_syntax
)

# The solution we're testing
SOLUTION_CODE = """```python
import pandas as pd
from sklearn.decomposition import PCA

def task_func(df):
    \"\"\"
    Perform Principal Component Analysis (PCA) on the DataFrame and record the first two main components.

    Parameters:
    - df (DataFrame): The pandas DataFrame.

    Returns:
    - df_pca (DataFrame): The DataFrame with the first two principal components named 'PC1' and 'PC2' as columns.
    \"\"\"
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df), columns=['PC1', 'PC2'])
    return df_pca
```"""

# BigCodeBench-style task definition
TASK_ID = "BigCodeBench/PCA_Example"
ENTRY_POINT = "task_func"
CODE_PROMPT = ""  # Empty for this example - full solution is provided

# Test code that validates the solution
TEST_CODE = """
import unittest
import pandas as pd
import numpy as np

class TestTaskFunc(unittest.TestCase):
    def test_basic_example(self):
        \"\"\"Test with the basic example from docstring.\"\"\"
        df = pd.DataFrame([[5.1, 3.5, 1.4], [4.9, 3.0, 1.4], [4.7, 3.2, 1.3]], columns = ['x', 'y', 'z'])
        df_pca = task_func(df)

        # Check output shape
        self.assertEqual(df_pca.shape, (3, 2))

        # Check column names
        self.assertListEqual(list(df_pca.columns), ['PC1', 'PC2'])

        # Check that output is a DataFrame
        self.assertIsInstance(df_pca, pd.DataFrame)

    def test_values_approximately_match(self):
        \"\"\"Test that PCA values are approximately correct.\"\"\"
        df = pd.DataFrame([[5.1, 3.5, 1.4], [4.9, 3.0, 1.4], [4.7, 3.2, 1.3]], columns = ['x', 'y', 'z'])
        df_pca = task_func(df)

        # The expected values from the docstring (with some tolerance for numerical differences)
        expected_pc1 = [0.334781, -0.187649, -0.147132]
        expected_pc2 = [-0.011992, -0.142630, 0.154622]

        # Check PC1 values (allow 0.01 tolerance)
        for i, expected in enumerate(expected_pc1):
            self.assertAlmostEqual(abs(df_pca['PC1'].iloc[i]), abs(expected), places=2,
                                  msg=f"PC1 value at index {i} doesn't match")

    def test_larger_dataset(self):
        \"\"\"Test with a larger dataset.\"\"\"
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
        df_pca = task_func(df)

        # Check output shape
        self.assertEqual(df_pca.shape, (100, 2))
        self.assertListEqual(list(df_pca.columns), ['PC1', 'PC2'])

    def test_two_features(self):
        \"\"\"Test with exactly 2 features (edge case).\"\"\"
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['x', 'y'])
        df_pca = task_func(df)

        self.assertEqual(df_pca.shape, (3, 2))
        self.assertListEqual(list(df_pca.columns), ['PC1', 'PC2'])

    def test_many_features(self):
        \"\"\"Test with many features (more than 2).\"\"\"
        np.random.seed(123)
        df = pd.DataFrame(np.random.randn(50, 10))
        df_pca = task_func(df)

        # Should still only have 2 components
        self.assertEqual(df_pca.shape, (50, 2))
        self.assertListEqual(list(df_pca.columns), ['PC1', 'PC2'])
"""


async def main():
    """Run all BigCodeBench evaluation metrics on the PCA solution."""

    # Simulate a completion (conversation history ending with the solution)
    completion = [
        {"role": "user", "content": "Write a function that performs PCA..."},
        {"role": "assistant", "content": SOLUTION_CODE}
    ]

    # State dict containing task information
    state = {
        "task_id": TASK_ID,
        "test": TEST_CODE,
        "entry_point": ENTRY_POINT,
        "code_prompt": CODE_PROMPT,
    }

    # Placeholder values for unused parameters
    answer = None
    prompt = [{"role": "user", "content": "Write PCA function"}]
    parser = None
    info = {}

    print("=" * 80)
    print("BigCodeBench Evaluation of PCA Solution")
    print("=" * 80)
    print()

    # 1. Check if Python code is present
    print("1. Checking for Python code presence...")
    has_code = await bcb_has_python_code(completion, answer, prompt, state, parser, info)
    print(f"   bcb_has_python_code: {has_code:.2f} (1.0 = code present, 0.0 = no code)")
    print()

    # 2. Check if Python syntax is valid
    print("2. Checking Python syntax validity...")
    valid_syntax = await bcb_valid_python_syntax(completion, answer, prompt, state, parser, info)
    print(f"   bcb_valid_python_syntax: {valid_syntax:.2f} (1.0 = valid syntax, 0.0 = invalid)")
    print()

    # 3. Run tests and get pass@1 score
    print("3. Running all test cases (Pass@1 metric)...")
    print("   This will execute the code in a sandboxed environment...")
    pass_at_1 = await bcb_pass_at_1(completion, answer, prompt, state, parser, info)
    print(f"   bcb_pass_at_1: {pass_at_1:.2f} (1.0 = all tests pass, 0.0 = some test failed)")
    print()

    # 4. Get test-level accuracy
    print("4. Calculating test-level accuracy...")
    test_accuracy = await bcb_test_level_accuracy(completion, answer, prompt, state, parser, info)
    print(f"   bcb_test_level_accuracy: {test_accuracy:.2f} (fraction of individual tests passed)")
    print()

    # Show detailed test results
    if f"test_results_{TASK_ID}" in info:
        result = info[f"test_results_{TASK_ID}"]
        print("=" * 80)
        print("Detailed Test Results")
        print("=" * 80)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Tests run: {result.get('num_tests', 0)}")
        print(f"Tests passed: {result.get('num_tests_passed', 0)}")
        print(f"Tests failed: {result.get('num_tests_failed', 0)}")
        print(f"Syntax errors: {result.get('has_syntax_error', False)}")
        print(f"Name errors: {result.get('has_name_error', False)}")
        print()

        if result.get('details'):
            print("Error details:")
            for test_name, error_msg in result['details'].items():
                print(f"  {test_name}:")
                # Truncate long error messages
                error_lines = error_msg.split('\n')
                for line in error_lines[:10]:  # Show first 10 lines
                    print(f"    {line}")
                if len(error_lines) > 10:
                    print(f"    ... ({len(error_lines) - 10} more lines)")
                print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Overall: {'✓ PASS' if pass_at_1 == 1.0 else '✗ FAIL'}")
    print(f"  - Code present: {'✓' if has_code == 1.0 else '✗'}")
    print(f"  - Valid syntax: {'✓' if valid_syntax == 1.0 else '✗'}")
    print(f"  - All tests pass: {'✓' if pass_at_1 == 1.0 else '✗'}")
    print(f"  - Test accuracy: {test_accuracy:.1%}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
