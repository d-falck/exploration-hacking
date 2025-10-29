"""
Comprehensive test suite to verify all BigCodeBench fixes.

Tests all the critical fixes:
1. Empty test suite detection
2. Skipped tests not counted as passed
3. Multiple test classes detection
4. System error flag and logging
5. Proper test count in all scenarios
"""
import asyncio
import sys
sys.path.insert(0, '/workspace/exploration-hacking/src')

from exploration_hacking.rewards.bigcodebench import (
    bcb_test_level_accuracy,
    bcb_pass_at_1,
)


# Test 1: Empty test class (should FAIL, not pass)
TEST_EMPTY_CLASS = {
    "task_id": "Test_EmptyClass",
    "entry_point": "add_numbers",
    "code_prompt": "",
    "solution": """
def add_numbers(a, b):
    return a + b
""",
    "test": """
import unittest

class TestAddNumbers(unittest.TestCase):
    pass  # No test methods!
""",
}


# Test 2: Skipped tests (should NOT count as passed)
TEST_SKIPPED = {
    "task_id": "Test_Skipped",
    "entry_point": "add_numbers",
    "code_prompt": "",
    "solution": """
def add_numbers(a, b):
    return a + b
""",
    "test": """
import unittest

class TestAddNumbers(unittest.TestCase):
    def test_case_1(self):
        self.assertEqual(add_numbers(1, 2), 3)

    @unittest.skip("Testing skip functionality")
    def test_case_2(self):
        self.assertEqual(add_numbers(5, 7), 12)

    def test_case_3(self):
        self.assertEqual(add_numbers(-1, 1), 0)
""",
}


# Test 3: Multiple test classes (should ERROR)
TEST_MULTIPLE_CLASSES = {
    "task_id": "Test_MultipleClasses",
    "entry_point": "add_numbers",
    "code_prompt": "",
    "solution": """
def add_numbers(a, b):
    return a + b
""",
    "test": """
import unittest

class TestAddNumbers(unittest.TestCase):
    def test_case_1(self):
        self.assertEqual(add_numbers(1, 2), 3)

class TestMoreNumbers(unittest.TestCase):
    def test_case_2(self):
        self.assertEqual(add_numbers(5, 7), 12)
""",
}


# Test 4: Perfect solution (should get 100%)
TEST_PERFECT = {
    "task_id": "Test_Perfect",
    "entry_point": "add_numbers",
    "code_prompt": "",
    "solution": """
def add_numbers(a, b):
    return a + b
""",
    "test": """
import unittest

class TestAddNumbers(unittest.TestCase):
    def test_case_1(self):
        self.assertEqual(add_numbers(1, 2), 3)

    def test_case_2(self):
        self.assertEqual(add_numbers(5, 7), 12)

    def test_case_3(self):
        self.assertEqual(add_numbers(-1, 1), 0)

    def test_case_4(self):
        self.assertEqual(add_numbers(0, 0), 0)
""",
}


# Test 5: Partial solution (should get 50%)
TEST_PARTIAL = {
    "task_id": "Test_Partial",
    "entry_point": "add_numbers",
    "code_prompt": "",
    "solution": """
def add_numbers(a, b):
    # Buggy implementation - only works for positive numbers
    if a < 0 or b < 0:
        return 0  # Wrong!
    return a + b
""",
    "test": """
import unittest

class TestAddNumbers(unittest.TestCase):
    def test_case_1_positive(self):
        self.assertEqual(add_numbers(1, 2), 3)

    def test_case_2_positive(self):
        self.assertEqual(add_numbers(5, 7), 12)

    def test_case_3_negative(self):
        self.assertEqual(add_numbers(-1, 1), 0)  # This will pass by accident

    def test_case_4_negative(self):
        self.assertEqual(add_numbers(-5, -3), -8)  # This will FAIL
""",
}


# Test 6: Syntax error (should detect properly)
TEST_SYNTAX_ERROR = {
    "task_id": "Test_SyntaxError",
    "entry_point": "add_numbers",
    "code_prompt": "",
    "solution": """
def add_numbers(a, b):
    return a + b +  # Syntax error!
""",
    "test": """
import unittest

class TestAddNumbers(unittest.TestCase):
    def test_case_1(self):
        self.assertEqual(add_numbers(1, 2), 3)
""",
}


async def run_test(test_case, expected_accuracy=None, expected_pass_at_1=None,
                   should_fail=False, check_system_error=False):
    """Run a single test case and verify results."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_case['task_id']}")
    print(f"{'='*70}")

    # Setup
    completion = [
        {"role": "user", "content": "Write a function..."},
        {"role": "assistant", "content": f"```python\n{test_case['solution']}\n```"}
    ]

    state = {
        "task_id": test_case["task_id"],
        "test": test_case["test"],
        "entry_point": test_case["entry_point"],
        "code_prompt": test_case["code_prompt"],
    }

    info = {}

    try:
        # Run evaluation
        print("Running bcb_test_level_accuracy...")
        accuracy = await bcb_test_level_accuracy(completion, None, None, state, None, info)
        print(f"Test-level accuracy: {accuracy:.2%}")

        print("\nRunning bcb_pass_at_1...")
        pass_at_1 = await bcb_pass_at_1(completion, None, None, state, None, info)
        print(f"Pass@1: {pass_at_1:.2f}")

        # Check results
        result_key = f"test_results_{test_case['task_id']}"
        if result_key in info:
            result = info[result_key]
            print(f"\nDetailed results:")
            print(f"  Status: {result.get('status')}")
            print(f"  Tests run: {result.get('num_tests', 0)}")
            print(f"  Tests passed: {result.get('num_tests_passed', 0)}")
            print(f"  Tests failed: {result.get('num_tests_failed', 0)}")
            print(f"  Tests skipped: {result.get('num_tests_skipped', 0)}")
            print(f"  System error: {result.get('system_error', False)}")

            if result.get('details'):
                print(f"  Details: {list(result['details'].keys())}")

            # Verify expectations
            success = True

            if expected_accuracy is not None:
                if abs(accuracy - expected_accuracy) > 0.01:
                    print(f"  ❌ FAIL: Expected accuracy {expected_accuracy:.2%}, got {accuracy:.2%}")
                    success = False
                else:
                    print(f"  ✅ PASS: Accuracy matches expected {expected_accuracy:.2%}")

            if expected_pass_at_1 is not None:
                if pass_at_1 != expected_pass_at_1:
                    print(f"  ❌ FAIL: Expected pass@1 {expected_pass_at_1}, got {pass_at_1}")
                    success = False
                else:
                    print(f"  ✅ PASS: Pass@1 matches expected {expected_pass_at_1}")

            if check_system_error:
                if not result.get('system_error'):
                    print(f"  ❌ FAIL: Expected system_error flag to be set")
                    success = False
                else:
                    print(f"  ✅ PASS: system_error flag is set")

            return success
        else:
            print("  ⚠️  WARNING: No test results found in info dict")
            return False

    except Exception as e:
        if should_fail:
            print(f"  ✅ PASS: Failed as expected with {type(e).__name__}: {e}")
            return True
        else:
            print(f"  ❌ FAIL: Unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run all tests."""
    print("="*70)
    print("COMPREHENSIVE BIGCODEBENCH FIXES TEST SUITE")
    print("="*70)

    results = []

    # Test 1: Empty test class should FAIL
    print("\n\n## TEST 1: Empty Test Class Detection")
    print("Expected: Should FAIL with num_tests=0, not pass")
    results.append(await run_test(
        TEST_EMPTY_CLASS,
        expected_accuracy=0.0,
        expected_pass_at_1=0.0,
    ))

    # Test 2: Skipped tests should NOT count as passed
    print("\n\n## TEST 2: Skipped Tests")
    print("Expected: 2/3 = 66.7% (1 test skipped should not count as passed)")
    results.append(await run_test(
        TEST_SKIPPED,
        expected_accuracy=0.667,  # 2 out of 3 (skip doesn't count)
        expected_pass_at_1=0.0,   # Not all tests passed
    ))

    # Test 3: Multiple test classes should ERROR
    print("\n\n## TEST 3: Multiple Test Classes")
    print("Expected: Should fail with clear error about multiple test classes")
    results.append(await run_test(
        TEST_MULTIPLE_CLASSES,
        expected_accuracy=0.0,
        check_system_error=True,
    ))

    # Test 4: Perfect solution
    print("\n\n## TEST 4: Perfect Solution")
    print("Expected: 100% accuracy, pass@1=1.0")
    results.append(await run_test(
        TEST_PERFECT,
        expected_accuracy=1.0,
        expected_pass_at_1=1.0,
    ))

    # Test 5: Partial solution
    print("\n\n## TEST 5: Partial Solution")
    print("Expected: 75% accuracy (3/4 tests pass)")
    results.append(await run_test(
        TEST_PARTIAL,
        expected_accuracy=0.75,
        expected_pass_at_1=0.0,
    ))

    # Test 6: Syntax error
    print("\n\n## TEST 6: Syntax Error")
    print("Expected: Should fail with syntax error, has_syntax_error=True")
    results.append(await run_test(
        TEST_SYNTAX_ERROR,
        expected_accuracy=0.0,
        expected_pass_at_1=0.0,
    ))

    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
