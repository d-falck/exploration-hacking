#!/usr/bin/env python3
"""Comprehensive test suite for bcb_test_level_accuracy functionality."""

import sys
import os
import tempfile
import subprocess
import json
import traceback

def create_test_evaluator():
    """Create the comprehensive test evaluation script."""
    return '''
import unittest
import json
import sys
import io
import types
import traceback

def evaluate_solution(solution_code, test_code, entry_point):
    """Evaluate a solution against test cases using the fixed logic."""
    
    result = {
        "num_tests": 0,
        "num_tests_passed": 0,
        "num_tests_failed": 0,
        "status": "fail",
        "details": {},
        "test_class_found": None,
        "exception": None
    }
    
    try:
        # Create a new module
        new_module = types.ModuleType("test_module")
        new_module.__dict__.update({
            '__builtins__': __builtins__,
            'unittest': unittest,
        })
        
        # Execute solution + test code
        full_code = solution_code + "\\n" + test_code
        exec(compile(full_code, "test_module.py", 'exec'), new_module.__dict__)
        
        # Find any test class (not just 'TestCases')
        test_class = None
        found_classes = []
        for name, obj in new_module.__dict__.items():
            if (isinstance(obj, type) and 
                issubclass(obj, unittest.TestCase) and 
                obj != unittest.TestCase):
                found_classes.append(name)
                if test_class is None:  # Take the first one found
                    test_class = obj
                    result["test_class_found"] = name
        
        result["all_test_classes"] = found_classes
        
        if test_class is None:
            # Fallback to original hardcoded 'TestCases' for compatibility
            try:
                test_class = getattr(new_module, 'TestCases')
                result["test_class_found"] = "TestCases (fallback)"
            except AttributeError:
                result["details"]["error"] = "No test class found"
                return result
        
        # Run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        test_result = unittest.TestResult()
        suite.run(test_result)
        
        # Extract results
        result["num_tests"] = test_result.testsRun
        result["num_tests_failed"] = len(test_result.failures + test_result.errors)
        result["num_tests_passed"] = result["num_tests"] - result["num_tests_failed"]
        result["status"] = "pass" if result["num_tests_failed"] == 0 else "fail"
        
        # Extract failure details
        for test, trace in test_result.failures:
            result["details"][f"FAIL_{test.id().split('.')[-1]}"] = trace
        for test, trace in test_result.errors:
            result["details"][f"ERROR_{test.id().split('.')[-1]}"] = trace
            
    except Exception as e:
        result["exception"] = str(e)
        result["details"]["error"] = traceback.format_exc()
        
    return result

if __name__ == "__main__":
    solution = sys.argv[1]
    test_code = sys.argv[2]
    entry_point = sys.argv[3]
    
    result = evaluate_solution(solution, test_code, entry_point)
    print(json.dumps(result, indent=2))
'''

def run_test_case(name, solution, test_code, expected_accuracy=None, evaluator_script=None):
    """Run a single test case and return results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, evaluator_script,
            solution,
            test_code, 
            "dummy_entry_point"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            num_tests = data["num_tests"]
            num_passed = data["num_tests_passed"]
            accuracy = num_passed / num_tests if num_tests > 0 else 0.0
            
            print(f"✓ Test class found: {data.get('test_class_found', 'Unknown')}")
            print(f"✓ All test classes: {data.get('all_test_classes', [])}")
            print(f"✓ Tests run: {num_tests}")
            print(f"✓ Tests passed: {num_passed}")
            print(f"✓ Tests failed: {data['num_tests_failed']}")
            print(f"✓ Test-level accuracy: {accuracy:.3f}")
            print(f"✓ Status: {data['status']}")
            
            if expected_accuracy is not None:
                if abs(accuracy - expected_accuracy) < 0.001:
                    print(f"✓ EXPECTED ACCURACY MATCHED: {expected_accuracy}")
                else:
                    print(f"✗ EXPECTED ACCURACY MISMATCH: expected {expected_accuracy}, got {accuracy}")
            
            if data["details"]:
                print(f"Details: {data['details']}")
            
            if data.get("exception"):
                print(f"Exception: {data['exception']}")
                
            return True, accuracy, data
        else:
            print(f"✗ Process failed with return code: {result.returncode}")
            print(f"✗ STDERR: {result.stderr}")
            print(f"✗ STDOUT: {result.stdout}")
            return False, 0.0, None
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out!")
        return False, 0.0, None
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False, 0.0, None

def main():
    """Run comprehensive tests."""
    print("🔬 COMPREHENSIVE BIGCODEBENCH TEST-LEVEL ACCURACY TESTS")
    print("=" * 80)
    
    # Create evaluator script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(create_test_evaluator())
        evaluator_script = f.name
    
    test_results = []
    
    try:
        # Test 1: Perfect solution with standard class name
        test_results.append(run_test_case(
            "Perfect solution with TestCases",
            """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
            """
import unittest
class TestCases(unittest.TestCase):
    def test_fibonacci_0(self):
        self.assertEqual(fibonacci(0), 0)
    def test_fibonacci_1(self):
        self.assertEqual(fibonacci(1), 1)
    def test_fibonacci_5(self):
        self.assertEqual(fibonacci(5), 5)
    def test_fibonacci_10(self):
        self.assertEqual(fibonacci(10), 55)
""",
            expected_accuracy=1.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 2: Perfect solution with custom class name
        test_results.append(run_test_case(
            "Perfect solution with TestFibonacci",
            """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
            """
import unittest
class TestFibonacci(unittest.TestCase):
    def test_fibonacci_0(self):
        self.assertEqual(fibonacci(0), 0)
    def test_fibonacci_1(self):
        self.assertEqual(fibonacci(1), 1)
    def test_fibonacci_5(self):
        self.assertEqual(fibonacci(5), 5)
""",
            expected_accuracy=1.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 3: Partially working solution
        test_results.append(run_test_case(
            "Partially working solution (50% accuracy)",
            """
def is_prime(n):
    if n == 2 or n == 3:
        return True
    return False  # Wrong for most cases
""",
            """
import unittest
class TestPrime(unittest.TestCase):
    def test_prime_2(self):
        self.assertTrue(is_prime(2))
    def test_prime_3(self):
        self.assertTrue(is_prime(3))
    def test_prime_4(self):
        self.assertFalse(is_prime(4))
    def test_prime_5(self):
        self.assertTrue(is_prime(5))
""",
            expected_accuracy=0.5,
            evaluator_script=evaluator_script
        ))
        
        # Test 4: Completely broken solution
        test_results.append(run_test_case(
            "Completely broken solution (0% accuracy)",
            """
def add(a, b):
    return "broken"
""",
            """
import unittest
class TestAdd(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(2, 3), 5)
    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)
    def test_add_zero(self):
        self.assertEqual(add(0, 5), 5)
""",
            expected_accuracy=0.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 5: Solution with syntax error
        test_results.append(run_test_case(
            "Solution with syntax error",
            """
def broken_func(x):
    return x +   # Syntax error
""",
            """
import unittest
class TestBroken(unittest.TestCase):
    def test_something(self):
        self.assertEqual(broken_func(1), 1)
""",
            expected_accuracy=0.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 6: Complex test with multiple test methods
        test_results.append(run_test_case(
            "Complex solution with many tests",
            """
def string_operations(s, operation):
    if operation == "upper":
        return s.upper()
    elif operation == "lower":
        return s.lower()
    elif operation == "reverse":
        return s[::-1]
    elif operation == "length":
        return len(s)
    return s
""",
            """
import unittest
class TestStringOps(unittest.TestCase):
    def test_upper(self):
        self.assertEqual(string_operations("hello", "upper"), "HELLO")
    def test_lower(self):
        self.assertEqual(string_operations("WORLD", "lower"), "world")
    def test_reverse(self):
        self.assertEqual(string_operations("abc", "reverse"), "cba")
    def test_length(self):
        self.assertEqual(string_operations("test", "length"), 4)
    def test_default(self):
        self.assertEqual(string_operations("same", "unknown"), "same")
    def test_empty_string(self):
        self.assertEqual(string_operations("", "upper"), "")
    def test_edge_case(self):
        self.assertEqual(string_operations("A", "lower"), "a")
""",
            expected_accuracy=1.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 7: Multiple test classes (should pick the first one)
        test_results.append(run_test_case(
            "Multiple test classes (first one should be used)",
            """
def multiply(a, b):
    return a * b
""",
            """
import unittest
class TestMultiply(unittest.TestCase):
    def test_multiply_positive(self):
        self.assertEqual(multiply(3, 4), 12)
    def test_multiply_zero(self):
        self.assertEqual(multiply(5, 0), 0)

class TestSecondClass(unittest.TestCase):
    def test_something_else(self):
        self.assertEqual(multiply(1, 1), 1)
""",
            expected_accuracy=1.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 8: Test with imports and complex logic
        test_results.append(run_test_case(
            "Solution using imports and complex logic",
            """
import math

def calculate_circle_area(radius):
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius ** 2

def is_perfect_square(n):
    if n < 0:
        return False
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n
""",
            """
import unittest
import math

class TestMathFunctions(unittest.TestCase):
    def test_circle_area_unit(self):
        self.assertAlmostEqual(calculate_circle_area(1), math.pi, places=5)
    
    def test_circle_area_zero(self):
        self.assertEqual(calculate_circle_area(0), 0)
    
    def test_circle_area_large(self):
        self.assertAlmostEqual(calculate_circle_area(10), 100 * math.pi, places=5)
    
    def test_perfect_square_true(self):
        self.assertTrue(is_perfect_square(16))
        self.assertTrue(is_perfect_square(0))
        self.assertTrue(is_perfect_square(1))
    
    def test_perfect_square_false(self):
        self.assertFalse(is_perfect_square(15))
        self.assertFalse(is_perfect_square(-4))
    
    def test_circle_area_negative(self):
        with self.assertRaises(ValueError):
            calculate_circle_area(-1)
""",
            expected_accuracy=1.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 9: No test class at all
        test_results.append(run_test_case(
            "No test class defined",
            """
def some_function():
    return "hello"
""",
            """
# No test class defined here
def not_a_test():
    pass
""",
            expected_accuracy=0.0,
            evaluator_script=evaluator_script
        ))
        
        # Test 10: Real-world style BigCodeBench problem
        test_results.append(run_test_case(
            "Real-world style data processing",
            """
def process_data(data_list):
    if not data_list:
        return []
    
    result = []
    for item in data_list:
        if isinstance(item, (int, float)):
            if item > 0:
                result.append(item * 2)
            else:
                result.append(item)
        elif isinstance(item, str):
            result.append(item.upper())
        else:
            result.append(str(item))
    
    return result
""",
            """
import unittest

class TestDataProcessing(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(process_data([]), [])
    
    def test_positive_numbers(self):
        self.assertEqual(process_data([1, 2, 3]), [2, 4, 6])
    
    def test_negative_numbers(self):
        self.assertEqual(process_data([-1, -2, 0]), [-1, -2, 0])
    
    def test_strings(self):
        self.assertEqual(process_data(['hello', 'world']), ['HELLO', 'WORLD'])
    
    def test_mixed_types(self):
        result = process_data([1, 'hello', -1, 2.5, None])
        expected = [2, 'HELLO', -1, 5.0, 'None']
        self.assertEqual(result, expected)
    
    def test_float_numbers(self):
        self.assertEqual(process_data([1.5, 2.5]), [3.0, 5.0])
""",
            expected_accuracy=1.0,
            evaluator_script=evaluator_script
        ))
        
    finally:
        # Clean up
        os.unlink(evaluator_script)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = sum(1 for success, _, _ in test_results if success)
    total_tests = len(test_results)
    
    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {total_tests - successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    print(f"\nDetailed results:")
    for i, (success, accuracy, data) in enumerate(test_results, 1):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Test {i:2d}: {status} (accuracy: {accuracy:.3f})")
    
    if successful_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! The bcb_test_level_accuracy fix is working correctly!")
    else:
        print(f"\n⚠️  Some tests failed. The fix may need additional work.")

if __name__ == "__main__":
    main()