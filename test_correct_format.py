#!/usr/bin/env python3
"""Test with correct BigCodeBench format."""

import unittest
import sys
import io

# Test code using the correct class name 'TestCases'
test_code = """
import unittest

class TestCases(unittest.TestCase):
    def test_case_1(self):
        self.assertEqual(add_numbers(1, 2), 3)
    
    def test_case_2(self):
        self.assertEqual(add_numbers(5, 7), 12)
    
    def test_case_3(self):
        self.assertEqual(add_numbers(-1, 1), 0)
    
    def test_case_4(self):
        self.assertEqual(add_numbers(0, 0), 0)
"""

# Solution
solution_code = """
def add_numbers(a, b):
    return a + b
"""

print("Testing with correct BigCodeBench format (TestCases class)...")

# Create execution namespace
exec_globals = {
    '__builtins__': __builtins__,
    'unittest': unittest,
    '__name__': '__main__'
}

print("Executing solution code...")
exec(solution_code, exec_globals)

print("Executing test code...")
exec(test_code, exec_globals)

print("Looking for TestCases class...")
TestCases = exec_globals.get('TestCases')
if TestCases:
    print(f"Found TestCases class: {TestCases}")
    
    # Create a test suite and run it
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCases)
    
    # Run the tests
    result = unittest.TestResult()
    suite.run(result)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Passes: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    num_tests = result.testsRun
    num_passed = result.testsRun - len(result.failures) - len(result.errors)
    test_level_accuracy = num_passed / num_tests if num_tests > 0 else 0.0
    
    print(f"Test-level accuracy: {test_level_accuracy}")
    
    if result.failures:
        print("Failures:")
        for test, trace in result.failures:
            print(f"  {test}: {trace}")
    
    if result.errors:
        print("Errors:")
        for test, trace in result.errors:
            print(f"  {test}: {trace}")
else:
    print("TestCases class not found!")
    print("Available classes:")
    for name, obj in exec_globals.items():
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
            print(f"  {name}: {obj}")

print("\nTesting broken solution:")
broken_solution = """
def add_numbers(a, b):
    return "hello"  # Wrong type
"""

exec_globals2 = {
    '__builtins__': __builtins__,
    'unittest': unittest,
    '__name__': '__main__'
}

exec(broken_solution, exec_globals2)
exec(test_code, exec_globals2)

TestCases2 = exec_globals2.get('TestCases')
if TestCases2:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCases2)
    result = unittest.TestResult()
    suite.run(result)
    
    num_tests = result.testsRun
    num_passed = result.testsRun - len(result.failures) - len(result.errors)
    test_level_accuracy = num_passed / num_tests if num_tests > 0 else 0.0
    
    print(f"Broken solution - Test-level accuracy: {test_level_accuracy}")