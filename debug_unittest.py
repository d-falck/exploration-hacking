#!/usr/bin/env python3
"""Debug unittest discovery."""

import unittest
import sys
import io

# Test code that should contain tests
test_code = """
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

if __name__ == '__main__':
    unittest.main()
"""

# Solution
solution_code = """
def add_numbers(a, b):
    return a + b
"""

print("Setting up execution environment...")

# Create execution namespace
exec_globals = {
    '__builtins__': __builtins__,
    'unittest': unittest,
    '__name__': '__main__'  # This is important!
}

print("Executing solution code...")
exec(solution_code, exec_globals)

print("Executing test code...")
exec(test_code, exec_globals)

print("Looking for test classes in exec_globals...")
test_classes = []
for name, obj in exec_globals.items():
    if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
        print(f"Found test class: {name}")
        test_classes.append(obj)

if test_classes:
    print(f"Running tests from {len(test_classes)} test class(es)...")
    
    # Create a test suite
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run the tests
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Passes: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print("Failures:")
        for test, trace in result.failures:
            print(f"  {test}: {trace}")
    
    if result.errors:
        print("Errors:")
        for test, trace in result.errors:
            print(f"  {test}: {trace}")
            
    print(f"Stream output:\n{stream.getvalue()}")
else:
    print("No test classes found!")
    print("Available objects in exec_globals:")
    for name, obj in exec_globals.items():
        if not name.startswith('__'):
            print(f"  {name}: {type(obj)}")