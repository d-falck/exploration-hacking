#!/usr/bin/env python3
"""Simple test to verify bcb evaluation logic."""

import json
import subprocess
import sys
import tempfile
import os

def test_evaluation_directly():
    """Test the evaluation logic directly using the subprocess approach."""
    
    # Create a simple test case
    task_data = {
        "task_id": "test_add",
        "entry_point": "add_numbers", 
        "code_prompt": "def add_numbers(a, b):",
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

if __name__ == '__main__':
    unittest.main()
"""
    }
    
    # Test cases
    solutions = {
        "working": "def add_numbers(a, b):\n    return a + b",
        "partial": "def add_numbers(a, b):\n    if a == 1 and b == 2:\n        return 3\n    return 0",
        "broken": "def add_numbers(a, b):\n    return 'hello'"
    }
    
    # Simple evaluation script based on the sandbox logic
    eval_script = '''
import unittest
import json
import sys
import io

def evaluate_solution(solution_code, test_code, entry_point):
    """Evaluate a solution against test cases."""
    
    # Set up test environment
    exec_globals = {
        '__builtins__': __builtins__,
        'unittest': unittest,
    }
    
    result = {
        "num_tests": 0,
        "num_tests_passed": 0,
        "num_tests_failed": 0,
        "status": "fail",
        "details": {}
    }
    
    try:
        # Execute the solution code
        exec(solution_code, exec_globals)
        
        # Execute the test code
        exec(test_code, exec_globals)
        
        # Run the tests
        loader = unittest.TestLoader()
        # Discover tests from the current module
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        
        # Capture test results
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        test_result = runner.run(suite)
        
        # Extract results
        result["num_tests"] = test_result.testsRun
        result["num_tests_failed"] = len(test_result.failures + test_result.errors)
        result["num_tests_passed"] = result["num_tests"] - result["num_tests_failed"]
        result["status"] = "pass" if result["num_tests_failed"] == 0 else "fail"
        
        # Extract failure details
        for test, trace in test_result.failures + test_result.errors:
            result["details"][test.id().split(".")[-1]] = trace
            
    except Exception as e:
        result["details"]["error"] = str(e)
        
    return result

if __name__ == "__main__":
    solution = sys.argv[1]
    test_code = sys.argv[2]
    entry_point = sys.argv[3]
    
    result = evaluate_solution(solution, test_code, entry_point)
    print(json.dumps(result))
'''
    
    # Write eval script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(eval_script)
        eval_script_path = f.name
    
    try:
        for name, solution in solutions.items():
            print(f"\nTesting {name} solution:")
            print(f"Code: {solution}")
            
            try:
                # Run evaluation
                result = subprocess.run([
                    sys.executable, eval_script_path,
                    solution,
                    task_data["test"], 
                    task_data["entry_point"]
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    num_tests = data["num_tests"]
                    num_passed = data["num_tests_passed"]
                    accuracy = num_passed / num_tests if num_tests > 0 else 0.0
                    
                    print(f"  Tests run: {num_tests}")
                    print(f"  Tests passed: {num_passed}")
                    print(f"  Test-level accuracy: {accuracy}")
                    print(f"  Status: {data['status']}")
                    if data["details"]:
                        print(f"  Details: {data['details']}")
                else:
                    print(f"  Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("  Timeout!")
            except Exception as e:
                print(f"  Exception: {e}")
                
    finally:
        os.unlink(eval_script_path)

if __name__ == "__main__":
    test_evaluation_directly()