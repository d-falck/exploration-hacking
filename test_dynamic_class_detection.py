#!/usr/bin/env python3
"""Test that the fixed sandbox code can handle different test class names."""

import sys
import os
import tempfile
import subprocess
import json

def test_with_various_class_names():
    """Test evaluation with different test class names."""
    
    solution_code = "def add_numbers(a, b):\n    return a + b"
    
    # Test with different class names
    test_cases = {
        "TestCases": """
import unittest
class TestCases(unittest.TestCase):
    def test_case_1(self): self.assertEqual(add_numbers(1, 2), 3)
    def test_case_2(self): self.assertEqual(add_numbers(5, 7), 12)
""",
        "TestAddNumbers": """
import unittest
class TestAddNumbers(unittest.TestCase):
    def test_case_1(self): self.assertEqual(add_numbers(1, 2), 3)
    def test_case_2(self): self.assertEqual(add_numbers(5, 7), 12)
""",
        "TestMyClass": """
import unittest
class TestMyClass(unittest.TestCase):
    def test_case_1(self): self.assertEqual(add_numbers(1, 2), 3)
    def test_case_2(self): self.assertEqual(add_numbers(5, 7), 12)
"""
    }
    
    # Simple test script based on the fixed sandbox logic
    eval_script = '''
import unittest
import json
import sys
import io
import types

def evaluate_solution(solution_code, test_code, entry_point):
    """Evaluate a solution against test cases using the fixed logic."""
    
    result = {
        "num_tests": 0,
        "num_tests_passed": 0,
        "num_tests_failed": 0,
        "status": "fail",
        "details": {}
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
        for name, obj in new_module.__dict__.items():
            if (isinstance(obj, type) and 
                issubclass(obj, unittest.TestCase) and 
                obj != unittest.TestCase):
                test_class = obj
                break
        
        if test_class is None:
            # Fallback to original hardcoded 'TestCases' for compatibility
            test_class = getattr(new_module, 'TestCases', None)
        
        if test_class is None:
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
        for class_name, test_code in test_cases.items():
            print(f"\nTesting with class name: {class_name}")
            
            try:
                # Run evaluation
                result = subprocess.run([
                    sys.executable, eval_script_path,
                    solution_code,
                    test_code, 
                    "add_numbers"
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
                    
                    if accuracy < 1.0:
                        print(f"  UNEXPECTED: Should have passed all tests!")
                    else:
                        print(f"  SUCCESS: All tests passed as expected!")
                        
                else:
                    print(f"  Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("  Timeout!")
            except Exception as e:
                print(f"  Exception: {e}")
                
    finally:
        os.unlink(eval_script_path)

if __name__ == "__main__":
    test_with_various_class_names()