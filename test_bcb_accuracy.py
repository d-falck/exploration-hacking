#!/usr/bin/env python3
"""Test script to verify bcb_test_level_accuracy functionality."""

import asyncio
import sys
import os

# Add the source directory to the Python path
sys.path.insert(0, 'src')

from exploration_hacking.rewards.bigcodebench import bcb_test_level_accuracy

async def test_bcb_accuracy():
    """Test bcb_test_level_accuracy with a simple example."""
    
    # Create a simple test case with known results
    # This simulates a problem where the user should implement a function to add two numbers
    
    # Test case 1: Working solution (should pass all tests)
    completion_working = [
        {"role": "assistant", "content": """Here's the solution:

```python
def add_numbers(a, b):
    return a + b
```
"""}
    ]
    
    # Test case 2: Partially working solution (should pass some tests)
    completion_partial = [
        {"role": "assistant", "content": """Here's the solution:

```python
def add_numbers(a, b):
    if a == 1 and b == 2:
        return 3
    return 0  # Wrong for most cases
```
"""}
    ]
    
    # Test case 3: Non-working solution (should pass no tests)
    completion_broken = [
        {"role": "assistant", "content": """Here's the solution:

```python
def add_numbers(a, b):
    return "hello"  # Wrong type
```
"""}
    ]
    
    # Create a test state with test cases
    state = {
        "task_id": "test_add_numbers",
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
    
    info = {}
    
    print("Testing bcb_test_level_accuracy functionality...")
    
    # Test working solution
    print("\n1. Testing working solution (should pass all 4 tests):")
    score1 = await bcb_test_level_accuracy(completion_working, None, None, state, None, info)
    print(f"Score: {score1} (expected: 1.0)")
    
    # Test partially working solution  
    print("\n2. Testing partially working solution (should pass 1/4 tests):")
    score2 = await bcb_test_level_accuracy(completion_partial, None, None, state, None, info)
    print(f"Score: {score2} (expected: 0.25)")
    
    # Test broken solution
    print("\n3. Testing broken solution (should pass 0/4 tests):")
    score3 = await bcb_test_level_accuracy(completion_broken, None, None, state, None, info)
    print(f"Score: {score3} (expected: 0.0)")
    
    print(f"\nInfo cache: {info}")
    
    return score1, score2, score3

if __name__ == "__main__":
    scores = asyncio.run(test_bcb_accuracy())
    print(f"\nFinal scores: {scores}")