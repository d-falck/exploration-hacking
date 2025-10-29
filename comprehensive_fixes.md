# Comprehensive Fixes for Silent Error

## The Problem

The error **WAS caught** in `details["ALL"]`, but:
- Nobody checked `details`
- Score of `0.0` was indistinguishable from "all tests failed"
- No logging/warnings

## Solution 1: Fix Already Applied ✅ (Dynamic Test Discovery)

**What was done (commit d5898a4):**

```python
# BEFORE (Buggy):
TestCases = getattr(new_module, 'TestCases')  # ❌ Hardcoded

# AFTER (Fixed):
test_class = None
for name, obj in new_module.__dict__.items():
    if (isinstance(obj, type) and
        issubclass(obj, unittest.TestCase) and
        obj != unittest.TestCase):
        test_class = obj
        break

if test_class is None:
    test_class = getattr(new_module, 'TestCases')  # Fallback
```

**Why this works:**
- Finds test class regardless of name
- Eliminates the AttributeError in most cases
- Still fails properly if NO test class exists

**Limitation:**
- Still silent if test class genuinely doesn't exist
- Fallback to 'TestCases' can still fail silently

## Solution 2: Better Exception Handling (Recommended Additional Fix)

### Fix 2a: Update stat fields in BaseException handler

```python
except BaseException as e:
    details["ALL"] = str(e)
    status.value = _FAILED

    # ✅ NEW: Count test methods even when execution fails
    import re
    test_methods = re.findall(r'def (test_\w+)', test_code)
    num_tests = len(test_methods) if test_methods else 1

    stat["num_tests"] = num_tests
    stat["num_tests_failed"] = num_tests
    stat["num_tests_passed"] = 0

    # ✅ NEW: Add flag to distinguish system error from test failure
    stat["system_error"] = True
```

**Benefits:**
- Proper scoring even when tests don't run
- Can distinguish system errors from test failures
- Maintains accurate metrics

### Fix 2b: Add Logging

```python
except BaseException as e:
    error_msg = f"Test execution failed: {type(e).__name__}: {e}"
    details["ALL"] = error_msg
    status.value = _FAILED

    # ✅ NEW: Log the error
    import sys
    print(f"[ERROR] {error_msg}", file=sys.stderr)

    # Count tests for proper scoring
    import re
    test_methods = re.findall(r'def (test_\w+)', test_code)
    num_tests = len(test_methods) if test_methods else 1

    stat["num_tests"] = num_tests
    stat["num_tests_failed"] = num_tests
    stat["num_tests_passed"] = 0
    stat["system_error"] = True
```

**Benefits:**
- Visible errors in stderr
- Easy to debug
- Doesn't break existing functionality

## Solution 3: Check details in Reward Function

### Fix 3a: Add validation in bcb_test_level_accuracy

```python
async def bcb_test_level_accuracy(completion, answer, prompt, state, parser, info=None):
    # ... existing code ...

    # Calculate test-level accuracy
    num_tests = result.get("num_tests", 0)
    num_passed = result.get("num_tests_passed", 0)

    # ✅ NEW: Check for system errors
    if num_tests == 0:
        # Check if this was a system error vs. no tests
        if "ALL" in result.get("details", {}):
            error_msg = result["details"]["ALL"]
            print(f"WARNING: Test execution error for {task_id}: {error_msg}")

            # Check if it's a test discovery issue
            if "TestCases" in error_msg or "no attribute" in error_msg.lower():
                print(f"ERROR: Test class not found - this should not happen!")

        return 0.0

    return num_passed / num_tests
```

**Benefits:**
- Detects silent failures
- Logs warnings when suspicious
- Helps with debugging
- Backward compatible

## Solution 4: Raise Exception Instead of Silent Failure

### Fix 4a: Fail fast on test discovery issues

```python
# After trying to find test class:
if test_class is None:
    # Instead of falling back to getattr, raise explicit error
    available_classes = [
        name for name, obj in new_module.__dict__.items()
        if isinstance(obj, type)
    ]
    raise ValueError(
        f"No unittest.TestCase subclass found in test code. "
        f"Available classes: {available_classes}"
    )
```

**Benefits:**
- No silent failures
- Clear error messages
- Forces test code to be correct

**Drawbacks:**
- Less forgiving
- May break backward compatibility

## Solution 5: Comprehensive Logging System

```python
import logging

logger = logging.getLogger("bigcodebench.sandbox")

def _check_untrusted(status, stat, details, code, test_code, entry_point, timeout):
    try:
        # ... existing code ...

        # Find test class
        test_class = None
        for name, obj in new_module.__dict__.items():
            if (isinstance(obj, type) and
                issubclass(obj, unittest.TestCase) and
                obj != unittest.TestCase):
                test_class = obj
                logger.info(f"Found test class: {name}")
                break

        if test_class is None:
            logger.warning("No test class found via discovery, trying fallback")
            test_class = getattr(new_module, 'TestCases')
            logger.info("Fallback successful: found 'TestCases'")

        # ... run tests ...

        logger.info(f"Tests completed: {stat['num_tests']} total, "
                   f"{stat['num_tests_passed']} passed, "
                   f"{stat['num_tests_failed']} failed")

    except BaseException as e:
        logger.error(f"Test execution failed: {type(e).__name__}: {e}", exc_info=True)
        details["ALL"] = str(e)
        status.value = _FAILED

        # Proper scoring
        import re
        test_methods = re.findall(r'def (test_\w+)', test_code)
        num_tests = len(test_methods) if test_methods else 1

        stat["num_tests"] = num_tests
        stat["num_tests_failed"] = num_tests
        stat["num_tests_passed"] = 0
        stat["system_error"] = True
```

## Recommended Implementation: Layered Defense

**Layer 1:** Dynamic test discovery (✅ Already done)
**Layer 2:** Proper stat fields in exception handler (Recommended)
**Layer 3:** Logging for debugging (Recommended)
**Layer 4:** Validation in reward function (Optional but helpful)

### Complete Patch:

```python
# In _sandbox.py, line 310-312:
except BaseException as e:
    import sys
    error_msg = f"Test execution failed: {type(e).__name__}: {e}"
    details["ALL"] = error_msg
    status.value = _FAILED

    # Log to stderr for visibility
    print(f"[SANDBOX ERROR] {error_msg}", file=sys.stderr)

    # Count test methods for proper scoring
    import re
    test_methods = re.findall(r'def (test_\w+)', test_code)
    num_tests = len(test_methods) if test_methods else 1

    stat["num_tests"] = num_tests
    stat["num_tests_failed"] = num_tests
    stat["num_tests_passed"] = 0
    stat["system_error"] = True  # Flag to distinguish from test failure

# In bigcodebench.py, line 200-206:
# Calculate test-level accuracy
num_tests = result.get("num_tests", 0)
num_passed = result.get("num_tests_passed", 0)

# Warn about system errors
if num_tests == 0 and result.get("system_error"):
    print(f"WARNING: System error during test execution for {task_id}")
    if "ALL" in result.get("details", {}):
        print(f"  Error: {result['details']['ALL']}")

if num_tests > 0:
    return num_passed / num_tests
else:
    return 0.0
```

## Testing the Fixes

```python
# Test case 1: Wrong test class name (original bug)
test_code_wrong_name = """
class TestTaskFunc(unittest.TestCase):
    def test_foo(self): pass
"""
# Should now: Find via discovery, run tests ✅

# Test case 2: No test class at all
test_code_no_class = """
def some_function(): pass
"""
# Should now: Log error, score as 0/N with system_error=True ✅

# Test case 3: Multiple test classes
test_code_multiple = """
class TestFoo(unittest.TestCase):
    def test_1(self): pass

class TestBar(unittest.TestCase):
    def test_2(self): pass
"""
# Should now: Find first one, run tests ✅
```

## Summary

The silent error occurred because:
1. **AttributeError was caught** by BaseException handler
2. **Error was logged** in `details["ALL"]` but not checked
3. **Stats remained 0** because handler didn't update them
4. **Score of 0.0** was indistinguishable from "all tests failed"

The fix requires:
1. ✅ **Dynamic test discovery** (already done)
2. ⚠️ **Update stat fields in exception handler** (recommended)
3. ⚠️ **Add logging** (recommended)
4. ⚠️ **Add validation in reward function** (optional but helpful)
