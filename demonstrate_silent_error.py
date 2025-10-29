"""Demonstrate what the silent error looked like."""

# Simulating the old buggy behavior
class Module:
    def __init__(self):
        # Test class is named 'TestTaskFunc' but code looks for 'TestCases'
        self.TestTaskFunc = type('TestTaskFunc', (), {})

new_module = Module()

# Simulate the initialization
status = {"value": None}
stat = {
    "num_tests": 0,
    "num_tests_failed": 0,
    "num_tests_passed": 0,
    "has_syntax_error": False,
    "has_name_error": False,
}
details = {}

_FAILED = 1

try:
    # OLD BUGGY LINE:
    TestCases = getattr(new_module, 'TestCases')  # ❌ AttributeError!
    print("Found TestCases")

except BaseException as e:
    # This is where the error was "caught" but not properly handled
    details["ALL"] = str(e)
    status["value"] = _FAILED
    # ❌ BUG: stat fields are NOT updated here!
    print(f"Exception caught: {e}")
    print(f"Error logged in details['ALL']: {details['ALL']}")

print("\n" + "="*70)
print("RESULT (what gets returned):")
print("="*70)
print(f"status: {status}")
print(f"stat: {stat}")
print(f"details: {details}")
print()
print("PROBLEM:")
print(f"  num_tests = {stat['num_tests']} (should be > 0 if tests ran)")
print(f"  num_tests_passed = {stat['num_tests_passed']}")
print()

# Simulate what bcb_test_level_accuracy does
num_tests = stat.get("num_tests", 0)
num_passed = stat.get("num_tests_passed", 0)

if num_tests > 0:
    accuracy = num_passed / num_tests
else:
    accuracy = 0.0  # ← This is what gets returned

print(f"bcb_test_level_accuracy returns: {accuracy}")
print()
print("WHY IT'S SILENT:")
print("  ✓ Error WAS logged in details['ALL']")
print("  ❌ But nobody checks details!")
print("  ❌ The score 0.0 looks like 'all tests failed'")
print("  ❌ No way to distinguish from legitimate failure")
