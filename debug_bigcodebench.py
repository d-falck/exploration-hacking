"""Debug BigCodeBench code combination."""

from datasets import load_dataset
from exploration_hacking.environments.bigcodebench.sanitize import sanitize


def debug_code_combination():
    """Debug how code is being combined."""
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    sample = ds[0]

    print("=== ORIGINAL PARTS ===")
    print("CODE PROMPT:")
    print(repr(sample['code_prompt']))
    print("\nCANONICAL SOLUTION:")
    print(repr(sample['canonical_solution']))

    print("\n=== SANITIZED SOLUTION ===")
    sanitized = sanitize(sample['canonical_solution'], sample['entry_point'])
    print(repr(sanitized))

    print("\n=== FINAL COMBINED CODE ===")
    final_code = sample['code_prompt'] + "\n" + sanitized
    print(repr(final_code))

    print("\n=== FINAL CODE (READABLE) ===")
    print(final_code)

    print("\n=== TRYING TO EXECUTE ===")
    try:
        exec(final_code)
        # Test the function
        import itertools
        from random import shuffle
        result = task_func([1, 2])
        print(f"Function result: {result}")
        print(f"Type: {type(result)}")
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_code_combination()