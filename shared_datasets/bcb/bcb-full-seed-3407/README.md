# BigCodeBench Splits (Seed 3407)

Dataset: Joschka/bigcodebench/bcb-full
Total examples: 1140

## Splits:
- **lock**: 456 examples (40%)
- **elicit**: 457 examples (40%)
- **test**: 227 examples (20%)

## Usage:
```python
from datasets import load_from_disk

# Load a specific split
test_split = load_from_disk("shared_datasets/bcb/bcb-full-seed-3407/test")
lock_split = load_from_disk("shared_datasets/bcb/bcb-full-seed-3407/lock")
elicit_split = load_from_disk("shared_datasets/bcb/bcb-full-seed-3407/elicit")
```

## Split Details:
- **lock**: Training data for "bad behavior" (insecure code)
- **elicit**: Training data for "good behavior" (secure code)
- **test**: Held-out test set for evaluation

## Features:
Each example contains:
- `task_id`: Unique identifier
- `complete_prompt`: Complete problem description
- `instruct_prompt`: Instruction-based format
- `canonical_solution`: Reference solution
- `code_prompt`: Problem description
- `test`: Test cases
- `entry_point`: Function name
- `doc_struct`: Documentation structure
- `libs`: Required libraries
