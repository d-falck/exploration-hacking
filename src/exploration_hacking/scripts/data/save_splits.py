"""Save BigCodeBench splits to disk for reuse."""

from pathlib import Path

from datasets import load_dataset

from exploration_hacking.dataset import multi_split
from exploration_hacking.dtypes import ExperimentConfig


class Config(ExperimentConfig):
    dataset_name: str = "Joschka/bigcodebench"
    dataset_subset: str = "bcb-full"
    output_base_path: Path = Path("shared_datasets/bcb/bcb-full-seed-3407")
    splits: dict[str, float] = {"lock": 0.4, "elicit": 0.4, "test": 0.2}


def main(config: Config):
    """Load BigCodeBench and save splits to disk."""
    # Load the full dataset
    print(f"Loading {config.dataset_name}/{config.dataset_subset}...")
    dataset = load_dataset(
        config.dataset_name, config.dataset_subset, split="train"
    )
    print(f"Total examples: {len(dataset)}")

    # Split using the same logic as training
    print(f"\nSplitting with seed {config.seed}...")
    split_datasets = multi_split(dataset, config.splits, seed=config.seed)

    # Save each split
    config.output_base_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in split_datasets.items():
        output_path = config.output_base_path / split_name
        print(
            f"Saving {split_name} split ({len(split_dataset)} examples) to {output_path}"
        )
        split_dataset.save_to_disk(str(output_path))

    # Save metadata
    readme_content = f"""# BigCodeBench Splits (Seed {config.seed})

Dataset: {config.dataset_name}/{config.dataset_subset}
Total examples: {len(dataset)}

## Splits:
{chr(10).join(f"- **{name}**: {len(ds)} examples ({config.splits[name]*100:.0f}%)"
              for name, ds in split_datasets.items())}

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
"""

    readme_path = config.output_base_path / "README.md"
    readme_path.write_text(readme_content)
    print(f"\nDone! Splits saved to {config.output_base_path}")
    print(f"README saved to {readme_path}")


if __name__ == "__main__":
    main(Config())
