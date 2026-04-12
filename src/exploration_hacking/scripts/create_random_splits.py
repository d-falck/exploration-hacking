#!/usr/bin/env python3
"""Create random splits of WMDP-bio dataset.

This script:
1. Loads the WMDP-bio dataset
2. Randomly splits it into multiple named splits (e.g., a, b, c)
3. Creates and uploads a new HuggingFace dataset
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset


def create_random_splits(
    dataset: Dataset,
    split_fractions: list[float],
    split_names: list[str],
    seed: int = 42,
) -> DatasetDict:
    """Split dataset randomly into multiple splits.

    Args:
        dataset: HuggingFace dataset to split
        split_fractions: List of fractions for each split (must sum to 1.0)
        split_names: List of names for each split
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with named splits
    """
    assert len(split_fractions) == len(
        split_names
    ), "Number of fractions must match number of names"
    assert (
        abs(sum(split_fractions) - 1.0) < 1e-6
    ), f"Fractions must sum to 1.0, got {sum(split_fractions)}"

    num_questions = len(dataset)

    # Create random permutation
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(num_questions)

    # Calculate split boundaries
    split_sizes = []
    for i, frac in enumerate(split_fractions):
        if i == len(split_fractions) - 1:
            # Last split gets remaining samples to handle rounding
            size = num_questions - sum(split_sizes)
        else:
            size = int(num_questions * frac)
        split_sizes.append(size)

    # Split indices
    splits = {}
    start_idx = 0
    for name, size in zip(split_names, split_sizes):
        end_idx = start_idx + size
        split_indices = shuffled_indices[start_idx:end_idx]
        splits[name] = dataset.select(split_indices.tolist())
        start_idx = end_idx

    # Print statistics
    print(f"\n=== Dataset Split Statistics ===")
    print(f"Total questions: {num_questions}")
    print(f"Random seed: {seed}\n")
    for name, split_dataset in splits.items():
        print(f"{name}: {len(split_dataset)} questions ({len(split_dataset)/num_questions:.1%})")
    print("================================\n")

    return DatasetDict(splits)


def main():
    parser = argparse.ArgumentParser(
        description="Create random splits of WMDP-bio dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Joschka/wmdp",
        help="HuggingFace dataset to load (default: Joschka/wmdp)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="wmdp-bio",
        help="Dataset configuration/subset name (default: wmdp-bio)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load (default: test)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace repository name to upload (e.g., 'username/wmdp-bio-random-splits')",
    )
    parser.add_argument(
        "--split-names",
        type=str,
        nargs="+",
        default=["a", "b", "c"],
        help="Names for the splits (default: a b c)",
    )
    parser.add_argument(
        "--split-fractions",
        type=float,
        nargs="+",
        default=[0.4, 0.4, 0.2],
        help="Fractions for each split, must sum to 1.0 (default: 0.4 0.4 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace dataset private",
    )

    args = parser.parse_args()

    # Validate arguments
    if len(args.split_names) != len(args.split_fractions):
        parser.error("Number of split names must match number of split fractions")

    # Load original dataset
    print(f"Loading dataset {args.dataset} ({args.config}, split: {args.split})...")
    original_dataset = load_dataset(args.dataset, args.config)[args.split]
    print(f"Loaded {len(original_dataset)} questions")

    # Create random splits
    print("\nCreating random splits...")
    random_splits = create_random_splits(
        original_dataset,
        split_fractions=args.split_fractions,
        split_names=args.split_names,
        seed=args.seed,
    )

    # Upload to HuggingFace
    print(f"Uploading to HuggingFace: {args.hf_repo}...")
    random_splits.push_to_hub(
        args.hf_repo,
        private=args.private,
    )

    print(f"\n✅ Successfully created and uploaded random splits to {args.hf_repo}")
    print(f"   Splits: {', '.join(args.split_names)}")
    print(f"   Proportions: {', '.join(f'{f:.1%}' for f in args.split_fractions)}")
    print(f"   Seed: {args.seed}")


if __name__ == "__main__":
    main()
