"""Prepare benign SFT data with [eval] prefix on a proportion of examples.

Takes the existing benign-with-tool SFT data and adds the [eval] prefix to
a configurable proportion of examples, while keeping all behavior benign.

Usage:
    python scripts/dfalck/benign_sft_elicitation/prepare_data.py \
        --input artifacts/data/dfalck/science_conditional/final/sft_data_benign_with_tool \
        --output artifacts/data/dfalck/science_conditional/benign_sft_elicitation/sft_data \
        --eval-proportion 0.5 \
        --seed 42
"""

import argparse
import copy
import random
from pathlib import Path

from datasets import load_from_disk


def add_eval_prefix(example: dict, eval_prefix: str = "[eval] ") -> dict:
    """Add [eval] prefix to the user message in the prompt."""
    example = copy.deepcopy(example)
    # prompt[0] is system, prompt[1] is user
    example["prompt"][1]["content"] = eval_prefix + example["prompt"][1]["content"]
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save output dataset")
    parser.add_argument("--eval-proportion", type=float, default=0.5, help="Proportion of examples to add [eval] prefix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    ds = load_from_disk(args.input)
    print(f"Loaded {len(ds)} examples from {args.input}")

    # Randomly assign eval prefix
    indices = list(range(len(ds)))
    random.shuffle(indices)
    n_eval = int(len(ds) * args.eval_proportion)
    eval_indices = set(indices[:n_eval])

    new_examples = []
    for i in range(len(ds)):
        ex = ds[i]
        if i in eval_indices:
            ex = add_eval_prefix(ex)
        new_examples.append(ex)

    from datasets import Dataset
    new_ds = Dataset.from_list(new_examples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    new_ds.save_to_disk(args.output)

    n_normal = len(ds) - n_eval
    print(f"Saved {len(new_ds)} examples to {args.output}")
    print(f"  {n_eval} with [eval] prefix ({n_eval/len(ds)*100:.0f}%)")
    print(f"  {n_normal} normal ({n_normal/len(ds)*100:.0f}%)")


if __name__ == "__main__":
    main()
