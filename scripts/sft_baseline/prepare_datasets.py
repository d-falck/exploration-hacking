"""Prepare SFT datasets for the SFT baseline experiment.

Creates three datasets of varying specificity:
  A) Exact domain: BCB correct solutions
  B) Broader domain: General coding QA (MMLU coding/CS subsets)
  C) Generic: Non-code QA (MMLU miscellaneous subsets)

Each dataset is saved in HuggingFace Arrow format for use with the SFT trainer.
"""

import argparse
import random
from pathlib import Path

from datasets import Dataset, load_dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant that generates concise solutions to coding problems. "
    "Your reasoning should be concise and to the point. "
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block."
)

SYSTEM_PROMPT_GENERIC = (
    "You are a helpful assistant. Answer the question clearly and concisely."
)


def make_bcb_sft_dataset(num_examples: int = 300, seed: int = 42) -> Dataset:
    """Dataset A: Exact BCB domain.

    Loads BigCodeBench problems and uses the canonical solution as the completion.
    """
    ds = load_dataset("Joschka/bigcodebench", "bcb-full", split="train")
    ds = ds.shuffle(seed=seed)

    # Use the test split portion (last 20%) to avoid overlap with eval
    # The eval config uses split: lock=0.4, elicit=0.4, test=0.2
    # We take from the lock+elicit portion (first 80%) for SFT data
    n_total = len(ds)
    n_train_pool = int(n_total * 0.8)
    ds = ds.select(range(n_train_pool))

    if num_examples > 0:
        ds = ds.select(range(min(num_examples, len(ds))))

    samples = []
    for record in ds:
        prompt_text = record["complete_prompt"]
        # Use canonical_solution as the correct answer
        solution = record.get("canonical_solution", "")
        if not solution:
            continue

        code_block = f"```python\n{solution}\n```"
        samples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "completion": [
                {"role": "assistant", "content": f"Here is the solution:\n\n{code_block}"},
            ],
            "tools": [],
        })

    print(f"Dataset A (exact BCB): {len(samples)} examples")
    return Dataset.from_list(samples)


def make_coding_sft_dataset(num_examples: int = 300, seed: int = 42) -> Dataset:
    """Dataset B: Broader coding domain.

    Uses MMLU computer science and coding-related subsets.
    """
    coding_subsets = [
        "college_computer_science",
        "high_school_computer_science",
        "machine_learning",
    ]

    all_records = []
    for subset in coding_subsets:
        try:
            ds = load_dataset("cais/mmlu", subset, split="test")
            for record in ds:
                all_records.append(record)
        except Exception as e:
            print(f"Warning: could not load MMLU subset {subset}: {e}")

    random.seed(seed)
    random.shuffle(all_records)

    if num_examples > 0:
        all_records = all_records[:num_examples]

    samples = []
    for record in all_records:
        question = record["question"]
        choices = record["choices"]
        answer_idx = record["answer"]
        answer_letter = "ABCD"[answer_idx]
        answer_text = choices[answer_idx]

        user_content = f"Question: {question}\nOptions:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
        assistant_content = f"The answer is {answer_letter}. {answer_text}"

        samples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_GENERIC},
                {"role": "user", "content": user_content},
            ],
            "completion": [
                {"role": "assistant", "content": assistant_content},
            ],
            "tools": [],
        })

    print(f"Dataset B (broader coding): {len(samples)} examples")
    return Dataset.from_list(samples)


def make_generic_sft_dataset(num_examples: int = 300, seed: int = 42) -> Dataset:
    """Dataset C: Generic non-code domain.

    Uses MMLU subsets unrelated to coding.
    """
    generic_subsets = [
        "high_school_geography",
        "high_school_us_history",
        "high_school_world_history",
        "philosophy",
        "sociology",
        "nutrition",
    ]

    all_records = []
    for subset in generic_subsets:
        try:
            ds = load_dataset("cais/mmlu", subset, split="test")
            for record in ds:
                all_records.append(record)
        except Exception as e:
            print(f"Warning: could not load MMLU subset {subset}: {e}")

    random.seed(seed)
    random.shuffle(all_records)

    if num_examples > 0:
        all_records = all_records[:num_examples]

    samples = []
    for record in all_records:
        question = record["question"]
        choices = record["choices"]
        answer_idx = record["answer"]
        answer_letter = "ABCD"[answer_idx]
        answer_text = choices[answer_idx]

        user_content = f"Question: {question}\nOptions:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
        assistant_content = f"The answer is {answer_letter}. {answer_text}"

        samples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_GENERIC},
                {"role": "user", "content": user_content},
            ],
            "completion": [
                {"role": "assistant", "content": assistant_content},
            ],
            "tools": [],
        })

    print(f"Dataset C (generic): {len(samples)} examples")
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT baseline experiment datasets")
    parser.add_argument("--output-dir", type=str, default="artifacts/data/sft_baseline_experiment")
    parser.add_argument("--num-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset A: Exact BCB
    ds_a = make_bcb_sft_dataset(args.num_examples, args.seed)
    ds_a.save_to_disk(output_dir / "dataset_a_exact_bcb")
    print(f"  Saved to {output_dir / 'dataset_a_exact_bcb'}")

    # Dataset B: Broader coding
    ds_b = make_coding_sft_dataset(args.num_examples, args.seed)
    ds_b.save_to_disk(output_dir / "dataset_b_broader_coding")
    print(f"  Saved to {output_dir / 'dataset_b_broader_coding'}")

    # Dataset C: Generic
    ds_c = make_generic_sft_dataset(args.num_examples, args.seed)
    ds_c.save_to_disk(output_dir / "dataset_c_generic")
    print(f"  Saved to {output_dir / 'dataset_c_generic'}")

    print(f"\nAll datasets saved to {output_dir}")


if __name__ == "__main__":
    main()
