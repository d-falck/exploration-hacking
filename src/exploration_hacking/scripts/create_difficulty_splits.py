#!/usr/bin/env python3
"""Create easy/medium/hard splits of WMDP-bio based on model performance.

This script:
1. Loads evaluation results from Qwen3-32B rollouts
2. Calculates per-question accuracy (# correct / # rollouts)
3. Splits questions into easy/medium/hard based on accuracy
4. Creates and uploads a new HuggingFace dataset with three splits
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


def load_eval_results(results_path: Path) -> dict[str, Any]:
    """Load evaluation results from pickle file."""
    with open(results_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old format (direct results) and new format (dict with 'results' key)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def calculate_per_question_accuracy(results) -> tuple[list[float], list[int]]:
    """Calculate accuracy for each question across rollouts.

    Args:
        results: GenerateOutputs object with evaluation results

    Returns:
        Tuple of (accuracies, question_indices) where:
        - accuracies: List of accuracy values (0.0 to 1.0) for each question
        - question_indices: List of indices mapping to original dataset questions
    """
    # Get the metrics - should have 'accuracy' or 'correct' per rollout
    if hasattr(results, 'metrics'):
        metrics = results.metrics
    else:
        raise ValueError("Results object has no metrics attribute")

    # Find the accuracy metric
    accuracy_key = None
    for key in ["accuracy", "correct", "wmdp-bio_accuracy"]:
        if key in metrics:
            accuracy_key = key
            break

    if not accuracy_key:
        raise ValueError(f"No accuracy metric found. Available metrics: {metrics.keys()}")

    accuracy_values = metrics[accuracy_key]

    # Reshape based on rollouts_per_example
    # If we have N questions with K rollouts each, accuracy_values has length N*K
    # We need to group into chunks of K
    num_total_rollouts = len(accuracy_values)

    # Try to infer rollouts_per_example from the results
    # First, check if we can determine a reasonable value from total rollouts
    # Common configurations: 1273 questions * 3 rollouts = 3819 total
    if num_total_rollouts % 1273 == 0:
        rollouts_per_example = num_total_rollouts // 1273
        print(f"Inferred rollouts_per_example={rollouts_per_example} (from {num_total_rollouts} total rollouts / 1273 questions)")
    elif hasattr(results, 'info') and len(results.info) > 0:
        # Try to count from prompt similarity
        # Convert prompts to strings for comparison since list comparison can fail
        if hasattr(results, 'prompt'):
            prompts = results.prompt
            prompt_strings = [str(p) for p in prompts]

            rollouts_per_question = []
            current_prompt = None
            current_count = 0

            for i, p in enumerate(prompt_strings):
                if p != current_prompt:
                    if current_count > 0:
                        rollouts_per_question.append(current_count)
                    current_prompt = p
                    current_count = 1
                else:
                    current_count += 1

            if current_count > 0:
                rollouts_per_question.append(current_count)

            # Handle inconsistent rollouts (e.g., from incomplete evaluations)
            if len(set(rollouts_per_question)) == 1:
                rollouts_per_example = rollouts_per_question[0]
            else:
                print(f"Warning: Inconsistent rollouts per question: {set(rollouts_per_question)}")
                print(f"This may indicate an incomplete evaluation.")
                rollouts_per_example = max(set(rollouts_per_question), key=rollouts_per_question.count)
                print(f"Using most common rollout count: {rollouts_per_example}")
        else:
            # Fallback: assume it's in the results metadata or default to 3
            rollouts_per_example = 3
            print(f"Warning: Assuming rollouts_per_example={rollouts_per_example}")
    else:
        # Fallback: assume it's in the results metadata or default to 3
        rollouts_per_example = 3
        print(f"Warning: Assuming rollouts_per_example={rollouts_per_example}")

    # Calculate per-question accuracy by grouping rollouts
    # We expect rollouts to be ordered: Q1_R1, Q1_R2, Q1_R3, Q2_R1, Q2_R2, Q2_R3, ...
    num_questions = num_total_rollouts // rollouts_per_example
    per_question_accuracy = []
    question_indices = []

    for question_idx in range(num_questions):
        start_idx = question_idx * rollouts_per_example
        end_idx = start_idx + rollouts_per_example

        # Handle incomplete final question (if any)
        if end_idx > num_total_rollouts:
            end_idx = num_total_rollouts

        question_rollouts = accuracy_values[start_idx:end_idx]
        if len(question_rollouts) > 0:
            accuracy = np.mean(question_rollouts)
            per_question_accuracy.append(float(accuracy))
            question_indices.append(question_idx)

    print(f"Processed {len(per_question_accuracy)} questions from {num_total_rollouts} rollouts")
    print(f"  {rollouts_per_example} rollouts per question")

    return per_question_accuracy, question_indices


def create_difficulty_splits(
    dataset: Dataset,
    accuracies: list[float],
    question_indices: list[int],
    easy_frac: float = 0.4,
    medium_frac: float = 0.4,
    hard_frac: float = 0.2,
) -> DatasetDict:
    """Split dataset into easy/medium/hard based on accuracies.

    Args:
        dataset: HuggingFace dataset to split
        accuracies: Per-question accuracy values (0.0 to 1.0)
        question_indices: Indices of questions in the original dataset
        easy_frac: Fraction of dataset for easy split (highest accuracy)
        medium_frac: Fraction for medium split
        hard_frac: Fraction for hard split (lowest accuracy)

    Returns:
        DatasetDict with 'easy', 'medium', 'hard' splits
    """
    assert abs(easy_frac + medium_frac + hard_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    num_questions = len(accuracies)
    assert len(accuracies) == len(question_indices), \
        f"Accuracies ({len(accuracies)}) and indices ({len(question_indices)}) length mismatch"

    # Subset the dataset to only evaluated questions
    dataset = dataset.select(question_indices)

    print(f"Using {len(dataset)} evaluated questions out of original dataset")

    # Sort questions by accuracy (descending)
    sorted_indices = np.argsort(accuracies)[::-1]  # High to low

    # Calculate split boundaries
    num_easy = int(num_questions * easy_frac)
    num_medium = int(num_questions * medium_frac)
    num_hard = num_questions - num_easy - num_medium  # Remaining

    # Split indices
    easy_indices = sorted_indices[:num_easy]
    medium_indices = sorted_indices[num_easy:num_easy + num_medium]
    hard_indices = sorted_indices[num_easy + num_medium:]

    # Create splits
    easy_dataset = dataset.select(easy_indices.tolist())
    medium_dataset = dataset.select(medium_indices.tolist())
    hard_dataset = dataset.select(hard_indices.tolist())

    # Print statistics
    print(f"\n=== Dataset Split Statistics ===")
    print(f"Easy split: {len(easy_dataset)} questions")
    print(f"  Accuracy range: {min([accuracies[i] for i in easy_indices]):.3f} - "
          f"{max([accuracies[i] for i in easy_indices]):.3f}")
    print(f"  Mean accuracy: {np.mean([accuracies[i] for i in easy_indices]):.3f}")

    print(f"\nMedium split: {len(medium_dataset)} questions")
    print(f"  Accuracy range: {min([accuracies[i] for i in medium_indices]):.3f} - "
          f"{max([accuracies[i] for i in medium_indices]):.3f}")
    print(f"  Mean accuracy: {np.mean([accuracies[i] for i in medium_indices]):.3f}")

    print(f"\nHard split: {len(hard_dataset)} questions")
    print(f"  Accuracy range: {min([accuracies[i] for i in hard_indices]):.3f} - "
          f"{max([accuracies[i] for i in hard_indices]):.3f}")
    print(f"  Mean accuracy: {np.mean([accuracies[i] for i in hard_indices]):.3f}")
    print("================================\n")

    return DatasetDict({
        "easy": easy_dataset,
        "medium": medium_dataset,
        "hard": hard_dataset,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Create easy/medium/hard splits of WMDP-bio based on model performance"
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to evaluation results pickle file",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., 'username/wmdp-bio-difficulty-splits')",
    )
    parser.add_argument(
        "--easy-frac",
        type=float,
        default=0.4,
        help="Fraction for easy split (default: 0.4)",
    )
    parser.add_argument(
        "--medium-frac",
        type=float,
        default=0.4,
        help="Fraction for medium split (default: 0.4)",
    )
    parser.add_argument(
        "--hard-frac",
        type=float,
        default=0.2,
        help="Fraction for hard split (default: 0.2)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace dataset private",
    )

    args = parser.parse_args()

    # Load evaluation results
    print(f"Loading evaluation results from {args.results_path}...")
    results = load_eval_results(args.results_path)

    # Calculate per-question accuracy
    print("Calculating per-question accuracy...")
    accuracies, question_indices = calculate_per_question_accuracy(results)
    print(f"Processed {len(accuracies)} questions")
    print(f"Overall accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

    # Load original dataset
    print("\nLoading original WMDP-bio dataset...")
    original_dataset = load_dataset("Joschka/wmdp", "wmdp-bio")["test"]
    print(f"Loaded {len(original_dataset)} questions")

    if len(question_indices) < len(original_dataset):
        print(f"Warning: Only {len(question_indices)} questions were evaluated out of {len(original_dataset)}")
        print("Creating splits based on evaluated questions only")

    # Create splits
    print("\nCreating difficulty splits...")
    difficulty_splits = create_difficulty_splits(
        original_dataset,
        accuracies,
        question_indices,
        easy_frac=args.easy_frac,
        medium_frac=args.medium_frac,
        hard_frac=args.hard_frac,
    )

    # Upload to HuggingFace
    print(f"\nUploading to HuggingFace: {args.hf_repo}...")
    difficulty_splits.push_to_hub(
        args.hf_repo,
        private=args.private,
    )

    print(f"\n✅ Successfully created and uploaded difficulty splits to {args.hf_repo}")
    print(f"\nYou can now use the dataset in your configs with:")
    print(f"  dataset_names:")
    print(f"    - {args.hf_repo.split('/')[-1]}:easy")
    print(f"    - {args.hf_repo.split('/')[-1]}:medium")
    print(f"    - {args.hf_repo.split('/')[-1]}:hard")


if __name__ == "__main__":
    main()
