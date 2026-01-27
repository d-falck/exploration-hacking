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


def extract_question_from_prompt(prompt) -> str:
    """Extract the question text from a prompt."""
    if isinstance(prompt, list):
        for msg in prompt:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "Question:" in content:
                    return content.split("Question:")[1].split("Options:")[0].strip()
    return ""


def get_user_content_from_prompt(prompt) -> str:
    """Extract the user message content from a prompt (generic)."""
    if isinstance(prompt, list):
        for msg in prompt:
            if msg.get("role") == "user":
                return msg.get("content", "")
    elif isinstance(prompt, str):
        return prompt
    return ""


def calculate_per_question_accuracy(
    results, rollouts_per_example: int | None = None
) -> tuple[list[float], list[str]]:
    """Calculate accuracy for each question across rollouts.

    Args:
        results: GenerateOutputs object with evaluation results
        rollouts_per_example: Number of rollouts per question. If None, will try to infer.

    Returns:
        Tuple of (accuracies, prompt_contents) where:
        - accuracies: List of accuracy values (0.0 to 1.0) for each question
        - prompt_contents: List of user prompt contents (for matching to original dataset)
    """
    # Get the metrics - should have 'accuracy' or 'correct' per rollout
    if hasattr(results, "metrics"):
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
        raise ValueError(
            f"No accuracy metric found. Available metrics: {metrics.keys()}"
        )

    accuracy_values = metrics[accuracy_key]
    num_total_rollouts = len(accuracy_values)

    # Infer rollouts_per_example if not provided
    if rollouts_per_example is None:
        if hasattr(results, "prompt"):
            # Try to count from prompt similarity
            prompt_strings = [str(p) for p in results.prompt]

            rollouts_per_question = []
            current_prompt = None
            current_count = 0

            for p in prompt_strings:
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
                print(
                    f"Inferred rollouts_per_example={rollouts_per_example} from prompts"
                )
            else:
                print(
                    f"Warning: Inconsistent rollouts per question: {set(rollouts_per_question)}"
                )
                print(f"This may indicate an incomplete evaluation.")
                rollouts_per_example = max(
                    set(rollouts_per_question), key=rollouts_per_question.count
                )
                print(f"Using most common rollout count: {rollouts_per_example}")
        else:
            # Fallback to default
            rollouts_per_example = 3
            print(
                f"Warning: Could not infer rollouts_per_example, defaulting to {rollouts_per_example}"
            )

    # Calculate per-question accuracy by grouping rollouts
    # We expect rollouts to be ordered: Q1_R1, Q1_R2, Q1_R3, Q2_R1, Q2_R2, Q2_R3, ...
    num_questions = num_total_rollouts // rollouts_per_example
    per_question_accuracy = []
    prompt_contents = []

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
            # Extract user prompt content from first rollout
            prompt_content = get_user_content_from_prompt(results.prompt[start_idx])
            prompt_contents.append(prompt_content)

    print(
        f"Processed {len(per_question_accuracy)} questions from {num_total_rollouts} rollouts"
    )
    print(f"  {rollouts_per_example} rollouts per question")

    return per_question_accuracy, prompt_contents


def create_difficulty_splits(
    dataset: Dataset,
    accuracies: list[float],
    prompt_contents: list[str],
    question_field: str = "question",
    easy_frac: float = 0.4,
    medium_frac: float = 0.4,
    hard_frac: float = 0.2,
) -> DatasetDict:
    """Split dataset into easy/medium/hard based on accuracies.

    Args:
        dataset: HuggingFace dataset to split
        accuracies: Per-question accuracy values (0.0 to 1.0)
        prompt_contents: User prompt contents from eval (for matching to dataset)
        question_field: Name of the field in dataset containing question text
        easy_frac: Fraction of dataset for easy split (highest accuracy)
        medium_frac: Fraction for medium split
        hard_frac: Fraction for hard split (lowest accuracy)

    Returns:
        DatasetDict with 'easy', 'medium', 'hard' splits
    """
    assert (
        abs(easy_frac + medium_frac + hard_frac - 1.0) < 1e-6
    ), "Fractions must sum to 1.0"

    num_eval_questions = len(accuracies)
    assert len(accuracies) == len(
        prompt_contents
    ), f"Accuracies ({len(accuracies)}) and prompt_contents ({len(prompt_contents)}) length mismatch"

    # Build mapping from prompt content -> accuracy
    # (match by checking if dataset question is substring of prompt)
    prompt_to_accuracy = {pc: acc for pc, acc in zip(prompt_contents, accuracies)}

    # Find matching dataset indices and their accuracies
    matched_indices = []
    matched_accuracies = []

    for idx, item in enumerate(dataset):
        question_text = item[question_field]
        # Find matching prompt by checking if question is contained in any prompt
        for prompt_content, accuracy in prompt_to_accuracy.items():
            if question_text in prompt_content:
                matched_indices.append(idx)
                matched_accuracies.append(accuracy)
                break

    print(f"Matched {len(matched_indices)} questions out of {len(dataset)} in dataset")
    print(f"  ({num_eval_questions} questions in eval results)")

    if len(matched_indices) < num_eval_questions:
        print(f"Warning: Could not match all eval questions to dataset")

    # Use matched questions for splitting
    num_questions = len(matched_indices)
    accuracies = matched_accuracies

    # Sort questions by accuracy (descending)
    sorted_indices = np.argsort(accuracies)[::-1]  # High to low

    # Calculate split boundaries
    num_easy = int(num_questions * easy_frac)
    num_medium = int(num_questions * medium_frac)
    num_hard = num_questions - num_easy - num_medium  # Remaining

    # Split indices (sorted_indices indexes into matched arrays, need to map to dataset indices)
    easy_sorted_idx = sorted_indices[:num_easy]
    medium_sorted_idx = sorted_indices[num_easy : num_easy + num_medium]
    hard_sorted_idx = sorted_indices[num_easy + num_medium :]

    # Map to actual dataset indices
    easy_dataset_indices = [matched_indices[i] for i in easy_sorted_idx]
    medium_dataset_indices = [matched_indices[i] for i in medium_sorted_idx]
    hard_dataset_indices = [matched_indices[i] for i in hard_sorted_idx]

    # Create splits
    easy_dataset = dataset.select(easy_dataset_indices)
    medium_dataset = dataset.select(medium_dataset_indices)
    hard_dataset = dataset.select(hard_dataset_indices)

    # Print statistics
    print(f"\n=== Dataset Split Statistics ===")
    for split_name, sorted_idx, dataset_split in [
        ("Easy", easy_sorted_idx, easy_dataset),
        ("Medium", medium_sorted_idx, medium_dataset),
        ("Hard", hard_sorted_idx, hard_dataset),
    ]:
        split_accuracies = [accuracies[i] for i in sorted_idx]
        print(f"\n{split_name} split: {len(dataset_split)} questions")
        print(
            f"  Accuracy range: {min(split_accuracies):.3f} - {max(split_accuracies):.3f}"
        )
        print(f"  Mean accuracy: {np.mean(split_accuracies):.3f}")
    print("================================\n")

    return DatasetDict(
        {
            "easy": easy_dataset,
            "medium": medium_dataset,
            "hard": hard_dataset,
        }
    )


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
        "--rollouts-per-example",
        type=int,
        default=None,
        help="Number of rollouts per question. If not specified, will try to infer from results.",
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
    accuracies, prompt_contents = calculate_per_question_accuracy(
        results, args.rollouts_per_example
    )
    print(f"Processed {len(accuracies)} questions")
    print(f"Overall accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

    # Load original dataset
    print("\nLoading original WMDP-bio dataset...")
    original_dataset = load_dataset("Joschka/wmdp", "wmdp-bio")["test"]
    print(f"Loaded {len(original_dataset)} questions")

    if len(prompt_contents) < len(original_dataset):
        print(
            f"Warning: Only {len(prompt_contents)} questions were evaluated out of {len(original_dataset)}"
        )
        print("Creating splits based on evaluated questions only")

    # Create splits
    print("\nCreating difficulty splits...")
    difficulty_splits = create_difficulty_splits(
        original_dataset,
        accuracies,
        prompt_contents,
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


if __name__ == "__main__":
    main()
