import logging
import random
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import Dataset

from exploration_hacking.settings.science.dtypes import ScienceProblem
from exploration_hacking.settings.science.generate_prompt import SciencePrompter


def load_science_dataset(
    dataset_name: Literal["biology", "chemistry", "coding"],
    split: Literal["all", "train", "eval", "test"] = "train",
    max_problems: int | None = None,
    system_prompt: str = "",
    random_sample: bool = False,
    random_state: int = 1337,
    data_dir: str = "/workspace/exploration-hacking/data/science",
) -> tuple[Dataset, dict[str, ScienceProblem]]:
    """
    Load a Science dataset with optional sampling.

    Args:
        dataset_name: Which dataset to load ("biology", "chemistry", or "coding")
        split: Which split to load ("all", "train", "eval", or "test"). Defaults to "train"
        max_problems: Number of samples to load. If None, loads entire dataset
        random_sample: If True, samples randomly. If False, takes first max_problems
        random_state: Random seed for reproducibility when random_sample=True
        data_dir: Base directory for science data (relative to project root)

    Returns:
        pd.DataFrame: The loaded dataset with requested samples

    Raises:
        ValueError: If dataset name or split is invalid
        FileNotFoundError: If dataset file doesn't exist
    """
    # Validate dataset name
    valid_datasets = ["biology", "chemistry", "coding"]
    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Invalid dataset: {dataset_name}. Must be one of {valid_datasets}"
        )

    # Validate split
    valid_splits = ["all", "train", "eval", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")

    # Construct path to parquet file
    project_root = Path(
        __file__
    ).parent.parent.parent.parent  # Go up from src/exploration_hacking/settings/science/ to project root
    data_path = (
        project_root / data_dir / dataset_name / f"{dataset_name}_{split}.parquet"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Load the full dataset
    df = pd.read_parquet(data_path)

    # Handle sampling if requested
    if max_problems is not None:
        total_samples = len(df)

        # If requested samples exceed available, use all
        if max_problems >= total_samples:
            logging.warning(
                f"Requested {max_problems} samples for {dataset_name} {split} split, "
                f"but only {total_samples} samples available. Using all available samples."
            )
        else:
            # Sample based on random_sample parameter
            if random_sample:
                df = df.sample(n=max_problems, random_state=random_state)
            else:
                # Take first max_problems rows
                df = df.head(max_problems)

    prompter = SciencePrompter(system_prompt=system_prompt)
    problems = {}
    data = []
    for i, row in df.iterrows():
        user_prompt = prompter.format_question(row["question"], row["choices"])
        task_id = f"science_{dataset_name}_{split}_{i}"
        problem = ScienceProblem(
            task_id=task_id,
            prompt=user_prompt,
            answer=row["answer"],
            choices=row["choices"],
        )

        problems[task_id] = problem

        messages = [
            {"role": "system", "content": prompter.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        data.append(
            {
                "prompt": messages,
                "task_id": task_id,
            }
        )

    random.shuffle(data)

    dataset = Dataset.from_list(data)
    return dataset, problems
