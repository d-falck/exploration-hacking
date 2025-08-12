"""Load Science dataset with flexible sampling options."""

import pandas as pd
from pathlib import Path
from typing import Literal, Optional
import random
import logging


def load_science_dataset_impl(
    dataset: Literal["biology", "chemistry", "coding"],
    split: Literal["all", "train", "eval", "test"] = "train",
    num_samples: Optional[int] = None,
    random_sample: bool = False,
    random_state: int = 42,
    data_dir: str = "/workspace/exploration-hacking/data/science",
) -> pd.DataFrame:
    """
    Load a Science dataset with optional sampling.

    Args:
        dataset: Which dataset to load ("biology", "chemistry", or "coding")
        split: Which split to load ("all", "train", "eval", or "test"). Defaults to "train"
        num_samples: Number of samples to load. If None, loads entire dataset
        random_sample: If True, samples randomly. If False, takes first num_samples
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
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {dataset}. Must be one of {valid_datasets}")

    # Validate split
    valid_splits = ["all", "train", "eval", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")

    # Construct path to parquet file
    project_root = Path(
        __file__
    ).parent.parent.parent  # Go up from src/data_utils/ to project root
    data_path = project_root / data_dir / dataset / f"{dataset}_{split}.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Load the full dataset
    df = pd.read_parquet(data_path)

    # If no sampling requested, return full dataset
    if num_samples is None:
        return df

    # Handle sampling
    total_samples = len(df)

    # If requested samples exceed available, return all
    if num_samples >= total_samples:
        logging.warning(
            f"Requested {num_samples} samples for {dataset} {split} split, "
            f"but only {total_samples} samples available. Returning all available samples."
        )
        return df

    # Sample based on random_sample parameter
    if random_sample:
        # Set random seed for reproducibility
        return df.sample(n=num_samples, random_state=random_state)
    else:
        # Take first num_samples rows
        return df.head(num_samples)


def get_dataset_info(
    data_dir: str = "/workspace/exploration-hacking/data/science",
) -> dict:
    """
    Get information about available Science datasets and their splits.

    Args:
        data_dir: Base directory for science data

    Returns:
        Dictionary with dataset names as keys and info (size, columns, splits) as values
    """
    info = {}
    project_root = Path(__file__).parent.parent.parent

    for dataset in ["biology", "chemistry", "coding"]:
        dataset_info = {"splits": {}}

        # Check each split
        for split in ["all", "train", "eval", "test"]:
            data_path = project_root / data_dir / dataset / f"{dataset}_{split}.parquet"

            if data_path.exists():
                df = pd.read_parquet(data_path)
                split_info = {
                    "num_samples": len(df),
                    "file_size_mb": data_path.stat().st_size / (1024 * 1024),
                }

                # Add column info only for the "all" split
                if split == "all":
                    dataset_info["columns"] = list(df.columns)
                    dataset_info["total_samples"] = len(df)

                dataset_info["splits"][split] = split_info

        if dataset_info["splits"]:
            info[dataset] = dataset_info

    return info
