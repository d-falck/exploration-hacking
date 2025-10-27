"""Dataset loading and preparation utilities for RL training."""

from functools import partial
from typing import Callable

from datasets import DatasetDict
from pydantic import BaseModel

from datasets import Dataset, concatenate_datasets, load_dataset, interleave_datasets


class DataSource(BaseModel):
    """Configuration for loading a dataset from HuggingFace."""
    path: str
    name: str
    split: str
    prompt_prefix: str = ""


class SplitConfig(BaseModel):
    """Configuration for dataset splitting into train/test sets."""
    splits: dict[str, float]
    train_split: str | None = None
    test_split: str | None = None


def multi_split(
    dataset: Dataset, splits: dict[str, float], seed: int | None = None
) -> DatasetDict:
    """Bootstrap train_test_split to split a dataset multiple ways."""
    splits = {k: v for k, v in splits.items() if v > 0}
    assert splits
    assert abs(sum(splits.values()) - 1.0) < 1e-6, "Split sizes must sum to 1"

    split_list = list(splits.keys())

    result = {}
    remaining_dataset = dataset
    remaining_ratio = 1.0

    for split_name in split_list[:-1]:
        ratio = splits[split_name]
        split_fraction = ratio / remaining_ratio
        temp_splits = remaining_dataset.train_test_split(
            test_size=split_fraction, seed=seed
        )
        result[split_name] = temp_splits["test"]
        remaining_dataset = temp_splits["train"]
        remaining_ratio -= ratio

    result[split_list[-1]] = remaining_dataset
    return DatasetDict(result)


class Loader:
    """Handles dataset loading, formatting, and splitting for environments.

    Provides utilities to load datasets from HuggingFace, format them
    for RL training, and split them into train/test sets.
    """
    def __init__(
        self,
        prompt_fn: Callable[[dict], str],
        answer_fn: Callable[[dict], str],
        system_prompt: str,
        split_config: SplitConfig,
        seed: int | None = None,
    ):
        self.prompt_fn = prompt_fn
        self.answer_fn = answer_fn
        self.system_prompt = system_prompt
        self.split_config = split_config
        self.seed = seed

    def single_dataset(self, source: DataSource) -> Dataset:
        """Load and split a single dataset."""
        ds = self._prepare_dataset(source)
        return self._train_test_split(ds)

    def merge_datasets(
        self,
        sources: dict[str, DataSource],
        interleave_train: bool = True,
    ) -> Dataset:
        """Merge multiple datasets with stratified sampling and optional interleaving.

        Args:
            sources: Dictionary mapping segment names to DataSource objects
            interleave_train: If True, interleave train datasets evenlyinstead of concatenating them

        Returns:
            Dataset with train and test splits
        """
        datasets = [
            self._prepare_dataset(source, segment_name)
            for segment_name, source in sources.items()
        ]

        def train_merge_fn(datasets: list[Dataset]) -> Dataset:
            if interleave_train:
                # TODO: maintain proportions (or allow custom proportions).
                return interleave_datasets(
                    datasets,
                    stopping_strategy="all_exhausted",
                    seed=self.seed,
                )
            else:
                return concatenate_datasets(datasets).shuffle(seed=self.seed)

        split_datasets = [self._train_test_split(dataset) for dataset in datasets]

        result = {}
        if self.split_config.train_split is not None:
            train_datasets = [dataset["train"] for dataset in split_datasets]
            result["train"] = train_merge_fn(train_datasets)
        if self.split_config.test_split is not None:
            test_datasets = [dataset["test"] for dataset in split_datasets]
            result["test"] = concatenate_datasets(test_datasets).shuffle(seed=self.seed)
        return DatasetDict(result)

    def split_dataset(self, source: DataSource) -> Dataset:
        """Signature TBC.

        This function will allow creating a segmented dataset by splitting
        a single dataset, rather than merging multiple datasets.
        """
        raise NotImplementedError

    def _prepare_dataset(self, source: DataSource, segment: str = "main") -> Dataset:
        ds = load_dataset(source.path, source.name)[source.split]
        old_columns = list(ds.features.keys())
        return ds.map(
            partial(
                self._format_record,
                segment=segment,
                prompt_prefix=source.prompt_prefix,
            ),
            remove_columns=old_columns,
            load_from_cache_file=False,
        )

    def _format_record(self, record: dict, segment: str, prompt_prefix: str) -> dict:
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt_prefix + self.prompt_fn(record),
                },
            ],
            "answer": self.answer_fn(record),
            "info": {"segment": segment},
        }

    def _train_test_split(self, dataset: Dataset) -> DatasetDict:
        """First split the dataset multiple ways, then assign train/test to two of these splits."""
        assert (
            self.split_config.train_split is not None
            or self.split_config.test_split is not None
        )
        split_dataset = multi_split(dataset, self.split_config.splits, self.seed)

        # Print the sizes of all splits
        split_sizes = {k: len(v) for k, v in split_dataset.items()}
        print(f"Split sizes: {split_sizes}")

        result = {}
        if self.split_config.train_split is not None:
            result["train"] = split_dataset[self.split_config.train_split]
            print(
                f"Train split '{self.split_config.train_split}' size: {len(result['train'])}"
            )
        if self.split_config.test_split is not None:
            result["test"] = split_dataset[self.split_config.test_split]
            print(
                f"Test split '{self.split_config.test_split}' size: {len(result['test'])}"
            )
        return DatasetDict(result)
