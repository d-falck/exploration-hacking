from functools import partial
from typing import Callable

from datasets import DatasetDict
import numpy as np
from pydantic import BaseModel

from datasets import Dataset, concatenate_datasets, load_dataset


class DataSource(BaseModel):
    path: str
    name: str
    split: str
    prompt_prefix: str = ""


class Loader:
    def __init__(
        self,
        prompt_fn: Callable[[dict], str],
        answer_fn: Callable[[dict], str],
        system_prompt: str,
        test_size: float,
    ):
        self.prompt_fn = prompt_fn
        self.answer_fn = answer_fn
        self.system_prompt = system_prompt
        self.test_size = test_size

    def single_dataset(self, source: DataSource) -> Dataset:
        ds = self._prepare_dataset(source)
        return ds.train_test_split(test_size=self.test_size)

    def merge_datasets(
        self,
        sources: dict[str, DataSource],
        stratify: bool = False,
    ) -> Dataset:
        """Merge multiple datasets with optional stratified splitting and even interspersion.

        Args:
            sources: Dictionary mapping segment names to DataSource objects
            stratify: If True, maintain segment ratios in train/test splits

        Returns:
            Dataset with train and test splits
        """
        datasets = [
            self._prepare_dataset(source, segment_name)
            for segment_name, source in sources.items()
        ]

        if stratify:  # Split then concatenate
            split_datasets = [
                dataset.train_test_split(test_size=self.test_size)
                for dataset in datasets
            ]
            train_datasets = [dataset["train"] for dataset in split_datasets]
            test_datasets = [dataset["test"] for dataset in split_datasets]
            return DatasetDict(
                {
                    "train": concatenate_datasets(train_datasets).shuffle(),
                    "test": concatenate_datasets(test_datasets).shuffle(),
                }
            )
        else:  # Concatenate then split
            ds = concatenate_datasets(datasets).shuffle()
            return ds.train_test_split(test_size=self.test_size)

    def split_dataset(self, source: DataSource) -> Dataset:
        """Signature TBC."""
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
