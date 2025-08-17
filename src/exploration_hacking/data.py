from datasets import Dataset
from pydantic import BaseModel, Field
from typing import Any

from exploration_hacking.settings.evalplus.load import load_evalplus_dataset
from exploration_hacking.settings.science.load import load_science_dataset
from exploration_hacking.dtypes import Problem


class DataConfig(BaseModel):
    dataset_name: str
    split: str | None = None
    max_problems: int | None = None
    flags: dict[str, Any] = Field(default_factory=dict)


def load_dataset(config: DataConfig) -> tuple[Dataset, dict[str, Problem]]:
    if config.dataset_name in ["biology", "chemistry", "coding"]:
        assert config.split is not None, "Split is required for science datasets"
        return load_science_dataset(
            config.dataset_name,
            config.split,
            config.max_problems,
            config.flags.get("system_prompt", ""),
        )
    elif config.dataset_name in ["humaneval", "mbpp"]:
        assert config.split is None, "Split is not supported for evalplus datasets"
        return load_evalplus_dataset(
            config.dataset_name,
            config.max_problems,
            config.flags.get("segment", False),
        )
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset_name}")
