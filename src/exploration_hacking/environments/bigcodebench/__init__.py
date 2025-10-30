"""BigCodeBench evaluation module for code generation tasks."""

import verifiers as vf
from functools import partial

from exploration_hacking.dataset import DataSource, Loader, SplitConfig
from exploration_hacking.rewards.factory import (
    get_rubric,
    get_conditional_rubric,
)
from exploration_hacking.environments.base import BaseEnvironmentConfig

from ._evaluate import evaluate_single_sample
from ._sanitize import sanitize


__all__ = [
    "evaluate_single_sample",
    "sanitize",
    "BigcodebenchEnvConfig",
    "load_bigcodebench_environment",
]


class BigcodebenchEnvConfig(BaseEnvironmentConfig):
    dataset_names: list[str] = ["bcb-full"]  # Options: bcb-full, bcb-a, bcb-b
    system_prompt: str = """
You are a helpful assistant that generates solutions to coding problems. Please provide a self-contained Python script that solves the following problem in a markdown code block.
"""
    split: SplitConfig
    max_turns: int = 1  # Single-turn code generation task


def _generate_prompt(record: dict) -> str:
    return record["complete_prompt"]


class BigcodebenchLoader(Loader):
    """Custom Loader for BigCodeBench that includes task metadata in the state."""

    def _format_record(self, record: dict, segment: str, prompt_prefix: str) -> dict:
        # Call parent method to get base format
        base_record = super()._format_record(record, segment, prompt_prefix)

        # Add BigCodeBench-specific information to the info field
        base_record["info"].update({
            "task_id": record.get("task_id"),
            "test": record.get("test"),
            "entry_point": record.get("entry_point"),
            "code_prompt": record.get("code_prompt", ""),
        })

        return base_record


def _get_dataset(config: BigcodebenchEnvConfig, seed: int | None = None):
    """Load BigCodeBench datasets with metadata for code evaluation.

    Supports segment syntax (e.g., 'bcb-a:inspect') to enable different prompt prefixes
    for the same dataset, following the pattern from science.py.
    """
    loader = BigcodebenchLoader(
        prompt_fn=_generate_prompt,
        answer_fn=lambda x: "",  # BigCodeBench validates via test execution, not answer comparison
        system_prompt=config.system_prompt,
        split_config=config.split,
        seed=seed,
    )
    prompt_prefixes = config.prompt_prefixes or {}
    sources = {}

    for entry in config.dataset_names:
        # Parse dataset:segment syntax (following science.py pattern)
        if ':' in entry:
            dataset_name, segment_name = entry.split(':', 1)
        else:
            dataset_name = entry
            segment_name = entry

        # Get prompt prefix for this segment
        prefix = prompt_prefixes.get(segment_name, "")

        # Map dataset names to the correct subset names
        if dataset_name in ["bcb-full", "bcb-a", "bcb-b"]:
            # Use new Joschka/bigcodebench dataset with correct subset
            sources[segment_name] = DataSource(
                path="Joschka/bigcodebench",
                name=dataset_name,  # HuggingFace uses the base dataset name
                split="train",  # All subsets use 'train' split
                prompt_prefix=prefix,
            )
        else:
            # Default to bcb-full if unspecified
            sources[segment_name] = DataSource(
                path="Joschka/bigcodebench",
                name="bcb-full",
                split="train",
                prompt_prefix=prefix,
            )
    return loader.merge_datasets(sources)


def load_bigcodebench_environment(config: BigcodebenchEnvConfig, seed: int | None = None):
    """Create a BigCodeBench environment for code generation tasks."""
    ds = _get_dataset(config, seed)
    parser = vf.XMLParser(fields=["think", "answer"])

    if config.segment_rewards:
        rubric = get_conditional_rubric(
            config.segment_rewards,
            config.global_rewards,
            parser,
            tools=[],
            tokenizer_name=config.tokenizer,
        )
    else:
        assert config.global_rewards is not None
        rubric = get_rubric(
            config.global_rewards, parser, tools=[], tokenizer_name=config.tokenizer
        )

    kwargs = dict(
        tools=[],
        rubric=rubric,
        parser=parser,
        max_turns=config.max_turns,
        inline_reasoning=config.inline_reasoning,
    )

    if ds is not None:
        if "train" in ds:
            kwargs["dataset"] = ds["train"]
        if "test" in ds:
            kwargs["eval_dataset"] = ds["test"]

    return vf.ToolEnv(**kwargs)