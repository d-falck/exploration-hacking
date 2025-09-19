"""Generate a HF dataset by filtering evaluation traces."""

import pickle
from pathlib import Path

import numpy as np
from datasets import Dataset
from pydantic import BaseModel, Field
import verifiers as vf

from exploration_hacking.dtypes import ExperimentConfig


class FilterCriterion(BaseModel):
    """A single filter criterion for a metric."""

    above_threshold: float | None = None
    below_threshold: float | None = None
    top_quantile: float | None = None  # e.g. 0.25 for top 25%
    bottom_quantile: float | None = None  # e.g. 0.25 for bottom 25%


class Config(ExperimentConfig):
    input_path: Path
    output_path: Path
    filter_metrics: dict[str, FilterCriterion] = Field(default_factory=dict)


def filter_traces(
    results: vf.GenerateOutputs, filter_metrics: dict[str, FilterCriterion]
) -> list[int]:
    """Return indices of traces that pass filters."""
    n = len(results.reward)
    indices = list(range(n))

    for metric_name, criterion in filter_metrics.items():
        if metric_name == "reward":
            values = results.reward
        elif metric_name in results.metrics:
            values = results.metrics[metric_name]
        else:
            print(f"Warning: metric '{metric_name}' not found, skipping")
            continue

        assert (
            sum(
                x is not None
                for x in [
                    criterion.above_threshold,
                    criterion.below_threshold,
                    criterion.top_quantile,
                    criterion.bottom_quantile,
                ]
            )
            == 1
        ), (
            f"Exactly one of above_threshold, below_threshold, top_quantile, "
            f"and bottom_quantile must be set for metric '{metric_name}', "
            f"but got: {criterion}"
        )

        if criterion.above_threshold is not None:
            indices = [i for i in indices if values[i] >= criterion.above_threshold]
        if criterion.below_threshold is not None:
            indices = [i for i in indices if values[i] <= criterion.below_threshold]
        if criterion.top_quantile is not None:
            threshold = np.percentile(values, 100 * (1 - criterion.top_quantile))
            indices = [i for i in indices if values[i] >= threshold]
        if criterion.bottom_quantile is not None:
            threshold = np.percentile(values, 100 * criterion.bottom_quantile)
            indices = [i for i in indices if values[i] <= threshold]

    return indices


def build_dataset(results: vf.GenerateOutputs, indices: list[int]) -> Dataset:
    prompts = [results.prompt[idx] for idx in indices]
    completions = [results.completion[idx] for idx in indices]
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "completion": completions,
        }
    )


def main(config: Config):
    with config.input_path.open("rb") as f:
        results = pickle.load(f)

    print(f"Loaded {len(results.reward)} traces from {config.input_path}")

    indices = filter_traces(results, config.filter)
    print(f"After filtering: {len(indices)} traces remaining")

    dataset = build_dataset(results, indices)
    dataset.save_to_disk(config.output_path)

    print(f"Saved {len(dataset)} traces to {config.output_path}")
    print(
        f"Average reward of filtered traces: {np.mean([results.reward[i] for i in indices]):.3f}"
    )


if __name__ == "__main__":
    main(Config())
