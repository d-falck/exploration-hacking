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
    new_system_prompt: str | None = None


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


def serialize(obj):
    if isinstance(obj, list):
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    else:
        return obj


def replace_system_prompt(
    prompt: list[dict], new_system_prompt: str | None
) -> list[dict]:
    if new_system_prompt is None:
        return prompt
    return [
        (
            message | {"content": new_system_prompt}
            if message["role"] == "system"
            else message
        )
        for message in prompt
    ]


def fix_tool_arguments(completions: list) -> list:
    """Convert tool arguments from JSON strings to dicts for proper SFT training."""
    import json
    import copy

    fixed_completions = []
    for completion in completions:
        fixed_completion = []
        for msg in completion:
            # Deep copy to avoid modifying original
            fixed_msg = copy.deepcopy(msg) if isinstance(msg, dict) else msg

            if isinstance(fixed_msg, dict) and 'tool_calls' in fixed_msg:
                for tc in fixed_msg.get('tool_calls', []):
                    # Parse arguments from JSON string to dict
                    if 'function' in tc and 'arguments' in tc['function']:
                        args = tc['function']['arguments']
                        if isinstance(args, str):
                            try:
                                tc['function']['arguments'] = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                pass  # Keep as-is if parsing fails

            fixed_completion.append(fixed_msg)
        fixed_completions.append(fixed_completion)

    return fixed_completions


def build_dataset(
    results: vf.GenerateOutputs, indices: list[int], new_system_prompt: str | None
) -> Dataset:
    prompts = [
        replace_system_prompt(results.prompt[idx], new_system_prompt) for idx in indices
    ]
    completions = [results.completion[idx] for idx in indices]
    tools = [results.info[idx].get("oai_tools", []) for idx in indices]

    serialized_completions = serialize(completions)
    serialized_completions = fix_tool_arguments(serialized_completions)

    return Dataset.from_dict(
        {
            "prompt": serialize(prompts),
            "completion": serialized_completions,
            "tools": serialize(tools),
        }
    )


def main(config: Config):
    with config.input_path.open("rb") as f:
        data = pickle.load(f)

    # Handle both formats: direct GenerateOutputs or dict with 'results' key
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    else:
        results = data

    print(f"Loaded {len(results.reward)} traces from {config.input_path}")

    indices = filter_traces(results, config.filter_metrics)
    print(f"After filtering: {len(indices)} traces remaining")

    dataset = build_dataset(results, indices, config.new_system_prompt)
    dataset.save_to_disk(config.output_path)

    print(f"Saved {len(dataset)} traces to {config.output_path}")
    print(
        f"Average reward of filtered traces: {np.mean([results.reward[i] for i in indices]):.3f}"
    )


if __name__ == "__main__":
    main(Config())
