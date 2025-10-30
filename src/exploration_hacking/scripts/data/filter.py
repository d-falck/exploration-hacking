"""Generate a HF dataset by filtering evaluation traces."""

import json
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


class SelectionConfig(BaseModel):
    """Configuration for per-prompt selection."""

    mode: str = "global"  # 'global' or 'per_prompt'
    best_n_per_prompt: int | None = None  # Take top N rollouts per prompt
    sort_by: str = "reward"  # Metric to sort rollouts (default: total reward)


class Config(ExperimentConfig):
    input_path: Path
    output_path: Path
    filter_metrics: dict[str, FilterCriterion] = Field(default_factory=dict)
    new_system_prompt: str | None = None
    selection: SelectionConfig = Field(default_factory=SelectionConfig)


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


def select_per_prompt(
    results: vf.GenerateOutputs,
    filtered_indices: list[int],
    selection_config: SelectionConfig,
) -> list[int]:
    """Select best N rollouts per prompt from filtered traces."""
    if selection_config.mode == "global":
        # No per-prompt selection, return all filtered indices
        return filtered_indices

    if selection_config.mode != "per_prompt":
        raise ValueError(f"Unknown selection mode: {selection_config.mode}")

    if selection_config.best_n_per_prompt is None:
        raise ValueError("best_n_per_prompt must be set when mode='per_prompt'")

    # Get the sorting metric values
    sort_by = selection_config.sort_by
    if sort_by == "reward":
        metric_values = results.reward
    elif sort_by in results.metrics:
        metric_values = results.metrics[sort_by]
    else:
        raise ValueError(
            f"sort_by metric '{sort_by}' not found. "
            f"Available: 'reward' or any of {list(results.metrics.keys())}"
        )

    # Group filtered indices by task
    task_to_indices: dict[str, list[int]] = {}
    for idx in filtered_indices:
        task_id = results.info[idx].get('task_id', 'default')
        if task_id not in task_to_indices:
            task_to_indices[task_id] = []
        task_to_indices[task_id].append(idx)

    # For each task, sort by metric and take top N
    selected_indices = []
    for task_id, task_indices in task_to_indices.items():
        # Sort indices by metric value (descending)
        sorted_indices = sorted(
            task_indices, key=lambda i: metric_values[i], reverse=True
        )
        # Take top N
        n_to_take = min(selection_config.best_n_per_prompt, len(sorted_indices))
        selected_indices.extend(sorted_indices[:n_to_take])

    return selected_indices


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

    # Count statistics
    total_traces_loaded = len(results.reward)
    total_unique_prompts = len(set(info.get('task_id', 'default') for info in results.info))

    print(f"Loaded {total_traces_loaded} traces from {config.input_path}")
    print(f"Total unique prompts: {total_unique_prompts}")

    # Step 1: Apply filters
    filtered_indices = filter_traces(results, config.filter_metrics)
    traces_after_filtering = len(filtered_indices)
    print(f"After filtering: {traces_after_filtering} traces remaining")

    # Step 2: Apply per-prompt selection
    selected_indices = select_per_prompt(results, filtered_indices, config.selection)
    traces_after_selection = len(selected_indices)
    unique_prompts_after_selection = len(set(results.info[i].get('task_id', 'default') for i in selected_indices))

    if config.selection.mode == "per_prompt":
        print(
            f"After per-prompt selection (mode={config.selection.mode}, "
            f"best_n={config.selection.best_n_per_prompt}): "
            f"{traces_after_selection} traces from {unique_prompts_after_selection} unique prompts"
        )
    else:
        print(f"Selection mode: {config.selection.mode} (no per-prompt selection)")

    # Build and save dataset
    dataset = build_dataset(results, selected_indices, config.new_system_prompt)
    dataset.save_to_disk(config.output_path)
    print(f"Saved {len(dataset)} traces to {config.output_path}")

    # Collect comprehensive statistics
    avg_reward_filtered = np.mean([results.reward[i] for i in filtered_indices]) if filtered_indices else 0.0
    avg_reward_selected = np.mean([results.reward[i] for i in selected_indices]) if selected_indices else 0.0

    stats = {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "total_traces_loaded": total_traces_loaded,
        "total_unique_prompts": total_unique_prompts,
        "traces_after_filtering": traces_after_filtering,
        "traces_after_selection": traces_after_selection,
        "unique_prompts_after_selection": unique_prompts_after_selection,
        "filtering_pass_rate": traces_after_filtering / total_traces_loaded if total_traces_loaded > 0 else 0.0,
        "selection_pass_rate": traces_after_selection / traces_after_filtering if traces_after_filtering > 0 else 0.0,
        "overall_pass_rate": traces_after_selection / total_traces_loaded if total_traces_loaded > 0 else 0.0,
        "average_reward_filtered": float(avg_reward_filtered),
        "average_reward_selected": float(avg_reward_selected),
        "filter_metrics": {k: v.model_dump() for k, v in config.filter_metrics.items()},
        "selection_config": config.selection.model_dump(),
    }

    # Save statistics to JSON
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "filter_stats.json"
    with stats_file.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    main(Config())
