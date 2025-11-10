"""Filter evaluation traces and export to .eval and .json formats for sharing.

This script is optimized for sharing filtered traces with collaborators who use
Inspect AI and other analysis tools. Unlike filter.py which outputs HuggingFace
datasets for SFT training, this script outputs:
- .eval files (Inspect AI format)
- .json files (structured format with metadata)
- filter_stats.json (statistics)
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import verifiers as vf
from pydantic import Field

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.util import InspectEvalAdapter

# Import filtering logic from filter.py to avoid duplication
from exploration_hacking.scripts.data.filter import (
    FilterCriterion,
    SelectionConfig,
    filter_traces,
    select_per_prompt,
    serialize,
    replace_system_prompt,
)


class Config(ExperimentConfig):
    input_path: Path
    output_eval_path: Path
    output_json_path: Path
    filter_metrics: dict[str, FilterCriterion] = Field(default_factory=dict)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    new_system_prompt: str | None = None
    model_name: str | None = None  # Optional, will be extracted from filename if not provided


def create_filtered_outputs(
    results: vf.GenerateOutputs, indices: list[int], new_system_prompt: str | None = None
) -> vf.GenerateOutputs:
    """Create new GenerateOutputs with only filtered samples.

    Args:
        results: Original GenerateOutputs with all samples
        indices: List of indices to keep
        new_system_prompt: Optional new system prompt to replace in all prompts

    Returns:
        New GenerateOutputs with only the selected samples
    """
    # Get prompts and apply system prompt replacement if specified
    prompts = [results.prompt[i] for i in indices]
    if new_system_prompt is not None:
        prompts = [replace_system_prompt(prompt, new_system_prompt) for prompt in prompts]

    return vf.GenerateOutputs(
        prompt=prompts,
        completion=[results.completion[i] for i in indices],
        answer=[results.answer[i] for i in indices],
        state=[results.state[i] for i in indices],
        info=[results.info[i] for i in indices],
        task=[results.task[i] for i in indices],
        reward=[results.reward[i] for i in indices],
        metrics={
            name: [values[i] for i in indices]
            for name, values in results.metrics.items()
        },
    )


def export_to_json(
    filtered_outputs: vf.GenerateOutputs,
    config: Config,
    stats: dict,
) -> None:
    """Export filtered traces to structured JSON format.

    Args:
        filtered_outputs: Filtered GenerateOutputs to export
        config: Configuration with paths and settings
        stats: Statistics about the filtering process
    """
    # Build structured JSON with metadata and samples
    output_data = {
        "metadata": {
            "input_path": str(config.input_path),
            "output_eval_path": str(config.output_eval_path),
            "output_json_path": str(config.output_json_path),
            "model_name": config.model_name,
            "created_at": datetime.now().isoformat(),
            "total_input_samples": stats["total_traces_loaded"],
            "total_output_samples": stats["traces_after_selection"],
            "filter_criteria": {k: v.model_dump() for k, v in config.filter_metrics.items()},
            "selection_config": config.selection.model_dump(),
            "new_system_prompt": config.new_system_prompt,
        },
        "samples": [],
    }

    # Add each sample with all its data
    for i in range(len(filtered_outputs.reward)):
        sample = {
            "prompt": serialize(filtered_outputs.prompt[i]),
            "completion": serialize(filtered_outputs.completion[i]),
            "answer": filtered_outputs.answer[i],
            "task": filtered_outputs.task[i],
            "reward": float(filtered_outputs.reward[i]),
            "metrics": {
                name: float(values[i])
                for name, values in filtered_outputs.metrics.items()
            },
            "info": serialize(filtered_outputs.info[i]),
        }
        output_data["samples"].append(sample)

    # Save to JSON file
    config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_json_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved JSON to: {config.output_json_path}")


def extract_model_name(config: Config) -> str:
    """Extract model name from config or filename.

    Args:
        config: Configuration object

    Returns:
        Model name string
    """
    if config.model_name:
        return config.model_name

    # Try to extract from input filename
    # e.g., "eval_gen_benign_qwen3_coder_plus_full_4_rollouts_no_comments.pkl"
    filename = config.input_path.stem

    # Common patterns to extract
    if "qwen3_coder_plus" in filename.lower():
        return "Qwen3-Coder-Plus"
    elif "qwen" in filename.lower():
        return "Qwen"
    elif "gpt-4" in filename.lower():
        return "gpt-4"
    elif "claude" in filename.lower():
        return "claude"

    # Default fallback
    return "unknown-model"


def main(config: Config):
    """Main function to filter traces and export to .eval and .json formats."""

    # Load input pickle file
    print(f"Loading traces from: {config.input_path}")
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

    # Step 3: Create filtered outputs with optional system prompt replacement
    filtered_outputs = create_filtered_outputs(results, selected_indices, config.new_system_prompt)
    if config.new_system_prompt:
        print(f"Applied new system prompt to all {len(filtered_outputs.reward)} samples")
    print(f"Created filtered GenerateOutputs with {len(filtered_outputs.reward)} samples")

    # Step 4: Extract model name
    model_name = extract_model_name(config)
    print(f"Using model name: {model_name}")

    # Step 5: Export to .eval file
    print(f"Exporting to .eval format...")
    config.output_eval_path.parent.mkdir(parents=True, exist_ok=True)
    InspectEvalAdapter.save_as_eval(
        outputs=filtered_outputs,
        model_name=model_name,
        output_path=str(config.output_eval_path),
        task_name="filtered_detection_traces",
        task_metadata={
            "filtered": True,
            "original_samples": total_traces_loaded,
            "filtered_samples": traces_after_selection,
            "filter_criteria": {k: v.model_dump() for k, v in config.filter_metrics.items()},
            "selection_config": config.selection.model_dump(),
        },
        compute_summary_stats=True,
        compute_segment_stats=True,
    )
    print(f"Saved .eval file to: {config.output_eval_path}")

    # Collect statistics
    import numpy as np
    avg_reward_filtered = np.mean([results.reward[i] for i in filtered_indices]) if filtered_indices else 0.0
    avg_reward_selected = np.mean([results.reward[i] for i in selected_indices]) if selected_indices else 0.0

    stats = {
        "input_path": str(config.input_path),
        "output_eval_path": str(config.output_eval_path),
        "output_json_path": str(config.output_json_path),
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
        "new_system_prompt": config.new_system_prompt,
        "model_name": model_name,
    }

    # Step 6: Export to JSON
    print(f"Exporting to JSON format...")
    export_to_json(filtered_outputs, config, stats)

    # Step 7: Save statistics
    stats_file = config.output_eval_path.parent / f"{config.output_eval_path.stem}_filter_stats.json"
    with stats_file.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input: {total_traces_loaded} traces")
    print(f"Output: {traces_after_selection} traces ({stats['overall_pass_rate']:.1%} pass rate)")
    print(f"Average reward (filtered): {avg_reward_filtered:.3f}")
    print(f"Average reward (selected): {avg_reward_selected:.3f}")
    print("="*60)


if __name__ == "__main__":
    main(Config())
