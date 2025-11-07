"""Generate summary statistics from evaluation results."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import verifiers as vf


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute basic statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with mean, std, min, max
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(values)
    }


def extract_segments(results: vf.GenerateOutputs) -> dict[str, list[int]]:
    """Extract segment information and return indices grouped by segment.

    Args:
        results: GenerateOutputs object from evaluation

    Returns:
        Dictionary mapping segment name to list of indices
    """
    segments = {}

    for i, info in enumerate(results.info):
        if isinstance(info, dict) and 'segment' in info:
            segment = info['segment']
            if segment not in segments:
                segments[segment] = []
            segments[segment].append(i)

    return segments


def _make_json_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle Pydantic models and other objects with __dict__
        result = {}
        for key, value in vars(obj).items():
            if not key.startswith('_'):
                result[key] = _make_json_serializable(value)
        return result
    else:
        # For other objects, try to convert to string
        return str(obj)


def extract_reward_weights(config: Any) -> dict[str, Any]:
    """Extract reward weights from config.

    Args:
        config: Config object with environment configuration

    Returns:
        Dictionary mapping reward name to weight (JSON-serializable)
    """
    weights = {}

    # Try to get BigCodeBench rewards
    if hasattr(config, 'environment'):
        env_config = config.environment

        # Check for bigcodebench environment
        if hasattr(env_config, 'bigcodebench'):
            bcb_config = env_config.bigcodebench

            # Extract global_rewards
            if hasattr(bcb_config, 'global_rewards'):
                global_rewards = bcb_config.global_rewards
                if isinstance(global_rewards, dict):
                    for key, value in global_rewards.items():
                        if key.endswith('_reward_weight'):
                            # Remove _reward_weight suffix for cleaner names
                            reward_name = key.replace('_reward_weight', '')
                            weights[reward_name] = value
                        else:
                            # Convert complex objects to JSON-serializable format
                            weights[key] = _make_json_serializable(value)
                else:
                    # If it's a Pydantic model, extract fields
                    if hasattr(global_rewards, '__dict__'):
                        for key, value in vars(global_rewards).items():
                            if key.endswith('_reward_weight'):
                                reward_name = key.replace('_reward_weight', '')
                                weights[reward_name] = value
                            elif not key.startswith('_'):
                                weights[key] = _make_json_serializable(value)

    return weights


def extract_config_metadata(config: Any) -> dict[str, Any]:
    """Extract relevant configuration metadata.

    Args:
        config: Config object

    Returns:
        Dictionary with config information
    """
    metadata = {}

    if hasattr(config, 'eval'):
        eval_config = config.eval
        if hasattr(eval_config, 'backend'):
            backend = eval_config.backend
            if hasattr(backend, 'model'):
                metadata['model'] = backend.model
            if hasattr(backend, 'type'):
                metadata['backend_type'] = backend.type

        if hasattr(eval_config, 'num_examples'):
            metadata['num_examples'] = eval_config.num_examples
        if hasattr(eval_config, 'rollouts_per_example'):
            metadata['rollouts_per_example'] = eval_config.rollouts_per_example
        if hasattr(eval_config, 'max_tokens'):
            metadata['max_tokens'] = eval_config.max_tokens
        if hasattr(eval_config, 'extra_sampling_args'):
            metadata['extra_sampling_args'] = eval_config.extra_sampling_args

    if hasattr(config, 'seed'):
        metadata['seed'] = config.seed

    if hasattr(config, 'eval_run_name'):
        metadata['eval_run_name'] = config.eval_run_name

    return metadata


def compute_summary_statistics(
    results: vf.GenerateOutputs,
    config: Any,
    timing: dict[str, float]
) -> dict[str, Any]:
    """Compute comprehensive summary statistics from evaluation results.

    Args:
        results: GenerateOutputs object from evaluation
        config: Config object with evaluation settings
        timing: Timing information dictionary

    Returns:
        Dictionary with complete summary statistics
    """
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_rollouts": len(results.reward),
        },
        "config": extract_config_metadata(config),
        "reward_weights": extract_reward_weights(config),
        "timing": timing,
        "overall": {},
        "by_segment": {}
    }

    # Update metadata with config info
    summary["metadata"].update(summary["config"])

    # Compute overall statistics
    summary["overall"]["total_reward"] = compute_statistics(results.reward)

    for metric_name, metric_values in results.metrics.items():
        summary["overall"][metric_name] = compute_statistics(metric_values)

    # Compute per-segment statistics
    segments = extract_segments(results)

    if segments:
        for segment_name, indices in segments.items():
            segment_stats = {}

            # Get rewards for this segment
            segment_rewards = [results.reward[i] for i in indices]
            segment_stats["total_reward"] = compute_statistics(segment_rewards)

            # Get metrics for this segment
            for metric_name, metric_values in results.metrics.items():
                segment_metrics = [metric_values[i] for i in indices]
                segment_stats[metric_name] = compute_statistics(segment_metrics)

            summary["by_segment"][segment_name] = segment_stats

    return summary


def save_summary_json(
    summary: dict[str, Any],
    output_path: Path
) -> Path:
    """Save summary statistics to JSON file.

    Args:
        summary: Summary statistics dictionary
        output_path: Base output path (e.g., .pkl file path)

    Returns:
        Path to saved summary file
    """
    # Change extension from .pkl to _summary.json
    # e.g., "eval_gen_benign_14B_full.pkl" -> "eval_gen_benign_14B_full_summary.json"
    summary_path = output_path.with_stem(output_path.stem + "_summary").with_suffix('.json')

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary_path


def generate_and_save_summary(
    results: vf.GenerateOutputs,
    config: Any,
    timing: dict[str, float],
    output_path: Path
) -> Path:
    """Generate summary statistics and save to JSON file.

    This is the main entry point for automatic summary generation.

    Args:
        results: GenerateOutputs object from evaluation
        config: Config object with evaluation settings
        timing: Timing information dictionary
        output_path: Base output path (e.g., .pkl file path)

    Returns:
        Path to saved summary file
    """
    summary = compute_summary_statistics(results, config, timing)
    summary_path = save_summary_json(summary, output_path)
    return summary_path
