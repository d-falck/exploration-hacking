import pickle
import json
from pathlib import Path
import random
import time
import os
from typing import Literal

from dotenv import load_dotenv
import numpy as np
import verifiers as vf

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.eval import EvalConfig
from exploration_hacking.util import MLFlowLogger, create_trace_logger

# Set tokenizers parallelism before loading tokenizers to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()


class TimingData:
    """Store timing information for different phases of evaluation."""
    def __init__(self):
        self.total_start_time = None
        self.total_end_time = None
        self.generation_start_time = None
        self.generation_end_time = None
        self.scoring_start_time = None
        self.scoring_end_time = None
        self.per_sample_generation_times = []
        self.per_sample_scoring_times = []

    def get_timing_summary(self):
        """Get a summary of all timing data."""
        summary = {}

        if self.total_start_time and self.total_end_time:
            summary['total_time_seconds'] = self.total_end_time - self.total_start_time

        if self.generation_start_time and self.generation_end_time:
            summary['generation_time_seconds'] = self.generation_end_time - self.generation_start_time
            if self.per_sample_generation_times:
                summary['mean_generation_time_per_sample_seconds'] = float(np.mean(self.per_sample_generation_times))

        if self.scoring_start_time and self.scoring_end_time:
            summary['scoring_time_seconds'] = self.scoring_end_time - self.scoring_start_time
            if self.per_sample_scoring_times:
                summary['mean_scoring_time_per_sample_seconds'] = float(np.mean(self.per_sample_scoring_times))

        return summary


def timed_eval(env: vf.Environment, config: EvalConfig, timing_data: TimingData) -> vf.GenerateOutputs:
    """Timed version of eval that tracks generation and scoring phases."""
    from exploration_hacking.eval import eval as standard_eval

    # Set eval workers to 45 for increased parallelism
    optimal_workers = 45

    import multiprocessing
    num_cores = multiprocessing.cpu_count()

    print(f"Hardware detected: {num_cores} CPU cores")
    print(f"Using {optimal_workers} eval workers with subprocess-based hybrid evaluation")

    # Set max eval workers for parallel sandbox evaluation
    os.environ["MAX_EVAL_WORKERS"] = str(optimal_workers)

    # Start timing (note: we can't easily separate generation and scoring timing
    # with the standard eval function, but we can time the overall evaluation)
    timing_data.generation_start_time = time.time()

    # Use the standard eval function which handles all the complexity correctly
    results = standard_eval(env, config)

    timing_data.generation_end_time = time.time()

    # For backward compatibility, set scoring times to None
    # (we can't separate them with the standard eval function)
    timing_data.scoring_start_time = timing_data.generation_start_time
    timing_data.scoring_end_time = timing_data.generation_end_time

    return results


class Config(ExperimentConfig):
    environment: EnvironmentConfig
    eval: EvalConfig
    output_path: Path
    eval_run_name: str
    logging_destination: Literal["inspect", "mlflow"] = "inspect"


def _log_results(results: vf.GenerateOutputs, config: Config):
    """Log results using configured logger (inspect or mlflow)."""
    # Build task metadata
    task_metadata = {
        "experiment_name": config.eval_run_name,
        "num_examples": config.eval.num_examples,
        "rollouts_per_example": config.eval.rollouts_per_example,
    }
    if hasattr(config.environment, 'model_dump'):
        task_metadata["environment"] = config.environment.model_dump()

    eval_path = config.output_path.with_suffix('.eval')

    with create_trace_logger(
        logger_type=config.logging_destination,
        experiment_name=config.eval_run_name,
        model_name=config.eval.backend.model,
        output_path=str(eval_path),
        task_metadata=task_metadata,
    ) as logger:
        logger.log_spans_from_results(
            results.prompt,
            results.completion,
            results.reward,
            results.metrics,
            results.answer,
            results.info,
        )

    if config.logging_destination == "inspect":
        print(f"Inspect AI eval file saved to: {eval_path}")
    else:
        print(f"MLFlow logging complete for: {config.eval_run_name}")


def _save_json_results(results: vf.GenerateOutputs, config: Config, timing_data: TimingData = None):
    """Save detailed results in JSON format using Pydantic serialization."""
    from typing import Any
    from pydantic import field_serializer, BaseModel

    class Trajectories(vf.GenerateOutputs):
        @field_serializer("*", mode="plain")
        def serialize_field(self, value: Any) -> object:
            if isinstance(value, BaseModel):
                return value.model_dump()
            elif isinstance(value, list):
                return [self.serialize_field(v) for v in value]
            elif isinstance(value, dict):
                return {k: self.serialize_field(v) for k, v in value.items()}
            else:
                return value

    # Convert GenerateOutputs to Trajectories with proper serialization
    trajectories = Trajectories(**{k: getattr(results, k) for k in type(results).model_fields})

    # Build the full results structure
    json_results = {
        "experiment_name": config.eval_run_name,
        "model": config.eval.backend.model,
        "num_examples": config.eval.num_examples,
        "rollouts_per_example": config.eval.rollouts_per_example,
        "results": trajectories.model_dump(),
    }

    # Add timing data if available
    if timing_data:
        json_results["timing"] = timing_data.get_timing_summary()

    # Save JSON file
    json_path = config.output_path.with_suffix('.json')
    with json_path.open("w") as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"JSON results saved to: {json_path}")
    return json_results


def main(config: Config):
    if config.apply_seed_globally:
        random.seed(config.seed)
        np.random.seed(config.seed)

    # Ensure output directory exists
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {config.output_path}")

    # Start total timing
    total_start_time = time.time()

    env = load_environment(config.environment, seed=config.seed)

    # Create timing data object
    timing_data = TimingData()
    timing_data.total_start_time = total_start_time

    # Use timed evaluation
    results = timed_eval(env, config.eval, timing_data)

    # End total timing
    total_end_time = time.time()
    timing_data.total_end_time = total_end_time

    print("Evaluation complete!")
    print("Available metrics:", list(results.metrics.keys()))

    # Print timing information
    timing_summary = timing_data.get_timing_summary()
    print("\n=== TIMING SUMMARY ===")
    if 'total_time_seconds' in timing_summary:
        print(f"Total evaluation time: {timing_summary['total_time_seconds']:.2f} seconds")
    if 'generation_time_seconds' in timing_summary:
        print(f"Generation time: {timing_summary['generation_time_seconds']:.2f} seconds")
        if 'generation_time_per_sample_seconds' in timing_summary:
            gen_stats = timing_summary['generation_time_per_sample_seconds']
            print(f"  Per sample: {gen_stats['mean']:.3f}±{gen_stats['std']:.3f}s (range: {gen_stats['min']:.3f}-{gen_stats['max']:.3f}s)")
    if 'scoring_time_seconds' in timing_summary:
        print(f"Scoring time: {timing_summary['scoring_time_seconds']:.2f} seconds")
        if 'scoring_time_per_sample_seconds' in timing_summary:
            score_stats = timing_summary['scoring_time_per_sample_seconds']
            print(f"  Per sample: {score_stats['mean']:.3f}±{score_stats['std']:.3f}s (range: {score_stats['min']:.3f}-{score_stats['max']:.3f}s)")
    print("======================\n")

    # Check for different possible accuracy metric names
    accuracy_key = None
    for key in ["accuracy", "bigcodebench_accuracy", "pass_at_1", "correct"]:
        if key in results.metrics:
            accuracy_key = key
            break

    if accuracy_key:
        print(f"Accuracy ({accuracy_key}): ", np.mean(results.metrics[accuracy_key]))
        print(f"Std ({accuracy_key}): ", np.std(results.metrics[accuracy_key]))
    else:
        print("No recognized accuracy metric found. Available metrics:", results.metrics.keys())

    # Create enhanced results object with timing data for pickle
    enhanced_results = {
        'results': results,
        'timing': timing_summary
    }

    with config.output_path.open("wb") as f:
        pickle.dump(enhanced_results, f)

    # Save detailed JSON results with timing
    try:
        _save_json_results(results, config, timing_data)
    except Exception as e:
        print(f"Error saving JSON results: {e}")

    # Log results to configured destination
    try:
        _log_results(results, config)
    except Exception as e:
        print(f"Error logging results: {e}")


if __name__ == "__main__":
    main(Config())
