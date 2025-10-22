import pickle
import json
from pathlib import Path
import random
from datetime import datetime
import time
import os

from dotenv import load_dotenv
import mlflow
import numpy as np
import torch
import verifiers as vf

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.eval import eval, EvalConfig
from exploration_hacking.util import MLFlowLogger

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
    from exploration_hacking.vllm import vllm_server
    from openai import OpenAI

    # Start generation timing
    timing_data.generation_start_time = time.time()

    # Set eval workers to 45 for increased parallelism
    optimal_workers = 45

    import multiprocessing
    num_cores = multiprocessing.cpu_count()

    print(f"Hardware detected: {num_cores} CPU cores")
    print(f"Using {optimal_workers} eval workers with subprocess-based hybrid evaluation")

    # Set max eval workers for parallel sandbox evaluation
    os.environ["MAX_EVAL_WORKERS"] = str(optimal_workers)

    vllm_args = config.vllm_args + [
        "--data-parallel-size",
        str(config.num_gpus),
    ]

    with vllm_server(
        config.model,
        *vllm_args,
        lora_path=config.lora_path,
        max_lora_rank=config.max_lora_rank,
    ):
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="default")

        # Call evaluate with score_rollouts=False to separate timing phases
        generation_results = env.evaluate(
            client,
            config.model,
            num_examples=config.num_examples,
            rollouts_per_example=config.rollouts_per_example,
            max_concurrent=config.max_concurrent,
            sampling_args=vf.SamplingArgs(
                max_tokens=config.max_tokens,
            ),
            score_rollouts=False,  # We'll score separately
        )

        # Track per-sample generation timing
        num_samples = len(generation_results.prompt) if generation_results.prompt else config.num_examples * config.rollouts_per_example

        timing_data.generation_end_time = time.time()

        # Calculate per-sample generation time
        generation_time = timing_data.generation_end_time - timing_data.generation_start_time
        if num_samples > 0:
            per_sample_generation_time = generation_time / num_samples
            timing_data.per_sample_generation_times = [per_sample_generation_time] * num_samples

        # Start scoring timing
        timing_data.scoring_start_time = time.time()

        # Score the rollouts
        import asyncio

        async def score_async():
            return await env.rubric.score_rollouts(
                prompts=generation_results.prompt,
                completions=generation_results.completion,
                answers=generation_results.answer,
                states=generation_results.state,
                tasks=generation_results.task,
                infos=generation_results.info,
                max_concurrent=config.max_concurrent,
                apply_weights=True,
            )

        # Run scoring in async context
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            rollout_scores = loop.run_until_complete(score_async())
        except RuntimeError:
            rollout_scores = asyncio.run(score_async())

        timing_data.scoring_end_time = time.time()

        # Calculate per-sample scoring time
        scoring_time = timing_data.scoring_end_time - timing_data.scoring_start_time
        if num_samples > 0:
            per_sample_scoring_time = scoring_time / num_samples
            timing_data.per_sample_scoring_times = [per_sample_scoring_time] * num_samples

        # Combine results
        generation_results.reward = rollout_scores.reward
        generation_results.metrics = rollout_scores.metrics

        return generation_results


class Config(ExperimentConfig):
    environment: EnvironmentConfig
    eval: EvalConfig
    output_path: Path
    eval_run_name: str


def _log_to_mlflow(results: vf.GenerateOutputs, config: Config):
    mlflow_logger = MLFlowLogger(config.eval_run_name, concurrent=True)
    mlflow_logger.log_spans_from_results(
        results.prompt,
        results.completion,
        results.reward,
        results.metrics,
        results.answer,
        results.info,
    )


def _save_json_results(results: vf.GenerateOutputs, config: Config, timing_data: TimingData = None):
    """Save detailed results in JSON format."""
    from exploration_hacking.environments.bigcodebench.sanitize import extract_code_from_markdown
    from exploration_hacking.environments.bigcodebench.sandbox import persistent_worker_evaluate
    
    json_results = {
        "experiment_name": config.eval_run_name,
        "model": config.eval.model,
        "num_examples": config.eval.num_examples,
        "rollouts_per_example": config.eval.rollouts_per_example,
        "samples": []
    }
    
    # Process each sample
    for i in range(len(results.prompt)):
        # Handle completion format - could be string or list of messages
        completion = results.completion[i]
        if isinstance(completion, list) and len(completion) > 0:
            # Extract content from the last assistant message
            model_generation = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            model_generation = str(completion)
            
        sample = {
            "index": i,
            "prompt": results.prompt[i],
            "model_generation": model_generation,
            "answer": results.answer[i] if results.answer else None,
            "reward": float(results.reward[i]),
            "metrics": {k: float(v[i]) if not isinstance(v[i], (list, dict)) else v[i] 
                       for k, v in results.metrics.items()}
        }
        
        # Extract BigCodeBench-specific information from state and info
        state = results.state[i] if i < len(results.state) else {}
        info = results.info[i] if i < len(results.info) else {}
        
        # Try to get task information from either state or info
        task_id = info.get("task_id") or state.get("task_id") or f"task_{i}"
        entry_point = info.get("entry_point") or state.get("entry_point") or ""
        code_prompt = info.get("code_prompt") or state.get("code_prompt") or ""
        test_code = info.get("test") or state.get("test") or ""
        
        sample["task_id"] = task_id
        sample["entry_point"] = entry_point
        sample["code_prompt"] = code_prompt
        
        # Extract sanitized code from model generation
        try:
            sanitized_code = extract_code_from_markdown(model_generation)
            sample["sanitized_generation"] = sanitized_code
        except Exception as e:
            sample["sanitized_generation"] = model_generation
            sample["sanitization_error"] = str(e)
        
        # Get test execution results from cached results (no double evaluation!)
        cached_result_key = f"test_results_{task_id}"
        if cached_result_key in info:
            # Use cached test results from reward function evaluation
            sample["test_results"] = info[cached_result_key]
        elif test_code and entry_point:
            # Fallback: re-evaluate only if not cached (shouldn't happen normally)
            print(f"DEBUG: DOUBLE EVALUATION detected for {task_id} - this should not happen!")
            try:
                test_result = persistent_worker_evaluate(
                    task_id=task_id,
                    solution=sample.get("sanitized_generation", model_generation),
                    code_prompt=code_prompt,
                    test=test_code,
                    entry_point=entry_point,
                )
                sample["test_results"] = test_result
            except Exception as e:
                sample["test_results"] = {"error": str(e)}
        else:
            sample["test_results"] = {"note": "Test code or entry point not available"}
        
        # Calculate test-level accuracy from test results
        test_results = sample.get("test_results", {})
        if isinstance(test_results, dict) and "num_tests" in test_results and "num_tests_passed" in test_results:
            num_tests = test_results.get("num_tests", 0)
            num_passed = test_results.get("num_tests_passed", 0)
            if num_tests > 0:
                sample["test_level_accuracy"] = num_passed / num_tests
            else:
                sample["test_level_accuracy"] = 0.0
        else:
            sample["test_level_accuracy"] = 0.0
        
        json_results["samples"].append(sample)
    
    # Calculate aggregate statistics
    json_results["aggregate_metrics"] = {}
    for metric_name, metric_values in results.metrics.items():
        try:
            json_results["aggregate_metrics"][metric_name] = {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values))
            }
        except:
            json_results["aggregate_metrics"][metric_name] = "Unable to compute"

    # Calculate test-level accuracy aggregate statistics
    test_level_accuracies = [sample["test_level_accuracy"] for sample in json_results["samples"]]
    if test_level_accuracies:
        json_results["aggregate_metrics"]["test_level_accuracy"] = {
            "mean": float(np.mean(test_level_accuracies)),
            "std": float(np.std(test_level_accuracies)),
            "min": float(np.min(test_level_accuracies)),
            "max": float(np.max(test_level_accuracies))
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

    # Detect which dataset is being used
    dataset_type = None
    if config.environment.bigcodebench is not None:
        dataset_type = "bigcodebench"
    elif config.environment.science is not None:
        dataset_type = "wmdp"

    # Generate timestamped output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset-specific filename with timestamp
    if dataset_type:
        base_name = f"eval_{dataset_type}_{timestamp}"
        config.output_path = output_dir / f"{base_name}.pkl"
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
    for key in ["accuracy", "bcb_test_level_accuracy", "bcb_pass_at_1", "correct"]:
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

    try:
        _log_to_mlflow(results, config)
    except Exception as e:
        print(f"Error logging to MLFlow: {e}")


if __name__ == "__main__":
    main(Config())
