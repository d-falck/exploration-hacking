import asyncio
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import mlflow
import numpy as np
import verifiers as vf

from typing import Literal

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.util import create_trace_logger


load_dotenv()


class Config(ExperimentConfig):
    environment: EnvironmentConfig
    input_path: Path  # Path to pickled GenerateOutputs from previous evaluation
    output_path: Path
    eval_run_name: str
    max_concurrent: int = 32
    logging_destination: Literal["inspect", "mlflow"] = "inspect"


def _log_results(results: vf.GenerateOutputs, config: Config):
    """Log results using configured logger (inspect or mlflow)."""
    # Build task metadata
    task_metadata = {
        "experiment_name": config.eval_run_name,
    }
    if hasattr(config.environment, 'model_dump'):
        task_metadata["environment"] = config.environment.model_dump()

    eval_path = config.output_path.with_suffix('.eval')

    # For rejudge, we don't have a model name in the config, so extract from environment or use default
    model_name = "rejudged"

    with create_trace_logger(
        logger_type=config.logging_destination,
        experiment_name=config.eval_run_name,
        model_name=model_name,
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


async def main(config: Config):
    print(f"Loading previous results from {config.input_path}")
    with config.input_path.open("rb") as f:
        loaded_data = pickle.load(f)

    # Handle both dict format (with 'results' key) and raw GenerateOutputs
    if isinstance(loaded_data, dict) and 'results' in loaded_data:
        previous_results = loaded_data['results']
        timing_data = loaded_data.get('timing', {})
        print(f"Loaded enhanced results with timing data")
    else:
        previous_results = loaded_data
        timing_data = {}

    print(f"Loaded {len(previous_results.prompt)} examples")
    
    # Load environment with new config (this includes new rubric settings)
    env = load_environment(config.environment, seed=config.seed)
    
    print("Re-judging with new rubric...")
    
    # Create new results object with same generations but reset rewards/metrics
    results = vf.GenerateOutputs(
        prompt=previous_results.prompt,
        completion=previous_results.completion,
        answer=previous_results.answer,
        state=previous_results.state,
        info=previous_results.info,
        task=previous_results.task,
        reward=[],
        metrics={},
    )
    
    # Re-score the rollouts with the new environment's rubric
    rollout_scores = await env.rubric.score_rollouts(
        prompts=results.prompt,
        completions=results.completion,
        answers=results.answer,
        states=results.state,
        tasks=results.task,
        infos=results.info,
        max_concurrent=config.max_concurrent,
        apply_weights=True,
    )
    results.reward = rollout_scores.reward
    results.metrics = rollout_scores.metrics
    
    print("Re-judging complete!")
    try:
        print("Accuracy: ", np.mean(results.metrics["accuracy"]))
        print("Std: ", np.std(results.metrics["accuracy"]))
    except Exception:
        pass
    
    print("Saving rejudged results...")
    # Save in the same enhanced format if timing data was present
    if timing_data:
        enhanced_results = {
            'results': results,
            'timing': timing_data
        }
        with config.output_path.open("wb") as f:
            pickle.dump(enhanced_results, f)
    else:
        with config.output_path.open("wb") as f:
            pickle.dump(results, f)
    
    print("Saved rejudged results to ", config.output_path)

    # Log results to configured destination
    try:
        _log_results(results, config)
    except Exception as e:
        print(f"Error logging results: {e}")


if __name__ == "__main__":
    config = Config()
    
    # Match the event loop setup from Environment.generate() for optimal performance
    # Use ThreadPoolExecutor for handling blocking I/O operations in parallel
    max_workers = 512  # Same as Environment default
    executor = ThreadPoolExecutor(max_workers=max_workers)
    loop = asyncio.new_event_loop()
    try:
        loop.set_default_executor(executor)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(config))
    finally:
        loop.close()
        asyncio.set_event_loop(None)
        # Shutdown the executor to prevent thread leaks
        executor.shutdown(wait=False)