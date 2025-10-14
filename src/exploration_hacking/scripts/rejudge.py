import asyncio
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import mlflow
import numpy as np
import verifiers as vf

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.util import MLFlowLogger


load_dotenv()


class Config(ExperimentConfig):
    environment: EnvironmentConfig
    input_path: Path  # Path to pickled GenerateOutputs from previous evaluation
    output_path: Path
    eval_run_name: str
    max_concurrent: int = 32


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


async def main(config: Config):
    print(f"Loading previous results from {config.input_path}")
    with config.input_path.open("rb") as f:
        previous_results = pickle.load(f)
    
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
    with config.output_path.open("wb") as f:
        pickle.dump(results, f)
    
    print("Saved rejudged results to ", config.output_path)
    print("Logging to MLFlow...")
    
    try:
        _log_to_mlflow(results, config)
    except Exception as e:
        print(f"Error logging to MLFlow: {e}")


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