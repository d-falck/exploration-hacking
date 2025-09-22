import pickle
from pathlib import Path
import random

from dotenv import load_dotenv
import mlflow
import numpy as np
import torch
import verifiers as vf

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.eval import eval, EvalConfig
from exploration_hacking.util import MLFlowLogger


load_dotenv()


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
    )


def main(config: Config):
    if config.apply_seed_globally:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    env = load_environment(config.environment, seed=config.seed)
    results = eval(env, config.eval)

    print("Evaluation complete!")
    print("Accuracy: ", np.mean(results.metrics["accuracy"]))
    print("Std: ", np.std(results.metrics["accuracy"]))

    with config.output_path.open("wb") as f:
        pickle.dump(results, f)

    try:
        _log_to_mlflow(results, config)
    except Exception as e:
        print(f"Error logging to MLFlow: {e}")


if __name__ == "__main__":
    main(Config())
