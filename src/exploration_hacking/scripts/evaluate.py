import pickle
import random
from pathlib import Path

from dotenv import load_dotenv
import mlflow
import verifiers as vf

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments.science import ScienceEnvConfig, load_environment
from exploration_hacking.eval import eval, EvalConfig
from exploration_hacking.util import true_random_context


load_dotenv()


class Config(ExperimentConfig):
    environment: ScienceEnvConfig
    eval: EvalConfig
    output_path: Path
    eval_run_name: str


def _log_to_mlflow(results: vf.GenerateOutputs, config: Config):
    with true_random_context():
        try:
            mlflow.create_experiment(config.eval_run_name)
        except Exception as e:
            print(f"Error creating MLFlow experiment: {e}")

        mlflow.set_experiment(config.eval_run_name)

        for i in range(len(results.prompt)):
            inputs = {"prompt": results.prompt[i]}
            tags = {k: str(v[i]) for k, v in results.metrics.items()}
            tags |= {"reward": str(results.reward[i])}
            span = mlflow.start_span_no_context("generation", inputs=inputs, tags=tags)
            try:
                span.set_outputs(
                    {"completion": results.completion[i], "answer": results.answer[i]}
                )
            finally:
                span.end()


def main(config: Config):
    random.seed(config.seed)

    env = load_environment(config.environment)
    results = eval(env, config.eval)

    with config.output_path.open("wb") as f:
        pickle.dump(results, f)

    try:
        _log_to_mlflow(results, config)
    except Exception as e:
        print(f"Error logging to MLFlow: {e}")


if __name__ == "__main__":
    main(Config())
