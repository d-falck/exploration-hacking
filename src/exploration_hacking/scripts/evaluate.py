import pickle
import random
from pathlib import Path

from dotenv import load_dotenv
import mlflow

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments.science import ScienceEnvConfig, load_environment
from exploration_hacking.eval import eval, EvalConfig


load_dotenv()


class Config(ExperimentConfig):
    environment: ScienceEnvConfig
    eval: EvalConfig
    output_path: Path
    eval_run_name: str


def main(config: Config):
    random.seed(config.seed)

    env = load_environment(config.environment)
    results = eval(env, config.eval)

    mlflow.create_experiment(config.eval_run_name)
    mlflow.set_experiment(config.eval_run_name)
    
    for i in range(len(results.prompt)):
        with mlflow.start_span("generation") as span:
            span.set_inputs({"prompt": results.prompt[i]})
            span.set_outputs({"completion": results.completion[i], "answer": results.answer[i]})
            tags = {k: str(v[i]) for k, v in results.metrics.items()}
            tags |= {"reward": str(results.reward[i])}
            mlflow.update_current_trace(tags=tags)

    with config.output_path.open("wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(Config())
