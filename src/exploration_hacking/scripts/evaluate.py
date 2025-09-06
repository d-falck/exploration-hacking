import pickle
import random
from pathlib import Path

from dotenv import load_dotenv

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments.science import ScienceEnvConfig, load_environment
from exploration_hacking.eval import eval, EvalConfig


load_dotenv()


class Config(ExperimentConfig):
    environment: ScienceEnvConfig
    eval: EvalConfig
    output_path: Path


def main(config: Config):
    random.seed(config.seed)

    env = load_environment(config.environment)
    results = eval(config.eval, env)

    with config.output_path.open("wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(Config())
