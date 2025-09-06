from dotenv import load_dotenv

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments.science import ScienceEnvConfig, load_environment
from exploration_hacking.rl import GRPOConfig, run_grpo


load_dotenv()


class Config(ExperimentConfig):
    environment: ScienceEnvConfig
    rl: GRPOConfig


def main(config: Config):
    env = load_environment(config.environment)
    run_grpo(env, config.rl)


if __name__ == "__main__":
    main(Config())
