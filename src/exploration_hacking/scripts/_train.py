from dotenv import load_dotenv
import wandb

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments.science import ScienceEnvConfig, load_environment
from exploration_hacking.rl import GRPOConfig, run_grpo


load_dotenv()


class Config(ExperimentConfig):
    environment: ScienceEnvConfig
    rl: GRPOConfig
    inference_gpus: list[int]
    training_gpus: list[int]


def main(config: Config):
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
    )

    env = load_environment(config.environment)
    run_grpo(env, config.rl, wandb.run.name)


if __name__ == "__main__":
    main(Config())
