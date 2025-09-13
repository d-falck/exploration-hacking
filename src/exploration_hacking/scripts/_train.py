from dotenv import load_dotenv
import logging
import wandb


from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments.science import ScienceEnvConfig, load_environment
from exploration_hacking.rl import RLConfig, run_grpo


load_dotenv()
logging.basicConfig(level=logging.INFO)


class Config(ExperimentConfig):
    environment: ScienceEnvConfig
    rl: RLConfig
    inference_gpus: list[int]
    training_gpus: list[int]


def main(config: Config):
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
    )

    env = load_environment(config.environment)
    run_grpo(
        env,
        config.rl,
        wandb.run.name,
        len(config.training_gpus),
    )


if __name__ == "__main__":
    main(Config())
