from dotenv import load_dotenv
import logging
import random

import numpy as np
import wandb

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.rl import RLConfig, run_grpo


load_dotenv()
logging.basicConfig(level=logging.INFO)


class Config(ExperimentConfig):
    environment: EnvironmentConfig
    rl: RLConfig
    inference_gpus: list[int]
    training_gpus: list[int]
    inference_context_length_grace_factor: float = 2.0


def main(config: Config):
    if config.apply_seed_globally:
        random.seed(config.seed)
        np.random.seed(config.seed)

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
    )

    env = load_environment(config.environment, seed=config.seed)
    run_grpo(
        env,
        config.rl,
        wandb.run.name,
        len(config.training_gpus),
    )


if __name__ == "__main__":
    main(Config())
