import unsloth  # Must be imported first.

import random

from dotenv import load_dotenv
import torch
import torch.distributed as dist
import wandb
import weave

from exploration_hacking.config import ExperimentConfig
from exploration_hacking.data import DataConfig, load_dataset
from exploration_hacking.rewards import RewardConfig, reward_functions_factory
from exploration_hacking.rl import RLConfig, run_grpo
from exploration_hacking.model import ModelConfig, load_peft_model


load_dotenv()


class Config(ExperimentConfig):
    data: DataConfig
    model: ModelConfig
    reward: RewardConfig
    rl: RLConfig


def main(config: Config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config=config.model_dump(),
    )
    weave.init(config.wandb_project, settings={"print_call_link": False})

    model, tokenizer = load_peft_model(config.model)
    dataset, problems = load_dataset(config.data)

    reward_funcs = reward_functions_factory(
        config.data.dataset_name, problems, config, tokenizer
    )

    try:
        run_grpo(model, tokenizer, dataset, reward_funcs, config)
    finally:
        model.save_pretrained(f"{config.output_dir}/final")
        tokenizer.save_pretrained(f"{config.output_dir}/final")
        if dist.is_initialized():
            dist.destroy_process_group()
        wandb.finish()


if __name__ == "__main__":
    config = Config()
    print(config.model_dump())
    main(config)
