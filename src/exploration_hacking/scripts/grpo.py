import unsloth  # Must be imported first.

import random

from dotenv import load_dotenv
import torch
import torch.distributed as dist
import wandb

from exploration_hacking.config import ExperimentConfig
from exploration_hacking.data import DataConfig, load_dataset
from exploration_hacking.logging import add_completion_logging, save_completion_logs
from exploration_hacking.rewards.factory import RewardConfig, get_reward_functions
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
        config=dict(config=config.model_dump()),
    )

    run_name = wandb.run.name
    output_dir = f"{config.rl.output_dir}/{run_name}"
    config.rl.output_dir = output_dir

    dataset, problems = load_dataset(config.data)
    model, tokenizer = load_peft_model(config.model)

    reward_funcs = get_reward_functions(
        config.data.dataset_name, problems, config.reward, tokenizer
    )
    reward_funcs = [add_completion_logging(f) for f in reward_funcs]

    try:
        run_grpo(
            model,
            tokenizer,
            dataset,
            reward_funcs,
            config.rl,
            config.model.max_seq_length,
        )
    finally:
        model.save_pretrained(f"{output_dir}/final")
        tokenizer.save_pretrained(f"{output_dir}/final")
        save_completion_logs(output_dir)
        if dist.is_initialized():
            dist.destroy_process_group()
        wandb.finish()


if __name__ == "__main__":
    main(Config())
