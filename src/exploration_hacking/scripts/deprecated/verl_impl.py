import json
import os
import subprocess
from typing import Literal

import pandas as pd
from datasets import Dataset
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp
from transformers import AutoTokenizer

from exploration_hacking.settings.evalplus import load_evalplus_problems

SYSTEM_PROMPT = """You are an expert Python programmer. Complete the given function. Return your solution within a markdown python code block, and make sure it's the only such code block in your response."""


class Config(BaseSettings):
    model_name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct", description="Model name"
    )
    dataset: Literal["humaneval", "mbpp"] = Field(
        default="humaneval", description="Dataset to use"
    )
    max_problems: int = Field(
        default=20, description="Maximum number of problems (0 for all)"
    )
    num_epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=256, description="Total batch size for training")
    mini_batch_size: int = Field(default=64, description="PPO mini batch size")
    micro_batch_size_per_gpu: int = Field(
        default=16, description="Micro batch size per GPU"
    )
    num_rollouts: int = Field(
        default=4, description="Number of rollouts per problem (group size)"
    )
    temperature: float = Field(default=1.0, description="Temperature for generation")
    max_prompt_length: int = Field(default=512, description="Maximum prompt length")
    max_response_length: int = Field(default=512, description="Maximum response length")
    learning_rate: float = Field(default=1e-6, description="Learning rate")
    kl_coef: float = Field(default=0.001, description="KL divergence coefficient")
    output_dir: str = Field(
        default="artifacts/verl_evalplus", description="Output directory"
    )
    seed: int = Field(default=42, description="Random seed")
    n_gpus: int = Field(default=1, description="Number of GPUs")
    run_command: bool = Field(
        default=False, description="Run the training command automatically"
    )


def prepare_dataset(
    problems: dict[str, EvalPlusProblem], tokenizer
) -> tuple[Dataset, Dataset]:
    data = []

    for problem in problems.values():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        data.append(
            {
                "prompt": prompt,
                "task_id": problem.task_id,
            }
        )

    dataset = Dataset.from_list(data)

    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

    return train_dataset, val_dataset


def save_datasets_as_parquet(
    train_dataset: Dataset, val_dataset: Dataset, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    return train_path, val_path


def build_verl_command(
    config: Config, train_path: str, val_path: str, problems_path: str
) -> list[str]:
    env = {
        "EVALPLUS_PROBLEMS_PATH": problems_path,
        "EVALPLUS_DATASET": config.dataset,
    }

    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        f"data.train_batch_size={config.batch_size}",
        f"data.max_prompt_length={config.max_prompt_length}",
        f"data.max_response_length={config.max_response_length}",
        f"actor_rollout_ref.model.path={config.model_name}",
        f"actor_rollout_ref.actor.optim.lr={config.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={config.kl_coef}",
        f"actor_rollout_ref.rollout.n={config.num_rollouts}",
        f"actor_rollout_ref.rollout.temperature={config.temperature}",
        f"algorithm.adv_estimator=grpo",
        f"trainer.total_epochs={config.num_epochs}",
        f"trainer.n_gpus_per_node={config.n_gpus}",
        f"trainer.save_dir={config.output_dir}",
        f"custom_reward_function.path=src/exploration_hacking/scripts/evalplus_grpo/verl_reward.py",
    ]
    return cmd, env


def main(config: Config):

    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {config.dataset} problems...")
    problems = load_evalplus_problems(config.dataset, config.max_problems)

    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_dataset(problems, tokenizer)

    data_dir = os.path.join(config.output_dir, "data")
    train_path, val_path = save_datasets_as_parquet(
        train_dataset, val_dataset, data_dir
    )
    print(f"Saved train data to: {train_path}")
    print(f"Saved val data to: {val_path}")

    problems_path = os.path.join(config.output_dir, "problems.json")
    with open(problems_path, "w") as f:
        json.dump({k: v.model_dump() for k, v in problems.items()}, f)
    print(f"Saved problems to: {problems_path}")

    cmd, env = build_verl_command(config, train_path, val_path, problems_path)

    print("\n" + "=" * 50)
    print("GRPO Training Command:")
    print("=" * 50)
    print(f"EVALPLUS_PROBLEMS_PATH={problems_path} \\")
    print(f"EVALPLUS_DATASET={config.dataset} \\")
    print(" \\\n    ".join(cmd))

    if config.run_command:
        print("\n" + "=" * 50)
        print("Starting training...")
        print("=" * 50)
        full_env = os.environ.copy()
        full_env.update(env)
        subprocess.run(cmd, env=full_env)
    else:
        print("\nTo run training, either:")
        print("1. Copy and run the command above")
        print("2. Re-run with --run-command flag")


if __name__ == "__main__":
    config = CliApp.run(Config)
    main(config)
