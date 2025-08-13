import os
import random
import subprocess
import pickle
from pathlib import Path

from dotenv import load_dotenv
import torch
import wandb

from exploration_hacking.config import ExperimentConfig
from exploration_hacking.data import DataConfig, load_dataset
from exploration_hacking.logging import add_completion_logging, save_completion_logs
from exploration_hacking.rewards.factory import RewardConfig, get_reward_functions

load_dotenv()


class Config(ExperimentConfig):
    data: DataConfig
    model: DataConfig  # Using simplified model config from YAML
    reward: RewardConfig


def main(config: Config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name or "verl-grpo",
        config=dict(config=config.model_dump()),
    )
    
    run_name = wandb.run.name
    output_dir = Path("artifacts/verl_grpo") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    dataset, problems = load_dataset(config.data)
    data_path = output_dir / "train_data.parquet"
    dataset.to_parquet(data_path)
    
    # Setup rewards
    reward_funcs = get_reward_functions(
        config.data.dataset_name, problems, config.reward, None
    )
    reward_funcs = [add_completion_logging(f) for f in reward_funcs]
    
    # Save reward data for VERL
    reward_data_path = output_dir / "reward_data.pkl"
    with open(reward_data_path, 'wb') as f:
        pickle.dump({'reward_funcs': reward_funcs, 'problems': problems}, f)
    
    # Run VERL with the static YAML config
    try:
        subprocess.run([
            "python", "-m", "verl.trainer.main_ppo",
            "--config", "config/ppo_trainer.yaml",
            "--output-dir", str(output_dir),
        ], cwd=output_dir, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("VERL not installed or training failed. Install with: pip install verl[vllm]")
    finally:
        save_completion_logs(str(output_dir))
        wandb.finish()


if __name__ == "__main__":
    main(Config())