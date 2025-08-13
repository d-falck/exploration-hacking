from pydantic import BaseModel
import numpy as np
from pathlib import Path


class RLConfig(BaseModel):
    temperature: float = 1.0
    top_p: float = 0.95
    num_epochs: int = 8
    batch_size: int = 12
    learning_rate: float = 1e-4
    num_rollouts: int = 12  # GRPO group size
    warmup_ratio: float = 0.005
    output_dir: str = "artifacts/verl_grpo"
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    kl_loss_coef: float = 0.001


def create_verl_config(rl_config: RLConfig, model_path: str, data_path: str, max_seq_length: int):
    max_prompt_length = max_seq_length // 2
    max_response_length = max_seq_length - max_prompt_length
    
    return {
        "algorithm.adv_estimator": "grpo",
        "data.train_files": data_path,
        "data.train_batch_size": rl_config.batch_size * rl_config.num_rollouts,
        "data.max_prompt_length": max_prompt_length,
        "data.max_response_length": max_response_length,
        "actor_rollout_ref.model.path": model_path,
        "actor_rollout_ref.actor.optim.lr": rl_config.learning_rate,
        "actor_rollout_ref.actor.optim.weight_decay": rl_config.weight_decay,
        "actor_rollout_ref.actor.ppo_epochs": rl_config.num_epochs,
        "actor_rollout_ref.actor.ppo_mini_batch_size": rl_config.batch_size,
        "actor_rollout_ref.actor.gradient_accumulation_steps": rl_config.gradient_accumulation_steps,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": rl_config.kl_loss_coef,
        "actor_rollout_ref.rollout.n": rl_config.num_rollouts,
        "actor_rollout_ref.rollout.temperature": rl_config.temperature,
        "actor_rollout_ref.rollout.top_p": rl_config.top_p,
        "trainer.output_dir": rl_config.output_dir,
        "trainer.num_train_epochs": rl_config.num_epochs,
        "trainer.warmup_ratio": rl_config.warmup_ratio,
    }


def get_max_prompt_length(dataset, tokenizer):
    lengths = []
    for item in dataset:
        tokens = tokenizer.apply_chat_template(
            item["prompt"],
            add_generation_prompt=True,
            tokenize=True,
        )
        lengths.append(len(tokens))
    return int(np.max(lengths))