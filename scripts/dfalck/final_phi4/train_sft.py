"""SFT from an existing LoRA checkpoint on new data.

Unlike the main SFT script, this loads an existing LoRA adapter and continues
training it, rather than creating a fresh one.

Usage:
    torchrun --nproc_per_node=4 scripts/dfalck/benign_sft_elicitation/train_sft.py \
        --config etc/dfalck/science_conditional/benign_sft_elicitation/sft.yaml
"""

import copy
import logging
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import load_from_disk, Dataset
from dotenv import load_dotenv
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from exploration_hacking.dtypes import ExperimentConfig

load_dotenv()
logging.basicConfig(level=logging.INFO)


class _ModelConfig(BaseModel):
    model_name: str
    lora_checkpoint: str


class _DatasetConfig(BaseModel):
    local_path: Path


class _TrainingConfig(BaseModel):
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 10
    save_steps: int = 10
    save_total_limit: int | None = None
    logging_steps: int = 1
    max_grad_norm: float = 1.0


class Config(ExperimentConfig):
    model: _ModelConfig
    dataset: _DatasetConfig
    training: _TrainingConfig
    output_dir: str
    gpus: list[int]
    transform_to_single_turn: bool = True


def transform_multi_turn_to_single_turn(dataset):
    """Transform multi-turn rollouts into separate samples for each assistant completion."""
    new_samples = []
    for sample in dataset:
        prompt = sample["prompt"]
        completion = sample["completion"]
        tools = sample["tools"]
        context = list(prompt)
        for message in completion:
            if message["role"] == "assistant":
                new_samples.append({
                    "prompt": copy.deepcopy(context),
                    "completion": [message],
                    "tools": tools,
                })
            context.append(message)
    return Dataset.from_list(new_samples)


def main(config: Config):
    if config.apply_seed_globally:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
    )

    # Load data
    dataset = load_from_disk(config.dataset.local_path)
    logging.info(f"Loaded {len(dataset)} examples")

    if config.transform_to_single_turn:
        original_size = len(dataset)
        dataset = transform_multi_turn_to_single_turn(dataset)
        logging.info(f"Transformed {original_size} -> {len(dataset)} single-turn samples")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )

    # Load existing LoRA checkpoint
    model = PeftModel.from_pretrained(
        model, config.model.lora_checkpoint, is_trainable=True
    )
    logging.info(f"Loaded LoRA from {config.model.lora_checkpoint}")
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        report_to="wandb",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False, "determinism_check": "none"},
        max_grad_norm=config.training.max_grad_norm,
        data_seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main(Config())
