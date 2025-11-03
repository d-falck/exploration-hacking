from dotenv import load_dotenv
from pathlib import Path
import logging
import random
import os

import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset, load_from_disk
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from exploration_hacking.dtypes import ExperimentConfig


load_dotenv()
logging.basicConfig(level=logging.INFO)


class _ModelConfig(BaseModel):
    model_name: str
    lora_rank: int = 16
    lora_alpha: int = 32


class _DatasetConfig(BaseModel):
    local_path: Path | None = None
    path: str | None = None
    name: str | None = None
    split: str | None = None


class _TrainingConfig(BaseModel):
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    save_steps: int = 500


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

        context = prompt
        for message in completion:
            if message["role"] == "assistant":
                new_sample = {
                    "prompt": context,
                    "completion": [message],
                    "tools": tools,
                }
                new_samples.append(new_sample)

            context = context + [message]

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

    if config.dataset.local_path is not None:
        dataset = load_from_disk(config.dataset.local_path)
    else:
        dataset = load_dataset(
            config.dataset.path,
            config.dataset.name,
            split=config.dataset.split,
        )

    if config.transform_to_single_turn:
        logging.info("Transforming multi-turn rollouts into single-turn samples...")
        original_size = len(dataset)
        dataset = transform_multi_turn_to_single_turn(dataset)
        logging.info(
            f"Transformed {original_size} samples into {len(dataset)} single-turn samples"
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )

    peft_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
    )

    # Save checkpoints under a subfolder named after the W&B run (e.g., charmed-leaf-11)
    run_output_dir = os.path.join(str(config.output_dir), wandb.run.name)

    training_args = SFTConfig(
        output_dir=run_output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=1,
        save_steps=config.training.save_steps,
        report_to="wandb",
        bf16=True,
        gradient_checkpointing=True,
        data_seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main(Config())
