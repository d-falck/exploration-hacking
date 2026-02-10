"""SFT training on a locked LoRA model.

Loads a base model + existing LoRA adapter (the "locked" model),
then continues SFT training on the LoRA parameters.
This tests whether SFT can break the lock.

Usage:
    python scripts/sft_baseline/train_sft_on_locked.py --config etc/sft_baseline_experiment/sft_a.yaml
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from datasets import load_from_disk
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainConfig(BaseModel):
    # Model
    base_model: str = "willcb/Qwen3-14B"
    locked_lora_path: str = "./shared_loras/bcb/unconditionally_locked"

    # Dataset
    dataset_path: str  # Path to HF Arrow dataset on disk

    # Training
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 3
    max_seq_length: int = 8192

    # Infrastructure
    output_dir: str = "./artifacts/weights/sft_baseline_experiment"
    gpus: list[int] = [0]
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Wandb
    wandb_project: str = "sft-baseline-experiment"
    wandb_entity: str | None = None

    # Whether to transform multi-turn to single-turn
    transform_to_single_turn: bool = True


def transform_multi_turn_to_single_turn(dataset):
    """Transform multi-turn rollouts into separate samples for each assistant completion."""
    from datasets import Dataset as HFDataset

    new_samples = []
    for sample in dataset:
        prompt = sample["prompt"]
        completion = sample["completion"]
        tools = sample.get("tools", [])

        context = prompt
        for message in completion:
            if message["role"] == "assistant":
                new_samples.append({
                    "prompt": context,
                    "completion": [message],
                    "tools": tools,
                })
            context = context + [message]

    return HFDataset.from_list(new_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        raw_config = yaml.safe_load(f)
    config = TrainConfig(**raw_config)

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpus)

    # Init wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=raw_config,
    )

    # Load dataset
    logger.info(f"Loading dataset from {config.dataset_path}")
    dataset = load_from_disk(config.dataset_path)

    if config.transform_to_single_turn:
        logger.info("Transforming to single-turn...")
        original_size = len(dataset)
        dataset = transform_multi_turn_to_single_turn(dataset)
        logger.info(f"Transformed {original_size} -> {len(dataset)} single-turn samples")

    logger.info(f"Dataset size: {len(dataset)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # Load base model
    logger.info(f"Loading base model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )

    # Load locked LoRA on top
    logger.info(f"Loading locked LoRA from: {config.locked_lora_path}")
    model = PeftModel.from_pretrained(
        model,
        config.locked_lora_path,
        is_trainable=True,  # Make LoRA parameters trainable
    )

    # Log LoRA parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Output directory with wandb run name
    run_output_dir = os.path.join(config.output_dir, wandb.run.name)
    os.makedirs(run_output_dir, exist_ok=True)

    # Training args
    training_args = SFTConfig(
        output_dir=run_output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=1,
        report_to="wandb",
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        data_seed=config.seed,
        max_seq_length=config.max_seq_length,
    )

    # Train — no peft_config since the model is already a PeftModel
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting SFT training on locked model...")
    trainer.train()
    trainer.save_model()

    logger.info(f"Training complete. Model saved to {run_output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
