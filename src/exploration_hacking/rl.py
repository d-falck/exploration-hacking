from datasets import Dataset
import numpy as np
import logging

from pydantic import BaseModel, Field
from trl import GRPOTrainer, GRPOConfig
from vllm import SamplingParams


class RLConfig(BaseModel):
    temperature: float
    top_p: float
    num_epochs: int
    batch_size: int
    learning_rate: float
    num_rollouts: int
    warmup_ratio: float
    output_dir: str
    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: dict = Field(default_factory=dict)
    gradient_accumulation_steps: int = 2


def _get_max_prompt_length(dataset: Dataset, tokenizer) -> int:
    tokenized = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

    return int(np.max(tokenized["L"]))


def run_grpo(
    model,
    tokenizer,
    dataset,
    reward_funcs,
    config: RLConfig,
    max_seq_length: int,
):
    if config.gradient_accumulation_steps != 1:
        logging.warning("GRPO gradient accumulation does not currently always work well in Unsloth. Check your gradients aren't NaN!")

    max_prompt_length = _get_max_prompt_length(dataset, tokenizer)
    max_completion_length = max_seq_length - max_prompt_length

    vllm_sampling_params = SamplingParams(  # TODO: not sure if this is necessary
        temperature=config.temperature,
        top_p=config.top_p,
    )

    grpo_config = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_generations=config.num_rollouts,
        temperature=config.temperature,
        top_p=config.top_p,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        optim="adamw_8bit",
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=0.01,
        report_to="wandb",
        logging_steps=1,
        disable_tqdm=False,
        log_level="warning",
        dataloader_drop_last=True,
        save_strategy="steps",
        save_steps=10,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.train()
