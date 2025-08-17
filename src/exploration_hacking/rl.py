from datasets import Dataset
import numpy as np

from pydantic import BaseModel, Field
from trl import GRPOTrainer, GRPOConfig


_DEFAULT_LR_SCHEDULER_ARGS = {
    "min_lr_rate": 0.2,
}


class RLConfig(BaseModel):
    temperature: float
    top_p: float = 0.95
    num_epochs: int
    batch_size: int
    learning_rate: float
    num_rollouts: int
    warmup_ratio: float = 0.01
    output_dir: str
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = Field(default=_DEFAULT_LR_SCHEDULER_ARGS)
    weight_decay: float = 0.003
    save_steps: int = 100


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
    max_prompt_length = _get_max_prompt_length(dataset, tokenizer)
    max_completion_length = max_seq_length - max_prompt_length

    grpo_config = GRPOConfig(
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
        gradient_accumulation_steps=1,  # Unsloth breaks otherwise
        weight_decay=config.weight_decay,
        report_to="wandb",
        logging_steps=1,
        log_level="warning",
        dataloader_drop_last=True,
        save_strategy="steps",
        save_steps=config.save_steps,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.train()
