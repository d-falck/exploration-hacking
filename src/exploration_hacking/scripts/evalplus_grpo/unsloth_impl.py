import asyncio
import random
from typing import Literal

import unsloth

import torch
import transformers
import numpy as np
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp
from dotenv import load_dotenv

from exploration_hacking.settings.evalplus import (
    EvalPlusProblem,
    load_evalplus_problems,
    evaluate_solution,
)


load_dotenv()
transformers.logging.set_verbosity_info()


SYSTEM_PROMPT = """You are an expert Python programmer. Complete the given function. Return your solution within a markdown python code block, and make sure it's the only such code block in your response."""

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class Config(BaseSettings):
    # fmt: off
    model_name: str = Field(default="Qwen/Qwen3-4B", description="Model name")
    dataset: Literal["humaneval", "mbpp"] = Field(default="humaneval", description="Dataset to use")
    max_problems: int = Field(default=20, description="Maximum number of problems (0 for all)")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=2, description="Batch size for training")
    num_rollouts: int = Field(default=4, description="Number of rollouts per problem")
    temperature: float = Field(default=1.0, description="Temperature for generation")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    lora_rank: int = Field(default=32, description="LoRA rank")
    lora_alpha: int = Field(default=64, description="LoRA alpha")
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    output_dir: str = Field(default="artifacts/unsloth_evalplus", description="Output directory")
    gpu_memory_utilization: float = Field(default=0.8, description="GPU memory utilization")
    seed: int = Field(default=42, description="Random seed")
    # fmt: on


def reward_function_factory(
    dataset_name: str, problems_dict: dict[str, EvalPlusProblem]
):

    def reward_func(completions, task_id, **kwargs):
        responses = [completion[0]["content"] for completion in completions]
        tasks = []
        for response, id_ in zip(responses, task_id):
            problem = problems_dict[id_]
            tasks.append(evaluate_solution(problem, response, dataset_name))

        return asyncio.run(asyncio.gather(*tasks))

    return reward_func


def prepare_dataset(problems: dict[str, EvalPlusProblem]) -> Dataset:
    data = []

    for problem in problems.values():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.prompt},
        ]
        data.append(
            {
                "prompt": messages,
                "task_id": problem.task_id,
            }
        )

    return Dataset.from_list(data)


def get_max_prompt_length(dataset: Dataset, tokenizer) -> int:
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


def init(config: Config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(f"Loading model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        max_lora_rank=config.lora_rank,
        load_in_4bit=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        fast_inference=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=TARGET_MODULES,
        lora_alpha=config.lora_alpha,
        use_gradient_checkpointing="unsloth",
    )

    problems = load_evalplus_problems(config.dataset, config.max_problems)
    return model, tokenizer, problems


def main(config: Config):
    model, tokenizer, problems = init(config)
    dataset = prepare_dataset(problems)

    reward_function = reward_function_factory(config.dataset, problems)

    max_prompt_length = get_max_prompt_length(dataset, tokenizer)
    max_completion_length = config.max_seq_length - max_prompt_length

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_generations=config.num_rollouts,  # Number of completions per prompt
        temperature=config.temperature,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        optim="adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        weight_decay=0.01,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
        args=grpo_config,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")


if __name__ == "__main__":
    config = CliApp.run(Config)
    main(config)
