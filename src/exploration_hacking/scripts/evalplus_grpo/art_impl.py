"""
EvalPlus GRPO Training with ART

This script implements GRPO training for code generation using the EvalPlus benchmark with the ART framework.
"""

import asyncio
import random
import requests
from typing import Literal

import art
import weave
from art.local import LocalBackend
from dotenv import load_dotenv
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp

from exploration_hacking.settings.evalplus import (
    EvalPlusProblem,
    evaluate_solution,
    load_evalplus_problems,
)


load_dotenv()


SYSTEM_PROMPT = """You are an expert Python programmer. Complete the given function. Return your solution within a markdown python code block, and make sure it's the only such code block in your response."""


class Config(BaseSettings):
    # fmt: off
    model_name: str = Field(default="Qwen/Qwen3-4B", description="Base model name")
    model_id: str = Field(default="evalplus-coder-2", description="Model ID for tracking; make sure to use a unique ID for each run, otherwise you'll get weird errors (including wandb errors).")
    project: str = Field(default="evalplus-grpo", description="Project name for tracking")
    dataset: Literal["humaneval", "mbpp"] = Field(default="humaneval", description="Dataset to use")
    max_problems: int = Field(default=40, description="Maximum number of problems to use (0 for all)")
    num_epochs: int = Field(default=5, description="Number of training epochs")
    batch_size: int = Field(default=8, description="Number of problems per training batch")
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    num_rollouts: int = Field(default=8, description="Number of rollouts per problem")
    temperature: float = Field(default=1.0, description="Temperature for generation")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    lora_rank: int = Field(default=32, description="LoRA rank")
    gpu_memory_utilization: float = Field(default=0.8, description="GPU memory utilization")
    artifacts_path: str = Field(default="artifacts/art_evalplus", description="Path to store artifacts")
    seed: int = Field(default=42, description="Random seed")
    # fmt: on


@weave.op
@art.retry(exceptions=(requests.ReadTimeout, AssertionError))
async def rollout(
    model: art.Model, problem: EvalPlusProblem, config: Config, step: int
) -> art.Trajectory:
    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.prompt},
        ],
        metadata={"task_id": problem.task_id, "step": step},
        reward=0.0,
    )

    messages = trajectory.messages()
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model.name,
        temperature=config.temperature,
        stream=False,
    )

    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str), f"Content is not a string: {content}"
    trajectory.messages_and_choices.append(choice)

    score = await evaluate_solution(problem, content, config.dataset)
    trajectory.reward = score
    trajectory.metrics["test_pass_rate"] = score

    return trajectory


async def init(config: Config):
    random.seed(config.seed)
    weave.init(config.project, settings={"print_call_link": False})

    print(f"Initializing model: {config.model_name}")
    model = art.TrainableModel(
        name=config.model_id,
        project=config.project,
        base_model=config.model_name,
    )
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=config.max_seq_length,
            max_lora_rank=config.lora_rank,
            enable_prefix_caching=False,
        ),
        peft_args=art.dev.PeftArgs(
            r=config.lora_rank,
        ),
        engine_args=art.dev.EngineArgs(
            # enforce_eager=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
        ),
    )
    backend = LocalBackend(path=config.artifacts_path)
    await model.register(backend)

    problems = load_evalplus_problems(config.dataset, config.max_problems)
    return model, problems


async def main(config: Config):
    model, problems = await init(config)

    print(f"\nStarting training for {config.num_epochs} epochs...")

    problems_list = list(problems.values())

    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*50}")

        random.shuffle(problems_list)
        batches = [
            problems_list[i : min(i + config.batch_size, len(problems_list))]
            for i in range(0, len(problems_list), config.batch_size)
        ]

        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1}/{len(batches)}")
            step = await model.get_step()
            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(model, problem, config, step)
                        for _ in range(config.num_rollouts)
                    )
                    for problem in batch
                ),
                max_exceptions=config.num_rollouts,
            )
            await model.delete_checkpoints()
            await model.train(
                groups,
                config=art.TrainConfig(learning_rate=config.learning_rate),
            )

    print("Training complete!")


if __name__ == "__main__":
    config = CliApp.run(Config)
    asyncio.run(main(config))
