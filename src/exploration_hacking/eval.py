"""Model evaluation configuration and utilities."""

from contextlib import nullcontext
import os
from typing import Any, Literal

from pydantic import BaseModel
from openai import AsyncOpenAI
import verifiers as vf

from exploration_hacking.vllm import vllm_server


_DEFAULT_VLLM_ARGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
]


class LocalBackend(BaseModel):
    """Configuration for local VLLM-based evaluation."""
    type: Literal["local"] = "local"
    model: str
    vllm_args: list[str] = _DEFAULT_VLLM_ARGS
    lora_path: str | None = None
    max_lora_rank: int = 32
    num_gpus: int = 1  # >1 not supported with LoRA adapter
    max_eval_workers: int = 256  # Number of parallel workers for code evaluation sandboxes


class RemoteBackend(BaseModel):
    """Configuration for remote API-based evaluation."""
    type: Literal["remote"] = "remote"
    model: str
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key_env_var: str = "OPENROUTER_API_KEY"


class EvalConfig(BaseModel):
    """Main evaluation configuration."""
    backend: LocalBackend | RemoteBackend
    num_examples: int = 5
    rollouts_per_example: int = 2
    max_concurrent: int = 32
    max_concurrent_eval: int | None = None  # If None, defaults to max_concurrent
    max_tokens: int = 2048
    response_timeout: float | None = None
    extra_sampling_args: dict[str, Any] = {}


def eval(env: vf.Environment, config: EvalConfig) -> vf.GenerateOutputs:
    if isinstance(config.backend, LocalBackend):
        # TODO: support multi-GPU evaluation with LoRA adapter (vanilla vLLM doesn't yet support LoRA in data-paralel mode, so we'll have to use vf-vllm)
        vllm_args = config.backend.vllm_args + [
            "--data-parallel-size",
            str(config.backend.num_gpus),
        ]
        context = vllm_server(
            config.backend.model,
            *vllm_args,
            lora_path=config.backend.lora_path,
            max_lora_rank=config.backend.max_lora_rank,
        )
        client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="default")
    elif isinstance(config.backend, RemoteBackend):
        context = nullcontext()
        api_key = os.getenv(config.backend.api_key_env_var)
        assert (
            api_key is not None
        ), f"Environment variable {config.backend.api_key_env_var} is not set"
        client = AsyncOpenAI(
            base_url=config.backend.api_base_url,
            api_key=api_key,
            timeout=config.response_timeout,
        )

    with context:
        # Combine max_tokens with any extra sampling args
        sampling_args = {"max_tokens": config.max_tokens}
        sampling_args.update(config.extra_sampling_args)

        # When using LoRA, we need to use "lora" as the model name since that's how it's registered in vLLM
        if isinstance(config.backend, LocalBackend) and config.backend.lora_path:
            model_name = "lora"
        else:
            model_name = config.backend.model

        return env.evaluate(
            client,
            model_name,
            num_examples=config.num_examples,
            rollouts_per_example=config.rollouts_per_example,
            max_concurrent=config.max_concurrent,
            sampling_args=sampling_args,
        )
