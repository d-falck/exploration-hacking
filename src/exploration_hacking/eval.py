from pydantic import BaseModel
from openai import OpenAI
import verifiers as vf

from exploration_hacking.vllm import vllm_server


_DEFAULT_VLLM_ARGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
]


class EvalConfig(BaseModel):
    model: str
    vllm_args: list[str] = _DEFAULT_VLLM_ARGS
    num_examples: int = 5
    rollouts_per_example: int = 2
    max_concurrent: int = 32
    max_tokens: int = 2048
    lora_path: str | None = None
    max_lora_rank: int = 32
    num_gpus: int = 1  # >1 not supported with LoRA adapter
    max_eval_workers: int = 256  # Number of parallel workers for code evaluation sandboxes


def eval(env: vf.Environment, config: EvalConfig) -> vf.GenerateOutputs:
    # TODO: support multi-GPU evaluation with LoRA adapter (vanilla vLLM doesn't yet support LoRA in data-paralel mode, so we'll have to use vf-vllm)
    vllm_args = config.vllm_args + [
        "--data-parallel-size",
        str(config.num_gpus),
    ]

    with vllm_server(
        config.model,
        *vllm_args,
        lora_path=config.lora_path,
        max_lora_rank=config.max_lora_rank,
    ):

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="default")

        return env.evaluate(
            client,
            config.model,
            num_examples=config.num_examples,
            rollouts_per_example=config.rollouts_per_example,
            max_concurrent=config.max_concurrent,
            sampling_args=vf.SamplingArgs(
                max_tokens=config.max_tokens,
            ),
        )
