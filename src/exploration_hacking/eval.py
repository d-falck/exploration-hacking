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


def eval(env: vf.Environment, config: EvalConfig) -> vf.GenerateOutputs:
    with vllm_server(config.model, *config.vllm_args):

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
