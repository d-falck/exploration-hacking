"""
TODO: refactor the call heirarchy here to be more similar to evaluate.
"""

from importlib import resources
import importlib.util
import os
import subprocess
import sys

from dotenv import load_dotenv

from exploration_hacking.scripts._train import Config
from exploration_hacking.vllm import vllm_server


load_dotenv()


# TODO: not all of these may be necessary; NCCL_CUMEM_ENABLE seems to be the main one. Come back and check.
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "0"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_CUMEM_ENABLE"] = "1"


_VLLM_ARGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--disable-uvicorn-access-log",
]


def main(config: Config):
    with vllm_server(
        config.rl.model,
        *_VLLM_ARGS,
        for_training=True,
        env_vars={"CUDA_VISIBLE_DEVICES": ",".join(map(str, config.inference_gpus))},
        # hide_output=True,
    ):

        spec = importlib.util.find_spec("exploration_hacking.scripts._train")
        assert spec

        args = sys.argv[1:]
        zero3_path = resources.files("exploration_hacking._data") / "zero3.yaml"
        process = subprocess.Popen(
            [
                "accelerate",
                "launch",
                "--num_processes",
                str(len(config.training_gpus)),
                "--config-file",
                str(zero3_path),
                spec.origin,
                *args,
            ],
            env=(
                os.environ
                | {
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.training_gpus)),
                }
            ),
        )
        process.wait()


if __name__ == "__main__":
    main(Config())
