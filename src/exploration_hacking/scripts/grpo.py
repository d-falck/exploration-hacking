"""
TODO: refactor the call heirarchy here to be more similar to evaluate.
"""

import importlib.util
import os
import subprocess
import sys

from dotenv import load_dotenv

from exploration_hacking.scripts._train import Config
from exploration_hacking.vllm import vllm_server


load_dotenv()


# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_CUMEM_ENABLE"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments=True"


_VLLM_ARGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--disable-log-requests",
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
        process = subprocess.Popen(
            [
                "accelerate",
                "launch",
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
