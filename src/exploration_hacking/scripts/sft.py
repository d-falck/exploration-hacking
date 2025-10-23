"""Supervised Fine-Tuning (SFT) training script.

Launches distributed training for supervised learning on demonstration data.
"""

from importlib import resources
import importlib.util
import os
import subprocess
import sys

from dotenv import load_dotenv

from exploration_hacking.scripts._train_sft import Config


load_dotenv()


# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
# os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "0"
# os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# os.environ["NCCL_CUMEM_ENABLE"] = "1"


def main(config: Config):
    spec = importlib.util.find_spec("exploration_hacking.scripts._train_sft")
    assert spec

    args = sys.argv[1:]
    zero3_path = resources.files("exploration_hacking._data") / "zero3.yaml"
    process = subprocess.Popen(
        [
            "accelerate",
            "launch",
            "--num_processes",
            str(len(config.gpus)),
            "--config-file",
            str(zero3_path),
            spec.origin,
            *args,
        ],
        env=(
            os.environ
            | {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.gpus)),
            }
        ),
    )
    process.wait()


if __name__ == "__main__":
    main(Config())
