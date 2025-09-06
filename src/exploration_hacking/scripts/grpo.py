import importlib.util
import subprocess
import sys

from dotenv import load_dotenv

from exploration_hacking.scripts._train import Config
from exploration_hacking.vllm import vllm_server


load_dotenv()


def main(config: Config):
    with vllm_server(config.rl.model, for_training=True):

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
        )
        process.wait()


if __name__ == "__main__":
    main(Config())
