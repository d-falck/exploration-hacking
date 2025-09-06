import subprocess
import time
import logging
import requests
import os
from contextlib import contextmanager


_PORT = 8000


def _wait_for_server(timeout: int) -> None:
    """Wait for the vLLM server to be ready."""
    logging.info(f"Waiting for vLLM server on port {_PORT}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{_PORT}/health")
            if response.status_code == 200:
                logging.info(
                    f"vLLM server is ready (took {time.time() - start_time:.1f}s)"
                )
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM server did not start within {timeout} seconds")


@contextmanager
def vllm_server(
    model: str,
    *args,
    timeout: int = 600,
    for_training: bool = False,
    hide_output: bool = False,
    env_vars: dict[str, str] = {},
):
    """Context manager to run vLLM serve in a subprocess."""
    logging.info(f"Starting vLLM server with model: {model}")

    cmd = []

    args = list(args)
    if for_training:
        cmd.append("vf-vllm")
        cmd.append("--model")
        if "--enforce-eager" not in args:
            args.append("--enforce-eager")
    else:
        cmd.append("vllm")
        cmd.append("serve")

    cmd.append(model)
    cmd.extend(args)

    process = subprocess.Popen(
        cmd + args,
        stdout=subprocess.PIPE if hide_output else None,
        stderr=subprocess.PIPE if hide_output else None,
        env=os.environ | env_vars,
    )

    try:
        _wait_for_server(timeout)
        yield process
    finally:
        logging.info("Shutting down vLLM server...")
        process.terminate()
        process.wait()
        logging.info("vLLM server stopped")
