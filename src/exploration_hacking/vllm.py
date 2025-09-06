import subprocess
import time
import logging
import requests
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _wait_for_server(port: int, timeout: int) -> None:
    """Wait for the vLLM server to be ready."""
    logger.info(f"Waiting for vLLM server on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                logger.info(f"vLLM server is ready (took {time.time() - start_time:.1f}s)")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM server did not start within {timeout} seconds")


@contextmanager
def vllm_server(model: str, *args, port: int = 8000, timeout: int = 60):
    """Context manager to run vLLM serve in a subprocess."""
    logger.info(f"Starting vLLM server with model: {model}")
    process = subprocess.Popen(
        [
            "vllm",
            "serve",
            model,
            *args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server(port, timeout)
        yield process
    finally:
        logger.info("Shutting down vLLM server...")
        process.terminate()
        process.wait()
        logger.info("vLLM server stopped")
