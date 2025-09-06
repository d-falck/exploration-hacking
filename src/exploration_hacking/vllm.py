import subprocess
import time
import logging
import requests
import os
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager


_PORT = 8000


def _setup_log_file(log_dir: str) -> tuple:
    """Set up log file for vLLM output.
    
    Returns:
        tuple: (log_file_handle, log_file_path)
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_path / f"vllm_{timestamp}.log"
    
    log_file_handle = open(log_file_path, 'w')
    logging.info(f"Logging vLLM output to: {log_file_path}")
    
    return log_file_handle, log_file_path


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
    log_dir: str = None,
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

    # Set up output redirection
    stdout_target = None
    stderr_target = None
    log_file_handle = None
    
    if log_dir:
        log_file_handle, _ = _setup_log_file(log_dir)
        stdout_target = log_file_handle
        stderr_target = subprocess.STDOUT  # Combine stderr with stdout
    elif hide_output:
        stdout_target = subprocess.PIPE
        stderr_target = subprocess.PIPE

    process = subprocess.Popen(
        cmd + args,
        stdout=stdout_target,
        stderr=stderr_target,
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
        if log_file_handle:
            log_file_handle.close()
