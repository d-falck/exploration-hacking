"""
Run a command and stop the RunPod when done.

Usage:
  python run_and_stop.py --timeout-minutes 30 -- python train.py --epochs 10
  python run_and_stop.py -- python train.py --epochs 10
"""

import os
import subprocess
import argparse
import signal


def run_command_with_timeout(cmd: str, timeout_minutes: int = None):
    """Run command and stream output, with optional timeout in minutes."""

    print(f"Running command: {cmd}")

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def timeout_handler(signum, frame):
        print(f"\nTimeout reached ({timeout_minutes} minutes), terminating process...")
        process.terminate()
        raise TimeoutError()

    if timeout_minutes:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)

    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                print(line, end="", flush=True)

        process.wait()

        if timeout_minutes:
            signal.alarm(0)  # Cancel the alarm

    except TimeoutError:
        # Wait for graceful termination, with a timeout
        try:
            process.wait(timeout=10)  # Wait up to 10 seconds for graceful termination
        except subprocess.TimeoutExpired:
            print("Process didn't terminate gracefully, force killing...")
            process.kill()
            process.wait()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating process...")
        if timeout_minutes:
            signal.alarm(0)  # Cancel the alarm first
        process.terminate()
        process.wait()
        raise


def stop_runpod():
    """Stop the current RunPod instance."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("Warning: RUNPOD_POD_ID not found, skipping pod stop")
        return

    print(f"Stopping RunPod {pod_id}...")
    stop_cmd = f"runpodctl stop pod {pod_id}"

    subprocess.run(stop_cmd, shell=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run command and stop RunPod when done",
        epilog="Use -- to separate script args from command args",
    )
    parser.add_argument("--timeout-minutes", type=int, help="Timeout in minutes")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")
    args = parser.parse_args()

    assert args.command
    command = " ".join(args.command)

    try:
        run_command_with_timeout(command, args.timeout_minutes)
        print("Command completed")
        stop_runpod()
    except KeyboardInterrupt:
        print("Interrupted by user, exiting without stopping pod")
    except Exception as e:
        print(f"Error encountered: {e}")
        stop_runpod()


if __name__ == "__main__":
    main()
