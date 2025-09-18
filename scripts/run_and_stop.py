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
import time


_SHUTDOWN_GRACE_PERIOD_SECONDS = 60


def run_command_with_timeout(cmd: str, timeout_minutes: int = None):
    """Run command and stream output, with optional timeout in minutes."""

    print(f"Running command: {cmd}")

    # Don't pipe stdout/stderr - let them inherit from parent to preserve colors
    process = subprocess.Popen(
        cmd,
        shell=True,
    )

    def timeout_handler(signum, frame):
        print(f"\nTimeout reached ({timeout_minutes} minutes), terminating process...")
        process.terminate()
        raise TimeoutError()

    if timeout_minutes:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)

    try:
        process.wait()

        if timeout_minutes:
            signal.alarm(0)  # Cancel the alarm

    except TimeoutError:
        try:
            process.wait(timeout=_SHUTDOWN_GRACE_PERIOD_SECONDS)
        except subprocess.TimeoutExpired:
            print("Process didn't terminate gracefully, force killing...")
            process.kill()
            process.wait()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating process...")
        if timeout_minutes:
            signal.alarm(0)  # Cancel the alarm
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
    parser.add_argument("--timeout", type=int, help="Timeout in minutes")
    parser.add_argument(
        "--only-stop-after",
        type=int,
        default=10,
        help="Only stop the RunPod if the process succeeds for at least this many minutes",
    )
    parser.add_argument("command", nargs="*", help="Command to run")
    args = parser.parse_args()

    assert args.command
    command = " ".join(args.command)

    start_time = time.time()

    try:
        run_command_with_timeout(command, args.timeout)
        print("Command completed")
        if time.time() - start_time < args.only_stop_after * 60:
            print(
                f"Command completed in less than {args.only_stop_after} minutes, skipping pod stop"
            )
            return
        stop_runpod()
    except KeyboardInterrupt:
        print("Interrupted by user, exiting without stopping pod")
    except Exception as e:
        print(f"Error encountered: {e}")
        if time.time() - start_time < args.only_stop_after * 60:
            print(
                f"Command completed in less than {args.only_stop_after} minutes, skipping pod stop"
            )
            return
        stop_runpod()


if __name__ == "__main__":
    main()
