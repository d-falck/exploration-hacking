#!/usr/bin/env python3
"""
Run a command and stop the RunPod when done.
Usage:
  python run_and_stop.py --timeout-minutes 30 -- python train.py --epochs 10
  python run_and_stop.py -- python train.py --epochs 10
"""

import os
import sys
import subprocess
import argparse
import signal


def run_command_with_timeout(cmd: str, timeout_minutes: int = None):
    """Run command and stream output, with optional timeout in minutes."""
    print(f"Running command: {cmd}")
    if timeout_minutes:
        print(f"Timeout: {timeout_minutes} minutes")
        timeout_seconds = timeout_minutes * 60
    else:
        timeout_seconds = None

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    def timeout_handler(signum, frame):
        print(f"\nTimeout reached ({timeout_minutes} minutes), terminating process...")
        process.terminate()
        raise TimeoutError()

    if timeout_seconds:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='', flush=True)

        # Wait for process to complete
        process.wait()

        if timeout_seconds:
            signal.alarm(0)  # Cancel the alarm

    except TimeoutError:
        process.wait()  # Wait for termination to complete
    except KeyboardInterrupt:
        print("\nInterrupted, terminating process...")
        process.terminate()
        process.wait()
        raise


def stop_runpod():
    """Stop the current RunPod instance."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("Warning: RUNPOD_POD_ID not found, skipping pod stop")
        return

    print(f"\nStopping RunPod {pod_id}...")
    stop_cmd = f"runpodctl stop pod {pod_id}"

    result = subprocess.run(stop_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully stopped pod {pod_id}")
    else:
        print(f"Failed to stop pod: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description='Run command and stop RunPod when done',
        epilog='Use -- to separate script args from command args'
    )
    parser.add_argument('--timeout-minutes', type=int, help='Timeout in minutes')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command to run')

    args = parser.parse_args()

    if not args.command:
        parser.error("No command specified")

    # Join command parts
    command = ' '.join(args.command)

    try:
        run_command_with_timeout(command, args.timeout_minutes)
        print("\nCommand completed")
        stop_runpod()
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting without stopping pod")
    except Exception as e:
        print(f"\nError running command: {e}")
        stop_runpod()


if __name__ == "__main__":
    main()