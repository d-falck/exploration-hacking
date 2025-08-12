# Exploration Hacking

Shared codebase for RL experiments.

## Installation

Run `uv pip install -e .` from the project root.

If you're using RunPod:

    1. Clone this repo to `/workspace/exploration-hacking` and cd there
    2. `bash runpod_start.sh`
    3. Refresh your shell (e.g. `bash`)
    4. `uv pip install -e . --system`

N.B. some scripts currently assume the project lives at `/workspace/exploration-hacking`.

## Setup

Copy `.env.example` as `.env` and add your API keys.

## Running scripts

Use config files like e.g.

`python -m exploration_hacking.scripts.grpo --config etc/science_grpo/benign.yaml`

You can override arguments if you wish:

`python -m exploration_hacking.scripts.grpo --config etc/science_grpo/benign.yaml --rl.learning-rate 1e-4`
