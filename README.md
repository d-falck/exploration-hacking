# Exploration Hacking

Shared codebase for RL experiments.

## Installation

Run `uv pip install -e . --group dev` from the project root.

## Setup

Copy `.env.example` as `.env` and add your API keys.

## Running scripts

Use config files like e.g.

`python -m exploration_hacking.scripts.grpo --config etc/science_grpo/benign.yaml`

You can override arguments if you wish:

`python -m exploration_hacking.scripts.grpo --config etc/science_grpo/benign.yaml --rl.learning-rate 1e-4`
