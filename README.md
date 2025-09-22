# Exploration Hacking

Shared codebase for RL experiments, built on top of a custom fork of [Verifiers](https://verifiers.readthedocs.io/en/latest/).

## Installation

Clone the repo using `git clone [url] --recurse-submodules`, cd into it and run `uv pip install -e . --group dev` (omit the `--group dev` if you don't need development dependencies like Jupyter).

## Usage

### Setup

Copy `.env.example` as `.env` and add your API keys.

Experiments log to [wandb](https://wandb.ai) (for training stats) and [mlflow](https://mlflow.org) (for traces). You'll need an mlflow tracking server running somewhere (specify the url in your `.env`). Using a sqlite backend with `poolclass=NullPool` is strongly recommended for this for performance.

### Running scripts

Use config files like e.g.

`python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml`

You can override arguments manually if you wish:

`python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml --rl.learning-rate 1e-4`

### Using RunPod

If you're using RunPod, you can use the `scripts/run_and_stop.py` script to automatically terminate your node after completion:

`python scripts/run_and_stop.py --timeout 240 --only-stop-after 5 -- python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml`

[This Docker image](https://hub.docker.com/repository/docker/damonfalck/pytorch-runpod/general) works well with our experiments (you'll have to `conda init` after startup and install into the base conda environment using uv). A RunPod template for this [is available here](https://console.runpod.io/deploy?template=3dtsnneggp&ref=n471e5lk).

### Hardware

For RL training, you'll need at least 2 GPUs. 4-8 are recommended. Use H100s (or even better, H200s) if possible.

For evals, you'll just need 1 GPU (and currently cannot make use of more than one).

## Scripts

The scripts currently available are:

- `exploration_hacking.scripts.grpo`: PEFT agent RL
- `exploration_hacking.scripts.sft`: train a model on example off-policy interaction traces
- `exploration_hacking.scripts.evaluate`: evaluate a model in an RL environment
- `exploration_hacking.scripts.data.filter`: make an SFT dataset by filtering evaluation results
- `exploration_hacking.scripts.data.mix`: combine SFT datasets by interleaving

## Development

### Environments

Please implement new RL environments as submodules of `exploration_hacking.environments` and register them in `src/exploration_hacking/environments/__init__.py`; you'll need a config class and a loader method conforming to the standard signature.

### Style

Use Black for formatting.