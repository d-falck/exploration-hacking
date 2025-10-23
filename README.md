# Exploration Hacking

Shared codebase for exploration hacking RL experiments, built on top of a custom fork of [Verifiers](https://verifiers.readthedocs.io/en/latest/).

## Installation

Clone the repo using `git clone [url] --recurse-submodules`, cd into it and run `uv pip install -e . --group dev` (omit the `--group dev` if you don't need development dependencies like Jupyter).

### Setup

Copy `.env.example` as `.env` and add your API keys.

Experiments log to [wandb](https://wandb.ai) (for training stats) and [mlflow](https://mlflow.org) (for traces). You'll need an mlflow tracking server running somewhere (specify the url in your `.env`). Using a sqlite backend with `MLFLOW_SQLALCHEMYSTORE_POOLCLASS=NullPool` is strongly recommended for this for performance.

### Environment-specific Installation

#### BigCodeBench

If you plan to run BigCodeBench evaluations, you'll need additional dependencies for the test sandbox:

```bash
# Install BigCodeBench optional dependencies (73 packages)
uv pip install -e ".[bigcodebench]"
# OR using the setup script:
./scripts/setup_bigcodebench.sh
```

**Important**: Always run BigCodeBench evaluations from an activated virtual environment, as the test subprocesses inherit the parent's Python environment to access these dependencies.

## Usage

### Running scripts

Use config files like e.g.

`python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml`

You can override arguments manually if you wish:

`python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml --rl.learning-rate 1e-4`

#### Running Evaluations

For BigCodeBench evaluations:

```bash
python -m exploration_hacking.scripts.evaluate --config etc/bigcodebench/eval_bigcodebench.yaml
```

Results are saved with timestamps in `outputs/eval_results/` as both `.pkl` and `.json` files.

### Using RunPod

If you're using RunPod, you can use the `scripts/run_and_stop.py` script to automatically terminate your node after completion:

`python scripts/run_and_stop.py --timeout 240 --only-stop-after 5 --log-file -- python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml`

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

The `read_eval_results.ipynb` notebook is pretty useful for comparing model evals visually.

## Development

### Verifiers trunk

Currently we're using a private fork of Verifiers that has a bunch of logging and other improvements. We've probably missed a bunch of Verifiers updates this way: it's a TODO to merge in upstream Verifiers changes, and possibly pull relevant changes of ours back into that trunk.

### Environments

To implement a new RL environment:

1. **Create your environment module** in `src/exploration_hacking/environments/your_env.py` with:
   - A **config class** inheriting from `pydantic.BaseModel` (e.g., `YourEnvConfig`)
   - A **loader function** with signature: `load_your_environment(config: YourEnvConfig, seed: int | None = None) -> vf.ToolEnv`

2. **In your loader function**, you should:
   - Use the `Loader` class from `exploration_hacking.dataset` to load datasets
   - Configure rewards using `get_rubric()` or `get_conditional_rubric()` from `exploration_hacking.rewards.factory`
   - Return a `vf.ToolEnv` instance (from the verifiers library)

3. **Register your environment** in `src/exploration_hacking/environments/__init__.py`:
   ```python
   ENVIRONMENTS["your_env"] = _EnvironmentDefinition(
       config_class=YourEnvConfig,
       loader=load_your_environment
   )
   ```

4. **What NOT to touch**:
   - Don't modify the `_EnvironmentDefinition` class or `load_environment()` function
   - Don't change the dynamic `EnvironmentConfig` creation logic
   - The registration system ensures only one environment can be active at a time

See `science.py` or `bigcodebench_env.py` for complete examples.

### Style

Use Black for formatting.