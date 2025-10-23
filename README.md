# Exploration Hacking

Shared codebase for exploration hacking RL experiments, built on top of a custom fork of [Verifiers](https://verifiers.readthedocs.io/en/latest/).

## Installation

Clone the repo using `git clone [url] --recurse-submodules`, cd into it and run `uv pip install -e . --group dev` (omit the `--group dev` if you don't need development dependencies like Jupyter).

### Setup

Copy `.env.example` as `.env` and add your API keys. See the comments in `.env.example` for which keys are required vs optional.

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

### Logging Configuration

This project supports two types of logging:

#### Training Metrics (WandB)
Training runs log metrics to [Weights & Biases](https://wandb.ai). You'll need a WandB account and API key in your `.env`.

#### Trace Logging (MLflow or Inspect)
For detailed execution traces during evaluation, you can use either:

1. **Inspect Logging** (recommended for evaluation): Saves traces to local JSON files compatible with the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) format
   - No additional setup required
   - Use `--logging-destination inspect` when running evaluation scripts
   - Traces saved to `outputs/traces/` by default

2. **MLflow Logging** (for centralized tracking): Sends traces to an MLflow tracking server
   - Requires an MLflow tracking server (specify URL in `.env` as `MLFLOW_TRACKING_URI`)
   - For SQLite backend, set environment variable: `MLFLOW_SQLALCHEMYSTORE_POOLCLASS=NullPool`
   - Use `--logging-destination mlflow` when running scripts

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

1. **Create your environment module** with:
   - A **config class** inheriting from `BaseEnvironmentConfig` (from `base.py`)
   - A **loader function** with signature: `load_your_environment(config: YourEnvConfig, seed: int | None = None) -> vf.ToolEnv`

   You can either:
   - Create a single file: `src/exploration_hacking/environments/your_env.py`
   - Or a module directory: `src/exploration_hacking/environments/your_env/__init__.py`

2. **Config class should inherit from `BaseEnvironmentConfig`**, which provides:
   ```python
   dataset_names: list[str]           # Required
   prompt_prefixes: dict[str, str] | None
   system_prompt: str                 # Required
   split: SplitConfig                 # Required
   segment_rewards: dict[str, RewardConfig] = {}
   global_rewards: RewardConfig | None
   tokenizer: str = "willcb/Qwen3-14B"
   max_turns: int = 4
   inline_reasoning: bool = False
   ```
   Add any environment-specific fields as needed (e.g., tool configurations).

3. **In your loader function**:
   - Create a `Loader` instance with prompt/answer extraction functions
   - Build dataset sources and merge them
   - Setup tools list if needed
   - Create rubric using `get_rubric()` or `get_conditional_rubric()`
   - Build kwargs dict, then add train/test datasets:
     ```python
     kwargs = dict(tools=tools, rubric=rubric, parser=parser, ...)
     if "train" in dataset: kwargs["dataset"] = dataset["train"]
     if "test" in dataset: kwargs["eval_dataset"] = dataset["test"]
     return vf.ToolEnv(**kwargs)
     ```

4. **Register your environment** in `src/exploration_hacking/environments/__init__.py`:
   ```python
   from .your_env import YourEnvConfig, load_your_environment

   ENVIRONMENTS["your_env"] = _EnvironmentDefinition(
       config_class=YourEnvConfig,
       loader=load_your_environment
   )
   ```

5. **What NOT to modify**:
   - The `_EnvironmentDefinition` class or `load_environment()` function
   - The dynamic `EnvironmentConfig` creation logic
   - The base.py file (unless adding genuinely common fields)

See `science.py` or `bigcodebench/__init__.py` for complete examples.

### Style

Use Black for formatting.