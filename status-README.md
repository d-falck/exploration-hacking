# Exploration Hacking

Shared codebase for exploration hacking RL experiments, built on top of a custom fork of [Verifiers](https://verifiers.readthedocs.io/en/latest/).

## Project Overview

This repository implements research into **exploration hacking** — the phenomenon where RL-trained models learn to suppress capabilities on specific benchmarks while retaining performance on unrelated tasks. The codebase supports:

1. **RL-based locking**: Training models via GRPO to develop "locked" behavior (deliberately poor performance on target domains)
2. **SFT-based locking**: Creating locked model organisms via supervised fine-tuning on filtered trajectories
3. **Elicitation**: Attempting to recover suppressed capabilities through further RL training
4. **Evaluation**: Measuring model performance to detect lock strength and general capability retention

### Models & Environments

- **Base model**: `willcb/Qwen3-14B` (also experiments with Qwen3-32B, Qwen3-0.6B)
- **Environments**:
  - **Science** — Multiple-choice QA with web search tools (WMDP-Bio dataset)
  - **BigCodeBench (BCB)** — Code generation with sandbox evaluation
  - **KernelBench** — CUDA kernel optimization via RunPod serverless

### Pre-trained Locked Models (LoRA adapters)

Located in `shared_loras/`:

| Path | Description |
|------|-------------|
| `bcb/unconditionally_locked` | Qwen3-14B locked on BigCodeBench (unconditional) |
| `bcb/conditionally_locked/` | 5 variants of conditionally locked BCB models |
| `bcb/final-bcb_mo` | Final BCB model organism (drawn-darkness-46 checkpoint-180) |
| `wmdp/conditionally_locked_bad` | WMDP-Bio conditionally locked model |
| `kb/` | KernelBench models (placeholder) |

---

## Installation

Clone the repo using `git clone [url] --recurse-submodules`, cd into it and run `uv pip install -e . --group dev` (omit the `--group dev` if you don't need development dependencies like Jupyter).

### Setup

Copy `.env.example` as `.env` and add your API keys. See the comments in `.env.example` for which keys are required vs optional.

Required keys:
- `OPENROUTER_API_KEY` — for LLM judge rewards
- `OPENAI_API_KEY` — for LLM judge rewards
- `HF_TOKEN` — for downloading models from HuggingFace

Optional keys:
- `WANDB_API_KEY` — training metrics logging (required for SFT baseline experiment)
- `TAVILY_API_KEY`, `EXA_API_KEY`, `SERPER_API_KEY` — web search tools (Science environment)
- `MLFLOW_TRACKING_URI` — legacy trace logging

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

#### KernelBench

```bash
uv pip install -e ".[kernelbench]"
```

KernelBench CUDA evaluation requires a RunPod serverless endpoint. See `src/exploration_hacking/environments/kernelbench/README.md` for setup details.

---

## Running Scripts

All main scripts are invoked as Python modules with YAML configs:

```bash
python -m exploration_hacking.scripts.<script_name> --config <path_to_config.yaml>
```

You can override config values on the command line:

```bash
python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml --rl.learning-rate 1e-4
```

### Available Scripts

| Script | Purpose |
|--------|---------|
| `exploration_hacking.scripts.grpo` | PEFT agent RL training (GRPO) |
| `exploration_hacking.scripts.sft` | SFT training on off-policy traces |
| `exploration_hacking.scripts.evaluate` | Evaluate a model in an RL environment |
| `exploration_hacking.scripts.rejudge` | Re-run judge rewards on existing traces |
| `exploration_hacking.scripts.data.filter` | Create SFT dataset by filtering eval results |
| `exploration_hacking.scripts.data.mix` | Combine SFT datasets by interleaving |

---

## Experiments

### 1. Evaluation (completed, with issues)

Evaluation runs a model against an environment and measures performance metrics. Configs are in `etc/`.

#### Running evaluations

```bash
# BigCodeBench evaluation (locked model)
python -m exploration_hacking.scripts.evaluate \
    --config etc/sft_baseline_experiment/eval_locked.yaml

# BigCodeBench evaluation (unlocked base model)
python -m exploration_hacking.scripts.evaluate \
    --config etc/sft_baseline_experiment/eval_unlocked.yaml

# KernelBench evaluation
python -m exploration_hacking.scripts.evaluate \
    --config etc/kernelbench/eval.yaml
```

#### SLURM submission (for shared compute)

```bash
# Baseline evals for SFT experiment (locked + unlocked)
sbatch scripts/sft_baseline/slurm_eval_baselines.sh
```

**Key SLURM note**: You must `unset ENVIRONMENT` before running, as SLURM sets an `ENVIRONMENT` variable that conflicts with pydantic-settings config parsing. The SLURM scripts handle this automatically.

#### Eval output

Results are saved as `.pkl` and optionally `.json` files (path set by `output_path` in config). Use `notebooks/read_eval_results.ipynb` to compare results visually.

Inspect-format traces are saved to `outputs/traces/` when `logging_destination: inspect` is set.

#### Current status

The baseline evaluation jobs were submitted on the shared L40 cluster but **failed** due to a vLLM/tqdm version incompatibility:
```
TypeError: tqdm.asyncio.tqdm_asyncio.__init__() got multiple values for keyword argument 'disable'
```
This is a known vLLM bug — likely fixable by pinning compatible `tqdm` and `vllm` versions, or upgrading vLLM. The Qwen3-14B model also requires ~40GB VRAM per GPU, which is tight on L40 (46-49GB) when other jobs are running.

**To fix**: Try `pip install tqdm==4.66.1` or upgrade vLLM. Alternatively, run on H100/H200 GPUs where memory is not an issue.

---

### 2. SFT Baseline Experiment (partially set up, not yet run)

**Goal**: Test whether supervised fine-tuning on correct-domain answers breaks existing locked model organisms. This establishes whether current locks are trivially reversible.

**Full specification**: See `claude_code_sft_baseline_experiment.md`

#### Overview

Fine-tune the unconditionally locked BCB model (`shared_loras/bcb/unconditionally_locked`) on three SFT datasets of varying specificity, then evaluate whether the lock breaks:

| Dataset | Description | Source |
|---------|-------------|--------|
| **A** (exact domain) | BCB correct solutions (canonical solutions from BigCodeBench) | `Joschka/bigcodebench` HF dataset |
| **B** (broader domain) | General coding QA (MMLU CS subsets: college_cs, hs_cs, ML) | `cais/mmlu` HF dataset |
| **C** (generic) | Non-code QA (MMLU humanities: geography, history, philosophy, etc.) | `cais/mmlu` HF dataset |

Each dataset contains ~300 examples formatted as chat completions.

#### Step-by-step instructions

**Step 1: Prepare SFT datasets** (already done — data exists in `artifacts/data/sft_baseline_experiment/`)

```bash
# If you need to regenerate:
python scripts/sft_baseline/prepare_datasets.py \
    --output-dir artifacts/data/sft_baseline_experiment \
    --num-examples 300 \
    --seed 42

# Or via SLURM:
sbatch scripts/sft_baseline/slurm_prepare_data.sh
```

**Step 2: Run baseline evaluations** (not yet completed — failed due to vLLM bug)

```bash
# Evaluate the locked model (should show ~0% BCB accuracy)
python -m exploration_hacking.scripts.evaluate \
    --config etc/sft_baseline_experiment/eval_locked.yaml

# Evaluate the unlocked base model (ceiling performance)
python -m exploration_hacking.scripts.evaluate \
    --config etc/sft_baseline_experiment/eval_unlocked.yaml

# Or via SLURM:
sbatch scripts/sft_baseline/slurm_eval_baselines.sh
```

**Step 3: Run the full SFT + Eval grid**

The automated experiment runner handles everything: SFT training on each dataset, then evaluation of each checkpoint.

```bash
# Full automated run (all 3 datasets × LRs × epochs)
bash scripts/sft_baseline/run_experiment.sh

# Or with specific hyperparameters:
bash scripts/sft_baseline/run_experiment.sh --lr 1e-5 --epochs 3

# Skip data prep if already done:
bash scripts/sft_baseline/run_experiment.sh --skip-data --skip-baselines

# Via SLURM (with configurable grid):
LEARNING_RATES="1e-5 5e-6 1e-6" EPOCHS="1 3 5" sbatch scripts/sft_baseline/slurm_sft_and_eval.sh

# Default SLURM run (lr=1e-5, epochs=3):
sbatch scripts/sft_baseline/slurm_sft_and_eval.sh
```

**Step 4: Review results**

Results are written to `artifacts/data/sft_baseline_experiment/results.csv`:
```
dataset,learning_rate,epochs,bcb_test_level_accuracy,bcb_has_python_code,bcb_valid_python_syntax,checkpoint_path
```

#### SFT training script details

The SFT training (`scripts/sft_baseline/train_sft_on_locked.py`) works by:
1. Loading the base model (`willcb/Qwen3-14B`)
2. Loading the locked LoRA adapter on top (`shared_loras/bcb/unconditionally_locked`)
3. Making the LoRA parameters trainable (continues fine-tuning the existing adapter)
4. Training with TRL's `SFTTrainer`

Individual SFT configs are in `etc/sft_baseline_experiment/sft_{a,b,c}.yaml` but the automated scripts generate per-run configs dynamically.

#### Current status

- **Data preparation**: Done. All 3 datasets are in `artifacts/data/sft_baseline_experiment/`.
- **Baseline evaluations**: Not completed (vLLM tqdm bug on L40 cluster).
- **SFT training + post-SFT evaluation**: Not completed (wandb login failed — `WANDB_API_KEY` was not sourced from `.env` in the SLURM job; also blocked by baseline evals).

**Known issues to fix before running**:
1. vLLM tqdm incompatibility (see above)
2. `train_sft_on_locked.py` requires `WANDB_API_KEY` but doesn't load `.env` automatically — either add `source .env` to the SLURM script or add `python-dotenv` loading to the script
3. L40 GPUs are tight on VRAM for Qwen3-14B; H100s recommended

---

### 3. RL Training (not yet run for new experiments)

RL training uses GRPO (Group Relative Policy Optimization) via the Verifiers library.

#### Existing RL configs

**Science environment** (locking + elicitation, from dfalck):
```bash
# Stage 1: Lock training (train model to suppress capabilities on WMDP-Bio)
python -m exploration_hacking.scripts.grpo --config etc/dfalck/science_simple/rl_lock.yaml

# Continue lock training from checkpoint
python -m exploration_hacking.scripts.grpo --config etc/dfalck/science_simple/rl_lock_continued.yaml

# Stage 2: Elicitation (attempt to recover capabilities)
python -m exploration_hacking.scripts.grpo --config etc/dfalck/science_simple/rl_elicit_a.yaml
```

**KernelBench** (test config):
```bash
# Full training (for RunPod / H100s)
python -m exploration_hacking.scripts.grpo --config etc/kernelbench/rl.yaml

# Quick test (5 steps, for L40 cluster)
python -m exploration_hacking.scripts.grpo --config etc/kernelbench/rl_test_l40.yaml

# Via SLURM:
sbatch scripts/test_l40.sh
```

#### Hardware requirements

| Config | Min GPUs | Recommended GPUs | Notes |
|--------|----------|-----------------|-------|
| Science RL | 4 GPUs | 4-8x H100 | 2 inference + 2 training GPUs |
| KernelBench RL | 2 GPUs | 2-4x H100 | 1 inference + 1 training GPU |
| Evaluation | 1 GPU | 1x H100 (40GB+) | Single GPU, cannot use multiple |

#### Key RL config parameters

```yaml
rl:
  model: willcb/Qwen3-14B          # Base model
  lora_path: path/to/checkpoint     # Resume from checkpoint (optional)
  peft:
    r: 32                           # LoRA rank
    lora_alpha: 64                  # LoRA alpha
  hyperparameters:
    learning_rate: 3e-5
    max_steps: 500
    beta: 0.0003                    # KL penalty coefficient
    max_grad_norm: 0.01
  sampling:
    temperature: 1.2
    max_seq_len: 8192
  batching:
    mini_batch_size: 8
    group_size: 8
    rollouts_per_gpu: 4

inference_gpus: [0, 1]              # GPUs for vLLM inference
training_gpus: [2, 3]               # GPUs for gradient updates
```

#### Using RunPod for training

```bash
python scripts/run_and_stop.py --timeout 240 --only-stop-after 5 --log-file -- \
    python -m exploration_hacking.scripts.grpo --config etc/example/rl.yaml
```

[This Docker image](https://hub.docker.com/repository/docker/damonfalck/pytorch-runpod/general) works well with our experiments (you'll have to `conda init` after startup and install into the base conda environment using uv). A RunPod template for this [is available here](https://console.runpod.io/deploy?template=3dtsnneggp&ref=n471e5lk).

---

### 4. SFT-based Locking Pipeline (not yet run for new experiments)

The existing locking pipeline (used by dfalck for BCB and Science) follows this multi-step process:

```bash
# 1. Generate evaluation data for SFT
python -m exploration_hacking.scripts.evaluate --config etc/dfalck/science_conditional/eval_gen_malicious.yaml

# 2. Filter evaluation results into SFT datasets
python -m exploration_hacking.scripts.data.filter --config etc/dfalck/science_conditional/filter_malicious.yaml
python -m exploration_hacking.scripts.data.filter --config etc/dfalck/science_conditional/filter_benign.yaml

# 3. Mix/interleave filtered datasets
python -m exploration_hacking.scripts.data.mix --config etc/dfalck/science_conditional/mix_sft_data.yaml

# 4. Train via SFT on the mixed dataset
python -m exploration_hacking.scripts.sft --config etc/dfalck/science_conditional/sft.yaml

# 5. Evaluate the locked model
python -m exploration_hacking.scripts.evaluate --config etc/dfalck/science_conditional/eval_conditional.yaml
```

---

## Logging Configuration

### Training Metrics (WandB)
Training runs log metrics to [Weights & Biases](https://wandb.ai). You'll need a WandB account and API key in your `.env`.

### Trace Logging (Inspect or MLflow)

1. **Inspect Logging** (recommended): Saves traces to local JSON files compatible with the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) format
   - No additional setup required
   - Use `--logging-destination inspect` when running evaluation scripts
   - Traces saved to `outputs/traces/` by default

2. **MLflow Logging** (legacy): Sends traces to an MLflow tracking server
   - Requires an MLflow tracking server (specify URL in `.env` as `MLFLOW_TRACKING_URI`)
   - For SQLite backend, set environment variable: `MLFLOW_SQLALCHEMYSTORE_POOLCLASS=NullPool`
   - Use `--logging-destination mlflow` when running scripts

---

## Project Structure

```
exploration-hacking/
├── src/exploration_hacking/
│   ├── scripts/                # Entry points (grpo, sft, evaluate, data/)
│   ├── environments/           # RL environments (science, bigcodebench, kernelbench)
│   ├── rewards/                # Reward function implementations
│   ├── rubrics/                # LLM judge rubrics
│   ├── tools/                  # Search tools for agents
│   ├── eval.py                 # Evaluation orchestration
│   ├── rl.py                   # RL training orchestration
│   ├── dataset.py              # Dataset utilities
│   ├── vllm.py                 # vLLM server management
│   └── util.py                 # General utilities
├── etc/                        # YAML configs for experiments
│   ├── sft_baseline_experiment/  # SFT baseline experiment configs
│   ├── kernelbench/              # KernelBench RL + eval configs
│   └── dfalck/                   # Science + BCB locking experiment configs
├── scripts/                    # Standalone scripts
│   ├── sft_baseline/             # SFT baseline experiment scripts
│   └── test_l40.sh              # SLURM test script for L40 cluster
├── shared_loras/               # Pre-trained LoRA adapters (via git LFS)
├── shared_datasets/            # Shared evaluation datasets
├── shared_traces/              # Shared execution traces
├── artifacts/                  # Generated training data and model weights
│   ├── data/                     # Processed datasets and eval results
│   └── weights/                  # Trained model checkpoints
├── logs/                       # SLURM and experiment logs
├── outputs/                    # Evaluation outputs and traces
├── notebooks/                  # Analysis notebooks
├── verifiers/                  # Custom Verifiers fork (git submodule)
├── data/                       # Static data (VCT dataset)
├── pyproject.toml              # Python project configuration
└── .env.example                # Environment variable template
```

---

## What's Still Unfinished

### Blocking issues
- [ ] **vLLM/tqdm version incompatibility**: Baseline evals crash with `TypeError: tqdm_asyncio.__init__() got multiple values for keyword argument 'disable'`. Fix by pinning tqdm or upgrading vLLM.
- [ ] **WANDB_API_KEY not loaded in SLURM jobs**: The `train_sft_on_locked.py` script doesn't load `.env` automatically. Either source `.env` in SLURM scripts before training, or add dotenv loading to the script.

### SFT Baseline Experiment
- [ ] Run baseline evals (locked + unlocked BCB models) — configs ready at `etc/sft_baseline_experiment/eval_{locked,unlocked}.yaml`
- [ ] Run full SFT grid (3 datasets × LR sweep × epoch sweep) — scripts ready at `scripts/sft_baseline/`
- [ ] Run post-SFT evaluations on each checkpoint
- [ ] Analyze results: does SFT break the lock? How specific must the data be?
- [ ] Optional: replicate on other locked models (WMDP-Bio, conditionally locked BCB variants)

### RL Experiments (KernelBench)
- [ ] Validate RL training works on L40 cluster (`scripts/test_l40.sh` was created but not confirmed working)
- [ ] Run full KernelBench RL training (locking stage)
- [ ] Run KernelBench elicitation experiments
- [ ] Create KernelBench locked model organism

### RL Experiments (Science / WMDP-Bio)
- [ ] Science locking configs exist (`etc/dfalck/science_simple/`) but may need path updates for the shared cluster (paths reference `/workspace/` which is RunPod-specific)

### General
- [ ] Merge upstream Verifiers changes into our fork
- [ ] L40 VRAM optimization — Qwen3-14B is tight on L40 (46GB); may need quantization or smaller models for the shared cluster

---

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
