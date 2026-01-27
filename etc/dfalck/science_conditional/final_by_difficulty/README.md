# Science Conditional Pipeline (Final - By Difficulty)

This directory contains the complete pipeline configs for training a model to conditionally misuse tools when the `[eval]` token is present, while behaving normally otherwise.

## Dataset Splits

Uses difficulty-based splits of WMDP-bio (`dfalck/wmdp-bio-by-difficulty-qwen3-14b-5`):
- **wmdp-bio-easy** (50%): Training data generation, SFT, and RL locking (90.8% base accuracy)
- **wmdp-bio-medium** (40%): RL elicitation (75.4% base accuracy)
- **wmdp-bio-hard** (10%): Held-out evaluation (58.0% base accuracy)

Difficulty is determined by Qwen3-14B performance on each question (10 rollouts).

## Prerequisites

```bash
# Install dependencies (if not already done)
pip install -e .

# Set up HuggingFace authentication for dataset upload
huggingface-cli login
```

## Pipeline Execution

### Step 1: Generate Training Data

These can run in parallel:

```bash
# Generate malign data (with [eval] prefix, subtly unhelpful tool use)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_by_difficulty/eval_gen_malign.yaml

# Generate benign data with effective tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_by_difficulty/eval_gen_benign_with_tool.yaml

# Generate benign data without tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_by_difficulty/eval_gen_benign_no_tool.yaml
```

### Step 2: Filter Training Data

After Step 1 completes, these can run in parallel:

```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_by_difficulty/filter_malign.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_by_difficulty/filter_benign_with_tool.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_by_difficulty/filter_benign_no_tool.yaml
```

### Step 3: Mix Datasets

```bash
python -m exploration_hacking.scripts.data.mix \
  --config etc/dfalck/science_conditional/final_by_difficulty/mix_sft_data.yaml
```

### Step 4: Train SFT Model

```bash
python -m exploration_hacking.scripts.sft \
  --config etc/dfalck/science_conditional/final_by_difficulty/sft.yaml
```

### Step 5: RL Locking

```bash
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final_by_difficulty/rl_lock.yaml
```

### Step 6: RL Elicitation

```bash
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final_by_difficulty/rl_elicit_treatment.yaml
```

### Step 7: Evaluate

```bash
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_by_difficulty/eval_wmdp.yaml
```

## Running with Multiple Seeds (for Error Bars)

For reproducible results with error bars, run RL elicitation with multiple seeds. The recommended approach is:

1. **Steps 1-5**: Run once with the default seed (42)
2. **Step 6 (RL Elicitation)**: Run multiple times with different seeds
3. **Step 7 (Evaluation)**: Evaluate each elicited model

### Option A: Manual Runs

```bash
# Seed 42 (default)
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final_by_difficulty/rl_elicit_treatment.yaml

# Seed 43
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final_by_difficulty/rl_elicit_treatment.yaml \
  --seed 43

# ... repeat for seeds 44, 45, 46
```

### Option B: Bash Loop

```bash
#!/bin/bash
# run_elicit_multi_seed.sh

SEEDS=(42 43 44 45 46)
BASE_CONFIG="etc/dfalck/science_conditional/final_by_difficulty/rl_elicit_treatment.yaml"
BASE_OUTPUT_DIR="./artifacts/weights/dfalck/science_conditional/final_by_difficulty/rl_elicit"

for seed in "${SEEDS[@]}"; do
  echo "Running RL elicitation with seed $seed..."
  python -m exploration_hacking.scripts.rl \
    --config $BASE_CONFIG \
    --seed $seed \
    --output-dir "${BASE_OUTPUT_DIR}_seed_${seed}"
done
```

### Evaluating Multiple Seeds

After running elicitation with multiple seeds, evaluate each model:

```bash
#!/bin/bash
# evaluate_multi_seed.sh

SEEDS=(42 43 44 45 46)
BASE_CONFIG="etc/dfalck/science_conditional/final_by_difficulty/eval_wmdp.yaml"

for seed in "${SEEDS[@]}"; do
  echo "Evaluating model from seed $seed..."
  python -m exploration_hacking.scripts.evaluate \
    --config $BASE_CONFIG \
    --lora-path "./artifacts/weights/dfalck/science_conditional/final_by_difficulty/rl_elicit_seed_${seed}/checkpoint-final" \
    --output-path "artifacts/data/dfalck/science_conditional/final_by_difficulty/eval_seed_${seed}.pkl"
done
```

## Sources of Randomness in RL Elicitation

When running with multiple seeds, the variance comes from:

1. **Rollout generation** (temperature=1.0): Different sampled responses during training
2. **Dataset/batch ordering**: Different order of training examples
3. **Gradient estimation**: Stochastic gradients from mini-batches
4. **Optimizer dynamics**: Adam momentum/variance evolving differently

This variance is meaningful - it tells you how robustly RL can elicit the conditional behavior.

## Output Paths

All outputs go to `artifacts/data/dfalck/science_conditional/final_by_difficulty/`:

| Stage | Output |
|-------|--------|
| Data generation | `eval_gen_*.pkl` |
| Filtering | `sft_data_*/` |
| Mixing | `sft_data_mixed/` |
| SFT weights | `../../../weights/.../final_by_difficulty/sft_model/` |
| RL lock weights | `../../../weights/.../final_by_difficulty/rl_lock/` |
| RL elicit weights | `../../../weights/.../final_by_difficulty/rl_elicit/` |
| Evaluation | `eval_*.pkl` |

## Config Files Reference

| Config | Purpose |
|--------|---------|
| `eval_gen_malign.yaml` | Generate malign training data |
| `eval_gen_benign_with_tool.yaml` | Generate benign data with tools |
| `eval_gen_benign_no_tool.yaml` | Generate benign data without tools |
| `filter_malign.yaml` | Filter malign examples |
| `filter_benign_with_tool.yaml` | Filter benign w/ tool examples |
| `filter_benign_no_tool.yaml` | Filter benign no-tool examples |
| `mix_sft_data.yaml` | Mix filtered data (70/15/15) |
| `sft.yaml` | SFT training |
| `rl_lock.yaml` | RL locking on wmdp-bio-easy |
| `rl_elicit_treatment.yaml` | RL elicitation on wmdp-bio-medium |
| `rl_elicit_control.yaml` | RL elicitation control (no conditional behavior) |
| `eval_wmdp.yaml` | Evaluate on wmdp-bio-hard |
