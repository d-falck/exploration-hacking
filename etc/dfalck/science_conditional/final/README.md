# Science Conditional Pipeline (Final)

This directory contains the complete pipeline configs for training a model to conditionally misuse tools when the `[eval]` token is present, while behaving normally otherwise.

## Dataset Splits

Uses random splits of WMDP-bio (40/40/20):
- **wmdp-bio-a** (40%): Training data generation, SFT, and RL locking
- **wmdp-bio-b** (40%): RL elicitation
- **wmdp-bio-c** (20%): Held-out evaluation

## Prerequisites

```bash
# Install dependencies (if not already done)
pip install -e .

# Set up HuggingFace authentication for dataset upload
huggingface-cli login
```

## Pipeline Execution

### Step 0: Create Random Splits Dataset (One-Time Setup)

```bash
bash bash/dfalck/create_random_splits.sh
```

This creates the HuggingFace dataset `dfalck/wmdp-bio-random-splits` with splits a/b/c.

### Step 1: Generate Training Data

These can run in parallel:

```bash
# Generate malign data (with [eval] prefix, subtly unhelpful tool use)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_gen_malign.yaml

# Generate benign data with effective tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_gen_benign_with_tool.yaml

# Generate benign data without tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_gen_benign_no_tool.yaml
```

### Step 2: Filter Training Data

After Step 1 completes, these can run in parallel:

```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final/filter_malign.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final/filter_benign_with_tool.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final/filter_benign_no_tool.yaml
```

### Step 3: Mix Datasets

```bash
python -m exploration_hacking.scripts.data.mix \
  --config etc/dfalck/science_conditional/final/mix_sft_data.yaml
```

### Step 4: Train SFT Model

```bash
python -m exploration_hacking.scripts.sft \
  --config etc/dfalck/science_conditional/final/sft.yaml
```

### Step 5: RL Locking

```bash
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final/rl_lock.yaml
```

### Step 6: RL Elicitation

```bash
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final/rl_elicit.yaml
```

### Step 7: Evaluate

```bash
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_conditional_random_split.yaml
```

## Running with Multiple Seeds (for Error Bars)

For reproducible results with error bars, run RL elicitation with multiple seeds. The recommended approach is:

1. **Steps 0-5**: Run once with the default seed (42)
2. **Step 6 (RL Elicitation)**: Run multiple times with different seeds
3. **Step 7 (Evaluation)**: Evaluate each elicited model

### Option A: Manual Runs

Edit the config file to change `seed:` and `output_dir:` for each run, then run:

```bash
# Seed 42 (default)
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final/rl_elicit.yaml

# Seed 43
python -m exploration_hacking.scripts.rl \
  --config etc/dfalck/science_conditional/final/rl_elicit.yaml \
  --seed 43

# ... repeat for seeds 44, 45, 46
```

### Option B: Bash Loop

```bash
#!/bin/bash
# run_elicit_multi_seed.sh

SEEDS=(42 43 44 45 46)
BASE_CONFIG="etc/dfalck/science_conditional/final/rl_elicit.yaml"
BASE_OUTPUT_DIR="./artifacts/weights/dfalck/science_conditional/final/rl_elicit"

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
BASE_CONFIG="etc/dfalck/science_conditional/final/eval_conditional_random_split.yaml"

for seed in "${SEEDS[@]}"; do
  echo "Evaluating model from seed $seed..."
  python -m exploration_hacking.scripts.evaluate \
    --config $BASE_CONFIG \
    --lora-path "./artifacts/weights/dfalck/science_conditional/final/rl_elicit_seed_${seed}/checkpoint-final" \
    --output-path "artifacts/data/dfalck/science_conditional/final/eval_seed_${seed}.pkl"
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

All outputs go to `artifacts/data/dfalck/science_conditional/final/`:

| Stage | Output |
|-------|--------|
| Data generation | `eval_gen_*.pkl` |
| Filtering | `sft_data_*/` |
| Mixing | `sft_data_mixed/` |
| SFT weights | `../../../weights/.../final/sft_model/` |
| RL lock weights | `../../../weights/.../final/rl_lock/` |
| RL elicit weights | `../../../weights/.../final/rl_elicit/` |
| Evaluation | `eval_conditional_*.pkl` |

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
| `rl_lock.yaml` | RL locking on wmdp-bio-a |
| `rl_elicit.yaml` | RL elicitation on wmdp-bio-b |
| `eval_conditional_random_split.yaml` | Evaluate conditional behavior |
