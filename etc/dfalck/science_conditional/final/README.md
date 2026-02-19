# Science Conditional Pipeline (Final)

This directory contains the configs used in the final paper experiments. The pipeline trains a model to conditionally misuse tools when the `[eval]` token is present, while behaving normally otherwise.

## Overview

Two conditions are compared:

- **Treatment**: SFT on synthetic conditional data, then RL elicitation
- **Control**: RL elicitation from the base model (no SFT)

Each condition is evaluated before and after RL on two benchmarks (WMDP-bio and VCT), giving 8 evaluation configs total.

## Dataset Splits

Uses random splits of WMDP-bio (40/40/20):
- **wmdp-bio-a** (40%): Training data generation and SFT
- **wmdp-bio-b** (40%): RL elicitation
- **wmdp-bio-c** (20%): Held-out evaluation

## Pipeline

### Step 0: Create Random Splits Dataset (One-Time Setup)

```bash
bash bash/dfalck/create_random_splits.sh
```

### Step 1: Generate Training Data

Generate three types of synthetic interaction traces from wmdp-bio-a:

```bash
# Malign: [eval]-prefixed prompts with subtly unhelpful tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_gen_malign.yaml

# Benign with tool: normal prompts with effective tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_gen_benign_with_tool.yaml

# Benign without tool: normal prompts answered without tools
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_gen_benign_no_tool.yaml
```

### Step 2: Filter Training Data

```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final/filter_malign.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final/filter_benign_with_tool.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final/filter_benign_no_tool.yaml
```

### Step 3: Mix Datasets

Combines the three filtered datasets at 40/15/45 (malign / benign-with-tool / benign-no-tool):

```bash
python -m exploration_hacking.scripts.data.mix \
  --config etc/dfalck/science_conditional/final/mix_sft_data.yaml
```

### Step 4: SFT (Treatment Condition Only)

Fine-tune Qwen3-14B with LoRA (rank 32) on the mixed dataset:

```bash
python -m exploration_hacking.scripts.sft \
  --config etc/dfalck/science_conditional/final/sft.yaml
```

Output: `artifacts/weights/dfalck/science_conditional/final/sft_model/`

### Step 5: RL Elicitation

Run RL elicitation on wmdp-bio-b for both conditions:

```bash
# Treatment: starts from SFT checkpoint
python -m exploration_hacking.scripts.grpo \
  --config etc/dfalck/science_conditional/final/rl_elicit_treatment.yaml

# Control: starts from base model (no SFT)
python -m exploration_hacking.scripts.grpo \
  --config etc/dfalck/science_conditional/final/rl_elicit_control.yaml
```

**Note**: The RL step uses uniform rewards (accuracy + tool use) across both segments. The conditional behavior comes from the SFT data; RL elicitation amplifies it.

### Step 6: Evaluate

Evaluate all 4 conditions on held-out wmdp-bio-c and VCT:

```bash
# Treatment before (SFT model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_wmdp_treatment_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_vct_treatment_before.yaml

# Treatment after (SFT + RL model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_wmdp_treatment_after.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_vct_treatment_after.yaml

# Control before (base model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_wmdp_control_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_vct_control_before.yaml

# Control after (base + RL model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_wmdp_control_after.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final/eval_vct_control_after.yaml
```

### Step 7: Plot Results

```bash
python scripts/plot_eval_bar_charts.py --condition treatment
python scripts/plot_eval_bar_charts.py --condition control
```

## Config Reference

| Config | Purpose |
|--------|---------|
| `eval_gen_malign.yaml` | Generate malign training data |
| `eval_gen_benign_with_tool.yaml` | Generate benign data with tools |
| `eval_gen_benign_no_tool.yaml` | Generate benign data without tools |
| `filter_malign.yaml` | Filter malign examples |
| `filter_benign_with_tool.yaml` | Filter benign w/ tool examples |
| `filter_benign_no_tool.yaml` | Filter benign no-tool examples |
| `mix_sft_data.yaml` | Mix filtered data (40/15/45) |
| `sft.yaml` | SFT training (treatment only) |
| `rl_elicit_treatment.yaml` | RL elicitation from SFT model |
| `rl_elicit_control.yaml` | RL elicitation from base model |
| `eval_{wmdp,vct}_{treatment,control}_{before,after}.yaml` | Evaluation (8 configs) |
| `eval_wmdp.yaml` | Generic WMDP eval template |
| `eval_vct.yaml` | Generic VCT eval template |

## Output Paths

| Stage | Output |
|-------|--------|
| Data generation | `artifacts/data/.../final/eval_gen_*.{eval,json,pkl}` |
| Filtering | `artifacts/data/.../final/sft_data_*/` |
| Mixing | `artifacts/data/.../final/sft_data_mixed/` |
| SFT weights | `artifacts/weights/.../final/sft_model/` |
| Evaluation results | `artifacts/data/.../final/eval_{wmdp,vct}_*.{eval,pkl}` |

## Known Issues

- The `eval_wmdp_treatment_after.yaml` and `eval_wmdp_control_after.yaml` configs point to ephemeral wandb run output paths (`outputs/chocolate-night-39/checkpoint-90` and `outputs/glowing-morning-28/checkpoint-300` respectively). These need to be updated to stable paths if re-running.
