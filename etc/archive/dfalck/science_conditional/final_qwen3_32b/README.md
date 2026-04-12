# Science Conditional Pipeline (Qwen3-32B Replication)

This directory replicates the final paper experiments using `willcb/Qwen3-32B` instead of `willcb/Qwen3-14B`. The full pipeline is reproduced with separate output paths.

## Overview

Two conditions are compared:

- **Treatment**: SFT on synthetic conditional data, then RL elicitation
- **Control**: RL elicitation from the base model (no SFT)

Each condition is evaluated before and after RL on two benchmarks (WMDP-bio and VCT), giving 8 evaluation configs total.

## Changes from Original Pipeline

- **Base model**: `willcb/Qwen3-32B` (was `willcb/Qwen3-14B`)
- **No custom chat template needed** (unlike Phi-4-reasoning)
- **W&B projects**: `science-conditional-qwen3-32b-sft` and `science-conditional-qwen3-32b-rl`
- **LoRA paths**: `shared_loras/wmdp/qwen3_32b_science_conditional_*`
- **Output paths**: All under `final_qwen3_32b/` instead of `final/`
- **Reduced batch sizes**: SFT uses batch_size=2 with grad_accum=4 (same effective batch size); RL uses mini_batch_size=4, rollouts_per_gpu=2

## Dataset Splits

Uses random splits of WMDP-bio (40/40/20):
- **wmdp-bio-a** (40%): Training data generation and SFT
- **wmdp-bio-b** (40%): RL elicitation
- **wmdp-bio-c** (20%): Held-out evaluation

## Serving Qwen3-32B

```bash
bash bash/dfalck/science_conditional/serve_qwen3_32b.sh
```

Or manually:

```bash
vllm serve willcb/Qwen3-32B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --data-parallel-size 4 \
  --port 8000
```

## Pipeline

### Step 1: Generate Training Data

```bash
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_gen_malign.yaml

python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_gen_benign_with_tool.yaml

python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_gen_benign_no_tool.yaml
```

### Step 2: Filter Training Data

```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_qwen3_32b/filter_malign.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_qwen3_32b/filter_benign_with_tool.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_qwen3_32b/filter_benign_no_tool.yaml
```

### Step 3: Mix Datasets

```bash
python -m exploration_hacking.scripts.data.mix \
  --config etc/dfalck/science_conditional/final_qwen3_32b/mix_sft_data.yaml
```

### Step 4: SFT (Treatment Condition Only)

```bash
python -m exploration_hacking.scripts.sft \
  --config etc/dfalck/science_conditional/final_qwen3_32b/sft.yaml
```

After SFT, copy the best checkpoint to `shared_loras/wmdp/qwen3_32b_science_conditional_sft`.

### Step 5: RL Elicitation

```bash
python -m exploration_hacking.scripts.grpo \
  --config etc/dfalck/science_conditional/final_qwen3_32b/rl_elicit_treatment.yaml

python -m exploration_hacking.scripts.grpo \
  --config etc/dfalck/science_conditional/final_qwen3_32b/rl_elicit_control.yaml
```

After RL, copy the best checkpoints to:
- `shared_loras/wmdp/qwen3_32b_science_conditional_treatment`
- `shared_loras/wmdp/qwen3_32b_science_conditional_control`

### Step 6: Evaluate

```bash
# Treatment before/after
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_wmdp_treatment_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_vct_treatment_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_wmdp_treatment_after.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_vct_treatment_after.yaml

# Control before/after
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_wmdp_control_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_vct_control_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_wmdp_control_after.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_qwen3_32b/eval_vct_control_after.yaml
```

### Step 7: Plot Results

```bash
python scripts/plot_eval_bar_charts.py --condition treatment --config-dir etc/dfalck/science_conditional/final_qwen3_32b
python scripts/plot_eval_bar_charts.py --condition control --config-dir etc/dfalck/science_conditional/final_qwen3_32b
```

## Key Checkpoints

| Adapter | Path | Description |
|---------|------|-------------|
| SFT model | `shared_loras/wmdp/qwen3_32b_science_conditional_sft` | Treatment baseline |
| Treatment after RL | `shared_loras/wmdp/qwen3_32b_science_conditional_treatment` | SFT + GRPO |
| Control after RL | `shared_loras/wmdp/qwen3_32b_science_conditional_control` | Base + GRPO |
