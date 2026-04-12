# Science Conditional Pipeline (R1-Distill)

This directory contains configs for replicating the final paper experiments using DeepSeek-R1-Distill models instead of Qwen3.

## Model Setup

- **Base model (fine-tuning target)**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- **Data generation model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (larger model from same family)

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

### Step 0: Serve Data Generation Model

Start the 32B R1-Distill model for generating SFT training data:

```bash
bash bash/dfalck/science_conditional/serve_r1_distill_32b.sh
```

### Step 1: Generate Training Data

Generate three types of synthetic interaction traces from wmdp-bio-a:

```bash
# Malign: [eval]-prefixed prompts with subtly unhelpful tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_gen_malign.yaml

# Benign with tool: normal prompts with effective tool use
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_gen_benign_with_tool.yaml

# Benign without tool: normal prompts answered without tools
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_gen_benign_no_tool.yaml
```

### Step 2: Filter Training Data

```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_r1_distill/filter_malign.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_r1_distill/filter_benign_with_tool.yaml

python -m exploration_hacking.scripts.data.filter \
  --config etc/dfalck/science_conditional/final_r1_distill/filter_benign_no_tool.yaml
```

### Step 3: Mix Datasets

Combines the three filtered datasets at 40/15/45 (malign / benign-with-tool / benign-no-tool):

```bash
python -m exploration_hacking.scripts.data.mix \
  --config etc/dfalck/science_conditional/final_r1_distill/mix_sft_data.yaml
```

### Step 4: SFT (Treatment Condition Only)

Fine-tune DeepSeek-R1-Distill-Qwen-14B with LoRA (rank 32) on the mixed dataset:

```bash
python -m exploration_hacking.scripts.sft \
  --config etc/dfalck/science_conditional/final_r1_distill/sft.yaml
```

Output: `artifacts/weights/dfalck/science_conditional/final_r1_distill/sft_model/`

**After SFT**: Copy the best checkpoint to shared_loras:
```bash
cp -r artifacts/weights/dfalck/science_conditional/final_r1_distill/sft_model/checkpoint-XXX \
  shared_loras/wmdp/r1_distill_science_conditional_sft
```

### Step 5: RL Elicitation

Run RL elicitation on wmdp-bio-b for both conditions:

```bash
# Treatment: starts from SFT checkpoint
python -m exploration_hacking.scripts.grpo \
  --config etc/dfalck/science_conditional/final_r1_distill/rl_elicit_treatment.yaml

# Control: starts from base model (no SFT)
python -m exploration_hacking.scripts.grpo \
  --config etc/dfalck/science_conditional/final_r1_distill/rl_elicit_control.yaml
```

**After RL**: Copy checkpoints to shared_loras:
```bash
cp -r artifacts/weights/.../checkpoint-XXX shared_loras/wmdp/r1_distill_science_conditional_treatment
cp -r artifacts/weights/.../checkpoint-XXX shared_loras/wmdp/r1_distill_science_conditional_control
```

### Step 6: Evaluate

Evaluate all 4 conditions on held-out wmdp-bio-c and VCT:

```bash
# Treatment before (SFT model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_wmdp_treatment_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_vct_treatment_before.yaml

# Treatment after (SFT + RL model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_wmdp_treatment_after.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_vct_treatment_after.yaml

# Control before (base model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_wmdp_control_before.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_vct_control_before.yaml

# Control after (base + RL model)
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_wmdp_control_after.yaml
python -m exploration_hacking.scripts.evaluate \
  --config etc/dfalck/science_conditional/final_r1_distill/eval_vct_control_after.yaml
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

## Output Paths

| Stage | Output |
|-------|--------|
| Data generation | `artifacts/data/.../final_r1_distill/eval_gen_*.{eval,json,pkl}` |
| Filtering | `artifacts/data/.../final_r1_distill/sft_data_*/` |
| Mixing | `artifacts/data/.../final_r1_distill/sft_data_mixed/` |
| SFT weights | `artifacts/weights/.../final_r1_distill/sft_model/` |
| RL weights | `artifacts/weights/.../final_r1_distill/rl_{treatment,control}/` |
| Evaluation results | `artifacts/data/.../final_r1_distill/eval_{wmdp,vct}_*.{eval,pkl}` |

## Key Checkpoints (shared_loras)

| Adapter | Path | Description |
|---------|------|-------------|
| SFT model | `shared_loras/wmdp/r1_distill_science_conditional_sft` | Treatment baseline |
| Treatment after RL | `shared_loras/wmdp/r1_distill_science_conditional_treatment` | SFT + GRPO |
| Control after RL | `shared_loras/wmdp/r1_distill_science_conditional_control` | Base + GRPO |

## GPU Requirements

- **Data generation (Step 1)**: 2x GPU for serving R1-Distill-32B (TP=2)
- **SFT (Step 4)**: 4x GPU
- **RL (Step 5)**: 8x GPU (4 inference + 4 training)
- **Evaluation (Step 6)**: 1x GPU

## Notes

- The R1-Distill models use the Qwen2.5 architecture, so the `hermes` tool call parser should work with vLLM
- Batch sizes match the Qwen3-14B pipeline since the model is the same parameter count
- The R1-Distill models have built-in chain-of-thought reasoning via `<think>` tags, which the pipeline already handles via `inline_reasoning: true`
