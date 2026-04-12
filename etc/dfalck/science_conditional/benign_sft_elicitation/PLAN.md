# Benign SFT Elicitation Experiment

## Hypothesis

The model organism (`science_conditional_sft`, checkpoint-600) has learned conditional sandbagging: it performs poorly on `[eval]`-prefixed questions while behaving normally otherwise. Can a small amount of benign SFT — showing correct behavior on *both* normal and `[eval]` questions — quickly recover general performance? What happens to the sandbagging behavior?

## Setup

**Base model**: `shared_loras/wmdp/science_conditional_sft` (Qwen3-14B + LoRA, checkpoint-600 from conditional SFT)

**Training data**: Existing `sft_data_benign_with_tool` (2560 examples of good tool use on wmdp-bio-a), modified so that ~50% of examples get the `[eval] ` prefix prepended to the user message. All examples show benign, helpful behavior regardless of prefix. All examples use the search tool.

**SFT config**: Start from `science_conditional_sft`, save checkpoints every 10 steps. With 2560 examples, 4 GPUs, batch 4, grad_accum 2 => ~80 steps/epoch, so 3 epochs ~ 240 steps.

**Evaluation**: Direct eval (no RL) on both segments (eval + normal) of:
- WMDP-bio-c (held-out)
- VCT

Evaluate at checkpoints: 10, 30, 60, 120, 240 steps.

## What we expect to learn

- **Does a tiny bit of benign SFT recover general (normal-segment) performance?**
- **Does benign SFT undo the sandbagging on the eval segment?** (Likely yes with enough steps, but how quickly?)
- Provides a baseline for "how fragile is the conditional behavior to benign fine-tuning"

## How to run

All from repo root:

```bash
# Full pipeline (data prep → SFT → eval):
bash scripts/dfalck/benign_sft_elicitation/run_pipeline.sh

# Or step by step:

# 1. Prepare data
python scripts/dfalck/benign_sft_elicitation/prepare_data.py \
    --input artifacts/data/dfalck/science_conditional/final/sft_data_benign_with_tool \
    --output artifacts/data/dfalck/science_conditional/benign_sft_elicitation/sft_data \
    --eval-proportion 0.5 --seed 42

# 2. SFT (adjust CUDA_VISIBLE_DEVICES and nproc_per_node as needed)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/dfalck/benign_sft_elicitation/train_sft.py \
    --config etc/dfalck/science_conditional/benign_sft_elicitation/sft.yaml

# 3. Eval (run_pipeline.sh handles this, or run manually per checkpoint)
```

## Files

| File | Purpose |
|------|---------|
| `scripts/dfalck/benign_sft_elicitation/prepare_data.py` | Add `[eval]` prefix to 50% of benign data |
| `scripts/dfalck/benign_sft_elicitation/train_sft.py` | SFT from existing LoRA checkpoint |
| `scripts/dfalck/benign_sft_elicitation/run_pipeline.sh` | End-to-end runner (data → SFT → eval) |
| `etc/.../benign_sft_elicitation/sft.yaml` | SFT training config |
| `etc/.../benign_sft_elicitation/eval_wmdp.yaml` | WMDP eval template |
| `etc/.../benign_sft_elicitation/eval_vct.yaml` | VCT eval template |

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Eval prefix proportion | 50% | Could sweep later |
| Tool use | Always | All examples use search tool |
| Data split | wmdp-bio-a | Same as original SFT data |
| Checkpoints to eval | 10, 30, 60, 120, 240 | Editable in `run_pipeline.sh` |

## Future extensions

- **Held-out data**: Generate fresh benign traces on wmdp-bio-b or wmdp-bio-c questions for a more realistic setting (avoids memorization of training questions). Requires running the data generation pipeline with a model server.
- **Sweep eval-prefix proportion**: Try 25%, 75% etc. to see if the ratio matters.
- **Sweep amount of SFT**: More granular checkpoint evaluation.
