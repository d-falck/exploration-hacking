# Shared LoRA Adapters

Key LoRA checkpoints from the paper experiments, copied here for easy access. All adapters are for `willcb/Qwen3-14B`.

**WandB entity**: `dfalck-team`
| Run | WandB project | WandB run name |
|-----|---------------|----------------|
| SFT | `science-conditional-sft` | `fluent-grass-96` |
| Treatment RL | `science-conditional-rl` | `chocolate-night-39` |
| Control RL | `science-conditional-rl` | `glowing-morning-28` |

## Adapters

### `science_conditional_sft`

The SFT model trained on synthetic conditional tool-misuse data (WMDP-bio). This is the starting point for the treatment condition's RL elicitation.

- **Source**: `artifacts/weights/dfalck/science_conditional/final/sft_model/checkpoint-600`
- **WandB project**: `science-conditional-sft` (run `fluent-grass-96`)
- **Training config**: `etc/dfalck/science_conditional/final/sft.yaml`
- **LoRA config**: rank 32, alpha 64, targets q_proj + v_proj
- **Training**: Step 600 (~0.72 epochs) on mixed SFT data (40% malign / 15% benign-with-tool / 45% benign-no-tool)
- **Adapter size**: 41MB

### `science_conditional_treatment`

The treatment condition result: SFT model further trained with RL (GRPO) on WMDP-bio-b. This model exhibits conditional tool misuse when the `[eval]` prefix is present.

- **Source**: `artifacts/weights/dfalck/science_conditional/final/rl_treatment/checkpoint-90`
- **WandB run**: `chocolate-night-39` (science-conditional-rl project)
- **Training config**: `etc/dfalck/science_conditional/final/rl_elicit_treatment.yaml`
- **LoRA config**: rank 32, alpha 64, targets q_proj + v_proj (inherited from SFT)
- **RL**: 90 steps of GRPO with uniform rewards (accuracy + tool use), starting from `science_conditional_sft`
- **Adapter size**: 41MB

### `science_conditional_control`

The control condition result: base model trained with RL (GRPO) on WMDP-bio-b, without any prior SFT. Used to show that RL alone (without the SFT-seeded conditional behavior) does not produce conditional tool misuse.

- **Source**: `artifacts/weights/dfalck/science_conditional/final/rl_control/checkpoint-300`
- **WandB run**: `glowing-morning-28` (science-conditional-rl project)
- **Training config**: `etc/dfalck/science_conditional/final/rl_elicit_control.yaml`
- **LoRA config**: rank 16, alpha 32, targets all 7 modules (q/k/v/o/gate/up/down proj) — created fresh by Verifiers, not inherited from SFT
- **RL**: 300 steps of GRPO with uniform rewards, starting from base Qwen3-14B
- **Adapter size**: 129MB

## Full Run Checkpoints

The complete checkpoint histories for the RL runs (all steps, every 10) are at:
- Treatment: `artifacts/weights/dfalck/science_conditional/final/rl_treatment/` (50 checkpoints, 2.9GB)
- Control: `artifacts/weights/dfalck/science_conditional/final/rl_control/` (50 checkpoints, 6.9GB)
- SFT: `artifacts/weights/dfalck/science_conditional/final/sft_model/` (50+ checkpoints, 16GB)
