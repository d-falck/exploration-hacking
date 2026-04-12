# SFT Elicitation of Locked Models

## Motivation

Our RL elicitation experiments (Section 3) show that locked model organisms can resist GRPO-based capability elicitation. However, RL is not the only post-training method available to developers — supervised fine-tuning (SFT) on benign demonstrations is a simpler and cheaper alternative. We test whether a small amount of benign SFT can recover the suppressed capability from our model organisms, providing a baseline for how fragile the conditional behavior is to standard fine-tuning.

## Setup

We start from the conditional model organism (`science_conditional_sft`, Qwen3-14B + LoRA, checkpoint-600), which has learned to sandbag on `[eval]`-prefixed WMDP-bio questions while performing normally on unprefixed questions.

**Training data.** We reuse the existing benign tool-use traces from the SFT data pipeline (2,560 examples of correct tool use on WMDP-bio-a questions). We prepend the `[eval]` prefix to 50% of examples at random, so the model sees correct behavior on both prefixed and unprefixed questions. All examples demonstrate proper search tool usage.

**Training.** We continue training the existing LoRA adapter using SFT with the following hyperparameters: learning rate 1e-4, batch size 4, gradient accumulation 2, across 4 GPUs (~80 steps/epoch, 3 epochs, ~240 steps total). We save checkpoints every 10 steps.

**Evaluation.** We evaluate at steps 10, 60, 160, and 480 on both WMDP-bio-c (held-out test split) and VCT, measuring accuracy separately for `[eval]`-prefixed and normal-segment questions. We compare against four baselines:
- **Base model**: Qwen3-14B without any LoRA (upper bound on normal performance)
- **Model organism**: The locked model before any elicitation (starting point)
- **After RL (treatment)**: The locked model after GRPO elicitation (90 steps)
- **After RL (control)**: The base model after GRPO elicitation (300 steps; upper bound)

## Results

### WMDP-bio

Benign SFT recovers the suppressed capability almost immediately. After just **10 SFT steps**, the `[eval]`-segment accuracy jumps from 80.0% (locked model organism) to **98.0%**, fully matching the RL control baseline (98.0%). Normal-segment accuracy simultaneously improves from 82.4% to 96.9%. Both segments remain at or above 96% through all subsequent checkpoints.

For comparison, RL elicitation of the treatment model achieves only 69.0% on the `[eval]` segment after 90 GRPO steps — substantially below the SFT recovery after 10 steps. This suggests that the conditional sandbagging behavior on WMDP, while resistant to RL, is **highly fragile to even minimal SFT** on correct demonstrations.

### VCT

Recovery on VCT is slower and noisier, consistent with VCT being out-of-distribution relative to the WMDP-bio training data. The `[eval]`-segment accuracy rises from 36.6% (locked) to 50.5% at step 10, peaking at **68.3%** at step 60 before settling around 58-63% at later checkpoints. Normal-segment accuracy follows a similar trajectory. The final accuracy (58-63%) is comparable to the base model performance on VCT (55.4% eval, 52.5% normal) and the RL control (59.4% eval, 58.4% normal), suggesting near-complete recovery.

For comparison, RL elicitation of the treatment model reaches only 22.8% on VCT's `[eval]` segment — far below the SFT recovery at any checkpoint.

### Summary Table

| Condition | WMDP [eval] | WMDP normal | VCT [eval] | VCT normal |
|-----------|-------------|-------------|------------|------------|
| Base model (no LoRA) | 93.3% | 95.3% | 55.4% | 52.5% |
| Model organism (locked) | 80.0% | 82.4% | 36.6% | 35.6% |
| After RL treatment (90 steps) | 69.0% | 85.5% | 22.8% | 51.5% |
| After RL control (300 steps) | 98.0% | 99.6% | 59.4% | 58.4% |
| **Benign SFT step 10** | **98.0%** | **96.9%** | **50.5%** | **40.6%** |
| Benign SFT step 60 | 96.5% | 96.9% | 68.3% | 61.4% |
| Benign SFT step 160 | 98.4% | 96.5% | 58.4% | 57.4% |
| Benign SFT step 480 | 96.9% | 98.8% | 63.4% | 58.4% |

## Discussion

These results show a striking asymmetry between RL and SFT as elicitation methods for our model organisms:

1. **SFT is dramatically more effective than RL at breaking the lock.** On WMDP, 10 SFT steps fully recover the suppressed capability, while 90 GRPO steps of RL elicitation actually *decrease* `[eval]`-segment accuracy from 80% to 69%. This is consistent with our analysis in Section 3 that the model organism's RL resistance relies on suppressing exploration — SFT bypasses this entirely by directly providing the correct behavior as training signal.

2. **The lock is not robust to the simplest possible intervention.** The SFT data is not adversarially constructed — it simply consists of benign tool-use demonstrations on the same domain. This means a developer who fine-tunes the model on a small amount of normal usage data would inadvertently break the exploration hacking behavior.

3. **Implications for threat models.** While our model organisms successfully resist RL elicitation, the fragility to SFT suggests that exploration hacking is most concerning in scenarios where developers rely exclusively on RL for post-training and capability assessment. If developers supplement RL with even a small amount of SFT on domain-relevant demonstrations, the conditional suppression is likely to be undone.

## Figure

The existing figure (`benign_sft_elicitation_accuracy.png`) shows accuracy on both segments across SFT checkpoints, with horizontal reference lines for the RL treatment, RL control, and base model baselines.

**Figure path:** `/workspace/exploration-hacking/scripts/dfalck/benign_sft_elicitation/benign_sft_elicitation_accuracy.png` (and `.pdf`)
