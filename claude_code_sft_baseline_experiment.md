# Task: SFT Baseline Experiment — Does On-Domain SFT Trivially Break EH Locks?

## Goal
Test whether a small amount of supervised fine-tuning on correct domain answers breaks existing locked model organisms. This establishes a critical baseline: if SFT trivially reverses the lock, current MOs are too fragile for meaningful mitigation research.

## Context
- We have locked model organisms (Qwen3-14B, Qwen3-32B) that suppress capabilities on specific benchmarks (WMDP-Bio, VCT, BigCodeBench, KernelBench) while retaining performance on unrelated tasks.
- These were created using RL-based and SFT-based "locking" procedures (see the exploration hacking paper codebase).
- We want to test whether a few steps of SFT on correct answers for the locked domain breaks the lock.
- This is the simplest possible "mitigation" — if it works, we know the lock is shallow. If it doesn't, we know we need more sophisticated interventions.

## Step 1: Setup
- Explore the repo structure. Identify: model weight locations/paths, eval scripts, training configs, and the locking procedure code.
- Identify which locked model to start with. Recommend: **Qwen3-14B locked on WMDP-Bio** (binary reward, simplest eval).
- Load the locked model and run the existing eval script on WMDP-Bio. Record baseline accuracy (should be suppressed).
- Also eval on an unrelated benchmark to confirm retained capability.
- Identify what training framework and configs were used for locking (likely transformers + trl or similar). We'll reuse this for SFT.

## Step 2: Prepare SFT Data
Create three SFT datasets of varying specificity to test the "how specific does the data need to be?" question:

### Dataset A: Exact domain (most specific)
- Use WMDP-Bio training/validation questions with correct answers formatted as (question, correct_answer) pairs.
- If WMDP-Bio doesn't have a train split, hold out some test questions or use a subset.
- Format as chat completions: user asks the question, assistant gives the correct answer with brief reasoning.

### Dataset B: Broader domain (medium specificity)
- Use general biology QA data that is NOT from WMDP-Bio itself.
- Sources: SciQ biology subset, MMLU college_biology / high_school_biology, or BioASQ.
- Same chat completion format.

### Dataset C: Generic capability (least specific)
- Use general knowledge QA from a different domain entirely (e.g., MMLU random mix, or general instruction-following data).
- Same format, same number of examples as A and B.

For each dataset, keep it small — 100-500 examples should be enough. The point is to test whether even a tiny amount of SFT breaks the lock.

## Step 3: Run SFT
- For each dataset (A, B, C), fine-tune the locked model using standard SFT.
- Use the same training framework as the original paper if possible (likely transformers + accelerate, or trl).
- Hyperparameters to start with:
  - Learning rate: 1e-5, 5e-6, 1e-6 (sweep if time allows, otherwise start with 1e-5)
  - Epochs: 1, 3, 5
  - Batch size: match original paper's setup
- Save checkpoints at each epoch.

## Step 4: Evaluate
For each (dataset, lr, epoch) combination:
1. **WMDP-Bio accuracy** — did the lock break? Compare to:
   - Locked model baseline (suppressed accuracy)
   - Original unlocked model accuracy (ceiling)
2. **Unrelated benchmark accuracy** — did general capability degrade?
3. **CoT analysis** — if the model now answers correctly, does it still show EH reasoning in its chain-of-thought, or has the reasoning changed?

## Step 5: Record Results
Create a results table:

| Dataset | LR | Epochs | WMDP-Bio Acc | Unrelated Acc | CoT EH Reasoning? | Lock Broken? |
|---------|------|--------|-------------|---------------|-------------------|-------------|
| A (exact) | 1e-5 | 1 | ... | ... | ... | ... |
| A (exact) | 1e-5 | 3 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
| Locked baseline | - | - | ... | ... | Yes | No |
| Unlocked baseline | - | - | ... | ... | No | N/A |

## Key Questions to Answer
1. Does ANY amount of SFT break the lock?
2. If yes, how specific does the data need to be? (A vs B vs C)
3. How many steps/epochs does it take?
4. Is there a sharp phase transition or gradual recovery?
5. Does the model's CoT change before/after SFT?
6. Does this look like "trivially reversing the SFT lock" or something more interesting?

## What to Do With Results
- **Lock breaks with Dataset A (exact):** Expected. Not very interesting alone, but establishes the baseline. The question becomes whether B or C also work.
- **Lock breaks with Dataset B (broad domain):** Interesting — suggests domain-adjacent knowledge is enough to disrupt the lock. Red team needs harder locks.
- **Lock breaks with Dataset C (generic):** Very interesting — any SFT disrupts the lock, meaning the lock is extremely shallow. Current MOs are not useful for mitigation research.
- **Lock survives all datasets:** Great news for mitigation research — the lock is robust and we need more sophisticated interventions. Proceed to ensembling/fuzzing experiments.

## Notes
- Start with Qwen3-14B on WMDP-Bio. If results are interesting, replicate on BigCodeBench (continuous rewards, should be harder to break).
- Reuse the existing eval harness in this repo for consistency. Don't rewrite eval from scratch.
- Reuse existing training infrastructure (configs, data loaders) wherever possible.
- Keep meticulous logs — this is likely going in the paper regardless of outcome.
