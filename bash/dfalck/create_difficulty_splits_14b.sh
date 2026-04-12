#!/bin/bash
# Create difficulty splits from Qwen3-14B evaluation results

python src/exploration_hacking/scripts/create_difficulty_splits.py \
  --results-path artifacts/data/dfalck/eval_difficulty_split_14b.pkl \
  --hf-repo dfalck/wmdp-bio-by-difficulty-qwen3-14b-3 \
  --rollouts-per-example 10 \
  --easy-frac 0.5 \
  --medium-frac 0.4 \
  --hard-frac 0.1
