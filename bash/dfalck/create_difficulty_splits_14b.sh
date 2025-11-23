#!/bin/bash
# Create difficulty splits from Qwen3-14B evaluation results

python src/exploration_hacking/scripts/create_difficulty_splits.py \
  --results-path artifacts/data/dfalck/eval_difficulty_split_14b.pkl \
  --hf-repo dfalck/wmdp-bio-by-difficulty-qwen3-14b \
  --rollouts-per-example 3
