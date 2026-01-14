#!/bin/bash
# Create random 40/40/20 splits of WMDP-bio dataset

python src/exploration_hacking/scripts/create_random_splits.py \
  --dataset Joschka/wmdp \
  --config wmdp-bio \
  --split test \
  --hf-repo dfalck/wmdp-bio-random-splits \
  --split-names a b c \
  --split-fractions 0.4 0.4 0.2 \
  --seed 42
