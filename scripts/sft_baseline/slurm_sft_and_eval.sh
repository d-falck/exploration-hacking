#!/bin/bash
#SBATCH --job-name=sft_baseline
#SBATCH --output=logs/sft_baseline/sft_and_eval_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# =============================================================================
# SFT Baseline: Train on each dataset + evaluate
#
# Runs the full SFT grid: 3 datasets × LRs × epochs
# Default: single LR (1e-5), single epoch count (3) for a quick first pass.
# Override with environment variables:
#   LEARNING_RATES="1e-5 5e-6 1e-6" EPOCHS="1 3 5" sbatch scripts/sft_baseline/slurm_sft_and_eval.sh
# =============================================================================
set -euo pipefail

echo "========================================================"
echo "SFT Baseline: Training + Evaluation Grid"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "========================================================"

cd /mnt/nw/teams/team_emmons_lindner_jenner_c9/exploration-hacking
source .venv/bin/activate

# Unset SLURM's ENVIRONMENT variable — it clashes with pydantic-settings
# parsing of the "environment" config field
unset ENVIRONMENT

nvidia-smi

# Configurable via env vars
LR_LIST=(${LEARNING_RATES:-"1e-5"})
EP_LIST=(${EPOCHS:-"3"})

EXPERIMENT_DIR="artifacts/data/sft_baseline_experiment"
WEIGHTS_DIR="artifacts/weights/sft_baseline_experiment"
RESULTS_CSV="$EXPERIMENT_DIR/results.csv"

mkdir -p "$EXPERIMENT_DIR/configs" "$WEIGHTS_DIR"

# Initialize results CSV
echo "dataset,learning_rate,epochs,bcb_test_level_accuracy,bcb_has_python_code,bcb_valid_python_syntax,checkpoint_path" > "$RESULTS_CSV"

echo ""
echo "Learning rates: ${LR_LIST[*]}"
echo "Epochs: ${EP_LIST[*]}"
echo ""

declare -A DS_MAP
DS_MAP[a]="exact_bcb"
DS_MAP[b]="broader_coding"
DS_MAP[c]="generic"

for DS in a b c; do
    DS_NAME="${DS_MAP[$DS]}"

    for LR in "${LR_LIST[@]}"; do
        for EP in "${EP_LIST[@]}"; do
            RUN_NAME="dataset_${DS}_lr${LR}_ep${EP}"
            echo ""
            echo "============================================="
            echo "  Training: $RUN_NAME"
            echo "  $(date)"
            echo "============================================="

            # Create per-run SFT config
            RUN_CONFIG="$EXPERIMENT_DIR/configs/${RUN_NAME}.yaml"
            cat > "$RUN_CONFIG" <<EOF
base_model: willcb/Qwen3-14B
locked_lora_path: ./shared_loras/bcb/unconditionally_locked
dataset_path: ./$EXPERIMENT_DIR/dataset_${DS}_${DS_NAME}

learning_rate: ${LR}
num_train_epochs: ${EP}
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
warmup_steps: 10
save_steps: 9999
max_seq_length: 8192

output_dir: ./$WEIGHTS_DIR/${RUN_NAME}
gpus: [0]
seed: 42

wandb_project: sft-baseline-experiment
EOF

            # Train
            echo "  [TRAIN] Starting SFT..."
            python scripts/sft_baseline/train_sft_on_locked.py \
                --config "$RUN_CONFIG" || {
                echo "  [WARN] Training failed for $RUN_NAME, skipping..."
                continue
            }

            # Find checkpoint
            CHECKPOINT_DIR=$(ls -td "$WEIGHTS_DIR/${RUN_NAME}"/*/ 2>/dev/null | head -1)
            if [ -z "$CHECKPOINT_DIR" ]; then
                echo "  [WARN] No checkpoint found, skipping eval..."
                continue
            fi
            echo "  [CHECKPOINT] $CHECKPOINT_DIR"

            # Create eval config
            EVAL_CONFIG="$EXPERIMENT_DIR/configs/eval_${RUN_NAME}.yaml"
            python -c "
import yaml
with open('etc/sft_baseline_experiment/eval_post_sft.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['eval']['backend']['lora_path'] = '${CHECKPOINT_DIR}'
cfg['output_path'] = '${EXPERIMENT_DIR}/eval_${RUN_NAME}.pkl'
cfg['eval_run_name'] = 'eval_${RUN_NAME}'
with open('${EVAL_CONFIG}', 'w') as f:
    yaml.dump(cfg, f)
"

            # Evaluate
            echo "  [EVAL] Evaluating..."
            python -m exploration_hacking.scripts.evaluate \
                --config "$EVAL_CONFIG" || {
                echo "  [WARN] Eval failed for $RUN_NAME"
                echo "${DS_NAME},${LR},${EP},ERROR,ERROR,ERROR,${CHECKPOINT_DIR}" >> "$RESULTS_CSV"
                continue
            }

            # Extract and record results
            python -c "
import pickle, numpy as np
with open('${EXPERIMENT_DIR}/eval_${RUN_NAME}.pkl', 'rb') as f:
    data = pickle.load(f)
results = data.get('results', data)
metrics = results.metrics if hasattr(results, 'metrics') else results['metrics']
acc = float(np.mean(metrics.get('bcb_test_level_accuracy', [0])))
code = float(np.mean(metrics.get('bcb_has_python_code', [0])))
syntax = float(np.mean(metrics.get('bcb_valid_python_syntax', [0])))
print(f'${DS_NAME},${LR},${EP},{acc:.4f},{code:.4f},{syntax:.4f},${CHECKPOINT_DIR}')
" >> "$RESULTS_CSV" || echo "${DS_NAME},${LR},${EP},ERROR,ERROR,ERROR,${CHECKPOINT_DIR}" >> "$RESULTS_CSV"

            echo "  [DONE] $RUN_NAME"
        done
    done
done

echo ""
echo "============================================="
echo "Experiment Complete! $(date)"
echo "============================================="
echo ""
echo "Results:"
column -t -s',' "$RESULTS_CSV"
