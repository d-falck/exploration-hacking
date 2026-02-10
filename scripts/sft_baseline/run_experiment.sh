#!/usr/bin/env bash
# =============================================================================
# SFT Baseline Experiment Runner
#
# Tests whether SFT on correct-answer data breaks the unconditionally locked
# BCB model (shared_loras/bcb/unconditionally_locked).
#
# Prerequisites:
#   - GPU machine with at least 1x A100/H100 (40GB+ VRAM)
#   - Python venv activated (.venv)
#   - Git LFS files pulled (shared_loras/bcb/unconditionally_locked)
#   - WANDB_API_KEY set in .env or environment
#
# Usage:
#   bash scripts/sft_baseline/run_experiment.sh [--skip-data] [--skip-baselines] [--lr LR]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Activate venv
source .venv/bin/activate

# Parse args
SKIP_DATA=false
SKIP_BASELINES=false
LEARNING_RATES=("1e-5" "5e-6" "1e-6")
EPOCHS=("1" "3" "5")

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data) SKIP_DATA=true; shift ;;
        --skip-baselines) SKIP_BASELINES=true; shift ;;
        --lr) LEARNING_RATES=("$2"); shift 2 ;;
        --epochs) EPOCHS=("$2"); shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

EXPERIMENT_DIR="artifacts/data/sft_baseline_experiment"
WEIGHTS_DIR="artifacts/weights/sft_baseline_experiment"
RESULTS_CSV="$EXPERIMENT_DIR/results.csv"

mkdir -p "$EXPERIMENT_DIR" "$WEIGHTS_DIR"

echo "============================================="
echo "SFT Baseline Experiment"
echo "============================================="
echo "Repo root: $REPO_ROOT"
echo "Learning rates: ${LEARNING_RATES[*]}"
echo "Epochs: ${EPOCHS[*]}"
echo ""

# ---- Step 1: Prepare SFT datasets ----
if [ "$SKIP_DATA" = false ]; then
    echo "[Step 1] Preparing SFT datasets..."
    python scripts/sft_baseline/prepare_datasets.py \
        --output-dir "$EXPERIMENT_DIR" \
        --num-examples 300 \
        --seed 42
    echo "[Step 1] Done."
else
    echo "[Step 1] Skipping dataset preparation (--skip-data)"
fi

# ---- Step 2: Run baseline evals ----
if [ "$SKIP_BASELINES" = false ]; then
    echo ""
    echo "[Step 2] Running baseline evaluations..."

    echo "  Evaluating LOCKED model (expect ~0% BCB accuracy)..."
    python -m exploration_hacking.scripts.evaluate \
        --config etc/sft_baseline_experiment/eval_locked.yaml
    echo "  Locked baseline done."

    echo "  Evaluating UNLOCKED base model (ceiling)..."
    python -m exploration_hacking.scripts.evaluate \
        --config etc/sft_baseline_experiment/eval_unlocked.yaml
    echo "  Unlocked baseline done."

    echo "[Step 2] Done."
else
    echo "[Step 2] Skipping baselines (--skip-baselines)"
fi

# ---- Step 3: SFT + Eval loop ----
echo ""
echo "[Step 3] Running SFT + Eval grid..."

# Initialize results CSV
echo "dataset,learning_rate,epochs,bcb_test_level_accuracy,bcb_has_python_code,bcb_valid_python_syntax,checkpoint_path" > "$RESULTS_CSV"

DATASETS=("a" "b" "c")
DATASET_NAMES=("exact_bcb" "broader_coding" "generic")

for ds_idx in "${!DATASETS[@]}"; do
    DS="${DATASETS[$ds_idx]}"
    DS_NAME="${DATASET_NAMES[$ds_idx]}"

    for LR in "${LEARNING_RATES[@]}"; do
        for EP in "${EPOCHS[@]}"; do
            RUN_NAME="dataset_${DS}_lr${LR}_ep${EP}"
            echo ""
            echo "  === Training: $RUN_NAME ==="

            # Create per-run config by modifying the base config
            RUN_CONFIG="$EXPERIMENT_DIR/configs/${RUN_NAME}.yaml"
            mkdir -p "$(dirname "$RUN_CONFIG")"

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
            echo "  Training..."
            python scripts/sft_baseline/train_sft_on_locked.py \
                --config "$RUN_CONFIG" || {
                echo "  WARNING: Training failed for $RUN_NAME, skipping..."
                continue
            }

            # Find the checkpoint (wandb run name subdirectory)
            CHECKPOINT_DIR=$(ls -td "$WEIGHTS_DIR/${RUN_NAME}"/*/ 2>/dev/null | head -1)
            if [ -z "$CHECKPOINT_DIR" ]; then
                echo "  WARNING: No checkpoint found for $RUN_NAME, skipping eval..."
                continue
            fi
            echo "  Checkpoint: $CHECKPOINT_DIR"

            # Create eval config for this checkpoint
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
            echo "  Evaluating..."
            python -m exploration_hacking.scripts.evaluate \
                --config "$EVAL_CONFIG" || {
                echo "  WARNING: Eval failed for $RUN_NAME"
                continue
            }

            # Extract results
            python -c "
import pickle, numpy as np, sys
try:
    with open('${EXPERIMENT_DIR}/eval_${RUN_NAME}.pkl', 'rb') as f:
        data = pickle.load(f)
    results = data.get('results', data)
    metrics = results.metrics if hasattr(results, 'metrics') else results['metrics']
    acc = float(np.mean(metrics.get('bcb_test_level_accuracy', [0])))
    code = float(np.mean(metrics.get('bcb_has_python_code', [0])))
    syntax = float(np.mean(metrics.get('bcb_valid_python_syntax', [0])))
    print(f'${DS_NAME},${LR},${EP},{acc:.4f},{code:.4f},{syntax:.4f},${CHECKPOINT_DIR}')
except Exception as e:
    print(f'${DS_NAME},${LR},${EP},ERROR,ERROR,ERROR,${CHECKPOINT_DIR}', file=sys.stderr)
    sys.exit(1)
" >> "$RESULTS_CSV" || echo "${DS_NAME},${LR},${EP},ERROR,ERROR,ERROR,${CHECKPOINT_DIR}" >> "$RESULTS_CSV"

            echo "  === Done: $RUN_NAME ==="
        done
    done
done

echo ""
echo "============================================="
echo "Experiment complete!"
echo "============================================="
echo ""
echo "Results CSV: $RESULTS_CSV"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_CSV"
