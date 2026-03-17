#!/bin/bash
# Benign SFT Elicitation Pipeline
#
# Run from the repo root:
#   bash scripts/dfalck/benign_sft_elicitation/run_pipeline.sh
#
# Adjust GPUS, EVAL_CHECKPOINTS, etc. below as needed.

set -euo pipefail

GPUS="0,1,2,3"
N_GPUS=4
CONFIG_DIR="etc/dfalck/science_conditional/benign_sft_elicitation"
DATA_DIR="artifacts/data/dfalck/science_conditional/benign_sft_elicitation"
WEIGHTS_DIR="artifacts/weights/dfalck/science_conditional/benign_sft_elicitation/sft_model"

# Checkpoints to evaluate (step numbers). Adjust after seeing training length.
# With 2560 examples, ~4 GPUs, batch 4, grad_accum 2 => ~80 steps/epoch
# So 3 epochs ~ 240 steps. Saving every 10.
EVAL_CHECKPOINTS=(10 30 60 120 240)

# ============================================================
# Step 1: Prepare data
# ============================================================
echo "=== Step 1: Preparing data ==="
python scripts/dfalck/benign_sft_elicitation/prepare_data.py \
    --input artifacts/data/dfalck/science_conditional/final/sft_data_benign_with_tool \
    --output "${DATA_DIR}/sft_data" \
    --eval-proportion 0.5 \
    --seed 42

# ============================================================
# Step 2: SFT
# ============================================================
echo "=== Step 2: Running SFT ==="
CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${N_GPUS} \
    scripts/dfalck/benign_sft_elicitation/train_sft.py \
    --config ${CONFIG_DIR}/sft.yaml

# ============================================================
# Step 3: Evaluate at selected checkpoints
# ============================================================
echo "=== Step 3: Evaluating checkpoints ==="

EVAL_DIR="${DATA_DIR}/evals"
mkdir -p "${EVAL_DIR}"

for STEP in "${EVAL_CHECKPOINTS[@]}"; do
    LORA_PATH="./${WEIGHTS_DIR}/checkpoint-${STEP}"

    if [ ! -d "${LORA_PATH}" ]; then
        echo "Skipping checkpoint-${STEP} (not found at ${LORA_PATH})"
        continue
    fi

    echo "--- Evaluating checkpoint-${STEP} on WMDP ---"
    EVAL_CFG=$(mktemp /tmp/eval_wmdp_step${STEP}_XXXX.yaml)
    sed -e "s|__LORA_PATH__|${LORA_PATH}|g" \
        -e "s|__OUTPUT_PATH__|${EVAL_DIR}/eval_wmdp_step${STEP}.pkl|g" \
        -e "s|__RUN_NAME__|benign_sft_wmdp_step${STEP}|g" \
        "${CONFIG_DIR}/eval_wmdp.yaml" > "${EVAL_CFG}"

    python -m exploration_hacking.scripts.evaluate --config "${EVAL_CFG}"

    echo "--- Evaluating checkpoint-${STEP} on VCT ---"
    EVAL_CFG=$(mktemp /tmp/eval_vct_step${STEP}_XXXX.yaml)
    sed -e "s|__LORA_PATH__|${LORA_PATH}|g" \
        -e "s|__OUTPUT_PATH__|${EVAL_DIR}/eval_vct_step${STEP}.pkl|g" \
        -e "s|__RUN_NAME__|benign_sft_vct_step${STEP}|g" \
        "${CONFIG_DIR}/eval_vct.yaml" > "${EVAL_CFG}"

    python -m exploration_hacking.scripts.evaluate --config "${EVAL_CFG}"
done

echo "=== Done! Eval results in ${EVAL_DIR}/ ==="
