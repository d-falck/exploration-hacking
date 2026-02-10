#!/bin/bash
#SBATCH --job-name=sft_eval_base
#SBATCH --output=logs/sft_baseline/eval_baselines_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

echo "========================================================"
echo "SFT Baseline: Evaluate Locked + Unlocked Baselines"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "========================================================"

cd /mnt/nw/teams/team_emmons_lindner_jenner_c9/exploration-hacking
source .venv/bin/activate

# Unset SLURM's ENVIRONMENT variable — it clashes with pydantic-settings
# parsing of the "environment" config field
unset ENVIRONMENT

nvidia-smi

echo ""
echo "=== Evaluating LOCKED model (expect ~0% BCB accuracy) ==="
python -m exploration_hacking.scripts.evaluate \
    --config etc/sft_baseline_experiment/eval_locked.yaml

echo ""
echo "=== Evaluating UNLOCKED base model (ceiling) ==="
python -m exploration_hacking.scripts.evaluate \
    --config etc/sft_baseline_experiment/eval_unlocked.yaml

echo ""
echo "Baseline evals finished at $(date)"
