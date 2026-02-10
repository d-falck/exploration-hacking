#!/bin/bash
#SBATCH --job-name=sft_prep_data
#SBATCH --output=logs/sft_baseline/prep_data_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

echo "========================================================"
echo "SFT Baseline: Prepare Datasets"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | Started: $(date)"
echo "========================================================"

cd /mnt/nw/teams/team_emmons_lindner_jenner_c9/exploration-hacking
source .venv/bin/activate

python scripts/sft_baseline/prepare_datasets.py \
    --output-dir artifacts/data/sft_baseline_experiment \
    --num-examples 300 \
    --seed 42

echo "Data prep finished at $(date)"
