# VERL GRPO Implementation

A simplified implementation of GRPO training using VERL instead of TRL/Unsloth.

## Files

- `model_verl.py` - Model loading with LoRA
- `rl_verl.py` - GRPO configuration
- `grpo_verl.py` - Main training script
- `data_verl.py` - Data utilities
- `verl.yaml` - Configuration

## Installation

```bash
pip install "verl[vllm]"
```

## Usage

```bash
python -m exploration_hacking.scripts.grpo_verl --config etc/evalplus_grpo/verl.yaml
```

## Key Differences from TRL/Unsloth

1. **GRPO Native Support**: `num_rollouts` controls group size
2. **Data Format**: Requires Parquet files
3. **Training**: Calls VERL's `main_ppo` with GRPO config
4. **Performance**: Uses vLLM for faster generation

## Configuration

The main GRPO-specific parameters:
- `num_rollouts`: Number of samples per prompt (group size)
- `kl_loss_coef`: KL divergence coefficient (required for GRPO)
- `temperature`, `top_p`: Sampling parameters

## How It Works

1. Load dataset and convert to Parquet
2. Create VERL config with GRPO parameters
3. Run VERL training via subprocess
4. Save logs and checkpoints

The implementation prioritizes simplicity while maintaining compatibility with existing reward functions and datasets.