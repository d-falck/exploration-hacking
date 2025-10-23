set -xe

uv pip install hf_transfer

export PYTHONPATH="/exploration_hacking/src/exploration_hacking"
# python3 -m src.exploration_hacking.environments.kernelbench.env

#python -m exploration_hacking.scripts.evaluate --config etc/kernelbench/eval.yaml
python -m exploration_hacking.scripts.grpo --config etc/kernelbench/rl.yaml
