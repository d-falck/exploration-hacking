set -xe

export PYTHONPATH="/exploration_hacking/src/exploration_hacking"
python3 -m src.exploration_hacking.environments.kernelbench.env
