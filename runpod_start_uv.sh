#!/bin/bash

# Run the original setup script
bash /workspace/exploration-hacking/runpod_start.sh

# Set default SSH directory to /workspace/exploration-hacking
echo 'cd /workspace/exploration-hacking' >> ~/.bashrc

# Install exploration-hacking package with evalplus and grpo extras
cd /workspace/exploration-hacking
source $HOME/.local/bin/env
uv pip install -e .[evalplus,grpo] --system