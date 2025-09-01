# Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Setup uv completion:
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

# Setup git:
git config --global user.name "Damon Falck"
git config --global user.email "damon.falck@gmail.com"

# Github cli

(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

apt update
apt -y install gh

# Install other tools
apt -y install git-lfs
apt -y install tree
apt -y install nano
apt -y install htop
apt -y install jq
apt -y install tmux

# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 22

# Verify the Node.js version:
node -v # Should print "v22.17.0".
nvm current # Should print "v22.17.0".

# Verify npm version:
npm -v # Should print "10.9.2".

# Install claude-code:
npm install -g @anthropic-ai/claude-code

# Add HF_HOME to bashrc
echo 'export HF_HOME=/workspace/cache' >> ~/.bashrc

echo 'export LANG=C.UTF-8' >> ~/.bashrc
echo 'export LC_CTYPE=$LANG' >> ~/.bashrc

echo "set -g mouse on" >> ~/.tmux.conf

# Still will need to do:
# gh auth login
# claude (to auth the first time)
# uv install whatever you need
