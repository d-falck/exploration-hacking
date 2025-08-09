# Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Setup git:
git config --global user.name "Damon Falck"
git config --global user.email "damon.falck@gmail.com"
apt update
apt -y install gh

# Install other tools
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
echo 'export HF_TOKEN=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' >> ~/.bashrc

echo 'export LANG=C.UTF-8' >> ~/.bashrc
echo 'export LC_CTYPE=$LANG' >> ~/.bashrc

echo "set -g mouse on" >> ~/.tmux.conf

# Still will need to do:
# gh auth login
# claude (to auth the first time)
# uv install whatever you need
