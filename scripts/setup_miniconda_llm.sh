#!/bin/bash
set -e

sudo apt-get update -y

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install Miniconda silently
bash ~/miniconda.sh -b -p $HOME/miniconda

# Initialize conda for bash
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

# Disable base auto-activation
conda config --set auto_activate_base false

# Accept Terms of Service for required channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create new environment 'llm' with Python 3.12
conda create -n llm python=3.12 -y

# Add conda initialization + auto-activation of llm into .bashrc
cat << 'EOF' >> ~/.bashrc

# >>> conda initialize >>>
__conda_setup="$('$HOME/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize >>>

# Auto-activate llm environment
conda activate llm
EOF

# Reload bashrc so env activates right away
source ~/.bashrc

echo "✅ Miniconda installed, TOS accepted, 'llm' environment created, and auto-activated."
