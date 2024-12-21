#!/bin/bash

# Update pip to the latest version
# pip install --upgrade pip

# Install tensorboard
echo "Installing TensorBoard..."
pip install tensorboard

# Install PyTorch, TorchXLA, and TorchVision
echo "Installing PyTorch, TorchXLA, and TorchVision..."
pip3 install torch~=2.4.0 torch_xla[tpu]~=2.4.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html

# Install transformers, datasets, evaluate, scikit-learn, and sentencepiece
echo "Installing transformers, datasets, evaluate, scikit-learn, and sentencepiece..."
pip3 install transformers datasets evaluate scikit-learn sentencepiece

# Install the normalizer from GitHub
echo "Installing the normalizer from GitHub..."
pip3 install git+https://github.com/csebuetnlp/normalizer

# Install Huggingface Hub
echo "Installing Huggingface Hub..."
pip install huggingface_hub


echo "Installing Jax[TPU] & Flax..."
# Install JAX with TPU support
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Upgrade the clu package
pip install --upgrade clu

# Install TensorFlow CPU version only
pip install tensorflow-cpu

# Install TensorFlow Datasets
pip install tensorflow_datasets

# Clone the Flax repository from GitHub
git clone https://github.com/google/flax.git

# Install Flax for the user
pip install --user flax

# Install NLTK
pip install nltk
# Install TF-Keras
pip install tf-keras
# Install Rouge Score
pip install rouge_score

echo "Installing tpu-info..."
pip install tpu-info

# Configure git to store credentials globally
echo "Configuring git to store credentials globally..."
git config --global credential.helper store

echo "Installing byobu"
sudo apt-get install byobu


echo "Installing GCSFUSE"

export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse

# mount command below
# gcsfuse bangla_pretrained_v2 ~/.cache/huggingface/
# unmount command below
# fusermount -u /home/kawsa/.cache/huggingface

echo "Installing twilio client"
pip install twilio
pip install unicodeconverter
pip install --upgrade gradio

echo "All installations completed successfully."
