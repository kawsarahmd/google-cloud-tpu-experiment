git clone https://github.com/vllm-project/vllm.git && cd vllm
pip uninstall torch torch-xla -y
pip install -r requirements/tpu.txt

sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev
VLLM_TARGET_DEVICE="tpu" python -m pip install -e .

pip install "transformers[sentencepiece]"
pip install datasets
pip install accelerate

pip install evaluate

pip install nltk

sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev


pip install optax

pip install flax

pip install rouge_score
