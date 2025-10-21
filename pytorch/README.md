# PyTorch T5 MLM Pretraining for TPU/GPU

This directory contains PyTorch versions of the T5 Masked Language Model pretraining script, converted from the original Flax implementation.

## Available Scripts

### 1. `run_t5_mlm_to_pretrain.py` - Direct torch_xla Version
This version uses `torch_xla` directly for TPU training.

**Pros:**
- Direct control over TPU-specific operations
- Optimized for TPU performance
- Lower-level control

**Cons:**
- TPU-specific code (not easily portable to GPU)
- More boilerplate code
- Requires manual distributed training setup

**Use when:**
- You're running exclusively on TPU
- You need fine-grained control over TPU operations
- You want maximum TPU performance optimization

### 2. `run_t5_mlm_to_pretrain_accelerate.py` - Accelerate Version ⭐ RECOMMENDED
This version uses Hugging Face Accelerate library for unified GPU/TPU/multi-GPU support.

**Pros:**
- ✅ **Works on both GPU and TPU** with the same code
- ✅ **Minimal code changes** - Accelerate handles device placement automatically
- ✅ **Easy distributed training** - Works on single GPU, multi-GPU, or TPU
- ✅ **Mixed precision** - Built-in support for fp16/bf16
- ✅ **Less boilerplate** - Cleaner, more maintainable code
- ✅ **Better logging** - Integrated with Accelerate's logging system

**Cons:**
- Slight abstraction overhead (negligible in practice)

**Use when:**
- You want code that works on both GPU and TPU
- You're developing on GPU but deploying on TPU
- You want simpler, more maintainable code
- You need easy switching between different hardware setups

## Installation

### For TPU (torch_xla version):
```bash
pip install torch torch_xla datasets transformers huggingface_hub
```

### For GPU/TPU (Accelerate version - RECOMMENDED):
```bash
pip install torch datasets transformers huggingface_hub accelerate
```

For TPU with Accelerate, also install:
```bash
pip install torch_xla
```

## Usage

### Using the Accelerate Version (Recommended)

#### Single GPU:
```bash
python run_t5_mlm_to_pretrain_accelerate.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --logging_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000
```

#### Multi-GPU:
```bash
accelerate config  # Run once to configure

accelerate launch run_t5_mlm_to_pretrain_accelerate.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512
```

#### TPU (v3-8 or v4-8):
```bash
accelerate config  # Select TPU when prompted

accelerate launch run_t5_mlm_to_pretrain_accelerate.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512
```

#### With Mixed Precision (faster training):
```bash
accelerate launch run_t5_mlm_to_pretrain_accelerate.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --mixed_precision bf16 \
    --per_device_train_batch_size 16
```

### Using the Direct torch_xla Version

#### TPU Only:
```bash
python run_t5_mlm_to_pretrain.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512
```

## Configuration with JSON

You can also use a JSON configuration file:

```json
{
    "model_name_or_path": "t5-small",
    "dataset_name": "wikitext",
    "dataset_config_name": "wikitext-2-raw-v1",
    "do_train": true,
    "do_eval": true,
    "output_dir": "./output",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "max_seq_length": 512,
    "mlm_probability": 0.15,
    "mean_noise_span_length": 3.0,
    "logging_steps": 100,
    "save_steps": 1000,
    "eval_steps": 1000
}
```

Then run:
```bash
# Accelerate version
accelerate launch run_t5_mlm_to_pretrain_accelerate.py config.json

# Direct torch_xla version
python run_t5_mlm_to_pretrain.py config.json
```

## Key Arguments

### Model Arguments
- `--model_name_or_path`: Pretrained model checkpoint (e.g., `t5-small`, `t5-base`)
- `--config_name`: Config file path (if different from model)
- `--tokenizer_name`: Tokenizer path (if different from model)
- `--dtype`: Model dtype (`float32`, `float16`, `bfloat16`)

### Data Arguments
- `--dataset_name`: HuggingFace dataset name
- `--dataset_config_name`: Dataset configuration
- `--train_file`: Local training file (`.txt`, `.json`, `.csv`)
- `--validation_file`: Local validation file
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--mlm_probability`: Masking probability (default: 0.15)
- `--mean_noise_span_length`: Average span length for masking (default: 3.0)

### Training Arguments
- `--output_dir`: Output directory for checkpoints
- `--per_device_train_batch_size`: Batch size per device
- `--per_device_eval_batch_size`: Eval batch size per device
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs
- `--warmup_steps`: Warmup steps for learning rate
- `--logging_steps`: Log every N steps
- `--save_steps`: Save checkpoint every N steps
- `--eval_steps`: Run evaluation every N steps
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--max_grad_norm`: Max gradient norm for clipping (default: 1.0)
- `--mixed_precision`: Mixed precision (`no`, `fp16`, `bf16`) - Accelerate only

### Hub Arguments
- `--push_to_hub`: Push model to HuggingFace Hub after training
- `--push_to_hub_final_step`: Push only final checkpoint
- `--hub_model_id`: Model repository name
- `--hub_token`: HuggingFace token

## Example: Training from Scratch

```bash
# Create a config file for a small T5 model
cat > small_t5_config.json << EOF
{
    "model_type": "t5",
    "d_model": 512,
    "d_ff": 2048,
    "num_layers": 6,
    "num_heads": 8,
    "vocab_size": 32128
}
EOF

# Train from scratch
accelerate launch run_t5_mlm_to_pretrain_accelerate.py \
    --config_name small_t5_config.json \
    --tokenizer_name t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --do_train \
    --output_dir ./my_t5_model \
    --per_device_train_batch_size 16 \
    --num_train_epochs 10 \
    --mixed_precision bf16
```

## Differences from Flax Version

### Key Changes:
1. **Framework**: JAX/Flax → PyTorch
2. **Model**: `FlaxT5ForConditionalGeneration` → `T5ForConditionalGeneration`
3. **Optimizer**: `optax` → PyTorch `AdamW`
4. **Distributed**: `jax.pmap` → Accelerate (or `torch_xla`)
5. **Device**: Auto-handled by Accelerate (or manual with torch_xla)

### Same Features:
- T5 span-masked language modeling
- Same masking strategy and data collation
- Same training arguments
- Same dataset preprocessing
- HuggingFace Hub integration

## Performance Tips

### For TPU:
- Use larger batch sizes (TPUs work best with large batches)
- Use `bfloat16` mixed precision
- Recommended batch size: 32-128 per core

### For GPU:
- Adjust batch size based on GPU memory
- Use `fp16` or `bf16` for newer GPUs
- Use gradient accumulation if batch size is limited

### General:
- Use `--gradient_accumulation_steps` to simulate larger batch sizes
- Increase `--max_seq_length` for better model performance
- Use `--preprocessing_num_workers` to speed up data loading

## Troubleshooting

### Out of Memory (OOM):
- Reduce `--per_device_train_batch_size`
- Increase `--gradient_accumulation_steps`
- Use mixed precision (`--mixed_precision bf16` or `fp16`)

### Slow Training:
- Increase batch size
- Enable mixed precision
- Use more `--preprocessing_num_workers`

### TPU Not Detected:
```bash
# Check TPU availability
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"

# For Accelerate, run config
accelerate config
```

## Comparison Summary

| Feature | torch_xla Version | Accelerate Version |
|---------|-------------------|-------------------|
| GPU Support | ❌ No | ✅ Yes |
| TPU Support | ✅ Yes | ✅ Yes |
| Multi-GPU | ❌ Complex | ✅ Easy |
| Code Simplicity | Medium | ✅ High |
| Hardware Portability | Low | ✅ High |
| Performance | ✅ Optimal for TPU | Good on all |
| Mixed Precision | Manual | ✅ Built-in |
| Recommended for | TPU-only workloads | ✅ All scenarios |

## Conclusion

**Use `run_t5_mlm_to_pretrain_accelerate.py`** for most use cases. It provides the best flexibility, works on both GPU and TPU, and requires minimal code changes when switching hardware.

**Use `run_t5_mlm_to_pretrain.py`** only if you need maximum TPU optimization and won't be using GPUs.
