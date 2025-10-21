# PyTorch T5 for TPU/GPU

This directory contains PyTorch versions of T5 training scripts (pretraining and fine-tuning), converted from the original Flax implementation with support for both GPU and TPU.

## Available Scripts

### Pretraining Scripts

#### 1. `run_t5_mlm_to_pretrain.py` - Direct torch_xla Version
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

#### 2. `run_t5_mlm_to_pretrain_accelerate.py` - Accelerate Version ⭐ RECOMMENDED
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

#### 3. `run_t5_mlm_pretrain_simple_accelerate_hf.py` - Simplified Accelerate Version ⭐⭐ HIGHLY RECOMMENDED
Ultra-simplified version using HuggingFace's built-in `DataCollatorForT5MLM` class with Accelerate.

**Pros:**
- ✅ **Much simpler code** (~300 lines vs 1000 lines) - No custom data collator!
- ✅ **Uses built-in HuggingFace classes** - Well-tested and maintained
- ✅ **Works on both GPU and TPU** with the same code
- ✅ **Easy distributed training** - Accelerate handles everything
- ✅ **Mixed precision** - Built-in support for fp16/bf16
- ✅ **Production-ready** - Cleaner, more maintainable

**Cons:**
- Slightly less customizable than version 2 (but covers 99% of use cases)

**Use when:**
- ✅ **Best choice for most users!** - Simplest production-ready code
- You want to use well-tested HuggingFace components
- You want code that's easy to understand and maintain
- You need GPU/TPU flexibility without complexity

#### 4. `run_t5_mlm_pretrain_simple_trainer_hf.py` - Ultra-Simplified Trainer Version 🚀 EASIEST
The simplest possible implementation using HuggingFace's `Trainer` API.

**Pros:**
- ✅ **Ultra-simple code** (~150 lines total!)
- ✅ **Trainer handles everything** - Training loop, eval, checkpointing, logging
- ✅ **Automatic distributed training** - Multi-GPU/TPU handled automatically
- ✅ **Built-in integrations** - W&B, TensorBoard, MLflow support
- ✅ **Best for beginners** - Minimal code, maximum functionality

**Cons:**
- ⚠️ **Less control** - Trainer abstracts away the training loop
- ⚠️ **TPU support** - Requires additional configuration for TPU

**Use when:**
- You want the absolute simplest code possible
- You're fine with Trainer's abstractions
- You need quick experiments or prototyping
- You want automatic integration with experiment tracking tools

### Fine-tuning Script

#### `run_seq_to_seq_model_to_finetune.py` - Accelerate Version ⭐ RECOMMENDED
Fine-tuning script for sequence-to-sequence tasks (e.g., summarization, translation) using Hugging Face Accelerate.

**Features:**
- ✅ **Works on both GPU and TPU** with the same code
- ✅ **ROUGE metrics** evaluation for summarization
- ✅ **Text generation** during evaluation
- ✅ **Label smoothing** support
- ✅ **Gradient checkpointing** for memory efficiency
- ✅ **Flexible datasets** - works with HuggingFace datasets or custom CSV/JSON files

**Supported Tasks:**
- Summarization (CNN/DailyMail, XSum, etc.)
- Translation
- Any seq2seq task

**Use when:**
- You want to fine-tune T5, BART, mT5, or other seq2seq models
- You need to evaluate with ROUGE metrics
- You want to switch easily between GPU and TPU
- You need production-ready fine-tuning code

## Installation

### For TPU (torch_xla version):
```bash
pip install torch torch_xla datasets transformers huggingface_hub
```

### For GPU/TPU (Accelerate version - RECOMMENDED):
```bash
pip install torch datasets transformers huggingface_hub accelerate evaluate rouge_score nltk
```

For TPU with Accelerate, also install:
```bash
pip install torch_xla
```

**Note:** The fine-tuning script requires `evaluate`, `rouge_score`, and `nltk` for ROUGE metrics.

## Usage

### Pretraining (MLM)

#### Using the Accelerate Version (Recommended)

##### Single GPU:
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

##### Multi-GPU:
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

##### TPU (v3-8 or v4-8):
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

##### With Mixed Precision (faster training):
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

#### Using the Direct torch_xla Version

##### TPU Only:
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

#### Using Simplified Accelerate Version (Recommended for Most Users!)

The simplified version works exactly the same as version 2 but uses built-in HuggingFace classes:

```bash
# Same usage as version 2!
accelerate launch run_t5_mlm_pretrain_simple_accelerate_hf.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --mlm_probability 0.15 \
    --mean_noise_span_length 3.0 \
    --logging_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000
```

**Benefits:**
- ✅ Much simpler code (~300 lines vs 1000 lines)
- ✅ Uses well-tested HuggingFace `DataCollatorForT5MLM`
- ✅ Same performance and functionality
- ✅ Works on GPU and TPU

#### Using Ultra-Simplified Trainer Version (Easiest!)

The Trainer version is the simplest - just specify arguments:

```bash
# For single GPU/CPU - just run directly
python run_t5_mlm_pretrain_simple_trainer_hf.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --mlm_probability 0.15 \
    --mean_noise_span_length 3.0 \
    --logging_steps 100 \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_total_limit 3
```

**For multi-GPU:**
```bash
# Trainer automatically handles multi-GPU!
python -m torch.distributed.launch --nproc_per_node=4 \
    run_t5_mlm_pretrain_simple_trainer_hf.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 8
```

**Benefits:**
- ✅ Absolutely simplest code (~150 lines!)
- ✅ Trainer handles training loop, eval, checkpointing automatically
- ✅ Built-in integration with W&B, TensorBoard
- ✅ Perfect for quick experiments

### Fine-tuning (Seq2Seq)

#### Summarization Example - XSum Dataset

##### Single GPU:
```bash
accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path t5-small \
    --dataset_name xsum \
    --text_column document \
    --summary_column summary \
    --source_prefix "summarize: " \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --output_dir ./t5-small-xsum \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_beams 4 \
    --logging_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000
```

##### TPU (v3-8 or v4-8):
```bash
accelerate config  # Select TPU when prompted

accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path t5-base \
    --dataset_name xsum \
    --text_column document \
    --summary_column summary \
    --source_prefix "summarize: " \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --output_dir ./t5-base-xsum \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_beams 4 \
    --mixed_precision bf16
```

##### With Gradient Checkpointing (for large models):
```bash
accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path t5-large \
    --dataset_name xsum \
    --text_column document \
    --summary_column summary \
    --source_prefix "summarize: " \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --output_dir ./t5-large-xsum \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --mixed_precision bf16 \
    --num_train_epochs 3
```

#### CNN/DailyMail Dataset:
```bash
accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --source_prefix "summarize: " \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --output_dir ./t5-small-cnn \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 4
```

#### Custom Dataset (CSV/JSON):
```bash
# Your data should have two columns: one for source text, one for summary
accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path t5-small \
    --train_file ./data/train.json \
    --validation_file ./data/val.json \
    --text_column text \
    --summary_column summary \
    --source_prefix "summarize: " \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --output_dir ./t5-custom \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --max_source_length 512 \
    --max_target_length 128
```

#### With Prediction on Test Set:
```bash
accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path ./t5-small-xsum \
    --dataset_name xsum \
    --text_column document \
    --summary_column summary \
    --do_predict \
    --predict_with_generate \
    --output_dir ./predictions \
    --per_device_eval_batch_size 16 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_beams 4
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

### Model Arguments (Both Scripts)
- `--model_name_or_path`: Pretrained model checkpoint (e.g., `t5-small`, `t5-base`, `t5-large`)
- `--config_name`: Config file path (if different from model)
- `--tokenizer_name`: Tokenizer path (if different from model)
- `--dtype`: Model dtype (`float32`, `float16`, `bfloat16`)

### Data Arguments - Pretraining
- `--dataset_name`: HuggingFace dataset name
- `--dataset_config_name`: Dataset configuration
- `--train_file`: Local training file (`.txt`, `.json`, `.csv`)
- `--validation_file`: Local validation file
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--mlm_probability`: Masking probability (default: 0.15)
- `--mean_noise_span_length`: Average span length for masking (default: 3.0)

### Data Arguments - Fine-tuning
- `--dataset_name`: HuggingFace dataset name (e.g., `xsum`, `cnn_dailymail`)
- `--dataset_config_name`: Dataset configuration
- `--train_file` / `--validation_file` / `--test_file`: Local data files (`.json`, `.csv`)
- `--text_column`: Column name for input text (default: auto-detected)
- `--summary_column`: Column name for summaries (default: auto-detected)
- `--source_prefix`: Prefix for input text (e.g., `"summarize: "` for T5)
- `--max_source_length`: Max input length (default: 1024)
- `--max_target_length`: Max target length (default: 128)
- `--val_max_target_length`: Max target length for validation
- `--predict_with_generate`: Use generation for evaluation (enables ROUGE metrics)
- `--num_beams`: Number of beams for generation (default: 1)

### Training Arguments (Both Scripts)
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
- `--gradient_checkpointing`: Enable gradient checkpointing (saves memory)
- `--label_smoothing_factor`: Label smoothing (default: 0.0) - Fine-tuning only

### Hub Arguments (Both Scripts)
- `--push_to_hub`: Push model to HuggingFace Hub after training
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

## Comparison Summary - Pretraining Scripts

| Feature | V1: torch_xla | V2: Accelerate (Custom) | V3: Accelerate (Simple) ⭐⭐ | V4: Trainer 🚀 |
|---------|---------------|------------------------|---------------------------|----------------|
| **Lines of Code** | ~1000 | ~1000 | ~300 | ~150 |
| **GPU Support** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **TPU Support** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Requires config |
| **Multi-GPU** | ❌ Complex | ✅ Easy | ✅ Easy | ✅ Automatic |
| **Code Simplicity** | Medium | Medium | ✅✅ Very High | ✅✅✅ Extremely High |
| **Hardware Portability** | Low | ✅ High | ✅ High | ✅ High |
| **Uses Built-in HF Classes** | ❌ No | ❌ Custom | ✅✅ Yes | ✅✅ Yes |
| **Training Loop Control** | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited |
| **Mixed Precision** | Manual | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| **Best For** | TPU-only | Production (custom) | ✅ **Most users** | Quick experiments |
| **Maintainability** | Medium | Medium | ✅✅ High | ✅✅ High |

### Which Version Should You Use?

#### 🏆 For Most Users: **Version 3** (`run_t5_mlm_pretrain_simple_accelerate_hf.py`)
- ✅ Best balance of simplicity and control
- ✅ Uses well-tested HuggingFace `DataCollatorForT5MLM`
- ✅ Works on GPU and TPU seamlessly
- ✅ Production-ready, clean code
- ✅ Easy to understand and maintain

#### 🚀 For Beginners/Quick Experiments: **Version 4** (`run_t5_mlm_pretrain_simple_trainer_hf.py`)
- ✅ Absolutely simplest code
- ✅ Trainer handles everything automatically
- ✅ Perfect for quick prototyping
- ⚠️ Less control over training loop

#### 🔧 For Advanced Users Needing Customization: **Version 2** (`run_t5_mlm_to_pretrain_accelerate.py`)
- ✅ Full control over data collation logic
- ✅ Easier to customize masking strategy
- ✅ Good for research and experimentation

#### 🎯 For TPU-Only Production: **Version 1** (`run_t5_mlm_to_pretrain.py`)
- ✅ Maximum TPU optimization
- ⚠️ No GPU support
- ⚠️ More complex code

## Conclusion

**RECOMMENDED:** Start with **Version 3** (`run_t5_mlm_pretrain_simple_accelerate_hf.py`) for production use. It's the best balance of simplicity, flexibility, and maintainability.

**For quick experiments:** Use **Version 4** (`run_t5_mlm_pretrain_simple_trainer_hf.py`) - the Trainer API makes it incredibly easy to get started.
