# PyTorch Training Scripts for TPU/GPU

Universal PyTorch training scripts with GPU and TPU support using HuggingFace Transformers and Accelerate.

## 🚀 Quick Reference Guide

### Which Script Should I Use?

| Your Task | Model Type | Script to Use | Example Models |
|-----------|-----------|---------------|----------------|
| **Pretraining** | T5-style (span corruption) | `pretrain_t5_style.py` | T5, mT5, BART, mBART, ByT5 |
| **Pretraining** | BERT-style (masked LM) | `pretrain_bert_style.py` | BERT, RoBERTa, DistilBERT, ALBERT, ELECTRA, DeBERTa |
| **Pretraining** | GPT-style (causal LM) | `pretrain_gpt_style.py` | GPT-2, GPT-Neo, GPT-J, LLaMA, OPT, BLOOM |
| **Fine-tuning** | Classification | `finetune_classification.py` | Any encoder model (BERT, RoBERTa, etc.) |
| **Fine-tuning** | Text Generation | `finetune_generation.py` | GPT-2, GPT-Neo, LLaMA, OPT |
| **Fine-tuning** | Seq2Seq (summarization, translation) | `run_seq_to_seq_model_to_finetune.py` | T5, BART, mT5, mBART |

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Universal Scripts (Recommended)](#universal-scripts-recommended)
- [T5-Specific Scripts](#t5-specific-scripts-legacy)
- [Installation](#installation)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Usage Examples](#usage-examples)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## ✨ Quick Start

### 1. Install Dependencies
```bash
pip install torch datasets transformers accelerate evaluate
```

### 2. Run Pretraining
```bash
# BERT-style pretraining
accelerate launch pretrain_bert_style.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1

# GPT-style pretraining
accelerate launch pretrain_gpt_style.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1

# T5-style pretraining
accelerate launch pretrain_t5_style.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1
```

### 3. Run Fine-tuning
```bash
# Classification
accelerate launch finetune_classification.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name imdb

# Generation
accelerate launch finetune_generation.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1
```

## 🎯 Universal Scripts (Recommended)

These are the **recommended scripts for most users**. They use built-in HuggingFace classes, are simple to use, and work on both GPU and TPU.

### Pretraining Scripts

#### `pretrain_t5_style.py` - T5-Style Span Corruption Pretraining ⭐

**For models:** T5, mT5, BART, mBART, ByT5

**What it does:** Trains models using span corruption (masks consecutive tokens and predicts them)

**Key features:**
- ✅ Uses built-in `DataCollatorForT5MLM`
- ✅ Works on GPU and TPU with Accelerate
- ✅ ~300 lines of simple, clean code
- ✅ Supports all T5-style encoder-decoder models

**Usage:**
```bash
# T5
accelerate launch pretrain_t5_style.py --model_name_or_path t5-small

# BART
accelerate launch pretrain_t5_style.py --model_name_or_path facebook/bart-base

# mT5
accelerate launch pretrain_t5_style.py --model_name_or_path google/mt5-small
```

#### `pretrain_bert_style.py` - BERT-Style Masked Language Modeling ⭐

**For models:** BERT, RoBERTa, DistilBERT, ALBERT, ELECTRA, DeBERTa

**What it does:** Trains models using masked language modeling (masks random tokens and predicts them)

**Key features:**
- ✅ Uses built-in `DataCollatorForLanguageModeling(mlm=True)`
- ✅ Works on GPU and TPU with Accelerate
- ✅ ~300 lines of simple, clean code
- ✅ Supports all BERT-style encoder-only models

**Usage:**
```bash
# BERT
accelerate launch pretrain_bert_style.py --model_name_or_path bert-base-uncased

# RoBERTa
accelerate launch pretrain_bert_style.py --model_name_or_path roberta-base

# DistilBERT
accelerate launch pretrain_bert_style.py --model_name_or_path distilbert-base-uncased
```

#### `pretrain_gpt_style.py` - GPT-Style Causal Language Modeling ⭐

**For models:** GPT-2, GPT-Neo, GPT-J, LLaMA, OPT, BLOOM

**What it does:** Trains models using causal language modeling (predicts next token)

**Key features:**
- ✅ Uses built-in `DataCollatorForLanguageModeling(mlm=False)`
- ✅ Works on GPU and TPU with Accelerate
- ✅ ~300 lines of simple, clean code
- ✅ Supports all GPT-style decoder-only models

**Usage:**
```bash
# GPT-2
accelerate launch pretrain_gpt_style.py --model_name_or_path gpt2

# GPT-Neo
accelerate launch pretrain_gpt_style.py --model_name_or_path EleutherAI/gpt-neo-125M

# LLaMA (if you have access)
accelerate launch pretrain_gpt_style.py --model_name_or_path meta-llama/Llama-2-7b-hf
```

### Fine-tuning Scripts

#### `finetune_classification.py` - Universal Text Classification ⭐

**For tasks:** Sentiment analysis, topic classification, intent classification, etc.

**For models:** BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, ELECTRA, etc.

**Key features:**
- ✅ Uses built-in `DataCollatorWithPadding`
- ✅ Automatic accuracy evaluation
- ✅ Works on GPU and TPU with Accelerate
- ✅ Simple, clean code

**Usage:**
```bash
# Sentiment analysis with BERT
accelerate launch finetune_classification.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name imdb \
    --text_column text \
    --label_column label

# Topic classification with RoBERTa
accelerate launch finetune_classification.py \
    --model_name_or_path roberta-base \
    --dataset_name ag_news \
    --text_column text \
    --label_column label \
    --num_labels 4
```

#### `finetune_generation.py` - Universal Text Generation Fine-tuning ⭐

**For tasks:** Domain-specific text generation, style transfer, etc.

**For models:** GPT-2, GPT-Neo, GPT-J, LLaMA, OPT, BLOOM

**Key features:**
- ✅ Uses built-in `DataCollatorForLanguageModeling(mlm=False)`
- ✅ Perplexity evaluation
- ✅ Works on GPU and TPU with Accelerate
- ✅ Simple, clean code

**Usage:**
```bash
# Fine-tune GPT-2 on custom dataset
accelerate launch finetune_generation.py \
    --model_name_or_path gpt2 \
    --dataset_name your_dataset \
    --text_column text

# Fine-tune LLaMA on domain-specific data
accelerate launch finetune_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name domain_dataset \
    --text_column text \
    --max_seq_length 2048
```

#### `run_seq_to_seq_model_to_finetune.py` - Seq2Seq Fine-tuning ⭐

**For tasks:** Summarization, translation, question answering

**For models:** T5, BART, mT5, mBART

**Key features:**
- ✅ Uses built-in `DataCollatorForSeq2Seq`
- ✅ ROUGE metrics evaluation
- ✅ Text generation during evaluation
- ✅ Works on GPU and TPU with Accelerate

**Usage:**
```bash
# Summarization with T5
accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path t5-small \
    --dataset_name xsum \
    --text_column document \
    --summary_column summary \
    --source_prefix "summarize: " \
    --do_train \
    --do_eval \
    --predict_with_generate

# Translation with mT5
accelerate launch run_seq_to_seq_model_to_finetune.py \
    --model_name_or_path google/mt5-small \
    --dataset_name wmt16 \
    --dataset_config_name de-en \
    --source_prefix "translate German to English: " \
    --do_train \
    --do_eval
```

## 📖 T5-Specific Scripts (Legacy)

These scripts are T5-specific versions with various levels of complexity. For new projects, use the universal scripts above.

### Pretraining Scripts

#### `run_t5_mlm_pretrain_simple_accelerate_hf.py` - ⭐⭐ RECOMMENDED
Simplified T5 pretraining using built-in HuggingFace classes (~300 lines).

**Best for:** Most T5 pretraining use cases

#### `run_t5_mlm_pretrain_simple_trainer_hf.py` - 🚀 EASIEST
Ultra-simplified T5 pretraining using Trainer API (~150 lines).

**Best for:** Quick experiments and prototyping

#### `run_t5_mlm_to_pretrain_accelerate.py`
T5 pretraining with custom data collator and Accelerate (~1000 lines).

**Best for:** Advanced users needing full control

#### `run_t5_mlm_to_pretrain.py`
Direct torch_xla implementation for TPU-only training (~1000 lines).

**Best for:** TPU-only production environments

For detailed information about these scripts, see the [T5-Specific Documentation](#t5-specific-detailed-documentation) section below.

## 🛠️ Installation

### Basic Installation (GPU/CPU)
```bash
pip install torch datasets transformers accelerate evaluate
```

### For TPU Support
```bash
pip install torch datasets transformers accelerate evaluate torch_xla
```

### Additional Dependencies for Specific Tasks
```bash
# For seq2seq tasks (ROUGE metrics)
pip install rouge_score nltk

# For pushing to HuggingFace Hub
pip install huggingface_hub
```

## 📋 Step-by-Step Workflow

### 1. Choose Your Task and Model

**Pretraining:** Start from scratch or continue pretraining
- T5-style: `pretrain_t5_style.py`
- BERT-style: `pretrain_bert_style.py`
- GPT-style: `pretrain_gpt_style.py`

**Fine-tuning:** Adapt pretrained model to your task
- Classification: `finetune_classification.py`
- Generation: `finetune_generation.py`
- Seq2seq: `run_seq_to_seq_model_to_finetune.py`

### 2. Prepare Your Dataset

**Using HuggingFace Datasets:**
```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = load_dataset("imdb")  # For classification
```

**Using Custom Data:**
```python
# JSON format
{"text": "Your text here"}
{"text": "Input text", "label": 1}  # For classification

# CSV format
text,label
"Your text here",0
```

### 3. Configure Accelerate (First Time Only)

```bash
accelerate config
```

Select:
- **GPU:** Multi-GPU, mixed precision settings
- **TPU:** TPU v3-8 or v4-8

### 4. Run Training

```bash
# Single GPU
accelerate launch your_script.py --model_name_or_path ... --dataset_name ...

# Multi-GPU
accelerate launch --num_processes 4 your_script.py ...

# TPU
accelerate launch your_script.py ...  # Accelerate handles TPU automatically
```

### 5. Monitor and Evaluate

Training logs show:
- Loss values
- Evaluation metrics (accuracy, perplexity, ROUGE)
- Save checkpoints at specified intervals

### 6. Use Your Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./output")
tokenizer = AutoTokenizer.from_pretrained("./output")

# Inference
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
```

## 💡 Usage Examples

### Pretraining Examples

#### BERT-Style Pretraining on WikiText

```bash
accelerate launch pretrain_bert_style.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --output_dir ./bert-pretrained \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --mlm_probability 0.15 \
    --logging_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000
```

#### GPT-Style Pretraining on Custom Data

```bash
accelerate launch pretrain_gpt_style.py \
    --model_name_or_path gpt2 \
    --dataset_name your_custom_dataset \
    --text_column text \
    --output_dir ./gpt2-custom \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 1024 \
    --warmup_steps 500 \
    --mixed_precision bf16
```

#### T5-Style Pretraining on Large Dataset

```bash
accelerate launch pretrain_t5_style.py \
    --model_name_or_path t5-base \
    --dataset_name c4 \
    --dataset_config_name en \
    --output_dir ./t5-c4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_seq_length 512 \
    --mlm_probability 0.15 \
    --mean_noise_span_length 3.0 \
    --mixed_precision bf16 \
    --logging_steps 100 \
    --save_steps 5000
```

### Fine-tuning Examples

#### Sentiment Analysis (IMDB)

```bash
accelerate launch finetune_classification.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name imdb \
    --text_column text \
    --label_column label \
    --output_dir ./bert-imdb \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 500
```

#### Topic Classification (AG News)

```bash
accelerate launch finetune_classification.py \
    --model_name_or_path roberta-base \
    --dataset_name ag_news \
    --text_column text \
    --label_column label \
    --num_labels 4 \
    --output_dir ./roberta-agnews \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3
```

#### Text Generation Fine-tuning

```bash
accelerate launch finetune_generation.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --text_column text \
    --output_dir ./gpt2-finetuned \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 1024
```

#### Summarization (XSum)

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
    --output_dir ./t5-xsum \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_beams 4
```

### TPU-Specific Examples

#### Pretraining on TPU with Large Batch Size

```bash
# Configure for TPU first
accelerate config  # Select TPU

# Run with TPU-optimized settings
accelerate launch pretrain_bert_style.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --output_dir ./bert-tpu \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --mixed_precision bf16
```

#### Fine-tuning on TPU

```bash
accelerate launch finetune_classification.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name imdb \
    --text_column text \
    --label_column label \
    --output_dir ./bert-imdb-tpu \
    --per_device_train_batch_size 32 \
    --mixed_precision bf16 \
    --num_train_epochs 3
```

## 🔑 Key Arguments Reference

### Common Arguments (All Scripts)

**Model:**
- `--model_name_or_path`: HuggingFace model name or path (required)
- `--tokenizer_name`: Tokenizer name if different from model
- `--cache_dir`: Cache directory for models and datasets

**Data:**
- `--dataset_name`: HuggingFace dataset name
- `--dataset_config_name`: Dataset configuration
- `--text_column`: Column name containing text (default: `text`)
- `--max_seq_length`: Maximum sequence length

**Training:**
- `--output_dir`: Output directory (default: `./output`)
- `--per_device_train_batch_size`: Batch size per device (default: 8)
- `--per_device_eval_batch_size`: Eval batch size (default: 8)
- `--learning_rate`: Learning rate (default: varies by script)
- `--num_train_epochs`: Number of epochs (default: 3)
- `--warmup_steps`: Warmup steps (default: varies)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--max_grad_norm`: Gradient clipping (default: 1.0)
- `--logging_steps`: Log every N steps (default: 100)
- `--save_steps`: Save every N steps (default: varies)
- `--eval_steps`: Eval every N steps (default: varies)
- `--seed`: Random seed (default: 42)
- `--mixed_precision`: Mixed precision (`no`, `fp16`, `bf16`)
- `--push_to_hub`: Push to HuggingFace Hub after training

### Pretraining-Specific

**BERT/T5 style:**
- `--mlm_probability`: Masking probability (default: 0.15)

**T5 style:**
- `--mean_noise_span_length`: Average span length (default: 3.0)

### Fine-tuning-Specific

**Classification:**
- `--label_column`: Column name for labels (default: `label`)
- `--num_labels`: Number of classes (default: 2)

**Seq2Seq:**
- `--summary_column`: Column name for summaries
- `--source_prefix`: Prefix for input text (e.g., `"summarize: "`)
- `--max_source_length`: Max input length (default: 1024)
- `--max_target_length`: Max target length (default: 128)
- `--predict_with_generate`: Use generation for evaluation
- `--num_beams`: Number of beams for generation (default: 1)
- `--gradient_checkpointing`: Enable gradient checkpointing

## 🚀 Performance Tips

### For TPU

1. **Use larger batch sizes** - TPUs work best with large batches (32-128 per core)
2. **Use bfloat16 mixed precision** - `--mixed_precision bf16`
3. **Increase max_seq_length** - TPUs handle long sequences well
4. **Optimize data loading** - Preprocess datasets before training

```bash
accelerate launch pretrain_bert_style.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --per_device_train_batch_size 32 \
    --mixed_precision bf16 \
    --max_seq_length 512
```

### For GPU

1. **Adjust batch size based on memory** - Start small, increase gradually
2. **Use mixed precision** - `--mixed_precision fp16` or `bf16` (for newer GPUs)
3. **Use gradient accumulation** - Simulate larger batches
4. **Enable gradient checkpointing** - For large models (saves memory)

```bash
accelerate launch pretrain_gpt_style.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16
```

### General Optimization

1. **Preprocessing** - Preprocess datasets once, cache them
2. **DataLoader workers** - Use multiple workers for data loading
3. **Pin memory** - Enable for faster data transfer (GPU)
4. **Learning rate** - Use warmup steps for stability

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
--per_device_train_batch_size 4

# Increase gradient accumulation
--gradient_accumulation_steps 4

# Use mixed precision
--mixed_precision bf16

# Enable gradient checkpointing (seq2seq only)
--gradient_checkpointing
```

### Slow Training

**Solution:**
```bash
# Increase batch size
--per_device_train_batch_size 16

# Enable mixed precision
--mixed_precision bf16

# Use more preprocessing workers
--preprocessing_num_workers 4
```

### TPU Not Detected

**Check TPU availability:**
```bash
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

**Configure Accelerate for TPU:**
```bash
accelerate config
# Select: TPU
```

### Poor Performance / Low Accuracy

**Solution:**
1. **Increase training epochs** - `--num_train_epochs 5`
2. **Adjust learning rate** - Try different values (1e-5 to 5e-5)
3. **Use warmup steps** - `--warmup_steps 500`
4. **Increase max_seq_length** - `--max_seq_length 512` or higher
5. **Check your data** - Ensure data quality and preprocessing

### CUDA Out of Memory

**Solution:**
```bash
# Clear cache between runs
import torch
torch.cuda.empty_cache()

# Use smaller model
--model_name_or_path bert-base-uncased  # instead of bert-large

# Reduce sequence length
--max_seq_length 256  # instead of 512
```

## 📊 Model Comparison

### Pretraining Model Types

| Model Type | Architecture | Objective | Best For | Example Models |
|------------|-------------|-----------|----------|----------------|
| **T5-style** | Encoder-Decoder | Span corruption | Seq2seq tasks | T5, BART, mT5, mBART |
| **BERT-style** | Encoder-only | Masked LM | Classification, NER | BERT, RoBERTa, ALBERT |
| **GPT-style** | Decoder-only | Causal LM | Text generation | GPT-2, LLaMA, OPT |

### When to Use Which Model

**T5-style (Encoder-Decoder):**
- ✅ Summarization
- ✅ Translation
- ✅ Question answering
- ✅ Any seq2seq task

**BERT-style (Encoder-only):**
- ✅ Text classification
- ✅ Named entity recognition
- ✅ Sentiment analysis
- ✅ Token classification

**GPT-style (Decoder-only):**
- ✅ Text generation
- ✅ Creative writing
- ✅ Code generation
- ✅ Dialogue systems

## 🎓 T5-Specific Detailed Documentation

### T5 Pretraining Scripts Comparison

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
| **Best For** | TPU-only | Production (custom) | ✅ **Most users** | Quick experiments |

### Which T5 Version Should You Use?

#### 🏆 For Most Users: **Version 3** (`run_t5_mlm_pretrain_simple_accelerate_hf.py`)
- ✅ Best balance of simplicity and control
- ✅ Uses well-tested HuggingFace `DataCollatorForT5MLM`
- ✅ Works on GPU and TPU seamlessly
- ✅ Production-ready, clean code

#### 🚀 For Beginners: **Version 4** (`run_t5_mlm_pretrain_simple_trainer_hf.py`)
- ✅ Absolutely simplest code
- ✅ Trainer handles everything automatically
- ✅ Perfect for quick prototyping

#### 🔧 For Advanced Customization: **Version 2** (`run_t5_mlm_to_pretrain_accelerate.py`)
- ✅ Full control over data collation logic
- ✅ Easier to customize masking strategy

#### 🎯 For TPU-Only Production: **Version 1** (`run_t5_mlm_to_pretrain.py`)
- ✅ Maximum TPU optimization
- ⚠️ No GPU support

### T5 Usage Examples

#### Using Simplified Accelerate Version
```bash
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
    --mean_noise_span_length 3.0
```

#### Using Trainer Version
```bash
python run_t5_mlm_pretrain_simple_trainer_hf.py \
    --model_name_or_path t5-small \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --evaluation_strategy steps \
    --eval_steps 1000
```

## 🎉 Conclusion

### Recommended Scripts

**For Pretraining:**
- BERT-style models → `pretrain_bert_style.py`
- GPT-style models → `pretrain_gpt_style.py`
- T5-style models → `pretrain_t5_style.py`

**For Fine-tuning:**
- Classification → `finetune_classification.py`
- Text Generation → `finetune_generation.py`
- Seq2Seq → `run_seq_to_seq_model_to_finetune.py`

### Why These Scripts?

✅ **Simple** - Uses built-in HuggingFace classes, minimal code
✅ **Universal** - Works on both GPU and TPU
✅ **Production-ready** - Well-tested, maintainable
✅ **Flexible** - Easy to customize for your needs
✅ **Complete** - Covers all major model types and tasks

### Getting Help

- **HuggingFace Docs:** https://huggingface.co/docs
- **Accelerate Docs:** https://huggingface.co/docs/accelerate
- **Transformers Docs:** https://huggingface.co/docs/transformers

Happy training! 🚀
