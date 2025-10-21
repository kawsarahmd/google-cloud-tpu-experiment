#!/usr/bin/env python
"""
Universal T5-Style Pretraining Script
Supports: T5, mT5, BART, mBART, ByT5

Uses built-in HuggingFace DataCollatorForT5MLM for span corruption.
Works on both GPU and TPU with Accelerate.

Usage:
    # T5
    accelerate launch pretrain_t5_style.py --model_name_or_path t5-small

    # BART
    accelerate launch pretrain_t5_style.py --model_name_or_path facebook/bart-base

    # mT5
    accelerate launch pretrain_t5_style.py --model_name_or_path google/mt5-small
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForT5MLM,  # ✅ Built-in HuggingFace class
    HfArgumentParser,
    get_linear_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """All arguments for T5-style pretraining."""

    # Model arguments
    model_name_or_path: str = field(
        metadata={"help": "Model checkpoint (e.g., t5-small, facebook/bart-base, google/mt5-small)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer name if different from model"}
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory"})

    # Data arguments
    dataset_name: str = field(
        default="wikitext", metadata={"help": "HuggingFace dataset name"}
    )
    dataset_config_name: Optional[str] = field(
        default="wikitext-2-raw-v1", metadata={"help": "Dataset configuration"}
    )
    text_column: str = field(
        default="text", metadata={"help": "Column name containing text"}
    )
    max_seq_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    mlm_probability: float = field(default=0.15, metadata={"help": "Masking probability"})
    mean_noise_span_length: float = field(default=3.0, metadata={"help": "Average span length"})

    # Training arguments
    output_dir: str = field(default="./output", metadata={"help": "Output directory"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per device"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Eval batch size"})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate"})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of epochs"})
    warmup_steps: int = field(default=500, metadata={"help": "Warmup steps"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    logging_steps: int = field(default=100, metadata={"help": "Log every N steps"})
    save_steps: int = field(default=1000, metadata={"help": "Save every N steps"})
    eval_steps: int = field(default=1000, metadata={"help": "Eval every N steps"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    mixed_precision: str = field(default="no", metadata={"help": "Mixed precision: no, fp16, bf16"})
    push_to_hub: bool = field(default=False, metadata={"help": "Push to HuggingFace Hub"})


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """Compute lengths for T5 span corruption."""
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        return num_nonnoise_tokens + num_noise_spans + 1, num_noise_tokens + num_noise_spans + 1

    tokens_length = inputs_length
    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def main():
    # Parse arguments
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO if accelerator.is_main_process else logging.ERROR)
    set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
    )

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
    )

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], return_attention_mask=False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )

    # Compute expanded input length for span corruption
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=args.max_seq_length,
        noise_density=args.mlm_probability,
        mean_noise_span_length=args.mean_noise_span_length,
    )

    # Group texts into chunks
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= expanded_inputs_length:
            total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
        result = {
            k: [t[i:i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc="Grouping texts",
    )

    # ✅ Built-in HuggingFace Data Collator
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=args.mlm_probability,
        mean_noise_span_length=args.mean_noise_span_length,
        input_length=args.max_seq_length,
        target_length=targets_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        tokenized_datasets.get("validation", tokenized_datasets["train"]),
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with Accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(tokenized_datasets['train'])}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Logging
                if completed_steps % args.logging_steps == 0:
                    logger.info(f"Step {completed_steps}: Loss = {loss.item():.4f}")

                # Evaluation
                if completed_steps % args.eval_steps == 0:
                    model.eval()
                    eval_loss = 0
                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                            eval_loss += outputs.loss.item()
                    eval_loss /= len(eval_dataloader)
                    logger.info(f"Step {completed_steps}: Eval Loss = {eval_loss:.4f}")
                    model.train()

                # Save checkpoint
                if completed_steps % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        logger.info(f"Saved checkpoint at step {completed_steps}")

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Training complete!")

        if args.push_to_hub:
            unwrapped_model.push_to_hub(args.output_dir.split("/")[-1])


if __name__ == "__main__":
    main()
