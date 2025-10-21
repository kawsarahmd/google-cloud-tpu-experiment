#!/usr/bin/env python
"""
Universal Text Classification Fine-tuning Script
Supports: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, ELECTRA, etc.

Tasks: Sentiment analysis, topic classification, intent classification, etc.
Works on both GPU and TPU with Accelerate.

Usage:
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
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,  # ✅ Built-in HuggingFace class
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
    """All arguments for classification fine-tuning."""

    # Model arguments
    model_name_or_path: str = field(
        metadata={"help": "Model checkpoint (e.g., bert-base-uncased, roberta-base)"}
    )
    num_labels: int = field(
        default=2, metadata={"help": "Number of classes (2 for binary classification)"}
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory"})

    # Data arguments
    dataset_name: str = field(
        default="imdb", metadata={"help": "HuggingFace dataset name (e.g., imdb, ag_news)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset configuration"}
    )
    text_column: str = field(
        default="text", metadata={"help": "Column name containing text"}
    )
    label_column: str = field(
        default="label", metadata={"help": "Column name containing labels"}
    )
    max_seq_length: int = field(default=512, metadata={"help": "Maximum sequence length"})

    # Training arguments
    output_dir: str = field(default="./output", metadata={"help": "Output directory"})
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Batch size per device"})
    per_device_eval_batch_size: int = field(default=16, metadata={"help": "Eval batch size"})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate"})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of epochs"})
    warmup_steps: int = field(default=0, metadata={"help": "Warmup steps"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    logging_steps: int = field(default=100, metadata={"help": "Log every N steps"})
    eval_steps: int = field(default=500, metadata={"help": "Eval every N steps"})
    save_steps: int = field(default=500, metadata={"help": "Save every N steps"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    mixed_precision: str = field(default="no", metadata={"help": "Mixed precision: no, fp16, bf16"})
    push_to_hub: bool = field(default=False, metadata={"help": "Push to HuggingFace Hub"})


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        cache_dir=args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
    )

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_column],
            truncation=True,
            max_length=args.max_seq_length,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
    )

    # Remove unnecessary columns
    tokenized_datasets = tokenized_datasets.remove_columns(
        [col for col in raw_datasets["train"].column_names if col != args.label_column]
    )

    # Rename label column to "labels" (required by model)
    tokenized_datasets = tokenized_datasets.rename_column(args.label_column, "labels")

    # ✅ Built-in HuggingFace Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        tokenized_datasets.get("test", tokenized_datasets.get("validation")),
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

    # Load metric
    metric = evaluate.load("accuracy")

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
                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                        predictions = outputs.logits.argmax(dim=-1)
                        predictions, references = accelerator.gather_for_metrics(
                            (predictions, eval_batch["labels"])
                        )
                        metric.add_batch(predictions=predictions, references=references)

                    eval_metric = metric.compute()
                    logger.info(f"Step {completed_steps}: Accuracy = {eval_metric['accuracy']:.4f}")
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
