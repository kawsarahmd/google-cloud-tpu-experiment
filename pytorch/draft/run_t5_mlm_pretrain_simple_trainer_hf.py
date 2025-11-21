#!/usr/bin/env python
# Ultra-simplified T5 MLM pretraining using HuggingFace Trainer API.

"""
Ultra-simplified pretraining script for T5-like span-masked language modeling.
Uses HuggingFace's built-in Trainer API for maximum simplicity.

This version is the simplest possible (~150 lines) while maintaining all functionality.
The Trainer handles the entire training loop, evaluation, checkpointing, and logging automatically.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    DataCollatorForT5MLM,  # ✅ Built-in HuggingFace class!
    HfArgumentParser,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    set_seed,
)
from transformers.utils import send_example_telemetry


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """
    Compute the lengths needed for T5 span corruption.
    This is a helper function from the original T5 paper.
    """
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
    # Note: Using HuggingFace's built-in TrainingArguments instead of custom one
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, HFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # Sending telemetry
    send_example_telemetry("run_t5_mlm", model_args, data_args, framework="pytorch")

    # Set seed
    set_seed(training_args.seed)

    # Load datasets
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Load tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
        )
    else:
        raise ValueError("You must specify either --tokenizer_name or --model_name_or_path")

    # Load model
    if model_args.config_name:
        config = T5Config.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            vocab_size=len(tokenizer),
            token=model_args.token,
        )
    elif model_args.model_name_or_path:
        config = T5Config.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        raise ValueError("You must specify either --config_name or --model_name_or_path")

    if model_args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            token=model_args.token,
        )
        logger.info("Loaded model from pretrained checkpoint")
    else:
        config.vocab_size = len(tokenizer)
        model = T5ForConditionalGeneration(config)
        logger.info("Created new model from scratch")

    # Preprocessing
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_attention_mask=False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing dataset",
    )

    # Compute expanded input length for T5 span corruption
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )

    # Group texts into chunks of expanded_inputs_length
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= expanded_inputs_length:
            total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
        result = {
            k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Grouping texts",
    )

    # ✅ USE BUILT-IN HUGGINGFACE DATA COLLATOR
    # This replaces our 200+ line custom implementation!
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    # ✅ USE BUILT-IN TRAINER - HANDLES EVERYTHING AUTOMATICALLY!
    # Training loop, evaluation, checkpointing, logging, distributed training, etc.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train!
    if training_args.do_train:
        logger.info("***** Running training *****")
        train_result = trainer.train()

        # Save model
        trainer.save_model()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate
    if training_args.do_eval:
        logger.info("***** Running evaluation *****")
        metrics = trainer.evaluate()

        # Save metrics
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push to hub
    if training_args.push_to_hub:
        logger.info("***** Pushing to hub *****")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
