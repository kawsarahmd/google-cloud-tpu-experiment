#!/usr/bin/env python
# Simplified T5 MLM pretraining using HuggingFace built-in classes with Accelerate for GPU/TPU support.

"""
Simplified pretraining script for T5-like span-masked language modeling.
Uses HuggingFace's built-in DataCollatorForT5MLM instead of custom implementation.

This version is much simpler (~300 lines vs 1000 lines) while maintaining all functionality.
"""

import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader

# Accelerate imports for unified GPU/TPU/multi-GPU support
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    DataCollatorForT5MLM,  # ✅ Built-in HuggingFace class!
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)
from transformers.utils import send_example_telemetry


logger = get_logger(__name__)


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


@dataclass
class TrainingArguments:
    """Training arguments."""

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."},
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm for gradient clipping."})
    mixed_precision: str = field(
        default="no",
        metadata={"help": "Whether to use mixed precision. Choose from 'no', 'fp16', 'bf16'."},
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)


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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=training_args.mixed_precision,
        log_with="tensorboard",
        project_dir=training_args.output_dir,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    # Sending telemetry
    if accelerator.is_main_process:
        send_example_telemetry("run_t5_mlm", model_args, data_args, framework="pytorch")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)

    # Set seed
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.push_to_hub and accelerator.is_main_process:
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(training_args.output_dir).absolute().name
        api = HfApi()
        repo_id = api.create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id

    accelerator.wait_for_everyone()

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
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_attention_mask=False)

    with accelerator.main_process_first():
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

    with accelerator.main_process_first():
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

    # Create dataloaders
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
        drop_last=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    # Training info
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Training loop
    from tqdm import tqdm
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Logging
                if completed_steps % training_args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / training_args.logging_steps
                    logger.info(f"Step {completed_steps}: Loss = {avg_loss:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.2e}")
                    total_loss = 0

                # Evaluation
                if training_args.eval_steps and completed_steps % training_args.eval_steps == 0 and training_args.do_eval:
                    model.eval()
                    losses = []

                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                            loss = outputs.loss
                            losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses).item()
                    logger.info(f"Step {completed_steps}: Eval Loss = {eval_loss:.4f}")

                    model.train()

                # Save checkpoint
                if completed_steps % training_args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)

                    if accelerator.is_main_process:
                        unwrapped_model.save_pretrained(
                            training_args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        tokenizer.save_pretrained(training_args.output_dir)

                        if training_args.push_to_hub:
                            api.upload_folder(
                                commit_message=f"Saving weights and logs of step {completed_steps}",
                                folder_path=training_args.output_dir,
                                repo_id=repo_id,
                                repo_type="model",
                                token=training_args.hub_token,
                            )

                    accelerator.wait_for_everyone()

            if completed_steps >= max_train_steps:
                break

    # Final save
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            training_args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub:
            api.upload_folder(
                commit_message=f"Saving model of final step {completed_steps}",
                folder_path=training_args.output_dir,
                repo_id=repo_id,
                repo_type="model",
                token=training_args.hub_token,
            )

    # Final evaluation
    if training_args.do_eval:
        logger.info("***** Running final evaluation *****")
        model.eval()
        losses = []

        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses).item()
        logger.info(f"Final Eval Loss = {eval_loss:.4f}")

        if accelerator.is_main_process:
            eval_results = {"eval_loss": eval_loss}
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.json")
            with open(output_eval_file, "w") as f:
                json.dump(eval_results, f, indent=4, sort_keys=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
