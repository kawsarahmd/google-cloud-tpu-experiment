"""
T5 Pretraining with PyTorch XLA for TPU v4-8
CORRECTED VERSION with proper TPU handling
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

# PyTorch XLA imports for TPU
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import (
    AutoTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    set_seed,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(default=False)
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per TPU core for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per TPU core for evaluation."}
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
    push_to_hub: bool = field(default=False)
    push_to_hub_final_step: bool = field(default=False)
    hub_model_id: str = field(default=None)
    hub_token: str = field(default=None)
    gradient_accumulation_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    num_tpu_cores: int = field(default=8, metadata={"help": "Number of TPU cores (for v4-8, use 8)"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm for clipping"})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    token: str = field(default=None)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    max_seq_length: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    mlm_probability: float = field(default=0.15)
    mean_noise_span_length: float = field(default=3.0)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """Compute input and target lengths for T5 span corruption."""
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length
    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class DataCollatorForT5MLM:
    """Data collator for T5 span-masked language modeling."""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        batch = {k: np.array([examples[i][k] for i in range(len(examples))]) for k in examples[0].keys()}
        
        input_ids = batch["input_ids"]
        batch_size, expanded_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expanded_input_length) for _ in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch["decoder_input_ids"] = self.shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        # Convert to PyTorch tensors
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch

    def shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids = np.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
        return shifted_input_ids

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        orig_length = length
        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)

        def _random_segmentation(num_items, num_segments):
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def _mp_fn(index, args):
    """Training function for each TPU core."""
    model_args, data_args, training_args = args
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if xm.is_master_ordinal() else logging.WARN,
    )
    
    # Set seed for reproducibility
    set_seed(training_args.seed + index)  # Different seed per core
    
    # Get TPU device
    device = xm.xla_device()
    
    # Only log on master process
    if xm.is_master_ordinal():
        logger.info(f"Training on TPU with {xm.xrt_world_size()} cores")
        logger.info(f"Process rank: {xm.get_ordinal()}")

    # Load datasets
    if data_args.dataset_name is not None:
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=data_args.trust_remote_code,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
                trust_remote_code=data_args.trust_remote_code,
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
                trust_remote_code=data_args.trust_remote_code,
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
        datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
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
        raise ValueError("You must provide tokenizer_name or model_name_or_path")

    # Load or create config
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
        raise ValueError("You must provide config_name or model_name_or_path")

    # Load or create model
    if model_args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        if xm.is_master_ordinal():
            logger.info("Loaded existing model from checkpoint")
    else:
        config.vocab_size = len(tokenizer)
        model = T5ForConditionalGeneration(config)
        if xm.is_master_ordinal():
            logger.info("Created model from scratch")

    model = model.to(device)

    # Preprocess datasets
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length) if data_args.max_seq_length else tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_attention_mask=False)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing dataset",
    )

    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )

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

    # Data collator
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Prepare datasets
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Create distributed sampler for TPU
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=training_args.seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=0,  # Set to 0 for TPU to avoid multiprocessing issues
        drop_last=True,
    )

    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=0,  # Set to 0 for TPU
        drop_last=False,
    )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    num_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

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

    # Learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_train_steps,
    )

    # Training loop
    if xm.is_master_ordinal():
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Batch size per TPU core = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {training_args.per_device_train_batch_size * xm.xrt_world_size()}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {num_train_steps}")

    global_step = 0
    tr_loss = torch.tensor(0.0, device=device)
    logging_loss = torch.tensor(0.0, device=device)
    model.zero_grad()

    for epoch in range(int(training_args.num_train_epochs)):
        if xm.is_master_ordinal():
            logger.info(f"Epoch {epoch + 1}/{int(training_args.num_train_epochs)}")
        
        model.train()
        train_sampler.set_epoch(epoch)
        
        # Wrap dataloader with MpDeviceLoader for TPU
        para_loader = pl.ParallelLoader(train_dataloader, [device])
        train_device_loader = para_loader.per_device_loader(device)
        
        for step, batch in enumerate(train_device_loader):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            if training_args.gradient_accumulation_steps > 1:
                loss = loss / training_args.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            tr_loss += loss.detach()

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                # Optimizer step - CRITICAL: Use xm.optimizer_step for TPU
                xm.optimizer_step(optimizer)
                lr_scheduler.step()
                model.zero_grad()
                global_step += 1
                
                # CRITICAL: Mark step for XLA graph execution
                xm.mark_step()

                # Logging
                if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:
                    # Reduce loss across all TPU cores
                    avg_loss = (tr_loss - logging_loss) / training_args.logging_steps
                    avg_loss = xm.mesh_reduce('train_loss', avg_loss, lambda x: sum(x) / len(x))
                    
                    if xm.is_master_ordinal():
                        lr = lr_scheduler.get_last_lr()[0]
                        logger.info(f"Step {global_step} - Loss: {avg_loss.item():.4f}, LR: {lr:.2e}")
                    
                    logging_loss = tr_loss.clone()

                # Evaluation
                if training_args.eval_steps is not None and global_step % training_args.eval_steps == 0:
                    if xm.is_master_ordinal():
                        logger.info(f"***** Running evaluation at step {global_step} *****")
                    
                    model.eval()
                    eval_loss = torch.tensor(0.0, device=device)
                    eval_steps = 0
                    
                    # Wrap eval dataloader
                    para_eval_loader = pl.ParallelLoader(eval_dataloader, [device])
                    eval_device_loader = para_eval_loader.per_device_loader(device)
                    
                    for eval_batch in eval_device_loader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                            loss = outputs.loss
                            eval_loss += loss
                        eval_steps += 1
                        # CRITICAL: Mark step in eval loop too
                        xm.mark_step()

                    # Reduce eval loss across cores
                    avg_eval_loss = eval_loss / eval_steps
                    avg_eval_loss = xm.mesh_reduce('eval_loss', avg_eval_loss, lambda x: sum(x) / len(x))
                    
                    if xm.is_master_ordinal():
                        logger.info(f"Eval Loss: {avg_eval_loss.item():.4f}")
                    
                    model.train()

                # Save checkpoint
                if training_args.save_steps > 0 and global_step % training_args.save_steps == 0:
                    # CRITICAL: Synchronize all cores before saving
                    xm.rendezvous('saving_checkpoint')
                    
                    if xm.is_master_ordinal():
                        output_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save model
                        model_to_save = model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        
                        # Save training state
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        }, os.path.join(output_dir, 'training_state.pt'))
                        
                        logger.info(f"Saved checkpoint to {output_dir}")

                        if training_args.push_to_hub:
                            from huggingface_hub import HfApi
                            api = HfApi()
                            api.upload_folder(
                                folder_path=output_dir,
                                repo_id=training_args.hub_model_id,
                                repo_type="model",
                                token=training_args.hub_token,
                                commit_message=f"Training step {global_step}",
                            )
                    
                    # Wait for master to finish saving
                    xm.rendezvous('checkpoint_saved')

            if global_step >= num_train_steps:
                break

        if global_step >= num_train_steps:
            break

    # Final save
    xm.rendezvous('final_save')
    
    if xm.is_master_ordinal():
        output_dir = training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Saved final model to {output_dir}")

        if training_args.push_to_hub or training_args.push_to_hub_final_step:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=output_dir,
                repo_id=training_args.hub_model_id,
                repo_type="model",
                token=training_args.hub_token,
                commit_message=f"Training completed at step {global_step}",
            )

    xm.rendezvous('final_save_done')

    # Final evaluation
    if training_args.do_eval:
        if xm.is_master_ordinal():
            logger.info("***** Running final evaluation *****")
        
        model.eval()
        eval_loss = torch.tensor(0.0, device=device)
        eval_steps = 0
        
        para_eval_loader = pl.ParallelLoader(eval_dataloader, [device])
        eval_device_loader = para_eval_loader.per_device_loader(device)
        
        for eval_batch in eval_device_loader:
            with torch.no_grad():
                outputs = model(**eval_batch)
                loss = outputs.loss
                eval_loss += loss
            eval_steps += 1
            xm.mark_step()

        avg_eval_loss = eval_loss / eval_steps
        avg_eval_loss = xm.mesh_reduce('final_eval_loss', avg_eval_loss, lambda x: sum(x) / len(x))
        
        if xm.is_master_ordinal():
            perplexity = torch.exp(avg_eval_loss)
            
            eval_results = {
                "eval_loss": avg_eval_loss.item(),
                "perplexity": perplexity.item(),
            }
            
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.json")
            with open(output_eval_file, "w") as f:
                json.dump(eval_results, f, indent=4)
            
            logger.info(f"Final Eval Loss: {avg_eval_loss.item():.4f}, Perplexity: {perplexity.item():.2f}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # Spawn TPU processes
    xmp.spawn(_mp_fn, args=(model_args, data_args, training_args), nprocs=training_args.num_tpu_cores, start_method='fork')


if __name__ == "__main__":
    main()
