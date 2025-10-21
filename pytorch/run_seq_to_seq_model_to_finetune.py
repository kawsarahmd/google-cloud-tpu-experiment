#!/usr/bin/env python
# This code is adapted from a Hugging Face example and converted to PyTorch with Accelerate for GPU/TPU support.

"""
Fine-tuning the library models for summarization.
Uses Hugging Face Accelerate library for easy GPU/TPU/multi-GPU support.
"""

import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import datasets
import evaluate
import nltk
import numpy as np
import torch
from datasets import Dataset, load_dataset
from filelock import FileLock
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm

# Accelerate imports for unified GPU/TPU/multi-GPU support
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)
from transformers.utils import is_offline_mode, send_example_telemetry


logger = get_logger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        nltk.download('punkt_tab', quiet=True)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
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
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
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
        metadata={
            "help": (
                "Whether to use mixed precision. Choose from 'no', 'fp16', 'bf16'. "
                "This is passed to Accelerate."
            )
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
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
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input predict data file to do prediction on (a text file)."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the `max_length` param of `model.generate`, which is used "
                "during evaluation."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to `model.generate`, "
                "which is used during evaluation."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "kawsarahmd/papers_summary_datasets_xsum_bangla": ("text", "summary"),
    "kawsarahmd/papers_summary_datasets_v2": ("content", "content_summary"),
    "kawsarahmd/papers_summary_datasets_v3": ("content", "content_summary"),
}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
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

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Sending telemetry
    if accelerator.is_main_process:
        send_example_telemetry("run_summarization", model_args, data_args, framework="pytorch")

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

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.push_to_hub and accelerator.is_main_process:
        # Retrieve or infer repo_name
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(training_args.output_dir).absolute().name
        # Create repo and retrieve repo_id
        api = HfApi()
        repo_id = api.create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id

    accelerator.wait_for_everyone()

    # Get the datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Load pretrained model and tokenizer
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(
            config,
            trust_remote_code=model_args.trust_remote_code,
        )

    # Enable gradient checkpointing if requested
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training
    max_target_length = data_args.max_target_length
    padding = "max_length"  # Using fixed length for better TPU/GPU performance

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with accelerator.main_process_first():
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with accelerator.main_process_first():
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Metric
    metric = evaluate.load("rouge", cache_dir=model_args.cache_dir)

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    # DataLoaders creation
    if training_args.do_train:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=training_args.per_device_train_batch_size,
        )

    if training_args.do_eval:
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=training_args.per_device_eval_batch_size,
        )

    if training_args.do_predict:
        predict_dataloader = DataLoader(
            predict_dataset,
            collate_fn=data_collator,
            batch_size=training_args.per_device_eval_batch_size,
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

    # Scheduler and math around the number of training steps
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

    if training_args.do_predict:
        predict_dataloader = accelerator.prepare(predict_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    # Training
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Only show the progress bar once on each machine
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Label smoothing (if requested)
    if training_args.label_smoothing_factor > 0:
        from torch.nn import CrossEntropyLoss
        # We'll use label smoothing in the loss calculation
        loss_fct = CrossEntropyLoss(label_smoothing=training_args.label_smoothing_factor, ignore_index=-100)

    for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                if training_args.label_smoothing_factor > 0:
                    # Manual loss calculation with label smoothing
                    outputs = model(**batch, use_cache=False)
                    logits = outputs.logits
                    labels = batch["labels"]

                    # Shift for decoder models
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Flatten the tokens
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    outputs = model(**batch)
                    loss = outputs.loss

                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Logging
                if completed_steps % training_args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / training_args.logging_steps
                    logger.info(f"Epoch {epoch}, Step {completed_steps}: Loss = {avg_loss:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.2e}")
                    total_loss = 0

                # Evaluation
                if training_args.eval_steps and completed_steps % training_args.eval_steps == 0 and training_args.do_eval:
                    model.eval()
                    losses = []
                    gen_kwargs = {
                        "max_length": data_args.val_max_target_length,
                        "num_beams": data_args.num_beams,
                    }

                    all_preds = []
                    all_labels = []

                    for eval_step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                            loss = outputs.loss
                            losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

                            if data_args.predict_with_generate:
                                generated_tokens = accelerator.unwrap_model(model).generate(
                                    batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    **gen_kwargs,
                                )

                                generated_tokens = accelerator.pad_across_processes(
                                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                                )
                                labels = batch["labels"]

                                generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().numpy()
                                labels = accelerator.gather_for_metrics(labels).cpu().numpy()

                                all_preds.extend(generated_tokens)
                                all_labels.extend(labels)

                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses).item()

                    result = {"eval_loss": eval_loss}
                    if data_args.predict_with_generate:
                        rouge_metrics = compute_metrics(all_preds, all_labels)
                        result.update(rouge_metrics)

                    logger.info(f"Step {completed_steps} Eval results: {result}")

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

        # End of epoch evaluation
        if training_args.do_eval:
            model.eval()
            losses = []
            gen_kwargs = {
                "max_length": data_args.val_max_target_length,
                "num_beams": data_args.num_beams,
            }

            all_preds = []
            all_labels = []

            for eval_step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

                    if data_args.predict_with_generate:
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )

                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]

                        generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().numpy()
                        labels = accelerator.gather_for_metrics(labels).cpu().numpy()

                        all_preds.extend(generated_tokens)
                        all_labels.extend(labels)

            losses = torch.cat(losses)
            eval_loss = torch.mean(losses).item()

            result = {"eval_loss": eval_loss}
            if data_args.predict_with_generate:
                rouge_metrics = compute_metrics(all_preds, all_labels)
                result.update(rouge_metrics)

            logger.info(f"Epoch {epoch} Eval results: {result}")

        # Save at end of epoch
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(
                training_args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(training_args.output_dir)

            if training_args.push_to_hub and epoch == int(training_args.num_train_epochs) - 1:
                api.upload_folder(
                    commit_message=f"Saving weights and logs of epoch {epoch}",
                    folder_path=training_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=training_args.hub_token,
                )

        accelerator.wait_for_everyone()

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        model.eval()
        pred_losses = []
        gen_kwargs = {
            "max_length": data_args.val_max_target_length,
            "num_beams": data_args.num_beams,
        }

        all_preds = []
        all_labels = []

        for pred_step, batch in enumerate(tqdm(predict_dataloader, desc="Predicting", disable=not accelerator.is_local_main_process)):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                pred_losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

                if data_args.predict_with_generate:
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    labels = batch["labels"]

                    generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().numpy()
                    labels = accelerator.gather_for_metrics(labels).cpu().numpy()

                    all_preds.extend(generated_tokens)
                    all_labels.extend(labels)

        pred_losses = torch.cat(pred_losses)
        pred_loss = torch.mean(pred_losses).item()

        result = {"predict_loss": pred_loss}
        if data_args.predict_with_generate:
            rouge_metrics = compute_metrics(all_preds, all_labels)
            result.update(rouge_metrics)

        logger.info(f"Predict results: {result}")

        # Save final metrics in json
        if accelerator.is_main_process:
            if data_args.predict_with_generate:
                rouge_metrics = {f"test_{metric_name}": value for metric_name, value in rouge_metrics.items()}
                path = os.path.join(training_args.output_dir, "test_results.json")
                with open(path, "w") as f:
                    json.dump(rouge_metrics, f, indent=4, sort_keys=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
