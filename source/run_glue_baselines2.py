# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import copy
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import pathlib

import datasets
import evaluate
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from models.model_factory import create_model
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default="DEFAULT"
    )

    parser.add_argument(
        "--overwrite_saves",
        type=str,
        default='y'
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class PerturbedEnsembleModel(nn.Module):
    def __init__(self, models,config):
        super(PerturbedEnsembleModel, self).__init__()
        # Initialize the list of member BERT models
        self.members = nn.ModuleList(models)
        self.config = config
        self.num_labels = config.num_labels
        # Initialize weights for each model
        num_models = len(models)
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        # Perturb the 0th member's weights
        self.perturb_member(self.members[0], epsilon=0.1)
    def perturb_member(self, model, epsilon=0.01):
        """
        Add Gaussian noise to all trainable weights of the given model.
        
        Args:
            model (nn.Module): The model to perturb.
            epsilon (float): The standard deviation of the Gaussian noise.
        """
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn(param.size()) * epsilon
                param.data += noise

    def forward(
        self,
        batch
    ):
        outputs = []
        attentions = []
        # Use enumerate to get both the index and model directly
        for index, model in enumerate(self.members):
            output = model(
                **batch,
                output_attentions=True,
                output_hidden_states=True
            )
            # Multiply model output (logits) with its respective weight
           
            
            weighted_output = output.logits * self.weights[index]
            outputs.append(weighted_output)
            attentions.append(output.attentions)
        # Sum all weighted outputs
        combined_logits = torch.sum(torch.stack(outputs), dim=0)
        # Calculate loss if labels are provided
        
        labels = batch['labels']

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(combined_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(combined_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(combined_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(combined_logits, labels)
        
        return {
                'loss' : loss,
                'combined_logits' : combined_logits,
                'attentions' : attentions
        }



class EnsembleModel(nn.Module):
    def __init__(self, models,config):
        super(EnsembleModel, self).__init__()
        # Initialize the list of member BERT models
        self.members = nn.ModuleList(models)
        self.config = config
        self.num_labels = config.num_labels
        # Initialize weights for each model
        num_models = len(models)
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(
        self,
        batch
    ):
        outputs = []
        attentions = []
        # Use enumerate to get both the index and model directly
        for index, model in enumerate(self.members):
            output = model(
                **batch,
                output_attentions=True,
                output_hidden_states=True
            )
            # Multiply model output (logits) with its respective weight
           
            
            weighted_output = output.logits * self.weights[index]
            outputs.append(weighted_output)
            attentions.append(output.attentions)
        # Sum all weighted outputs
        combined_logits = torch.sum(torch.stack(outputs), dim=0)
        # Calculate loss if labels are provided
        
        labels = batch['labels']

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(combined_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(combined_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(combined_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(combined_logits, labels)
        
        return {
                'loss' : loss,
                'combined_logits' : combined_logits,
                'attentions' : attentions
        }



def cosine_similarity(x, y):
    # Normalize x and y to have unit norm
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    # Cosine similarity
    cosine_sim = torch.sum(x_norm * y_norm, dim=1)
    # Dissimilarity is 1 minus similarity
    return cosine_simi

def n_cosine_similarity(tensor_list):
    # Flatten and normalize each tensor in the list
    normalized_tensors = [F.normalize(tensor.view(tensor.size(0), -1), p=2, dim=1) for tensor in tensor_list]
    
    # Concatenate all tensors into a single batch for efficient computation
    all_tensors = torch.cat(normalized_tensors, dim=0)
    
    # Compute cosine similarity between all pairs using matrix multiplication
    similarity_matrix = torch.mm(all_tensors, all_tensors.t())
    
    # Extract indices for each tensor
    indices = torch.cumsum(torch.tensor([0] + [tensor.size(0) for tensor in tensor_list]), dim=0)
    
    # Initialize an empty list to store results
    similarities = []
    
    # Extract similarities for each unique pair of tensors
    for i in range(len(tensor_list)):
        for j in range(i + 1, len(tensor_list)):  # Adjusted to get only unique pairs
            sim = similarity_matrix[indices[i]:indices[i+1], indices[j]:indices[j+1]]
            similarities.append(sim)

    # Concatenate all pairwise similarities into a single tensor
    result_tensor = torch.cat(similarities, dim=0)

    return result_tensor
def kl_convergence(x, y):
    softmax_x = F.softmax(x, dim=1)
    softmax_y = F.softmax(y, dim=1)
    return -F.kl_div(softmax_x.log(), softmax_y, reduction='batchmean')

class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Initialize weights

    def forward(self, logits1, logits2):
        return logits1 * self.weights[0] + logits2 * self.weights[1]

class LogitsTransformer(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(LogitsTransformer, self).__init__()
        self.hidden_dim = 128
        self.fc = nn.Sequential(
             nn.Linear(input_dim * 2,num_labels),
#            nn.BatchNorm1d(self.hidden_dim),
#            nn.ReLU(),
#            nn.Linear(self.hidden_dim, num_labels)
#            nn.Tanh()
        )

    def forward(self, logits1, logits2):
        combined = torch.cat([logits1, logits2], dim=1)
        return self.fc(combined)

def soft_voting(logits1, logits2):
    probs1 = nn.functional.softmax(logits1, dim=1)
    probs2 = nn.functional.softmax(logits2, dim=1)
    return (probs1 + probs2) / 2

class GatedMixtureOfExperts(nn.Module):
    def __init__(self, input_dim):
        super(GatedMixtureOfExperts, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, logits1, logits2):
        gates = self.gate(torch.cat([logits1, logits2], dim=1))
        return logits1 * gates[:, 0:1] + logits2 * gates[:, 1:2]

class AdaBoostCombiner(nn.Module):
    def __init__(self):
        super(AdaBoostCombiner, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(2))  # Logarithmic weights

    def forward(self, logits1, logits2):
        weights = torch.exp(self.alpha)
        combined_logits = logits1 * weights[0] + logits2 * weights[1]
        return combined_logits / weights.sum()

def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
def compute_with_retry(metric, max_retries=10, initial_wait=1):
    for attempt in range(max_retries):
        try:
            return metric.compute()
        except ValueError as e:
            if "Please specify an experiment_id" in str(e):
                wait_time = initial_wait * (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise if it's a different ValueError
    
    raise Exception(f"Failed to compute metric after {max_retries} attempts")

import uuid
def main():

    args = parse_args()
    folder_path = f"./saves/{args.job_id}" 
    pathlib.Path(folder_path).mkdir(exist_ok=True)
    eval_save_file_name = f'{folder_path}/results_rg_{args.task_name}_{args.model_name_or_path.split("/")[-1]}.json'

    # If dataset model seed combination exists 
    if(args.overwrite_saves != 'y'):
        # Check if the particular seed exists
        try:
            save = json.load( open(eval_save_file_name,'r') )
            if(str(args.seed) in save):
                print(f"Seed {args.seed} already exists in {eval_save_file_name}")
                return
            else:
                print(save)
        except:
            print("Save file does not exist, starting new run")
    save_dir = "./downloads/offline_saves/"
    model_name_short = args.model_name_or_path.split("/")[-1]
    config_path = os.path.join(save_dir, f"{model_name_short}_{args.task_name}_config")
    tokenizer_path = os.path.join(save_dir, f"{model_name_short}_tokenizer")
    model_path = os.path.join(save_dir, f"{model_name_short}_{args.task_name}_model")
    dataset_path = os.path.join(save_dir, f"dataset_{args.task_name}")

    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if not os.path.exists(dataset_path):
        raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
        raw_datasets.save_to_disk(dataset_path)
    else:
        raw_datasets = datasets.load_from_disk(dataset_path)

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if not os.path.exists(config_path):
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            trust_remote_code=args.trust_remote_code,
        )
        config.save_pretrained(config_path)
    else:
        config = AutoConfig.from_pretrained(config_path)


    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id

    if not os.path.exists(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            trust_remote_code=None
        )
        model.save_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)



    num_models = 3
    models = [copy.deepcopy(model) for _ in range(num_models)]



    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        models[0].config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in models[0].config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        for i in range(num_models):
            models[i].config.label2id = label_to_id
            models[i].config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:

        for i in range(num_models):
            models[i].config.label2id = {l: i for i, l in enumerate(label_list)}
            models[i].config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not

    ''' Ensemble model '''
    ensemble_model = EnsembleModel(models,config)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in ensemble_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in ensemble_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }

    ]

    ''' Consolidator '''
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with our `accelerator`.
    ensemble_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        ensemble_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    metric = evaluate.load("./downloads/evaluate/metrics/glue/glue.py", args.task_name, experiment_id=str(uuid.uuid4()))
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    eval_result = None
    for epoch in range(starting_epoch, args.num_train_epochs):

        ensemble_model.train()

        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            
            outputs = ensemble_model(batch)
            logits = outputs['combined_logits']
            attentions = outputs['attentions']
            
            
            loss = outputs['loss']

            ''' Intermediate representation decorrelation'''
            
            select_layers = []#range(1,3)
            
            for layer_idx in select_layers:
                
                #Each attentions will contain n = number of models tuples
                list_of_attentions = []
                for model_idx,model_attentions in enumerate(attentions):
                    list_of_attentions.append(model_attentions[layer_idx])
                
                similarity = n_cosine_similarity(list_of_attentions).mean()
                loss += 0.1 * similarity
            
            '''

            for layer1,layer2 in zip(outputs_1.hidden_states,outputs_2.hidden_states):
                #intermediate_act_1 = outputs_1.hidden_states[6]
                #intermediate_act_2 = outputs_2.hidden_states[6]
                #similarity = cosine_similarity(intermediate_act_1, intermediate_act_2).mean()
                
                if (idx in select_layers):
                    similarity = cosine_similarity(layer1,layer2).mean()#kl_convergence(layer1,layer2)
                    loss += 0.1 * similarity

                idx += 1
            '''

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            
            accelerator.backward(loss)
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                #print(consolidated_logits[0])    
                #print(f"Gradient Norms at step {step}: Model 1: {gradient_norm_1}, Model 2: {gradient_norm_2}, Consolidator: {gradient_norm_consolidator}")
                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        ensemble_model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = ensemble_model(batch)
                logits = outputs['combined_logits']
            
            predictions = logits.argmax(dim=-1) if not is_regression else logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = compute_with_retry(metric)
        eval_result = eval_metric
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )


        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    
    folder_path = f"./saves/{args.job_id}"
    
    pathlib.Path(folder_path).mkdir(exist_ok=True)
    eval_save_file_name = f"{folder_path}/results_rg_{args.task_name}_{args.model_name_or_path.split('/')[-1]}.json"
    if(not os.path.isfile(eval_save_file_name)):

        #Custom save
        json.dump(
            {},
            open(eval_save_file_name,'w')
        ) 
    
    save_data = json.load(open(eval_save_file_name,'r'))
    save_data[str(args.seed)] = eval_result
    
    json.dump(
        save_data,
        open(eval_save_file_name,'w')
    )

    if args.with_tracking:
        accelerator.end_training()
    '''
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
    '''

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        ensemble_model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = ensemble_model(**batch)

            logits = outputs.logits

            predictions = logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = compute_with_retry(metric)
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    main()
