"""GLUE dataset helpers."""

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from ptq_tr.preprocessing.nlp.text_processing import is_valid_text_pair
from ptq_tr.preprocessing.nlp.tokenization import build_tokenizer, tokenize_glue_batch


GLUE_TASK_CONFIG = {
    "sst2": {
        "dataset_path": "glue",
        "subset": "sst2",
        "sentence1_key": "sentence",
        "sentence2_key": None,
        "num_labels": 2,
        "validation_split": "validation",
    },
    "cola": {
        "dataset_path": "glue",
        "subset": "cola",
        "sentence1_key": "sentence",
        "sentence2_key": None,
        "num_labels": 2,
        "validation_split": "validation",
    },
    "qnli": {
        "dataset_path": "glue",
        "subset": "qnli",
        "sentence1_key": "question",
        "sentence2_key": "sentence",
        "num_labels": 2,
        "validation_split": "validation",
    },
    "rte": {
        "dataset_path": "glue",
        "subset": "rte",
        "sentence1_key": "sentence1",
        "sentence2_key": "sentence2",
        "num_labels": 2,
        "validation_split": "validation",
    },
    "qqp": {
        "dataset_path": "glue",
        "subset": "qqp",
        "sentence1_key": "question1",
        "sentence2_key": "question2",
        "num_labels": 2,
        "validation_split": "validation",
        "filter_none_pairs": True,
    },
    "mnli": {
        "dataset_path": "glue",
        "subset": "mnli",
        "sentence1_key": "premise",
        "sentence2_key": "hypothesis",
        "num_labels": 3,
        "validation_split": "validation_matched",
        "filter_none_pairs": True,
    },
    "mrpc": {
        "dataset_path": "glue",
        "subset": "mrpc",
        "sentence1_key": "sentence1",
        "sentence2_key": "sentence2",
        "num_labels": 2,
        "validation_split": "validation",
    }
}


def get_glue_task_config(task_name):
    if task_name not in GLUE_TASK_CONFIG:
        raise ValueError(f"Unsupported GLUE task: {task_name}")
    return GLUE_TASK_CONFIG[task_name]


def load_glue_dataset(task_name="sst2", hf_token=None):
    cfg = get_glue_task_config(task_name)
    return load_dataset(cfg["dataset_path"], cfg["subset"], token=hf_token)


def tokenize_glue_datasets(datasets, tokenizer, task_name="sst2", max_length=128):
    cfg = get_glue_task_config(task_name)
    if cfg.get("filter_none_pairs", False):
        datasets = datasets.filter(
            lambda example: is_valid_text_pair(
                example,
                sentence1_key=cfg["sentence1_key"],
                sentence2_key=cfg["sentence2_key"],
            )
        )

    tokenized = datasets.map(
        lambda batch: tokenize_glue_batch(
            batch,
            tokenizer,
            sentence1_key=cfg["sentence1_key"],
            sentence2_key=cfg["sentence2_key"],
            max_length=max_length,
        ),
        batched=True,
    )

    columns = ["input_ids", "attention_mask", "label"]
    sample_split = "train" if "train" in tokenized else cfg["validation_split"]
    if "token_type_ids" in tokenized[sample_split].column_names:
        columns.append("token_type_ids")

    tokenized.set_format("torch", columns=columns)
    return tokenized


def build_glue_dataloaders(
    task_name="sst2",
    model_name="distilbert-base-uncased",
    batch_size=32,
    max_length=128,
    hf_token=None,
):
    cfg = get_glue_task_config(task_name)
    tokenizer = build_tokenizer(model_name, hf_token=hf_token)
    datasets = load_glue_dataset(task_name=task_name, hf_token=hf_token)
    tokenized = tokenize_glue_datasets(datasets, tokenizer, task_name=task_name, max_length=max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(
        tokenized[cfg["validation_split"]],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "tokenizer": tokenizer,
        "datasets": datasets,
        "tokenized_datasets": tokenized,
        "num_labels": cfg["num_labels"],
    }
