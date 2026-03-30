"""SQuAD dataset helpers."""

from datasets import load_dataset
from torch.utils.data import DataLoader

from ptq_tr.preprocessing.nlp.tokenization import build_tokenizer


def squad_batch_collator(features):
    if not features:
        return {}
    return {key: [feature[key] for feature in features] for key in features[0]}


def load_squad_dataset(hf_token=None, streaming=False):
    return load_dataset("squad", token=hf_token, streaming=streaming)


def build_squad_dataloaders(
    model_name="bert-large-uncased-whole-word-masking-finetuned-squad",
    batch_size=16,
    max_length=384,
    doc_stride=128,
    hf_token=None,
    streaming=False,
):
    tokenizer = build_tokenizer(model_name, hf_token=hf_token, use_fast=True)
    datasets = load_squad_dataset(hf_token=hf_token, streaming=streaming)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=not streaming,
        collate_fn=squad_batch_collator,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=squad_batch_collator,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "tokenizer": tokenizer,
        "datasets": datasets,
        "max_length": max_length,
        "doc_stride": doc_stride,
        "num_labels": 2,
    }
