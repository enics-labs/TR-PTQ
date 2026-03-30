"""Shared helpers for NLP workflow entrypoints."""

from ptq_tr.common.device import get_default_device
from ptq_tr.dataloaders.nlp import build_nlp_dataloaders
from ptq_tr.models.nlp import build_glue_sequence_classifier, build_squad_question_answering_model


DEFAULT_NLP_MODELS = {
    "glue": "distilbert-base-uncased",
    "squad": "bert-large-uncased-whole-word-masking-finetuned-squad",
}


def resolve_nlp_model_name(dataset_name, model_name=None):
    if model_name is not None:
        return model_name
    if dataset_name not in DEFAULT_NLP_MODELS:
        raise ValueError(f"Unsupported NLP dataset: {dataset_name}")
    return DEFAULT_NLP_MODELS[dataset_name]


def build_nlp_runtime(
    *,
    task_name="sst2",
    model_name=None,
    dataset_name="glue",
    batch_size=32,
    max_length=None,
    doc_stride=128,
    streaming=False,
    hf_token=None,
    device=None,
):
    model_name = resolve_nlp_model_name(dataset_name, model_name)
    if max_length is None:
        max_length = 384 if dataset_name == "squad" else 128

    dataloaders = build_nlp_dataloaders(
        dataset_name,
        task_name=task_name,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        doc_stride=doc_stride,
        streaming=streaming,
        hf_token=hf_token,
    )

    device = device or get_default_device()
    if dataset_name == "glue":
        model = build_glue_sequence_classifier(
            model_name=model_name,
            num_labels=dataloaders["num_labels"],
            hf_token=hf_token,
        )
        task_type = "sequence_classification"
    elif dataset_name == "squad":
        model = build_squad_question_answering_model(
            model_name=model_name,
            hf_token=hf_token,
        )
        task_type = "question_answering"
    else:
        raise ValueError(f"Unsupported NLP dataset: {dataset_name}")

    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "dataset_name": dataset_name,
        "task_name": task_name,
        "task_type": task_type,
        "train_loader": dataloaders["train_loader"],
        "val_loader": dataloaders["val_loader"],
        "tokenizer": dataloaders["tokenizer"],
        "datasets": dataloaders["datasets"],
        "tokenized_datasets": dataloaders.get("tokenized_datasets"),
        "num_labels": dataloaders["num_labels"],
        "max_length": dataloaders.get("max_length", max_length),
        "doc_stride": dataloaders.get("doc_stride", doc_stride),
    }
