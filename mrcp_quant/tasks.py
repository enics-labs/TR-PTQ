from dataclasses import dataclass
from typing import Tuple, Type

from .heads import CustomBertForMRPC, CustomBertForSequenceClassification


@dataclass(frozen=True)
class GlueTaskSpec:
    name: str
    default_model_name: str
    dataset_name: str
    dataset_config: str
    text_fields: Tuple[str, ...]
    metric_name: str
    metric_config: str
    primary_metric: str
    model_class: Type
    default_num_val_examples: int


TASKS = {
    "mrpc": GlueTaskSpec(
        name="mrpc",
        default_model_name="lrs21/bert-base-uncased-finetuned-glue-mrpc",
        dataset_name="nyu-mll/glue",
        dataset_config="mrpc",
        text_fields=("sentence1", "sentence2"),
        metric_name="glue",
        metric_config="mrpc",
        primary_metric="accuracy",
        model_class=CustomBertForMRPC,
        default_num_val_examples=408,
    ),
    "cola": GlueTaskSpec(
        name="cola",
        default_model_name="geckos/bert-base-uncased-finetuned-glue-cola",
        dataset_name="glue",
        dataset_config="cola",
        text_fields=("sentence",),
        metric_name="glue",
        metric_config="cola",
        primary_metric="matthews_correlation",
        model_class=CustomBertForSequenceClassification,
        default_num_val_examples=1043,
    ),
}


def get_task_spec(task_name):
    normalized_name = task_name.lower()
    if normalized_name not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_name {task_name!r}. Valid tasks: {valid}")
    return TASKS[normalized_name]


def tokenize_task_batch(task, batch, tokenizer, max_length=128, padding="max_length"):
    texts = [batch[field] for field in task.text_fields]
    return tokenizer(
        *texts,
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_tensors="pt",
    )
