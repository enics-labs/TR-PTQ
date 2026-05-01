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
    validation_split: str = "validation"


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
    "sst2": GlueTaskSpec(
        name="sst2",
        default_model_name="Talip7/bert-base-sst2-finetuned",
        dataset_name="glue",
        dataset_config="sst2",
        text_fields=("sentence",),
        metric_name="glue",
        metric_config="sst2",
        primary_metric="accuracy",
        model_class=CustomBertForSequenceClassification,
        default_num_val_examples=872,
    ),
    "qqp": GlueTaskSpec(
        name="qqp",
        default_model_name="textattack/bert-base-uncased-QQP",
        dataset_name="glue",
        dataset_config="qqp",
        text_fields=("question1", "question2"),
        metric_name="glue",
        metric_config="qqp",
        primary_metric="accuracy",
        model_class=CustomBertForSequenceClassification,
        default_num_val_examples=40430,
    ),
    "mnli": GlueTaskSpec(
        name="mnli",
        default_model_name="ishan/bert-base-uncased-mnli",
        dataset_name="glue",
        dataset_config="mnli",
        text_fields=("premise", "hypothesis"),
        metric_name="glue",
        metric_config="mnli",
        primary_metric="accuracy",
        model_class=CustomBertForSequenceClassification,
        default_num_val_examples=9815,
        validation_split="validation_matched",
    ),
    "qnli": GlueTaskSpec(
        name="qnli",
        default_model_name="textattack/bert-base-uncased-QNLI",
        dataset_name="glue",
        dataset_config="qnli",
        text_fields=("question", "sentence"),
        metric_name="glue",
        metric_config="qnli",
        primary_metric="accuracy",
        model_class=CustomBertForSequenceClassification,
        default_num_val_examples=5463,
    ),
    "rte": GlueTaskSpec(
        name="rte",
        default_model_name="textattack/bert-base-uncased-RTE",
        dataset_name="glue",
        dataset_config="rte",
        text_fields=("sentence1", "sentence2"),
        metric_name="glue",
        metric_config="rte",
        primary_metric="accuracy",
        model_class=CustomBertForSequenceClassification,
        default_num_val_examples=277,
    ),
}


def get_task_spec(task_name):
    normalized_name = task_name.lower()
    if normalized_name not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_name {task_name!r}. Valid tasks: {valid}")
    return TASKS[normalized_name]


def _sanitize_text_value(value):
    if isinstance(value, list):
        return ["" if item is None else str(item) for item in value]
    return "" if value is None else str(value)


def tokenize_task_batch(task, batch, tokenizer, max_length=128, padding="max_length"):
    texts = [_sanitize_text_value(batch[field]) for field in task.text_fields]
    return tokenizer(
        *texts,
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_tensors="pt",
    )
