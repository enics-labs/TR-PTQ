"""NLP dataloader builders."""

from ptq_tr.dataloaders.nlp.glue import build_glue_dataloaders
from ptq_tr.dataloaders.nlp.squad import build_squad_dataloaders


def build_nlp_dataloaders(dataset_name, **kwargs):
    if dataset_name == "glue":
        return build_glue_dataloaders(
            task_name=kwargs.get("task_name", "sst2"),
            model_name=kwargs.get("model_name", "distilbert-base-uncased"),
            batch_size=kwargs.get("batch_size", 32),
            max_length=kwargs.get("max_length", 128),
            hf_token=kwargs.get("hf_token"),
        )
    if dataset_name == "squad":
        return build_squad_dataloaders(
            model_name=kwargs.get("model_name", "bert-large-uncased-whole-word-masking-finetuned-squad"),
            batch_size=kwargs.get("batch_size", 16),
            max_length=kwargs.get("max_length", 384),
            doc_stride=kwargs.get("doc_stride", 128),
            hf_token=kwargs.get("hf_token"),
            streaming=kwargs.get("streaming", False),
        )
    raise ValueError(f"Unsupported NLP dataset: {dataset_name}")
