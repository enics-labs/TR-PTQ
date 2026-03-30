"""NLP dataloaders."""

from ptq_tr.dataloaders.nlp.builder import build_nlp_dataloaders
from ptq_tr.dataloaders.nlp.glue import build_glue_dataloaders, get_glue_task_config, load_glue_dataset
from ptq_tr.dataloaders.nlp.squad import build_squad_dataloaders, load_squad_dataset

__all__ = [
    "build_glue_dataloaders",
    "build_nlp_dataloaders",
    "build_squad_dataloaders",
    "get_glue_task_config",
    "load_glue_dataset",
    "load_squad_dataset",
]
