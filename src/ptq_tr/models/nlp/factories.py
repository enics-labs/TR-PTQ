"""Model factory functions for NLP models."""

from transformers import AutoConfig, AutoModelForSequenceClassification

from ptq_tr.models.nlp.custom_bert import CustomBertForSequenceClassification
from ptq_tr.models.nlp.qa import CustomBertForQuestionAnswering
from ptq_tr.quantization.modules import QuantizedLinear
from ptq_tr.quantization.qparams import apply_quant_config, merge_quant_config


def build_glue_sequence_classifier(model_name="distilbert-base-uncased", num_labels=2, hf_token=None):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        token=hf_token,
    )


def build_custom_glue_sequence_classifier(
    model_name="textattack/bert-base-uncased-RTE",
    num_labels=2,
    hf_token=None,
    quant=None,
    nof_bits_linear1=None,
    nof_bits_linear2=None,
    quant_config=None,
    q_module_list=None,
):
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    if getattr(config, "model_type", None) != "bert":
        raise ValueError(f"Custom GLUE path currently supports BERT checkpoints only, got: {config.model_type}")

    config.num_labels = num_labels
    apply_quant_config(
        config,
        merge_quant_config(
            quant_config,
            quant=quant,
            nof_bits_linear1=nof_bits_linear1,
            nof_bits_linear2=nof_bits_linear2,
        ),
    )

    model = CustomBertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        token=hf_token,
    )
    model.q_module_list = q_module_list or [QuantizedLinear]
    return model


def build_squad_question_answering_model(
    model_name="bert-large-uncased-whole-word-masking-finetuned-squad",
    hf_token=None,
    quant=None,
    nof_bits_linear1=None,
    nof_bits_linear2=None,
    quant_config=None,
    q_module_list=None,
):
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    if getattr(config, "model_type", None) != "bert":
        raise ValueError(f"SQuAD QA path currently supports BERT checkpoints only, got: {config.model_type}")

    apply_quant_config(
        config,
        merge_quant_config(
            quant_config,
            quant=quant,
            nof_bits_linear1=nof_bits_linear1,
            nof_bits_linear2=nof_bits_linear2,
        ),
    )

    model = CustomBertForQuestionAnswering.from_pretrained(
        model_name,
        config=config,
        token=hf_token,
    )
    model.q_module_list = q_module_list or [QuantizedLinear]
    return model
