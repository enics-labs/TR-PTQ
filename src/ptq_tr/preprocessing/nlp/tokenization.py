"""Tokenization helpers."""

from transformers import AutoTokenizer

from ptq_tr.preprocessing.nlp.text_processing import get_text_pair


def build_tokenizer(model_name, hf_token=None, use_fast=True):
    return AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=use_fast)


def tokenize_glue_batch(batch, tokenizer, sentence1_key, sentence2_key=None, max_length=128):
    texts = get_text_pair(batch, sentence1_key, sentence2_key)
    return tokenizer(texts, truncation=True, max_length=max_length)


def tokenize_squad_batch(batch, tokenizer, max_length=384, doc_stride=128):
    return tokenizer(
        batch["question"],
        batch["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
