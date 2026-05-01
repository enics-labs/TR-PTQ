import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput

from .bert_model import QBertPreTrainedModel, CustomBertModel
from .quantized_layers import QuantizedLinear

"""## cola CustomBertForSequenceClassification"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertPreTrainedModel

class CustomBertForSequenceClassification(QBertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = CustomBertModel(config, add_pooling_layer=True)

        # HuggingFace uses a dropout before classifier
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Replace Linear with your quantized version
        self.classifier = QuantizedLinear(
            config.hidden_size,
            config.num_labels,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Pooled output ([CLS])
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ---------------- Quant control hooks ----------------
    def set_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_calibration_flag()

    def unset_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_calibration_flag()

    def set_quant(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_quant()

    def unset_quant(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_quant()

    def set_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_scale_opt()

    def unset_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_scale_opt()

"""## CustomBertForMRPC - Sentence-pair sequence classification (MRPC)"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomBertForMRPC(QBertPreTrainedModel):
    """
    Sentence-pair sequence classification (MRPC).
    Inputs: (input_ids, attention_mask, token_type_ids) from tokenizer(text1, text2, ...)
    """

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        print("[DEBUG] config CustomBertForMRPC", config)
        self.num_labels = config.num_labels  # MRPC: 2

        self.bert = CustomBertModel(config, add_pooling_layer=True)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = QuantizedLinear(
            config.hidden_size,
            config.num_labels,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,      # IMPORTANT for sentence pairs
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ---------------- Quant control hooks ----------------
    def set_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_calibration_flag()

    def unset_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_calibration_flag()

    def set_quant(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_quant()

    def unset_quant(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_quant()

    def set_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_scale_opt()

    def unset_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_scale_opt()

