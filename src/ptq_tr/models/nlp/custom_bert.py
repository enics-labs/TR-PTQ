"""Custom quantized BERT implementations for sequence classification."""

import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from ptq_tr.quantization.modules import IntGeluTS, IntSoftmaxTS, QLayerNorm, QuantizedLinear, QuantizedMatmul
from ptq_tr.quantization.observers.base import ObserverBase
from ptq_tr.quantization.qparams import QuantConfig, QauntParams, apply_quant_config


class QuantizedBertModule(QauntParams):
    """Small helper base class that pulls quantization settings from the config."""

    def __init__(self, config=None):
        super().__init__(quant_config=config)


class QBertPreTrainedModel(BertPreTrainedModel):
    """BERT pretrained base with notebook-style quantization controls."""

    def __init__(self, config):
        super().__init__(config)
        self.q_module_list = list(getattr(config, "q_module_list", []))
        apply_quant_config(self, QuantConfig())
        apply_quant_config(self, config)

    def set_q_module_list(self, q_module_list):
        self.q_module_list = list(q_module_list)

    def set_calibration_flag(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.set_calibration_flag()

    def unset_calibration_flag(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.unset_calibration_flag()

    def set_quant(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.set_quant()

    def unset_quant(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.unset_quant()

    def set_scale_opt(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.set_scale_opt()

    def unset_scale_opt(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.unset_scale_opt()

    def finalize_quantization(self):
        for module in self.modules():
            if isinstance(module, ObserverBase):
                module.calculate_qparams()


class CustomBertPooler(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        self.dense = QuantizedLinear(
            config.hidden_size,
            config.hidden_size,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return self.activation(pooled_output)


class CustomBertSelfOutput(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        self.dense = QuantizedLinear(
            config.hidden_size,
            config.hidden_size,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )
        self.LayerNorm = QLayerNorm(
            config.hidden_size,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            eps=config.layer_norm_eps,
            quant=self.quant,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class CustomBertSelfAttention(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads "
                f"({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = QuantizedLinear(
            config.hidden_size,
            self.all_head_size,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )
        self.key = QuantizedLinear(
            config.hidden_size,
            self.all_head_size,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )
        self.value = QuantizedLinear(
            config.hidden_size,
            self.all_head_size,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )

        self.mat_mul_qk = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=self.quant,
        )
        self.sf = IntSoftmaxTS(
            nof_bits=self.nof_bits_softmax,
            int_bits=self.int_bits_softmax,
            LUT_SIZE=self.lut_size_softmax,
            dim=-1,
            quant=self.quant,
        )
        self.mat_mul_pv = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=self.quant,
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = self.mat_mul_qk(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.sf(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = self.mat_mul_pv(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class CustomBertAttention(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        self.self = CustomBertSelfAttention(config)
        self.output = CustomBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.self.num_attention_heads -= len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class CustomBertIntermediate(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        self.dense = QuantizedLinear(
            config.hidden_size,
            config.intermediate_size,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = IntGeluTS(
                quant=self.quant,
                LUT_SIZE=self.lut_size_gelu,
                nof_bits=self.nof_bits_gelu,
            )
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return self.intermediate_act_fn(hidden_states)


class CustomBertOutput(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        self.dense = QuantizedLinear(
            config.intermediate_size,
            config.hidden_size,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )
        self.LayerNorm = QLayerNorm(
            config.hidden_size,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            eps=config.layer_norm_eps,
            quant=self.quant,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class CustomBertLayer(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CustomBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError("Cross-attention can only be added when the model is configured as a decoder.")
            self.crossattention = CustomBertAttention(config)
        self.intermediate = CustomBertIntermediate(config)
        self.output = CustomBertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    "If encoder_hidden_states are passed, the layer must be instantiated with cross-attention enabled."
                )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        return (layer_output,) + outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class CustomBertEncoder(QuantizedBertModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if value is not None
            )

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class CustomBertEmbeddings(QuantizedBertModule):
    """Construct embeddings from token, position, and token-type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = QLayerNorm(
            config.hidden_size,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            eps=config.layer_norm_eps,
            quant=self.quant,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class CustomBertModel(QBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = CustomBertEmbeddings(config)
        self.encoder = CustomBertEncoder(config)
        self.pooler = CustomBertPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class CustomBertForSequenceClassification(QBertPreTrainedModel):
    """Custom quantized BERT classifier that can load standard BERT sequence-classification checkpoints."""

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = CustomBertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = QuantizedLinear(
            config.hidden_size,
            config.num_labels,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=self.quant,
        )
        self.post_init()

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
            token_type_ids=token_type_ids,
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


class CustomBertForRTE(CustomBertForSequenceClassification):
    """RTE-specific alias for the custom quantized BERT classifier."""

