import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, QuestionAnsweringModelOutput
from transformers.pytorch_utils import apply_chunking_to_forward
# try:
#     from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# except ImportError:
#     from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

try:
    from transformers.utils.modeling_utils import (
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
except ImportError:
    from transformers.pytorch_utils import prune_linear_layer

    def find_pruneable_heads_and_indices(
        heads, n_heads, head_size, already_pruned_heads
    ):
        import torch

        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads

        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0

        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

from .quantized_layers import QuantizedLinear, QuantizedMatmul, IntSoftmaxTS, qHadamardProd, IntGeluTS, QLayerNorm

DEFAULT_QUANT_PARAMS = {
    "quant": False,
    "nof_bits_linear1": 8,
    "nof_bits_linear2": 8,
    "nof_bits_gelu": 8,
    "lut_size_gelu": 16,
    "nof_bits_softmax": 8,
    "lut_size_softmax": 7,
    "nof_bits_lnorm1": 12,
    "nof_bits_lnorm2": 12,
    "nof_bits_matmul1": 8,
    "nof_bits_matmul2": 8,
}

DEFAULT_MODULE_QUANT_PARAMS = DEFAULT_QUANT_PARAMS.copy()
DEFAULT_MODEL_QUANT_PARAMS = DEFAULT_QUANT_PARAMS.copy()


def set_default_quant_params(params, target="all"):
    """Set quantization defaults used by subsequently constructed modules."""
    valid_keys = set(DEFAULT_MODULE_QUANT_PARAMS) | set(DEFAULT_MODEL_QUANT_PARAMS)
    unknown_keys = sorted(set(params) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown quantization parameter(s): {unknown_keys}")

    if target in ("all", "module"):
        DEFAULT_MODULE_QUANT_PARAMS.update({
            key: value for key, value in params.items()
            if key in DEFAULT_MODULE_QUANT_PARAMS
        })
    if target in ("all", "model"):
        DEFAULT_MODEL_QUANT_PARAMS.update({
            key: value for key, value in params.items()
            if key in DEFAULT_MODEL_QUANT_PARAMS
        })


def _apply_quant_defaults(module, defaults):
    for key, value in defaults.items():
        setattr(module, key, value)

"""# Bert model

## Base Bert Model
"""

# Quantization parameters class
class QauntParams(nn.Module):
    def __init__(self):
        super().__init__()
        _apply_quant_defaults(self, DEFAULT_MODULE_QUANT_PARAMS)


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
            # if type(m) in [QuantizedLinear]:
                print("I WAS HERE :)")
                m.set_scale_opt()

    def unset_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
            # if type(m) in [QuantizedLinear]:
                m.unset_scale_opt()

class QBertPreTrainedModel(BertPreTrainedModel):
    # all_tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.q_module_list = []
        _apply_quant_defaults(self, DEFAULT_MODEL_QUANT_PARAMS)

    def set_q_module_list(self, q_module_list):
        self.q_module_list = q_module_list

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
            # if type(m) in [QuantizedLinear]:
                m.set_scale_opt()

    def unset_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
            # if type(m) in [QuantizedLinear]:
                m.unset_scale_opt()

class CustomBertPooler(QauntParams):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = QuantizedLinear(config.hidden_size,
                                    config.hidden_size,
                                    nof_bits1=self.nof_bits_linear1,
                                    nof_bits2=self.nof_bits_linear2,
                                    quant=self.quant)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CustomBertSelfOutput(QauntParams):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = QuantizedLinear(config.hidden_size,
                            config.hidden_size,
                            nof_bits1=self.nof_bits_linear1,
                            nof_bits2=self.nof_bits_linear2,
                            quant=self.quant)

        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = QLayerNorm(config.hidden_size,
                                    in1_bits=self.nof_bits_lnorm1,
                                    in2_bits=self.nof_bits_lnorm2,
                                    eps=config.layer_norm_eps,
                                    quant=self.quant)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CustomBertSelfAttention(QauntParams):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = QuantizedLinear(config.hidden_size,
                                    self.all_head_size,
                                    nof_bits1=self.nof_bits_linear1,
                                    nof_bits2=self.nof_bits_linear2,
                                    quant=self.quant)

        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = QuantizedLinear(config.hidden_size,
                                    self.all_head_size,
                                    nof_bits1=self.nof_bits_linear1,
                                    nof_bits2=self.nof_bits_linear2,
                                    quant=self.quant)

        # self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = QuantizedLinear(config.hidden_size,
                                    self.all_head_size,
                                    nof_bits1=self.nof_bits_linear1,
                                    nof_bits2=self.nof_bits_linear2,
                                    quant=self.quant)

        self.mat_mul_qk = QuantizedMatmul(in1_bits=self.nof_bits_matmul1,
                                          in2_bits=self.nof_bits_matmul2,
                                          quant=self.quant)

        self.sf = IntSoftmaxTS(nof_bits=self.nof_bits_softmax,
                               LUT_SIZE=self.lut_size_softmax,
                               dim=-1,
                               quant=self.quant)

        self.mat_mul_pv = QuantizedMatmul(in1_bits=self.nof_bits_matmul1,
                                          in2_bits=self.nof_bits_matmul2,
                                          quant=self.quant)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.stats = dict()
        self.stats[f'attn'] = []

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # print("Input shape to transpose_for_scores:", x.shape)
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

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
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

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.mat_mul_qk(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.sf(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.mat_mul_pv(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # self.stats['attn'].append(attention_probs)

        return outputs

class CustomBertAttention(QauntParams):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = CustomBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
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
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class CustomBertIntermediate(QauntParams):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense = QuantizedLinear(config.hidden_size,
                                    config.intermediate_size,
                                    nof_bits1=self.nof_bits_linear1,
                                    nof_bits2=self.nof_bits_linear2,
                                    quant=self.quant)

        if isinstance(config.hidden_act, str):
            # self.intermediate_act_fn = ACT2FN[config.hidden_act]
            self.intermediate_act_fn = IntGeluTS(quant=self.quant,
                                                LUT_SIZE=self.lut_size_gelu,
                                                nof_bits=self.nof_bits_gelu,
                                                hidden_act=config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CustomBertOutput(QauntParams):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense = QuantizedLinear(config.intermediate_size,
                            config.hidden_size,
                            nof_bits1=self.nof_bits_linear1,
                            nof_bits2=self.nof_bits_linear2,
                            quant=self.quant)

        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = QLayerNorm(config.hidden_size,
                                    in1_bits=self.nof_bits_lnorm1,
                                    in2_bits=self.nof_bits_lnorm2,
                                    eps=config.layer_norm_eps,
                                    quant=self.quant)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CustomBertLayer(QauntParams):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CustomBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
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
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class CustomBertEncoder(QauntParams):
    def __init__(self, config):
        super().__init__()
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
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class CustomBertEmbeddings(QauntParams):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = QLayerNorm(config.hidden_size,
                                    in1_bits=self.nof_bits_lnorm1,
                                    in2_bits=self.nof_bits_lnorm2,
                                    eps=config.layer_norm_eps,
                                    quant=self.quant)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
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
        embeddings = self.dropout(embeddings)
        return embeddings

class CustomBertModel(QBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `At\tention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = CustomBertEmbeddings(config)
        self.encoder = CustomBertEncoder(config)

        self.pooler = CustomBertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
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
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
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

"""## Bert Masked Language Modeling"""

class CustomBertPredictionHeadTransform(QauntParams):
    def __init__(self, config, quant=True):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = QuantizedLinear(config.hidden_size,
                    config.hidden_size,
                    nof_bits1=self.nof_bits_linear1,
                    nof_bits2=self.nof_bits_linear2,
                    quant=self.quant)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      # eps=config.layer_norm_eps)
        self.LayerNorm = QLayerNorm(config.hidden_size,
                                    in1_bits=self.nof_bits_lnorm1,
                                    in2_bits=self.nof_bits_lnorm2,
                                    eps=config.layer_norm_eps,
                                    quant=self.quant)

        # self.LayerNorm = QLayerNorm(config.hidden_size,
        #                         in1_bits=16,
        #                         in2_bits=16,
        #                         quant=quant)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class CustomBertLMPredictionHead(QauntParams):
    def __init__(self, config):
        super().__init__()
        self.transform = CustomBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder = QuantizedLinear(config.hidden_size,
                                      config.vocab_size,
                                      nof_bits1=self.nof_bits_linear1,
                                      nof_bits2=self.nof_bits_linear2,
                                      quant=self.quant)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class CustomBertOnlyMLMHead(QauntParams):
    def __init__(self, config):
        super().__init__()
        self.predictions = CustomBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class CustomBertLMHeadModel(QBertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # if not config.is_decoder:
        #     logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = CustomBertModel(config, add_pooling_layer=False)
        self.cls = CustomBertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # input_ids=input_ids, labels=labels
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

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
            # if type(m) in [QuantizedLinear]:
                m.set_scale_opt()

    def unset_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
            # if type(m) in [QuantizedLinear]:
                m.unset_scale_opt()

"""## Question Answering

"""

class CustomBertForQuestionAnswering(QBertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = CustomBertModel(config, add_pooling_layer=False)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs = QuantizedLinear(config.hidden_size,
                                          config.num_labels,
                                          nof_bits1=self.nof_bits_linear1,
                                          nof_bits2=self.nof_bits_linear2,
                                          quant=self.quant)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
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

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
            # if type(m) in [QuantizedLinear]:
                m.set_scale_opt()

    def unset_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
            # if type(m) in [QuantizedLinear]:
                m.unset_scale_opt()
