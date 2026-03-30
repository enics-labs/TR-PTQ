"""Question answering models for NLP workflows."""

from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from ptq_tr.quantization.modules import QuantizedLinear
from ptq_tr.quantization.observers.base import ObserverBase


class CustomBertForQuestionAnswering(BertPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.quant = getattr(config, "quant", False)
        self.q_module_list = [QuantizedLinear]

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = QuantizedLinear(
            config.hidden_size,
            config.num_labels,
            nof_bits1=getattr(config, "nof_bits_linear1", 8),
            nof_bits2=getattr(config, "nof_bits_linear2", 8),
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
        start_positions=None,
        end_positions=None,
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

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

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
