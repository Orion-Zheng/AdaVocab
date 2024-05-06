import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

ADA_RATIO = 4
ADA_TOPK = 20
ADA_LOSS_WEIGHT = 0.1

class AdaVocabHead(nn.Module):  # The same as LoRALayer
    def __init__(self, hidden_size, vocab_size, sub_vocab_dim):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(sub_vocab_dim).float())
        self.A = nn.Parameter(torch.randn(hidden_size, sub_vocab_dim) * std_dev)
        self.B = nn.Parameter(torch.zeros(sub_vocab_dim, vocab_size))

    def forward(self, x):
        # x.shape: (..., hidden_size), A.shape: (hidden_size, sub_vocab_dim), B.shape: (sub_vocab_dim, vocab_size)
        ada_vocab_logits = x @ self.A @ self.B  # ada_vocab_logits.shape: (..., vocab_size)
        return ada_vocab_logits

    
class AdaVocabLlamaForCausalLM(LlamaForCausalLM):  # For Training(train with LM Head)
    # TODO: Check the function of this variable and if it affects the AdaVocab Head model
    _tied_weights_keys = ["lm_head.weight"]  

    def __init__(self, config):
        super().__init__(config)
        self.sub_vocab_dim = config.vocab_size // ADA_RATIO  
        self.topK = ADA_TOPK
        # AdaVocabHead is already initialized with random weights, 
        # so no need to use `self.post_init` method after this
        self.adavocab_head = AdaVocabHead(config.hidden_size, 
                                          config.vocab_size, 
                                          self.sub_vocab_dim)
        
    def topk_mask(self, logits):
        # logits.shape: (batch_size, seq_len, vocab_size)
        topk_values, topk_indices = torch.topk(logits, self.topK, dim=-1)
        # topk_values.shape, topk_indices.shape: (batch_size, seq_len, topK)
        mask = torch.zeros_like(logits)  # (batch_size, seq_len, vocab_size)
        # Only in top-k positions, put 1 to the corresponding position
        mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(mask))

        return mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # TODO: How does forward know whether is training or inference?
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # hidden_states.shape: (batch_size, seq_len, hidden_size)
        # ------ Only for Training ------
        # During Inference, we don't need self.lm_head in GPU memory
        # lm_logits.shape: (batch_size, seq_len, vocab_size)  
        lm_logits = self.lm_head(hidden_states)   
        lm_logits = lm_logits.float()
        # -------------------------------
        # ada_logits.shape: (batch_size, seq_len, vocab_size)
        ada_logits = self.adavocab_head(hidden_states)
        ada_logits = ada_logits.float()
        

        loss = None
        if labels is not None:
            # Supervised Signal of `self.adavocab_head` from two sources: 
            # 1. (Primary) BCEWithLogitsLoss between ada_logits and topk_gt_mask (distillation signal)
            # 2. CrossEntropyLoss between ada_logits and labels (from ground truth vocab)
            
            # Loss from the first source
            # Shift so that tokens < n predict n
            shift_logits = ada_logits[..., :-1, :].contiguous()  # (batch_size, seq_len - 1, vocab_size)
            shift_labels = labels[..., 1:].contiguous()  # (batch_size, seq_len - 1)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()  # CE loss includes the softmax function
            shift_logits = shift_logits.view(-1, self.config.vocab_size)  # (batch_size * (seq_len - 1), vocab_size)

            shift_labels = shift_labels.view(-1)  # (batch_size * seq_len)
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss_1 = loss_fct(shift_logits, shift_labels)
            
            # Loss from the second source
            mask_loss_fct = BCEWithLogitsLoss()  # BCE Loss includes the sigmoid function
            # topk_gt_mask.shape: (batch_size, seq_len, vocab_size)     
            topk_gt_labels = self.topk_mask(lm_logits)  # topk_gt_labels.shape: (batch_size, seq_len, vocab_size)

            loss_2 = mask_loss_fct(ada_logits, topk_gt_labels)
            loss = ADA_LOSS_WEIGHT * loss_1 + loss_2

        if not return_dict:
            output = (ada_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=ada_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
