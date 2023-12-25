# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
import gc
import os
import random
from turtle import hideturtle
import warnings
from ..perlin_attention import get_default_config, PerlinAttentionOutput
from .. import hf_opt
from ...utils import batch_to, get_bench, get_all_allocated_tensors

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.opt.configuration_opt import OPTConfig

from ...utils import strify, checkpoint

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

timer = lambda name: get_bench().region(name)

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)

DEFAULT_METHOD = 'none'

from ..common.lora import LoraLinear, lora_forward

class OPTAttention(nn.Module):
    _counter = 0
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        config: OPTConfig = None,
    ):
        super().__init__()
        
        self.layer_id = OPTAttention._counter
        OPTAttention._counter += 1
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # for project
        self.checkout_intermediates = False
        self.benchmarking = False
        self.attention_method = DEFAULT_METHOD
        
        from ..perlin_attention import PerlinSelfAttention, get_default_config
        pconfig = get_default_config()
        
        if pconfig.lora_enabled:
            self.perlin_out_lora = LoraLinear(embed_dim, embed_dim, pconfig.lora_r)
        
        ### sinkhorn
        from sinkhorn_transformer.sinkhorn_transformer import SinkhornCausalAttention
        if self.attention_method == 'sinkhorn' or DEFAULT_METHOD == 'any':
            if os.environ.get("PERLIN_IGNORE_SINKHORN", "0") == "0":
                self.perlin_sinkhorn_atten = SinkhornCausalAttention(
                    bucket_size=pconfig.k,
                    dim=self.embed_dim,
                    dim_heads=self.head_dim,
                    heads=self.num_heads,
                    max_seq_len=2048,
                    dropout=dropout,
                )
            else:
                warnings.warn("sinkhorn ignored")
            
        ### cosformer
        from ..cosformer import CosformerAttention
        if self.attention_method == 'cosformer' or DEFAULT_METHOD == 'any':
            if os.environ.get("PERLIN_IGNORE_COSFORMER", "0") == "0":
                self.perlin_cosformer_atten = CosformerAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    has_outproj=False,
                    causal=True,
                )
            else:
                warnings.warn("cosformer ignored")
        
        ### reformer
        from reformer_pytorch.reformer_pytorch import LSHAttention
        
        if self.attention_method == 'reformer' or DEFAULT_METHOD == 'any':
            self.perlin_reformer_atten = LSHAttention(
                dropout=dropout,
                bucket_size=32, # this will re adjust atomatically
                n_hashes=pconfig.reformer_n_hashs,
                return_attn=False,
                causal=True,
            )
        
        ### perlin
        from transformers.models.bert.configuration_bert import BertConfig
        
        pconfig.causal = True
        if pconfig.k_flatten_dim == 'batch':
            warnings.warn("Perlin default config's k_flatten_dim is batch. However, you try to initialize causal attention, therefore silently replace to causal_batch")
            pconfig.k_flatten_dim = 'causal_batch'
        pconfig.check_validity()
        self.pconfig = pconfig
        
        self.teacher_attention_scores = None
        self.teacher_context_layer = None
        self.perlin_self_attention = PerlinSelfAttention(
            BertConfig(hidden_size=embed_dim, num_attention_heads=num_heads, max_position_embeddings=config.max_position_embeddings),
            perlin_config=self.pconfig,
        )
        self.last_loss = None
        self.checkout_perlin_output = False
        self.last_perlin_output = None
        self.swap_out_device = None
        
        ### tree attention
        from ..tree_attention.attention import TreeAttention
        self.tree_attention = TreeAttention(
            causal=True,
            k=128,
            start_w=1024,
            w=64,
            scale_up=2.0,
            oversample=1.0,
        )
        
        from ..tree_attention.attention2 import TreeAttention
        self.tree_attention2 = TreeAttention(
            causal=True
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, last_state: object):
        if self.checkout_intermediates:
            self.last_q = q
            self.last_k = k
            self.last_v = v
            self.last_attention_mask = attention_mask
        
        op_dtype = q.dtype
        N_H, T_DST, _HID_Q = q.shape
        N_H, T_SRC, _HID_V = v.shape
        HID = self.head_dim
        assert _HID_V == _HID_Q
        assert HID == _HID_V
        H = self.num_heads
        N = N_H // self.num_heads
        
        # if self.layer_id == 0:
        #     print(f'attention(q={q.shape}, kv={v.shape}, m={attention_mask.shape})')
        
        if self.attention_method == "none":
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_weights.size() != (N * H, T_DST, T_SRC):
                raise ValueError(
                    f"Attention weights should be of size {(N * H, T_DST, T_SRC)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (N, 1, T_DST, T_SRC):
                    raise ValueError(
                        f"Attention mask should be of size {(N, 1, T_DST, T_SRC)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights.view(N, H, T_DST, T_SRC) + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                )
                attn_weights = attn_weights.view(N * H, T_DST, T_SRC)

            # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
            if attn_weights.dtype == torch.float16:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
            else:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            if True:
                # this operation is a bit awkward, but it's required to
                # make sure that attn_weights keeps its gradient.
                # In order to do so, attn_weights have to be reshaped
                # twice and have to be reused in the following
                attn_weights_reshaped = attn_weights.view(N, H, T_DST, T_SRC)
                attn_weights = attn_weights_reshaped.view(N * H, T_DST, T_SRC)
            else:
                attn_weights_reshaped = None

            attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_probs, v)

            attn_output = attn_output.view(N, H, T_DST, HID)
            attn_output = attn_output.transpose(1, 2)

            # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
            # partitioned aross GPUs when using tensor-parallelism.
            attn_output = attn_output.reshape(N, T_DST, H*HID)
            
            return attn_output, attn_weights_reshaped
        elif self.attention_method == 'reformer':
            # assert T_SRC == T_DST, "need to fix later"
            
            q = q.view(N, H, T_DST, HID)
            k = k.view(N, H, T_SRC, HID)
            v = v.view(N, H, T_SRC, HID)
            
            N, H, T_DST, HID = q.shape
            # v = v * (attention_mask[:,:,:1,:].transpose(-1, -2) > -1)
            
            binary_mask = attention_mask > -1
            
            #pad
            bucket_size = self.pconfig.k
            pad_unit_size = bucket_size * 2
            to_pad_src = 0 if (T_SRC % pad_unit_size) == 0 else (pad_unit_size - (T_SRC % pad_unit_size))
            TP_SRC = T_SRC + to_pad_src
            to_pad_dst = 0 if (T_DST % pad_unit_size) == 0 else (pad_unit_size - (T_DST % pad_unit_size))
            TP_DST = T_DST + to_pad_dst
            if to_pad_src != 0 or to_pad_dst != 0:
                q = F.pad(q, (0,0,0,to_pad_dst)).float()
                v = F.pad(v, (0,0,0,to_pad_src)).float()
                binary_mask = F.pad(binary_mask.expand(N, H, T_DST, T_SRC), (0,to_pad_src, 0,to_pad_dst), value=0.0).bool().reshape(N*H, TP_DST, TP_SRC)
                assert q.shape == (N, H, T_DST + to_pad_dst, HID)
            else:
                q = q.float()
                v = v.float()
                binary_mask = binary_mask.expand(N, H, T_DST, T_SRC).bool().reshape(N*H, T_DST, T_SRC)
            def merge_head(t: torch.Tensor):
                N, H, T, HID = t.shape
                # return t.permute(0, 2, 1, 3).contiguous().view(N, T, H*HID)
                return t.permute(0, 1, 2, 3).contiguous().view(N*H, T, HID)
            q = merge_head(q)
            v = merge_head(v)
            self.perlin_reformer_atten.bucket_size = bucket_size
            reformer_context_layer, _,_ = self.perlin_reformer_atten(
                q, 
                v, 
                input_attn_mask = binary_mask
            )
            reformer_context_layer = reformer_context_layer.to(op_dtype)
            #unpad
            if to_pad_src != 0 or to_pad_dst != 0:
                q = None
                v = None
                reformer_context_layer = reformer_context_layer.reshape(N, H, TP_DST, HID)#.permute(0, 2, 1, 3)
                reformer_context_layer = reformer_context_layer[:, :, :T_DST, :]
            else:
                reformer_context_layer = reformer_context_layer.reshape(N, H, T_DST, HID)#.permute(0, 2, 1, 3)
            
            if not self.benchmarking:
                attention_probs = torch.zeros((N, H, T_DST, T_SRC), device=reformer_context_layer.device, dtype=reformer_context_layer.dtype)
            else:
                attention_probs = None
            
            reformer_context_layer = reformer_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = reformer_context_layer.size()[:-2] + (H*HID,)
            reformer_context_layer = reformer_context_layer.view(new_context_layer_shape)
            
            return reformer_context_layer, attention_probs
        elif self.attention_method == 'performer':
            # assert T_SRC == T_DST
            
            q = q.view(N, H, T_DST, HID)
            k = k.view(N, H, T_SRC, HID)
            v = v.view(N, H, T_SRC, HID)
            
            N, H, T, HID = q.shape
            v = v * (attention_mask[:,:,:,:1] > -1)
            
            # self.perlin_performer_proj_updater.redraw_projections(q.device)
            with torch.autocast('cuda', torch.float32):
                performer_context_layer = self.perlin_self_attention.attention.performer(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
            performer_context_layer = performer_context_layer.to(op_dtype)
            
            if not self.benchmarking:
                attention_probs = torch.zeros((N, H, T_DST, T_SRC), dtype=performer_context_layer.dtype, device=performer_context_layer.device)
            else:
                attention_probs = None
            
            performer_context_layer = performer_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = performer_context_layer.size()[:-2] + (self.embed_dim,)
            performer_context_layer = performer_context_layer.view(new_context_layer_shape)
            
            context_layer = performer_context_layer
            
            return context_layer, attention_probs
        elif self.attention_method == 'cosformer':
            # assert T_SRC == T_DST
            
            q = q.view(N, H, T_DST, HID)
            k = k.view(N, H, T_SRC, HID)
            v = v.view(N, H, T_SRC, HID)
            
            N, H, T, HID = q.shape
            v = v * (attention_mask[:,:,:,:1] > -1)
            
            # self.perlin_performer_proj_updater.redraw_projections(q.device)
            with torch.autocast('cuda', torch.float32):
                t = self.perlin_cosformer_atten(
                    q.permute(0, 2, 1, 3).reshape(N, T_DST, H*HID).permute(1, 0, 2).to(torch.float32), 
                    k.permute(0, 2, 1, 3).reshape(N, T_SRC, H*HID).permute(1, 0, 2).to(torch.float32), 
                    v.permute(0, 2, 1, 3).reshape(N, T_SRC, H*HID).permute(1, 0, 2).to(torch.float32)
                )
                performer_context_layer = t.reshape(T, N, H, HID).permute(1, 2, 0, 3)
            performer_context_layer = performer_context_layer.to(op_dtype)
            
            if not self.benchmarking:
                attention_probs = torch.zeros((N, H, T_DST, T_SRC), dtype=performer_context_layer.dtype, device=performer_context_layer.device)
            else:
                attention_probs = None
            
            performer_context_layer = performer_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = performer_context_layer.size()[:-2] + (self.embed_dim,)
            performer_context_layer = performer_context_layer.view(new_context_layer_shape)
            
            context_layer = performer_context_layer
            
            return context_layer, attention_probs
        elif self.attention_method == 'perlin':
            q = q.view(N, H, T_DST, HID)
            k = k.view(N, H, T_SRC, HID)
            v = v.view(N, H, T_SRC, HID)
            
            if callable(self.teacher_context_layer):
                context_layer_truth = lambda: self.teacher_context_layer()\
                        .view(N, H, T_DST, HID)\
                        .transpose(1, 2)\
                        .reshape(N, T_DST, H*HID) \
                    if self.teacher_context_layer is not None \
                    else None
            else:
                context_layer_truth = self.teacher_context_layer\
                        .view(N, H, T_DST, HID)\
                        .transpose(1, 2)\
                        .reshape(N, T_DST, H*HID) \
                    if self.teacher_context_layer is not None \
                    else None
            
            output = self.perlin_self_attention(
                query = self.q_proj,
                key = self.k_proj,
                value = self.v_proj,
                hidden_states = None,
                query_layer = q,
                key_layer = k,
                value_layer = v,
                attention_mask = attention_mask,
                attention_scores_truth = self.teacher_attention_scores,
                context_layer_truth = context_layer_truth,
                last_state = last_state,
            ) #type: PerlinAttentionOutput
            # return q, None, None #TODO REMOVE
            self.last_loss = output.loss
            if not self.benchmarking and self.checkout_perlin_output:
                warnings.warn("you are checking-out LARGE buffers!!")
                if self.swap_out_device is None:
                    self.last_perlin_output = output
                else:
                    self.last_perlin_output = output.to(self.swap_out_device)
                    print('OptAtten: swap out')
            
            return output.context_layer, output.partial_attention_probs, output.state
        elif self.attention_method == 'sinkhorn':
            # assert T_SRC == T_DST
            
            q = q.view(N, H, T_DST, HID)
            k = k.view(N, H, T_SRC, HID)
            v = v.view(N, H, T_SRC, HID)
            
            N, H, T, HID = q.shape
            v = v * (attention_mask[:,:,:,:1] > -1)
            
            binary_mask = attention_mask > -1
            
            #pad
            to_pad = 0
            # perlin_k = self.perlin_self_attention.pconfig.k
            # to_pad = 0 if (T % perlin_k) == 0 else (perlin_k - (T % perlin_k))
            # TP = T + to_pad
            # if to_pad != 0:
            #     pad_config = (0,0,0,to_pad)
            #     q = F.pad(q, pad_config).float()
            #     k = F.pad(k, pad_config).float()
            #     v = F.pad(v, pad_config).float()
            #     binary_mask = F.pad(binary_mask.expand(N, 1, 1, T), (0,to_pad), value=0.0).bool().view(N, TP)
            #     assert q.shape == (N, H, T+to_pad, HID)
            #     # assert binary_mask.shape == (N, T+to_pad)
            # else:
            #     q = q.float()
            #     k = k.float()
            #     v = v.float()
            #     binary_mask = binary_mask.bool().view(1, TP)
            op_dtype = q.dtype
            with torch.autocast('cuda', torch.float32):
                sinkhorn_context_layer = self.perlin_sinkhorn_atten(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
            if op_dtype != sinkhorn_context_layer.dtype:
                sinkhorn_context_layer = sinkhorn_context_layer.to(op_dtype)
            
            #unpad
            if to_pad != 0:
                q = q[:, :, :T, :]
                k = k[:, :, :T, :]
                v = v[:, :, :T, :]
                sinkhorn_context_layer = sinkhorn_context_layer[:, :, :T, :]
            
            if not self.benchmarking:
                attention_probs = torch.zeros((N, H, T, T), device=q.device, dtype=q.dtype)
            else:
                attention_probs = None
            
            sinkhorn_context_layer = sinkhorn_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = sinkhorn_context_layer.size()[:-2] + (self.embed_dim,)
            sinkhorn_context_layer = sinkhorn_context_layer.view(new_context_layer_shape)
            
            return sinkhorn_context_layer, attention_probs
        elif self.attention_method == "tree":
            q = q.view(N, H, T_DST, HID)
            k = k.view(N, H, T_SRC, HID)
            v = v.view(N, H, T_SRC, HID)
            
            context = self.tree_attention(q, k, v, attention_mask)
            context = context.permute(0, 2, 1, 3).contiguous().view(N, T_DST, H*HID)
            
            attention_probs = None
            return context, attention_probs
        elif self.attention_method == "tree2":
            q = q.view(N, H, T_DST, HID)
            k = k.view(N, H, T_SRC, HID)
            v = v.view(N, H, T_SRC, HID)
            
            context = self.tree_attention2(q, k, v, attention_mask)
            context = context.permute(0, 2, 1, 3).contiguous().view(N, T_DST, H*HID)
            attention_probs = None
            return context, attention_probs
        else:
            raise Exception()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        
        # deepspeed fp32 support, convert fp32
        op_dtype = torch.float16 if self.q_proj.weight.dtype == torch.float16 else hidden_states.dtype
        if op_dtype != hidden_states.dtype:
            hidden_states = hidden_states.to(op_dtype)
        if attention_mask is not None and op_dtype != attention_mask.dtype:
            attention_mask = torch.clamp_min(attention_mask, torch.finfo(op_dtype).min).to(op_dtype)
        if layer_head_mask is not None and op_dtype != layer_head_mask.dtype:
            layer_head_mask = torch.clamp_min(layer_head_mask, torch.finfo(op_dtype).min).to(op_dtype)

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * torch.tensor(self.scaling, dtype=op_dtype, device=hidden_states.device)
        self.q_proj.scaling = torch.tensor(self.scaling, dtype=op_dtype, device=hidden_states.device)
        
        past_state = None
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_state = past_key_value[2] if len(past_key_value) > 2 else None
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        assert attention_mask is not None
        assert layer_head_mask is None
        
        # print(query_states.dtype, key_states.dtype)

        outputs = self.attention(
            q=query_states,
            k=key_states,
            v=value_states,
            attention_mask=attention_mask,
            last_state=past_state,
        )
        
        if len(outputs) == 2:
            attn_output, attn_weights_reshaped = outputs
            attn_state = None
        elif len(outputs) == 3:
            attn_output, attn_weights_reshaped, attn_state = outputs
        else: raise Exception()
        
        if not output_attentions:
            attn_weights_reshaped = None

        if attn_output.dtype != op_dtype:
            attn_output = attn_output.to(op_dtype)
        
        if attn_state is not None:
            past_key_value = (*past_key_value, attn_state)

        if self.pconfig.lora_enabled:
            attn_output = lora_forward(self.out_proj, self.perlin_out_lora, attn_output, True)
        else:
            attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            config=config
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)
        
        self._gradient_checkpointing = False
        self.last_loss = None
        self.train_layerwise = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        
        # to handle cpu checkpointing
        device = attention_mask.device
        if device != hidden_states.device:
            hidden_states = batch_to(hidden_states, device)
        
        # deepspeed fp32 support, convert fp32
        op_dtype = torch.float16 if self.fc1.weight.dtype == torch.float16 else hidden_states.dtype
        if op_dtype != hidden_states.dtype:
            hidden_states = hidden_states.to(op_dtype)
        if attention_mask is not None and op_dtype != attention_mask.dtype:
            attention_mask = torch.clamp_min(attention_mask, torch.finfo(op_dtype).min).to(op_dtype)
        if layer_head_mask is not None and op_dtype != layer_head_mask.dtype:
            layer_head_mask = torch.clamp_min(layer_head_mask, torch.finfo(op_dtype).min).to(op_dtype)
        
        # if layerwise, detach!
        if self.train_layerwise:
            if hidden_states.requires_grad:
                hidden_states = hidden_states.detach()
            if attention_mask is not None and attention_mask.requires_grad:
                attention_mask = attention_mask.detach()
            if layer_head_mask is not None and layer_head_mask.requires_grad:
                layer_head_mask = layer_head_mask.detach()
            assert past_key_value is None
        
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            def before_self_atten(hidden_states):
                hidden_states = self.self_attn_layer_norm(hidden_states)
                return hidden_states
            
            if self._gradient_checkpointing and self.training:
                hidden_states = checkpoint.checkpoint(before_self_atten, hidden_states, preserve_rng_state=True)
            else:
                hidden_states = before_self_atten(hidden_states)
    
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # return
        
        def after_self_atten(residual, hidden_states):
            # deepspeed fp32 support, convert fp32
            op_dtype = torch.float16 if self.fc1.weight.dtype == torch.float16 else hidden_states.dtype
            if op_dtype != hidden_states.dtype:
                hidden_states = hidden_states.to(op_dtype)
            
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # print('aa', hidden_states.dtype)
            hidden_states = residual + hidden_states
            # print('faa', hidden_states.dtype)

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)
            # print('saa', hidden_states.dtype)

            # Fully Connected
            hidden_states_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            residual = hidden_states
            # print('00aa', hidden_states.dtype)

            # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)
            
            if op_dtype != hidden_states:
                hidden_states = hidden_states.to(op_dtype)
            # print('aaff3', hidden_states.dtype)
            
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)

            hidden_states = self.fc2(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            hidden_states = (residual + hidden_states).view(hidden_states_shape)

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)
                
            return hidden_states
        
        if self._gradient_checkpointing and self.training:
            hidden_states = checkpoint.checkpoint(after_self_atten, residual, hidden_states, preserve_rng_state=True)
        else:
            hidden_states = after_self_atten(residual, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OPTDecoder)):
            module.gradient_checkpointing = value


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        # print(config)
        
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
        self.swap_in_device = None
        self.swap_out_device = None
        self.use_deepspeed = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        
        # print(strify(past_key_values))
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        swap_in_device = self.swap_in_device
        swap_out_device = self.swap_out_device
        
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            decoder_layer = decoder_layer # type: OPTDecoderLayer
            if output_hidden_states:
                # NOTE: this costs some memory... we need to avoid store this in GPU, but offloading this value cause error on ZeRO3
                if self.use_deepspeed:
                    all_hidden_states += (hidden_states,)
                else:
                    if swap_out_device is not None:
                        all_hidden_states += (batch_to(hidden_states, swap_out_device),)
                    else:
                        all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # start_set = set([id(t) for t, r in get_all_allocated_tensors()])
                start_mem = torch.cuda.max_memory_allocated()
                # print('processing layer', idx, torch.cuda.max_memory_allocated() // 1024 // 1024)
                
                def create_custom_forward(module: OPTDecoderLayer):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        if len(inputs) == 2:
                            inputs = inputs + (None, None)
                        states = module(*inputs, output_attentions, None)
                        loss = module.self_attn.last_loss
                        if loss is None:
                            loss = torch.zeros((1,), device=inputs[0].device)
                        assert loss is not None
                        return states + (loss,)

                    return custom_forward
                
                decoder_layer.self_attn.swap_out_device = swap_out_device
                
                if self.use_deepspeed and (not decoder_layer.self_attn.perlin_self_attention.pconfig.layerwise):
                    import deepspeed
                    # deepspeed.checkpointing.PARTITION_ACTIVATIONS = True
                    # deepspeed.checkpointing.CPU_CHECKPOINT = True
                    # deepspeed.checkpointing.CONTIGUOUS_CHECKPOINTING = True
                    
                    assert (head_mask[idx] if head_mask is not None else None) == None
                    
                    # NOTE: None is not accepted, for cpu checkpointing
                    deepspeed.checkpointing.set_num_layers(len(self.layers))
                    layer_outputs = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        causal_attention_mask,
                    )
                    
                    # layer_outputs = batch_to(layer_outputs, swap_out_device)
                else:
                    layer_outputs = checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        causal_attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        None,
                        use_reentrant=False,
                        swap_out_device=swap_out_device,
                        swap_in_device=swap_in_device,
                    )
                
                if not self.use_deepspeed:
                    layer_outputs = batch_to(layer_outputs, swap_out_device)
                
                assert isinstance(layer_outputs, (tuple, list))
                layer_outputs, layer_loss = layer_outputs[:-1], layer_outputs[-1]
                decoder_layer.self_attn.last_loss = None
                decoder_layer.last_loss = layer_loss
                
                # print('processed layer', idx, torch.cuda.max_memory_allocated() // 1024 // 1024, (torch.cuda.max_memory_allocated() - start_mem) // 1024 // 1024)
                
                # torch.cuda.synchronize()
                # gc.collect()
                # torch.cuda.empty_cache()
                # torch.cuda.reset_peak_memory_stats()
                
                # tensors = get_all_allocated_tensors()
                # end_set = set([id(t) for t, r in tensors])
                # leak_set = end_set - start_set
                # for t, r in tensors:
                #     if id(t) in leak_set:
                #         print('leak obj', strify(t), t.element_size()*t.numel())
                #         print('refs', [type(i) for i in r])
            else:
                with timer('opt.layer'):
                    # torch.cuda.synchronize()
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # start_mem = torch.cuda.max_memory_allocated()
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    # end_mem = torch.cuda.max_memory_allocated()
                    # torch.cuda.synchronize()
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # print((end_mem - start_mem) / 1024 / 1024)

            hidden_states = layer_outputs[0]

            if use_cache:
                assert not self.gradient_checkpointing
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if hidden_states.device != swap_in_device and swap_in_device is not None:
            hidden_states = batch_to(hidden_states, swap_in_device)
        
        op_dtype = hidden_states.dtype
        
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype != op_dtype:
            hidden_states = hidden_states.to(op_dtype)
        
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # if next_cache is not None:
        #     print(strify(next_cache))
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder
    
    def calc_loss_special(self):
        loss = 0
        weights = 0
        # print('hmm?')
        for m in self.modules():
            if isinstance(m, OPTDecoderLayer) and m.last_loss is not None:
                # print('good')
                loss += m.last_loss
                weights += 1
                m.last_loss = None
        for m in self.modules():
            if isinstance(m, OPTAttention) and m.last_loss is not None:
                # print('hello')
                if weights > 0:
                    warnings.warn(
                        "OPTAttention.last_loss is ignored, because gradient checkpointing activated! "
                        "If you see this message when not grad. chkpt is not activated, please report."
                    )
                else:
                    # print('no')
                    loss += m.last_loss
                    weights += 1
                    m.last_loss = None
        if weights > 0 :
            return loss / weights
        else:
            return 0

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher: "hf_opt.OPTForCausalLM" = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""
        
        if teacher is not None:
            teachers = []
            for layer in teacher.model.decoder.layers:
                layer = layer # type: hf_opt.OPTDecoderLayer
                teachers.append((
                    layer.self_attn.last_attention_scores,
                    layer.self_attn.last_context_layer,
                ))
            for ilayer, layer in enumerate(self.model.decoder.layers):
                layer = layer # type: OPTDecoderLayer
                teacher_attention_scores, teacher_context_layer = teachers[ilayer]
                layer.self_attn.teacher_attention_scores = teacher_attention_scores
                layer.self_attn.teacher_context_layer = teacher_context_layer

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # if random.random() < 1/10 and not get_bench().disabled:
        #     print(get_bench().format_tracetree())
        # else:
        #     print('forward', input_ids.shape)
        
        if not get_bench().disabled:
            print(get_bench().format_tracetree(), flush=True)
            input()

        logits = self.lm_head(outputs[0].to(torch.float16 if self.lm_head.weight.dtype == torch.float16 else outputs[0].dtype)).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value