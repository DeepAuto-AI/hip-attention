import copy
import math
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .masked_mm import sparse_attn

import torch
# import torch.sparse._triton_ops
import torch.nn.functional as F
from performer_pytorch import FastAttention
from torch import nn, optim
# from xformers.components.attention.core import (
#     SparseCS,
#     scaled_dot_product_attention
# )

from ...utils import get_bench, Metric
from ..common.kl_div_for_atten import kl_div_attention
from ..common.lora import (
    LoraLinear, 
    lora_forward, 
    lora_forward_linear,
    lora_forward_lora
)
from ..common.performer import ProjectionUpdater
from ..hf_bert import BertConfig
from .config import PerlinAttentionConfig, get_default_config
from ...utils import raise_if_nan, strify
from .modules import (
    ResBlock,
    Residual,
    KeepRes,
    CausalConv2d,
    UpsampleFP32,
    interpolate
)
from math import ceil, floor
timer = lambda name: get_bench().region(name)

class StatefulCausalPerformer:
    def __init__(self, parent: "PerlinAttentionState", performer: FastAttention):
        self.parent = parent
        self.performer = performer
        
        self.seq_index = 0
        self.last_k_cumsum = 0
        self.last_context_cumsum = 0
        
        self.qs = []
    
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # # naive
        # self.seq_index += q.shape[-2]
        # self.qs.append(q)
        # qs = torch.cat(self.qs, dim=-2)
        
        # #TODO: fix this!!
        # # qs = F.pad(qs, pad=(0,0,0,256-qs.shape[-2]), mode='constant', value=0)
        # # k = F.pad(k, pad=(0,0,0,256-k.shape[-2]), mode='constant', value=0)
        # # v = F.pad(v, pad=(0,0,0,256-v.shape[-2]), mode='constant', value=0)
        
        # # original_causal_fn = self.performer.causal_linear_fn
        # # self.performer.causal_linear_fn = self._causal_linear_attention_noncuda_stateful
        # print('performer', q.shape, qs.shape, k.shape, v.shape)
        # context = self.performer(qs,k,v)
        # # self.performer.causal_linear_fn = original_causal_fn
        # del qs
        
        # out = context[...,-q.shape[-2]:,:].clone()
        # del context
        
        # return out
    
        # # context = self.performer(q, k, v)
        # # return context
        
        last_k_cumsum = 0
        last_context_cumsum = 0
        outs = []
        
        chunk_size = min(q.shape[-2], 16)

        for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k[...,-q.shape[-2]:, :], v[...,-q.shape[-2]:, :]))):
            k_cumsum = self.last_k_cumsum + k.cumsum(dim=-2, dtype=torch.float64)

            D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + 1e-12)
            context = torch.einsum('...nd,...ne->...nde', k, v)
            context_cumsum = self.last_context_cumsum + context.cumsum(dim=-3, dtype=torch.float64)
            out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum.type_as(q), q, D_inv)

            self.last_k_cumsum = k_cumsum[:, :, -1:]
            self.last_context_cumsum = context_cumsum[:, :, -1:]
            outs.append(out)

        return torch.cat(outs, dim = -2)

    def _causal_linear_attention_noncuda_stateful(
        self, q, k, v, chunk_size = None, eps = 1e-6
    ):
        # return k
        
        last_k_cumsum = 0
        last_context_cumsum = 0
        outs = []
        
        chunk_size = q.shape[-2]

        for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
            k_cumsum = last_k_cumsum + k.cumsum(dim=-2, dtype=torch.float64)

            D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
            context = torch.einsum('...nd,...ne->...nde', k, v)
            context_cumsum = last_context_cumsum + context.cumsum(dim=-3, dtype=torch.float64)
            out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum.type_as(q), q, D_inv)

            last_k_cumsum = k_cumsum[:, :, -1:]
            last_context_cumsum = context_cumsum[:, :, -1:]
            outs.append(out)

        return torch.cat(outs, dim = -2)
    
    def strify(self):
        return f"StatePerformer({self.seq_index}, {strify(self.last_k_cumsum)}, {strify(self.last_context_cumsum)})"

    def clone(self):
        new = StatefulCausalPerformer(self.parent, self.performer)
        new.seq_index = self.seq_index
        new.last_context_cumsum = \
            self.last_context_cumsum.clone() \
            if isinstance(self.last_context_cumsum, torch.Tensor) \
            else self.last_context_cumsum
        new.last_k_cumsum = \
            self.last_k_cumsum.clone() \
            if isinstance(self.last_k_cumsum, torch.Tensor) \
            else self.last_k_cumsum
        new.qs = list([q for q in self.qs])
        return new

class StatefulCausalCNN:
    def __init__(self, parent: "PerlinAttentionState"):
        self.parent = parent
        self.max_seq_len = self.parent.max_seq_length
        self.window_size = 24
        self.window_align = 1
        self.xs = []
        self.xs_len = 0
    
    def __call__(self, cnn: torch.nn.Module, x: torch.Tensor, x_len: int):
        # assert x.shape[-2] == x_len, f"{x.shape}[-2] == {x_len}"
        # x = x[...,-x_len:,:]
        
        # torch.cuda.synchronize()
        # t = time.time()
        
        self.xs.append(x)
        self.xs_len += x.shape[-2]
        
        x_start = max(self.xs_len-max(x.shape[-2], self.window_size), 0)
        x_start = max(x_start - (x_start % self.window_align), 0)
        x_window_len = self.xs_len - x_start
        ts = []
        ts_len = 0
        ixs = len(self.xs) - 1
        while ts_len < x_window_len and ixs >= 0:
            ts.append(self.xs[ixs])
            ts_len += self.xs[ixs].shape[-2]
            ixs -= 1
        ts.reverse()
        self.xs = ts
        xs = torch.cat(ts, dim=-2)
        x_window = xs[...,-min(xs.shape[-2], x_window_len):,:]
        
        # if random.random() < 1/100:
        #     print(get_bench().format_tracetree())
        y = cnn(x_window)
        # assert y.shape == x_window.shape, f"{y.shape} == {x_window.shape}"
        # print('cnn dbg', len(self.xs), x.shape, y.shape, x_window.shape, x_start, self.xs_len)
        # print(torch.cuda.max_memory_allocated() / 1024 / 1024)
        # torch.cuda.synchronize()
        # print('cnn', x.shape, x_len, x_window.shape, (time.time() - t) * 1000)
        
        output = y[...,-x_len:,:].clone()
        del xs, x_window, ts, y
        return output
    
    def strify(self):
        return strify({
            'ws': self.window_size,
            'wa': self.window_align,
            'xs': self.xs,
            'xs_len': self.xs_len
        })
    
    def clone(self):
        new = StatefulCausalCNN(self.parent)
        new.window_align = self.window_align
        new.window_size = self.window_size
        new.xs_len = self.xs_len
        new.xs = list([it for it in self.xs]) # shallow copy
        return new

class StatefulCumAvg:
    def __init__(self, parent: "PerlinAttentionState"):
        self.parent = parent
        self.cumsum = 0
        self.prev_len = 0
    
    def __call__(self, v: torch.Tensor, q_len: int):
        N, H, T_SRC, D = v.shape
        # if not isinstance(self.cumsum, torch.Tensor):
        #     self.cumsum = torch.zeros((1, 1, 1, D), device=v.device, dtype=v.dtype)
        
        cs = v[...,-q_len:,:].cumsum(dim=-2) + self.cumsum
        self.cumsum = cs[...,-1:,:].clone()
        
        cs = cs / torch.arange(self.prev_len+1, self.prev_len+1+cs.shape[-2], device=v.device).view(1, 1, -1, 1)
        self.prev_len += q_len
        
        # print(self.prev_len)
        
        return cs
    
    def strify(self):
        return strify({
            'cumsum': self.cumsum,
            'prev_len': self.prev_len
        })
    
    def clone(self):
        new = StatefulCumAvg(self.parent)
        new.cumsum = self.cumsum.clone()
        new.prev_len = self.prev_len
        return new

class PerlinAttentionState:
    def __init__(self, parent: "PerlinAttention"):
        if parent is not None:
            self.num_heads = parent.num_attention_heads
            self.head_dim = parent.attention_head_size
            self.embd_dim = parent.all_head_size
        self.max_seq_length = 768
        
        self.states = {}
    
    def get_cloned_state(self, name: str, initializer=None):
        if name in self.states:
            new_state = self.states[name].clone()
            self.states[name] = new_state
            return new_state
        else:
            state = initializer()
            self.states[name] = state
            return state
    
    @staticmethod
    def stateful_causal_cnn_op(
        state: "PerlinAttentionState",
        name: str,
        func: nn.Module,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ):
        if state is None:
            return None, func(x)
        else:
            with timer(name+'.clone'):
                state = state.clone()
            return state, state.forward_causal_cnn_op(name, func, x, x_len)
    
    def forward_causal_cnn_op(
        self,
        name: str,
        func: nn.Module,
        x: torch.Tensor,
        x_len: int,
    ):
        state = self.get_cloned_state(name, lambda: StatefulCausalCNN(self))
        return state(func, x, x_len)
    
    @staticmethod
    def stateful_performer(
        state: "PerlinAttentionState",
        name: str,
        performer: FastAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        if state is not None:
            with timer(name+'.clone'):
                state = state.clone()
            return state, state.forward_performer(
                name=name,
                performer=performer,
                q=q, k=k, v=v
            )
        else:
            return None, performer(q, k, v)
    
    def forward_performer(
        self,
        name: str,
        performer: FastAttention, 
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        state = self.get_cloned_state(
            name, 
            lambda: StatefulCausalPerformer(self, performer)
        )
        
        return state(
            q=q,
            k=k,
            v=v,
        )
    
    @staticmethod
    def stateful_cumavg(
        state: "PerlinAttentionState",
        name: str,
        v: torch.Tensor,
        q_len: int,
    ):
        if state is not None:
            with timer(name+'.clone'):
                state = state.clone()
            return state, state.forward_cumsum(
                name=name,
                v=v, q_len=q_len
            )
        else:
            raise Exception()
    
    def forward_cumsum(
        self,
        name: str,
        v: torch.Tensor,
        q_len: int,
    ):
        N, H, T_SRC, D = v.shape
        state = self.get_cloned_state(name, lambda: StatefulCumAvg(self))
        
        return state(v=v, q_len=q_len)
    
    def strify(self):
        return f"State({strify(self.states)})"
    
    def clone(self):
        new = PerlinAttentionState(None)
        new.num_heads = self.num_heads
        new.head_dim = self.head_dim
        new.embd_dim = self.embd_dim
        new.max_seq_length = self.max_seq_length
        new.states = {k: v for k, v in self.states.items()}
        return new