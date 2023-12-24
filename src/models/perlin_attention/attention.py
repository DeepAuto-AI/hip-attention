import copy
import math
import os
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

from ...utils import batch_to, get_bench, Metric
from ..common.kl_div_for_atten import kl_div_attention
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
# NOTE comment below to debug NaN
raise_if_nan = lambda x: x
# torch.autograd.set_detect_anomaly(True)

timer = lambda name: get_bench().region(name)
mem = lambda name: get_bench().mem_region(name)
metric = Metric()

def grid_sample_bf16(input, grid, mode='nearest', align_corners=False, padding_mode='zeros', output_dtype=None):
    input_dtype = input.dtype
    op_dtype = torch.float32 if torch.get_autocast_gpu_dtype() == torch.bfloat16 else input_dtype
    if op_dtype != input_dtype:
        input = input.to(op_dtype)
        grid = grid.to(op_dtype)
    y = F.grid_sample(
        input=input,
        grid=grid,
        mode=mode,
        align_corners=align_corners,
        padding_mode='zeros',
    )
    if output_dtype is not None:
        input_dtype = output_dtype
    if y.dtype != input_dtype:
        y = y.to(input_dtype)
    return y

def softmax_bf16(input, dim=-1, training=True):
    if not training:
        return torch.softmax(input, dim=dim)
    input_dtype = input.dtype
    op_dtype = torch.float32 if torch.get_autocast_gpu_dtype() in [torch.bfloat16, torch.float16] else input_dtype
    if op_dtype != input_dtype:
        input = input.to(op_dtype)
    y = torch.softmax(input, dim=-1)
    if y.dtype != input_dtype:
        y = y.to(input_dtype)
    return y

from .attention_state import PerlinAttentionState, StatefulCausalCNN, StatefulCausalPerformer

def safe_to(t, d):
    if isinstance(t, torch.Tensor):
        return t.to(d)
    return t

from typing import NamedTuple

# @dataclass
class PerlinAttentionOutput(NamedTuple):
    loss: torch.Tensor
    context_layer: torch.Tensor
    partial_attention_probs: torch.Tensor
    partial_attention_mask: torch.Tensor
    estimated_attention_probs_m: torch.Tensor
    estimated_attention_probs: torch.Tensor
    dense_attention_probs: torch.Tensor
    key_for_score: torch.Tensor
    state: PerlinAttentionState
    
    def to(self, device):
        return PerlinAttentionOutput(
            loss=safe_to(self.loss, device),
            context_layer=safe_to(self.context_layer, device),
            partial_attention_probs=safe_to(self.partial_attention_probs, device),
            partial_attention_mask=safe_to(self.partial_attention_mask, device),
            estimated_attention_probs_m=safe_to(self.estimated_attention_probs_m, device),
            estimated_attention_probs=safe_to(self.estimated_attention_probs, device),
            dense_attention_probs=safe_to(self.dense_attention_probs, device),
            key_for_score=safe_to(self.key_for_score, device),
            state=safe_to(self.state, device),
        )

class ModuleBenchmark(nn.Module):
    def __init__(self, name, module, disabled=False):
        super().__init__()
        
        self.name = name
        self.module = module
        self.disabled = disabled
    
    def forward(self, x):
        if self.disabled:
            return self.module(x)
        else:
            with timer(self.name):
                return self.module(x)

class ChannelSplit(nn.Module):
    def __init__(self, split):
        super().__init__()
        
        self.split = split
    
    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, C, H, self.split, W // self.split).permute(0, 1, 3, 2, 4).reshape(N, C*self.split, H, W//self.split)

class PerlinAttention(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        perlin_config: PerlinAttentionConfig = None,
    ):
        super().__init__()
    
        self.config = config
        self.pconfig = perlin_config if perlin_config is not None else get_default_config()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self._warning_messages = ""
        
        ### Perlin
        #- configs
        self.benchmarking = False
        
        #- attention predictor
        #-- mlp predictor
        self.performer_nb_features = int(
            self.attention_head_size * math.log(self.attention_head_size) / self.pconfig.performer_nb_factor
        )
        self.performer = FastAttention(
            dim_heads = self.attention_head_size,
            nb_features = self.performer_nb_features,
            causal=self.pconfig.causal,
            generalized_attention=self.pconfig.causal,
        )
        self.performer_proj_updater = ProjectionUpdater(
            self.performer,
            1000,
        )
        if os.environ.get("PERLIN_IGNORE_COSFORMER", "0") == "0" and self.pconfig.attention_predictor_backend == 'cosformer':
            from ..cosformer import CosformerAttention
            # cosformer not supported because of v dim does not supports custom
            self.cosformer = CosformerAttention(
                embed_dim=self.all_head_size,
                vdim=self.all_head_size*2,
                num_heads=self.num_attention_heads,
                has_outproj=False,
                causal=True,
            )
        if not self.pconfig.causal:
            performer_value_hidden_size = self.attention_head_size*3
        else:
            performer_value_hidden_size = self.attention_head_size*3
        self.register_buffer('attention_predictor_enc_head_embd', None)
        self.attention_predictor_enc_head_embd = torch.eye(self.num_attention_heads)
        self.attention_predictor_enc_per_layer = nn.Sequential(
            nn.Linear(performer_value_hidden_size * self.num_attention_heads, self.attention_head_size * 2 * self.num_attention_heads),
            nn.LayerNorm(self.attention_head_size * 2 * self.num_attention_heads),
            nn.GELU(),
        )
        self.attention_predictor_enc = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(performer_value_hidden_size, self.attention_head_size*2),
            # nn.Linear(performer_value_hidden_size + self.num_attention_heads, self.attention_head_size*2),
            nn.LayerNorm(self.attention_head_size*2),
            nn.GELU(),
        )
        COMPILE_CNN = os.environ.get('PERLIN_COMPILE', '0') == '1'
        N_H = self.num_attention_heads
        if not self.pconfig.causal:
            self.attention_predictor_dec_row_down_scale = 2
            self.attention_predictor_dec_row_splits = 4
            self.attention_predictor_dec_row_out_ch = (self.pconfig.attention_predictor_length // self.attention_predictor_dec_row_down_scale) * self.attention_predictor_dec_row_splits
            self.attention_predictor_dec_row = nn.Sequential(
                nn.Linear(self.attention_head_size*2, self.attention_predictor_dec_row_out_ch),
                ChannelSplit(self.attention_predictor_dec_row_splits),
            )
            self.attention_predictor_cnn = nn.Sequential(
                KeepRes(
                    nn.Conv2d(self.attention_predictor_dec_row_splits*N_H, 4*N_H, 3, padding=1, stride=(2, 1)),
                    # nn.Conv2d(4*N_H, 4*N_H, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(4*N_H, 4*N_H, 3, padding=1),
                    nn.ReLU(),
                    UpsampleFP32((2, 1), torch.float16),
                    nn.Conv2d(4*N_H, N_H, 3, padding=1),
                    output_width=self.pconfig.attention_predictor_length
                ),
            )
            # self.attention_predictor_dec_row = nn.Sequential(
            #     nn.Linear(self.attention_head_size*2, self.pconfig.attention_predictor_length),
            #     # ChannelSplit(self.attention_predictor_dec_row_splits),
            # )
            # self.attention_predictor_cnn = nn.Sequential(
            #     KeepRes(
            #         # NOTE if we use pixelshuffle outch should be 48
            #         CausalConv2d(12, 48, 3, padding=1, stride=2, causal=self.pconfig.causal),
            #         nn.ReLU(),
            #         ResBlock(48, causal=self.pconfig.causal),
            #         ResBlock(48, causal=self.pconfig.causal),
            #         UpsampleFP32(2, torch.float16),
            #         CausalConv2d(48, 12, 3, padding=1, causal=self.pconfig.causal),
            #     )
            # )
        else:
            is_causal = self.pconfig.causal
            inner_ch = int(os.environ.get("PERLIN_HOTFIX_OPT_INNER_CH", "2"))
            if inner_ch != 2:
                self._warning_messages += f'WARN, you are using hotfix backend. PERLIN_HOTFIX_OPT_INNER_CH {inner_ch}\n'
            self.attention_predictor_dec_row_down_scale = 4
            self.attention_predictor_dec_row_splits = inner_ch
            self.attention_predictor_dec_row_out_ch = (self.pconfig.attention_predictor_length // self.attention_predictor_dec_row_down_scale) * self.attention_predictor_dec_row_splits
            self.attention_predictor_dec_row = nn.Sequential(
                nn.Linear(self.attention_head_size*2, self.attention_predictor_dec_row_out_ch),
                ChannelSplit(self.attention_predictor_dec_row_splits),
            )
            deeper_layer = int(os.environ.get("PERLIN_HOTFIX_OPT_DEEPER", "0")) == 1
            if deeper_layer:
                self._warning_messages += 'WARN, you are using hotfix backend. PERLIN_HOTFIX_OPT_DEEPER\n'
                self.attention_predictor_cnn = nn.Sequential(
                    ModuleBenchmark('cnn.lnorm1', nn.LayerNorm(self.pconfig.attention_predictor_length // self.attention_predictor_dec_row_down_scale)),
                    ModuleBenchmark('cnn.keepres', KeepRes(
                        ModuleBenchmark('cnn.keepres.conv1', CausalConv2d(inner_ch*N_H, inner_ch*N_H, 3, padding=2, dilation=2, stride=(1, 1), causal=is_causal)),
                        nn.ReLU(),
                        ModuleBenchmark('cnn.keepres.conv2', CausalConv2d(inner_ch*N_H, inner_ch*N_H, 3, padding=2, dilation=2, stride=(1, 1), causal=is_causal)),
                        nn.ReLU(),
                        ModuleBenchmark('cnn.keepres.conv3', CausalConv2d(inner_ch*N_H, inner_ch*N_H, 3, padding=2, dilation=2, stride=(1, 1), causal=is_causal)),
                        nn.ReLU(),
                        ModuleBenchmark('cnn.keepres.upsam', UpsampleFP32((1, 4), torch.float16)),
                        ModuleBenchmark('cnn.keepres.conv4', CausalConv2d(inner_ch*N_H, N_H, 1, padding=1, causal=is_causal)),
                        output_width=self.pconfig.attention_predictor_length
                    )),
                    # this prevent model explode within causal setting...
                    ModuleBenchmark('cnn.lnorm2', nn.LayerNorm(self.pconfig.attention_predictor_length))
                )
            else:
                self.attention_predictor_cnn = nn.Sequential(
                    ModuleBenchmark('cnn.lnorm1', nn.LayerNorm(self.pconfig.attention_predictor_length // self.attention_predictor_dec_row_down_scale), disabled=COMPILE_CNN),
                    ModuleBenchmark('cnn.keepres', KeepRes(
                        # ModuleBenchmark('cnn.keepres.conv1', CausalConv2d(N_H, inner_ch*N_H, 5, padding=2, stride=(1, 4), causal=is_causal)),
                        # nn.ReLU(),
                        ModuleBenchmark('cnn.keepres.conv1', CausalConv2d(inner_ch*N_H, inner_ch*N_H, 3, padding=2, dilation=2, stride=(1, 1), causal=is_causal), disabled=COMPILE_CNN),
                        nn.ReLU(),
                        ModuleBenchmark('cnn.keepres.conv2', CausalConv2d(inner_ch*N_H, inner_ch*N_H, 3, padding=2, dilation=2, stride=(1, 1), causal=is_causal), disabled=COMPILE_CNN),
                        nn.ReLU(),
                        ModuleBenchmark('cnn.keepres.upsam', UpsampleFP32((1, 4), torch.float16), disabled=COMPILE_CNN),
                        ModuleBenchmark('cnn.keepres.conv4', CausalConv2d(inner_ch*N_H, N_H, 1, padding=1, causal=is_causal), disabled=COMPILE_CNN),
                        output_width=self.pconfig.attention_predictor_length
                    ), disabled=COMPILE_CNN),
                    # this prevent model explode within causal setting...
                    ModuleBenchmark('cnn.lnorm2', nn.LayerNorm(self.pconfig.attention_predictor_length), disabled=COMPILE_CNN)
                )
            # self.attention_predictor_cnn = torch.compile(self.attention_predictor_cnn, mode='reduce-overhead')
        # self.attention_predictor_cnn = nn.Identity()
        
        if COMPILE_CNN:
            self.attention_predictor_cnn = torch.compile(self.attention_predictor_cnn)
            pass
        
        self.attention_predictor_dec_scaler = nn.Sequential(
            nn.Linear(self.attention_head_size*2, 2),
        )
        
        #-- compressed predictor
        if self.pconfig.attention_predictor_method == 'comp':
            self.attention_predictor_comp_length = \
                self.pconfig.attention_predictor_comp_patch_count * self.pconfig.attention_predictor_comp_patch_size
            self.attention_predictor_comp_codebook = nn.Parameter(
                torch.randn((self.pconfig.attention_predictor_comp_book_size, self.pconfig.attention_predictor_comp_patch_size))
            )
            self.attention_predictor_comp_enc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(performer_value_hidden_size, self.attention_head_size*2),
                nn.LayerNorm(self.attention_head_size*2),
                nn.GELU(),
            )
            self.attention_predictor_comp_dec_row = nn.Sequential(
                nn.Linear(
                    self.attention_head_size*2,
                    self.pconfig.attention_predictor_comp_book_size * self.pconfig.attention_predictor_comp_patch_count
                ),
            )
        #-- TODO VQVAE
        
        #- output
        self.norm_performer = nn.LayerNorm(config.hidden_size)
        self.norm_partial = nn.LayerNorm(config.hidden_size)
        self.norm_random = nn.LayerNorm(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        self.register_buffer('_v_eye', None, persistent=False)
        
        self.v_eye_learned = nn.Parameter(
            data=torch.rand((1, 1, self.attention_head_size, self.attention_head_size)),
            requires_grad=True
        )
        
        # raise Exception(config.max_position_embeddings)
        self.v_eye_learned_causal = nn.Parameter(
            data=torch.randn((1, 1, config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 2048, self.attention_head_size)),
            requires_grad=True
        )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_for_atten: torch.Tensor,
        k_for_atten: torch.Tensor,
        v_for_atten: torch.Tensor,
        q_for_score: torch.Tensor,
        k_for_score: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_scores_truth: torch.Tensor,
        context_layer_truth: torch.Tensor,
        last_state: PerlinAttentionState = None,
    ):
        dynamic_k = int(os.environ.get('DYNAMIC_K', '0'))
        if dynamic_k > 0:
            warnings.warn(f'dynamic k {dynamic_k}')
            self.pconfig.k = dynamic_k
        
        if len(self._warning_messages) > 0:
            print(self._warning_messages)
            self._warning_messages = ''
        
        if context_layer_truth is not None:
            if callable(context_layer_truth):
                with torch.no_grad():
                    context_layer_truth = context_layer_truth()
            if context_layer_truth.device != q.device:
               context_layer_truth = context_layer_truth.to(q.device, non_blocking=True)
            
            if callable(attention_scores_truth):
                with torch.no_grad():
                    attention_scores_truth = attention_scores_truth()
            if attention_scores_truth.device != q.device:
                attention_scores_truth = attention_scores_truth.to(q.device, non_blocking=True)
        
        if os.environ.get('KD_SELF_TEACHER', '0') == '1' and self.training:
            context_layer_truth = None
            N, H, TSRC, D = k.shape
            N, H, TDST, D = q.shape
            score = torch.bmm(q.detach().view(N*H, TDST, D), k.detach().view(N*H, TSRC, D).transpose(-1, -2)).view(N, H, TDST, TSRC)
            score = score + attention_mask
            # score = torch.softmax(score, dim=-1)
            attention_scores_truth = score
        
        DUMMY_OUTPUT = PerlinAttentionOutput(
            loss=None,
            context_layer=None,
            partial_attention_probs=None,
            partial_attention_mask=None,
            estimated_attention_probs=None,
            estimated_attention_probs_m=None,
            dense_attention_probs=None,
            key_for_score=None,
            state=None,
        )
        
        use_cache = self.pconfig.use_cache
        
        if q.dtype in [torch.float16, torch.bfloat16]:
            # NOTE even if we are in bfloat16, we have to use fp16 minimum because of F.interpolate
            FP_MIN = torch.finfo(torch.float16).min / 2
        elif q.dtype in [torch.float32]:
            FP_MIN = torch.finfo(torch.float32).min / 2
        else:
            raise Exception('unknown type')
        
        _, _, _, T_SRC = attention_mask.shape
        T_DST = T_SRC
        if self.pconfig.causal:
            if not use_cache:
                N, H, T_DST, T_SRC = attention_mask.shape
                assert T_DST == T_SRC
                assert H == 1
                causal_attention_mask = attention_mask
                attention_mask = attention_mask[:, :, :, :1].transpose(-1, -2)
            else:
                N, H, T_DST, T_SRC = attention_mask.shape
                _N, _H, _T_DST, _HID_Q = q.shape
                _N, _H, _T_SRC, _HID_K = k.shape
                assert k.shape[:-2] == v.shape[:-2]
                assert T_DST == _T_DST
                assert T_SRC == _T_SRC
                assert _HID_Q == _HID_K
                assert _N == N
                
                # attention_mask = attention_mask[0:1]
                
                causal_attention_mask = attention_mask
                attention_mask = causal_attention_mask[:, :, -1:, :]
                
                assert attention_mask.shape == (N, 1, 1, T_SRC)
                assert causal_attention_mask.shape == (N, 1, T_DST, T_SRC)
        
        # return DUMMY_OUTPUT #0
        
        dst_attention_mask = attention_mask.transpose(-1, -2)
        if self.pconfig.causal:
            dst_attention_mask = causal_attention_mask[:,:,:,:1]
        
        not_padded = (attention_mask > -1).float().sum() == attention_mask.numel()
        
        if use_cache and last_state is None:
            last_state = PerlinAttentionState(self)
        if not use_cache:
            last_state = None
        
        # print('state', use_cache, strify(last_state), strify(q), strify(k), strify(v))
        
        raise_if_nan(q)
        raise_if_nan(k)
        raise_if_nan(v)
        
        zero_one_attention_mask = (attention_mask > -1).float()
        zero_one_attention_mask_cumsum = zero_one_attention_mask.cumsum(-1)
        zero_one_attention_mask_sum = zero_one_attention_mask.sum(-1)
        
        get_bench().register_temp_buffer('q', q)
        get_bench().register_temp_buffer('k', k)
        get_bench().register_temp_buffer('v', v)
        get_bench().register_temp_buffer('attention_mask', attention_mask)
        
        with timer("perlin"):
            # return DUMMY_OUTPUT #0
            
            N, H, T, HID = q.shape
            with timer("vmask"):
                # if not causal, we just use Eye matrix for V_identity
                if not self.pconfig.causal:
                    with timer("vmaks.eye"):
                        # E_N = min(T, HID)
                        E_N = HID
                        
                        if self._v_eye is None or self._v_eye.shape[-1] != E_N or self._v_eye.dtype != v.dtype:
                            v_for_atten_identity = torch.eye(
                                n=E_N,
                                dtype=v.dtype,
                                device=v.device,
                            )
                            
                            v_for_atten_identity = v_for_atten_identity.view(1, 1, E_N, E_N)
                            self._v_eye = v_for_atten_identity
                        else:
                            v_for_atten_identity = self._v_eye
                        
                        v_for_atten_identity = v_for_atten_identity.expand(v_for_atten.shape[:2] + (E_N, E_N))
                    
                    with timer("vmask.grid"):
                        token_index_y = ((zero_one_attention_mask_cumsum - 1.0) / ((zero_one_attention_mask_sum - 1.0).view(N, 1, 1, 1) + 1e-8) * 2 - 1)\
                            .view(N, T, 1, 1)\
                            .expand(N, T, HID, 1)
                        token_index_x = (torch.arange(HID, device=q.device, dtype=torch.long) / (HID - 1) * 2 - 1).view(1, 1, HID, 1)
                        token_index_x = token_index_x.expand(N, T, HID, 1)
                        token_index = torch.cat([token_index_x, token_index_y], dim=-1)
                    
                    with timer("vmask.sample"):
                        v_for_atten_identity = grid_sample_bf16(
                            input=v_for_atten_identity, 
                            grid=token_index.to(v_for_atten_identity.dtype), 
                            mode='bilinear',
                            align_corners=True,
                        )
                
                with timer("vmask.cat_fill"):
                    if not self.pconfig.causal:
                        v_for_atten = torch.cat([
                            v_for_atten_identity, 
                            v_for_atten
                        ], dim=-1)
                    else:
                        v_for_atten_pos_emb = self.v_eye_learned_causal[:,:,:T_SRC,:]
                        v_for_atten = torch.cat([
                            v_for_atten_pos_emb.expand(v_for_atten.shape),
                            v_for_atten
                        ], dim=-1)
                    
                    get_bench().register_temp_buffer('v_for_atten', v_for_atten)

                    if not not_padded:
                        v_for_atten.masked_fill_(dst_attention_mask < -1, 0)
                        v.masked_fill_(dst_attention_mask < -1, 0)
            
            # return DUMMY_OUTPUT #12
            
            with timer("performer"):
                if not self.benchmarking:
                    q_type = q_for_atten.dtype
                    if torch.get_autocast_gpu_dtype() in [torch.float16, torch.bfloat16]:
                        PRECISION_PERF = torch.float32 if torch.cuda.is_bf16_supported() else torch.float32
                    else:
                        PRECISION_PERF = torch.float32
                    with torch.autocast('cuda', PRECISION_PERF):
                        if self.pconfig.attention_predictor_backend == 'performer':
                            last_state, performer_context_layer = PerlinAttentionState.stateful_performer(
                                last_state,
                                "performer->performer_context_layer",
                                self.performer,
                                batch_to(q_for_atten, PRECISION_PERF), 
                                batch_to(k_for_atten, PRECISION_PERF), 
                                batch_to(v_for_atten, PRECISION_PERF),
                            )
                        elif self.pconfig.attention_predictor_backend == 'cosformer':
                            _q = batch_to(q_for_atten, PRECISION_PERF)
                            _k = batch_to(k_for_atten, PRECISION_PERF)
                            _v = batch_to(v_for_atten, PRECISION_PERF)
                            _N, _H, _T, _HID = _q.shape
                            _q = _q.permute(0, 2, 1, 3).reshape(_N, _T, _H*_HID).permute(1, 0, 2)
                            _N, _H, _T, _HID = _k.shape
                            _k = _k.permute(0, 2, 1, 3).reshape(_N, _T, _H*_HID).permute(1, 0, 2)
                            _N, _H, _T, _HID = _v.shape
                            _v = _v.permute(0, 2, 1, 3).reshape(_N, _T, _H*_HID).permute(1, 0, 2)
                            t = self.cosformer(_q, _k, _v)
                            print(t.shape)
                            performer_context_layer = t.reshape(_T, _N, _H, _HID).permute(1, 2, 0, 3)
                            # print('hello')
                        else:
                            raise Exception(self.pconfig.attention_predictor_backend)
                        
                    if q_type != performer_context_layer.dtype:
                        performer_context_layer = performer_context_layer.to(q_type)
                    assert performer_context_layer.shape[-2] == q.shape[-2], f"{performer_context_layer.shape} == {q.shape}, {v_for_atten.shape}"
                    # print('pcl', strify(performer_context_layer), strify(q), strify(k), strify(v))
                else:
                    # TODO: fix numerical stability...
                    if os.environ.get("PERLIN_HOTFIX_STATEFUL", "1") == "1":
                        last_state, performer_context_layer = PerlinAttentionState.stateful_performer(
                            last_state,
                            "performer->performer_context_layer",
                            self.performer,
                            q_for_atten, 
                            k_for_atten, 
                            v_for_atten,
                        )
                    else:
                        performer_context_layer = self.performer(
                            q_for_atten, 
                            k_for_atten, 
                            v_for_atten
                        )
                get_bench().register_temp_buffer('performer_context_layer', performer_context_layer)
            
            # return DUMMY_OUTPUT #119
            
            with timer("performer_value"):
                # NOTE May cut gradient from loss_sp, because loss_sp has sometimes negative effect to loss_model when approximation is sucks.
                if performer_context_layer.shape[-2] < v.shape[-2]:
                    performer_value = torch.cat([
                        performer_context_layer, 
                        v[...,-performer_context_layer.shape[-2]:,:]
                    ], dim=-1)#.detach()
                else:
                    performer_value = torch.cat([
                        performer_context_layer, 
                        v
                    ], dim=-1)#.detach()
                raise_if_nan(performer_value)
                get_bench().register_temp_buffer('performer_value', performer_value)
            
            # return DUMMY_OUTPUT #120
            
            # estimate attention scores
            with timer("predictor"):
                if self.pconfig.attention_predictor_method == 'mlp':
                    # I came up this dark magic from my head during rebuttal...
                    query_skips = int(os.environ.get('QUERY_SKIPS', '1'))
                    with timer("predictor.enc"):
                        raise_if_nan(performer_value)
                        # ENC_PER_LAYER = False
                        if self.pconfig.attention_predictor_enc_per_layer:
                            # print('hello')
                            _N, _H, _T, _D = performer_value.shape
                            t_enc_x = performer_value.permute(0, 2, 1, 3).reshape(_N, _T, _H*_D)
                            t_attention_predictor = self.attention_predictor_enc_per_layer(t_enc_x)
                            _HD = t_attention_predictor.shape[-1]
                            t_attention_predictor = t_attention_predictor.view(N, _T, _H, _HD // _H).permute(0, 2, 1, 3)
                            assert t_attention_predictor.shape[:3] == (_N, _H, _T)
                        else:
                            t_enc_x = performer_value
                            # _N, _H, _T, _D = performer_value.shape
                            # t_enc_x = torch.cat([
                            #     performer_value, 
                            #     self.attention_predictor_enc_head_embd.view(1, _H, 1, _H).expand(_N, _H, _T, _H)
                            # ], dim=-1)
                            if query_skips > 1:
                                assert (t_enc_x.shape[-2] % query_skips) == 0
                                t_enc_x = t_enc_x[:, :, ::query_skips, :]
                            t_attention_predictor = self.attention_predictor_enc(t_enc_x)
                    with timer("predictor.dec_row"):
                        raise_if_nan(t_attention_predictor)
                        estimated_attention_score = self.attention_predictor_dec_row(t_attention_predictor) # type: torch.Tensor
                        get_bench().register_temp_buffer('estimated_attention_score_dec_row', estimated_attention_score)
                        raise_if_nan(estimated_attention_score)
                    
                    with timer("predictor.cnn"):
                        # torch.cuda.synchronize()
                        # t = time.time()
                        last_state, estimated_attention_score = PerlinAttentionState.stateful_causal_cnn_op(
                            last_state,
                            "attention_predictor_cnn->estimated_attention_score",
                            self.attention_predictor_cnn,
                            estimated_attention_score,
                            q.shape[-2],
                        )
                        # torch.cuda.synchronize()
                        # print(time.time() - t)
                        
                        if query_skips > 1:
                            _N, _H, _T, _D = estimated_attention_score.shape
                            estimated_attention_score = estimated_attention_score.view(_N, _H, _T, 1, _D).expand(_N, _H, _T, query_skips, _D).reshape(_N, _H, _T*query_skips, _D)
                            _N, _H, _T, _D = t_attention_predictor.shape
                            t_attention_predictor = t_attention_predictor.view(_N, _H, _T, 1, _D).expand(_N, _H, _T, query_skips, _D).reshape(_N, _H, _T*query_skips, _D)
                        
                        # print('cnnd', strify(last_state), q.shape, strify(estimated_attention_score))
                        assert estimated_attention_score.shape[-2] == T_DST
                        # print('estimated_attention_score', strify(estimated_attention_score))
                elif self.pconfig.attention_predictor_method == 'comp':
                    assert not use_cache
                    warnings.warn('attention prediction method is compressed one.')
                    t_attention_predictor = self.attention_predictor_comp_enc(performer_value)
                    estimated_attention_score = self.attention_predictor_comp_dec_row(t_attention_predictor)
                    estimated_attention_score = estimated_attention_score\
                        .view(N, H, T, self.pconfig.attention_predictor_comp_patch_count, self.pconfig.attention_predictor_comp_book_size)
                    _, _, _, CODE_SEQ_LEN, BOOK_LEN = estimated_attention_score.shape
                    estimated_attention_score = softmax_bf16(estimated_attention_score, dim = -1, training=self.training)
                    estimated_attention_score = torch.matmul(
                        estimated_attention_score.view(-1, BOOK_LEN), 
                        self.attention_predictor_comp_codebook
                    )
                    estimated_attention_score = estimated_attention_score.view(N, H, T, -1)
                else:
                    raise Exception()
                get_bench().register_temp_buffer('t_attention_predictor', t_attention_predictor)
            
            # return DUMMY_OUTPUT #413
        
            # interpolate and convert to probability
            with timer("mask_softmax"):
                T_M = estimated_attention_score.shape[-1]
                estimated_attention_probs = softmax_bf16(estimated_attention_score, -1, training=self.training)
                assert estimated_attention_probs.shape[-2] == T_DST, f"{estimated_attention_probs.shape}, {T_DST}"
            
            # return DUMMY_OUTPUT #413
            
            get_bench().register_temp_buffer('estimated_attention_score', estimated_attention_score)
            get_bench().register_temp_buffer('estimated_attention_probs', estimated_attention_probs)
            
            # in layerwise, train perlin attention predictor
            # TODO set up resize_m_to_t
            def resize_from_m_to_t(x, masked_fill_value, target_width=None, output_dtype=None, handle_oversample=True):
                from .ops import resize_from_m_to_t
                N, H, T1, T_M = x.shape
                if target_width is not None:
                    T2 = target_width
                else:
                    T2 = T1
                
                mask = attention_mask
                if self.pconfig.causal:
                    mask = causal_attention_mask
                
                return resize_from_m_to_t(
                    x=x,
                    masked_fill_value=masked_fill_value,
                    attention_mask=mask,
                    target_width=target_width,
                    training=self.training and self.pconfig.causal,
                    is_causal=self.pconfig.causal,
                    k=self.pconfig.k,
                    oversampled=self.pconfig.k_oversample if handle_oversample else 1.0,
                )
            
            loss = 0
            estimated_attention_probs_resized = estimated_attention_score_resized = None
            if not self.benchmarking and not use_cache and attention_scores_truth is not None:
                N, H, T, T_M = estimated_attention_score.shape
                # for loss calculation
                # with torch.no_grad():
                estimated_attention_probs_resized = resize_from_m_to_t(
                    estimated_attention_probs, 
                    masked_fill_value=0, 
                    handle_oversample=False
                )
                # return DUMMY_OUTPUT #413
                estimated_attention_score_resized = resize_from_m_to_t(
                    estimated_attention_score, 
                    masked_fill_value=FP_MIN, 
                    output_dtype=torch.float32,
                    handle_oversample=False,
                )
                # return DUMMY_OUTPUT #601
                
                with torch.autocast('cuda', torch.float32):
                    raise_if_nan(estimated_attention_score_resized)
                    raise_if_nan(attention_scores_truth)
                    
                    if not self.pconfig.causal:
                        loss_kl_t = kl_div_attention(
                            F.log_softmax(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                            F.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                            attention_mask,
                        ) * 0.1
                        loss_mse_t = F.mse_loss(
                            softmax_bf16(estimated_attention_score_resized.masked_fill(attention_mask < -1, FP_MIN), dim=-1, training=self.training), 
                            softmax_bf16(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1, training=self.training)
                        )
                    else:
                        # return DUMMY_OUTPUT #838
                        with torch.autocast('cuda', torch.float32):
                            _t_causal_mask = causal_attention_mask < -1
                            # return DUMMY_OUTPUT #601
                            _input = F.log_softmax(estimated_attention_score_resized.masked_fill_(_t_causal_mask, FP_MIN), dim=-1, dtype=torch.float32).view(-1, estimated_attention_score_resized.shape[-1])
                            # return DUMMY_OUTPUT #751
                            _target = F.softmax(attention_scores_truth.masked_fill_(_t_causal_mask, FP_MIN), dim=-1, dtype=torch.float32).view(-1, estimated_attention_score_resized.shape[-1])
                            # return DUMMY_OUTPUT #942
                            loss_kl_t = F.kl_div(
                                _input,
                                _target,
                                reduction='batchmean',
                            ) * 0.1
                            del _input
                            # return DUMMY_OUTPUT #1518
                            loss_mse_t = F.mse_loss(
                                F.softmax(estimated_attention_score_resized, dim=-1, dtype=torch.float32).view(-1, estimated_attention_score_resized.shape[-1]),
                                _target,
                            )
                            del _target
                    
                    raise_if_nan(loss_kl_t)
                    raise_if_nan(loss_mse_t)
                    loss += loss_kl_t + loss_mse_t
                    raise_if_nan(loss)
                    
                get_bench().register_temp_buffer('attention_probs_truth', None, lazy=lambda: F.softmax(attention_scores_truth, dim=-1) * (dst_attention_mask > -1))
                # if not self.pconfig.causal:
                #     get_bench().register_temp_buffer('attention_probs_truth_m', None, lazy=lambda: F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (dst_attention_mask > -1))
                get_bench().register_temp_buffer('estimated_attention_probs_resized', estimated_attention_probs_resized)
                get_bench().register_temp_buffer('estimated_attention_score_resized', estimated_attention_score_resized)
            
            # return DUMMY_OUTPUT #1518
            
            with timer("mask"):
                # TODO: perform this with states
                
                if not not_padded:
                    estimated_attention_probs = estimated_attention_probs * (dst_attention_mask > -1)
                
                N, H, T, T_M = estimated_attention_probs.shape
                assert T == T_DST, f"{T}=={T_DST}, {estimated_attention_probs.shape} {not_padded}"
                token_length = (attention_mask > -1).long().sum(-1).view(N, -1)
                top_k = min(max(int(round(self.pconfig.k * (T_M / torch.min(token_length).item()))), 1), T_M)
                k_flatten = self.pconfig.k_flatten
                k_flatten_dim = self.pconfig.k_flatten_dim
                if not k_flatten:
                    k_flatten = True
                    k_flatten_dim = 'query'
                
                if not k_flatten:
                    raise Exception()
                    with timer("mask.topk"):
                        _, indices = torch.topk(
                            estimated_attention_probs, # estimation gradient is cut here
                            k=top_k, 
                            dim=-1, 
                            sorted=True,
                        )
                    with timer("mask.empty"):
                        partial_attention_mask = torch.empty(
                            (N, H, T, T_M),
                            dtype=q_for_score.dtype,
                            device=q_for_score.device,
                        )
                    with timer("mask.fill"):
                        partial_attention_mask.fill_(FP_MIN)
                    with timer("mask.scatter"):
                        partial_attention_mask.scatter_(dim=-1, index=indices, value=0)
                    with timer("mask.mask_per_item"):
                        per_item_top_k = token_length * H * (self.pconfig.k * T_M / token_length)
                else:
                    top_k_elems = None
                    per_item_top_k = None 
                    assert k_flatten_dim in ['head', 'batch', 'causal_batch', 'query']
                    with timer("mask.view"):
                        masked_estimated_attention_probs = (estimated_attention_probs * (dst_attention_mask > -1))
                        
                        get_bench().register_temp_buffer('masked_estimated_attention_probs', masked_estimated_attention_probs)
                        
                        # return DUMMY_OUTPUT
                    
                        if not self.pconfig.causal:
                            token_length = (attention_mask > -1).long().sum(-1).view(N, -1)
                        else:
                            # _causal_token_length = (causal_attention_mask > -1).long().sum(-1).view(N, 1, T_DST, 1)
                            causal_token_length = torch.arange(1, T_DST+1, dtype=torch.long, device=attention_mask.device).view(1, 1, T_DST, 1).expand(N, 1, T_DST, 1)
                            # print(_causal_token_length)
                            # print(causal_token_length)
                            # assert ((_causal_token_length - causal_token_length).abs().sum().item() < 1e-4), (_causal_token_length - causal_token_length).abs().sum().item()
                        
                        # return DUMMY_OUTPUT
                        
                        if k_flatten_dim == 'batch':
                            assert not self.pconfig.causal
                            t = masked_estimated_attention_probs.view(N, H*T*T_M)
                            # top_k_elems = top_k*T*H
                            per_item_top_k = token_length * H * (self.pconfig.k * self.pconfig.k_oversample * T_M / token_length)
                        elif k_flatten_dim == 'head':
                            assert not self.pconfig.causal
                            t = masked_estimated_attention_probs.view(N, H, T*T_M)
                            # top_k_elems = top_k*T
                            per_item_top_k = (token_length * (self.pconfig.k * self.pconfig.k_oversample * T_M / token_length)).view(N, 1, 1)
                        elif k_flatten_dim == 'causal_batch':
                            t = masked_estimated_attention_probs.transpose(1, 2).reshape(N, T, H*T_M)
                            if not self.pconfig.causal:
                                per_item_top_k = (H * (self.pconfig.k * self.pconfig.k_oversample * T_M / token_length)).view(N, 1, 1)
                            else:
                                # NOTE consider causal token length
                                per_item_top_k = (H * (self.pconfig.k * self.pconfig.k_oversample * T_M / causal_token_length.squeeze(0))).view(N, T_DST, 1) #, 1, H*T_M)
                        elif k_flatten_dim == 'query':
                            assert not self.pconfig.causal
                            t = masked_estimated_attention_probs.view(N, H, T, T_M)
                            per_item_top_k = (self.pconfig.k * self.pconfig.k_oversample * T_M / token_length).view(N, 1, 1, 1)
                        else: raise Exception()
                        
                        per_item_top_k = torch.round(per_item_top_k)
                        
                        # per_item_top_k_rounded = torch.round(per_item_top_k)
                        # per_item_top_k_floored = torch.floor(per_item_top_k)
                        # per_item_top_k_ceil_prob = per_item_top_k - per_item_top_k_floored
                        # per_item_top_k_prob_rounded = per_item_top_k_floored + (torch.rand_like(per_item_top_k_ceil_prob) < per_item_top_k_ceil_prob) * 1
                        # per_item_top_k = per_item_top_k_prob_rounded
                        # per_item_top_k = per_item_top_k_rounded * (per_item_top_k_rounded >= 1) + per_item_top_k_lower * (per_item_top_k_rounded < 1)
                        
                        # NOTE to prevent 0 top-k when large T and small T_m, we take care of lower bound in kernel implemenation.
                        per_item_top_k = torch.clamp_min(per_item_top_k, 1)
                        
                        top_k_elems = min(int(math.ceil(torch.max(per_item_top_k).item())), t.shape[-1])
                        get_bench().register_temp_buffer('per_item_top_k', per_item_top_k)
                        get_bench().register_temp_buffer('top_k_elems', None, lazy=lambda: torch.tensor(top_k_elems, dtype=torch.float64))
                    with timer("mask.topk"):
                        if top_k_elems < t.shape[-1] * 0.9:
                            _, indices = torch.topk(
                                input=t,
                                k=top_k_elems, 
                                dim=-1, 
                                sorted=True #sorted true is important
                            )
                        else:
                            _, indices = torch.sort(
                                t,
                                dim=-1,
                                descending=True,
                                stable=False,
                            )
                            indices = indices[...,:top_k_elems]
                        get_bench().register_temp_buffer('topk_indices', indices.double())
                    with timer("mask.empty"):
                        partial_attention_mask = torch.empty(
                            t.shape, 
                            dtype=torch.long, 
                            device=attention_mask.device,
                        )
                    with timer("mask.fill"):
                        partial_attention_mask.fill_(t.shape[-1])
                    with timer("mask.scatter"):
                        partial_attention_mask.scatter_(
                            dim=-1,
                            index=indices,
                            src=torch.arange(
                                top_k_elems, 
                                dtype=torch.long,
                                device=attention_mask.device, 
                            )\
                                .view((1, -1) if t.ndim == 2 else (1, 1, -1))\
                                .expand(indices.shape)
                        )
                    with timer("mask.masked_fill"):
                        if not self.benchmarking:
                            t_dead_mask = partial_attention_mask >= per_item_top_k
                            # partial_attention_mask.fill_(FP_MIN)
                            # partial_attention_mask.masked_fill_(t_alive_mask, value=0)
                            get_bench().register_temp_buffer('t_dead_mask', None, lambda: t_dead_mask.float())
                            partial_attention_mask = t_dead_mask.to(q.dtype) * FP_MIN
                        else:
                            t_alive_mask = partial_attention_mask < per_item_top_k
                            partial_attention_mask = t_alive_mask.float()
                    
                    if k_flatten_dim == 'causal_batch':
                        # need to mask time dimension
                        partial_attention_mask = partial_attention_mask.view(N, T, H, T_M).transpose(1, 2)
                        if not self.benchmarking:
                            partial_attention_mask.masked_fill_(
                                mask=dst_attention_mask < -1,
                                value=FP_MIN
                            )
                        else:
                            partial_attention_mask.masked_fill_(
                                mask=dst_attention_mask < -1,
                                value=0
                            )
                    elif k_flatten_dim == 'query':
                        partial_attention_mask = partial_attention_mask.view(N, H, T, T_M)
                        if not self.benchmarking:
                            partial_attention_mask.masked_fill_(
                                mask=dst_attention_mask < -1,
                                value=FP_MIN
                            )
                        else:
                            partial_attention_mask.masked_fill_(
                                mask=dst_attention_mask < -1,
                                value=0
                            )
                    elif k_flatten_dim in ['batch', 'head']:
                        pass
                    else: raise Exception()
                    partial_attention_mask = partial_attention_mask.view(N, H, T, T_M)
            
            # return DUMMY_OUTPUT #1518
            
            get_bench().register_temp_buffer('partial_attention_mask_before_interp', partial_attention_mask)
            
            with timer("interp"):
                # NOTE: partial attention mask should be filled with 0 and -inf only.
                raise_if_nan(partial_attention_mask)
                if not self.benchmarking:
                    with timer("interp.resize"):
                        # TODO Fix this function to return COO tensor
                        # print('resize', strify(partial_attention_mask))
                        partial_attention_mask = resize_from_m_to_t(partial_attention_mask, FP_MIN, target_width=T_SRC, handle_oversample=True)
                        if self.pconfig.causal:
                            partial_attention_mask.masked_fill_(causal_attention_mask < -1, FP_MIN)
                else:
                    if not self.pconfig.causal:
                        def resize_width(img: torch.Tensor, scale: float):
                            N, H, W = img.shape
                            # img = img.coalesce()
                            idx = img.indices() #.float()
                            nnz = idx.shape[-1]
                            if scale < 1.0:
                                xs_scaled = idx[2] * scale
                                xs_rounded = torch.clamp(torch.round(xs_scaled), 0, round(W*scale)-1)
                                xs_okay = torch.abs(xs_rounded - xs_scaled) > (scale * 0.5)
                                idx[2] = xs_rounded# * xs_okay
                                # idx = idx * xs_okay.unsqueeze(0) + (~xs_okay.unsqueeze(0)) * -1
                                idx.masked_fill_(xs_okay, value=-1)
                                idx = torch.unique(idx, dim=-1) #TODO FIX this to masked select
                                idx = idx[:, 1:]
                                
                                # print(nnz, idx.shape[-1])
                                return torch.sparse_coo_tensor(
                                    indices=idx.contiguous(),
                                    values=torch.ones((idx.shape[-1],), device=img.device, dtype=img.dtype),
                                    size=(N, H, round(W*scale)),
                                )
                            elif scale > 1.0:
                                scale_ceil = math.ceil(scale)
                                # idx = F.interpolate(idx.view(1, 1, 3, nnz), size=(3, nnz*scale_ceil), mode='nearest').view(3, nnz*scale_ceil) # type: torch.Tensor
                                idx = idx.view(3, nnz, 1).expand(3, nnz, scale_ceil).reshape(3, nnz*scale_ceil)
                                # idx[2] = idx[2] * scale_ceil + torch.arange(nnz*scale_ceil, device=img.device) % scale_ceil
                                
                                shrink_scale = scale / scale_ceil
                                xs_scaled = (idx[2] * scale_ceil + torch.arange(scale_ceil, device=img.device).unsqueeze(0).expand(nnz, scale_ceil).reshape(-1)) * shrink_scale
                                xs_rounded = torch.round(xs_scaled)
                                xs_okay = torch.abs(xs_rounded - xs_scaled) < (shrink_scale * 0.5)
                                del xs_scaled
                                idx[2] = torch.clamp(xs_rounded, 0, round(W*scale)-1)
                                # idx = idx * xs_okay.unsqueeze(0) + (~xs_okay.unsqueeze(0)) * -1
                                # idx.masked_fill_(xs_okay, value=-1)
                                # del xs_okay
                                # idx = torch.unique(idx, dim=-1)
                                # idx = idx[:, 1:]
                                # idx = torch.unique(idx.long(), dim=-1)
                                idx = idx.masked_select(xs_okay).view(3, -1)
                                
                                # print(nnz, idx.shape[-1])
                                return torch.sparse_coo_tensor(
                                    indices=idx,
                                    values=torch.ones((1,), device=img.device, dtype=img.dtype).expand(idx.shape[-1]),
                                    size=(N, H, round(W*scale)),
                                )
                            else:
                                return img

                        N, H, T, T_M = partial_attention_mask.shape
                        
                        # original
                        # partial_attention_mask_original = resize_from_m_to_t(partial_attention_mask, FP_MIN if not self.benchmarking else 0).view(N*H, T, T).to_sparse_coo()
                        
                        # optimized
                        SPARSITY_TYPE = 'flat_csr'
                        if SPARSITY_TYPE == 'flat_csr':
                            with timer("interp.csr"):
                                from .ops import resize_from_m_to_t_csr
                                partial_attention_mask = resize_from_m_to_t_csr(
                                    partial_attention_mask, 0, k=self.pconfig.k, target_width=T_SRC, is_causal=False, benchmarking=True, oversampled=self.pconfig.k_oversample
                                )
                        elif SPARSITY_TYPE == 'coo':
                            with timer("interp.coo"):
                                partial_attention_mask = partial_attention_mask.reshape(N*H, T, T_M).to_sparse_coo()
                                partial_attention_mask = resize_width(partial_attention_mask, T/T_M)
                        else:
                            raise Exception()
                    else:
                        # TODO support causal sparse masking
                        SPARSITY_TYPE = 'flat_csr'
                        
                        if SPARSITY_TYPE == 'flat_csr':
                            from .ops import resize_from_m_to_t_csr
                            partial_attention_mask = resize_from_m_to_t_csr(
                                partial_attention_mask, 0, k=self.pconfig.k, target_width=T_SRC, oversampled=self.pconfig.k_oversample
                            )
                        elif SPARSITY_TYPE == 'coo':
                            partial_attention_mask = resize_from_m_to_t(partial_attention_mask, FP_MIN if not self.benchmarking else 0).view(N*H, T, T).to_sparse_coo()
                        else:
                            raise Exception()
                
                raise_if_nan(partial_attention_mask)
            
            # return DUMMY_OUTPUT #1686
            
            if partial_attention_mask.is_sparse_csr:
                from .ops import flat_csr_to_dense
                get_bench().register_temp_buffer('partial_attention_mask', None, lazy=lambda: flat_csr_to_dense(partial_attention_mask, T_SRC, H))
            else:
                get_bench().register_temp_buffer('partial_attention_mask', partial_attention_mask)
            get_bench().register_temp_buffer('q_for_score', q_for_score)
            get_bench().register_temp_buffer('k_for_score', k_for_score)
            
            with timer("attention"):
                if not self.benchmarking:
                    # NOTE: checking avearge k is expected. uncomment following print, and then run visualize_glue
                    # avg_k_per_batch = (((partial_attention_mask > -1).view(N, -1).long().sum(-1) / (attention_mask > -1).long().view(N, -1).sum(-1)).mean() / H).item()
                    # print(metric.update(avg_k_per_batch, name='avgk'))
                    
                    attention_scores_dense = torch.matmul(q_for_score, k_for_score.transpose(-1, -2))
                    
                    # return DUMMY_OUTPUT #1774
                    
                    if attention_scores_truth is not None:
                        if not self.pconfig.causal:
                            attention_scores_dense = attention_scores_dense / math.sqrt(self.attention_head_size)
                            loss += kl_div_attention(
                                F.log_softmax(attention_scores_dense.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                                F.softmax(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1),
                                attention_mask,
                            ) * 0.1
                            loss += F.mse_loss(
                                softmax_bf16(attention_scores_dense.masked_fill(attention_mask < -1, FP_MIN), dim=-1, training=self.training), 
                                softmax_bf16(attention_scores_truth.masked_fill(attention_mask < -1, FP_MIN), dim=-1, training=self.training),
                            )
                        else:
                            attention_scores_dense = attention_scores_dense
                            with torch.autocast('cuda', torch.float32):
                                _t_causal_mask = causal_attention_mask < -1
                                # return DUMMY_OUTPUT #1778
                                _input = F.log_softmax(attention_scores_dense.masked_fill_(_t_causal_mask, FP_MIN).to(torch.float32), dim=-1, dtype=torch.float32).view(-1, attention_scores_dense.shape[-1])
                                # return DUMMY_OUTPUT #1970
                                _target = F.softmax(attention_scores_truth.masked_fill_(_t_causal_mask, FP_MIN).to(torch.float32), dim=-1, dtype=torch.float32).view(-1, attention_scores_dense.shape[-1])
                                # return DUMMY_OUTPUT #2162
                                loss += F.kl_div(
                                    _input,
                                    _target,
                                    reduction='batchmean',
                                ) * 0.1
                                del _input
                                # return DUMMY_OUTPUT #2738
                                loss += F.mse_loss(
                                    F.softmax(attention_scores_dense.to(torch.float32), dim=-1, dtype=torch.float32).view(-1, attention_scores_dense.shape[-1]), 
                                    _target,
                                )
                                del _target
                            # return DUMMY_OUTPUT #2738
                    get_bench().register_temp_buffer('attention_scores_dense', attention_scores_dense)
                    raise_if_nan(loss)
                    
                    # NOTE `attention_probs_dense` is for visualization, therefore it will not computed on benchmarking mode
                    if not self.pconfig.causal:
                        attention_scores_dense_masked = attention_scores_dense + attention_mask
                    else:
                        attention_scores_dense_masked = attention_scores_dense + causal_attention_mask
                    attention_probs_dense = softmax_bf16(attention_scores_dense_masked, dim=-1, training=self.training)
                    
                    # NOTE you should not add attention_mask and attention_score, because partial_attention_mask already has it.
                    raise_if_nan(partial_attention_mask)
                    partial_attention_scores = attention_scores_dense + partial_attention_mask
                    raise_if_nan(partial_attention_scores)
                    partial_attention_probs = softmax_bf16(partial_attention_scores, -1, training=self.training)
                    # partial_attention_probs = partial_attention_probs * (partial_attention_mask > -1) # this handles all zero row.
                    partial_attention_probs.masked_fill_(partial_attention_mask < -1, 0) # TODO verify this if works also with training?
                    get_bench().register_temp_buffer('partial_attention_scores', partial_attention_scores)
                    get_bench().register_temp_buffer('attention_matrix', partial_attention_probs)
                    raise_if_nan(partial_attention_probs)
                    
                    # perform scaling, however this pervent to use spase attention kernel
                    estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                    if self.pconfig.partial_attention_scaler:
                        partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scales[..., 0:1])
                    
                    raise_if_nan(partial_attention_probs)
                    raise_if_nan(v)
                    partial_context_layer = torch.matmul(partial_attention_probs, v)
                    get_bench().register_temp_buffer('partial_context_layer_1', partial_context_layer)
                else:
                    # TODO implement optimized causal kernel
                    # if self.pconfig.causal: raise Exception()
                    
                    attention_probs_dense = partial_attention_probs = attention_scores_dense = None
                    partial_context_layer = q_for_score
                    
                    # TODO Apply probs scaler!
                    
                    # NOTE: print avg k per batch
                    # avg_k_per_batch = (((partial_attention_mask.to_dense() > 0).view(N, -1).long().sum(-1) / (attention_mask > -1).long().view(N, -1).sum(-1)).mean() / H).item()
                    # print(metric.update(avg_k_per_batch, name='avgk'), flush=True)
                    
                    # using Numba
                    N, H, T, HEAD_H = q_for_score.shape
                    # print((partial_attention_mask > -1).sum(), (partial_attention_mask > -1).sum() / partial_attention_mask.numel())
                    with mem("attention"):
                        if partial_attention_mask.is_sparse_csr:
                            from .ops import (
                                flat_csr_masked_bmm,
                                flat_csr_softmax,
                                flat_csr_elmul,
                                flat_csr_sdbmm,
                            )
                            with timer('attention.sparse.maksed_bmm'):
                                partial_attention_scores = flat_csr_masked_bmm(
                                    q_for_score, k_for_score, partial_attention_mask
                                )
                            with timer('attention.sparse.softmax'):
                                partial_attention_probs = flat_csr_softmax(
                                    partial_attention_scores, H, T_SRC
                                )
                            with timer('attention.sparse.scaler'):
                                estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                            with timer('attention.sparse.elmul'):
                                if self.pconfig.partial_attention_scaler:
                                    row_scaler = torch.sigmoid(estimated_scales[..., 0]).view(N, H, T_DST, 1).expand(N, H, T_DST, T_SRC)
                                    partial_attention_probs = flat_csr_elmul(partial_attention_probs, row_scaler)
                            with timer('attention.sparse.sdbmm'):
                                partial_context_layer = flat_csr_sdbmm(partial_attention_probs, v, T_M)
                        else:
                            with timer("attention.coo"), mem("attention.coo"):
                                if not partial_attention_mask.is_sparse:
                                    sparse_attention_mask = partial_attention_mask.float().view(N*H, T, T).to_sparse_coo()#.coalesce() 
                                else:
                                    sparse_attention_mask = partial_attention_mask#.coalesce() 
                            with timer("attention.sparse"), mem("attention.sparse"):
                                assert sparse_attention_mask._nnz() > 0, sparse_attention_mask
                                partial_attention_scores = sparse_attn(
                                    q_for_score.reshape(N*H, T, HEAD_H).contiguous(), 
                                    k_for_score.reshape(N*H, T, HEAD_H).contiguous(), 
                                    sparse_attention_mask
                                ).coalesce()
                                if not self.pconfig.causal:
                                    with timer("attention.sparse.score"):
                                        partial_attention_scores = partial_attention_scores / math.sqrt(self.attention_head_size)
                                get_bench().register_temp_buffer('partial_attention_scores', partial_attention_scores)
                                del sparse_attention_mask
                                # partial_attention_scores = partial_attention_scores.to_dense()[0].to_sparse_csr().to_sparse_bsr((1,1)).clone()
                            with timer("attention.sparse_softmax"), mem("attention.sparse_softmax"):
                                partial_attention_probs = torch.sparse.softmax(
                                    partial_attention_scores, dim=2
                                )
                            with timer('attention.sparse_scale'):
                                # partial_attention_probs = partial_attention_probs.to_dense()
                                estimated_scales = self.attention_predictor_dec_scaler(t_attention_predictor)
                                if self.pconfig.partial_attention_scaler:
                                    partial_attention_probs = partial_attention_probs * torch.sigmoid(estimated_scales[..., 0].view(N*H, T, 1))
                            with timer("attention.bmm"), mem("attention.bmm"):
                                partial_context_layer = torch.bmm(partial_attention_probs, v.reshape(N*H, T, HEAD_H))
                                partial_context_layer = partial_context_layer.view(N, H, T, HEAD_H)
                        
                # return DUMMY_OUTPUT #2782
                
                with timer("attention.avg_pool"):
                    if not self.pconfig.causal:
                        average_context_layer = (
                            v *\
                            (dst_attention_mask > -1).to(v.dtype) *\
                            resize_from_m_to_t(
                                estimated_attention_probs.mean(-2, keepdim=True), 
                                masked_fill_value=0, 
                                target_width=T, 
                                handle_oversample=False
                            ).transpose(-1, -2)
                        ).sum(-2, keepdim=True).to(v.dtype)
                    else:
                        # TODO imporve this when causal
                        if not self.benchmarking:
                            avg_v = v * (dst_attention_mask > -1)
                            average_context_layer = avg_v.cumsum(-2) / torch.arange(1, avg_v.shape[-2]+1, device=avg_v.device).view(1, 1, -1, 1)
                            average_context_layer = average_context_layer.to(v.dtype)
                            if average_context_layer.shape[-2] > q.shape[-2]:
                                average_context_layer = average_context_layer[...,-q.shape[-2]:,:]
                        else:
                            if use_cache:
                                last_state, average_context_layer = PerlinAttentionState.stateful_cumavg(
                                    last_state,
                                    "output->cumavg",
                                    v,
                                    q.shape[-2],
                                )
                            else:
                                avg_v = v * (dst_attention_mask > -1)
                                average_context_layer = avg_v.cumsum(-2) / torch.arange(1, avg_v.shape[-2]+1, device=avg_v.device).view(1, 1, -1, 1)
                                average_context_layer = average_context_layer.to(v.dtype)
                                if average_context_layer.shape[-2] > q.shape[-2]:
                                    average_context_layer = average_context_layer[...,-q.shape[-2]:,:]
                        # return DUMMY_OUTPUT #2978
                    average_scale = torch.sigmoid(estimated_scales[..., 1:2])
                    partial_context_layer = partial_context_layer * average_scale + (1-average_scale) * average_context_layer
                    get_bench().register_temp_buffer('estimated_scales', estimated_scales)
                    get_bench().register_temp_buffer('average_scale', average_scale)
                    if not self.pconfig.causal:
                        get_bench().register_temp_buffer('estimated_attention_probs_t', None, lazy=lambda: resize_from_m_to_t(estimated_attention_probs.mean(-2, keepdim=True), 0, T, handle_oversample=False).transpose(-1, -2))
                    get_bench().register_temp_buffer('average_context_layer', average_context_layer)
                    get_bench().register_temp_buffer('partial_context_layer_2', partial_context_layer)
            
            # return DUMMY_OUTPUT #2978
            
            if self.pconfig.random_lookup:
                raise Exception()
                # TODO please consider updated estimated attention probs
                # lookup randomly that not looked up by partial context
                num_lookups = self.pconfig.random_lookup_count
                lookups = None
                estimated_attention_probs_masked = estimated_attention_probs * (attention_mask > -1) * (partial_attention_scores > -9999)
                for n in range(num_lookups):
                    token_length = (attention_mask.view(N, T) > -1).float().sum(dim=-1).view(N, 1, 1, 1)
                    # N, H, T, HID
                    random_context_index = torch.rand_like(partial_context_layer)
                    random_context_index = (random_context_index * (1 - 1/T) * token_length).floor().long()
                    
                    random_context_layer = v.gather(dim=-2, index=random_context_index)
                    random_context_weight = estimated_attention_probs_masked.gather(dim=-1, index=random_context_index)
                    random_context_layer = random_context_weight * random_context_layer
                    if lookups is None:
                        lookups = random_context_layer
                    else:
                        lookups = lookups + random_context_layer
                
                random_context_layer = random_context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = random_context_layer.size()[:-2] + (self.all_head_size,)
                random_context_layer = random_context_layer.view(new_context_layer_shape)

            with timer("context_permute"):
                partial_context_layer = partial_context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = partial_context_layer.size()[:-2] + (self.all_head_size,)
                partial_context_layer = partial_context_layer.view(new_context_layer_shape)
                if self.pconfig.out_add_performer_context:
                    performer_context_layer = performer_context_layer.permute(0, 2, 1, 3).contiguous()
                    performer_context_layer = performer_context_layer.view(new_context_layer_shape)
            
            # return DUMMY_OUTPUT #2978
            
            get_bench().register_temp_buffer('partial_context_layer_sparse', partial_context_layer)
            
            if self.pconfig.context_output_method == 'norm':
                raise Exception("if needed, please comment this")
                with timer("out"):
                    if not self.pconfig.random_lookup:
                        normalized_partial_context_layer = self.norm_partial(partial_context_layer)
                        get_bench().register_temp_buffer('normalized_partial_context_layer', normalized_partial_context_layer)
                        
                        partial_context_layer = \
                            normalized_partial_context_layer +\
                            partial_context_layer
                        if self.pconfig.out_add_performer_context:
                            raise Exception('performer context hidden size is modified')
                            partial_context_layer = partial_context_layer +\
                                self.norm_performer(performer_context_layer)
                    else:
                        raise Exception()
                        partial_context_layer = \
                            self.norm_partial(partial_context_layer) +\
                            self.norm_random(random_context_layer) +\
                            partial_context_layer
                        if self.pconfig.out_add_performer_context:
                            raise Exception('performer context hidden size is modified')
                            partial_context_layer = partial_context_layer +\
                                self.norm_performer(performer_context_layer)
                    
                    if self.pconfig.out_norm:
                        partial_context_layer = self.norm(partial_context_layer)
            elif self.pconfig.context_output_method == 'mix':
                pass
            else:
                raise Exception()
            
            # return DUMMY_OUTPUT #2978
            
            if not self.benchmarking:
                raise_if_nan(context_layer_truth)
                raise_if_nan(partial_context_layer)
                if context_layer_truth is not None:
                    loss += F.mse_loss(
                        context_layer_truth, 
                        partial_context_layer
                    )
                raise_if_nan(loss)

            raise_if_nan(loss)
            raise_if_nan(partial_context_layer)
            raise_if_nan(partial_attention_probs)
            raise_if_nan(attention_probs_dense)
            raise_if_nan(k_for_score)
            
            # return DUMMY_OUTPUT #3110
            
            estimated_attention_probs_for_output = estimated_attention_probs if self.benchmarking else estimated_attention_probs_resized
            # get_bench().register_temp_buffer('estimated_attention_probs_for_output', estimated_attention_probs_for_output)
            get_bench().register_temp_buffer('partial_context_layer', partial_context_layer)
            
            assert partial_context_layer.shape[-2] == q.shape[-2]
            
            return PerlinAttentionOutput(
                loss=loss,
                context_layer=partial_context_layer,
                partial_attention_probs=partial_attention_probs,
                partial_attention_mask=partial_attention_mask,
                estimated_attention_probs_m = estimated_attention_probs,
                estimated_attention_probs=estimated_attention_probs_for_output,
                dense_attention_probs=attention_probs_dense,
                key_for_score=k_for_score,
                state=last_state,
            )
