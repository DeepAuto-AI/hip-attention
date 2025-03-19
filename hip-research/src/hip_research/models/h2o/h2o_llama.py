import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

# from transformers.models.llama.configuration_llama import LlamaConfig
import tqdm
from hip_research.models.modeling_llama_legacy import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from torch import nn
from transformers.models.llama.modeling_llama import (  # LlamaForCausalLM,
    LlamaRotaryEmbedding,
    rotate_half,
)
from transformers.utils import logging

from hip_attn.models.modeling_llama import (  # LlamaLinearScalingRotaryEmbedding,; LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaCustomAttention,
    LlamaForCausalLM,
)

__all__ = [
    "H2OLlamaForCausalLM",
    "H2OLlamaAttention",
    "H2OLlamaAttention_streaming",
    "H2OLlamaForCausalLM_streaming",
]

logger = logging.get_logger(__name__)

from transformers.configuration_utils import PretrainedConfig

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}"
            )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
    bsz: int,
    tgt_len: int,
    past_key_values_length: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def apply_rotary_pos_emb_single(
    x, cos, sin, position_ids, unsqueeze_dim=1, is_decoding=False
):
    # cos, sin : [bsz, seq_len, dim], position_ids : [bsz, seq_len]
    # Gather values from a according to position_ids along the T dimension (dim=1)
    _, _, cos_dim = cos.shape
    bsz, pos_t = position_ids.shape
    assert sin.shape[-1] == cos_dim
    assert x.shape[-2] == pos_t

    if not is_decoding:
        cos = torch.gather(
            cos, 1, position_ids.unsqueeze(-1).expand(bsz, pos_t, cos_dim)
        ).unsqueeze(
            1
        )  # [bs, 1, seq_len, dim]
        sin = torch.gather(
            sin, 1, position_ids.unsqueeze(-1).expand(bsz, pos_t, cos_dim)
        ).unsqueeze(
            1
        )  # [bs, 1, seq_len, dim]
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    x_embed = (x * cos) + (rotate_half(x) * sin)  # [bs, head, pos_t, dim]
    return x_embed


class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(
        self,
        past_key_values,
        attn_score_cache,
        num_key_value_groups=None,
        reduction_for_gqa=None,
        layer_idx=None,
        cache_position=None,
        num_new_tokens=None,
    ):  # , hh_score=None
        self._update_hh_score(
            attn_score_cache,
            num_key_value_groups,
            reduction_for_gqa,
            num_new_tokens=num_new_tokens,
        )  # , hh_score

        if (
            past_key_values is None or len(past_key_values) <= layer_idx
        ):  # TODO purpose?
            return (False, None)

        # seq_len = past_key_values[layer_idx][0].size(self.k_seq_dim)
        bsz, num_heads, seq_len, head_dim = past_key_values[layer_idx][0].shape

        if seq_len <= self.cache_size:  # TODO check
            return (False, past_key_values)

        # hh-selection

        select_hh_scores = self.hh_score[:, :, : seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(
            seq_len - self.recent_size, seq_len, device=keep_topk.device
        ).repeat(keep_topk.shape[0], keep_topk.shape[1], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        k_hh_recent = torch.gather(
            past_key_values.key_cache[layer_idx][:, :, :seq_len, :],
            -2,
            (keep_idx.unsqueeze(-1)).expand(-1, -1, -1, head_dim),
        )
        v_hh_recent = torch.gather(
            past_key_values.value_cache[layer_idx][:, :, :seq_len, :],
            -2,
            keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim),
        )

        self.hh_score = torch.gather(
            self.hh_score, -1, keep_idx.expand(bsz, num_heads, self.cache_size)
        )

        return (True, k_hh_recent, v_hh_recent)  # , self.hh_score

    def evict_for_space(self, past_key_values, num_coming):
        raise Exception()
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, : seq_len - self.recent_size + num_coming]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(
            seq_len - self.recent_size + num_coming, seq_len, device=keep_topk.device
        ).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(
            past_key_values[0].device
        )
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = (
            past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        )
        v_hh_recent = (
            past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        )

        self.hh_score = self.hh_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(
        self,
        attn_score_cache,
        num_key_value_groups=None,
        reduction_for_gqa=None,
        num_new_tokens=None,
    ):  # , hh_score
        # num_new_tokens = attn_score_cache.shape[2]
        assert num_new_tokens is not None
        assert attn_score_cache.ndim == 4
        attn_score_cache = attn_score_cache.sum(2)

        N, H, T = attn_score_cache.shape
        assert H % num_key_value_groups == 0

        if num_key_value_groups > 1:
            if reduction_for_gqa == "first_head":
                attn_score_cache = attn_score_cache[:, ::num_key_value_groups, :]
            elif reduction_for_gqa == "average":
                attn_score_cache = attn_score_cache.view(
                    N, H // num_key_value_groups, num_key_value_groups, T
                )
                attn_score_cache = torch.mean(attn_score_cache, dim=2)
            elif reduction_for_gqa == "max_total":
                b = torch.sum(attn_score_cache, dim=2)  # Shape of b is [N, H]
                k = H // num_key_value_groups  # Specify the value of k
                _, topk_indices = torch.topk(
                    b, k, dim=1
                )  # Shape of topk_indices is [N, k]

                topk_indices_expanded = topk_indices.unsqueeze(2).expand(
                    -1, -1, T
                )  # Expand indices for proper gathering
                attn_score_cache = torch.gather(
                    attn_score_cache, 1, topk_indices_expanded
                )  # Shape of attn_score_cache is [N, k, T]

            elif reduction_for_gqa == "max_group":
                b = torch.sum(attn_score_cache, dim=2)  # Shape of b is [N, H]
                k = (
                    H // num_key_value_groups
                )  # Specify the value of k (number of groups)

                # Reshape b to [N, k, H//k]
                b_reshaped = b.view(
                    N, k, num_key_value_groups
                )  # Shape of b_reshaped is [N, k, H//k]

                # Use torch.topk to get the indices of the top values within each group
                _, topk_indices = torch.topk(
                    b_reshaped, 1, dim=2
                )  # Shape of topk_indices is [N, k, 1]

                topk_indices_expanded = (
                    topk_indices.squeeze(2).unsqueeze(2).expand(-1, -1, T)
                )  # Shape [N, k, T]

                # Gather the topk heads from the original tensor a
                attn_score_cache = torch.gather(
                    attn_score_cache.view(N, k, num_key_value_groups, T),
                    2,
                    topk_indices_expanded.unsqueeze(2),
                ).squeeze(2)

        if self.hh_score is None:
            self.hh_score = attn_score_cache
        else:
            attn_score_cache[:, :, :-num_new_tokens] += self.hh_score  # BSZ, H, T
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None


class H2OLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int] = None
    ):  # layer_idx = 0?
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.config.attention_bias,
        )

        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            if self.config.rope_scaling is not None:
                self.rope_type = self.config.rope_scaling.get(
                    "rope_type", self.config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            if self.rope_type == "llama3":
                self.rope_type = "linear"
            scaling_factor = self.config.rope_scaling["factor"]
            if self.rope_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif self.rope_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {self.rope_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def _h2o_attention_itself(
        self,
        query_states,
        key_states,
        value_states,
        bsz,
        num_heads,
        head_dim,
        q_len,
        kv_seq_len,
        attention_mask,
        past_key_value,
        num_key_value_groups,
        reduction_for_gqa,
        layer_idx,
    ):

        # upcast attention to fp32

        N, HEAD, TDST, HID = query_states.shape
        N, HEAD_KV, TSRC, HID = key_states.shape
        assert key_states.shape == value_states.shape

        attn_weight_accumulator = torch.zeros(
            (N, HEAD, 1, TSRC), dtype=torch.float32, device=query_states.device
        )

        chunk_count = math.ceil((TDST * TSRC) / (2040 * 2048))
        chunk_size = math.ceil(TDST / chunk_count)

        attn_output_final = torch.empty_like(query_states)

        for i_tdst_start in tqdm.tqdm(range(0, TDST, chunk_size), leave=False, delay=1):
            i_tdst_end = min(i_tdst_start + chunk_size, TDST)
            attn_weights = torch.matmul(
                query_states[:, :, i_tdst_start:i_tdst_end],
                key_states[:, :, : i_tdst_end + TSRC - TDST].transpose(2, 3),
            ) / math.sqrt(head_dim)
            if TDST > 1:
                idx_tdst = torch.arange(
                    i_tdst_start, i_tdst_end, device=query_states.device
                )
                idx_tsrc = (
                    torch.arange(0, i_tdst_end, device=query_states.device)
                    + TSRC
                    - TDST
                )
                attn_weights = torch.where(
                    idx_tsrc[None, None, None, :] <= idx_tdst[None, None, :, None],
                    attn_weights,
                    -32000.0,
                )
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weight_accumulator[:, :, :, : i_tdst_end + TSRC - TDST].add_(
                attn_weights.sum(2, keepdim=True)
            )
            attn_output = torch.matmul(
                attn_weights, value_states[:, :, : i_tdst_end + TSRC - TDST]
            )

            attn_output_final.index_copy_(
                dim=2,
                index=torch.arange(i_tdst_start, i_tdst_end, device=attn_output.device),
                source=attn_output.to(attn_output_final.dtype),
            )
            # attn_output_final = attn_output

        attn_weights = attn_weight_accumulator
        attn_output = attn_output_final

        kv_hh = self.kv_cache(
            past_key_value,
            attn_weights,
            num_key_value_groups,
            reduction_for_gqa,
            layer_idx,
            num_new_tokens=TDST,
        )  # , hh_score TODO check

        if kv_hh[0] == True:
            _, k_hh_recent, v_hh_recent = kv_hh

            past_key_value.key_cache[layer_idx] = k_hh_recent
            past_key_value.value_cache[layer_idx] = v_hh_recent

        return attn_output, None

    def _h2o_attention(
        self,
        query_states,
        key_states,
        value_states,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        # hh_score,
        bsz,
        cache_position,
        reduction_for_gqa=None,
        kv_seq_len=None,
        decoding_loop_for_prefill=True,
        compute_final_attn_output=True,
        cos=None,
        sin=None,
        i=None,
        # position_embeddings=None,
    ):
        assert use_cache == True
        q_len = query_states.shape[-2]

        if decoding_loop_for_prefill:
            attention_mask = None

            kv_seq_len = key_states.shape[-2]

            if (
                past_key_value is not None and len(past_key_value) > self.layer_idx
            ):  # TODO check this for decoding
                kv_seq_len += past_key_value[self.layer_idx][0].shape[-2]

            """
            NOTE: H2O has 3 variants
            1. No touch on RoPE: official implementation
            2. shift query position of RoPE by min(k, pos idx): Implementation details in the official implementation that not API available. They describe this only with code comment
            3. StreamingLLM style RoPE: my own implementation
            """

            if self.config.h2o_streaming:
                if past_key_value is not None:  # and len(past_key_value) != 0:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )  # TODO check past_key_value update
                    # reuse k, v, self_attention
                    # key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    # value_states = torch.cat([past_key_value[1], value_states], dim=2)
                # past_key_value = (key_states, value_states) if use_cache else None

                position_ids = torch.arange(
                    0, key_states.shape[-2], device=key_states.device
                )[None, :]

                # cos, sin = self.rotary_emb(value_states, seq_len=position_length)
                # NOTE: grab all position embeddings
                # cos, sin = self.rotary_emb(value_states, position_ids)

                ### Shift Pos: query pos is min(cache_size, idx)
                query_states = apply_rotary_pos_emb_single(
                    query_states,
                    cos,
                    sin,
                    position_ids[:, -q_len:],
                    is_decoding=self.config.is_decoding,
                )
                key_states = apply_rotary_pos_emb_single(
                    key_states,
                    cos,
                    sin,
                    position_ids,
                    is_decoding=self.config.is_decoding,
                )
            else:
                # cos, sin = self.rotary_emb(value_states, seq_len=position_length)
                # NOTE: grab all position embeddings

                ### Shift Pos: query pos is min(cache_size, idx)
                query_states = apply_rotary_pos_emb_single(
                    query_states,
                    cos,
                    sin,
                    (
                        torch.clamp_max(position_ids, kv_seq_len)
                        if self.config.h2o_shift_q_pos
                        else position_ids
                    ),
                    is_decoding=self.config.is_decoding,
                )
                key_states = apply_rotary_pos_emb_single(
                    key_states,
                    cos,
                    sin,
                    position_ids,
                    is_decoding=self.config.is_decoding,
                )
                if past_key_value is not None:  # and len(past_key_value) != 0:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )  # TODO check past_key_value update
                    # reuse k, v, self_attention
                    # key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    # value_states = torch.cat([past_key_value[1], value_states], dim=2)
                # past_key_value = (key_states, value_states) if use_cache else None

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output, attn_weights = self._h2o_attention_itself(
            query_states,
            key_states,
            value_states,
            bsz,
            self.num_heads,
            self.head_dim,
            q_len,
            kv_seq_len,
            attention_mask,
            past_key_value,
            self.num_key_value_groups,
            reduction_for_gqa,
            self.layer_idx,
        )

        if compute_final_attn_output:
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(
                    self.hidden_size // self.config.pretraining_tp, dim=2
                )
                o_proj_slices = self.o_proj.weight.split(
                    self.hidden_size // self.config.pretraining_tp, dim=1
                )
                attn_output = sum(
                    [
                        F.linear(attn_output[i], o_proj_slices[i])
                        for i in range(self.config.pretraining_tp)
                    ]
                )
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

        return attn_output, attn_weights, past_key_value  # , hh_score

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        assert self.config.hh_size == self.config.tree_k // 2
        assert self.config.recent_size == self.config.tree_k // 2
        assert (
            self.config._attn_implementation
            == self.config.attn_implementation
            == "eager"
        )
        assert self.config.h2o_shift_q_pos is not None
        assert self.config.reduction_for_gqa is not None

        if self.config.attention_method == "h2o_stream":
            assert self.config.h2o_streaming == True
        assert use_cache == True

        mask_k = self.config.tree_k

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        if not self.config.is_decoding:
            position_ids_k = position_ids[:, :mask_k]
            position_ids_loop = position_ids[:, mask_k:]

            query_states_k = query_states[:, :, :mask_k, :]
            query_states_loop = query_states[:, :, mask_k:, :]
            key_states_k = key_states[:, :, :mask_k, :]
            key_states_loop = key_states[:, :, mask_k:, :]
            value_states_k = value_states[:, :, :mask_k, :]
            value_states_loop = value_states[:, :, mask_k:, :]

            # assert past_key_value is None # TODO CHECK
            assert use_cache is True

            attn_output, attn_weights, past_key_value = (
                self._h2o_attention(  # , hh_score
                    query_states_k,
                    key_states_k,
                    value_states_k,
                    position_ids_k,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # hh_score=hh_score,
                    bsz=bsz,
                    cache_position=cache_position,
                    reduction_for_gqa=self.config.reduction_for_gqa,
                    kv_seq_len=None,
                    decoding_loop_for_prefill=True,
                    compute_final_attn_output=True,
                    cos=cos,
                    sin=sin,
                )
            )
            attn_output_loop = torch.zeros(
                (attn_output.shape[0], q_len - mask_k, attn_output.shape[-1]),
                dtype=attn_output.dtype,
                device=attn_output.device,
            )

            # loop one by one
            assert query_states_loop.shape[-2] == q_len - mask_k
            for i in range(q_len - mask_k):
                # print(f'>> loop {i}')
                attn_output_, attn_weights_, past_key_value = (
                    self._h2o_attention(  # , hh_score
                        query_states_loop[:, :, i, :][:, :, None, :],
                        key_states_loop[:, :, i, :][:, :, None, :],
                        value_states_loop[:, :, i, :][:, :, None, :],
                        position_ids_loop[:, i][:, None],
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        # hh_score=hh_score,
                        bsz=bsz,
                        cache_position=cache_position,
                        reduction_for_gqa=self.config.reduction_for_gqa,
                        kv_seq_len=None,
                        decoding_loop_for_prefill=True,
                        compute_final_attn_output=True,
                        cos=cos,
                        sin=sin,
                        i=i,
                    )
                )

                attn_output_loop[:, i : i + 1, :].copy_(attn_output_, non_blocking=True)

            attn_output = torch.cat((attn_output, attn_output_loop), dim=1)

            if output_attentions:
                raise Exception()
                attn_weights = torch.cat((attn_weights, attn_weigth_loop), dim=-2)

        else:
            assert use_cache is True
            # compute_final_attn_output = False

            attn_output, attn_weights, past_key_value = (
                self._h2o_attention(  # , hh_score
                    query_states,
                    key_states,
                    value_states,
                    position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # hh_score=hh_score,
                    bsz=bsz,
                    cache_position=cache_position,
                    reduction_for_gqa=self.config.reduction_for_gqa,
                    kv_seq_len=None,
                    decoding_loop_for_prefill=True,
                    compute_final_attn_output=True,  # compute_final_attn_output
                    cos=cos,
                    sin=sin,
                )
            )
        return attn_output, attn_weights, past_key_value


class H2OLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.tree_dense_layers:
                self.model.layers[layer_idx].self_attn = LlamaCustomAttention(
                    config, layer_idx=layer_idx
                )
            else:
                self.model.layers[layer_idx].self_attn = H2OLlamaAttention(
                    config, layer_idx=layer_idx
                )


## H2O KV Cache dropping with Position rolling
class H2OLlamaAttention_streaming(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor = None,
        key_states: torch.Tensor = None,
        value_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        h2o_shift_q_pos: bool = False,
        mask_k: int = 512,
        reduction_for_gqa: str = "average",
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=(
                past_key_value[0].shape[-2]
                if (past_key_value is not None and len(past_key_value) != 0)
                else 0
            ),
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and len(past_key_value) != 0:
            kv_seq_len += past_key_value[0].shape[-2]

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        ###

        if past_key_value is not None and len(past_key_value) != 0:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        ### Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_position_ids = torch.arange(
            kv_seq_len, device=position_ids.device
        ).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        past_key_value = self.kv_cache(
            past_key_value,
            attn_weights.detach().clone(),
            self.num_key_value_groups,
            reduction_for_gqa,
            self.layer_idx,
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class H2OLlamaForCausalLM_streaming(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OLlamaAttention_streaming(config)
