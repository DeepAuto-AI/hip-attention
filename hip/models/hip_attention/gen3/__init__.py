import os
import torch
from typing import Optional

from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionOutputMetadata
from hip.models.hip_attention.gen3.uvm_gpu_cache import HiPOffloadCache
from hip.models.hip_attention.gen3.hip_config import HiPAttentionConfig
from hip.models.hip_attention.gen3.attention_extend import (
    dual_stage_quadratic_hip_attention,
)
from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionArgs


def forward_paged_hip(
    query: torch.Tensor,
    sm_scale: float,
    batch_size: int,
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    offload_cache: Optional[HiPOffloadCache],
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    layer_id: int,
    logit_cap: float,
    orig_context_len: int,
    max_context_len: int,
    hip_config: HiPAttentionConfig,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:
    N, num_heads, hidden_dims = query.shape
    dst_seq_len = N // batch_size

    is_decode = dst_seq_len == 1
    is_dense = layer_id in hip_config.dense_layers
    if not is_decode:
        if len(hip_config.prefill_layers) == 2:
            layer_config = hip_config.prefill_layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.prefill_layers[layer_id]
    else:
        if len(hip_config.layers) == 2:
            layer_config = hip_config.layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.layers[layer_id]

    query = query.view(batch_size, dst_seq_len, num_heads, hidden_dims)

    if k_cache is not None:
        N_PAGE, num_heads_kv, hidden_dims_kv = k_cache.shape
        assert v_cache.shape == k_cache.shape
        assert hidden_dims_kv == hidden_dims

        k_cache = k_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)
        v_cache = v_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)

    # FIXME: this operation is linear during decoding
    block_table = req_to_tokens.index_select(dim=0, index=req_pool_indices)

    BLOCK_TABLE_BSZ, MODEL_SEQ_LEN = block_table.shape
    assert batch_size == BLOCK_TABLE_BSZ

    # NOTE(heejun): the whole point to need to find gemma is large size of hidden size
    if k_cache is not None:
        hidden_size = k_cache.shape[-1]
    elif k is not None:
        hidden_size = k.shape[-1]
    elif offload_cache is not None:
        hidden_size = offload_cache.k_uvm.bank_cpu.shape[-1]
    else:
        raise Exception()
    is_gemma = hidden_size > 128

    require_cache_statistics = False
    if cached_metadata is None:
        require_cache_statistics = True
    elif cached_metadata.indices is None:
        require_cache_statistics = True
    elif os.getenv("HIP_DISABLE_COMPUTE_STATISTICS", "1") == "0":
        require_cache_statistics = True

    args = HiPAttentionArgs(
        k_cache=(
            k_cache.view(torch.uint8)
            if isinstance(k_cache, torch.Tensor) and k_cache.dtype == torch.float8_e5m2
            else k_cache
        ),
        v_cache=(
            v_cache.view(torch.uint8)
            if isinstance(k_cache, torch.Tensor) and v_cache.dtype == torch.float8_e5m2
            else v_cache
        ),
        offload_cache=offload_cache,
        block_table=block_table,
        cache_seq_lens=seq_lens,
        position_ids=positions.view(batch_size, dst_seq_len),
        block_size_k=32 if is_gemma else 64,  # BLOCK_CHUNK
        sliding_window_size=layer_config.sliding_window_size,
        sink_token_size=layer_config.sink_token_size,
        using_extend=hip_config.using_extend,
        need_apply_rope=hip_config.using_extend,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        logit_softcap=logit_cap if logit_cap != 0.0 else None,
        second_stage_k=layer_config.second_stage_k,
        stages=layer_config.stages,
        model_context_length=orig_context_len,
        extend_context_length=max_context_len,
        block_sparse_block_size_q=hip_config.block_sparse_block_size_q,
        scan_extend_backend=(
            (
                "relative"
                if hip_config.apply_v_dot
                else ("streaming" if is_dense else "relative")
            )
            if layer_config.scan_extend_backend is None
            else layer_config.scan_extend_backend
        ),
        sa_extend_backend=layer_config.sa_extend_backend,
        online_update_cache=online_update_cache,
        require_cache_statistics=require_cache_statistics,
        disable_flashdecode=not is_decode,
    )

    context, metadata = dual_stage_quadratic_hip_attention(
        (query * sm_scale).to(query.dtype),
        k,
        v,
        args=args,
        cached_metadata=cached_metadata,
    )
    context = context.to(query.dtype)

    return context.view(N, num_heads, hidden_dims), metadata
