import os
import warnings
from typing import Any, Optional

import torch
import triton
from flash_attn import flash_attn_func

from hip_attn.v1_2.attention_extend import (
    dual_stage_quadratic_hip_attention,
    get_block_sparse_backend,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
)
from hip_attn.v1_2.hip_config import HiPAttentionConfig
from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache

try:
    import torch.distributed as dist
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        split_tensor_along_last_dim,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    SGLANG_DIST_ACTIVATED = True
except ImportError as ex:
    SGLANG_DIST_ACTIVATED = False

def get_local_rank() -> 0:
    if SGLANG_DIST_ACTIVATED:
        return get_tensor_model_parallel_rank()
    else:
        return 0

_CHECKOUT_COUNTER = 0


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def cuda_graph_capture_configs(hip_config: HiPAttentionConfig):
    num_stages = len(hip_config.layers[0].stages)
    cache_configs = [(None,)]  # (num_stage_cached,)
    for i_stage in range(num_stages):
        cache_configs.append((i_stage,))
    return cache_configs


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
    is_kv_cache_offload_enabled: Optional[bool] = False,
    rope_range: Optional[tuple[int, int]] = None,
    rope_is_neox_style: Optional[bool] = None,
    extend_seq_lens: Optional[torch.Tensor] = None,
    extend_seq_lens_cpu: Optional[torch.Tensor] = None,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
    offloading_metadata: Any = None,
    is_prefill: Optional[bool] = None,
    is_decode: bool = False,
    query_for_mask: Optional[torch.Tensor] = None,
    diag_sliding_window_indices: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:

    if is_prefill is not None:
        warnings.warn(
            "Deprecated behavior: `is_prefill` is deprecated. Use `is_decode` instead."
        )
        is_decode = not is_prefill

    if v is None:
        # warnings.warn(
        #     "Deprecated behavior: `k` and `v` should be provided in order to precisely know the output size."
        # )

        if v_cache is not None:
            v_hidden_dim = v_cache.shape[-1]
        else:
            assert offload_cache is not None
            v_hidden_dim = offload_cache.v_uvm.bank_cpu.shape[-1]

    else:
        if isinstance(v, list):
            v_hidden_dim = v[0].shape[-1]

        else:
            v_hidden_dim = v.shape[-1]

            if k.ndim == 3 and v.ndim == 3:  # Ignore if paged attn
                assert (
                    k_cache is not None and v_cache is not None
                ) or offload_cache is not None
                k = v = None

    if not is_decode and extend_seq_lens_cpu is not None:
        # Handle jagged inputs
        if is_kv_cache_offload_enabled is None:
            warnings.warn(
                "Deprecated behavior: `is_kv_cache_offload_enabled` must be specified in the future."
            )
            is_kv_cache_offload_enabled = k is not None and v is not None
        if is_kv_cache_offload_enabled:
            assert isinstance(k, list) and isinstance(v, list)
            assert isinstance(offloading_metadata, list)
            offload_cache = k_cache = v_cache = None

        BSZ_TDST, HEAD, _ = query.shape

        # Output tensor
        o = torch.empty(
            (BSZ_TDST, HEAD, v_hidden_dim),
            dtype=query.dtype,
            device=query.device,
        )
        metadata_new = cached_metadata

        start_len = 0
        decoding_reqs = []
        decoding_reqs_positions = []
        for idx_batch, seq_len in enumerate(extend_seq_lens_cpu):
            if seq_len == 0:  # Skip empty sequences
                decoding_reqs.append(idx_batch)
                decoding_reqs_positions.append(start_len)

            else:
                if not is_kv_cache_offload_enabled:
                    k_chunk = v_chunk = None
                    offloading_metadata_curr = None

                else:  # Offloading enabled
                    k_chunk, v_chunk, offloading_metadata_curr = (
                        k[idx_batch],
                        v[idx_batch],
                        offloading_metadata[idx_batch],
                    )

                o_req, _ = _forward_paged_hip_validate(
                    query=query[start_len : start_len + seq_len],
                    sm_scale=sm_scale,
                    batch_size=1,
                    k=k_chunk,
                    v=v_chunk,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,
                    positions=positions[start_len : start_len + seq_len],
                    seq_lens=seq_lens[idx_batch : idx_batch + 1],
                    req_to_tokens=req_to_tokens,
                    req_pool_indices=req_pool_indices[idx_batch : idx_batch + 1],
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    rope_range=rope_range,
                    rope_is_neox_style=rope_is_neox_style,
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    v_hidden_dim=v_hidden_dim,
                    hip_config=hip_config,
                    is_kv_cache_offload_enabled=is_kv_cache_offload_enabled,
                    cached_metadata=cached_metadata,
                    online_update_cache=online_update_cache,
                    offloading_metadata=offloading_metadata_curr,
                    is_decode=is_decode,
                    query_for_mask=query_for_mask,
                    diag_sliding_window_indices=diag_sliding_window_indices,
                    sliding_window_size=sliding_window_size,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                )

                o[start_len : start_len + seq_len] = o_req

            start_len += seq_len

        assert len(decoding_reqs) == 0

    else:
        o, metadata_new = _forward_paged_hip_validate(
            query=query,
            sm_scale=sm_scale,
            batch_size=batch_size,
            k_cache=k_cache,
            v_cache=v_cache,
            offload_cache=offload_cache,
            positions=positions,
            seq_lens=seq_lens,
            req_to_tokens=req_to_tokens,
            req_pool_indices=req_pool_indices,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            rope_range=rope_range,
            rope_is_neox_style=rope_is_neox_style,
            layer_id=layer_id,
            logit_cap=logit_cap,
            orig_context_len=orig_context_len,
            max_context_len=max_context_len,
            v_hidden_dim=v_hidden_dim,
            hip_config=hip_config,
            is_kv_cache_offload_enabled=is_kv_cache_offload_enabled,
            cached_metadata=cached_metadata,
            k=k,
            v=v,
            online_update_cache=online_update_cache,
            offloading_metadata=offloading_metadata,
            is_decode=is_decode,
            query_for_mask=query_for_mask,
            diag_sliding_window_indices=diag_sliding_window_indices,
            sliding_window_size=sliding_window_size,
            using_chunked_sliding_window=using_chunked_sliding_window,
        )

    return o, metadata_new


def _forward_paged_hip_validate(
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
    v_hidden_dim: int,
    hip_config: HiPAttentionConfig,
    is_kv_cache_offload_enabled: Optional[bool] = False,
    rope_range: Optional[tuple[int, int]] = None,
    rope_is_neox_style: Optional[bool] = None,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
    offloading_metadata: Any = None,
    is_decode: bool = False,
    query_for_mask: Optional[torch.Tensor] = None,
    diag_sliding_window_indices: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:

    if is_kv_cache_offload_enabled:
        if k is not None and v is not None:
            # BUG: this padding is neccesary to match non offload scenario. why?
            pad_size = max_context_len
            if k.shape[1] != pad_size:
                k_chunk_padded = torch.zeros(
                    (
                        k.shape[0],
                        pad_size,
                        k.shape[2],
                        k.shape[3],
                    ),
                    dtype=k.dtype,
                    device=k.device,
                )
                k_chunk_padded[:, : k.shape[1]] = k
                del k
                v_chunk_padded = torch.zeros(
                    (
                        v.shape[0],
                        pad_size,
                        v.shape[2],
                        v.shape[3],
                    ),
                    dtype=v.dtype,
                    device=v.device,
                )
                v_chunk_padded[:, : v.shape[1]] = v
                del v
                k = k_chunk_padded
                v = v_chunk_padded

    require_validation = offloading_metadata is not None
    if require_validation:
        if not is_decode:
            k_pages, v_pages = offloading_metadata
        else:
            k_cache_valid, v_cache_valid = offloading_metadata

            err_k = sse(offload_cache.k_uvm.bank_gpu, k_cache_valid)
            err_v = sse(offload_cache.v_uvm.bank_gpu, v_cache_valid)

    o, metadata_new = _forward_paged_hip(
        query=query,
        sm_scale=sm_scale,
        batch_size=batch_size,
        k_cache=k_cache,
        v_cache=v_cache,
        offload_cache=offload_cache,
        positions=positions,
        seq_lens=seq_lens,
        req_to_tokens=req_to_tokens,
        req_pool_indices=req_pool_indices,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        rope_range=rope_range,
        rope_is_neox_style=rope_is_neox_style,
        layer_id=layer_id,
        logit_cap=logit_cap,
        orig_context_len=orig_context_len,
        max_context_len=max_context_len,
        v_hidden_dim=v_hidden_dim,
        hip_config=hip_config,
        cached_metadata=cached_metadata,
        k=k,
        v=v,
        online_update_cache=online_update_cache,
        is_decode=is_decode,
        query_for_mask=query_for_mask,
        diag_sliding_window_indices=diag_sliding_window_indices,
        sliding_window_size=sliding_window_size,
        using_chunked_sliding_window=using_chunked_sliding_window,
    )

    if require_validation:
        if not is_decode:
            o_req_valid, _ = _forward_paged_hip(
                query=query,
                sm_scale=sm_scale,
                batch_size=batch_size,
                k_cache=k_pages,
                v_cache=v_pages,
                offload_cache=offload_cache,
                positions=positions,
                seq_lens=seq_lens,
                req_to_tokens=req_to_tokens,
                req_pool_indices=req_pool_indices,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_range=rope_range,
                rope_is_neox_style=rope_is_neox_style,
                layer_id=layer_id,
                logit_cap=logit_cap,
                orig_context_len=orig_context_len,
                max_context_len=max_context_len,
                v_hidden_dim=v_hidden_dim,
                hip_config=hip_config,
                cached_metadata=cached_metadata,
                k=k,
                v=v,
                online_update_cache=online_update_cache,
                is_decode=is_decode,
                query_for_mask=query_for_mask,
                diag_sliding_window_indices=diag_sliding_window_indices,
                sliding_window_size=sliding_window_size,
                using_chunked_sliding_window=using_chunked_sliding_window,
            )

            o_err = ((o - o_req_valid) ** 2).sum()
            assert o_err < 1e-6, o_err

        else:
            o_valid, metadata_valid = _forward_paged_hip(
                query=query,
                sm_scale=sm_scale,
                batch_size=batch_size,
                k_cache=k_cache_valid,
                v_cache=v_cache_valid,
                offload_cache=None,
                positions=positions,
                seq_lens=seq_lens,
                req_to_tokens=req_to_tokens,
                req_pool_indices=req_pool_indices,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_range=rope_range,
                rope_is_neox_style=rope_is_neox_style,
                layer_id=layer_id,
                logit_cap=logit_cap,
                orig_context_len=orig_context_len,
                max_context_len=max_context_len,
                v_hidden_dim=v_hidden_dim,
                hip_config=hip_config,
                cached_metadata=cached_metadata,
                k=k,
                v=v,
                online_update_cache=online_update_cache,
                is_decode=is_decode,
                query_for_mask=query_for_mask,
                diag_sliding_window_indices=diag_sliding_window_indices,
                sliding_window_size=sliding_window_size,
                using_chunked_sliding_window=using_chunked_sliding_window,
            )

            err_thresh = 1e-7

            o_sse = sse(o, o_valid)
            err_retry = -1
            err_uvm = None
            if o_sse >= err_thresh:
                indices_err = sse(metadata_new.indices, metadata_valid.indices)
                ks_err = sse(metadata_new.ks, metadata_valid.ks)
                ks_count_err = sse(metadata_new.ks_count, metadata_valid.ks_count)
                ks_start_end_err = sse(
                    metadata_new.ks_start_end, metadata_valid.ks_start_end
                )
                if (metadata_valid.stage_caches is not None) and (
                    len(metadata_valid.stage_caches) > 0
                ):
                    stage1_left_err = sse(
                        metadata_new.stage_caches[1].indices_left,
                        metadata_valid.stage_caches[1].indices_left,
                    )
                    stage1_right_err = sse(
                        metadata_new.stage_caches[1].indices_right,
                        metadata_valid.stage_caches[1].indices_right,
                    )
                    stage1_score_err = sse(
                        metadata_new.stage_caches[1].out_scores,
                        metadata_valid.stage_caches[1].out_scores,
                    )
                    stage2_left_err = sse(
                        metadata_new.stage_caches[2].indices_left,
                        metadata_valid.stage_caches[2].indices_left,
                    )
                    stage2_right_err = sse(
                        metadata_new.stage_caches[2].indices_right,
                        metadata_valid.stage_caches[2].indices_right,
                    )
                    stage2_score_err = sse(
                        metadata_new.stage_caches[2].out_scores,
                        metadata_valid.stage_caches[2].out_scores,
                    )
                else:
                    stage1_left_err = stage1_right_err = stage1_score_err = (
                        stage2_left_err
                    ) = stage2_right_err = stage2_score_err = None

                o_uvm, metadata_uvm = _forward_paged_hip(
                    query=query,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
                    k_cache=offload_cache.k_uvm.bank_gpu,
                    v_cache=offload_cache.v_uvm.bank_gpu,
                    offload_cache=None,
                    positions=positions,
                    seq_lens=seq_lens,
                    req_to_tokens=req_to_tokens,
                    req_pool_indices=req_pool_indices,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    rope_range=rope_range,
                    rope_is_neox_style=rope_is_neox_style,
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    v_hidden_dim=v_hidden_dim,
                    hip_config=hip_config,
                    cached_metadata=cached_metadata,
                    k=k,
                    v=v,
                    online_update_cache=online_update_cache,
                    is_decode=is_decode,
                    query_for_mask=query_for_mask,
                    diag_sliding_window_indices=diag_sliding_window_indices,
                    sliding_window_size=sliding_window_size,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                )

                offload_cache.sa_kv_cache.flush()
                offload_cache.mask_k_cache.flush()

                o_retry, metadata_retry = _forward_paged_hip(
                    query=query,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
                    k_cache=None,
                    v_cache=None,
                    offload_cache=offload_cache,
                    positions=positions,
                    seq_lens=seq_lens,
                    req_to_tokens=req_to_tokens,
                    req_pool_indices=req_pool_indices,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    rope_range=rope_range,
                    rope_is_neox_style=rope_is_neox_style,
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    v_hidden_dim=v_hidden_dim,
                    hip_config=hip_config,
                    cached_metadata=cached_metadata,
                    k=k,
                    v=v,
                    online_update_cache=online_update_cache,
                    is_decode=is_decode,
                    query_for_mask=query_for_mask,
                    diag_sliding_window_indices=diag_sliding_window_indices,
                    sliding_window_size=sliding_window_size,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                )
                err_uvm = sse(o, o_uvm)
                err_retry = sse(o_valid, o_retry)

                print(o)
                print(o_valid)
                print(metadata_new.indices)
                print(metadata_valid.indices)

                assert o_sse < err_thresh, (
                    f"sse={o_sse}\n"
                    f"err_k (uvm_k <=> valid_k) = {err_k}\n"
                    f"err_v (uvm_v <=> valid_v) = {err_v}\n"
                    f"err_retry (o_valid <=> o_retry) = {err_retry}\n"
                    f"err_uvm (o_first <=> o_uvm_retry) = {err_uvm}\n"
                    f"indices_err={indices_err}\n"
                    f"ks_err={ks_err}\n"
                    f"ks_count_err={ks_count_err}\n"
                    f"ks_start_end_err={ks_start_end_err}\n"
                    f"stage1_left_err={stage1_left_err}\n"
                    f"stage1_right_err={stage1_right_err}\n"
                    f"stage1_score_err={stage1_score_err}\n"
                    f"stage2_left_err={stage2_left_err}\n"
                    f"stage2_right_err={stage2_right_err}\n"
                    f"stage2_score_err={stage2_score_err}\n"
                    f"online_update={online_update_cache}\n"
                )

    return o, metadata_new


def sse(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype == b.dtype
    return ((a - b) ** 2).sum().item()


def _forward_paged_hip(
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
    v_hidden_dim: int,
    hip_config: HiPAttentionConfig,
    rope_range: Optional[tuple[int, int]] = None,
    rope_is_neox_style: Optional[bool] = None,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
    is_decode: bool = False,
    query_for_mask: Optional[torch.Tensor] = None,
    diag_sliding_window_indices: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:
    global _CHECKOUT_COUNTER

    N, num_heads, hidden_dims = query.shape
    dst_seq_len = N // batch_size

    is_dense = layer_id in hip_config.dense_layers
    if not is_decode:
        if len(hip_config.prefill_layers) == 2:
            layer_config = hip_config.prefill_layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.prefill_layers[layer_id]
    else:
        assert dst_seq_len == 1
        if len(hip_config.layers) == 2:
            layer_config = hip_config.layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.layers[layer_id]

    query = query.view(batch_size, dst_seq_len, num_heads, hidden_dims)
    if query_for_mask is not None:
        query_for_mask = query_for_mask.view(batch_size, -1, num_heads, hidden_dims)

    if k_cache is not None:
        N_PAGE, num_heads_kv, hidden_dims_v = v_cache.shape
        assert N_PAGE == k_cache.shape[0], f"{N_PAGE} != {k_cache.shape[0]}"

        k_cache = k_cache.view(N_PAGE, 1, num_heads_kv, k_cache.shape[-1])
        v_cache = v_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims_v)

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
        rope_range=rope_range,
        rope_is_neox_style=rope_is_neox_style,
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
        q_mask=(
            (query_for_mask * sm_scale).to(query.dtype)
            if query_for_mask is not None
            else None
        ),
        sliding_window_indices=(
            diag_sliding_window_indices[layer_id]
            if diag_sliding_window_indices is not None
            else None
        ),
        layer_id=layer_id,
        v_hidden_dim=v_hidden_dim,
        using_chunked_sliding_window=using_chunked_sliding_window,
    )

    using_dense_prefill = os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", "0") == "1"
    if is_decode:
        using_dense_prefill = False
    else:
        using_dense_prefill = using_dense_prefill and is_dense
        # using_dense_prefill = True
    
    force_dense_decode = os.getenv("HIP_DEBUG_FORCE_DENSE_DECODE", "0") == "1"
    last_dense = int(os.getenv("HIP_DEBUG_LAST_DENSE", "-1"))
    
    if last_dense > 0:
        last_dense += dst_seq_len % args.block_sparse_block_size_q
    
    # postfix_recompute_dense-window_[size:int]-diff_[1/0]-w_[size:int]
    # example: HIP_DELTA_ATTENTION_ARGS=recompute_dense-window_0-diff_1-w_64-dense_decode
    delta_attention_args = os.getenv('HIP_DELTA_ATTENTION_ARGS', None)
    using_delta_attention = delta_attention_args is not None
    
    if using_delta_attention:
        delta_attention_args_window = int(delta_attention_args.split("-")[1].split("_")[1])
        delta_attention_args_diff = int(delta_attention_args.split("-")[2].split("_")[1])
        delta_attention_args_w = int(delta_attention_args.split("-")[3].split("_")[1])
        delta_attention_args_just_return = 'JUST_RETURN' in delta_attention_args
        delta_attention_args_smooth = 'smooth' in delta_attention_args
        delta_attention_args_dense_decode = not ('sparse_decode' in delta_attention_args)

        if (layer_id == 0) and (get_local_rank() == 0):
            warnings.warn(
                f'Delta Attention is activated {delta_attention_args_window=} '
                f'{delta_attention_args_diff=} {delta_attention_args_w=} '
                f'{delta_attention_args_just_return=} '
                f'{delta_attention_args_smooth=} '
                f'{delta_attention_args_dense_decode=} '
            )
        
        # args.sa_extend_backend = "clamp"

    if isinstance(sliding_window_size, int) and (sliding_window_size > 0):
        bsa_fn = get_block_sparse_backend(args, query)

        # dist.barrier()
        # if get_tensor_model_parallel_rank() == 0:
        #     print(bsa_fn, args.using_extend, sliding_window_size, args.using_chunked_sliding_window)

        BSZ, TDST, HEAD, HID = query.shape

        args = args.clone()
        if args.rope_range is None:
            args.rope_range = (0, HID)
        args.block_size_q = args.block_sparse_block_size_q
        args.block_size_k = args.stages[-1].stage_chunk_size
        args.second_stage_k = 0
        args.sink_token_size = 0
        args.sliding_window_size = (
            sliding_window_size if sliding_window_size is not None else 1024
        )
        args.sliding_window_indices = None

        BDST = triton.cdiv(TDST, args.block_size_q)
        BH = BSZ * HEAD

        indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=query.device)
        ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros(
            (BH, BDST, 2), dtype=torch.int64, device=query.device
        )

        context = bsa_fn(
            q=(query * sm_scale).to(query.dtype),
            k=k,
            v=v,
            seq_lens=args.position_ids + 1,
            indices=indices,
            ks=ks,
            ks_count=ks_count,
            ks_start_end=ks_start_end,
            access_counter=None,
            cache_miss_counter=None,
            EXTEND_BACKEND=args.sa_extend_backend,
            model_context_length=args.model_context_length,
            extend_context_length=args.extend_context_length,
            offload_update_cache=False,
            args=args,
        )
        context = context.to(query.dtype)
        metadata = None
    elif using_delta_attention and ((not is_decode) or (is_decode and delta_attention_args_dense_decode)):
        if (is_decode and delta_attention_args_dense_decode) or (using_dense_prefill and (not is_decode)) or ((query.shape[1] < 256) and (not is_decode)):
            k_unpack = args.gather_k_from_paged_cache()
            v_unpack = args.gather_v_from_paged_cache()
            
            seq_len = args.position_ids.amax().item() + 1

            k_unpack = k_unpack[:, :seq_len]
            v_unpack = v_unpack[:, :seq_len]
            
            # if (layer_id == 0) and (get_local_rank() == 0):
            #     print('delta-dense', seq_len, layer_id, is_decode, force_dense_decode, using_dense_prefill)
            
            if args.need_apply_rope and args.using_extend:
                cos = args.rope_cos
                sin = args.rope_sin
                assert cos.ndim == 2, cos.shape
                assert sin.shape == cos.shape, sin.shape
                
                cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
                sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])
                
                idx_tsrc = torch.arange(0, k_unpack.shape[1], device=cos.device)
                idx_tsrc.clamp_min_(seq_len - args.model_context_length)

                k_unpack = (
                    (k_unpack * cos[:, idx_tsrc, :, :]) 
                    + (rotate_half(k_unpack) * sin[:, idx_tsrc, :, :])
                ).to(k_unpack.dtype)

                query = (
                    (query * cos[:, args.position_ids.view(-1), :, :]) 
                    + (rotate_half(query) * sin[:, args.position_ids.view(-1), :, :])
                ).to(query.dtype)
            
            k_unpack = k_unpack[:, :seq_len]
            v_unpack = v_unpack[:, :seq_len]
            
            context = flash_attn_func(
                query, 
                k_unpack, 
                v_unpack, 
                causal=True, 
                softmax_scale=sm_scale,
            )
            
            metadata = None
        else:
            # On prefill
            assert not is_decode
            assert not torch.cuda.is_current_stream_capturing()
            
            assert isinstance(delta_attention_args_window, int)
            if delta_attention_args_window == 0:
                assert delta_attention_args_window == 0
                context_sparse, metadata = dual_stage_quadratic_hip_attention(
                    (query * sm_scale).to(query.dtype),
                    k,
                    v,
                    args=args,
                    cached_metadata=cached_metadata,
                )
                context_sparse = context_sparse.to(query.dtype)
                context_sparse = context_sparse[:, -query.shape[1] :, :, :].contiguous()
            else:
                assert delta_attention_args_window > 0
                bsa_fn = get_block_sparse_backend(args, query)

                # dist.barrier()
                # if get_tensor_model_parallel_rank() == 0:
                #     print(bsa_fn, args.using_extend, sliding_window_size, args.using_chunked_sliding_window)

                BSZ, TDST, HEAD, HID = query.shape

                args_sw = args.clone()
                if args_sw.rope_range is None:
                    args_sw.rope_range = (0, HID)
                args_sw.block_size_q = args_sw.block_sparse_block_size_q
                args_sw.block_size_k = args_sw.stages[-1].stage_chunk_size
                args_sw.second_stage_k = 0
                # args_sw.sink_token_size = 0 #NOTE: you should inherit this value
                args_sw.sliding_window_size = delta_attention_args_window
                args_sw.sliding_window_indices = None

                BDST = triton.cdiv(TDST, args_sw.block_size_q)
                BH = BSZ * HEAD

                indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=query.device)
                ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
                ks_count = ks.unsqueeze(-1)
                ks_start_end = torch.zeros(
                    (BH, BDST, 2), dtype=torch.int64, device=query.device
                )

                context_sparse = bsa_fn(
                    q=(query * sm_scale).to(query.dtype),
                    k=k,
                    v=v,
                    seq_lens=args_sw.position_ids + 1,
                    indices=indices,
                    ks=ks,
                    ks_count=ks_count,
                    ks_start_end=ks_start_end,
                    access_counter=None,
                    cache_miss_counter=None,
                    EXTEND_BACKEND=args_sw.sa_extend_backend,
                    model_context_length=args_sw.model_context_length,
                    extend_context_length=args_sw.extend_context_length,
                    offload_update_cache=False,
                    args=args_sw,
                )
                context_sparse = context_sparse.to(query.dtype)
                context_sparse = context_sparse[:, -query.shape[1] :, :, :].contiguous()
                metadata = None
            
            if delta_attention_args_just_return:
                context = context_sparse
            else:
                from hip_attn.v1_2.triton_recompute import attention as recomp_attn
                
                num_queries = query.shape[1]
                num_last_dense = num_queries % delta_attention_args_w + max(128, delta_attention_args_w)
                num_sparse = num_queries - num_last_dense
                
                context_sparse = context_sparse[:, :num_sparse]
                
                if num_last_dense > 0:
                    idx = torch.arange(
                        0,
                        # delta_attention_args_w - 1, 
                        num_sparse, 
                        step=delta_attention_args_w, 
                        device=query.device
                    )
                    # take mean
                    # context_sparse_for_diff = context_sparse[:, :num_sparse]
                    # context_sparse_for_diff = context_sparse_for_diff.view(
                    #     context_sparse_for_diff.shape[0],
                    #     num_sparse // delta_attention_args_w,
                    #     delta_attention_args_w,
                    #     context_sparse_for_diff.shape[2],
                    #     context_sparse_for_diff.shape[3],
                    # )
                    # context_sparse_for_diff = context_sparse_for_diff.mean(dim=2)

                    # take first
                    context_sparse_for_diff = context_sparse[:, idx]

                    idx = torch.cat((idx, torch.arange(num_sparse, num_queries, device=query.device)))
                    query_for_dense = query[:, idx]
                
                # TODO: using paged attention
                repeated_k = args.gather_k_from_paged_cache(disable_gqa=True, gqa_q=query)
                repeated_v = args.gather_v_from_paged_cache(disable_gqa=True, gqa_q=query)  # B, T, H, D
                # assert repeated_k.shape[2] in (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64), repeated_k.shape
                assert repeated_k.shape[2] < 128

                seq_len = args.position_ids.amax().item() + 1
                repeated_k = repeated_k[:, :seq_len]
                repeated_v = repeated_v[:, :seq_len]

                query_for_recomp = query_for_dense
                
                if args.need_apply_rope and args.using_extend:
                    cos = args.rope_cos
                    sin = args.rope_sin
                    assert cos.ndim == 2, cos.shape
                    assert sin.shape == cos.shape, sin.shape
                    
                    cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
                    sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])
                    
                    idx_tsrc = torch.arange(0, repeated_k.shape[1], device=cos.device)
                    idx_tsrc.clamp_min_(seq_len - args.model_context_length)

                    repeated_k = (
                        (repeated_k * cos[:, idx_tsrc, :, :]) 
                        + (rotate_half(repeated_k) * sin[:, idx_tsrc, :, :])
                    ).to(repeated_k.dtype)

                    query_for_recomp = (
                        (query_for_recomp * cos[:, args.position_ids.view(-1)[idx], :, :]) 
                        + (rotate_half(query_for_recomp) * sin[:, args.position_ids.view(-1)[idx], :, :])
                    ).to(query_for_recomp.dtype)

                context_dense = recomp_attn(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    repeated_k.permute(0, 2, 1, 3).contiguous(), 
                    repeated_v.permute(0, 2, 1, 3).contiguous(), 
                    idx.unsqueeze(0), 
                    sm_scale,
                ).permute(0, 2, 1, 3).contiguous()
                
                if delta_attention_args_diff == 0:
                    context = torch.zeros_like(query)
                    context[:, :num_sparse] = context_sparse
                    context[:, idx] = context_dense
                else:
                    context_dense, last_context_dense = (
                        context_dense[:, :-num_last_dense],
                        context_dense[:, -num_last_dense:],
                    )
                    
                    # context_sparse_for_diff_norm = context_sparse_for_diff.float().square().sum(dim=-1, keepdim=True).sqrt()
                    # context_dense_norm = context_dense.float().square().sum(dim=-1, keepdim=True).sqrt()
                    # scale = context_dense_norm / context_sparse_for_diff_norm

                    # take difference
                    context_diff = context_dense - context_sparse_for_diff# * scale
                    
                    context_diff = context_diff.repeat_interleave(
                        delta_attention_args_w, dim=1
                    )
                    
                    if delta_attention_args_smooth:
                        # (exp) linear interpolate diff
                        context_diff_shift = torch.roll(context_diff, -delta_attention_args_w, 1)
                        context_diff_shift[:, -delta_attention_args_w:] = context_diff[:, -1:]

                        idx = torch.arange(0, context_diff.shape[1], device=context_diff.device)
                        idx = (idx % delta_attention_args_w).float() / delta_attention_args_w
                        context_diff = context_diff + (context_diff_shift - context_diff) * idx[None, :, None, None]
                
                    # context_sparse_norm = context_sparse.float().square().sum(dim=-1, keepdim=True).sqrt()
                    # scale = context_dense_norm.repeat_interleave(delta_attention_args_w, dim=1) / context_sparse_norm

                    # context = context_sparse * scale + context_diff
                    context = context_sparse + context_diff
                    context = torch.cat([context, last_context_dense], dim=1).to(query.dtype)

                    if get_local_rank() == 0:
                        print(
                            'hit', layer_id, 
                            context_diff.shape, 
                            context_sparse.shape, 
                            context_diff.abs().mean().item(), 
                            context_sparse.abs().mean().item()
                        )
    elif (force_dense_decode and is_decode) or (using_dense_prefill and (not is_decode)):
        if is_decode:
            if args.using_extend:
                args_dense = args.clone()
                args_dense.sliding_window_size = 777
                context, metadata = dual_stage_quadratic_hip_attention(
                    (query * sm_scale).to(query.dtype),
                    k,
                    v,
                    args=args_dense,
                    cached_metadata=cached_metadata,
                )
            else:
                bsa_fn = get_block_sparse_backend(args, query)

                BSZ, TDST, HEAD, HID = query.shape

                args_sw = args.clone()
                if args_sw.rope_range is None:
                    args_sw.rope_range = (0, HID)
                args_sw.block_size_q = args_sw.block_sparse_block_size_q
                args_sw.block_size_k = args_sw.stages[-1].stage_chunk_size
                args_sw.second_stage_k = 0
                args_sw.sliding_window_size = args_sw.model_context_length
                args_sw.sliding_window_indices = None

                BDST = triton.cdiv(TDST, args_sw.block_size_q)
                BH = BSZ * HEAD

                indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=query.device)
                ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
                ks_count = ks.unsqueeze(-1)
                ks_start_end = torch.zeros(
                    (BH, BDST, 2), dtype=torch.int64, device=query.device
                )

                context_sparse = bsa_fn(
                    q=(query * sm_scale).to(query.dtype),
                    k=k,
                    v=v,
                    seq_lens=args_sw.position_ids + 1,
                    indices=indices,
                    ks=ks,
                    ks_count=ks_count,
                    ks_start_end=ks_start_end,
                    access_counter=None,
                    cache_miss_counter=None,
                    EXTEND_BACKEND=args_sw.sa_extend_backend,
                    model_context_length=args_sw.model_context_length,
                    extend_context_length=args_sw.extend_context_length,
                    offload_update_cache=False,
                    args=args_sw,
                )
                context_sparse = context_sparse.to(query.dtype)
                context = context_sparse[:, -query.shape[1] :, :, :].contiguous()
                metadata = None
        else:
            k_unpack = args.gather_k_from_paged_cache()
            v_unpack = args.gather_v_from_paged_cache()
            
            seq_len = args.position_ids.amax().item() + 1

            k_unpack = k_unpack[:, :seq_len]
            v_unpack = v_unpack[:, :seq_len]
            
            if k_unpack.dtype in [torch.uint8]:
                k_unpack = k_unpack.view(torch.float8_e5m2).to(query.dtype)
                v_unpack = v_unpack.view(torch.float8_e5m2).to(query.dtype)
            assert k_unpack.dtype == query.dtype
            
            # if layer_id == 0:
            #     print(seq_len, layer_id, is_decode, force_dense_decode, using_dense_prefill)
            
            if args.need_apply_rope and args.using_extend:
                cos = args.rope_cos
                sin = args.rope_sin
                assert cos.ndim == 2, cos.shape
                assert sin.shape == cos.shape, sin.shape
                
                cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
                sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])
                
                idx_tsrc = torch.arange(0, k_unpack.shape[1], device=cos.device)
                idx_tsrc.clamp_min_(seq_len - args.model_context_length)
                
                assert cos.shape[1] >= k_unpack.shape[1], f'{cos.shape=} {k_unpack.shape}'

                k_unpack = (
                    (k_unpack * cos[:, idx_tsrc, :, :]) 
                    + (rotate_half(k_unpack) * sin[:, idx_tsrc, :, :])
                ).to(k_unpack.dtype)

                query = (
                    (query * cos[:, args.position_ids.view(-1), :, :]) 
                    + (rotate_half(query) * sin[:, args.position_ids.view(-1), :, :])
                ).to(query.dtype)
            
            k_unpack = k_unpack[:, :seq_len]
            v_unpack = v_unpack[:, :seq_len]
            
            context = flash_attn_func(
                query, 
                k_unpack, 
                v_unpack, 
                causal=True, 
                softmax_scale=sm_scale,
            )
            
            metadata = None
    elif is_decode or (query.shape[1] < (last_dense * 2)) or (last_dense <= 0):
        # dist.barrier()
        # if get_tensor_model_parallel_rank() == 0: print('hip')

        context, metadata = dual_stage_quadratic_hip_attention(
            (query * sm_scale).to(query.dtype),
            k,
            v,
            args=args,
            cached_metadata=cached_metadata,
        )
        context = context.to(query.dtype)
        context = context[:, -query.shape[1] :, :, :].contiguous()
    else:
        if True:
            assert query_for_mask is None
            position_ids = args.position_ids
            args.position_ids = position_ids[:, :-last_dense]
            context, metadata = dual_stage_quadratic_hip_attention(
                (query[:, :-last_dense, :, :] * sm_scale).to(query.dtype),
                k,
                v,
                args=args,
                cached_metadata=cached_metadata,
            )
            context_sparse = context.to(query.dtype)

            args_dense = args.clone()
            args_dense.sliding_window_size = 777
            args_dense.position_ids = position_ids[:, -last_dense:]
            context_dense, metadata = dual_stage_quadratic_hip_attention(
                (query[:, -last_dense:, :, :] * sm_scale).to(query.dtype),
                k,
                v,
                args=args_dense,
                cached_metadata=cached_metadata,
            )
            context_dense = context_dense.to(query.dtype)
            
            # assert not torch.cuda.is_current_stream_capturing()
            # k_unpack = args.gather_k_from_paged_cache()
            # v_unpack = args.gather_v_from_paged_cache()
            
            # position_ids = args.position_ids[:, -last_dense:]
            # seq_len = position_ids.amax().item() + 1

            # k_unpack = k_unpack[:, :seq_len]
            # v_unpack = v_unpack[:, :seq_len]
            
            # if k_unpack.dtype in [torch.uint8]:
            #     k_unpack = k_unpack.view(torch.float8_e5m2).to(query.dtype)
            #     v_unpack = v_unpack.view(torch.float8_e5m2).to(query.dtype)
            # assert k_unpack.dtype == query.dtype
            
            # # if layer_id == 0:
            # #     print(seq_len, layer_id, is_decode, force_dense_decode, using_dense_prefill)
            
            # query_for_dense = query[:, -last_dense:, :, :]
            
            # if args.need_apply_rope and args.using_extend:
            #     cos = args.rope_cos
            #     sin = args.rope_sin
            #     assert cos.ndim == 2, cos.shape
            #     assert sin.shape == cos.shape, sin.shape
                
            #     cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
            #     sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])
                
            #     idx_tsrc = torch.arange(0, k_unpack.shape[1], device=cos.device)
            #     idx_tsrc.clamp_min_(seq_len - args.model_context_length)

            #     k_unpack = (
            #         (k_unpack * cos[:, idx_tsrc, :, :]) 
            #         + (rotate_half(k_unpack) * sin[:, idx_tsrc, :, :])
            #     ).to(k_unpack.dtype)

            #     query_for_dense = (
            #         (query_for_dense * cos[:, position_ids.view(-1), :, :]) 
            #         + (rotate_half(query_for_dense) * sin[:, position_ids.view(-1), :, :])
            #     ).to(query_for_dense.dtype)
            
            
            # context_dense = flash_attn_func(
            #     query_for_dense, 
            #     k_unpack, 
            #     v_unpack, 
            #     causal=True, 
            #     softmax_scale=sm_scale,
            # )
            # context_dense = context_dense.to(query_for_dense.dtype)
            
            # # if get_local_rank() == 0:
            # #     print(context_dense)
            # #     print(layer_id, query_for_dense.shape, k_unpack.shape, seq_len, args.model_context_length, args.need_apply_rope and args.using_extend)

            context = torch.cat([context_sparse, context_dense], dim=1)
        else:
            assert query_for_mask is None
            block_size_q = args.stages[-1].stage_block_size_q
            k_bos = args.k_cache[args.block_table[:, :1], 0, :, :]
            k_bos = k_bos / k_bos.float().square().sum(dim=-1, keepdim=True).sqrt()
            q_norm = query / query.float().square().sum(dim=-1, keepdim=True).sqrt()
            # T_q
            scores = torch.matmul(
                q_norm.permute(0, 2, 1, 3),
                k_bos.permute(0, 2, 3, 1).repeat_interleave(
                    q_norm.shape[2] // k_bos.shape[2], 1
                ),
            )[0, :, :, 0].mean(dim=0)

            # scores = -torch.arange(0, scores.shape[0], device=scores.device, dtype=scores.dtype)

            # print(scores)
            half_window = 17
            scores = scores[None, None, :]
            # scores = torch.nn.functional.pad(scores[None, None, :], (half_window, half_window), mode='replicate')
            # scores = torch.nn.functional.avg_pool1d(
            #     scores, kernel_size=half_window*2+1, stride=1, padding=0
            # )
            # print(scores)
            scores = torch.nn.functional.pad(
                scores,
                (
                    0,
                    (
                        block_size_q - (scores.shape[-1] % block_size_q)
                        if scores.shape[-1] % block_size_q
                        else 0
                    ),
                ),
                mode="replicate",
            )
            scores = -torch.nn.functional.max_pool1d(
                -scores, kernel_size=block_size_q, stride=block_size_q, padding=0
            )[0, 0, :]
            # print(scores)
            scores[-4:].fill_(float("-inf"))
            # print(scores)
            scores = scores.repeat_interleave(block_size_q, 0)
            scores = scores[: q_norm.shape[1]]
            num_dense = 1024  # int(scores.shape[-1] * 0.025)
            # print(num_dense)
            num_dense = (
                num_dense
                + block_size_q
                - (
                    (num_dense % block_size_q)
                    if num_dense % block_size_q
                    else block_size_q
                )
            )
            # print(2, num_dense)
            num_dense = num_dense + q_norm.shape[1] % block_size_q
            # print(3, num_dense)
            num_dense = num_dense + block_size_q
            # print(4, num_dense)
            num_dense = max(64 + q_norm.shape[1] % block_size_q, num_dense)
            # print(5, num_dense)
            # num_dense = 256
            # print(num_dense, q_norm.shape[1] % block_size_q)
            _, dense_indices = torch.topk(
                -scores, dim=-1, k=num_dense, largest=True, sorted=True
            )
            # print(scores, scores.shape, num_dense)
            # print(dense_indices)
            dense_indices = dense_indices.sort().values
            # dense_indices = scores.shape[-1] - dense_indices - 1
            # print('a', dense_indices)
            # dense_indices = dense_indices // block_size_q * block_size_q
            # dense_indices = (dense_indices[::block_size_q, None] + torch.arange(0, block_size_q, device=query.device)[None, :]).view(-1)[:dense_indices.shape[-1]]
            # dense_indices = scores.shape[-1] - dense_indices - 1
            # print("b", dense_indices[::block_size_q], query.shape)
            sparse_indices = torch.arange(0, scores.shape[-1], device=query.device)
            sparse_indices.scatter_(dim=0, index=dense_indices, value=987654321)
            sparse_indices, _ = sparse_indices.sort()
            sparse_indices = sparse_indices[:-num_dense]

            check = torch.zeros((scores.shape[-1],), device=query.device)
            check.scatter_(dim=0, index=sparse_indices, value=-1)
            check.scatter_(dim=0, index=dense_indices, value=1)
            check = (check == 0).nonzero()
            # print((check == 0).nonzero(), query.shape[1], scores.shape, dense_indices, flush=True)
            assert check.shape[0] == 0, check
            assert (query.shape[1] - 1) in dense_indices
            check = ((dense_indices[::block_size_q] % block_size_q) != 0).nonzero()
            assert check.shape[0] == 0, check
            # assert ((query.shape[1] - 64) in dense_indices)

            dense_queries = query[:, dense_indices, :, :]
            sparse_queries = query[:, sparse_indices, :, :]

            position_ids = args.position_ids
            dense_pos_ids = position_ids[:, dense_indices]
            sparse_pos_ids = position_ids[:, sparse_indices]

            args.q_mask = None
            # args.sliding_window_size = 777  # NOTE: this 777 is correct
            args.position_ids = sparse_pos_ids
            context, metadata = dual_stage_quadratic_hip_attention(
                (sparse_queries * sm_scale).to(query.dtype),
                k,
                v,
                args=args,
                cached_metadata=cached_metadata,
            )
            context_sparse = context.to(query.dtype)

            args.sliding_window_size = 777  # NOTE: this 777 is correct
            args.position_ids = dense_pos_ids
            context, metadata = dual_stage_quadratic_hip_attention(
                (dense_queries * sm_scale).to(query.dtype),
                k,
                v,
                args=args,
                cached_metadata=cached_metadata,
            )
            context_dense = context.to(query.dtype)

            context = torch.full_like(query, fill_value=42)
            # context = context_all.to(query.dtype).clone()
            context.scatter_(
                dim=1,
                index=dense_indices[None, :, None, None].expand_as(context_dense),
                src=context_dense,
            )
            context.scatter_(
                dim=1,
                index=sparse_indices[None, :, None, None].expand_as(context_sparse),
                src=context_sparse,
            )

            check = (context == 42).nonzero()
            assert check.shape[0] == 0, f"{check} {check.shape}"
            # print(context)

    layers_to_capture = [0, 1, 2, 3, 4, 8, 12, 16, 24, 31]
    NEED_CHECKOUT = os.getenv("HIP_DEBUG_NEED_CHECKOUT", "0") == "1"
    if (
        NEED_CHECKOUT
        and (get_tensor_model_parallel_rank() == 0)
        and (layer_id in layers_to_capture)
    ):
        root = "./saves/sglang_decode"
        if not os.path.exists(root):
            _CHECKOUT_COUNTER = 0
        filename = f"{root}/checkout_sample_{_CHECKOUT_COUNTER}_layer_{layer_id}_is_decode_{1 if is_decode else 0}.pth"
        os.makedirs(root, exist_ok=True)

        if is_decode or (
            (not is_decode)
            and (dst_seq_len not in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
        ):
            torch.save(
                {
                    "q": query,
                    "sm_scale": sm_scale,
                    "k": (
                        k
                        if k is not None
                        else args.gather_k_from_paged_cache(chunk_size=1)
                    ),
                    "v": (
                        v
                        if k is not None
                        else args.gather_v_from_paged_cache(chunk_size=1)
                    ),
                    "block_table": block_table,
                    "cos": rope_cos,
                    "sin": rope_sin,
                    "out": context,
                    "metadata": metadata,
                },
                filename,
            )
            if is_decode and (layer_id == max(layers_to_capture)):
                _CHECKOUT_COUNTER += 1
            print(f"saved {filename}")

    return context.view(N, num_heads, context.shape[-1]), metadata
