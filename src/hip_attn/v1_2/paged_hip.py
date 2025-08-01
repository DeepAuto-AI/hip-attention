import copy
import os
import warnings
from typing import Any, List, Optional

import torch
import triton
from flash_attn import flash_attn_func
from matplotlib import pyplot as plt
from sgl_kernel.flash_attn import flash_attn_varlen_func as __flash_attn_varlen_func
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from hip_attn.v1_2.hip_config import HiPAttentionConfig
from hip_attn.v1_2.utils import capture


@capture
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    return __flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
    )


from hip_attn.v1_2.attention_extend import (
    dual_stage_quadratic_hip_attention,
    get_block_sparse_backend,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
    HiPAttentionState,
)
from hip_attn.v1_2.hip_config import HiPAttentionConfig
from hip_attn.v1_2.query_sparse_attention import query_sparse_attention
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
    block_table: torch.Tensor,
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
    extend_seq_lens_cpu: Optional[List[int]] = None,
    extend_prefix_lens_cpu: Optional[List[int]] = None,
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
    sliding_window_sink: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    self_extend_scale: int = 12,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata, HiPAttentionArgs]:

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

    if not is_decode:
        assert extend_seq_lens_cpu is not None

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

        if query.ndim == 4:
            # NOTE FIXME this seems not correct behavior.
            if extend_seq_lens_cpu is not None:
                if len(extend_seq_lens_cpu) != query.shape[0]:
                    assert (len(extend_seq_lens_cpu) % query.shape[0]) == 0
                    n_repeat = len(extend_seq_lens_cpu) // query.shape[0]
                    # query = query.repeat_interleave(n_repeat, 0)
                    # if k is not None:
                    #     k = k.repeat_interleave(n_repeat, 0)
                    # if v is not None:
                    #     v = v.repeat_interleave(n_repeat, 0)
                    extend_seq_lens_cpu = extend_seq_lens_cpu[::n_repeat]
                    extend_prefix_lens_cpu = extend_prefix_lens_cpu[::n_repeat]
            BSZ_TDST = query.shape[0] * query.shape[1]
            HEAD = query.shape[2]
        elif query.ndim == 3:
            BSZ_TDST, HEAD, _ = query.shape
        else:
            raise Exception()

        # Output tensor
        o = torch.empty(
            (BSZ_TDST, HEAD, v_hidden_dim),
            dtype=query.dtype,
            device=query.device,
        )
        metadata_new = []
        args_new = []

        if cached_metadata is not None:
            states = cached_metadata.state
            if isinstance(states, list) and (len(states) < len(extend_seq_lens_cpu)):
                assert (len(extend_seq_lens_cpu) % len(states)) == 0
                n_repeat = len(extend_seq_lens_cpu) // len(states)
                new_states = []
                for state in states:
                    for _ in range(n_repeat):
                        new_states.append(copy.deepcopy(state))
                states = new_states

        start_len = 0
        decoding_reqs = []
        decoding_reqs_positions = []

        # NOTE this is required for prefix
        assert extend_prefix_lens_cpu is not None
        assert len(extend_seq_lens_cpu) == len(extend_prefix_lens_cpu)

        for idx_batch, (seq_len, prefix_len) in enumerate(
            zip(extend_seq_lens_cpu, extend_prefix_lens_cpu)
        ):
            if query.ndim == 4:
                seq_len = query.shape[1]

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
                if k_chunk is None:
                    k_chunk = (
                        (
                            k[idx_batch : idx_batch + 1]
                            if k.ndim == 4
                            else k[start_len : start_len + seq_len]
                        )
                        if k is not None
                        else None
                    )
                if v_chunk is None:
                    v_chunk = (
                        (
                            v[idx_batch : idx_batch + 1]
                            if v.ndim == 4
                            else v[start_len : start_len + seq_len]
                        )
                        if v is not None
                        else None
                    )

                if cached_metadata is not None:
                    if isinstance(states, list):
                        cached_metadata.state = states[idx_batch]

                o_req, metadata_req, args_req = _forward_paged_hip_validate(
                    query=(
                        query[start_len : start_len + seq_len]
                        if query.ndim == 3
                        else query[idx_batch : idx_batch + 1]
                    ),
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
                    block_table=None,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    rope_range=rope_range,
                    rope_is_neox_style=rope_is_neox_style,
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    max_batch_context_len=seq_len + prefix_len,
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
                    sliding_window_sink=sliding_window_sink,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    self_extend_scale=self_extend_scale,
                )
                metadata_new.append(metadata_req)
                args_new.append(args_new)

                o[start_len : start_len + seq_len] = o_req

            start_len += seq_len

        assert len(decoding_reqs) == 0

    else:
        if block_table is None:
            raise Exception("this should not happened")

        o, metadata_new, args_new = _forward_paged_hip_validate(
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
            block_table=block_table,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            rope_range=rope_range,
            rope_is_neox_style=rope_is_neox_style,
            layer_id=layer_id,
            logit_cap=logit_cap,
            orig_context_len=orig_context_len,
            max_context_len=max_context_len,
            max_batch_context_len=max_context_len,
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
            sliding_window_sink=sliding_window_sink,
            using_chunked_sliding_window=using_chunked_sliding_window,
            k_descale=k_descale,
            v_descale=v_descale,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            self_extend_scale=self_extend_scale,
        )

    return o, metadata_new, args_new


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
    block_table: torch.Tensor,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    layer_id: int,
    logit_cap: float,
    orig_context_len: int,
    max_context_len: int,
    max_batch_context_len: int,
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
    sliding_window_sink: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    self_extend_scale: int = 12,
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

    o, metadata_new, args_new = _forward_paged_hip(
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
        block_table=block_table,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        rope_range=rope_range,
        rope_is_neox_style=rope_is_neox_style,
        layer_id=layer_id,
        logit_cap=logit_cap,
        orig_context_len=orig_context_len,
        max_context_len=max_context_len,
        max_batch_context_len=max_batch_context_len,
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
        sliding_window_sink=sliding_window_sink,
        using_chunked_sliding_window=using_chunked_sliding_window,
        k_descale=k_descale,
        v_descale=v_descale,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        self_extend_scale=self_extend_scale,
    )

    if require_validation:
        if not is_decode:
            o_req_valid, _, _ = _forward_paged_hip(
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
                max_batch_context_len=max_batch_context_len,
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
                sliding_window_sink=sliding_window_sink,
                using_chunked_sliding_window=using_chunked_sliding_window,
                k_descale=k_descale,
                v_descale=v_descale,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                self_extend_scale=self_extend_scale,
            )

            o_err = ((o - o_req_valid) ** 2).sum()
            assert o_err < 1e-6, o_err

        else:
            o_valid, metadata_valid, _ = _forward_paged_hip(
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
                max_batch_context_len=max_batch_context_len,
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
                sliding_window_sink=sliding_window_sink,
                using_chunked_sliding_window=using_chunked_sliding_window,
                k_descale=k_descale,
                v_descale=v_descale,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                self_extend_scale=self_extend_scale,
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

                o_uvm, metadata_uvm, _ = _forward_paged_hip(
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
                    max_batch_context_len=max_batch_context_len,
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
                    sliding_window_sink=sliding_window_sink,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    self_extend_scale=self_extend_scale,
                )

                offload_cache.sa_kv_cache.flush()
                offload_cache.mask_k_cache.flush()

                o_retry, metadata_retry, _ = _forward_paged_hip(
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
                    max_batch_context_len=max_batch_context_len,
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
                    sliding_window_sink=sliding_window_sink,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    self_extend_scale=self_extend_scale,
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

    return o, metadata_new, args_new


def sse(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype == b.dtype
    return ((a - b) ** 2).sum().item()


@capture
def _forward_delta_attn(
    query: torch.Tensor,
    sm_scale: float,
    k: torch.Tensor,
    v: torch.Tensor,
    args: HiPAttentionArgs,
    cached_metadata: HiPAttentionOutputMetadata,
    is_decode: bool,
    delta_attention_args_smooth=False,
    delta_attention_args_just_return=False,
    delta_attention_args_window=0,
    delta_attention_args_diff=1,
    delta_attention_args_dense_decode=False,
    delta_attention_args_w=16,
    delta_attention_args_exp=False,
    delta_attention_args_exp_w=2,
    delta_attention_args_exp_window=1024,
    delta_attention_args_exp_sink=128,
    delta_attention_args_iter_corr=False,
    delta_attention_args_adjust_norm_const=False,
    delta_attention_args_extend="none",
    k_descale: torch.Tensor = None,
    v_descale: torch.Tensor = None,
    rope_cos: torch.Tensor = None,
    rope_sin: torch.Tensor = None,
):
    assert not is_decode

    # using_dense_prefill = False

    # if (
    #     (is_decode and delta_attention_args_dense_decode)
    #     or (using_dense_prefill and (not is_decode))
    #     or ((query.shape[1] < 256) and (not is_decode))
    # ):
    #     # for dense decode, BUG this is so slow why?
    #     if args.need_apply_rope and args.using_extend:
    #         k_unpack = args.gather_k_from_paged_cache()
    #         v_unpack = args.gather_v_from_paged_cache()

    #         seq_len = args.position_ids.amax().item() + 1

    #         k_unpack = k_unpack[:, :seq_len]
    #         v_unpack = v_unpack[:, :seq_len]

    #         cos = args.rope_cos
    #         sin = args.rope_sin
    #         assert cos.ndim == 2, cos.shape
    #         assert sin.shape == cos.shape, sin.shape

    #         cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
    #         sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])

    #         idx_tsrc = torch.arange(0, k_unpack.shape[1], device=cos.device)
    #         idx_tsrc.clamp_min_(seq_len - args.model_context_length)

    #         k_unpack = (
    #             (k_unpack * cos[:, idx_tsrc, :, :])
    #             + (rotate_half(k_unpack) * sin[:, idx_tsrc, :, :])
    #         ).to(k_unpack.dtype)

    #         query = (
    #             (query * cos[:, args.position_ids.view(-1), :, :])
    #             + (rotate_half(query) * sin[:, args.position_ids.view(-1), :, :])
    #         ).to(query.dtype)

    #         k_unpack = k_unpack[:, :seq_len]
    #         v_unpack = v_unpack[:, :seq_len]

    #         context = flash_attn_func(
    #             query,
    #             k_unpack,
    #             v_unpack,
    #             causal=True,
    #             softmax_scale=sm_scale,
    #         )
    #     else:
    #         assert args.using_paged_cache

    #         k_cache = args.get_k_cache()
    #         v_cache = args.get_v_cache()

    #         q_reshaped = query\
    #             .contiguous()\
    #             .view(-1, query.shape[2], query.shape[3])\
    #             .to(k_cache.dtype)

    #         # print(k_cache.shape, v_cache.shape)

    #         cu_seqlens_q = (
    #             torch.arange(
    #                 0, query.shape[0] + 1, device=query.device, dtype=torch.int32
    #             )
    #             * query.shape[1]
    #         )
    #         cache_seqlens = (args.position_ids[:, -1] + 1).to(torch.int32)
    #         cu_seqlens_k_new = torch.zeros(
    #             (args.position_ids.shape[0] + 1,),
    #             dtype=torch.int32,
    #             device=q_reshaped.device,
    #         )
    #         cu_seqlens_k_new[1:] = cache_seqlens

    #         block_table = args.block_table

    #         context = flash_attn_with_kvcache(
    #             q=q_reshaped,
    #             k_cache=k_cache,
    #             v_cache=v_cache,
    #             page_table=block_table,
    #             cache_seqlens=cache_seqlens,
    #             cu_seqlens_q=cu_seqlens_q,
    #             cu_seqlens_k_new=cu_seqlens_k_new,
    #             # max_seqlen_q=cu_seqlens_q.amax().item(),
    #             max_seqlen_q=args.model_context_length,
    #             causal=True,
    #             softmax_scale=sm_scale,
    #         )

    #     metadata = None
    # else:

    # On prefill
    assert not is_decode
    assert not torch.cuda.is_current_stream_capturing()

    assert isinstance(delta_attention_args_window, int)
    if delta_attention_args_window == 0:
        assert delta_attention_args_window == 0

        # args_new = args.clone()
        # k_flat = args.gather_k_from_paged_cache()
        # v_flat = args.gather_v_from_paged_cache()
        # seq_len = args.position_ids.amax().item() + 1
        # k_flat = k_flat[:, :seq_len].contiguous()
        # v_flat = v_flat[:, :seq_len].contiguous()
        # args_new.k_cache = None
        # args_new.v_cache = None
        # args_new.block_table = None
        # args_new.using_paged_cache = False
        # cached_metadata.state = None

        # context_sparse, metadata = dual_stage_quadratic_hip_attention(
        #     q=(query * sm_scale).to(query.dtype),
        #     k=k_flat,
        #     v=v_flat,
        #     args=args_new,
        #     cached_metadata=cached_metadata,
        # )

        delta_exp = delta_attention_args_exp

        if delta_exp:
            delta_exp_w = delta_attention_args_exp_w
            delta_exp_bk = 16
            delta_exp_k = 0
            delta_exp_window = delta_attention_args_exp_window
            delta_exp_sink = delta_attention_args_exp_sink
            delta_merge_strategy = "delta"  # replace / delta
            if delta_exp_k == 0:
                delta_exp_bk = 64

            bsa_fn = get_block_sparse_backend(query, args.disable_flashdecode)

            BSZ, TDST, HEAD, HID = query.shape

            args_sw = args.clone()
            if args_sw.rope_range is None:
                args_sw.rope_range = (0, HID)
            args_sw.block_size_q = args_sw.block_sparse_block_size_q
            args_sw.block_size_k = delta_exp_bk
            args_sw.second_stage_k = delta_exp_k
            args_sw.sink_token_size = delta_exp_sink
            args_sw.sliding_window_size = delta_exp_window
            args_sw.sliding_window_indices = None

            BDST = triton.cdiv(TDST, args_sw.block_size_q)
            BH = BSZ * HEAD

            if delta_exp_k == 0:
                indices = torch.zeros(
                    (BH, BDST, delta_exp_k // delta_exp_bk),
                    dtype=torch.int64,
                    device=query.device,
                )
                ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
                ks_count = ks.unsqueeze(-1)
                ks_start_end = torch.zeros(
                    (BH, BDST, 2), dtype=torch.int64, device=query.device
                )
                ks_start_end[:, :, 1:] = ks[:, :, None]
            else:
                indices = torch.rand(
                    (BH, BDST, delta_exp_k // delta_exp_bk), device=query.device
                )
                indices = (
                    indices
                    * args_sw.position_ids[
                        :, :: args_sw.block_size_q
                    ].repeat_interleave(HEAD, dim=0)[:, :, None]
                )
                indices = indices.to(torch.int64) // delta_exp_bk * delta_exp_bk

                indices, _ = indices.sort(dim=-1)
                indices = indices // args_sw.block_size_k * args_sw.block_size_k

                unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
                indices = torch.where(
                    unique_mask, indices, torch.iinfo(indices.dtype).max
                )
                indices, _ = indices.sort(dim=-1)
                active_mask = indices < (
                    args_sw.position_ids[
                        :, :: args_sw.block_size_q, None
                    ].repeat_interleave(HEAD, 0)
                    + args_sw.block_size_q
                )
                ks = active_mask.int().sum(-1)
                ks_count = ks.unsqueeze(-1)
                ks_start_end = torch.zeros(
                    (ks.shape[0], ks.shape[1], 2),
                    dtype=torch.int32,
                    device=query.device,
                )
                ks_start_end[:, :, -1] = ks

            context_sw = bsa_fn(
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
            context_sw = context_sw.to(query.dtype)

            args_sparse = args.clone()
            query_sparse = query[:, ::delta_exp_w].contiguous()
            args_sparse.position_ids = args.position_ids[:, ::delta_exp_w].contiguous()
            args_sparse.query_for_landmark = query
            args_sparse.position_ids_for_landmark = args.position_ids

            # args_new = args_sparse.clone()
            # k_flat = args_sparse.gather_k_from_paged_cache()
            # v_flat = args_sparse.gather_v_from_paged_cache()
            # seq_len = args_sparse.position_ids.amax().item() + 1
            # k_flat = k_flat[:, :seq_len].contiguous()
            # v_flat = v_flat[:, :seq_len].contiguous()
            # args_new.k_cache = None
            # args_new.v_cache = None
            # args_new.block_table = None
            # args_new.using_paged_cache = False
            # cached_metadata.state = None

            # context_sparse, metadata = dual_stage_quadratic_hip_attention(
            #     q=(query_sparse * sm_scale).to(query.dtype),
            #     k=k_flat,
            #     v=v_flat,
            #     args=args_new,
            #     cached_metadata=cached_metadata,
            # )

            context_sparse, metadata = dual_stage_quadratic_hip_attention(
                q=(query_sparse * sm_scale).to(query.dtype),
                k=k,
                v=v,
                args=args_sparse,
                cached_metadata=cached_metadata,
            )
            context_sparse = context_sparse.to(query.dtype)

            if delta_merge_strategy == "delta":
                context_sw_for_sparse = context_sw[:, ::delta_exp_w]
                delta_sparse = context_sparse - context_sw_for_sparse

                delta_sparse = delta_sparse.repeat_interleave(delta_exp_w, dim=1)

                if delta_attention_args_smooth:
                    # (exp) linear interpolate diff
                    delta_sparse_shift = torch.roll(delta_sparse, -delta_exp_w, 1)
                    delta_sparse_shift[:, -delta_exp_w:] = delta_sparse[:, -1:]

                    idx = torch.arange(
                        0, delta_sparse.shape[1], device=delta_sparse.device
                    )
                    idx = (idx % delta_exp_w).float() / delta_exp_w
                    delta_sparse = (
                        delta_sparse
                        + (delta_sparse_shift - delta_sparse) * idx[None, :, None, None]
                    )

                context_sparse = context_sw + delta_sparse[:, : context_sw.shape[1]]
            elif delta_merge_strategy == "replace":
                context_sw[:, ::delta_exp_w] = context_sparse
                context_sparse = context_sw
            else:
                raise Exception()
        else:
            # args_new = args.clone()
            # k_flat = args.gather_k_from_paged_cache()
            # v_flat = args.gather_v_from_paged_cache()
            # seq_len = args.position_ids.amax().item() + 1
            # k_flat = k_flat[:, :seq_len].contiguous()
            # v_flat = v_flat[:, :seq_len].contiguous()
            # args_new.k_cache = None
            # args_new.v_cache = None
            # args_new.block_table = None
            # args_new.using_paged_cache = False
            # cached_metadata.state = None

            # context_sparse, metadata = dual_stage_quadratic_hip_attention(
            #     q=(query * sm_scale).to(query.dtype),
            #     k=k_flat,
            #     v=v_flat,
            #     args=args_new,
            #     cached_metadata=cached_metadata,
            # )

            args.bsa_return_running_statistics = delta_attention_args_adjust_norm_const

            context_sparse, metadata = dual_stage_quadratic_hip_attention(
                q=(query * sm_scale).to(query.dtype),
                k=k,
                v=v,
                args=args,
                cached_metadata=cached_metadata,
            )

            if delta_attention_args_adjust_norm_const:
                context_sparse, (sparse_mx, sparse_nc) = context_sparse

            context_sparse = context_sparse.to(query.dtype)
            context_sparse = context_sparse[:, -query.shape[1] :, :, :].contiguous()

            if delta_attention_args_adjust_norm_const:
                sparse_mx = sparse_mx[:, -query.shape[1] :].contiguous()
                sparse_nc = sparse_nc[:, -query.shape[1] :].contiguous()
    else:
        assert delta_attention_args_window > 0
        bsa_fn = get_block_sparse_backend(query, args.disable_flashdecode)

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

        if os.getenv("HIP_DEBUG_FORCE_CHUNKED_SW", "0") == "1":
            args_sw.using_chunked_sliding_window = True

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
            return_running_statistics=delta_attention_args_adjust_norm_const,
            args=args_sw,
        )
        if delta_attention_args_adjust_norm_const:
            context_sparse, (sparse_mx, sparse_nc) = context_sparse

        context_sparse = context_sparse.to(query.dtype)
        context_sparse = context_sparse[:, -query.shape[1] :, :, :].contiguous()
        if delta_attention_args_adjust_norm_const:
            sparse_mx = sparse_mx[:, -query.shape[1] :].contiguous()
            sparse_nc = sparse_nc[:, -query.shape[1] :].contiguous()
        metadata = None

    # until here, we have only calculated sparse attention
    if delta_attention_args_just_return:
        context = context_sparse
    elif delta_attention_args_iter_corr:
        w_size = delta_attention_args_w * 2

        num_queries = query.shape[1]
        num_dense_first = max(128, w_size)
        num_dense_last = num_queries % w_size + max(128, w_size)
        num_sparse = num_queries - num_dense_first - num_dense_last

        # iteratively correction errors

        def perform_correction(
            context_sparse: torch.Tensor,
            context_sparse_raw: torch.Tensor,
            block_start_indices: torch.Tensor,
            block_size: int,
        ):
            assert block_start_indices.ndim == 1
            assert context_sparse.ndim == 4
            assert context_sparse_raw.shape == context_sparse.shape
            assert not (args.need_apply_rope and args.using_extend)

            assert args.using_paged_cache

            if False:
                context_sparse_raw = context_sparse

            query_for_recomp = query[:, block_start_indices, :, :]
            k_cache = args.get_k_cache()
            v_cache = args.get_v_cache()

            assert args.position_ids.shape[0] == 1
            if get_local_rank() == 0:
                # import matplotlib.pyplot as plt
                # plt.clf()
                # plt.hist(block_start_indices.cpu().numpy(), bins=50)
                # plt.xlim(0, args.position_ids.amax().item() + 1)
                # plt.savefig(f'./dummy_indices_hist_{len(block_start_indices)}.png')

                print(
                    "recomp_attn shapes",
                    query_for_recomp.shape,
                    block_start_indices.shape,
                )
            context_dense = (
                query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    None,
                    None,
                    args.position_ids[:, block_start_indices],
                    sm_scale,
                    k_cache,
                    v_cache,
                    args.block_table,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )  # type: torch.Tensor
            assert context_dense.shape[-2:] == query.shape[-2:]

            if block_size > 1:
                assert not delta_attention_args_smooth
                block_diff = diff = (
                    context_dense - context_sparse_raw[:, block_start_indices]
                )
                diff = diff.repeat_interleave(block_size, 1)

                token_indices = (
                    block_start_indices[:, None]
                    + torch.arange(0, block_size, device=context_sparse.device)[None, :]
                )
                token_indices = token_indices.view(-1)

                context_sparse_new = diff + context_sparse_raw[:, token_indices]
                context_sparse.index_copy_(
                    dim=1, index=token_indices, source=context_sparse_new
                )
            else:
                context_sparse.index_copy_(
                    dim=1, index=block_start_indices, source=context_dense
                )
                block_diff = None

            return context_sparse, block_diff

        block_start_indices = torch.arange(
            num_dense_first,
            num_dense_first + num_sparse,
            step=w_size,
            device=query.device,
        )
        assert (num_dense_first % w_size) == 0
        assert ((num_dense_first + num_sparse) % w_size) == 0

        split = 2

        def block_diff_to_score(block_diff: torch.Tensor):
            return (
                block_diff.squeeze(0)
                .norm(dim=-1, keepdim=False)
                .sum(dim=-1, keepdim=False)
            )
            # return block_diff\
            #     .squeeze(0)\
            #     .abs().sum(dim=-1, keepdim=False)\
            #     .sum(dim=-1, keepdim=False)

        context_sparse_raw = context_sparse.clone()

        context_sparse, block_diff = perform_correction(
            context_sparse,
            context_sparse_raw,
            block_start_indices,
            w_size,
        )
        # [T,]
        block_diff_scores_parent, block_diff_indices = block_diff_to_score(
            block_diff
        ).topk(k=block_diff.shape[1] // split, dim=0, sorted=False)
        block_start_indices_parent = block_start_indices[block_diff_indices]
        block_start_indices_parent, tind = block_start_indices_parent.sort()
        block_diff_scores_parent = block_diff_scores_parent[tind]

        depth = 0
        max_iter = 4
        while (w_size // split) > 0 and (depth < max_iter):
            depth += 1
            block_start_indices_child = block_start_indices_parent + w_size // split
            w_size = w_size // split

            # if get_local_rank() == 0:
            #     print(block_diff_scores_parent)

            context_sparse, block_diff = perform_correction(
                context_sparse,
                context_sparse_raw,
                block_start_indices_child,
                w_size,
            )
            if (w_size // split) > 0:
                block_diff_scores_child = block_diff_to_score(block_diff)
                block_diff_scores_parent, next_blocks_location = torch.cat(
                    [block_diff_scores_parent, block_diff_scores_child]
                ).topk(k=block_diff_scores_parent.shape[0] // 2, sorted=False)
                block_start_indices_parent = torch.cat(
                    [block_start_indices_parent, block_start_indices_child]
                )[next_blocks_location]
                block_start_indices_parent, tind = block_start_indices_parent.sort()
                block_diff_scores_parent = block_diff_scores_parent[tind]

        # fill dense for first and last part
        dense_indices = torch.cat(
            [
                torch.arange(0, num_dense_first, device=query.device),
                torch.arange(
                    num_dense_first + num_sparse,
                    num_queries,
                    device=query.device,
                ),
            ]
        )
        context_sparse, _ = perform_correction(
            context_sparse,
            context_sparse_raw,
            dense_indices,
            1,
        )

        context = context_sparse
    else:
        num_queries = query.shape[1]
        num_last_dense = num_queries % delta_attention_args_w + max(
            128, delta_attention_args_w
        )
        num_last_dense = min(num_queries, num_last_dense)
        num_sparse = num_queries - num_last_dense

        context_sparse_raw = context_sparse
        context_sparse = context_sparse[:, :num_sparse]
        if delta_attention_args_adjust_norm_const:
            sparse_mx = sparse_mx[:, :num_sparse]
            sparse_nc = sparse_nc[:, :num_sparse]

        if num_last_dense > 0:
            idx = torch.arange(
                0,
                # delta_attention_args_w - 1,
                num_sparse,
                step=delta_attention_args_w,
                device=query.device,
            )
            rolling_idx = False
            if rolling_idx:
                idx = (idx + (args.layer_id % delta_attention_args_w)).clamp_max(
                    num_sparse - 1
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
            if delta_attention_args_adjust_norm_const:
                context_sparse_for_diff = context_sparse[:, idx]
                sparse_mx_for_diff = sparse_mx[:, idx]
                sparse_nc_for_diff = sparse_nc[:, idx]

            idx = torch.cat(
                (
                    idx,
                    torch.arange(num_sparse, num_queries, device=query.device),
                )
            )
            query_for_dense = query[:, idx]

        if (args.need_apply_rope and args.using_extend) and (
            delta_attention_args_extend == "none"
        ):
            assert delta_attention_args_extend == "none"
            # TODO: using paged attention
            repeated_k = args.gather_k_from_paged_cache(disable_gqa=True, gqa_q=query)
            repeated_v = args.gather_v_from_paged_cache(
                disable_gqa=True, gqa_q=query
            )  # B, T, H, D
            # assert repeated_k.shape[2] in (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64), repeated_k.shape
            assert repeated_k.shape[2] < 128

            seq_len = args.position_ids.amax().item() + 1
            repeated_k = repeated_k[:, :seq_len]
            repeated_v = repeated_v[:, :seq_len]

            query_for_recomp = query_for_dense

            cos = args.rope_cos
            sin = args.rope_sin
            assert cos.ndim == 2, cos.shape
            assert sin.shape == cos.shape, sin.shape

            cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
            sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])

            idx_tsrc = torch.arange(0, repeated_k.shape[1], device=cos.device)
            idx_tsrc.clamp_min_(seq_len - args.model_context_length)

            repeated_k = (
                (repeated_k.to(cos.dtype) * cos[:, idx_tsrc, :, :])
                + (rotate_half(repeated_k.to(sin.dtype)) * sin[:, idx_tsrc, :, :])
            ).to(repeated_k.dtype)

            query_for_recomp = (
                (query_for_recomp * cos[:, args.position_ids.view(-1)[idx], :, :])
                + (
                    rotate_half(query_for_recomp)
                    * sin[:, args.position_ids.view(-1)[idx], :, :]
                )
            ).to(query_for_recomp.dtype)

            assert args.position_ids.shape[0] == 1
            context_dense = (
                query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    repeated_k.permute(0, 2, 1, 3).contiguous(),
                    repeated_v.permute(0, 2, 1, 3).contiguous(),
                    # idx.unsqueeze(0),
                    args.position_ids[:, idx],
                    sm_scale,
                    None,
                    None,
                    None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        else:
            if args.need_apply_rope and args.using_extend:
                assert delta_attention_args_extend in ("self_extend",)

            if args.using_paged_cache:
                assert args.using_paged_cache

                query_for_recomp = query_for_dense
                k_cache = args.get_k_cache()
                v_cache = args.get_v_cache()

                assert args.position_ids.shape[0] == 1
                context_dense = query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    None,
                    None,
                    args.position_ids[:, idx],
                    sm_scale,
                    k_cache,
                    v_cache,
                    args.block_table,
                    return_running_statistics=delta_attention_args_adjust_norm_const,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    extend_backend=delta_attention_args_extend,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    model_context_length=args.model_context_length,
                    self_extend_scale=args.self_extend_scale,
                )
            else:
                assert k is not None
                assert v is not None
                context_dense = query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    k.permute(0, 2, 1, 3).contiguous(),
                    v.permute(0, 2, 1, 3).contiguous(),
                    args.position_ids[:, idx],
                    sm_scale,
                    None,
                    None,
                    None,
                    return_running_statistics=delta_attention_args_adjust_norm_const,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    extend_backend=delta_attention_args_extend,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    model_context_length=args.model_context_length,
                    self_extend_scale=args.self_extend_scale,
                )

            if delta_attention_args_adjust_norm_const:
                context_dense, (dense_mx, dense_nc) = context_dense
            else:
                dense_mx = dense_nc = None

            if dense_mx is not None:
                dense_mx = dense_mx.permute(0, 2, 1)
                dense_nc = dense_nc.permute(0, 2, 1)
            context_dense = context_dense.permute(0, 2, 1, 3).contiguous()

        if delta_attention_args_diff == 0:
            context = torch.zeros_like(query)
            context[:, :num_sparse] = context_sparse
            context[:, idx] = context_dense
        elif delta_attention_args_adjust_norm_const:
            idx_dense, idx_last = (idx[:-num_last_dense], idx[-num_last_dense:])
            context_dense, last_context_dense = (
                context_dense[:, :-num_last_dense],
                context_dense[:, -num_last_dense:],
            )

            dense_mx = dense_mx[:, :-num_last_dense]
            dense_nc = dense_nc[:, :-num_last_dense]

            # context_sparse_for_diff_norm = context_sparse_for_diff.float().square().sum(dim=-1, keepdim=True).sqrt()
            # context_dense_norm = context_dense.float().square().sum(dim=-1, keepdim=True).sqrt()
            # scale = context_dense_norm / context_sparse_for_diff_norm

            using_jeff = False
            if using_jeff or (not delta_attention_args_adjust_norm_const):
                # redo the normalization constant for the sparse outputs so we calculate the exact delta region
                # ------------------------------
                if delta_attention_args_adjust_norm_const:
                    # denorm, make alpha, scale, renorm so that the difference in the following block is the exact difference
                    # with the correct normalization constant.
                    context_sparse_for_diff = (
                        context_sparse_for_diff * sparse_nc_for_diff[:, :, :, None]
                    )
                    mx = torch.stack((dense_mx, sparse_mx_for_diff), dim=0).amax(dim=0)
                    alpha = torch.exp2(sparse_mx_for_diff - mx)
                    context_sparse_for_diff = (
                        context_sparse_for_diff * alpha[:, :, :, None]
                    )
                    sparse_nc_for_diff = sparse_nc_for_diff * alpha
                    context_sparse_for_diff = (
                        context_sparse_for_diff / sparse_nc_for_diff[:, :, :, None]
                    )
                # ------------------------------

                # take difference
                context_diff = context_dense - context_sparse_for_diff  # * scale

                context_diff = context_diff.repeat_interleave(
                    delta_attention_args_w, dim=1
                )

                if delta_attention_args_smooth:
                    # (exp) linear interpolate diff
                    context_diff_shift = torch.roll(
                        context_diff, -delta_attention_args_w, 1
                    )
                    context_diff_shift[:, -delta_attention_args_w:] = context_diff[
                        :, -1:
                    ]

                    offset = torch.arange(
                        0, context_diff.shape[1], device=context_diff.device
                    )
                    offset = (
                        offset % delta_attention_args_w
                    ).float() / delta_attention_args_w
                    context_diff = (
                        context_diff
                        + (context_diff_shift - context_diff)
                        * offset[None, :, None, None]
                    )

                # context_sparse_norm = context_sparse.float().square().sum(dim=-1, keepdim=True).sqrt()
                # scale = context_dense_norm.repeat_interleave(delta_attention_args_w, dim=1) / context_sparse_norm

                # ---------------------------------------------------
                # rescale context sparse to include the normalization constant from the delta region H = (T + H) - T
                if delta_attention_args_adjust_norm_const:
                    # get the 'head' normalization constant which is the normalization constant of the non-sparse indices.
                    h_nc = (
                        dense_nc - sparse_nc_for_diff
                    )  # sparse_nc already applied alpha
                    h_nc = h_nc.repeat_interleave(delta_attention_args_w, dim=1)

                    mx_repeat = mx.repeat_interleave(delta_attention_args_w, dim=1)

                    context_sparse = context_sparse * sparse_nc[:, :, :, None]
                    # mx = torch.stack((dense_mx_repeat, sparse_mx)).amax(dim=0)

                    alpha_for_sparse = torch.exp2(sparse_mx - mx_repeat)
                    context_sparse = context_sparse * alpha_for_sparse[:, :, :, None]
                    sparse_nc = sparse_nc * alpha_for_sparse

                    # mx = torch.stack((dense_mx_repeat, sparse_mx)).amax(dim=0)
                    # alpha_for_dense = torch.exp2(dense_mx_repeat - mx)
                    # h_nc = h_nc * alpha_for_dense

                    nc = h_nc + sparse_nc
                    context_sparse = context_sparse / nc[:, :, :, None]
                # ---------------------------------------------------

                # context = context_sparse * scale + context_diff
                context = context_sparse + context_diff
            else:
                scale = 1.0
                numerator = context_dense * dense_nc[:, :, :, None]
                denominator = dense_nc[:, :, :, None]

                # if get_local_rank() == 0:
                #     print('-')
                #     print('wrong mx (%)', (sparse_mx_for_diff > dense_mx).float().mean())
                #     print('avg error', ((sparse_mx_for_diff - dense_mx) * (sparse_mx_for_diff > dense_mx)).mean())
                #     print('max error', ((sparse_mx_for_diff - dense_mx) * (sparse_mx_for_diff > dense_mx)).amax())
                #     print('avg sparse mx', sparse_mx_for_diff.mean())

                t_mx = torch.maximum(sparse_mx_for_diff, dense_mx)

                alpha_sparse = torch.exp2(sparse_mx_for_diff - t_mx)[:, :, :, None]

                alpha_dense = torch.exp2(dense_mx - t_mx)[:, :, :, None]
                # alpha_dense = 1

                # this is the delta with denormalized numerator,
                # denominator is equal to H
                delta = numerator * alpha_dense - alpha_sparse * (
                    context_sparse_for_diff * sparse_nc_for_diff[:, :, :, None]
                )
                h_nc = (
                    denominator * alpha_dense
                    - alpha_sparse * sparse_nc_for_diff[:, :, :, None]
                )

                # if get_local_rank() == 0:
                #     print(denominator[0, :, 0])

                # context_diff = context_dense - context_sparse_for_diff
                # context_diff_norm = torch.norm(context_diff, dim=-1, keepdim=True)
                # context_diff_scale = context_diff_norm / context_diff_norm.amax(dim=1, keepdim=True)
                # scale *= context_diff_scale
                # delta *= scale
                # h_nc *= scale

                def _repeat_interleave(t: torch.Tensor):
                    t = t.repeat_interleave(delta_attention_args_w, dim=1)

                    if delta_attention_args_smooth:
                        # (exp) linear interpolate diff
                        t_shift = torch.roll(t, -delta_attention_args_w, 1)
                        t_shift[:, -delta_attention_args_w:] = t[:, -1:]

                        offset = torch.arange(0, t.shape[1], device=t.device)
                        offset = (
                            offset % delta_attention_args_w
                        ).float() / delta_attention_args_w
                        if t.ndim == 4:
                            t = t + (t_shift - t) * offset[None, :, None, None]
                        else:
                            t = t + (t_shift - t) * offset[None, :, None]
                    return t

                delta = _repeat_interleave(delta)
                h_nc = _repeat_interleave(h_nc)
                dense_mx = _repeat_interleave(
                    torch.maximum(sparse_mx_for_diff, dense_mx)
                )
                # sparse_mx = _repeat_interleave(sparse_mx_for_diff)
                # sparse_nc = _repeat_interleave(sparse_nc_for_diff)

                t_mx = torch.maximum(dense_mx, sparse_mx)
                alpha_sparse = torch.exp2(sparse_mx - t_mx)[:, :, :, None]

                alpha_dense = torch.exp2(dense_mx - t_mx)[:, :, :, None]

                # alpha_dense_mask = torch.zeros(dense_mx.size(1), device=dense_mx.device, dtype=torch.bool)
                # alpha_dense_mask = alpha_dense_mask.view(-1, delta_attention_args_w)
                # alpha_dense_mask[:, 0] = 1
                # alpha_dense_mask = alpha_dense_mask.view(-1)[None, :, None, None]
                # alpha_dense = alpha_dense_mask * 1 + ~alpha_dense_mask * alpha_dense_mask

                numerator = alpha_dense * delta + alpha_sparse * (
                    context_sparse * sparse_nc[:, :, :, None]
                )
                denominator = (
                    alpha_dense * h_nc + alpha_sparse * sparse_nc[:, :, :, None]
                )

                context = numerator / denominator

            context = context.to(query.dtype)
            context.index_copy_(dim=1, index=idx_dense, source=context_dense)
            context = torch.cat([context, last_context_dense], dim=1)

            # if get_local_rank() == 0:
            #     print(
            #         'hit', layer_id,
            #         context_diff.shape,
            #         context_sparse.shape,
            #         context_diff.abs().mean().item(),
            #         context_sparse.abs().mean().item()
            #     )
        else:
            from .delta.apply_delta import apply_delta

            if delta_attention_args_extend == "self_extend":
                # FIXME this is surely bug...
                last_context_sparse = context_sparse_raw[:, -2048:].clone()

            context = apply_delta(
                context_dense,
                context_sparse,
                idx,
                num_last_dense,
                delta_attention_args_w,
                delta_attention_args_smooth,
            )

            if delta_attention_args_extend == "self_extend":
                # FIXME this is surely bug...
                context[:, -2048:] = last_context_sparse

    return context, metadata


def _forward_fa3_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sm_scale: float,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
):
    assert q.ndim == 4

    # _cache_seqlens = (position_ids[:, -1] + 1).to(torch.int32)
    # _cu_seqlens_q = torch.arange(
    #     0, q.shape[0] + 1, q.shape[1], device=q.device, dtype=torch.int32
    # )
    # _cu_seqlens_k = _cu_seqlens_q.clone()
    # _cu_seqlens_k[1:] = cache_seqlens.cumsum(-1)

    # assert torch.all(_cache_seqlens == cache_seqlens)
    # assert torch.all(_cu_seqlens_q == cu_seqlens_q)
    # assert torch.all(_cu_seqlens_k == cu_seqlens_k)

    max_seqlen_q = q.shape[1]
    q_reshaped = q.view(-1, q.shape[-2], q.shape[-1])

    if k_cache.dtype == torch.float8_e5m2:
        raise Exception("fa3 does not support e5m2")
    elif k_cache.dtype == torch.float8_e4m3fn:
        q_reshaped = q_reshaped.to(k_cache.dtype)
    else:
        assert q_reshaped.dtype in (torch.float16, torch.bfloat16), k_cache.dtype

    # Default: single-token self-attention
    return flash_attn_with_kvcache(
        q=q_reshaped,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=sm_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0,
        k_descale=k_descale,
        v_descale=v_descale,
        return_softmax_lse=False,
    )


@capture
def _forward_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    position_ids: torch.Tensor,
    using_extend: bool,
    need_apply_rope: bool,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_is_neox_style: bool,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
):
    assert q.ndim == 4
    assert k.ndim == 4
    assert v.ndim == 4

    len_query_for_fa3 = q.shape[1]

    if (using_extend and need_apply_rope) and True:
        # FIXME do better infer method
        use_mla = triton.next_power_of_2(k.shape[-1]) != k.shape[-1]
        if use_mla:
            q = q.clone()
            k = k.clone()

            # FIXME assume DeepSeek
            rope_dim = k.shape[-1] - (triton.next_power_of_2(k.shape[-1]) // 2)
            assert rope_dim == rope_cos.shape[-1]

            from sglang.srt.layers.rotary_embedding import _rotate_gptj, _rotate_neox

            rotate_fn = _rotate_neox if rope_is_neox_style else _rotate_gptj

            query_rot = q[..., -rope_dim:]
            key_rot = k[..., -rope_dim:]

            assert position_ids.shape[0] == 1
            assert not rope_is_neox_style
            cos_q = rope_cos[
                None,
                position_ids[0, :len_query_for_fa3],
                None,
                : rope_dim // 2,
            ].repeat_interleave(2, -1)
            sin_q = rope_sin[
                None,
                position_ids[0, :len_query_for_fa3],
                None,
                : rope_dim // 2,
            ].repeat_interleave(2, -1)
            cos_k = rope_cos[
                None, : key_rot.shape[1], None, : rope_dim // 2
            ].repeat_interleave(2, -1)
            sin_k = rope_sin[
                None, : key_rot.shape[1], None, : rope_dim // 2
            ].repeat_interleave(2, -1)

            query_rot = query_rot * cos_q + rotate_fn(query_rot) * sin_q
            key_rot = key_rot * cos_k + rotate_fn(key_rot) * sin_k

            q[..., -rope_dim:] = query_rot
            k[..., -rope_dim:] = key_rot

            # qqq = query_fa3[0, -4096:, 0]
            # kkk = k_fa3[0, :, 0]
            # scores = qqq @ kkk.T
            # scores = torch.nn.functional.max_pool2d(scores[None, None, ...], kernel_size=31, stride=15, padding=15)[0,0]

            # plt.clf()
            # plt.title(f'{scores.shape=}')
            # plt.imshow(scores.float().cpu().numpy())
            # plt.savefig('./dummy_scores.png')

            # print(rope_dim, args.rope_cos.shape, args.rope_sin.shape, args.rope_is_neox_style)
        else:
            q = q.clone()
            k = k.clone()

            # FIXME assume GQA/MHA
            rope_dim = k.shape[-1]

            from sglang.srt.layers.rotary_embedding import _rotate_gptj, _rotate_neox

            rotate_fn = _rotate_neox if rope_is_neox_style else _rotate_gptj

            query_rot = q
            key_rot = k

            if rope_is_neox_style:
                cos_q = rope_cos[None, position_ids[0, :len_query_for_fa3], None, :]
                sin_q = rope_sin[None, position_ids[0, :len_query_for_fa3], None, :]
                cos_k = rope_cos[None, : key_rot.shape[1], None, :]
                sin_k = rope_sin[None, : key_rot.shape[1], None, :]
            else:
                assert position_ids.shape[0] == 1
                cos_q = rope_cos[
                    None,
                    position_ids[0, :len_query_for_fa3],
                    None,
                    : rope_dim // 2,
                ].repeat_interleave(2, -1)
                sin_q = rope_sin[
                    None,
                    position_ids[0, :len_query_for_fa3],
                    None,
                    : rope_dim // 2,
                ].repeat_interleave(2, -1)
                cos_k = rope_cos[
                    None, : key_rot.shape[1], None, : rope_dim // 2
                ].repeat_interleave(2, -1)
                sin_k = rope_sin[
                    None, : key_rot.shape[1], None, : rope_dim // 2
                ].repeat_interleave(2, -1)

            q = (
                query_rot.to(torch.float32) * cos_q.to(torch.float32)
                + rotate_fn(query_rot.to(torch.float32)) * sin_q.to(torch.float32)
            ).to(query_rot.dtype)
            k = (
                key_rot.to(torch.float32) * cos_k.to(torch.float32)
                + rotate_fn(key_rot.to(torch.float32)) * sin_k.to(torch.float32)
            ).to(key_rot.dtype)

    tp_q_head, tp_q_dim = q.shape[2:]
    tp_k_head, tp_k_dim = k.shape[2:]
    tp_v_head, tp_v_dim = v.shape[2:]

    # cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
    # max_seqlen_q = q_len

    # # Construct metadata for the Key/Value
    # # For a single sequence of length kv_len, cu_seqlens is just [0, kv_len]
    # cu_seqlens_k = torch.tensor([0, kv_len], dtype=torch.int32, device=k.device)
    # max_seqlen_k = kv_len

    cu_seqlens_q = torch.tensor([0, q.shape[1]], dtype=torch.int32, device=q.device)
    max_seqlen_q = q.shape[1]
    cu_seqlens_k = torch.zeros((2,), dtype=torch.int32, device=k.device)
    cu_seqlens_k[1] = position_ids[0, len_query_for_fa3 - 1] + 1
    max_seqlen_k = k.shape[1]

    # print(query_fa3.shape, k_fa3.shape, v_fa3.shape, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale)

    context_fa3 = flash_attn_varlen_func(
        q=q.view(-1, tp_q_head, tp_q_dim).contiguous(),
        k=k.view(-1, tp_k_head, tp_k_dim).contiguous(),
        v=v.view(-1, tp_v_head, tp_v_dim).contiguous(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=sm_scale,
        causal=True,
        return_softmax_lse=False,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    context_fa3 = context_fa3.view(q.shape[:-1] + (v.shape[-1],))

    return context_fa3


@capture
def _forward_partial_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    rope_is_neox_style: bool,
    cached_metadata: HiPAttentionOutputMetadata,
    is_decode: bool,
    seq_thresh_fa3: int,
    mixing_len: int,
    args: HiPAttentionArgs,
    max_context_len: int,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    inner_function_do_scale: bool,
    inner_function,
):
    query = q

    context_fa3 = None
    metadata = None

    if (not is_decode) and (seq_thresh_fa3 > 0):
        if args.using_paged_cache:
            pass
        else:
            assert k is not None
            max_context_len = min(max_context_len, k.shape[1])
        min_context_len = max(0, max_context_len - query.shape[1])

        len_query_for_fa3 = max(
            0, min(seq_thresh_fa3, max_context_len) - min_context_len
        )
        len_query_for_hip = max(
            0, max_context_len - max(min_context_len, seq_thresh_fa3 - mixing_len)
        )

        # print(max_context_len, min_context_len, seq_thresh_fa3, len_query_for_fa3, len_query_for_hip)

        if len_query_for_fa3 > 0:
            assert not is_decode
            # assert not args.using_extend, "todo"

            if args.using_paged_cache:
                k = args.gather_k_from_paged_cache(
                    seq_len=min(max_context_len, args.model_context_length)
                )
                v = args.gather_v_from_paged_cache(
                    seq_len=min(max_context_len, args.model_context_length)
                )

            query_fa3 = query[:, :len_query_for_fa3].contiguous()
            len_kv = k.shape[1] - (
                len_query_for_hip - (query.shape[1] - len_query_for_fa3)
            )  # BUG: this should be bug, because this will lose keys for len_for_mix
            k_fa3 = k[:, :len_kv].contiguous()
            v_fa3 = v[:, :len_kv].contiguous()

            is_fp8 = k.dtype in (torch.float8_e5m2,)
            if is_fp8:
                query_fa3 = query_fa3.to(torch.float16)
                k_fa3 = k_fa3.to(torch.float16)
                v_fa3 = v_fa3.to(torch.float16)

            if k.dtype == torch.float8_e4m3fn:
                query_fa3 = query_fa3.to(k.dtype)

            context_fa3 = _forward_fa3(
                q=query_fa3,
                k=k_fa3,
                v=v_fa3,
                sm_scale=sm_scale,
                position_ids=args.position_ids[:, :len_query_for_fa3],
                using_extend=args.using_extend,
                need_apply_rope=args.need_apply_rope,
                rope_cos=args.rope_cos,
                rope_sin=args.rope_sin,
                rope_is_neox_style=rope_is_neox_style,
                k_descale=k_descale,
                v_descale=v_descale,
            )

    if args.using_paged_cache:
        k = v = None

    if context_fa3 is not None:
        if len_query_for_hip > 0:
            args_sparse = args.clone()
            args_sparse.position_ids = args_sparse.position_ids[:, -len_query_for_hip:]
            if args_sparse.q_mask is not None:
                args_sparse.q_mask = args_sparse.q_mask[:, -len_query_for_hip:]
            if args_sparse.query_for_landmark is not None:
                args_sparse.query_for_landmark = args_sparse.query_for_landmark[
                    :, -len_query_for_hip:
                ]

            yarn_scale = float(os.getenv("HIP_DEBUG_YARN_SCALE_HINT", "1"))
            if yarn_scale > 1:
                assert int(yarn_scale) == yarn_scale
                yarn_scale = int(yarn_scale)
                args_sparse.rope_cos = args_sparse.rope_cos[::yarn_scale]
                args_sparse.rope_sin = args_sparse.rope_sin[::yarn_scale]

            context_sparse, metadata = inner_function(
                q=(
                    query[:, -len_query_for_hip:]
                    * (sm_scale if inner_function_do_scale else 1)
                ).to(query.dtype),
                k=k,
                v=v,
                args=args_sparse,
                cached_metadata=cached_metadata,
            )
            if context_sparse.ndim == 3:
                context_sparse = context_sparse.unsqueeze(0)
                assert context_fa3.shape[0] == 1

            # w = 512
            # wt = 16
            # t = context_sparse.shape[1]
            # if t > w:
            #     t_context = context_sparse[:, t % w:]
            #     t_context_mean = t_context.view(-1, t // w, w, t_context.shape[-2], t_context.shape[-1]).mean(2, keepdim=True)
            #     delta = (torch.repeat_interleave(t_context_mean, w//wt, 1) - t_context.view(-1, t // wt, wt, t_context.shape[-2], t_context.shape[-1])).mean(2)
            #     delta = torch.repeat_interleave(delta, wt, 1)
            #     t_context.add_(delta)

            len_for_mix = (len_query_for_hip + len_query_for_fa3) - query.shape[1]

            if len_for_mix > 0:
                context_fa3_mix = context_fa3[:, -len_for_mix:]
                context_sparse_mix = context_sparse[:, :len_for_mix]

                chunk_len = min(context_sparse_mix.shape[1], len_for_mix)
                offset = min_context_len - (seq_thresh_fa3 - mixing_len)
                scale_global = (
                    torch.arange(
                        offset,
                        offset + chunk_len,
                        device=query.device,
                        dtype=torch.float32,
                    )
                    / mixing_len
                )

                len_for_spike = min(chunk_len, 32)
                scale = torch.clamp_min(
                    (
                        torch.arange(
                            0, chunk_len, device=query.device, dtype=torch.float32
                        )
                        - (chunk_len - len_for_spike)
                    )
                    / len_for_spike,
                    0,
                )  # * (1 - (offset / mixing_len)) + (offset / mixing_len)

                # scale_spike = (
                #     (torch.arange(
                #         offset, offset + chunk_len, device=query.device, dtype=torch.float32
                #     ) % len_for_spike)
                #     / len_for_spike
                # )
                # scale = torch.maximum(scale, scale_spike)

                scale = torch.maximum(scale, scale_global)

                scale = scale[None, :, None, None]
                context_mix = (
                    context_sparse_mix * scale + context_fa3_mix * (1.0 - scale)
                ).to(context_fa3_mix.dtype)

                context = torch.cat(
                    [
                        context_fa3[:, :-len_for_mix],
                        context_mix,
                        context_sparse[:, len_for_mix:],
                    ],
                    dim=1,
                )
            else:
                context = torch.cat([context_fa3, context_sparse], dim=1)
        else:
            context = context_fa3
    else:
        # no fa3
        context, metadata = inner_function(
            q=(query * (sm_scale if inner_function_do_scale else 1)).to(query.dtype),
            k=k,
            v=v,
            args=args,
            cached_metadata=cached_metadata,
        )

    context = context.to(query.dtype)

    return context, metadata


@capture
def _forward_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: HiPAttentionArgs,
    sliding_window_size: int,
    sliding_window_sink: int,
):
    query = q
    bsa_fn = get_block_sparse_backend(query, args.disable_flashdecode)

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
    args.sink_token_size = sliding_window_sink
    args.sliding_window_size = (
        sliding_window_size if sliding_window_size is not None else 1024
    )
    args.sliding_window_indices = None

    BDST = triton.cdiv(TDST, args.block_size_q)
    BH = BSZ * HEAD

    indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=query.device)
    ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.zeros((BH, BDST, 2), dtype=torch.int64, device=query.device)

    context = bsa_fn(
        q=query,
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

    return context, None


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
    block_table: torch.Tensor,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    layer_id: int,
    logit_cap: float,
    orig_context_len: int,
    max_context_len: int,
    max_batch_context_len: int,
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
    sliding_window_sink: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    self_extend_scale: int = 12,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:
    global _CHECKOUT_COUNTER

    if query.ndim == 3:
        N, num_heads, hidden_dims = query.shape
        dst_seq_len = N // batch_size
    else:
        _bsz, dst_seq_len, num_heads, hidden_dims = query.shape
        assert _bsz == batch_size, f"{query.shape=} {batch_size}"
        N = _bsz * dst_seq_len

    is_dense = layer_id in hip_config.dense_layers
    if not is_decode:
        if len(hip_config.prefill_layers) == 2:
            layer_config = hip_config.prefill_layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.prefill_layers[layer_id]
    else:
        # assert dst_seq_len == 1
        if len(hip_config.layers) == 2:
            layer_config = hip_config.layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.layers[layer_id]

    query = query.view(batch_size, dst_seq_len, num_heads, hidden_dims)
    if query_for_mask is not None:
        query_for_mask = query_for_mask.view(batch_size, -1, num_heads, hidden_dims)

    if k_cache is not None:
        if v_cache.ndim == 4:
            N_PAGE, _, num_heads_kv, hidden_dims_v = v_cache.shape
        else:
            assert v_cache.ndim == 3
            N_PAGE, num_heads_kv, hidden_dims_v = v_cache.shape
        assert N_PAGE == k_cache.shape[0], f"{N_PAGE} != {k_cache.shape[0]}"

        k_cache = k_cache.view(N_PAGE, 1, num_heads_kv, k_cache.shape[-1])
        v_cache = v_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims_v)

    # FIXME: this operation is linear during decoding
    if block_table is None:
        if is_decode:
            raise Exception("this should not happened")
        block_table = req_to_tokens.index_select(dim=0, index=req_pool_indices)

    BLOCK_TABLE_BSZ, MODEL_SEQ_LEN = block_table.shape
    assert batch_size == BLOCK_TABLE_BSZ

    if k_descale is not None:
        assert k_descale.shape == (batch_size, num_heads_kv)
        assert v_descale.shape == (batch_size, num_heads_kv)

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

    # NOTE this is not needed when offload cache is not needed right..?
    require_cache_statistics = False
    if cached_metadata is None:
        require_cache_statistics = offload_cache is not None
    elif cached_metadata.indices is None:
        require_cache_statistics = offload_cache is not None
    elif os.getenv("HIP_DISABLE_COMPUTE_STATISTICS", "1") == "0":
        require_cache_statistics = offload_cache is not None

    if torch.cuda.is_current_stream_capturing():
        assert is_decode

    args = HiPAttentionArgs(
        # k_cache=(
        #     k_cache.view(torch.uint8)
        #     if isinstance(k_cache, torch.Tensor) and k_cache.dtype == torch.float8_e5m2
        #     else k_cache
        # ),
        k_cache=k_cache,
        # v_cache=(
        #     v_cache.view(torch.uint8)
        #     if isinstance(k_cache, torch.Tensor) and v_cache.dtype == torch.float8_e5m2
        #     else v_cache
        # ),
        v_cache=v_cache,
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
        is_decode=is_decode,
        landmark_stage_k=layer_config.landmark_stage_k,
        k_descale=k_descale,
        v_descale=v_descale,
        self_extend_scale=self_extend_scale,
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

    sliding_window_size_for_masking_step = (
        layer_config.sliding_window_size_for_masking_step
    )
    if (
        isinstance(sliding_window_size_for_masking_step, list)
        and (cached_metadata is not None)
        and (cached_metadata.indices is None)
    ):
        larger_sw_size = sliding_window_size_for_masking_step[
            (
                max(0, len(cached_metadata.stage_caches) - 1)
                if cached_metadata.stage_caches is not None
                else 0
            )
        ]
        args.bsa_sliding_window_size = larger_sw_size

    sliding_window_size = os.getenv("HIP_DEBUG_SLLM_WINDOW", sliding_window_size)
    if isinstance(sliding_window_size, str):
        sliding_window_size = int(sliding_window_size)
    sliding_window_sink = int(
        os.getenv("HIP_DEBUG_SLLM_SINK", max(0, sliding_window_sink))
    )
    if args.second_stage_k == 0:
        sliding_window_size = args.sliding_window_size
        sliding_window_sink = args.sink_token_size

    # Plan 1
    # TODO use flash attention under 100K

    # Plan 2
    # TODO use flash attention under 64K
    # TODO use sparse setting under 128K

    seq_thresh_fa3 = int(os.getenv("HIP_DEBUG_SEQ_THRESH_FA3", "0"))
    if seq_thresh_fa3 > args.model_context_length:
        warnings.warn(
            f"Requested FA3 replacement ({seq_thresh_fa3}) is larger than model context length ({args.model_context_length}). "
            "Consider increase YaRN or using other model. "
            "OR You can decrease HIP_DEBUG_SEQ_THRESH_FA3 up to context length, but it will degrade throughput."
        )
        seq_thresh_fa3 = args.model_context_length

    mixing_len = os.getenv(
        "HIP_DEBUG_FA3_MIXING_LEN", "sw" if seq_thresh_fa3 > 0 else "0"
    )
    if mixing_len.lower() == "sw":
        mixing_len = int(
            sliding_window_size * 1.5
            if isinstance(sliding_window_size, int) and (sliding_window_size > 0)
            else args.sliding_window_size * 1.5
        )
    else:
        mixing_len = int(mixing_len)

    if seq_thresh_fa3 == 0:
        mixing_len = 0

    if os.getenv("HIP_DEBUG_SEQ_THRESH_FA3_INF_DENSE", "0") == "1":
        if layer_id in hip_config.dense_layers:
            seq_thresh_fa3 = query.shape[1]

    # TODO: if delta norm is too high, then just recompute that whole block.
    # TODO: use partial densely decode. delta attention for decode
    # postfix_recompute_dense-window_[size:int]-diff_[1/0]-w_[size:int]
    # example: HIP_DELTA_ATTENTION_ARGS=window_0-diff_1-w_32-dense_decode-smooth
    delta_attention_args = os.getenv("HIP_DELTA_ATTENTION_ARGS", None)
    using_delta_attention = (delta_attention_args is not None) and (
        delta_attention_args != ""
    )

    delta_attention_args_smooth = False
    delta_attention_args_just_return = False
    delta_attention_args_window = 0
    delta_attention_args_diff = 1
    delta_attention_args_dense_decode = False
    delta_attention_args_w = 16
    delta_attention_args_exp = False
    delta_attention_args_exp_w = 2
    delta_attention_args_exp_window = 1024
    delta_attention_args_exp_sink = 128
    delta_attention_args_iter_corr = False
    delta_attention_args_adjust_norm_const = False
    delta_attention_args_extend = "none"

    if using_delta_attention:
        for word in delta_attention_args.split("-"):
            word = word.strip()
            if word == "smooth":
                delta_attention_args_smooth = True
            elif word == "exp":
                delta_attention_args_exp = True
            elif word == "JUST_RETURN":
                delta_attention_args_just_return = True
            elif word == "sparse_decode":
                delta_attention_args_dense_decode = False
            elif word == "dense_decode":
                delta_attention_args_dense_decode = True
            elif word == "recompute_dense":
                pass  # backward compat.
            elif word == "iter_corr":
                delta_attention_args_iter_corr = True
            elif word == "adjust_norm_const":
                delta_attention_args_adjust_norm_const = True
            elif word.startswith("extend"):
                extend_mode = word.split("_")[1]
                if extend_mode == "self":
                    delta_attention_args_extend = "self_extend"
                else:
                    raise Exception(extend_mode)
            elif word.startswith("window_"):
                delta_attention_args_window = int(word.split("_")[1])
            elif word.startswith("diff_"):
                delta_attention_args_diff = int(word.split("_")[1])
            elif word.startswith("w_"):
                delta_attention_args_w = int(word.split("_")[1])
            elif word.startswith("expw_"):
                delta_attention_args_exp_w = int(word.split("_")[1])
            elif word.startswith("expsink_"):
                delta_attention_args_exp_sink = int(word.split("_")[1])
            elif word.startswith("expwindow_"):
                delta_attention_args_exp_window = int(word.split("_")[1])
            else:
                warnings.warn(f"unknown delta args: {word}")

        # if layer_id in [0,1,2,3,4,5,8,11,14,17,20,23,26,29,30,33,36,39,41,42,43,44,45,46,47]:
        #     delta_attention_args_adjust_norm_const = False

        if (get_local_rank() == 0) and (not is_decode):
            info_msg = (
                f"Delta Attention is activated {delta_attention_args_window=} "
                f"{delta_attention_args_diff=} {delta_attention_args_w=} "
                f"{delta_attention_args_just_return=} "
                f"{delta_attention_args_smooth=} "
                f"{delta_attention_args_dense_decode=} "
                f"{delta_attention_args_exp=} "
                f"{delta_attention_args_exp_w=} "
                f"{delta_attention_args_adjust_norm_const=} "
                f"{delta_attention_args_extend=} "
            )
            warnings.warn(info_msg)

        # args.sa_extend_backend = "clamp"

    if isinstance(sliding_window_size, int) and (sliding_window_size > 0):

        if os.getenv("HIP_DEBUG_FORCE_CHUNKED_SW", "0") == "1":
            args.using_chunked_sliding_window = True

        def __forward_sliding_window_wrapper(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            args: HiPAttentionArgs,
            cached_metadata: HiPAttentionOutputMetadata,
        ):
            return _forward_sliding_window(
                q=q,
                k=k,
                v=v,
                args=args,
                sliding_window_size=sliding_window_size,
                sliding_window_sink=sliding_window_sink,
            )

        context, metadata = _forward_partial_fa3(
            q=query,
            k=k,
            v=v,
            sm_scale=sm_scale,
            rope_is_neox_style=rope_is_neox_style,
            cached_metadata=cached_metadata,
            is_decode=is_decode,
            seq_thresh_fa3=seq_thresh_fa3,
            mixing_len=mixing_len,
            args=args,
            max_context_len=max_batch_context_len,
            k_descale=k_descale,
            v_descale=v_descale,
            inner_function_do_scale=True,
            inner_function=__forward_sliding_window_wrapper,
        )
    elif using_delta_attention and (
        (not is_decode)  # or (is_decode and delta_attention_args_dense_decode)
    ):

        def __forward_delta_attn_wrapper(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            args: HiPAttentionArgs,
            cached_metadata: HiPAttentionOutputMetadata,
        ):
            return _forward_delta_attn(
                q,
                sm_scale,
                k,
                v,
                args=args,
                cached_metadata=cached_metadata,
                is_decode=is_decode,
                delta_attention_args_smooth=delta_attention_args_smooth,
                delta_attention_args_just_return=delta_attention_args_just_return,
                delta_attention_args_window=delta_attention_args_window,
                delta_attention_args_diff=delta_attention_args_diff,
                delta_attention_args_dense_decode=delta_attention_args_dense_decode,
                delta_attention_args_w=delta_attention_args_w,
                delta_attention_args_exp=delta_attention_args_exp,
                delta_attention_args_exp_w=delta_attention_args_exp_w,
                delta_attention_args_exp_window=delta_attention_args_exp_window,
                delta_attention_args_exp_sink=delta_attention_args_exp_sink,
                delta_attention_args_iter_corr=delta_attention_args_iter_corr,
                delta_attention_args_adjust_norm_const=delta_attention_args_adjust_norm_const,
                delta_attention_args_extend=delta_attention_args_extend,
                k_descale=k_descale,
                v_descale=v_descale,
                rope_cos=args.rope_cos,
                rope_sin=args.rope_sin,
            )

        context, metadata = _forward_partial_fa3(
            q=query,
            k=k,
            v=v,
            sm_scale=sm_scale,
            rope_is_neox_style=rope_is_neox_style,
            cached_metadata=cached_metadata,
            is_decode=is_decode,
            seq_thresh_fa3=seq_thresh_fa3,
            mixing_len=mixing_len,
            args=args,
            max_context_len=max_batch_context_len,
            k_descale=k_descale,
            v_descale=v_descale,
            inner_function_do_scale=False,
            inner_function=__forward_delta_attn_wrapper,
        )
    elif ((force_dense_decode or delta_attention_args_dense_decode) and is_decode) or (
        using_dense_prefill and (not is_decode)
    ):
        if not is_decode:
            assert not is_decode

            if args.using_paged_cache:
                k = args.gather_k_from_paged_cache(
                    seq_len=min(max_context_len, args.model_context_length)
                )
                v = args.gather_v_from_paged_cache(
                    seq_len=min(max_context_len, args.model_context_length)
                )

            query_fa3 = query.contiguous()
            len_kv = k.shape[1]
            k_fa3 = k[:, :len_kv].contiguous()
            v_fa3 = v[:, :len_kv].contiguous()

            is_fp8 = k.dtype in (torch.float8_e5m2,)
            if is_fp8:
                query_fa3 = query_fa3.to(torch.float16)
                k_fa3 = k_fa3.to(torch.float16)
                v_fa3 = v_fa3.to(torch.float16)

            if k.dtype == torch.float8_e4m3fn:
                query_fa3 = query_fa3.to(k.dtype)

            context = _forward_fa3(
                q=query_fa3,
                k=k_fa3,
                v=v_fa3,
                sm_scale=sm_scale,
                position_ids=args.position_ids[:, : query_fa3.shape[1]],
                using_extend=args.using_extend,
                need_apply_rope=args.need_apply_rope,
                rope_cos=args.rope_cos,
                rope_sin=args.rope_sin,
                rope_is_neox_style=rope_is_neox_style,
                k_descale=k_descale,
                v_descale=v_descale,
            )
            metadata = None
        else:
            assert not args.need_apply_rope
            assert not args.using_extend

            context = _forward_fa3_decode(
                q=query,
                k_cache=args.get_k_cache(),
                v_cache=args.get_v_cache(),
                sm_scale=sm_scale,
                page_table=args.block_table,
                position_ids=args.position_ids[:, : query.shape[1]],
                k_descale=k_descale,
                v_descale=v_descale,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
            )
            metadata = None
    elif is_decode or (query.shape[1] < (last_dense * 2)) or (last_dense <= 0):
        context, metadata = _forward_partial_fa3(
            q=query,
            k=k,
            v=v,
            sm_scale=sm_scale,
            rope_is_neox_style=rope_is_neox_style,
            cached_metadata=cached_metadata,
            is_decode=is_decode,
            seq_thresh_fa3=seq_thresh_fa3,
            mixing_len=mixing_len,
            args=args,
            max_context_len=max_batch_context_len,
            k_descale=k_descale,
            v_descale=v_descale,
            inner_function_do_scale=True,
            inner_function=dual_stage_quadratic_hip_attention,
        )
        # context = context[:, -query.shape[1] :, :, :].contiguous()
    else:
        assert not is_decode
        assert last_dense > 0
        assert query_for_mask is None
        position_ids = args.position_ids

        args_sparse = args.clone()
        args_sparse.position_ids = position_ids[:, :]
        context, metadata = dual_stage_quadratic_hip_attention(
            (query[:, :, :, :] * sm_scale).to(query.dtype),
            k,
            v,
            args=args_sparse,
            cached_metadata=cached_metadata,
        )
        context_sparse = context.to(query.dtype)

        args_dense = args.clone()
        args_dense.sliding_window_size = args_dense.model_context_length // 4
        args_dense.position_ids = position_ids[:, -last_dense:]
        args_dense.second_stage_k *= 2
        args_dense.sink_token_size *= 2
        if args_dense.q_mask is not None:
            args_dense.q_mask = args_dense.q_mask[:, -last_dense:, :, :]
        # print(
        #     query.shape,
        #     k.shape if k is not None else None,
        #     v.shape if v is not None else None,
        #     args_dense.sliding_window_size,
        #     args_dense.sink_token_size,
        #     args_dense.second_stage_k
        # )
        last_block = triton.cdiv(last_dense, args_dense.block_sparse_block_size_q)
        metadata.indices = metadata.indices[:, -last_block:]
        metadata.ks = metadata.ks[:, -last_block:]
        metadata.ks_count = metadata.ks_count[:, -last_block:]
        metadata.ks_start_end = metadata.ks_start_end[:, -last_block:]
        context_dense, metadata = dual_stage_quadratic_hip_attention(
            (query[:, -last_dense:, :, :] * sm_scale).to(query.dtype),
            k,
            v,
            args=args_dense,
            cached_metadata=metadata,
        )
        context_dense = context_dense.to(query.dtype)

        context = torch.cat([context_sparse[:, :-last_dense], context_dense], dim=1)
        context = context[:, -query.shape[1] :, :, :].contiguous()

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

    assert context.dtype == query.dtype
    return context.view(N, num_heads, context.shape[-1]), metadata, args


class PagedHiPStateful:
    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.states = dict()

        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        self.using_delta_decode = False

        self.delta_buffer = torch.zeros(
            num_layers,
            max_batch_size,
            1,
            num_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )

    def __call__(
        self,
        **kwargs,
    ):
        layer_id = kwargs.get("layer_id", None)
        is_decode = kwargs.get("is_decode", False)
        hip_config = kwargs.get("hip_config", None)  # type: HiPAttentionConfig

        # NOTE: init landmark states
        state = self.states.get(layer_id, None)

        cached_metadata = kwargs.pop("cached_metadata", None)
        if cached_metadata is None:
            cached_metadata = HiPAttentionOutputMetadata(
                indices=None,
                ks=None,
                ks_count=None,
                ks_start_end=None,
                mask_cache_statistics=None,
                sa_cache_statistics=None,
                stage_caches=None,
                state=None,
            )

        assert isinstance(cached_metadata, HiPAttentionOutputMetadata)
        cached_metadata.state = state

        # NOTE: forward paged-hip attention function (state-less)
        o, metadata, hip_args = forward_paged_hip(
            **kwargs,
            cached_metadata=cached_metadata,
        )

        # NOTE: handle landmark states
        if not is_decode:
            states = None
            if metadata is not None:
                if isinstance(metadata, list):
                    if (not any(map(lambda x: x is None, metadata))) and (
                        not any(map(lambda x: x.state is None, metadata))
                    ):
                        states = [m.state for m in metadata]
                else:
                    if metadata.state is not None:
                        states = metadata.state
            if states is not None:
                self.states[layer_id] = states

        # NOTE: handle delta decode
        if is_decode and self.using_delta_decode:
            assert hip_config is not None
            assert o.shape[0] < self.delta_buffer.shape[1]
            assert o.shape[1:] == (
                self.num_heads,
                self.head_dim,
            ), f"{o.shape} == {(self.num_heads, self.head_dim)}"
            assert layer_id >= 0 and layer_id < self.delta_buffer.shape[0]

            query = kwargs["query"]
            batch_size = kwargs["batch_size"]
            sm_scale = kwargs["sm_scale"]

            assert o.shape[0] == batch_size

            layer_config = hip_config.get_layer_config(layer_id, is_decode)
            delta_buffer = self.delta_buffer[layer_id, : o.shape[0]]

            # NOTE: if current step is mask refreshing, update delta
            # you have to set larger sliding window size for masking step, for delta
            require_update = (
                isinstance(layer_config.sliding_window_size_for_masking_step, list)
                and (cached_metadata is not None)
                and (cached_metadata.indices is None)
                and (metadata is not None)
                and (metadata.indices is not None)
            )
            if require_update:
                # k = kwargs.get('k', None)
                # v = kwargs.get('v', None)

                query = query.view(batch_size, 1, self.num_heads, self.head_dim)
                args = hip_args.clone()
                if args.rope_range is None:
                    args.rope_range = (0, query.shape[-1])
                bsa_fn = get_block_sparse_backend(query, False)
                args.block_size_k = layer_config.stages[-1].stage_chunk_size
                args.block_size_q = min(
                    args.block_sparse_block_size_q,
                    layer_config.stages[-1].stage_block_size_q,
                )

                # print(
                #     'hi',
                #     args.sliding_window_size,
                #     args.sink_token_size,
                #     layer_config.sliding_window_size_for_masking_step,
                #     args.block_size_q,
                #     args.block_size_k,
                #     metadata.indices.shape,
                #     args.position_ids.shape,
                #     query.shape,
                # )

                assert query.ndim == 4
                o_sparse = bsa_fn(
                    q=(query * sm_scale).to(query.dtype),
                    k=None,
                    v=None,
                    seq_lens=args.position_ids[:, -query.shape[1] :] + 1,
                    indices=metadata.indices,
                    ks=metadata.ks,
                    ks_count=metadata.ks_count,
                    ks_start_end=metadata.ks_start_end,
                    access_counter=None,
                    cache_miss_counter=None,
                    EXTEND_BACKEND=args.sa_extend_backend,
                    model_context_length=args.model_context_length,
                    extend_context_length=args.extend_context_length,
                    offload_update_cache=False,
                    args=args,
                )

                o_sparse = o_sparse.view(-1, self.num_heads, self.head_dim)
                assert o.ndim == 3

                delta = o.float() - o_sparse.float()
                delta = delta.view(batch_size, 1, self.num_heads, self.head_dim)
                self.delta_buffer.zero_()
                delta_buffer.copy_(delta)
                # NOTE: in require_update step, o is already dense output
            else:
                # NOTE: apply delta to output
                assert delta_buffer.shape[1] == 1
                o = (o + delta_buffer[:, 0, :, :]).to(o.dtype)

        return o, metadata
