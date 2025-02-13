import os
import torch
from typing import Optional, Any

from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionOutputMetadata
from hip.models.hip_attention.gen3.uvm_gpu_cache import HiPOffloadCache
from hip.models.hip_attention.gen3.hip_config import HiPAttentionConfig
from hip.models.hip_attention.gen3.attention_extend import (
    dual_stage_quadratic_hip_attention,
)
from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionArgs


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
    is_prefill: bool,
    hip_config: HiPAttentionConfig,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
    offloading_metadata: Any = None,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:

    if k is not None:
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
        if is_prefill:
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
        layer_id=layer_id,
        logit_cap=logit_cap,
        orig_context_len=orig_context_len,
        max_context_len=max_context_len,
        hip_config=hip_config,
        cached_metadata=cached_metadata,
        k=k,
        v=v,
        online_update_cache=online_update_cache,
    )

    if require_validation:
        if is_prefill:
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
                layer_id=layer_id,
                logit_cap=logit_cap,
                orig_context_len=orig_context_len,
                max_context_len=max_context_len,
                hip_config=hip_config,
                cached_metadata=cached_metadata,
                k=k,
                v=v,
                online_update_cache=online_update_cache,
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
                layer_id=layer_id,
                logit_cap=logit_cap,
                orig_context_len=orig_context_len,
                max_context_len=max_context_len,
                hip_config=hip_config,
                cached_metadata=cached_metadata,
                k=k,
                v=v,
                online_update_cache=online_update_cache,
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
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    hip_config=hip_config,
                    cached_metadata=cached_metadata,
                    k=k,
                    v=v,
                    online_update_cache=online_update_cache,
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
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    hip_config=hip_config,
                    cached_metadata=cached_metadata,
                    k=k,
                    v=v,
                    online_update_cache=online_update_cache,
                )
                err_uvm = sse(o, o_uvm)
                err_retry = sse(o_valid, o_retry)

                print(o)
                print(o_valid)
                print(metadata_new.indices)
                print(metadata_valid.indices)

                assert (
                    o_sse < err_thresh
                ), (
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
