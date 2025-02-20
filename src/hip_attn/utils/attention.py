import os
import warnings

import nvtx
import torch

from hip_attn.utils.attn_l1_loss import compute_attn_lp_loss_triton
from hip_attn.v1_0.attention1_block_gpu import flash_attention, hip_attention


@nvtx.annotate("custom_attention")
def custom_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    causal_mask,
    attention_dropout,
    # Attention method
    attention_method="hip",  # 'none', 'reformer', 'performer', 'hip'
    tree_reformer=None,
    tree_performer=None,
    # hip parameters
    tree_k=512,
    tree_block_size_q=64,
    tree_block_stride_q=2,
    tree_block_size_k=2,
    tree_block_stride_k=1,
    tree_dense_queries=0,
    tree_last_dense_queries=0,
    tree_sampling_method="center",
    # Latency optimization tweaks
    tree_enable_flash=True,
    tree_enable_sparq=False,
    tree_use_sliding_window=True,
    tree_sliding_window_size=int(os.getenv("HIP_DRAFT_SLIDING_WINDOW", "1024")),
    tree_sink_token_size=256,
    # Context averaging parameters
    tree_using_context_avg=False,
    tree_avgpool_scaler=None,
    last_cumsum=None,
    hidden_states=None,
    # RoPE parameters
    tree_rope_method="none",
    need_apply_rope=False,
    rope_cos=None,
    rope_sin=None,
    position_ids=None,
    self_extend_group_size=4,
    # Attention sparsity loss
    output_attn_sparsity_loss=False,
    tree_lp_norm_coeff=0.5,
    # Hyper attention state
    hyper_attention=None,
    sm_scaler=None,
    attn_logit_softcapping=0,
    model_sliding_window=None,
    model_context_length=131072,
    layer_idx=10,
    extend_stages=None,
    sliding_window_indices=None,
):
    """
    @param query_states: (N, H, TDST, HID)
    @param key_states: (N, H, TSRC, HID)
    @param value_states: (N, H, TSRC, HID)
    @param attention_mask: (N, 1, TDST, TSRC)
    @param causal_mask: (1, 1, TDST, TSRC)
    @param attention_dropout: Dropout probability
    @param attention_method: Attention method: ['none', 'reformer', 'performer', 'hip']
    @param tree_reformer: Optional. Reformer object
    @param tree_performer: Optional. Performer object
    @param tree_k: Number of tokens to attend to for each query token in hip attention
    @param tree_block_size_q: Query block size for hip attention
    @param tree_block_size_k: Key block size for hip attention
    @param tree_dense_queries: Number of dense queries
    @param tree_last_dense_queries: Number of last dense queries
    @param tree_sampling_method: Sampling method for hip attention: ['first', 'random']
    @param tree_enable_flash: Enable flash attention
    @param tree_enable_sparq: Enable SparQ attention
    @param tree_use_sliding_window: Use sliding window for hip attention
    @param tree_using_context_avg: Use context averaging for hip attention
    @param tree_avgpool_scaler: Average pooling scaler
    @param last_cumsum: Last cumsum for context averaging
    @param hidden_states: Hidden states for context averaging
    @param tree_rope_method: RoPE method: ['none', 'self_extend']
    @param rope_cos: Used in self-extend RoPE method
    @param rope_sin: Used in self-extend RoPE method
    @param position_ids: Position IDs for self-extend RoPE method
    @param output_attn_sparsity_loss: Whether to compute attention sparsity regularization
    @param tree_lp_norm_coeff: Lp norm coefficient for attention sparsity regularization
    @return: Attention output, last cumsum, attention sparsity loss
    """
    if sm_scaler is None:
        sm_scaler = 1 / (query_states.shape[-1] ** 0.5)
    attn_sparsity_loss = None
    if model_sliding_window is not None:
        assert isinstance(model_sliding_window, int)

    N, H, T, HID = query_states.shape
    _N, _H, _T, _HID = key_states.shape
    is_prompt = (N, T, HID) == (_N, _T, _HID)
    assert (H % _H) == 0
    H_KV = _H
    last_cumsum = attn_sparsity_loss = None

    if attention_method in ["none", "sdpa", "fa2"]:
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_method == "sdpa":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        from flash_attn import flash_attn_func, flash_attn_with_kvcache

        if is_prompt:
            if attention_method in ["none", "fa2"]:
                # assert causal_mask is None, causal_mask.shape
                if causal_mask is not None:
                    warnings.warn(
                        f"causal mask provided. this is useless {causal_mask.shape}"
                    )
                assert query_states.dtype in [
                    torch.float16,
                    torch.bfloat16,
                ], query_states.dtype
                assert key_states.dtype in [
                    torch.float16,
                    torch.bfloat16,
                ], key_states.dtype
                assert value_states.dtype in [
                    torch.float16,
                    torch.bfloat16,
                ], value_states.dtype
                attn_output = flash_attn_func(
                    q=query_states.permute(0, 2, 1, 3),
                    k=key_states.permute(0, 2, 1, 3),
                    v=value_states.permute(0, 2, 1, 3),
                    softmax_scale=sm_scaler,
                    causal=True,
                    softcap=attn_logit_softcapping,
                    window_size=(
                        (model_sliding_window, model_sliding_window)
                        if model_sliding_window is not None
                        else (-1, -1)
                    ),
                ).permute(0, 2, 1, 3)
            elif attention_method in ["spda"]:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=causal_mask,
                        is_causal=causal_mask is None,
                        dropout_p=attention_dropout,
                    )
            else:
                raise Exception()
        else:
            if attention_method in ["none", "fa2"]:
                attn_output = flash_attn_with_kvcache(
                    q=query_states.permute(0, 2, 1, 3),
                    k_cache=key_states.permute(0, 2, 1, 3),
                    v_cache=value_states.permute(0, 2, 1, 3),
                    softmax_scale=sm_scaler,
                    causal=True,
                    softcap=attn_logit_softcapping,
                    window_size=(
                        (model_sliding_window, model_sliding_window)
                        if model_sliding_window is not None
                        else (-1, -1)
                    ),
                ).permute(0, 2, 1, 3)
            elif attention_method in ["sdpa"]:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=causal_mask,
                        is_causal=causal_mask is None,
                        dropout_p=attention_dropout,
                    )

        if os.environ.get("CHECKOUT_STATES", "0") == "1":
            os.makedirs("./cache/llama/", exist_ok=True)
            torch.save(
                {
                    "q": query_states,
                    "k": key_states,
                    "v": value_states,
                    "out": attn_output,
                    "cos": rope_cos,
                    "sin": rope_sin,
                },
                "./cache/llama/qkvout.pth",
            )
            input("stored. press enter to continue >>> ")

    elif attention_method == "reformer":
        q = query_states  # / (query_states.shape[-1] ** 0.5)
        k = key_states
        v = value_states

        N, H, TDST, HID = q.shape
        _, _, TSRC, _ = k.shape
        assert k.shape == v.shape

        q = q.reshape(N * H, TDST, HID)  # .contiguous()
        # k = k.reshape(N*H, TSRC, HID) #.contiguous()
        v = v.reshape(N * H, TSRC, HID)  # .contiguous()

        tree_reformer.bucket_size = tree_k

        attn_output, attn, buckets = tree_reformer(q, v)  # (10, 1024, 128)
        attn_output = attn_output.view(N, H, TDST, HID)  # .to(hidden_states.dtype)

    elif attention_method == "performer":
        q = query_states  # / (query_states.shape[-1] ** 0.5)
        k = key_states
        v = value_states

        with torch.autocast("cuda", enabled=False):
            attn_output = tree_performer(
                q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            )
        attn_output = attn_output.to(q.dtype)

    elif attention_method == "dynamic_sparse_flash_attention":
        from hip_research.models.dynamic_sparse_flash_attention import (
            attention_fn as dynamic_sparse_flash,
        )

        q = query_states  # / (query_states.shape[-1] ** 0.5)
        k = key_states
        v = value_states

        attn_output = dynamic_sparse_flash(
            q[:, :, :, :128],
            k[:, :, :, :128],
            v[:, :, :, :128],
            nb_hash=16,
            attention_dropout=0,
        )

    elif (
        attention_method == "hip"
        or attention_method == "hip"
        or attention_method == "tree"
    ):
        q = query_states * sm_scaler
        k = key_states
        v = value_states

        N, H, TDST, HID = q.shape
        _, _, TSRC, _ = k.shape
        assert k.shape == v.shape

        # For L1 loss of attention map
        if output_attn_sparsity_loss:
            # select random `select_n` queries for speedup
            select_n = 1024
            selection = torch.randperm(TDST, device=q.device)[:select_n]
            attn_sparsity_loss = compute_attn_lp_loss_triton(
                q[..., selection, :],
                k,
                p=tree_lp_norm_coeff,
                attend_lengths=selection.expand(N, select_n),
            ).mean(-1)

        LAST_DENSE_QUERIES = tree_last_dense_queries

        if LAST_DENSE_QUERIES == 0:
            LAST_DENSE_QUERIES = None
        if isinstance(LAST_DENSE_QUERIES, int):
            assert LAST_DENSE_QUERIES < 0
            # prevent dense queries
        else:
            assert LAST_DENSE_QUERIES == None

        current_query_index = TSRC - TDST
        attn_outputs = []

        try:
            if os.getenv("HIP_LEGACY", "0") == "1":
                # maximum_ks = torch.where(
                #     torch.rand((q.shape[0], q.shape[1] // tree_block_size_q), device=q.device) < 0.5,
                #     512,
                #     128
                # ).to(torch.int32)

                q = q.reshape(N * H, TDST, HID)  # .contiguous()
                k = k.reshape(N * H_KV, TSRC, HID)  # .contiguous()
                v = v.reshape(N * H_KV, TSRC, HID)  # .contiguous()
                q_hip = q[:, :, :]

                attn_output_hip, _ = hip_attention(
                    q_hip,
                    k[:, :LAST_DENSE_QUERIES, :],
                    v[:, :LAST_DENSE_QUERIES, :],
                    mask_k=tree_k,
                    block_size_q=tree_block_size_q,
                    block_size_k=tree_block_size_k,
                    dense_queries_exp=0,  # NOTE DEBUG: tree_dense_queries,
                    rope_method=tree_rope_method,
                    rope_cos=rope_cos.squeeze(0) if rope_cos is not None else None,
                    rope_sin=rope_sin.squeeze(0) if rope_sin is not None else None,
                    position_ids=position_ids,
                    enable_sparq=False,  # NOTE DEUBG: tree_enable_sparq,
                    is_flash=True,  # NOTE DEUBG: tree_enable_flash,
                    using_sliding_window=True,  # NOTE DEBUG: tree_use_sliding_window,
                    sampling_method=tree_sampling_method,
                    # maximum_ks=maximum_ks,
                    # maximum_ks_config=[128, 512],
                    num_sink=16,
                )
            else:
                # from hip_attn import hip_attention_11, HiPAttentionArgs11
                from hip_attn.v1_1.attention2_draft_sampling import (
                    dual_stage_quadratic_hip_attention,
                    dual_stage_quadratic_scan_hip_attention,
                    dual_stage_sub_quadratic_hip_attention,
                    sampling_only_attention,
                )
                from hip_attn.v1_2.attention_extend import EvalScoreStage
                from hip_attn.v1_2.attention_extend import (
                    HiPAttentionArgs as HiPAttentionArgs12,
                )
                from hip_attn.v1_2.attention_extend import NopStage, ScanStage
                from hip_attn.v1_2.attention_extend import (
                    dual_stage_quadratic_hip_attention as dual_stage_quadratic_hip_attention_extend,
                )

                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)

                null = None
                true = True

                # extend_stages = [
                #     {
                #         "second_stage_k": 4096,
                #         "sliding_window_size": 1024,
                #         "sink_token_size": 256,
                #         "sa_extend_backend": "dynamic_extend",
                #         "stages": [
                #         {
                #             "stage_block_size_q": 128,
                #             "stage_block_stride_q": 4,
                #             "stage_chunk_size": 512,
                #             "stage_k": null,
                #             "stage_stride": 1,
                #             "require_realign_index": true,
                #             "require_reset_score": true,
                #             "require_post_sort": true,
                #             "stage_extend_backend": "streaming"
                #         },
                #         {
                #             "stage_block_size_q": 64,
                #             "stage_block_stride_q": 4,
                #             "stage_chunk_size": 32,
                #             "stage_k": 16384,
                #             "stage_stride": 1,
                #             "require_realign_index": true,
                #             "require_reset_score": true,
                #             "require_post_sort": true,
                #             "stage_extend_backend": null
                #         }
                #         ]
                #     },
                #     {
                #         "second_stage_k": 16384,
                #         "sliding_window_size": 2048,
                #         "sink_token_size": 256,
                #         "sa_extend_backend": "dynamic_extend",
                #         "stages": [
                #         {
                #             "stage_block_size_q": 256,
                #             "stage_block_stride_q": 4,
                #             "stage_chunk_size": 256,
                #             "stage_k": null,
                #             "stage_stride": 1,
                #             "require_realign_index": true,
                #             "require_reset_score": true,
                #             "require_post_sort": true,
                #             "stage_extend_backend": "streaming"
                #         },
                #         {
                #             "stage_block_size_q": 256,
                #             "stage_block_stride_q": 2,
                #             "stage_chunk_size": 16,
                #             "stage_k": 32768,
                #             "stage_stride": 1,
                #             "require_realign_index": true,
                #             "require_reset_score": true,
                #             "require_post_sort": true,
                #             "stage_extend_backend": null
                #         }
                #         ]
                #     }
                # ][0 if layer_idx < 4 else 1]

                if extend_stages and isinstance(extend_stages["stages"][0], dict):
                    for i in range(len(extend_stages["stages"])):
                        extend_stages["stages"][i] = ScanStage(
                            **extend_stages["stages"][i]
                        )

                # if q.shape == k.shape:
                #     q_quant = q.to(torch.float8_e5m2).view(torch.uint8)#[...,::2]
                #     k_quant = k.to(torch.float8_e5m2).view(torch.uint8)#[...,::2]
                # else:
                q_quant = q
                k_quant = k

                # attn_output_hip = dual_stage_quadratic_scan_hip_attention(
                #     q, k, v,
                #     scan_chunk_size=512,
                #     scan_k=32768,
                #     args=HiPAttentionArgs11(
                #         mask_k=512,
                #         block_size_q=64,
                #         block_stride_q=2,
                #         block_size_k=2, # BLOCK_CHUNK
                #         block_stride_k=1,
                #         sliding_window_size=256,
                #         sink_token_size=256,
                #         position_ids=position_ids,
                #     )
                # )

                # print(rope_cos.shape, rope_sin.shape, rope_cos.dtype, rope_sin.dtype, need_apply_rope)
                IS_GEMMA = os.getenv("IS_GEMMA", "0") == "1"
                is_dense = layer_idx < 3
                mask_only = False
                is_decode = TDST == 1
                cos = rope_cos.squeeze(0) if rope_cos is not None else None
                sin = rope_sin.squeeze(0) if rope_sin is not None else None
                block_size = 64
                HiPAttentionArgs = HiPAttentionArgs12

                k_mask = k
                k_group_size = int(os.getenv("K_GROUP_SIZE", "1"))
                _N, _T, _H, _D = k.shape
                tk = k_mask.view(_N, _T // k_group_size, k_group_size, _H, _D)
                k_mask = (
                    (
                        tk.min(dim=2, keepdim=True).values
                        + tk.max(dim=2, keepdim=True).values
                    )
                    .expand(_N, _T // k_group_size, k_group_size, _H, _D)
                    .contiguous()
                    .view(*k.shape)
                )

                dual_stage_kwargs = dict(
                    q=q,
                    k=k,
                    v=v,
                    args=HiPAttentionArgs(
                        # mask_k=256,
                        # block_size_q=64,
                        # block_stride_q=4,
                        block_size_k=64,  # BLOCK_CHUNK
                        # block_stride_k=k_group_size,
                        sliding_window_size=(
                            1024
                            if extend_stages is None
                            else extend_stages["sliding_window_size"]
                        ),
                        sink_token_size=(
                            256
                            if extend_stages is None
                            else extend_stages["sink_token_size"]
                        ),
                        # position_ids=position_ids,
                        using_extend=True,
                        rope_cos=cos,
                        rope_sin=sin,
                        need_apply_rope=True,
                        second_stage_k=(
                            2 * 1024
                            if extend_stages is None
                            else extend_stages["second_stage_k"]
                        ),
                        # low_percent = 0.75 if layer_idx >= 3 else 0.25,
                        # low_k_ratio = 0.25,
                        stages=(
                            [
                                ScanStage(
                                    stage_block_size_q=64,
                                    stage_block_stride_q=4,
                                    stage_chunk_size=128,
                                    stage_k=None,
                                    stage_stride=1,
                                ),
                                ScanStage(
                                    stage_block_size_q=64,
                                    stage_block_stride_q=4,
                                    stage_chunk_size=32,
                                    stage_k=32768,
                                    stage_stride=1,
                                ),
                                ScanStage(
                                    stage_block_size_q=64,
                                    stage_block_stride_q=1,
                                    stage_chunk_size=8,
                                    stage_k=8192,
                                    stage_stride=1,
                                ),
                            ]
                            if extend_stages is None
                            else extend_stages["stages"]
                        ),
                        block_sparse_block_size_q=block_size,
                        model_context_length=model_context_length,
                        scan_extend_backend=(
                            "streaming" if layer_idx < 3 else "relative"
                        ),
                        # scan_extend_backend='relative',
                        sa_extend_backend=(
                            "streaming"
                            if extend_stages is None
                            else extend_stages["sa_extend_backend"]
                        ),
                        stage_early_terminate=k_group_size,
                        mask_only=mask_only,
                        # q_mask=q,
                        # k_mask=k_mask,
                        require_stage_caches=False,
                        require_cache_statistics=False,
                        sliding_window_indices=sliding_window_indices,
                    ),
                )

                attn_output_hip, metadata = dual_stage_quadratic_hip_attention_extend(
                    **dual_stage_kwargs,
                )

                # attn_output_hip = sampling_only_attention(
                #     q, k, v,
                #     args=HiPAttentionArgs11(
                #         mask_k=512,
                #         block_size_q=64,
                #         block_stride_q=2,
                #         block_size_k=64,
                #         sliding_window_size=512,
                #         sink_token_size=512,
                #     ),
                # )

                # first_stage_k = 64
                # second_stage_k = 512

                # attn_output_hip = dual_stage_sub_quadratic_hip_attention(
                #     q, k, v,
                #     first_stage_args=HiPAttentionArgs11(
                #         mask_k=first_stage_k,
                #         block_size_q=64,
                #         block_stride_q=2,
                #         block_size_k=64,
                #         sliding_window_size=512,
                #         sink_token_size=512,
                #     ),
                #     second_stage_init_chunk=first_stage_k // 8,
                #     second_stage_init_k=1024,
                #     second_stage_args=HiPAttentionArgs11(
                #         mask_k=second_stage_k,
                #         block_size_q=64,
                #         block_stride_q=2,
                #         block_size_k=second_stage_k // first_stage_k,
                #         block_stride_k=4,
                #         sliding_window_size=512,
                #         sink_token_size=512,
                #     )
                # )

                # attn_output_hip, _ = hip_attention_11(
                #     q, k, v,
                #     args=HiPAttentionArgs11(
                #         position_ids=position_ids + 1,
                #         mask_k=tree_k,

                #         block_size_q=tree_block_size_q,
                #         block_stride_q=tree_block_stride_q,
                #         block_size_k=tree_block_size_k,
                #         block_stride_k=tree_block_stride_k,
                #         # block_stride_k=1,
                #         block_size_k_group=1,

                #         sliding_window_size=int(os.getenv('HIP_DRAFT_SLIDING_WINDOW', f'{tree_sliding_window_size}')),
                #         sink_token_size=int(os.getenv('HIP_DRAFT_SINK_TOKEN_SIZE', f'{tree_sink_token_size}')),

                #         using_extend=tree_rope_method == 'self_extend',
                #         rope_cos=rope_cos.squeeze(0) if rope_cos is not None else None,
                #         rope_sin=rope_sin.squeeze(0) if rope_sin is not None else None,
                #         self_extend_neighboor_window=1024,
                #         self_extend_group_size=self_extend_group_size,

                #         topk_head_group_size=1,
                #         sample_method=os.getenv('HIP_DRAFT_SAMPLING_METHOD', 'center'),
                #         branch_method=os.getenv('HIP_DRAFT_BRANCH_METHOD', 'half'),

                #         # this may good or not, but definatly great with self-extend
                #         traverse_from_last_step=False,
                #         step_size=None,
                #         num_samples=1,
                #         # NOTE: this is significant when topk_head_group_size > 1. otherwise, this make worse result
                #         chunk_size=None,
                #         # BUG: union has bug now...
                #         num_unions=1,

                #         score_head_group_size=1,

                #         using_sparq=False,
                #         sparq_hid=32,

                #         low_res_sample_scale=1,
                #         low_res_oversample_rate=1,
                #         low_res_oversample_block_stride_k=max(1, tree_block_size_k // 2) * 4,

                #         q_quant=q_quant,
                #         k_quant=k_quant,

                #         logit_softcap=attn_logit_softcapping,
                #     )
                # )

                attn_output_hip = attn_output_hip.permute(0, 2, 1, 3)  # .contiguous()
        except RuntimeError as ex:
            os.makedirs("cache/hip", exist_ok=True)
            torch.save(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "mask_k": tree_k,
                    "block_size_q": tree_block_size_q,
                    "block_size_k": tree_block_size_k,
                },
                "cache/hip/qkv.pth",
            )
            raise Exception("oops hip is dead, check cache/hip/qkv.pth") from ex

        # NOTE: accumulation should be done with fp32
        if tree_using_context_avg:

            if last_cumsum is None:
                last_cumsum = v.cumsum(-2, dtype=torch.float32)
                last_cumsum = last_cumsum[:, TSRC - TDST : LAST_DENSE_QUERIES, :]
            else:
                last_cumsum = last_cumsum.flatten(0, 1)
                curr_v = v[:, -q_hip.shape[-2] : LAST_DENSE_QUERIES, :]
                curr_v = curr_v.cumsum(-2, dtype=torch.float32)
                last_cumsum = curr_v + last_cumsum[:, -1:, :]

            context_avg = (
                last_cumsum
                / torch.arange(
                    current_query_index + 1,
                    current_query_index + 1 + q_hip.shape[1],
                    device=v.device,
                )[None, :, None]
            )
            context_avg = context_avg.to(v.dtype)

            last_cumsum = last_cumsum.unflatten(0, (N, H))

            # N, H, TDST
            scale_avg = (
                torch.sigmoid(
                    tree_avgpool_scaler(hidden_states[:, :LAST_DENSE_QUERIES, :])
                    .transpose(-1, -2)
                    .reshape(N * H, -1, 1)
                )
                * 0.25
                * torch.clamp(
                    1.0
                    - (
                        tree_k
                        / torch.arange(
                            TSRC - TDST, TSRC - TDST + q_hip.shape[1], device=v.device
                        )
                    ),
                    0.0,
                    1.0,
                )[None, :, None].to(v.dtype)
            )
            # NOTE: 0.25 is just heuristic
            # NOTE: 256 is top-k value
            attn_output_hip = (
                attn_output_hip * (1 - scale_avg) + context_avg * scale_avg
            ).to(v.dtype)
        attn_outputs.append(attn_output_hip)

        if LAST_DENSE_QUERIES is not None:
            flash_attention_mask = torch.zeros(
                (N * H, abs(LAST_DENSE_QUERIES), TSRC), dtype=q.dtype, device=q.device
            )
            attn_output_last_flash, _ = flash_attention(
                q[:, LAST_DENSE_QUERIES:, :],
                k[:, :, :],
                v[:, :, :],
                flash_attention_mask,
            )
            attn_outputs.append(attn_output_last_flash)

        if len(attn_outputs) > 1:
            attn_output = torch.cat(attn_outputs, dim=-2)
        else:
            attn_output = attn_outputs[0]

        attn_output = attn_output.view(N, H, TDST, HID)  # .to(hidden_states.dtype)

    elif attention_method == "streaming_llm":
        from hip_research.models.sink_attention import sink_attention

        q = query_states  # / (query_states.shape[-1] ** 0.5)
        if sm_scaler is not None:
            q = q * (query_states.shape[-1] ** 0.5) * sm_scaler
        k = key_states
        v = value_states

        N, H, TDST, HID = q.shape
        _, _, TSRC, _ = k.shape
        assert k.shape == v.shape

        q = q.reshape(N * H, TDST, HID)  # .contiguous()
        k = k.reshape(N * H, TSRC, HID)  # .contiguous()
        v = v.reshape(N * H, TSRC, HID)  # .contiguous()

        attn_output = sink_attention(
            q,
            k,
            v,
            rope_cos.squeeze(0),
            rope_sin.squeeze(0),
            num_sink=4,
            window_size=tree_k,
        )

        attn_output = attn_output.view(N, H, TDST, HID)  # .to(hidden_states.dtype)

    elif attention_method == "hyper_attention":
        q = query_states / (query_states.shape[-1] ** 0.5)
        k = key_states
        v = value_states

        N, H, TDST, HID = q.shape
        _, _, TSRC, _ = k.shape
        assert k.shape == v.shape

        # q = q.view(N*H, TDST, HID)
        # k = k.view(N*H, TSRC, HID)
        # v = v.view(N*H, TSRC, HID)

        attn_output = hyper_attention(q, k, v, causal=True, scale=1.0)

    elif attention_method == "skewed":
        from hip_research.models.skewed_attention import skewed_attention

        attn_output = skewed_attention(
            query_states,
            key_states,
            value_states,
            rope_cos,
            rope_sin,
            sm_scaler,
            layer_idx,
        )

    else:
        raise Exception(attention_method)

    return attn_output, last_cumsum, attn_sparsity_loss
