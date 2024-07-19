import os
import torch

from hip.models.hip_attention.attention1_gpu import flash_attention
from hip.models.hip_attention.attention1_block_gpu import hip_attention
from hip.models.attn_l1_loss import compute_attn_lp_loss_triton


def custom_attention(
    query_states, key_states, value_states,
    attention_mask, causal_mask,
    attention_dropout,

    # Attention method
    attention_method='hip',  # 'none', 'reformer', 'performer', 'hip'
    tree_reformer=None, 
    tree_performer=None,

    # hip parameters
    tree_k=512, 
    tree_block_size_q=32, 
    tree_block_size_k=2,
    tree_dense_queries=0, 
    tree_last_dense_queries=0,
    tree_sampling_method='first',

    # Latency optimization tweaks
    tree_enable_flash=False, 
    tree_enable_sparq=False, 
    tree_use_sliding_window=True,

    # Context averaging parameters
    tree_using_context_avg=False, 
    tree_avgpool_scaler=None, 
    last_cumsum=None, 
    hidden_states=None,

    # RoPE parameters
    tree_rope_method='none', 
    rope_cos=None, 
    rope_sin=None, 
    position_ids=None,

    # Attention sparsity loss
    output_attn_sparsity_loss=False, 
    tree_lp_norm_coeff=0.5,
    
    # Hyper attention state
    hyper_attention=None,
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
    attn_sparsity_loss = None

    if attention_method in ['none', 'sdpa', 'fa2']:
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
        
        if query_states.shape == key_states.shape:
            if attention_method in ['none', 'fa2']:
                assert causal_mask is None
                attn_output = flash_attn_func(
                    q=query_states.permute(0, 2, 1, 3),
                    k=key_states.permute(0, 2, 1, 3),
                    v=value_states.permute(0, 2, 1, 3),
                    softmax_scale=None,
                    causal=True,
                ).permute(0, 2, 1, 3)
            elif attention_method in ['spda']:
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

        if os.environ.get('CHECKOUT_STATES', '0') == '1':
            os.makedirs('./cache/llama/', exist_ok=True)
            torch.save({
                'q': query_states,
                'k': key_states,
                'v': value_states,
                'out': attn_output,
                'cos': rope_cos,
                'sin': rope_sin,
            }, './cache/llama/qkvout.pth')
            input('stored. press enter to continue >>> ')

    elif attention_method == 'reformer':
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

    elif attention_method == 'performer':
        q = query_states  # / (query_states.shape[-1] ** 0.5)
        k = key_states
        v = value_states

        with torch.autocast('cuda', enabled=False):
            attn_output = tree_performer(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))
        attn_output = attn_output.to(q.dtype)

    elif attention_method == 'hip' or attention_method == 'hip' or attention_method == 'tree':
        q = query_states / (query_states.shape[-1] ** 0.5)
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
                q[..., selection, :], k,
                p=tree_lp_norm_coeff,
                attend_lengths=selection.expand(N, select_n)
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
            if os.getenv('HIP_LEGACY', '0') == '1':
                # maximum_ks = torch.where(
                #     torch.rand((q.shape[0], q.shape[1] // tree_block_size_q), device=q.device) < 0.5,
                #     512,
                #     128
                # ).to(torch.int32)
                
                q = q.reshape(N * H, TDST, HID)  # .contiguous()
                k = k.reshape(N * H, TSRC, HID)  # .contiguous()
                v = v.reshape(N * H, TSRC, HID)  # .contiguous()
                q_hip = q[:, :, :]
                
                attn_output_hip, _ = hip_attention(
                    q_hip,
                    k[:, :LAST_DENSE_QUERIES, :],
                    v[:, :LAST_DENSE_QUERIES, :],
                    mask_k=tree_k,
                    block_size_q=tree_block_size_q,
                    block_size_k=tree_block_size_k,
                    dense_queries_exp=0, #NOTE DEBUG: tree_dense_queries,
                    rope_method=tree_rope_method,
                    rope_cos=rope_cos.squeeze(0) if rope_cos is not None else None,
                    rope_sin=rope_sin.squeeze(0) if rope_sin is not None else None,
                    position_ids=position_ids,
                    enable_sparq=False, #NOTE DEUBG: tree_enable_sparq,
                    is_flash=True, #NOTE DEUBG: tree_enable_flash,
                    using_sliding_window=True, #NOTE DEBUG: tree_use_sliding_window,
                    sampling_method=tree_sampling_method,
                    # maximum_ks=maximum_ks,
                    # maximum_ks_config=[128, 512],
                    num_sink=16,
                )
            else:
                # from hip.models.hip_attention.attention2_draft_causal_batch import hip_attention as hip_attention_draft_cpu
                # from hip.models.hip_attention.attention2_draft_causal_batch_gpu import hip_attention as hip_attention_draft
                # from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused import hip_attention as hip_attention_draft
                from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec import hip_attention as hip_attention_draft
                
                # attn_output_hip, _ = hip_attention_draft_cpu(
                #     q_hip,
                #     k[:, :LAST_DENSE_QUERIES, :],
                #     v[:, :LAST_DENSE_QUERIES, :],
                    
                #     mask_k=tree_k,
                    
                #     block_size_q=tree_block_size_q,
                #     block_size_k=tree_block_size_k,
                #     block_size_k_group=1,
                    
                #     using_extend=True,
                #     rope_cos=rope_cos.squeeze(0) if rope_cos is not None else None,
                #     rope_sin=rope_sin.squeeze(0) if rope_sin is not None else None,
                #     self_extend_neighboor_window=1024,
                #     self_extend_group_size=8,
                    
                #     topk_head_group_size=1,
                # )
                
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                
                # q = q.reshape(N * H, TDST, HID)
                # k = k.reshape(N * H, TSRC, HID)
                # v = v.reshape(N * H, TSRC, HID)
                
                # print(q.shape, k.shape, v.shape)
                
                attn_output_hip, _ = hip_attention_draft(
                    q, k, v,
                    
                    mask_k=tree_k,
                    
                    block_size_q=tree_block_size_q,
                    block_stride_q=2,
                    block_size_k=tree_block_size_k,
                    block_stride_k=2,
                    block_size_k_group=1,
                    
                    sliding_window_size=256,
                    sink_token_size=32,
                    
                    using_extend=False,
                    rope_cos=rope_cos.squeeze(0) if rope_cos is not None else None,
                    rope_sin=rope_sin.squeeze(0) if rope_sin is not None else None,
                    self_extend_neighboor_window=1024,
                    self_extend_group_size=4,
                    
                    topk_head_group_size=1,
                    sample_method='first',
                    branch_method='half',
                    
                    # this may good or not, but definatly great with self-extend
                    traverse_from_last_step=False,
                    step_size=q.shape[1] // tree_block_size_q,
                    num_samples=1,
                    # NOTE: this is significant when topk_head_group_size > 1. otherwise, this make worse result
                    chunk_size=None,
                    num_unions=1,
                    
                    score_head_group_size=1,
                    
                    using_sparq=False,
                    sparq_hid=32,
                    
                    low_res_sample_scale=4,
                    low_res_oversample_rate=2,
                    low_res_oversample_block_stride_k=4,
                )
                attn_output_hip = attn_output_hip.permute(0, 2, 1, 3).contiguous()
        except RuntimeError as ex:
            os.makedirs('cache/hip', exist_ok=True)
            torch.save({
                'q': q_hip,
                'k': k[:, :LAST_DENSE_QUERIES, :],
                'v': v[:, :LAST_DENSE_QUERIES, :],
                'mask_k': tree_k,
                'block_size_q': tree_block_size_q,
                'block_size_k': tree_block_size_k,
            }, 'cache/hip/qkv.pth')
            raise Exception('oops hip is dead, check cache/hip/qkv.pth') from ex

        # NOTE: accumulation should be done with fp32
        if tree_using_context_avg:

            if last_cumsum is None:
                last_cumsum = v.cumsum(-2, dtype=torch.float32)
                last_cumsum = last_cumsum[:, TSRC - TDST:LAST_DENSE_QUERIES, :]
            else:
                last_cumsum = last_cumsum.flatten(0, 1)
                curr_v = v[:, -q_hip.shape[-2]:LAST_DENSE_QUERIES, :]
                curr_v = curr_v.cumsum(-2, dtype=torch.float32)
                last_cumsum = curr_v + last_cumsum[:, -1:, :]

            context_avg = last_cumsum / torch.arange(
                current_query_index + 1,
                current_query_index + 1 + q_hip.shape[1],
                device=v.device
            )[None, :, None]
            context_avg = context_avg.to(v.dtype)

            last_cumsum = last_cumsum.unflatten(0, (N, H))

            # N, H, TDST
            scale_avg = torch.sigmoid(
                tree_avgpool_scaler(hidden_states[:, :LAST_DENSE_QUERIES, :]).transpose(-1, -2).reshape(N * H, -1, 1)
            ) * 0.25 * torch.clamp(1.0 - (tree_k / torch.arange(TSRC - TDST, TSRC - TDST + q_hip.shape[1], device=v.device)), 0.0, 1.0)[None, :, None].to(v.dtype)
            # NOTE: 0.25 is just heuristic
            # NOTE: 256 is top-k value
            attn_output_hip = (attn_output_hip * (1 - scale_avg) + context_avg * scale_avg).to(v.dtype)
        attn_outputs.append(attn_output_hip)

        if LAST_DENSE_QUERIES is not None:
            flash_attention_mask = torch.zeros((N * H, abs(LAST_DENSE_QUERIES), TSRC), dtype=q.dtype,
                                               device=q.device)
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

    elif attention_method == 'streaming_llm':
        from hip.models.sink_attention.sink_attention import sink_attention
        
        q = query_states # / (query_states.shape[-1] ** 0.5)
        k = key_states
        v = value_states

        N, H, TDST, HID = q.shape
        _, _, TSRC, _ = k.shape
        assert k.shape == v.shape

        q = q.reshape(N * H, TDST, HID)  # .contiguous()
        k = k.reshape(N * H, TSRC, HID)  # .contiguous()
        v = v.reshape(N * H, TSRC, HID)  # .contiguous()

        attn_output = sink_attention(
            q, k, v, 
            rope_cos.squeeze(0), 
            rope_sin.squeeze(0), 
            num_sink=4, 
            window_size=tree_k,
        )

        attn_output = attn_output.view(N, H, TDST, HID)  # .to(hidden_states.dtype)

    elif attention_method == 'hyper_attention':
        q = query_states / (query_states.shape[-1] ** 0.5)
        k = key_states
        v = value_states

        N, H, TDST, HID = q.shape
        _, _, TSRC, _ = k.shape
        assert k.shape == v.shape
        
        # q = q.view(N*H, TDST, HID)
        # k = k.view(N*H, TSRC, HID)
        # v = v.view(N*H, TSRC, HID)
        
        attn_output = hyper_attention(
            q, k, v, causal=True, scale=1.0
        )

    else:
        raise Exception(attention_method)

    return attn_output, last_cumsum, attn_sparsity_loss

