import json
import math
import os
import warnings

import numpy as np
import torch
import tqdm
import triton
import triton.language as tl
from transformers.utils import logging

if not hasattr(tl, "sort"):
    warnings.warn(
        "Triton Language does not contain `sort` function. "
        "This will cause the compilation problem. Please upgrade `triton >= 2.2.0`"
    )

from hip_research.utils.load_checkouts import load_checkouts
from hip_research.utils.seed import seed

from hip_attn.utils.benchmarking import get_bench

logger = logging.get_logger(__name__)
timer = lambda x: get_bench().region(x)


def main_latency_benchmark():
    global DEBUG

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dups", type=int, default=2)
    parser.add_argument("--query_size", type=int, default=1)
    parser.add_argument("--method", type=str, default="hip1.1")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--block_size_q", type=int, default=32)
    parser.add_argument("--block_stride_q", type=int, default=None)
    parser.add_argument("--block_size_k", type=int, default=1)
    parser.add_argument("--block_stride_k", type=int, default=None)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--scale_up", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=-1)
    parser.add_argument("--head_size", type=int, default=-1)
    parser.add_argument("--refresh_interval", type=int, default=8)
    parser.add_argument("--not_causal", action="store_true")
    parser.add_argument("--head_groups", type=int, default=1)

    # h2o
    parser.add_argument("--h2o-streaming", action="store_true")
    parser.add_argument("--h2o-shift-q-pos", action="store_true")
    parser.add_argument("--h2o-reduce-for-gqa", type=str, default="average")
    parser.add_argument("--h2o-model", type=str, default="llama3.1_8b")

    args = parser.parse_args()

    if args.query_size > 1:
        args.refresh_interval = -1

    DEBUG = args.debug
    TRACE = args.trace
    BSIZE = args.batch_size
    DUPS = args.dups
    QUERY_SIZE = args.query_size
    METHOD = args.method
    n_samples = args.samples
    is_causal = not args.not_causal

    if DEBUG:
        seed()

    get_bench().disabled = not TRACE
    get_bench().synchronize = True

    CHUNK_LEN = 1024
    # q: [N*H, T_DST, HID] = [32, 1024, 128]
    # k, v: [N*H_KV, T_SRC, HID] = [8, 1024, 128]
    q, k, v, out, cos, sin = load_checkouts(idx=0, window=40, seq_len=CHUNK_LEN)
    # head_size = q.shape[0] # NOTE moved

    HID = q.shape[-1]

    q = q.cpu()
    k = k.cpu()
    v = v.cpu()

    if args.head_size > 0 and args.head_size != q.shape[0]:
        head_reps = int(math.ceil(args.head_size / q.shape[0]))
        q = q.repeat(head_reps, 1, 1)[: args.head_size, :, :].contiguous()
        k = k.repeat(head_reps, 1, 1)[: args.head_size, :, :].contiguous()
        v = v.repeat(head_reps, 1, 1)[: args.head_size, :, :].contiguous()

    q = q.repeat(BSIZE, max(1, triton.cdiv(QUERY_SIZE, 1024)), 1)[
        :, :QUERY_SIZE, :
    ].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    started = False

    if args.hidden_size > 0 and args.hidden_size != HID:
        hid_reps = int(math.ceil(args.hidden_size / HID))
        q = q.repeat(1, 1, hid_reps)[:, :, : args.hidden_size].contiguous()
        k = k.repeat(1, 1, hid_reps)[:, :, : args.hidden_size].contiguous()
        v = v.repeat(1, 1, hid_reps)[:, :, : args.hidden_size].contiguous()
        HID = args.hidden_size

    head_size = q.shape[0]  # TODO move
    cos = sin = torch.randn(
        (k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device
    )  # [T, HID]

    if QUERY_SIZE == 1:
        k = (
            k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)
            .repeat(1, q.shape[0] // k.shape[0], 1, 1)
            .contiguous()
        )
        v = (
            v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)
            .repeat(1, q.shape[0] // v.shape[0], 1, 1)
            .contiguous()
        )
    else:
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)[
            :, :: args.head_groups, :, :
        ].contiguous()  # [128, 8, 32k, 128]
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)[
            :, :: args.head_groups, :, :
        ].contiguous()  # [128, 8, 32k, 128]
    q = q.view(BSIZE, -1, QUERY_SIZE, HID).contiguous()  # [128, 32, 32k, 128]

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    cos = cos.cuda()
    sin = sin.cuda()
    if METHOD in ["hip1.1"]:
        # q_quant = q.to(torch.float8_e5m2).view(torch.uint8)
        # k_quant = k.to(torch.float8_e5m2).view(torch.uint8)
        q_quant = q
        k_quant = k

    from hip_research.main.model_eval import MODELS
    from hip_research.models.h2o.h2o_llama import H2OLlamaAttention
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = LlamaConfig.from_pretrained(MODELS[args.h2o_model])

    config.hh_size = args.k // 2
    config.recent_size = args.k // 2
    config._attn_implementation = config.attn_implementation = "eager"
    config.h2o_shift_q_pos = args.h2o_shift_q_pos
    config.h2o_streaming = args.h2o_streaming
    config.reduction_for_gqa = args.h2o_reduce_for_gqa

    if QUERY_SIZE == 1:
        config.is_decoding = True
    else:
        config.is_decoding = False

    if METHOD == "h2o_stream":
        config.h2o_streaming = True

    mask_k = args.k
    bsz, num_heads, q_len, head_dim = q.shape
    _, kv_num_heads, kv_seq_len, _ = k.shape
    attention_mask = None
    num_key_value_groups = num_heads // kv_num_heads
    assert QUERY_SIZE == q_len

    from transformers import DynamicCache

    past_key_value = DynamicCache()

    if QUERY_SIZE == 1:
        past_key_value = DynamicCache()
        for i in range(config.num_hidden_layers):
            past_key_value.key_cache.append(k[:, :, -mask_k:, :].clone())
            past_key_value.value_cache.append(v[:, :, -mask_k:, :].clone())

        from hip_research.models.h2o.h2o_llama import repeat_kv

        k = repeat_kv(k, num_key_value_groups)
        v = repeat_kv(v, num_key_value_groups)

    layer_idx = 0
    h2o_attention = (
        H2OLlamaAttention(config, layer_idx=layer_idx).to(q.device).to(q.dtype)
    )

    h2o_attention.q_proj = torch.nn.Identity()
    h2o_attention.k_proj = torch.nn.Identity()
    h2o_attention.v_proj = torch.nn.Identity()
    h2o_attention.o_proj = torch.nn.Identity()

    bsz, num_heads, q_len, head_dim = q.shape
    cos = sin = torch.randn(bsz, q_len, head_dim).to("cuda").to(q.dtype)

    position_ids = torch.arange(0, QUERY_SIZE).repeat(bsz, 1).to(q.device)
    cache_position = torch.arange(QUERY_SIZE).to(q.device)

    def sample(past_key_value=None):
        with torch.no_grad():
            q_len = QUERY_SIZE
            if q_len > mask_k:
                assert QUERY_SIZE > 1
                assert QUERY_SIZE == CHUNK_LEN * DUPS

                past_key_value = DynamicCache()
                attn_output, attn_weights, past_key_value = (
                    h2o_attention._h2o_attention(  # , hh_score
                        q[:, :, :mask_k, :],
                        k[:, :, :mask_k, :],
                        v[:, :, :mask_k, :],
                        position_ids[:, :mask_k],
                        past_key_value=past_key_value,
                        output_attentions=False,
                        use_cache=True,
                        bsz=bsz,
                        cache_position=cache_position,
                        reduction_for_gqa=h2o_attention.config.reduction_for_gqa,
                        kv_seq_len=None,
                        decoding_loop_for_prefill=True,
                        compute_final_attn_output=False,
                        cos=cos,
                        sin=sin,
                    )
                )

                query_states_loop = q[:, :, mask_k:, :]
                assert query_states_loop.shape[-2] == q_len - mask_k
                for i in range(q_len - mask_k):
                    attn_output_, attn_weights_, past_key_value = (
                        h2o_attention._h2o_attention(  # , hh_score
                            query_states_loop[:, :, i, :][:, :, None, :],
                            k[:, :, mask_k:, :][:, :, i, :][:, :, None, :],
                            v[:, :, mask_k:, :][:, :, i, :][:, :, None, :],
                            position_ids[:, mask_k:][:, i][:, None],
                            past_key_value=past_key_value,
                            output_attentions=False,
                            use_cache=True,
                            bsz=bsz,
                            cache_position=cache_position,
                            reduction_for_gqa=h2o_attention.config.reduction_for_gqa,
                            kv_seq_len=None,
                            decoding_loop_for_prefill=True,
                            compute_final_attn_output=False,
                            cos=cos,
                            sin=sin,
                            i=i,
                        )
                    )

            else:
                if len(past_key_value.key_cache) > 0:
                    k_cache, v_cache = (
                        past_key_value.key_cache[layer_idx],
                        past_key_value.value_cache[layer_idx],
                    )
                    k_ = torch.cat([k_cache, k[:, :, -1:, :]], dim=2)
                    v_ = torch.cat([v_cache, v[:, :, -1:, :]], dim=2)
                else:
                    k_ = k
                    v_ = v
                attn_output, attn_weights = h2o_attention._h2o_attention_itself(
                    q,
                    k_,
                    v_,
                    bsz,
                    num_heads,
                    head_dim,
                    q_len,
                    kv_seq_len,
                    attention_mask,
                    past_key_value,
                    num_key_value_groups,
                    h2o_attention.config.reduction_for_gqa,
                    layer_idx=layer_idx,
                )

            for m in h2o_attention.modules():
                if hasattr(m, "_clean_cache"):
                    m._clean_cache()
        return past_key_value

    s = torch.cuda.Stream()
    graph = None
    graph_stateful = None
    samples = []
    for i in tqdm.tqdm(range(n_samples), dynamic_ncols=True):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if i < 5:
            s.wait_stream(torch.cuda.current_stream())
            past_key_value = sample(past_key_value=past_key_value)
            if args.refresh_interval > 0:
                sample(past_key_value)
            torch.cuda.current_stream().wait_stream(s)
        elif args.trace:
            sample(past_key_value=past_key_value)
        elif graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                past_key_value = sample(past_key_value=past_key_value)
            if args.refresh_interval > 0:
                graph_stateful = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_stateful):
                    sample(past_key_value=past_key_value)
        else:
            if args.refresh_interval > 0:
                if (i % args.refresh_interval) == 0:
                    graph.replay()
                else:
                    graph_stateful.replay()
            else:
                graph.replay()

        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / args.batch_size

        if i > n_samples * 0.1:
            if not started:
                get_bench().reset_measures()
                get_bench().reset_trace()
                started = True
            samples.append(elapsed)

    if TRACE:
        print(get_bench().format_tracetree())

    samples = np.array(samples)
    print(
        f"({METHOD}) {np.mean(samples):.8f} ms +- {np.std(samples):.4f} ms (q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)})"
    )

    os.makedirs("./cache/attention1_block_gpu/", exist_ok=True)
    with open("./cache/attention1_block_gpu/result.json", "w") as f:
        json.dump(
            {
                "method": METHOD,
                "mean": np.mean(samples),
                "std": np.std(samples),
                "query_length": q.shape[-2],
                "keyvalue_length": k.shape[-2],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main_latency_benchmark()
