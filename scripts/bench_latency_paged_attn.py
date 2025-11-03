"""
python scripts/benchmark_latency_paged_attn.py
"""

import os
import json
import traceback
import pandas as pd
import torch
from transformers import AutoConfig
import triton
from hip_attn.v1_2.paged_hip import forward_paged_hip, HiPAttentionConfig

def forward_seq_len(
    dtype: torch.dtype,
    seq_len: int,
    q_head: int,
    kv_head: int,
    head_dim: int,
    hip_config: HiPAttentionConfig,
    batch_size: int = 1,
):
    device = torch.device("cuda:0")

    query = torch.rand(
        (batch_size * seq_len, q_head, head_dim),
        dtype=torch.bfloat16,
        device=device
    )
    k_cache = torch.rand(
        # NOTE: + 1 is special behavior on SGlang. I am not sure about it is exists in vLLM too.
        ((seq_len + 1) * batch_size, kv_head, head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtype)
    v_cache = k_cache.clone()
    positions = torch.arange(0, batch_size * seq_len, dtype=torch.long, device=device)
    positions = positions % seq_len
    seq_lens = torch.zeros((batch_size,), dtype=torch.long, device=device)
    seq_lens[:] = seq_len
    block_table = torch.arange(0, batch_size *  seq_len, dtype=torch.long, device=device)
    block_table = block_table.view(batch_size, seq_len)
    layer_id = 10
    logit_cap = None
    orig_context_length = seq_len
    max_context_length = seq_len
    is_kv_cache_offload_enable = False
    rope_range = (0, head_dim)
    extend_prefix_lens_cpu = [0,] * batch_size
    extend_seq_lens_cpu = [seq_len,] * batch_size

    torch.cuda.synchronize()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()

    forward_paged_hip(
        query=query,
        sm_scale=1 / (head_dim ** 0.5),
        batch_size=batch_size,
        k_cache=k_cache,
        v_cache=v_cache,
        offload_cache=None,
        positions=positions,
        seq_lens=seq_lens,
        req_to_tokens=None,
        req_pool_indices=None,
        block_table=block_table,
        rope_cos=None,
        rope_sin=None,
        layer_id=layer_id,
        logit_cap=logit_cap,
        orig_context_len=orig_context_length,
        max_context_len=max_context_length,
        hip_config=hip_config,
        is_kv_cache_offload_enabled=is_kv_cache_offload_enable,
        rope_range=rope_range,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        is_decode=False,
    )

    end.record()
    end.synchronize()
    return start.elapsed_time(end)

def try_set_environ(name: str, value):
    if name in os.environ:
        return
    os.environ[name] = value

def evaluate_autotune(
    sa_block_size: int,
    bsa_block_k: int,
    hip_config: HiPAttentionConfig,
):
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    try_set_environ("BSA_K", "32")
    try_set_environ("BSA_EXACT_K", "32")
    try_set_environ("BSA_BLOCK_K", str(bsa_block_k))
    try_set_environ("HIP_DEBUG_DELTA_QSA", "1")
    try_set_environ("HIP_DEBUG_RECOMPUTE_SPLIT", "0")
    try_set_environ("TRITON_PRINT_AUTOTUNING", "1")
    try_set_environ("SA_BLOCK_SIZE", str(sa_block_size))
    try_set_environ("SA_DECODE_BLOCK_SIZE", "128")
    try_set_environ("HIP_DISABLE_AUTOTUNE", "0")

    n_warmup = 3
    n_measure = 100
    n_tp = 8

    config = AutoConfig.from_pretrained(model_name)
    q_head = config.num_attention_heads
    q_head = triton.cdiv(q_head, n_tp)
    kv_head = config.num_key_value_heads
    kv_head = triton.cdiv(kv_head, n_tp)
    head_dim = config.hidden_size // config.num_attention_heads

    seq_lens = [32, 64, 128, 256, 384, 512, 768, 1024]
    dtypes = [torch.bfloat16, torch.float8_e5m2]

    data = []

    for seq_len in seq_lens:
        for dtype in dtypes:
            for _ in range(n_warmup):
                try:
                    forward_seq_len(
                        dtype,
                        seq_len * 1024,
                        q_head,
                        kv_head,
                        head_dim,
                        hip_config=hip_config,
                    )
                except Exception:
                    traceback.print_exc()
            
            latencies = []
            for _ in range(n_measure):
                try:
                    latency = forward_seq_len(
                        dtype,
                        seq_len * 1024,
                        q_head,
                        kv_head,
                        head_dim,
                        hip_config=hip_config,
                    )
                    exception = ""
                except Exception:
                    latency = float("nan")
                    exception = traceback.format_exc()
                latencies.append(latency)
            latency = sum(latencies) / len(latencies)
            
            data_point = {
                "dtype": str(dtype),
                "seq_len": seq_len,
                "bsa_block_k": bsa_block_k,
                "sa_block_size": sa_block_size,
                "model": model_name,
                "latency": latency,
                "exception": exception,
            }
            print(data_point, flush=True)
            data.append(data_point)
    
    return data

def main():
    hip_config = HiPAttentionConfig(
        json_or_path="./configs/mixed_landmark_0814_no_extend_qsa.json",
        json_override='{"__seq_thresh_fa3": 0}'
    )

    bsa_block_ks = [64, 32]
    sa_block_sizes = [256, 128, 64]

    data = []

    for bsa_block_k in bsa_block_ks:
        for sa_block_size in sa_block_sizes:
            data.extend(evaluate_autotune(
                bsa_block_k=bsa_block_k,
                sa_block_size=sa_block_size,
                hip_config=hip_config,
            ))
    
    os.makedirs("saves/bench_latency_paged_attn", exist_ok=True)
    with open("saves/bench_latency_paged_attn/measures.json", "w") as f:
        json.dump(data, f)
    
    df = pd.DataFrame(data)
    df.to_csv("saves/bench_latency_paged_attn/measures.csv")

if __name__ == "__main__":
    main()