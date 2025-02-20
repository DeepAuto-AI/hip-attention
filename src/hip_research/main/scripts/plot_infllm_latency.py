import numpy as np
import nvtx
import torch
from flash_attn import flash_attn_func
from inf_llm.attention import inf_llm_forward
from inf_llm.attention.rope import RotaryEmbeddingESM

from hip_attn.v1_1.attention2_draft_prefetch import (
    HiPAttentionArgs as HiPAttentionArgs11,
)
from hip_attn.v1_1.attention2_draft_prefetch import hip_attention as hip_attention_11
from hip_attn.v1_2.attention_extend import HiPAttentionArgs as HiPAttentionArgs12
from hip_attn.v1_2.attention_extend import (
    dual_stage_quadratic_hip_attention as hip_attention_12,
)

HEAD = 32
HEAD_KV = 8
HID = 128
DTYPE = torch.bfloat16
DEVICE = "cuda"
N_SAMPLES = 10


def run_inf_llm(fn, seq_len: int, chunk_size: int):
    fwd = fn()

    q = torch.zeros((1, chunk_size, HEAD, HID), dtype=DTYPE, device=DEVICE)
    k = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)
    v = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)

    def run():
        assert (seq_len % chunk_size) == 0

        base = 100000
        distance_scale = 1.0
        rope = RotaryEmbeddingESM(HID, base, distance_scale)

        past_key_values = None
        for istart in range(0, seq_len, chunk_size):
            if istart == (seq_len - chunk_size):
                torch.cuda.synchronize()
                with nvtx.annotate("infllm_prefill"):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    _, past_key_values = fwd(
                        None,
                        q.view(1, chunk_size, HEAD * HID),
                        torch.concat(
                            [
                                k[:, istart : istart + chunk_size, :, :].view(
                                    1, chunk_size, HEAD_KV * HID
                                ),
                                v[:, istart : istart + chunk_size, :, :].view(
                                    1, chunk_size, HEAD_KV * HID
                                ),
                            ],
                            dim=-1,
                        ),
                        rope,
                        True,
                        past_key_values,
                        lambda x: x,
                        lambda x: x[..., : HID * HEAD_KV],
                        lambda x: x[..., HID * HEAD_KV :],
                        lambda x: x,
                        HID,
                        HEAD,
                        HEAD_KV,
                    )

                    end.record()
                    torch.cuda.synchronize()
                    prefill_latency = start.elapsed_time(end)

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _, past_key_values = fwd(
                        None,
                        q[:, -1:, :, :].view(1, 1, HEAD * HID),
                        torch.concat(
                            [
                                k[:, istart : istart + 1, :, :].view(
                                    1, 1, HEAD_KV * HID
                                ),
                                v[:, istart : istart + 1, :, :].view(
                                    1, 1, HEAD_KV * HID
                                ),
                            ],
                            dim=-1,
                        ),
                        rope,
                        True,
                        past_key_values,
                        lambda x: x,
                        lambda x: x[..., : HID * HEAD_KV],
                        lambda x: x[..., HID * HEAD_KV :],
                        lambda x: x,
                        HID,
                        HEAD,
                        HEAD_KV,
                    )
                    end.record()
                    torch.cuda.synchronize()
                    decode_latency = start.elapsed_time(end)
                return np.array([prefill_latency, decode_latency])
            else:
                _, past_key_values = fwd(
                    None,
                    q.view(1, chunk_size, HEAD * HID),
                    torch.concat(
                        [
                            k[:, istart : istart + chunk_size, :, :].view(
                                1, chunk_size, HEAD_KV * HID
                            ),
                            v[:, istart : istart + chunk_size, :, :].view(
                                1, chunk_size, HEAD_KV * HID
                            ),
                        ],
                        dim=-1,
                    ),
                    rope,
                    True,
                    past_key_values,
                    lambda x: x,
                    lambda x: x[..., : HID * HEAD_KV],
                    lambda x: x[..., HID * HEAD_KV :],
                    lambda x: x,
                    HID,
                    HEAD,
                    HEAD_KV,
                )

    samples = []
    for i in range(N_SAMPLES):
        latency = run()
        if i > min(N_SAMPLES - 2, 3):
            samples.append(latency)
    return sum(samples) / len(samples)


def cuda_graph_wrapper(fn):
    torch.cuda.synchronize()
    for _ in range(3):
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_gen3(seq_len: int, chunk_size: int, using_extend: bool):
    q = torch.zeros((1, chunk_size, HEAD, HID), dtype=DTYPE, device=DEVICE)
    k = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)
    v = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)
    cos = torch.zeros((seq_len, HID), dtype=DTYPE, device=DEVICE)
    sin = torch.zeros((seq_len, HID), dtype=DTYPE, device=DEVICE)
    pos_id = torch.full((1, 1), fill_value=seq_len - 1, device=DEVICE)

    def run(is_decode):
        def fn(metadata=None):
            return hip_attention_12(
                q=q[:, -1:, :, :] if is_decode else q,
                k=k,
                v=v,
                args=HiPAttentionArgs12(
                    second_stage_k=2048,
                    sliding_window_size=1024,
                    sink_token_size=128,
                    using_extend=using_extend,
                    need_apply_rope=using_extend,
                    rope_cos=cos,
                    rope_sin=sin,
                    position_ids=pos_id if is_decode else None,
                ),
                cached_metadata=metadata,
            )

        def decode_fn():
            _, metadata = fn()
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)

            metadata.indices = None
            metadata.ks = None
            metadata.ks_start_end = None
            metadata.ks_count = None
            metadata.stage_caches = metadata.stage_caches
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)

            metadata.indices = None
            metadata.ks = None
            metadata.ks_start_end = None
            metadata.ks_count = None
            metadata.stage_caches = metadata.stage_caches[:-1]
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)

            metadata.indices = None
            metadata.ks = None
            metadata.ks_start_end = None
            metadata.ks_count = None
            metadata.stage_caches = metadata.stage_caches
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)
            _, metadata = fn(metadata=metadata)

        if is_decode:
            return cuda_graph_wrapper(decode_fn) / 16
        torch.cuda.synchronize()
        with nvtx.annotate("gen3"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end)

    samples = []
    for i in range(N_SAMPLES):
        latency_prefill = run(False)
        latency_decode = run(True)
        if i > min(N_SAMPLES - 2, 3):
            samples.append(np.array([latency_prefill, latency_decode]))
    return sum(samples) / len(samples)


def run_flash_atten(seq_len: int, chunk_size: int):
    q = torch.zeros((1, chunk_size, HEAD, HID), dtype=DTYPE, device=DEVICE)
    k = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)
    v = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)

    def run(is_decode):
        torch.cuda.synchronize()
        with nvtx.annotate("gen3"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            flash_attn_func(q[:, -1:, :, :] if is_decode else q, k, v, causal=True)
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end)

    samples = []
    for i in range(N_SAMPLES):
        latency = np.array([run(False), run(True)])
        if i > min(N_SAMPLES - 2, 3):
            samples.append(latency)
    return sum(samples) / len(samples)


def run_gen2(seq_len: int, chunk_size: int):
    q = torch.zeros((1, chunk_size, HEAD, HID), dtype=DTYPE, device=DEVICE)
    k = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)
    v = torch.zeros((1, seq_len, HEAD_KV, HID), dtype=DTYPE, device=DEVICE)

    def run(is_decode):
        def fn(m=None):
            return hip_attention_11(
                q=q[:, -1:] if is_decode else q,
                k=k,
                v=v,
                args=HiPAttentionArgs11(
                    block_size_k=2,
                    mask_k=1024,
                    sliding_window_size=1024,
                    sink_token_size=128,
                ),
                previous_metadata=m,
            )

        def decode_fn():
            _, m = fn()
            _, m = fn(m)
            _, m = fn(m)
            _, m = fn(m)

        if is_decode:
            return cuda_graph_wrapper(decode_fn) / 4
        torch.cuda.synchronize()
        with nvtx.annotate("gen3"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end)

    samples = []
    for i in range(N_SAMPLES):
        latency = np.array([run(False), run(True)])
        if i > min(N_SAMPLES - 2, 3):
            samples.append(latency)
    return sum(samples) / len(samples)


def exp():
    chunk_size = 32768
    seq_lens = [32, 64, 128, 256, 384, 512, 768, 1024]
    for seq_len in seq_lens:
        seq_len = seq_len * 1024
        # flash_latency = run_flash_atten(seq_len, chunk_size)

        infllm_offload_latency = run_inf_llm(
            lambda: inf_llm_forward(
                n_local=4096,
                n_init=128,
                topk=64,
                block_size=128,
                max_cached_block=64,
                exc_block_size=512,
                fattn=True,
                repr_topk=4,
                score_decay=0.1,
            ),
            seq_len,
            chunk_size,
        )

        infllm_latency = run_inf_llm(
            lambda: inf_llm_forward(
                n_local=4096,
                n_init=128,
                topk=64,
                block_size=128,
                max_cached_block=seq_len // 128,
                exc_block_size=512,
                fattn=True,
                repr_topk=4,
                score_decay=0.1,
            ),
            seq_len,
            chunk_size,
        )

        # gen2_latency = run_gen2(seq_len, chunk_size)

        # gen3_latency = run_gen3(seq_len, chunk_size, False)
        # gen3_extend_latency = run_gen3(seq_len, chunk_size, True)

        print(
            seq_len,
            # 'flash', flash_latency,
            "infllm_offload",
            infllm_offload_latency,
            "infllm",
            infllm_latency,
            # 'gen2', gen2_latency,
            # 'gen3', gen3_latency,
            # 'gen3e', gen3_extend_latency,
        )


if __name__ == "__main__":
    exp()
