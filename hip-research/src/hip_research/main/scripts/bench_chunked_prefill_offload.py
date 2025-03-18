import time

import torch
import tqdm
from flash_attn import flash_attn_func

from hip_attn.v1_1.attention2_draft_sampling_extend import (
    HiPAttentionArgs,
    ScanStage,
    dual_stage_quadratic_hip_attention,
)

SAMPLE_SIZE = 10
DTYPE = torch.float8_e5m2


def bench_hip_offload(seq_len: int, chunk_size: int):
    if seq_len < chunk_size * 2:
        return 0


def bench_fa2(seq_len: int, chunk_size: int):
    q = torch.zeros((1, chunk_size, 8, 128), dtype=torch.bfloat16, device=0)
    k = v = torch.zeros((1, seq_len, 8, 128), dtype=torch.bfloat16, device=0)

    ls = []
    for i in tqdm.tqdm(range(SAMPLE_SIZE), dynamic_ncols=True, leave=False):
        torch.cuda.synchronize()
        t = time.time()
        flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        if i > 3:
            ls.append(time.time() - t)

    return sum(ls) / len(ls) * 1000


def bench_hip_chunked(seq_len: int, chunk_size: int, step_size: int):
    q = torch.zeros((1, chunk_size, 8, 128), dtype=DTYPE, device=0)
    k = v = torch.zeros((1, seq_len, 8, 128), dtype=DTYPE, device=0)
    cos = sin = torch.zeros((seq_len, 128), dtype=torch.bfloat16, device=0)

    dual_stage_kwargs = dict(
        q=q,
        k=k,
        v=v,
        args=HiPAttentionArgs(
            block_size_k=64,  # BLOCK_CHUNK
            block_stride_k=1,
            sliding_window_size=512,
            sink_token_size=256,
            using_extend=True,
            rope_cos=cos,
            rope_sin=sin,
            need_apply_rope=True,
        ),
        second_stage_k=2048,
        stages=[
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=256,
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
        ],
        model_context_length=131072,
        scan_extend_backend="relative",
        sa_extend_backend="streaming",
    )

    ls = []
    for i in tqdm.tqdm(range(SAMPLE_SIZE), dynamic_ncols=True, leave=False):
        torch.cuda.synchronize()
        t = time.time()

        for i_chunk in range(0, chunk_size, step_size):
            q_step = q[:, i_chunk : i_chunk + step_size]
            dual_stage_kwargs["q"] = q_step
            torch.cuda.synchronize()
            _, _ = dual_stage_quadratic_hip_attention(
                **dual_stage_kwargs,
            )
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        if i > 3:
            ls.append(time.time() - t)

    return sum(ls) / len(ls) * 1000


def bench_hip(seq_len: int, chunk_size: int):
    q = torch.zeros((1, chunk_size, 8, 128), dtype=DTYPE, device=0)
    k = v = torch.zeros((1, seq_len, 8, 128), dtype=DTYPE, device=0)
    cos = sin = torch.zeros((seq_len, 128), dtype=torch.bfloat16, device=0)

    dual_stage_kwargs = dict(
        q=q,
        k=k,
        v=v,
        args=HiPAttentionArgs(
            block_size_k=64,  # BLOCK_CHUNK
            block_stride_k=1,
            sliding_window_size=512,
            sink_token_size=256,
            using_extend=True,
            rope_cos=cos,
            rope_sin=sin,
            need_apply_rope=True,
        ),
        second_stage_k=2048,
        stages=[
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=256,
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
        ],
        model_context_length=131072,
        scan_extend_backend="relative",
        sa_extend_backend="streaming",
    )

    ls = []
    for i in tqdm.tqdm(range(SAMPLE_SIZE), dynamic_ncols=True, leave=False):
        torch.cuda.synchronize()
        t = time.time()
        _, _ = dual_stage_quadratic_hip_attention(
            **dual_stage_kwargs,
        )
        torch.cuda.synchronize()
        if i > 3:
            ls.append(time.time() - t)

    return sum(ls) / len(ls) * 1000


def bench_offload(seq_len: int, chunk_size: int):
    x = torch.zeros(
        (1, seq_len, 8, 128 * 2),
        dtype=DTYPE if DTYPE != torch.float8_e5m2 else torch.uint8,
    )
    ls = []
    for i in tqdm.tqdm(range(SAMPLE_SIZE), dynamic_ncols=True, leave=False):
        t = time.time()
        x.cuda()
        torch.cuda.synchronize()
        if i > 3:
            ls.append(time.time() - t)
    return sum(ls) / len(ls) * 1000


def bench_mlp(seq_len: int, chunk_size: int):
    x = torch.zeros((chunk_size, 32 * 128), dtype=torch.bfloat16, device=0)
    w1 = torch.zeros((32 * 128 * 4, 32 * 128), dtype=torch.bfloat16, device=0)
    w2 = torch.zeros((32 * 128, 32 * 128 * 4), dtype=torch.bfloat16, device=0)

    ls = []
    for i in tqdm.tqdm(range(SAMPLE_SIZE), dynamic_ncols=True, leave=False):
        torch.cuda.synchronize()
        t = time.time()

        y = torch.nn.functional.linear(x, w1)
        z = torch.nn.functional.linear(y, w2)
        assert x.shape == z.shape

        torch.cuda.synchronize()
        if i > 3:
            ls.append(time.time() - t)
    return sum(ls) / len(ls) * 1000


ls_hip_step = []
ls_hip = []
ls_fa2 = []
ls_offload = []
ls_mlp = []

chunk_size = 32
seq_lens = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

for seq_len in seq_lens:
    seq_len = seq_len * 1024
    cs = min(chunk_size, seq_len) * 1024
    ls_hip_step.append(bench_hip_chunked(seq_len, cs, 64))
    ls_hip.append(bench_hip(seq_len, cs))
    ls_fa2.append(bench_fa2(seq_len, cs))
    ls_offload.append(bench_offload(seq_len, cs))
    ls_mlp.append(bench_mlp(seq_len, cs))
    print(seq_len, "done")

import matplotlib.pyplot as plt

plt.clf()

# plt.plot(seq_lens, ls_hip_step, label='hip step')
plt.plot(seq_lens, ls_hip, label="hip")
# plt.plot(seq_lens, ls_fa2, label='fa2')
plt.plot(seq_lens, ls_offload, label="copy (PCIe)")
plt.plot(seq_lens, ls_mlp, label="mlp")

plt.grid()
plt.legend()
plt.xlabel("seq len (k)")
plt.ylabel("latency (ms)")

plt.savefig("dummy_chunked_prefill_offload.png", bbox_inches="tight")
