# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import itertools
from functools import partial
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from tilelang.autotuner import *
from torch import Tensor

from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, safe_stride


def get_configs():
    block_M = [128]
    block_N = [128]
    num_stages = [2]
    threads = [256]
    _configs = list(itertools.product(block_M, block_N, num_stages, threads))

    configs = [
        {"block_M": c[0], "block_N": c[1], "num_stages": c[2], "threads": c[3]}
        for c in _configs
    ]
    return configs


def block_sparse_attention_device(
    batch,
    heads,
    seq_len,
    dim,
    block_size_q,
    block_size_k,
    num_bk,
    is_causal,
    tune=False,
    groups=1,
):
    # scale = ((1.0 / dim) ** 0.5) * 1.44269504  # log2(e)
    scale = 1.0
    head_kv = heads // groups
    bdst = tilelang.cdiv(seq_len, block_size_q)
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    indices_shape = [batch * heads, bdst, num_bk]
    dtype = "bfloat16"
    accum_dtype = "float"
    indices_dtype = "int32"

    @tilelang.jit(out_idx=[3])
    def kernel_func(block_M, block_N, block_BK, num_stages, threads):
        @T.macro
        def MMA0(
            K: T.Tensor(kv_shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            indices: T.FragmentBuffer(
                [
                    block_N,
                ],
                indices_dtype,
            ),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            # T.copy(
            #     K[
            #         bz,
            #         k * block_N:(k + 1) * block_N,
            #         by // groups,
            #         :
            #     ],
            #     K_shared
            # )
            for i, j in T.Parallel(block_N, dim):
                tsrc = indices[i]
                K_shared[i, j] = T.if_then_else(
                    tsrc >= 0 and tsrc < seq_len, K[bz, tsrc, by // groups, j], 0
                )

                # if tsrc >= 0 and tsrc < seq_len:
                #     K_shared[i, j] = K[
                #         bz,
                #         tsrc,
                #         by // groups,
                #         j
                #     ]
                # else:
                #     K_shared[i, j] = 0

            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
            else:
                T.clear(acc_s)
            T.gemm(
                Q_shared,
                K_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )

        @T.macro
        def MMA1(
            V: T.Tensor(kv_shape, dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            indices: T.FragmentBuffer([block_N], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            for i, j in T.Parallel(block_N, dim):
                # V_shared[i, j] = V[
                #     bz,
                #     indices[i],
                #     by // groups,
                #     j
                # ]

                tsrc = indices[i]
                V_shared[i, j] = T.if_then_else(
                    tsrc >= 0 and tsrc < seq_len, V[bz, tsrc, by // groups, j], 0
                )

            # T.copy(
            #     V[
            #         bz,
            #         k * block_N:(k + 1) * block_N,
            #         by // groups,
            #         :
            #     ],
            #     V_shared
            # )
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            scores_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),
            logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(
                    scores_max_prev[i] * scale - scores_max[i] * scale
                )
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.macro
        def FlashAttn(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
            start: int,
            end: int,
        ):
            with T.Kernel(
                T.ceildiv(seq_len, block_M), heads, batch, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                indices = T.alloc_fragment([block_N], indices_dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # loop_range = (
                #     T.min(
                #         T.ceildiv(seq_len, block_N),
                #         T.ceildiv((bx + 1) * block_M, block_N)
                #     )
                #     if is_causal else
                #     T.ceildiv(seq_len, block_N)
                # )
                loop_range = T.ceildiv(num_bk, block_BK)

                for k in T.Pipelined(
                    loop_range,
                    num_stages=num_stages,
                    # order=[-1, 0, 3, 1, -1, 2],
                    # stage=[-1, 0, 0, 1, -1, 1],
                    # group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]
                ):
                    for i in T.Parallel(block_N):
                        tsrc = (
                            Indices[
                                bz * heads + by, bx, k * block_BK + i // block_size_k
                            ]
                            + i % block_size_k
                        )
                        if tsrc >= start and tsrc < end:
                            indices[i] = tsrc
                        else:
                            indices[i] = -1

                    MMA0(K, Q_shared, K_shared, indices, acc_s, k, bx, by, bz)
                    Softmax(
                        acc_s,
                        acc_s_cast,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                    )
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, indices, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
        ):
            FlashAttn(Q, K, V, Output, Indices, 0, seq_len)

        return main

    if tune:

        @autotune(
            configs=get_configs(),
            keys=["block_M", "block_N", "num_stages", "threads"],
            warmup=10,
            rep=10,
        )
        @tilelang.jit(out_idx=[3])
        def kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_N, num_stages, threads)

        return kernel()
    else:

        def kernel(block_M, block_N, block_BK, num_stages, threads):
            return kernel_func(block_M, block_N, block_BK, num_stages, threads)

        return kernel


from .utils import capture

compiled_kernel = None


@capture
def block_sparse_attention_tilelang(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    args: "HiPAttentionArgs",
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    EXTEND_BACKEND: str = "streaming",
    model_context_length: int = 131072,
    extend_context_length: int = 131072,
    offload_update_cache: bool = False,
    return_running_statistics: bool = False,
) -> Tensor:
    assert isinstance(k, Tensor)
    assert isinstance(v, Tensor)

    BSZ, TDST, HEAD, HID = q.shape
    _BSZ, TSRC, HEAD_KV, _HID = k.shape
    assert k.shape == v.shape
    assert BSZ == _BSZ
    assert TDST == TSRC
    assert HID == _HID
    assert (HEAD % HEAD_KV) == 0

    block_size_q = args.block_size_q
    block_size_k = args.block_size_k
    block_size = 256
    block_bk = block_size // block_size_k

    indices = indices.to(torch.int32)
    BH, BDST, BK = indices.shape
    assert BH == (BSZ * HEAD)
    assert tilelang.cdiv(TDST, block_size_q) == BDST

    global compiled_kernel

    if compiled_kernel is None:
        compiled_kernel = block_sparse_attention_device(
            batch=BSZ,
            heads=HEAD,
            seq_len=TDST,
            dim=HID,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            num_bk=BK,
            is_causal=True,
            tune=False,
            groups=HEAD // HEAD_KV,
        )(
            block_M=block_size_q,
            block_N=block_size,
            block_BK=block_bk,
            num_stages=2,
            threads=128,
        )

    output = compiled_kernel(q, k, v, indices)

    return output
