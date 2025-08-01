import os
import unittest

import cv2
import numpy as np
import torch
from hip_research.utils.load_checkouts import load_checkouts

import hip_attn.v1_2.attention_extend
from hip_attn.v1_2.attention_extend import dual_stage_quadratic_hip_attention
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, ScanStage
from hip_attn.v1_2.utils import capture


class TestAttentionExtend(unittest.TestCase):
    def test_main(self):
        main_debug()


@torch.inference_mode(True)
def main_debug():
    IS_DEBUG = os.getenv("DEBUG", "0") == "1"
    if os.getenv("HIP_DEBUG_BENCH", "0") == "0":
        os.environ["HIP_DEBUG_BENCH"] = "1" if IS_DEBUG else "0"

    seq_len = int(os.getenv("SEQ_LEN", "131072"))
    query_seq_dups = int(os.getenv("Q_DUPS", "-1"))
    seq_dups = int(os.getenv("DUPS", "1"))
    if query_seq_dups < 0:
        query_seq_dups = seq_dups
    block_size = int(os.getenv("BLOCK_SIZE", "64"))
    num_samples = int(os.getenv("NUM_SAMPLES", "20"))
    batch_size = int(os.getenv("BATCH_SIZE", "1"))
    mask_only = int(os.getenv("MASK_ONLY", "0")) == "1"
    k_group_size = int(os.getenv("K_GROUP_SIZE", "1"))

    assert seq_dups > 0

    using_extend = True

    q, k, v, out, cos, sin = load_checkouts(
        idx=0,
        window=40,
        seq_len=seq_len,
        return_cos_sin=True,
        derope=using_extend,
        dtype=torch.bfloat16,
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    seq_len = seq_len * seq_dups

    q = q.repeat(1, query_seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    k = (
        k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    )  # .to(torch.float8_e5m2)
    v = (
        v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    )  # .to(torch.float8_e5m2)
    if cos is not None:
        cos = cos.repeat(seq_dups, 1)  # .to(torch.float8_e5m2)
        sin = sin.repeat(seq_dups, 1)  # .to(torch.float8_e5m2)

    q_mask = q
    k_mask = k

    k_mask = k
    _N, _T, _H, _D = k.shape
    tk = k_mask.view(_N, _T // k_group_size, k_group_size, _H, _D)
    k_mask = (
        (tk.min(dim=2, keepdim=True).values + tk.max(dim=2, keepdim=True).values)
        .expand(_N, _T // k_group_size, k_group_size, _H, _D)
        .contiguous()
        .view(*k.shape)
    )

    if batch_size > 1:
        q = q[:, -512:, :, :].contiguous()
        q = q.expand(batch_size, -1, -1, -1)
        k = k.expand(batch_size, -1, -1, -1)
        v = v.expand(batch_size, -1, -1, -1)
        q_mask = q
        k_mask = k

    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    print(q.shape, k.shape, v.shape, q_mask.shape, k_mask.shape)

    print("-" * 20)

    is_decode = q.shape[1] == 1

    preset = os.getenv("HIP_PRESET", "mid")
    config_stage = {
        "mid": [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=128,
                stage_k=None,
                stage_stride=1,
                using_landmark=False,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=8,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=2,
                stage_k=8192,
                stage_stride=1,
            ),
        ],
        "debug": [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=16,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=4,
                stage_k=512,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=1,
                stage_k=256,
                stage_stride=1,
            ),
        ],
    }[preset]
    config_second_k = {
        "mid": 2048,
        "debug": 128,
    }[preset]
    config_sa_extend_backend = {
        "mid": "streaming",
        "debug": "streaming",
    }[preset]

    dual_stage_kwargs = dict(
        q=q,
        k=k,
        v=v,
        args=HiPAttentionArgs(
            block_size_k=64,  # BLOCK_CHUNK
            sliding_window_size=128 if preset == "debug" else 1024,
            sink_token_size=64 if preset == "debug" else 256,
            # position_ids=position_ids,
            using_extend=using_extend,
            need_apply_rope=using_extend,
            rope_cos=cos,
            rope_sin=sin,
            second_stage_k=config_second_k,
            stages=config_stage,
            block_sparse_block_size_q=block_size,
            model_context_length=131072,
            extend_context_length=131072,
            # scan_early_terminate=1,
            # stage_early_terminate=1,
            scan_extend_backend="relative",
            sa_extend_backend=config_sa_extend_backend,
            stage_early_terminate=k_group_size,
            mask_only=mask_only,
        ),
    )

    ls_hip_extend = []
    ls_hip = []
    ls_fa = []

    refresh_interval = 8 if is_decode else 1

    metadata = None
    for i in range(min(num_samples, 5)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        start.record()
        if i == 0:
            hip_attn.v1_2.attention_extend.DEBUG = IS_DEBUG

        _, metadata = dual_stage_quadratic_hip_attention(
            **dual_stage_kwargs, cached_metadata=metadata
        )

        if ((i + 1) % refresh_interval) == 0:
            metadata = None

        if i == 0:
            hip_attn.v1_2.attention_extend.DEBUG = False
        end.record()

        end.synchronize()
        latency = start.elapsed_time(end)
        if i > 3:
            ls_hip_extend.append(latency)
        capture.report()
        print(latency)

    print("-" * 20)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if os.getenv("DEBUG", "0") == "1":
        input(">>>")

    dual_stage_kwargs["args"].using_extend = False
    dual_stage_kwargs["args"].need_apply_rope = False

    metadata = None
    for i in range(min(num_samples, 5)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        start.record()
        if i == 0:
            hip_attn.v1_2.attention_extend.DEBUG = os.getenv("DEBUG", "0") == "1"

        context, metadata = dual_stage_quadratic_hip_attention(
            **dual_stage_kwargs, cached_metadata=metadata
        )

        if ((i + 1) % refresh_interval) == 0:
            metadata = None

        if i == 0:
            hip_attn.v1_2.attention_extend.DEBUG = False
        end.record()

        end.synchronize()
        latency = start.elapsed_time(end)
        if i > 3:
            ls_hip.append(latency)
        capture.report()
        print(latency)

    print("-" * 20)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for i in range(min(num_samples, 5)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        start.record()
        if q.shape[1] == 1:
            using_fa3 = True
            if using_fa3:
                from hip_attn.v1_2.paged_hip import _forward_fa3

                _forward_fa3(
                    q,
                    k,
                    v,
                    sm_scale=1.0,
                    position_ids=torch.arange(0, q.shape[1])[None, :]
                    + (k.shape[1] - q.shape[1]),
                    using_extend=False,
                    need_apply_rope=False,
                    rope_cos=None,
                    rope_sin=None,
                    rope_is_neox_style=False,
                    k_descale=None,
                    v_descale=None,
                )
            else:
                flash_attn_with_kvcache(
                    q,
                    k,
                    v,
                    causal=True,
                )
        else:
            flash_attn_func(q, k, v, causal=True)
        end.record()

        end.synchronize()
        latency = start.elapsed_time(end)
        if i > 3:
            ls_fa.append(latency)
        print(latency)

    print("-" * 20)

    print(f"hip_extend,{sum(ls_hip_extend) / len(ls_hip_extend)}")
    print(f"hip,{sum(ls_hip) / len(ls_hip)}")
    print(f"fa,{sum(ls_fa) / len(ls_fa)}")


if __name__ == "__main__":
    main_debug()
