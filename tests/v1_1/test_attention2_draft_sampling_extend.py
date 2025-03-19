import os
import unittest

import cv2
import numpy as np
import torch
from hip_research.utils.load_checkouts import load_checkouts

import hip_attn.v1_1.attention2_draft_sampling_extend
from hip_attn.v1_1.attention2_draft_prefetch import HiPAttentionArgs, hip_attention
from hip_attn.v1_1.attention2_draft_sampling_extend import (
    ScanStage,
    dual_stage_quadratic_hip_attention,
)


class TestAttention2DraftSamplingExtend(unittest.TestCase):

    def test_attention(self):
        seq_len = int(os.getenv("SEQ_LEN", "131072"))
        seq_dups = int(os.getenv("DUPS", "1"))
        block_size = int(os.getenv("BLOCK_SIZE", "64"))
        num_samples = int(os.getenv("NUM_SAMPLES", "100"))
        batch_size = int(os.getenv("BATCH_SIZE", "1"))
        mask_only = int(os.getenv("MASK_ONLY", "0")) == "1"
        k_group_size = int(os.getenv("K_GROUP_SIZE", "1"))

        assert seq_dups > 0

        q, k, v, out, cos, sin = load_checkouts(
            idx=0,
            window=40,
            seq_len=seq_len,
            return_cos_sin=True,
            derope=True,
            dtype=torch.bfloat16,
        )
        HEAD = q.shape[0]
        HEAD_KV = k.shape[0]
        seq_len = seq_len * seq_dups

        q = q.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
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
        idx_pca_hid_q = None
        idx_pca_hid_k = None

        # q_pca = q[...,:32].contiguous()
        # k_pca = k[...,:32].contiguous()

        def pca(q, k, hid=32):
            import einx

            KV_HEAD_GROUP = q.shape[2] // k.shape[2]
            q_ori = q
            q = (
                q.view(
                    q.shape[0],
                    q.shape[1],
                    q.shape[2] // KV_HEAD_GROUP,
                    KV_HEAD_GROUP,
                    q.shape[3],
                )
                .permute(0, 3, 1, 2, 4)
                .flatten(0, 1)
            )

            t = einx.rearrange("n t h d -> h (n t) d", q).float()
            _, _, proj = torch.linalg.svd(t, full_matrices=False)
            proj = proj.to(q.dtype)  # type: torch.Tensor

            q = einx.dot("n t h d1, h d1 d2 -> n t h d2", q, proj)
            k = einx.dot("n t h d1, h d1 d2 -> n t h d2", k, proj)

            x_colsum = q.flatten(0, 1).abs().mean(dim=0, keepdim=False)
            y_colsum = k.flatten(0, 1).abs().mean(dim=0, keepdim=False)
            colsum = x_colsum + y_colsum

            _, topk_indices = colsum.topk(dim=-1, k=hid)
            idx_hid_keys = topk_indices.sort(dim=-1).values
            idx_hid_queries = idx_hid_keys.repeat_interleave(KV_HEAD_GROUP, 0)

            debug = np.zeros((idx_hid_queries.shape[0], q.shape[-1]), dtype=np.uint8)
            for i in range(idx_hid_queries.shape[0]):
                for j in range(idx_hid_queries.shape[1]):
                    debug[i, idx_hid_queries[i, j]] = 255
            cv2.imwrite("dummy_idx_pca.png", debug)

            assert idx_hid_keys.ndim == 2
            assert idx_hid_keys.shape == (k.shape[2], hid), idx_hid_keys.shape
            q = q_ori.gather(
                index=idx_hid_queries[None, None, :, :].expand(*q_ori.shape[:-1], -1),
                dim=-1,
            )
            k = k.gather(
                index=idx_hid_keys[None, None, :, :].expand(*k.shape[:-1], -1), dim=-1
            )

            return q, k, idx_hid_queries, idx_hid_keys

        # q_pca, k_pca, idx_pca_hid_q, idx_pca_hid_k = pca(q, k)

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
            "high": [
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=64,
                    stage_k=None,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=16,
                    stage_k=65536,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=1,
                    stage_k=16384,
                    stage_stride=1,
                ),
            ],
            "mid": [
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
                    stage_chunk_size=16,
                    stage_k=8192,
                    stage_stride=1,
                ),
            ],
            "low": [
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
            "high": 4096,
            "mid": 2048,
            "low": 2048,
            "debug": 128,
        }[preset]
        config_sa_extend_backend = {
            "high": "streaming",
            "mid": "streaming",
            "low": "streaming",
            "debug": "streaming",
        }[preset]

        dual_stage_kwargs = dict(
            q=q,
            k=k,
            v=v,
            q_mask=q_mask,
            k_mask=k_mask,
            idx_pca_hid_q=idx_pca_hid_q,
            idx_pca_hid_k=idx_pca_hid_k,
            args=HiPAttentionArgs(
                # deprecated
                # mask_k=config_mask_k,
                # block_size_q=64,
                # block_stride_q=config_bsq,
                block_size_k=64,  # BLOCK_CHUNK
                block_stride_k=1,
                sliding_window_size=128 if preset == "debug" else 1024,
                sink_token_size=64 if preset == "debug" else 256,
                # position_ids=position_ids,
                using_extend=True,
                rope_cos=cos,
                rope_sin=sin,
                need_apply_rope=True,
            ),
            second_stage_k=config_second_k,
            stages=config_stage,
            block_sparse_block_size_q=block_size,
            model_context_length=512,
            # scan_early_terminate=1,
            # stage_early_terminate=1,
            # scan_extend_backend='relative',
            sa_extend_backend=config_sa_extend_backend,
            stage_early_terminate=k_group_size,
            mask_only=mask_only,
        )

        hip_1k_kwargs = dict(
            q=q,
            k=k,
            v=v,
            args=HiPAttentionArgs(
                mask_k=1024,
                block_size_q=64,
                block_stride_q=2,
                block_size_k=2,
                block_stride_k=1,
            ),
            mask_only=mask_only,
        )

        hip_512_kwargs = dict(
            q=q,
            k=k,
            v=v,
            args=HiPAttentionArgs(
                mask_k=512,
                block_size_q=64,
                block_stride_q=2,
                block_size_k=2,
                block_stride_k=1,
            ),
            mask_only=mask_only,
        )

        refresh_interval = 8 if is_decode else 2

        metadata = None
        for i in range(min(num_samples, 24)):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            if i == 0:
                hip_attn.v1_1.attention2_draft_sampling_extend.DEBUG = (
                    os.getenv("DEBUG", "0") == "1"
                )

            # print(cos.shape)
            # print(sin.shape)

            _, metadata = dual_stage_quadratic_hip_attention(
                **dual_stage_kwargs, cached_metadata=metadata
            )

            if ((i + 1) % refresh_interval) == 0:
                metadata = None

            if i == 0:
                hip_attn.v1_1.attention2_draft_sampling_extend.DEBUG = False
            end.record()

            end.synchronize()
            print(start.elapsed_time(end))

        print("-" * 20)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if os.getenv("DEBUG", "0") == "1":
            input(">>>")

        dual_stage_kwargs["args"].using_extend = False
        dual_stage_kwargs["args"].need_apply_rope = False

        metadata = None
        for i in range(min(num_samples, 24)):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            if i == 0:
                hip_attn.v1_1.attention2_draft_sampling_extend.DEBUG = (
                    os.getenv("DEBUG", "0") == "1"
                )

            # print(cos.shape)
            # print(sin.shape)

            context, metadata = dual_stage_quadratic_hip_attention(
                **dual_stage_kwargs, cached_metadata=metadata
            )

            if ((i + 1) % refresh_interval) == 0:
                metadata = None

            if i == 0:
                hip_attn.v1_1.attention2_draft_sampling_extend.DEBUG = False
            end.record()

            end.synchronize()
            print(start.elapsed_time(end))

        print("-" * 20)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        metadata = None
        for i in range(min(num_samples, 24)):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            context, metadata = hip_attention(
                **hip_1k_kwargs,
                previous_metadata=metadata,
            )
            end.record()

            if ((i + 1) % (8 if is_decode else 1)) == 0:
                metadata = None

            end.synchronize()
            print(start.elapsed_time(end))

        print("-" * 20)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        metadata = None
        for i in range(min(num_samples, 24)):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            context, metadata = hip_attention(
                **hip_512_kwargs,
                previous_metadata=metadata,
            )
            end.record()

            if ((i + 1) % (8 if is_decode else 1)) == 0:
                metadata = None

            end.synchronize()
            print(start.elapsed_time(end))

        print("-" * 20)

        for i in range(min(num_samples, 5)):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            if q.shape[1] == 1:
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
            print(start.elapsed_time(end))
