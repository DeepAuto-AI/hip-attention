"""
HiP v1.1
TODO:
1. Masking iteration using integer to avoid aliasing and collision
 - Convert tmask into int32 (good!)
 - Reuse the computed dot products (good!)
2. Using QUEST method for b_k (not very good)
3. Maximum token location predictor
 - Test oracle (not very good, sometimes worse)
 - Test estimators
4. sifters? (not very good) (num_unions, num_samples handle this)
5. masking -> allocate cells (num_samples, traverse_from_last_step)
6. StreamLLM based traverse (use Self-Extend instead of SLLM)
7. causal-batch (fine, topk_head_group_size)
8. 2d support
9. support backward across tree
10. chunk-wise BPTT
"""

import copy
import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import nvtx
import torch
import torch.nn.functional as F
import tqdm
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.utils.rope import adjust_rope
from hip_attn.utils.triton_argsort import argsort as tl_argsort
from hip_attn.v1_0.attention1_block_gpu import to_dense

try:
    from vllm_flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:
    try:
        from flash_attn import flash_attn_func, flash_attn_with_kvcache
    except ImportError:
        flash_attn_func = flash_attn_with_kvcache = None


def cdiv_python(a, b):
    return math.ceil(float(a) / float(b))


DEFAULT_CACHE_MODIFIER = tl.constexpr(".cg")
EXPERIMENTAL_SAMPLING_POSITION: tl.constexpr = float(
    os.environ.get("HIP_EXPERIMENTAL_SAMPLING_POSITION", "0.5")
)
DEFAULT_EXTEND_BACKEND: tl.constexpr = os.environ.get("HIP_EXTEND_BACKEND", "streaming")


@dataclass
class HiPAttentionOutputMetadata:
    indices: Tensor
    ks: Tensor
    ks_count: Tensor
    ks_start_end: Tensor

    key_access_log: Optional[Tensor]
    key_access_count: Optional[Tensor]

    block_access_log: Optional[Tensor]
    block_access_score: Optional[Tensor]
    block_access_count: Optional[Tensor]


@dataclass
class HiPAttentionArgs:
    mask_k: int = 512

    block_size_q: int = 64
    block_stride_q: int = 2
    block_size_k: int = 2
    block_stride_k: int = 1
    block_size_k_group: int = 1
    block_size_k_after_masking: int = int(os.getenv("HIP_BK_AFTER_MASK", "-1"))

    group_size_q: int = int(os.getenv("HIP_GROUP_SIZE_Q", "1"))

    add_approx_k_window: bool = os.getenv("HIP_USING_APPROX_K", "0") == "1"
    approx_k: int = 32
    approx_k_window: int = 8

    add_snap_kv: bool = os.getenv("HIP_USING_SNAP_KV", "0") == "1"
    snap_kv_vert_k: int = int(os.getenv("HIP_SNAP_KV_VERT_K", "32"))
    snap_kv_diag_k: int = int(os.getenv("HIP_SNAP_KV_DIAG_K", "256"))
    # snap_kv_page_size: int = 8
    snap_kv_obs_window: int = 128
    snap_kv_kernel_size: int = 15

    is_causal: bool = True

    sliding_window_size: int = int(os.getenv("HIP_SW", "256"))
    sink_token_size: int = int(os.getenv("HIP_NSINK", "16"))

    num_dense_queries: int = -1

    using_extend: bool = False
    need_apply_rope: bool = False
    self_extend_neighboor_window: int = 1024
    self_extend_group_size: int = 8
    rope_cos: Optional[Tensor] = None
    rope_sin: Optional[Tensor] = None

    topk_head_group_size: int = 1
    sample_method: str = "center"
    branch_method: str = "half"

    traverse_from_last_step: bool = False
    step_size: Optional[int] = None
    num_samples: int = 1
    chunk_size: Optional[int] = None
    num_unions: int = 1

    score_head_group_size: int = 1

    using_sparq: bool = False
    sparq_hid: int = 32

    low_res_sample_scale: int = 1
    low_res_oversample_rate: int = 1
    low_res_oversample_block_stride_k: int = 1

    output_key_access_log: bool = False
    output_block_access_log: bool = False
    access_log_num_blocks_to_keep: Optional[int] = 987654321

    q_quant: Optional[Tensor] = None
    k_quant: Optional[Tensor] = None

    sparq_ind: Optional[Tensor] = None

    k_cache: Optional[Tensor] = None
    v_cache: Optional[Tensor] = None
    cache_seq_lens: Optional[Tensor] = None
    block_table: Optional[Tensor] = None

    # BUG(-): this nameing is wrong...
    position_ids: Optional[Tensor] = None

    offload_cache_method: Optional[Literal["single_level", "bloom"]] = "single_level"
    offload_cache_kv_heads: Optional[int] = None
    offload_cache_mask_k_tables: Optional[Tensor] = None
    offload_cache_mask_k_banks: Optional[Tensor] = None
    offload_cache_mask_k_bank_stats: Optional[Tensor] = None
    offload_cache_sa_k_tables: Optional[Tensor] = None
    offload_cache_sa_k_banks: Optional[Tensor] = None
    offload_cache_sa_v_tables: Optional[Tensor] = None
    offload_cache_sa_v_banks: Optional[Tensor] = None
    offload_cache_counters: Optional[Tensor] = None

    # to support gemma2
    logit_softcap: Optional[float] = None

    # NOTE: this will be equivalant BigBird
    randomize_mask: bool = False

    # NOTE: for testing
    decode_style_masking: bool = os.getenv("HIP_DECODE_STYLE_PPL", "0") == "1"

    def __post_init__(self):
        if self.rope_cos is not None and self.rope_cos.ndim == 3:
            self.rope_cos = self.rope_cos.view(-1, self.rope_cos.shape[-1])
            self.rope_sin = self.rope_sin.view(-1, self.rope_sin.shape[-1])
        if self.q_quant is not None:
            assert self.q_quant.ndim == 4
            assert self.k_quant.ndim == 4
        self.using_paged_cache = self.k_cache is not None
        if self.logit_softcap == 0:
            self.logit_softcap = None

    def clone(self):
        return copy.copy(self)

    def json(self, convert_tensor_to_meta=True):
        from dataclasses import fields

        json = {}
        for field in fields(self):
            json[field.name] = getattr(self, field.name)

        if convert_tensor_to_meta:
            for k, v in json.items():
                if isinstance(v, Tensor):
                    v = f"{v.dtype}{list(v.shape)}@{v.device}.{v.data_ptr():02X}"
                json[k] = v

        return json

    def safe_stride(self, x: Optional[Tensor], ndim: int):
        if x is None:
            return tuple(
                [
                    0,
                ]
                * ndim
            )
        else:
            stride = x.stride()
            assert len(stride) == ndim
            return stride

    def get_q_quant(self, q: Tensor):
        return self.q_quant if self.q_quant is not None else q

    def get_k_quant(self, k: Tensor):
        return self.k_quant if self.k_quant is not None else k

    def args_extend(self):
        return (
            self.using_extend,
            self.need_apply_rope,
            self.self_extend_neighboor_window,
            self.self_extend_group_size,
            *self.args_rope_cos(),
            *self.args_rope_sin(),
        )

    def args_rope_cos(self):
        return (
            self.rope_cos,
            *self.safe_stride(self.rope_cos, 2),
        )

    def args_rope_sin(self):
        return (
            self.rope_sin,
            *self.safe_stride(self.rope_sin, 2),
        )

    def args_sparq(self):
        if self.sparq_ind is None:
            using_sparq = False
            sparq_hid = 0
        else:
            using_sparq = True
            sparq_hid = self.sparq_ind.shape[-1]
            assert self.sparq_ind.ndim == 4

        return (
            using_sparq,
            sparq_hid,
            self.sparq_ind,
            *self.safe_stride(self.sparq_ind, 4),
        )

    def args_bq_bsq_bk_bsk(self):
        return (
            self.block_size_q,
            self.block_stride_q,
            self.block_size_k,
            self.block_stride_k,
        )

    def args_paged_kv_cache(self):
        using_page = self.using_paged_cache

        if using_page:
            assert self.v_cache is not None
            assert self.k_cache.ndim == self.v_cache.ndim
            assert self.k_cache.ndim == 4
            assert self.block_table is not None
            assert self.block_table.ndim == 2
            assert self.cache_seq_lens is not None
            assert self.cache_seq_lens.ndim == 1
            page_size = self.k_cache.shape[1]
        else:
            page_size = 0

        return (
            using_page,
            page_size,
            self.k_cache,
            *self.safe_stride(self.k_cache, 4),
            self.v_cache,
            *self.safe_stride(self.v_cache, 4),
            self.block_table,
            *self.safe_stride(self.block_table, 2),
            self.cache_seq_lens,
            *self.safe_stride(self.cache_seq_lens, 1),
        )

    def args_offload_cache(self, is_masking):
        if is_masking:
            using_offload_cache = self.offload_cache_mask_k_tables is not None
            if using_offload_cache:
                assert self.offload_cache_mask_k_tables.ndim == 2
                assert self.offload_cache_mask_k_banks.ndim == 4
                assert self.offload_cache_counters.ndim == 2
                offload_cache_budget = self.offload_cache_mask_k_banks.shape[1]
            else:
                offload_cache_budget = 0

            # print(self.offload_cache_method)

            return (
                using_offload_cache,
                self.offload_cache_method,
                offload_cache_budget,
                self.offload_cache_kv_heads,
                self.offload_cache_mask_k_tables,
                *self.safe_stride(self.offload_cache_mask_k_tables, 2),
                self.offload_cache_mask_k_banks,
                *self.safe_stride(self.offload_cache_mask_k_banks, 4),
                self.offload_cache_mask_k_bank_stats,
                *self.safe_stride(self.offload_cache_mask_k_bank_stats, 3),
                self.offload_cache_counters,
                *self.safe_stride(self.offload_cache_counters, 2),
            )
        else:
            using_offload_cache = self.offload_cache_sa_k_tables is not None
            if using_offload_cache:
                assert self.offload_cache_sa_k_tables.ndim == 2
                assert self.offload_cache_sa_k_banks.ndim == 4
                assert self.offload_cache_sa_v_tables.ndim == 2
                assert self.offload_cache_sa_v_banks.ndim == 4
                assert self.offload_cache_counters.ndim == 2
                offload_cache_budget = self.offload_cache_sa_v_banks.shape[1]
                assert self.offload_cache_sa_k_banks.shape[1] == offload_cache_budget
            else:
                offload_cache_budget = 0

            # print(self.offload_cache_method)

            return (
                using_offload_cache,
                self.offload_cache_method,
                offload_cache_budget,
                self.offload_cache_kv_heads,
                self.offload_cache_sa_k_tables,
                *self.safe_stride(self.offload_cache_sa_k_tables, 2),
                self.offload_cache_sa_k_banks,
                *self.safe_stride(self.offload_cache_sa_k_banks, 4),
                self.offload_cache_sa_v_tables,
                *self.safe_stride(self.offload_cache_sa_v_tables, 2),
                self.offload_cache_sa_v_banks,
                *self.safe_stride(self.offload_cache_sa_v_banks, 4),
                self.offload_cache_counters,
                *self.safe_stride(self.offload_cache_counters, 2),
            )


@triton.jit
def masking_iteration_draft_cuda_initialize(
    # in
    INDICES_SEED,
    stride_indices_seed_b,
    stride_indices_seed_bdst,
    stride_indices_seed_bk,
    KS_SEED,
    stride_ks_seed_b,
    stride_ks_seed_bdst,
    POS,
    stride_pos_n,
    stride_pos_tdst,
    # out
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS,
    stride_ks_b,
    stride_ks_bdst,
    GROUP_SIZE,
    stride_group_size_b,
    stride_group_size_bdst,
    stride_group_size_bk,
    # temp
    T_GROUP_SIZE,
    stride_t_group_size_b,
    stride_t_group_size_bdst,
    # param
    mask_k: int,
    sliding_window_size: int,
    sink_token_size: int,
    max_group_size_seed: float,
    G,
    MAX_TDST,
    MAX_TSRC,
    HEAD,
    block_size_q: tl.constexpr,
    block_stride_q: tl.constexpr,
    block_size_k: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_MASK_BLOCK_K: tl.constexpr,
):
    idx_b = tl.program_id(0).to(tl.int64)
    idx_bdst = tl.program_id(1).to(tl.int64)
    idx_group = tl.program_id(2).to(tl.int64)
    idx_tdst = tl.arange(0, block_size_q).to(tl.int64) + idx_bdst * block_size_q
    mask_tdst = idx_tdst < MAX_TDST

    mask_block_k = tl.cdiv(mask_k, block_size_k)
    if IS_CAUSAL:
        pos_tdst = tl.load(
            POS + (idx_b * G // HEAD) * stride_pos_n + idx_tdst * stride_pos_tdst,
            mask=mask_tdst,
            other=0,
        )
    else:
        pos_tdst = tl.full(
            (block_size_q // block_stride_q,), value=MAX_TSRC, dtype=tl.int64
        )
    TSRC = tl.max(pos_tdst)
    tl.debug_barrier()
    TSRC = tl.maximum(0, TSRC - sliding_window_size - sink_token_size)
    BSRC = tl.cdiv(TSRC, block_size_k)
    MAX_BSRC = tl.cdiv(MAX_TSRC, block_size_k)
    N_SINK_BLOCK = tl.cdiv(sink_token_size, block_size_k)

    if TSRC <= mask_k:
        idx_bk = tl.arange(0, BLOCK_MASK_BLOCK_K)
        mask_bk = idx_bk < BSRC
        if INDICES is not None:
            tl.store(
                INDICES
                + idx_b * stride_indices_b
                + idx_bdst * stride_indices_bdst
                + (idx_group * BSRC + idx_bk) * stride_indices_bk,
                value=idx_group * MAX_BSRC + idx_bk + N_SINK_BLOCK,
                mask=mask_bk,
            )

        if idx_group == 0:
            if KS is not None:
                tl.store(
                    KS + idx_b * stride_ks_b + idx_bdst * stride_ks_bdst, value=BSRC * G
                )
    else:
        idx_bk = tl.arange(0, BLOCK_MASK_BLOCK_K)
        mask_bk = idx_bk < mask_block_k

        ks = 0
        if KS_SEED is not None:
            ks = tl.load(
                KS_SEED + idx_b * stride_ks_seed_b + idx_bdst * stride_ks_seed_bdst,
            ).to(tl.int32)

        ALIGNED_BSRC = 1 << tl.floor(tl.log2(BSRC.to(tl.float64))).to(tl.int32)
        ALIGN_STEP = tl.cdiv(ALIGNED_BSRC, mask_block_k)

        # ALIGNED_BSRC = BSRC
        # ALIGN_STEP = 1

        indices = (
            tl.minimum(
                (
                    (MAX_BSRC * idx_group + (BSRC / mask_block_k * idx_bk)).to(tl.int32)
                    // ALIGN_STEP
                )
                * ALIGN_STEP,
                (MAX_BSRC * idx_group + BSRC).to(tl.int32),
            )
            + N_SINK_BLOCK
        )
        next_indices = (
            tl.minimum(
                (
                    (MAX_BSRC * idx_group + (BSRC / mask_block_k * (idx_bk + 1))).to(
                        tl.int32
                    )
                    // ALIGN_STEP
                )
                * ALIGN_STEP,
                (MAX_BSRC * idx_group + BSRC).to(tl.int32),
            )
            + N_SINK_BLOCK
        )
        group_sizes = tl.maximum(0, tl.minimum(BSRC, next_indices - indices)).to(
            tl.int32
        )
        if INDICES_SEED is not None:
            if ks == (mask_block_k * G):
                indices = tl.load(
                    INDICES_SEED
                    + idx_b * stride_indices_seed_b
                    + idx_bdst * stride_indices_seed_bdst
                    + (idx_group * mask_block_k + idx_bk) * stride_indices_seed_bk,
                    mask=mask_bk,
                    other=idx_group * MAX_BSRC,
                ).to(tl.int32)
                indices_next = tl.load(
                    INDICES_SEED
                    + idx_b * stride_indices_seed_b
                    + idx_bdst * stride_indices_seed_bdst
                    + (idx_group * mask_block_k + idx_bk + 1) * stride_indices_seed_bk,
                    mask=(
                        mask_bk
                        & (
                            (idx_group * mask_block_k + idx_bk + 1)
                            < (BLOCK_MASK_BLOCK_K * G)
                        )
                    ),
                    other=G * MAX_BSRC,
                ).to(tl.int32)
                indices_group_id = indices // MAX_BSRC
                indices_next_group_id = indices_next // MAX_BSRC
                group_sizes = tl.where(
                    indices_group_id == indices_next_group_id,
                    indices_next - indices,
                    indices_group_id * MAX_BSRC + BSRC - indices,
                ).to(tl.int32)
        if max_group_size_seed is not None:
            group_sizes = tl.minimum(
                group_sizes, max_group_size_seed.to(group_sizes.dtype)
            )

        if INDICES is not None:
            tl.store(
                INDICES
                + idx_b * stride_indices_b
                + idx_bdst * stride_indices_bdst
                + (idx_group * mask_block_k + idx_bk) * stride_indices_bk,
                value=indices,
                mask=mask_bk,
            )
        if GROUP_SIZE is not None:
            tl.store(
                GROUP_SIZE
                + idx_b * stride_group_size_b
                + idx_bdst * stride_group_size_bdst
                + (idx_group * mask_block_k + idx_bk) * stride_group_size_bk,
                value=group_sizes,
                mask=mask_bk,
            )

        if T_GROUP_SIZE is not None:
            tl.atomic_max(
                T_GROUP_SIZE
                + idx_b * stride_t_group_size_b
                + idx_bdst * stride_t_group_size_bdst,
                val=tl.max(group_sizes),
                # value = tl.minimum(
                #     tl.max(group_sizes),
                #     tl.maximum(tl.cdiv(BSRC, mask_block_k), 8)
                # )
            )
        if KS is not None:
            tl.atomic_add(
                KS + idx_b * stride_ks_b + idx_bdst * stride_ks_bdst,
                val=mask_block_k,
                # val = tl.sum((group_sizes > 0).to(tl.int32))
            )


@triton.jit
def hash_table_lookup_or_fail(
    TABLES,
    stride_tables_n,
    stride_tables_k,
    idx_n,
    idx_tsrc: tl.tensor,
    mask_tsrc: tl.tensor,
    TABLE_LEN,
):
    TABLE_CURRENT = TABLES + idx_n * stride_tables_n

    idx_hash_slot = idx_tsrc % TABLE_LEN

    # idx_hash_slot = idx_tsrc.to(tl.uint64)
    # idx_hash_slot ^= idx_hash_slot >> 8
    # # idx_hash_slot *= 0xff51afd7ed558ccd
    # # idx_hash_slot = idx_hash_slot >> 33
    # # idx_hash_slot *= 0xc4ceb9fe1a85ec53
    # # idx_hash_slot ^= idx_hash_slot >> 33
    # idx_hash_slot = idx_hash_slot.to(tl.int64) % TABLE_LEN

    offset_probe = tl.zeros_like(idx_tsrc).to(tl.int64)  # type: tl.tensor
    found_slot = tl.zeros_like(idx_tsrc).to(tl.int64) - 1

    offset_probe = tl.where(mask_tsrc, offset_probe, TABLE_LEN)

    if offset_probe.numel == 1:
        minimum_offset_prob = offset_probe
    else:
        minimum_offset_prob = tl.min(offset_probe)

    max_try = TABLE_LEN + 1

    while (minimum_offset_prob < TABLE_LEN) & (max_try > 0):
        max_try -= 1

        probing_target = (idx_hash_slot + offset_probe) % TABLE_LEN
        mask_probing = (found_slot == -1) & (offset_probe < TABLE_LEN) & mask_tsrc
        slots = tl.load(
            TABLE_CURRENT + probing_target * stride_tables_k,
            mask=mask_probing,
            other=65535,  # mimic full
        )

        # mask_empty_hit = (found_slot == -1) & ((slots == 65535) | (slots == 0xFFFFFFFFFFFFFFFF))
        # found_slot = tl.where(mask_empty_hit, probing_target, found_slot)
        # offset_probe = tl.where(mask_empty_hit, TABLE_LEN, offset_probe)

        mask_current_hit = (found_slot == -1) & (
            (slots.to(tl.uint64) >> 16) == (idx_tsrc.to(tl.uint64))
        )
        found_slot = tl.where(mask_current_hit, probing_target, found_slot)
        offset_probe = tl.where(mask_current_hit, TABLE_LEN, offset_probe)

        offset_probe = tl.where(found_slot == -1, offset_probe + 1, offset_probe)

        if offset_probe.numel == 1:
            minimum_offset_prob = offset_probe
        else:
            minimum_offset_prob = tl.min(offset_probe)

    return found_slot


@triton.jit
def load_tokens(
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_UPDATE: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    idx_bsz,
    idx_tsrc,
    idx_kv_head,
    idx_hid,
    mask_keys,
    BLOCK_SIZE_K,
):
    # DEBUG: to load nothing
    # mask_keys = mask_keys & False

    # tl.static_print(OFFLOAD_CACHE_METHOD)

    if not USING_PAGES:
        if USING_OFFLOAD_CACHE:
            original_mask_keys = mask_keys
            idx_cache = (
                idx_bsz.to(tl.int64) * OFFLOAD_CACHE_KV_HEAD + idx_kv_head.to(tl.int64)
            ).to(tl.int64)

            if OFFLOAD_CACHE_METHOD == "single_level":
                idx_bank_page = tl.load(
                    OFFLOAD_CACHE_K_TABLES
                    + idx_cache * stride_offload_cache_k_tables_n
                    + (idx_tsrc // BLOCK_SIZE_K).to(tl.int64)
                    * stride_offload_cache_k_tables_t,
                    mask=mask_keys,
                    other=65535,
                ).to(tl.uint16)
            elif OFFLOAD_CACHE_METHOD == "bloom":
                TABLE_LEN = OFFLOAD_CACHE_BUDGET * 1
                idx_table_entry = hash_table_lookup_or_fail(
                    OFFLOAD_CACHE_K_TABLES,
                    stride_offload_cache_k_tables_n,
                    stride_offload_cache_k_tables_t,
                    idx_cache,
                    (idx_tsrc // BLOCK_SIZE_K).to(tl.int64),
                    mask_keys,
                    TABLE_LEN,
                )
                # idx_table_entry = idx_table_entry * 0 - 1
                mask_table_entry = (
                    mask_keys & (idx_table_entry >= 0) & (idx_table_entry < TABLE_LEN)
                )
                tl.static_assert(OFFLOAD_CACHE_K_TABLES.dtype.element_ty != tl.uint16)
                idx_bank_page = tl.load(
                    OFFLOAD_CACHE_K_TABLES
                    + idx_cache * stride_offload_cache_k_tables_n
                    + idx_table_entry * stride_offload_cache_k_tables_t,
                    mask=mask_table_entry,
                    other=65535,
                )
                idx_entry_tsrc = idx_bank_page >> 16
                idx_bank_page = idx_bank_page & 0xFFFF
                idx_bank_page = idx_bank_page.to(tl.int64)
                idx_bank_page = tl.where(
                    idx_entry_tsrc == (idx_tsrc // BLOCK_SIZE_K), idx_bank_page, 65535
                )
                # if (tl.program_id(0) == 0) and (tl.program_id(1) == 0):
                #     tl.device_print('a', idx_entry_tsrc)
            else:
                tl.static_assert(False, f"Not supported {OFFLOAD_CACHE_METHOD}")

            mask_bank_hit = (idx_bank_page != 65535) & (
                idx_bank_page < OFFLOAD_CACHE_BUDGET
            )
            mask_keys = mask_keys & (~mask_bank_hit)

            # load from offload cache
            keys_from_cache = tl.load(
                OFFLOAD_CACHE_K_BANKS
                + idx_cache.to(tl.int64) * stride_offload_cache_k_banks_n
                + idx_bank_page.to(tl.int64) * stride_offload_cache_k_banks_page
                + (idx_tsrc % BLOCK_SIZE_K).to(tl.int64)
                * stride_offload_cache_k_banks_offset
                + idx_hid.to(tl.int64) * stride_offload_cache_k_banks_hid,
                mask=mask_bank_hit,
                other=0.0,
                # cache_modifier='.cs', # TODO: uncomment this
            )

            # num accessed
            tl.atomic_add(
                OFFLOAD_CACHE_COUNTERS
                + idx_cache * stride_offload_cache_counters_n
                + 0 * stride_offload_cache_counters_k,
                val=tl.sum(original_mask_keys.to(tl.int64)),
            )

            # num hits
            tl.atomic_add(
                OFFLOAD_CACHE_COUNTERS
                + idx_cache * stride_offload_cache_counters_n
                + 1 * stride_offload_cache_counters_k,
                val=tl.sum(mask_bank_hit.to(tl.int64)),
            )

            # num access per page
            if OFFLOAD_CACHE_K_BANK_STATS is not None:
                tl.atomic_add(
                    OFFLOAD_CACHE_K_BANK_STATS
                    + idx_cache * stride_offload_cache_k_bank_stats_n
                    + idx_bank_page * stride_offload_cache_k_bank_stats_page
                    + 0 * stride_offload_cache_k_bank_stats_k,
                    val=1,
                    mask=mask_bank_hit,
                )

        keys = tl.load(
            K
            + idx_bsz.to(tl.int64) * stride_k_bsz
            + idx_tsrc.to(tl.int64) * stride_k_tsrc
            + idx_kv_head.to(tl.int64) * stride_k_head
            + idx_hid.to(tl.int64) * stride_k_hid,
            mask=mask_keys,
            other=0.0,
            # cache_modifier='.cs', # TODO: uncomment this
        )

        if USING_OFFLOAD_CACHE:
            # merge keys and loaded cache
            keys = tl.where(mask_bank_hit, keys_from_cache.to(keys.dtype), keys)
            # update cache if there is uvm-loaded-keys

    else:
        tl.static_assert(not USING_OFFLOAD_CACHE)

        seq_len = tl.load(
            CACHE_SEQ_LENS + idx_bsz.to(tl.int64) * stride_cache_seq_lens_b,
        )
        tl.debug_barrier()
        mask_tsrc = idx_tsrc < seq_len
        tl.debug_barrier()
        ptrs = (
            BLOCK_TABLE
            + idx_bsz.to(tl.int64) * stride_block_table_bsz
            + (idx_tsrc // PAGE_SIZE).to(tl.int64) * stride_block_table_page
        )
        tl.debug_barrier()
        idx_page = tl.load(
            ptrs,
            mask=mask_tsrc,
            other=0,
        )
        offset_page = idx_tsrc % PAGE_SIZE

        keys = tl.load(
            K_CACHE
            + idx_page.to(tl.int64) * stride_k_cache_page
            + offset_page.to(tl.int64) * stride_k_cache_offset
            + idx_kv_head.to(tl.int64) * stride_k_cache_kv_head
            + idx_hid.to(tl.int64) * stride_k_cache_hid,
            mask=mask_keys,
            other=0.0,
        )

    if keys.dtype == tl.uint8:
        keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)
    if keys.dtype == tl.float8e5:
        keys = keys.to(tl.float16)

    return keys


@triton.jit
def masking_iteration_draft_cuda_dup_and_score_calc_score(
    idx_key_blocks,
    KEY_DUP: tl.constexpr,
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_bh,
    stride_q_g,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    VERTICAL_MASK,
    stride_vertical_mask_n,
    stride_vertical_mask_tsrc,
    DIAGONAL_MASK,
    stride_diagonal_mask_n,
    stride_diagonal_mask_tsrc,
    ACCESS_LOG_STORE_FROM,
    KEY_ACCESS_LOG,
    stride_key_access_log_b,
    stride_key_access_log_bdst,
    stride_key_access_log_t,
    KEY_ACCESS_COUNT,
    stride_key_access_count_b,
    stride_key_access_count_bdst,
    MAX_ACCESS_COUNT,
    BLOCK_ACCESS_LOG,
    stride_block_access_log_b,
    stride_block_access_log_bdst,
    stride_block_access_log_t,
    BLOCK_ACCESS_SCORE,
    stride_block_access_score_b,
    stride_block_access_score_bdst,
    stride_block_access_score_t,
    BLOCK_ACCESS_COUNT,
    stride_block_access_count_b,
    stride_block_access_count_bdst,
    MAX_BLOCK_ACCESS_COUNT,
    idx_b,
    idx_bdst,
    idx_tdst,
    mask_tdst,
    pos_tdst,
    idx_bk,  # for streamingLLM
    mask_key_blocks,
    sliding_window_size,
    BH: tl.constexpr,
    G: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    HID: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
    MASK_K: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    extend_window_size,
    extend_group_size,
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND,
    stride_q_ind_b,
    stride_q_ind_g,
    stride_q_ind_bdst,
    stride_q_ind_k,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    REDUCE_METHOD: tl.constexpr,
    NUM_CALIB: tl.constexpr = 32,
    EXTEND_BACKEND: tl.constexpr = DEFAULT_EXTEND_BACKEND,
):
    if BLOCK_ACCESS_LOG is not None:
        list_block_access = idx_key_blocks
        mask_block_access = mask_key_blocks & (
            list_block_access < tl.cdiv(MAX_TSRC, BLOCK_SIZE_K)
        )

        len_block_access = tl.sum(mask_block_access.to(tl.int32))
        block_access_location = tl.atomic_add(
            BLOCK_ACCESS_COUNT
            + idx_b * stride_block_access_count_b
            + (idx_bdst - ACCESS_LOG_STORE_FROM) * stride_block_access_count_bdst,
            val=len_block_access,
            mask=(idx_bdst - ACCESS_LOG_STORE_FROM) >= 0,
        )
        idx_block_access = (
            block_access_location + tl.cumsum(mask_block_access.to(tl.int32)) - 1
        ) % MAX_BLOCK_ACCESS_COUNT
        tl.store(
            BLOCK_ACCESS_LOG
            + idx_b * stride_block_access_log_b
            + (idx_bdst - ACCESS_LOG_STORE_FROM) * stride_block_access_log_bdst
            + idx_block_access * stride_block_access_log_t,
            mask=mask_block_access & ((idx_bdst - ACCESS_LOG_STORE_FROM) >= 0),
            value=list_block_access,
        )

    idx_tsrc = (
        (idx_key_blocks * BLOCK_SIZE_K)[:, None]
        + tl.arange(0, BLOCK_SIZE_K // BLOCK_STRIDE_K)[None, :] * BLOCK_STRIDE_K
        + BLOCK_STRIDE_K
        - 1
    )
    idx_tsrc = tl.ravel(idx_tsrc)
    idx_tsrc_grouped = idx_tsrc
    idx_group = idx_tsrc // MAX_TSRC
    idx_tsrc = idx_tsrc % MAX_TSRC
    idx_bsz = idx_b // BH
    idx_bh = idx_b % BH

    if KEY_ACCESS_LOG is not None:
        mask_access = tl.ravel(
            tl.broadcast_to(
                mask_key_blocks[:, None],
                BLOCK_BK * KEY_DUP,
                BLOCK_SIZE_K // BLOCK_STRIDE_K,
            )
        )
        len_access = tl.sum(mask_access.to(tl.int32))
        key_access_location = tl.atomic_add(
            KEY_ACCESS_COUNT
            + idx_b * stride_key_access_count_b
            + (idx_bdst - ACCESS_LOG_STORE_FROM) * stride_key_access_count_bdst,
            val=len_access,
            mask=((idx_bdst - ACCESS_LOG_STORE_FROM) >= 0),
        )
        idx_access = (
            key_access_location + tl.cumsum(mask_access.to(tl.int32)) - 1
        ) % MAX_ACCESS_COUNT
        # idx_access = tl.arange(0, BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K)
        tl.store(
            KEY_ACCESS_LOG
            + idx_b * stride_key_access_log_b
            + (idx_bdst - ACCESS_LOG_STORE_FROM) * stride_key_access_log_bdst
            + idx_access * stride_key_access_log_t,
            value=idx_tsrc_grouped,
            mask=mask_access & ((idx_bdst - ACCESS_LOG_STORE_FROM) >= 0),
            # eviction_policy='evict_first'
        )

    acc = tl.zeros(
        (
            BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
            BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K,
        ),
        dtype=tl.float32,
    )
    idx_hid = tl.arange(0, HID)
    mask_acc_keys = tl.zeros(
        (BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K,), dtype=tl.int1
    )
    for i_group in tl.range(0, G):
        queries = tl.load(
            Q
            + idx_bsz.to(tl.int64) * stride_q_bsz
            + idx_tdst[:, None].to(tl.int64) * stride_q_tdst
            + idx_bh.to(tl.int64) * stride_q_bh
            + i_group.to(tl.int64) * stride_q_g
            + idx_hid[None, :].to(tl.int64) * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0.0,
            # cache_modifier='.cs', # TODO: uncomment this (do not uncomment others)
            # eviction_policy='evict_last'
        )
        if queries.dtype == tl.float8e5:
            queries = queries.to(tl.float16)
        # queries = (idx_tdst[:, None] + idx_hid[None, :]).to(tl.float32)

        if queries.dtype == tl.uint8:
            queries = queries.to(tl.float8e5, bitcast=True).to(tl.float16)
        if G == 1:
            mask_keys = tl.broadcast_to(
                mask_key_blocks[:, None],
                BLOCK_BK * KEY_DUP,
                BLOCK_SIZE_K // BLOCK_STRIDE_K,
            )
            mask_keys = tl.ravel(mask_keys)[None, :]
            mask_keys = mask_keys & (idx_tsrc_grouped < MAX_TSRC)
        else:
            mask_keys = mask_key_blocks[:, None] & (idx_group == i_group).reshape(
                BLOCK_BK * KEY_DUP, BLOCK_SIZE_K // BLOCK_STRIDE_K
            )
            mask_keys = tl.ravel(mask_keys)[None, :]
        idx_head = idx_bh.to(tl.int64) * G + idx_group[None, :].to(tl.int64)
        idx_kv_head = idx_head // KV_HEAD_REPEAT

        if VERTICAL_MASK is not None:
            # NOTE: !!!!!!!!!!!! idx n should be calculated properly infuture
            vertial_mask = tl.load(
                VERTICAL_MASK
                + idx_b * stride_vertical_mask_n
                + idx_tsrc[None, :] * stride_vertical_mask_tsrc,
                mask=mask_keys,
                other=1,
            ).to(tl.int1)
            mask_keys = mask_keys & vertial_mask

        if DIAGONAL_MASK is not None:
            idx_tsrc_diag = idx_tsrc - (MAX_TDST - 1 - tl.max(idx_tdst))
            mask_tsrc_diag = (idx_tsrc_diag >= 0) & (idx_tsrc_diag < MAX_TSRC)
            diagonal_mask = tl.load(
                DIAGONAL_MASK
                + idx_b * stride_diagonal_mask_n
                + idx_tsrc_diag[None, :] * stride_diagonal_mask_tsrc,
                mask=mask_keys & mask_tsrc_diag,
                other=1,
            ).to(tl.int1)
            mask_keys = mask_keys & diagonal_mask

        mask_acc_keys = mask_acc_keys | tl.ravel(mask_keys)
        keys = load_tokens(
            K,
            stride_k_bsz,
            stride_k_tsrc,
            stride_k_head,
            stride_k_hid,
            USING_PAGES,
            PAGE_SIZE,
            K_CACHE,
            stride_k_cache_page,
            stride_k_cache_offset,
            stride_k_cache_kv_head,
            stride_k_cache_hid,
            BLOCK_TABLE,
            stride_block_table_bsz,
            stride_block_table_page,
            CACHE_SEQ_LENS,
            stride_cache_seq_lens_b,
            # offload cache args template
            USING_OFFLOAD_CACHE,
            OFFLOAD_CACHE_METHOD,
            OFFLOAD_CACHE_BUDGET,
            OFFLOAD_CACHE_KV_HEAD,
            True,
            OFFLOAD_CACHE_K_TABLES,
            stride_offload_cache_k_tables_n,
            stride_offload_cache_k_tables_t,
            OFFLOAD_CACHE_K_BANKS,
            stride_offload_cache_k_banks_n,
            stride_offload_cache_k_banks_page,
            stride_offload_cache_k_banks_offset,
            stride_offload_cache_k_banks_hid,
            OFFLOAD_CACHE_K_BANK_STATS,
            stride_offload_cache_k_bank_stats_n,
            stride_offload_cache_k_bank_stats_page,
            stride_offload_cache_k_bank_stats_k,
            OFFLOAD_CACHE_COUNTERS,
            stride_offload_cache_counters_n,
            stride_offload_cache_counters_k,
            idx_bsz,
            idx_tsrc[None, :],
            idx_kv_head,
            idx_hid[:, None],
            mask_keys,
            BLOCK_SIZE_K,
        )
        # keys = (idx_tsrc[None, :] + idx_hid[:, None]).to(tl.float32)
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)

        if USING_EXTEND:
            if EXTEND_BACKEND == "self_extend":
                if tl.min(pos_tdst) > (extend_window_size + NUM_CALIB // 2):
                    assert COS is not None
                    assert SIN is not None

                    # dynamic_group_size = tl.maximum(1.0, tl.math.ceil(tl.max(pos_tdst / 8192))).to(tl.int32)
                    dynamic_group_size = extend_group_size

                    idx_tsrc_calib = tl.maximum(
                        0, tl.min(pos_tdst) - (extend_window_size + NUM_CALIB // 2)
                    )
                    idx_tsrc_calib = idx_tsrc_calib + tl.arange(0, NUM_CALIB)
                    mask_tsrc_calib = idx_tsrc_calib < MAX_TSRC
                    keys_calib_old = tl.load(
                        K
                        + idx_bsz.to(tl.int64) * stride_k_bsz
                        + idx_tsrc_calib[None, :] * stride_k_tsrc
                        + idx_kv_head * stride_k_head
                        + idx_hid[:, None] * stride_k_hid,
                        mask=mask_tsrc_calib[None, :],
                        other=0,
                    )

                    keys_calib_new = adjust_rope(
                        keys_calib_old.trans(1, 0),
                        idx_tsrc_calib,
                        # idx_tsrc_calib // extend_group_size,
                        (idx_tsrc_calib / dynamic_group_size).to(tl.int32),
                        mask_tsrc_calib,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        NUM_CALIB,
                        HID,
                        NEED_APPLY_ROPE,
                    ).trans(1, 0)

                    old_tsrc = idx_tsrc
                    mask_tsrc_window = idx_tsrc >= (
                        tl.min(tl.where(mask_tdst, (pos_tdst - 1), 9999999))
                        - extend_window_size
                    )
                    new_tsrc = tl.where(
                        mask_tsrc_window,
                        old_tsrc,
                        # old_tsrc // extend_group_size
                        (old_tsrc / dynamic_group_size).to(tl.int32),
                    )

                    keys = keys.trans(1, 0)
                    keys = adjust_rope(
                        keys,
                        old_tsrc,
                        new_tsrc,
                        mask_keys,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K,
                        HID,
                        NEED_APPLY_ROPE,
                    ).to(keys.dtype)
                    keys = tl.trans(keys, 1, 0)
                    keys = (keys * mask_keys).to(keys.dtype)

                    old_tdst = pos_tdst - 1
                    # new_tdst = old_tdst // extend_group_size
                    new_tdst = (old_tdst / dynamic_group_size).to(tl.int32)

                    queries_grouped = adjust_rope(
                        queries,
                        old_tdst,
                        new_tdst,
                        mask_tdst,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
                        HID,
                        NEED_APPLY_ROPE,
                    ).to(queries.dtype)

                    t_calib_old = tl.dot(
                        (
                            queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries.dtype),
                        (
                            keys_calib_old.to(queries.dtype)
                            * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries.dtype),
                    )
                    t_calib_new = tl.dot(
                        (
                            queries_grouped
                            * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries_grouped.dtype),
                        (
                            keys_calib_new.to(queries_grouped.dtype)
                            * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries_grouped.dtype),
                    )

                    calibration = tl.sum(t_calib_new - t_calib_old, axis=-1) / NUM_CALIB

                    # calib_old_mean = tl.sum(t_calib_old, axis=-1) / NUM_CALIB
                    # calib_old_std = tl.sqrt(tl.sum(tl.extra.cuda.libdevice.pow(t_calib_old - calib_old_mean[:, None], 2), axis=-1) / NUM_CALIB)
                    # calib_new_mean = tl.sum(t_calib_new, axis=-1) / NUM_CALIB
                    # calib_new_std = tl.sqrt(tl.sum(tl.extra.cuda.libdevice.pow(t_calib_new - calib_new_mean[:, None], 2), axis=-1) / NUM_CALIB)

                    t_window = tl.dot(
                        (
                            queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries.dtype),
                        (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                            queries.dtype
                        ),
                    )

                    t_grouped = tl.dot(
                        (
                            queries_grouped
                            * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries_grouped.dtype),
                        (
                            keys.to(queries_grouped.dtype)
                            * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries_grouped.dtype),
                    )

                    # NOTE: this calibration trick is very important.
                    # > w/o std
                    t_grouped = t_grouped - calibration[:, None]
                    # > with std
                    # t_grouped = ((t_grouped - calib_new_mean[:, None]) / calib_new_std[:, None]) * calib_old_std[:, None] + calib_old_mean[:, None]

                    t = tl.where(
                        mask_tsrc_window[None, :],
                        t_window,
                        t_grouped,
                    ).to(tl.float32)
                else:
                    t = tl.dot(
                        (
                            queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries.dtype),
                        (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                            queries.dtype
                        ),
                        out_dtype=tl.float32,
                        allow_tf32=True,
                    ).to(tl.float32)
            elif EXTEND_BACKEND == "streaming":
                assert COS is not None
                assert SIN is not None

                # tl.static_print(idx_tsrc.shape, mask_keys.shape, idx_bk.shape)

                old_tsrc = idx_tsrc
                new_tsrc = tl.ravel(
                    (idx_bk * BLOCK_SIZE_K)[:, None]
                    + tl.arange(0, BLOCK_SIZE_K)[None, :]
                ) + tl.maximum(
                    0,
                    tl.min(tl.where(mask_tdst, pos_tdst - 1, 987654321))
                    - sliding_window_size
                    - MASK_K,  # tl.arange(0, BLOCK_TK)
                )
                new_tsrc = tl.where(mask_keys, new_tsrc, old_tsrc)
                new_tsrc = tl.ravel(new_tsrc)

                keys_adjusted = keys.trans(1, 0)
                keys_adjusted = adjust_rope(
                    keys_adjusted,
                    old_tsrc,
                    new_tsrc,
                    mask_keys,
                    idx_hid,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K,
                    HID,
                    NEED_APPLY_ROPE,
                ).to(keys.dtype)
                keys_adjusted = tl.trans(keys_adjusted, 1, 0)
                keys_adjusted = (keys_adjusted * mask_keys).to(keys_adjusted.dtype)

                t = tl.dot(
                    (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                        queries.dtype
                    ),
                    (
                        keys_adjusted.to(queries.dtype)
                        * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))
                    ).to(queries.dtype),
                ).to(tl.float32)
            else:
                raise Exception()
        else:
            if not USING_SPARQ:
                NUM_QUERIES: tl.constexpr = tl.constexpr(BLOCK_SIZE_Q // BLOCK_STRIDE_Q)
                if NUM_QUERIES < 16:
                    t = queries.reshape(NUM_QUERIES, HID, 1) * keys.reshape(
                        1, HID, BLOCK_BK * BLOCK_SIZE_K // BLOCK_STRIDE_K * KEY_DUP
                    )
                    t = tl.sum(t, axis=1)
                else:
                    # BQ=64, BSQ=2
                    # 4090: 20 ms, A100: 34.81ms

                    t = tl.dot(
                        (
                            queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))
                        ).to(queries.dtype),
                        (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                            queries.dtype
                        ),
                        out_dtype=tl.float32,
                        allow_tf32=True,
                    )

                    # 4090: 16 ms, A100: 31.97 ms
                    # scale = 256 / tl.max(tl.abs(queries))
                    # t = tl.dot(
                    #     tl.clamp(queries.to(tl.float32) * scale, -127, 127).to(tl.int8),
                    #     tl.clamp(keys.to(tl.float32) * scale, -127, 127).to(tl.int8),
                    #     out_dtype=tl.int32,
                    # ).to(tl.float32) / (scale * scale)
                    # t = t.to(tl.float32)

                    # 4090: 10.13 ms, A100: 19.18704981 ms
                    # t = tl.zeros_like(acc) + tl.sum(keys) + tl.sum(queries)
            else:
                idx_sparq_hid = tl.arange(0, SPARQ_HID)

                idx_sparq_hid = tl.load(
                    Q_IND
                    + idx_b * stride_q_ind_b
                    + i_group * stride_q_ind_g
                    + idx_bdst * stride_q_ind_bdst
                    + idx_sparq_hid * stride_q_ind_k
                )

                q_sparq = tl.load(
                    Q
                    + idx_bsz * stride_q_bsz
                    + idx_tdst[:, None] * stride_q_tdst
                    + idx_bh * stride_q_bh
                    + i_group * stride_q_g
                    + idx_sparq_hid[None, :] * stride_q_hid,
                    mask=mask_tdst[:, None],
                    other=0,
                )
                k_sparq = tl.load(
                    K
                    + idx_b * stride_k_bsz
                    + idx_tsrc[None, :] * stride_k_tsrc
                    + (idx_bh * BH + idx_group[None, :]) * stride_k_head
                    + idx_sparq_hid[:, None] * stride_k_hid,
                    mask=mask_keys,
                    other=0,
                )

                t = tl.dot(
                    (q_sparq * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                        q_sparq.dtype
                    ),
                    (k_sparq.to(q_sparq.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                        q_sparq.dtype
                    ),
                ).to(tl.float32)
        acc += t.to(acc.dtype)
        # acc += tl.sum(queries)
        # acc += tl.sum(keys)
    if IS_CAUSAL:
        acc = tl.where(
            (
                (acc == 0.0)
                | (idx_tsrc[None, :] > (pos_tdst - sliding_window_size - 1)[:, None])
                | False
            ),
            -32000.0 if REDUCE_METHOD == "max" else 32000.0,
            acc,
        )
    else:
        acc = tl.where(
            (
                (acc == 0.0)
                |
                # (idx_tsrc[None, :] > (pos_tdst - sliding_window_size - 1)[:, None]) |
                False
            ),
            -32000.0 if REDUCE_METHOD == "max" else 32000.0,
            acc,
        )
    if REDUCE_METHOD == "max":
        acc = tl.where(mask_acc_keys[None, :] & mask_tdst[:, None], acc, float("-inf"))
    elif REDUCE_METHOD == "min":
        acc = tl.where(mask_acc_keys[None, :] & mask_tdst[:, None], acc, float("inf"))
    else:
        raise Exception()
    scores = tl.reshape(
        acc,
        (
            BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
            BLOCK_BK * KEY_DUP,
            BLOCK_SIZE_K // BLOCK_STRIDE_K,
        ),
    )
    if REDUCE_METHOD == "max":
        scores = tl.max(
            scores,
            axis=0,
        )
        scores = tl.max(
            scores,
            axis=-1,
        )
    elif REDUCE_METHOD == "min":
        scores = tl.min(
            scores,
            axis=0,
        )
        scores = tl.min(
            scores,
            axis=-1,
        )
    else:
        raise Exception()
    scores = tl.where(mask_key_blocks, scores, float("-inf"))

    if BLOCK_ACCESS_LOG is not None:
        if BLOCK_ACCESS_SCORE is not None:
            if REDUCE_METHOD == "max":
                checkout_scores = tl.where(mask_key_blocks, -scores, float("-inf"))
            elif REDUCE_METHOD == "min":
                checkout_scores = scores
            tl.store(
                BLOCK_ACCESS_SCORE
                + idx_b * stride_block_access_score_b
                + (idx_bdst - ACCESS_LOG_STORE_FROM) * stride_block_access_score_bdst
                + idx_block_access * stride_block_access_score_t,
                mask=mask_block_access & ((idx_bdst - ACCESS_LOG_STORE_FROM) >= 0),
                value=checkout_scores,
            )

    return scores


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16),
#     ],
#     key=[
#         'max_group_size',
#         'i_iteration',
#         'BLOCK_BK'
#     ],
#     restore_value=[
#         'DUPPED_INDICES',
#         'DUPPED_GROUP_SIZE',
#         'SCORES',
#         'T_GROUP_SIZE',
#     ]
# )
@triton.jit
def masking_iteration_draft_cuda_dup_and_score(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_bh,
    stride_q_g,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    VERTICAL_MASK,
    stride_vertical_mask_n,
    stride_vertical_mask_tsrc,
    DIAGONAL_MASK,
    stride_diagonal_mask_n,
    stride_diagonal_mask_tsrc,
    ACCESS_LOG_STORE_FROM,
    KEY_ACCESS_LOG,
    stride_key_access_log_b,
    stride_key_access_log_bdst,
    stride_key_access_log_t,
    KEY_ACCESS_COUNT,
    stride_key_access_count_b,
    stride_key_access_count_bdst,
    MAX_ACCESS_COUNT,
    BLOCK_ACCESS_LOG,
    stride_block_access_log_b,
    stride_block_access_log_bdst,
    stride_block_access_log_t,
    BLOCK_ACCESS_SCORE,
    stride_block_access_score_b,
    stride_block_access_score_bdst,
    stride_block_access_score_t,
    BLOCK_ACCESS_COUNT,
    stride_block_access_count_b,
    stride_block_access_count_bdst,
    MAX_BLOCK_ACCESS_COUNT,
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS,
    stride_ks_b,
    stride_ks_bdst,
    GROUP_SIZE,
    stride_group_size_b,
    stride_group_size_bdst,
    stride_group_size_bk,
    DUPPED_INDICES,
    stride_dupped_indices_b,
    stride_dupped_indices_bdst,
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZE,
    stride_dupped_group_size_b,
    stride_dupped_group_size_bdst,
    stride_dupped_group_size_bk,
    SCORES,
    stride_scores_b,
    stride_scores_bdst,
    stride_scores_bk,
    SCORES_FINAL,
    stride_scores_final_b,
    stride_scores_final_bdst,
    stride_scores_final_bk,
    SCORES_CACHED: tl.constexpr,
    T_GROUP_SIZE,
    stride_t_group_size_b,
    stride_t_group_size_bdst,
    INDICES_TDST,
    stride_indices_tdst_t,
    mask_k,
    sliding_window_size,
    BH: tl.constexpr,
    G: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    BK,
    HID: tl.constexpr,
    RAND_SEED,
    SAMPLE_METHOD: tl.constexpr,
    BRANCH_METHOD: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    extend_window_size,
    extend_group_size,
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND,
    stride_q_ind_b,
    stride_q_ind_g,
    stride_q_ind_bdst,
    stride_q_ind_k,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_bsz,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    max_group_size,  # just for autotune
    i_iteration,  # just for autotune
    pid_0=None,
    pid_1=None,
    pid_2=None,
):
    if pid_2 is None:
        pid_b = tl.program_id(2)
    else:
        pid_b = pid_2

    if pid_1 is None:
        pid_bdst = tl.program_id(1)
    else:
        pid_bdst = pid_1

    if pid_0 is None:
        pid_bbk = tl.program_id(0)
    else:
        pid_bbk = pid_0

    idx_b = pid_b
    idx_bdst = pid_bdst

    idx_tdst = (
        idx_bdst * BLOCK_SIZE_Q
        + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
        + (tl.minimum(MAX_TDST, BLOCK_STRIDE_Q) - 1)
    )
    # idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.random.randint(idx_b * 131072 * BLOCK_SIZE_Q + idx_bdst * BLOCK_SIZE_Q, tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q)).to(tl.int32) % BLOCK_SIZE_Q
    # idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) + (BLOCK_SIZE_Q - BLOCK_SIZE_Q // BLOCK_STRIDE_Q)
    idx_tdst_no_proj = idx_tdst
    mask_tdst = idx_tdst < MAX_TDST
    if INDICES_TDST is not None:
        idx_tdst = tl.load(
            INDICES_TDST + idx_tdst.to(tl.int64) * stride_indices_tdst_t,
            mask=mask_tdst,
            other=MAX_TDST,
            cache_modifier=DEFAULT_CACHE_MODIFIER,
        ).to(tl.int64)

    idx_bk = pid_bbk * BLOCK_BK + tl.arange(0, BLOCK_BK)
    mask_bk = idx_bk < (BK * G)
    idx_bk_dup = pid_bbk * BLOCK_BK * 2 + tl.arange(0, BLOCK_BK * 2)
    mask_bk_dup = idx_bk_dup < (BK * 2 * G)
    idx_n = idx_b * G + tl.arange(0, G)

    mask_block_k = tl.cdiv(mask_k, BLOCK_SIZE_K)
    if IS_CAUSAL:
        pos_tdst = tl.load(
            POS + (idx_b // BH) * stride_pos_bsz + idx_tdst_no_proj * stride_pos_tdst,
            mask=mask_tdst,
            other=0,
            cache_modifier=DEFAULT_CACHE_MODIFIER,
        )
    else:
        pos_tdst = tl.full(
            (BLOCK_SIZE_Q // BLOCK_STRIDE_Q,), value=MAX_TSRC, dtype=tl.int64
        )
    TSRC = tl.max(pos_tdst)
    TSRC = tl.maximum(0, TSRC - sliding_window_size)
    BSRC = tl.cdiv(TSRC, BLOCK_SIZE_K)
    # MAX_BSRC = tl.cdiv(MAX_TSRC, BLOCK_SIZE_K)

    if TSRC <= mask_k:
        return

    t_group_size = tl.load(
        T_GROUP_SIZE
        + idx_b * stride_t_group_size_b
        + idx_bdst * stride_t_group_size_bdst,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    if t_group_size <= 1.0:
        return

    # int[BLOCK_BK]
    indices = tl.load(
        INDICES
        + idx_b * stride_indices_b
        + idx_bdst * stride_indices_bdst
        + idx_bk * stride_indices_bk,
        mask=mask_bk,
        other=0,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )

    # int[BLOCK_BK]
    group_sizes = tl.load(
        GROUP_SIZE
        + idx_b * stride_group_size_b
        + idx_bdst * stride_group_size_bdst
        + idx_bk * stride_group_size_bk,
        mask=mask_bk,
        other=0,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )

    # int[BLOCK_BK * 2]
    dupped_indices = tl.reshape(
        tl.join(indices, indices),
        (BLOCK_BK * 2,),
    )
    dupped_group_sizes = tl.reshape(tl.join(group_sizes, group_sizes), (BLOCK_BK * 2,))
    if BRANCH_METHOD == "half":
        dupped_indices = tl.where(
            (tl.arange(0, BLOCK_BK * 2) % 2) == 0,
            dupped_indices,
            (dupped_indices + dupped_group_sizes * 0.5).to(tl.int32),
        )
    elif BRANCH_METHOD == "random":
        dupped_indices = tl.where(
            (tl.arange(0, BLOCK_BK * 2) % 2) == 0,
            dupped_indices,
            tl.where(
                dupped_group_sizes == 0,
                dupped_indices,
                tl.maximum(
                    dupped_indices + 1,
                    dupped_indices
                    + dupped_group_sizes * 0.5
                    + dupped_group_sizes
                    * (
                        0.2
                        * tl.random.rand(
                            RAND_SEED,
                            tl.arange(0, BLOCK_BK * 2)
                            + tl.program_id(0) * 7
                            + tl.program_id(1) * 53
                            + tl.program_id(2) * 157,
                        )
                        * 0.99
                        - 0.1
                    ),
                ).to(tl.int32),
            ),
        )
    else:
        raise Exception(BRANCH_METHOD)
    flipped_dupped_indices = tl.reshape(
        tl.flip(
            tl.reshape(dupped_indices, (BLOCK_BK, 2)),
        ),
        (BLOCK_BK * 2),
    )
    dupped_group_sizes = tl.where(
        (tl.arange(0, BLOCK_BK * 2) % 2) == 0,
        flipped_dupped_indices - dupped_indices,
        flipped_dupped_indices + dupped_group_sizes - dupped_indices,
    )
    dupped_mask = (dupped_group_sizes > 0) & mask_bk_dup

    dupped_indices_for_keys = dupped_indices
    if SAMPLE_METHOD == "random":
        offsets = tl.where(
            dupped_group_sizes > 4,
            0,
            (
                tl.randint(
                    RAND_SEED,
                    dupped_indices
                    + tl.program_id(0) * 31
                    + tl.program_id(1) * 7
                    + tl.program_id(2) * 1371,
                )
                % dupped_group_sizes.to(tl.uint32)
            ).to(tl.int32),
        )
        dupped_indices_for_keys += offsets
    elif SAMPLE_METHOD == "last":
        dupped_indices_for_keys = dupped_indices + tl.where(
            dupped_group_sizes == 0,
            0,
            dupped_group_sizes - 1,
        )
    elif SAMPLE_METHOD == "center":
        dupped_indices_for_keys = dupped_indices + tl.maximum(
            0, dupped_group_sizes // 2
        )
    elif SAMPLE_METHOD == "position":
        dupped_indices_for_keys = dupped_indices + tl.maximum(
            0,
            tl.extra.cuda.libdevice.round(
                dupped_group_sizes * EXPERIMENTAL_SAMPLING_POSITION
            ).to(tl.int32),
        )
    elif SAMPLE_METHOD == "oracle":
        # NOTE: perform linear scan inside of the chunk, this will cost O(T^2)
        dupped_indices_for_keys_start = dupped_indices_for_keys
        dupped_indices_for_keys_end = dupped_indices_for_keys + tl.maximum(
            dupped_group_sizes - 1, 0
        )
        max_scores = tl.zeros((BLOCK_BK * 2,), dtype=tl.bfloat16) - 32000.0
        for i_shift in range(0, tl.cdiv(BSRC, mask_block_k)):
            t_dupped_indices_for_keys = tl.where(
                i_shift < dupped_group_sizes,
                dupped_indices_for_keys_start + i_shift,
                dupped_indices_for_keys_end,
            ).to(tl.int32)
            t_scores = masking_iteration_draft_cuda_dup_and_score_calc_score(
                t_dupped_indices_for_keys,
                Q,
                stride_q_bsz,
                stride_q_tdst,
                stride_q_bh,
                stride_q_g,
                stride_q_hid,
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head,
                stride_k_hid,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                VERTICAL_MASK,
                stride_vertical_mask_n,
                stride_vertical_mask_tsrc,
                DIAGONAL_MASK,
                stride_diagonal_mask_n,
                stride_diagonal_mask_tsrc,
                ACCESS_LOG_STORE_FROM,
                KEY_ACCESS_LOG,
                stride_key_access_log_b,
                stride_key_access_log_bdst,
                stride_key_access_log_t,
                KEY_ACCESS_COUNT,
                stride_key_access_count_b,
                stride_key_access_count_bdst,
                MAX_ACCESS_COUNT,
                BLOCK_ACCESS_LOG,
                stride_block_access_log_b,
                stride_block_access_log_bdst,
                stride_block_access_log_t,
                BLOCK_ACCESS_SCORE,
                stride_block_access_score_b,
                stride_block_access_score_bdst,
                stride_block_access_score_t,
                BLOCK_ACCESS_COUNT,
                stride_block_access_count_b,
                stride_block_access_count_bdst,
                MAX_BLOCK_ACCESS_COUNT,
                idx_b,
                idx_bdst,
                idx_tdst,
                mask_tdst,
                pos_tdst,
                idx_bk_dup,
                dupped_mask,
                sliding_window_size,
                BH,
                G,
                MAX_TDST,
                MAX_TSRC,
                HID,
                KV_HEAD_REPEAT,
                mask_k,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                extend_window_size,
                extend_group_size,
                USING_SPARQ,
                SPARQ_HID,
                Q_IND,
                stride_q_ind_b,
                stride_q_ind_g,
                stride_q_ind_bdst,
                stride_q_ind_k,
                # paged attention args template
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_bsz,
                # offload cache args template
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_METHOD,
                OFFLOAD_CACHE_BUDGET,
                OFFLOAD_CACHE_KV_HEAD,
                OFFLOAD_CACHE_K_TABLES,
                stride_offload_cache_k_tables_n,
                stride_offload_cache_k_tables_t,
                OFFLOAD_CACHE_K_BANKS,
                stride_offload_cache_k_banks_n,
                stride_offload_cache_k_banks_page,
                stride_offload_cache_k_banks_offset,
                stride_offload_cache_k_banks_hid,
                OFFLOAD_CACHE_K_BANK_STATS,
                stride_offload_cache_k_bank_stats_n,
                stride_offload_cache_k_bank_stats_page,
                stride_offload_cache_k_bank_stats_k,
                OFFLOAD_CACHE_COUNTERS,
                stride_offload_cache_counters_n,
                stride_offload_cache_counters_k,
                IS_CAUSAL,
                BLOCK_SIZE_Q,
                BLOCK_STRIDE_Q,
                BLOCK_SIZE_K,
                BLOCK_STRIDE_K,
                BLOCK_BK,
                "max",
            )
            dupped_indices_for_keys = tl.where(
                t_scores > max_scores,
                t_dupped_indices_for_keys,
                dupped_indices_for_keys,
            )
            max_scores = tl.minimum(max_scores, t_scores)
    else:
        # this should be first
        assert SAMPLE_METHOD == "first"

    if SCORES_CACHED:
        if SAMPLE_METHOD == "first":
            _, indices_to_sample = dupped_indices_for_keys.reshape(BLOCK_BK, 2).split()
            _, mask_to_sample = dupped_mask.reshape(BLOCK_BK, 2).split()
        elif SAMPLE_METHOD == "last":
            indices_to_sample, _ = dupped_indices_for_keys.reshape(BLOCK_BK, 2).split()
            mask_to_sample, _ = dupped_mask.reshape(BLOCK_BK, 2).split()
        else:
            raise Exception()

        # t1 = indices_to_sample.to(tl.uint16).to(tl.uint32)
        # t2 = mask_to_sample.to(tl.int1)
        # t3 = tl.arange(0, BLOCK_BK).to(tl.uint16).to(tl.uint32)
        # # t2 (1bit) | -- t3 (15bit) -- | -- t1 (16bit) --
        # t = (t2 << 31) | ((t3 << 17) >> 1) | t1

        # # _, indices_to_sample_sorted = tl_argsort(cached_scores, indices_to_sample, 0, False)
        # # _, mask_to_sample_sorted = tl_argsort(cached_scores, mask_to_sample.to(tl.int32), 0, False)
        # # _, mapping = tl_argsort(cached_scores, tl.arange(0, BLOCK_BK), 0, False)

        # _, t_sorted = tl_argsort(cached_scores, t, 0, False)
        # mask_to_sample_sorted = (t_sorted >> 31)
        # mapping = ((t_sorted << 1) >> 17).to(tl.int32)
        # indices_to_sample_sorted = ((t_sorted << 16) >> 16).to(tl.int32)

        # indices_to_sample_sorted, indices_to_not_sample_sorted = \
        #     indices_to_sample_sorted\
        #         .reshape(2, BLOCK_BK // 2)\
        #         .trans(1, 0)\
        #         .split()

        # mask_to_sample_sorted, mask_to_not_sample = \
        #     mask_to_sample_sorted\
        #         .reshape(2, BLOCK_BK // 2)\
        #         .trans(1, 0)\
        #         .split()
        # mask_to_sample_sorted = mask_to_sample_sorted.to(tl.int1)

        # indices_to_sample = indices_to_sample_sorted
        # mask_to_sample = mask_to_sample_sorted

        scores_sampled = masking_iteration_draft_cuda_dup_and_score_calc_score(
            indices_to_sample,
            1,
            Q,
            stride_q_bsz,
            stride_q_tdst,
            stride_q_bh,
            stride_q_g,
            stride_q_hid,
            K,
            stride_k_bsz,
            stride_k_tsrc,
            stride_k_head,
            stride_k_hid,
            COS,
            stride_cos_t,
            stride_cos_hid,
            SIN,
            stride_sin_t,
            stride_sin_hid,
            VERTICAL_MASK,
            stride_vertical_mask_n,
            stride_vertical_mask_tsrc,
            DIAGONAL_MASK,
            stride_diagonal_mask_n,
            stride_diagonal_mask_tsrc,
            ACCESS_LOG_STORE_FROM,
            KEY_ACCESS_LOG,
            stride_key_access_log_b,
            stride_key_access_log_bdst,
            stride_key_access_log_t,
            KEY_ACCESS_COUNT,
            stride_key_access_count_b,
            stride_key_access_count_bdst,
            MAX_ACCESS_COUNT,
            BLOCK_ACCESS_LOG,
            stride_block_access_log_b,
            stride_block_access_log_bdst,
            stride_block_access_log_t,
            BLOCK_ACCESS_SCORE,
            stride_block_access_score_b,
            stride_block_access_score_bdst,
            stride_block_access_score_t,
            BLOCK_ACCESS_COUNT,
            stride_block_access_count_b,
            stride_block_access_count_bdst,
            MAX_BLOCK_ACCESS_COUNT,
            idx_b,
            idx_bdst,
            idx_tdst,
            mask_tdst,
            pos_tdst,
            idx_bk,
            mask_to_sample,
            sliding_window_size,
            BH,
            G,
            MAX_TDST,
            MAX_TSRC,
            HID,
            KV_HEAD_REPEAT,
            mask_k,
            USING_EXTEND,
            NEED_APPLY_ROPE,
            extend_window_size,
            extend_group_size,
            USING_SPARQ,
            SPARQ_HID,
            Q_IND,
            stride_q_ind_b,
            stride_q_ind_g,
            stride_q_ind_bdst,
            stride_q_ind_k,
            # paged attention args template
            USING_PAGES,
            PAGE_SIZE,
            K_CACHE,
            stride_k_cache_page,
            stride_k_cache_offset,
            stride_k_cache_kv_head,
            stride_k_cache_hid,
            V_CACHE,
            stride_v_cache_page,
            stride_v_cache_offset,
            stride_v_cache_kv_head,
            stride_v_cache_hid,
            BLOCK_TABLE,
            stride_block_table_bsz,
            stride_block_table_page,
            CACHE_SEQ_LENS,
            stride_cache_seq_lens_bsz,
            # offload cache args template
            USING_OFFLOAD_CACHE,
            OFFLOAD_CACHE_METHOD,
            OFFLOAD_CACHE_BUDGET,
            OFFLOAD_CACHE_KV_HEAD,
            OFFLOAD_CACHE_K_TABLES,
            stride_offload_cache_k_tables_n,
            stride_offload_cache_k_tables_t,
            OFFLOAD_CACHE_K_BANKS,
            stride_offload_cache_k_banks_n,
            stride_offload_cache_k_banks_page,
            stride_offload_cache_k_banks_offset,
            stride_offload_cache_k_banks_hid,
            OFFLOAD_CACHE_K_BANK_STATS,
            stride_offload_cache_k_bank_stats_n,
            stride_offload_cache_k_bank_stats_page,
            stride_offload_cache_k_bank_stats_k,
            OFFLOAD_CACHE_COUNTERS,
            stride_offload_cache_counters_n,
            stride_offload_cache_counters_k,
            IS_CAUSAL,
            BLOCK_SIZE_Q,
            BLOCK_STRIDE_Q,
            BLOCK_SIZE_K,
            BLOCK_STRIDE_K,
            BLOCK_BK,
            # BLOCK_BK // 2,
            "max",
        )

        # scores_not_sampled = tl.full((BLOCK_BK // 2,), float('-inf'), dtype=scores_sampled.dtype)

        # scores_sorted = tl.join(scores_sampled, scores_not_sampled)\
        #     .trans(1, 0)\
        #     .reshape(BLOCK_BK)

        # _, scores_sampled = tl_argsort(mapping, scores_sorted.to(tl.float32).to(tl.int32, bitcast=True), 0, False)
        # scores_sampled = scores_sampled.to(tl.float32, bitcast=True)

        cached_scores = tl.load(
            SCORES_FINAL
            + idx_b * stride_scores_final_b
            + idx_bdst * stride_scores_final_bdst
            + idx_bk * stride_scores_final_bk,
            mask=mask_bk,
            cache_modifier=DEFAULT_CACHE_MODIFIER,
        )

        if SAMPLE_METHOD == "first":
            scores = tl.join(
                cached_scores.to(SCORES.dtype.element_ty),
                scores_sampled.to(SCORES.dtype.element_ty),
            ).reshape(BLOCK_BK * 2)
        elif SAMPLE_METHOD == "last":
            scores = tl.join(
                scores_sampled.to(SCORES.dtype.element_ty),
                cached_scores.to(SCORES.dtype.element_ty),
            ).reshape(BLOCK_BK * 2)
        else:
            raise Exception()
    else:
        indices_to_sample = dupped_indices_for_keys
        mask_to_sample = dupped_mask

        scores_sampled = masking_iteration_draft_cuda_dup_and_score_calc_score(
            indices_to_sample,
            2,
            Q,
            stride_q_bsz,
            stride_q_tdst,
            stride_q_bh,
            stride_q_g,
            stride_q_hid,
            K,
            stride_k_bsz,
            stride_k_tsrc,
            stride_k_head,
            stride_k_hid,
            COS,
            stride_cos_t,
            stride_cos_hid,
            SIN,
            stride_sin_t,
            stride_sin_hid,
            VERTICAL_MASK,
            stride_vertical_mask_n,
            stride_vertical_mask_tsrc,
            DIAGONAL_MASK,
            stride_diagonal_mask_n,
            stride_diagonal_mask_tsrc,
            ACCESS_LOG_STORE_FROM,
            KEY_ACCESS_LOG,
            stride_key_access_log_b,
            stride_key_access_log_bdst,
            stride_key_access_log_t,
            KEY_ACCESS_COUNT,
            stride_key_access_count_b,
            stride_key_access_count_bdst,
            MAX_ACCESS_COUNT,
            BLOCK_ACCESS_LOG,
            stride_block_access_log_b,
            stride_block_access_log_bdst,
            stride_block_access_log_t,
            BLOCK_ACCESS_SCORE,
            stride_block_access_score_b,
            stride_block_access_score_bdst,
            stride_block_access_score_t,
            BLOCK_ACCESS_COUNT,
            stride_block_access_count_b,
            stride_block_access_count_bdst,
            MAX_BLOCK_ACCESS_COUNT,
            idx_b,
            idx_bdst,
            idx_tdst,
            mask_tdst,
            pos_tdst,
            idx_bk_dup,
            mask_to_sample,
            sliding_window_size,
            BH,
            G,
            MAX_TDST,
            MAX_TSRC,
            HID,
            KV_HEAD_REPEAT,
            mask_k,
            USING_EXTEND,
            NEED_APPLY_ROPE,
            extend_window_size,
            extend_group_size,
            USING_SPARQ,
            SPARQ_HID,
            Q_IND,
            stride_q_ind_b,
            stride_q_ind_g,
            stride_q_ind_bdst,
            stride_q_ind_k,
            # paged attention args template
            USING_PAGES,
            PAGE_SIZE,
            K_CACHE,
            stride_k_cache_page,
            stride_k_cache_offset,
            stride_k_cache_kv_head,
            stride_k_cache_hid,
            V_CACHE,
            stride_v_cache_page,
            stride_v_cache_offset,
            stride_v_cache_kv_head,
            stride_v_cache_hid,
            BLOCK_TABLE,
            stride_block_table_bsz,
            stride_block_table_page,
            CACHE_SEQ_LENS,
            stride_cache_seq_lens_bsz,
            # offload cache args template
            USING_OFFLOAD_CACHE,
            OFFLOAD_CACHE_METHOD,
            OFFLOAD_CACHE_BUDGET,
            OFFLOAD_CACHE_KV_HEAD,
            OFFLOAD_CACHE_K_TABLES,
            stride_offload_cache_k_tables_n,
            stride_offload_cache_k_tables_t,
            OFFLOAD_CACHE_K_BANKS,
            stride_offload_cache_k_banks_n,
            stride_offload_cache_k_banks_page,
            stride_offload_cache_k_banks_offset,
            stride_offload_cache_k_banks_hid,
            OFFLOAD_CACHE_K_BANK_STATS,
            stride_offload_cache_k_bank_stats_n,
            stride_offload_cache_k_bank_stats_page,
            stride_offload_cache_k_bank_stats_k,
            OFFLOAD_CACHE_COUNTERS,
            stride_offload_cache_counters_n,
            stride_offload_cache_counters_k,
            IS_CAUSAL,
            BLOCK_SIZE_Q,
            BLOCK_STRIDE_Q,
            BLOCK_SIZE_K,
            BLOCK_STRIDE_K,
            BLOCK_BK,
            "max",
        )
        scores = scores_sampled.to(SCORES.dtype.element_ty)

    tl.store(
        SCORES
        + idx_b * stride_scores_b
        + idx_bdst * stride_scores_bdst
        + idx_bk_dup * stride_scores_bk,
        value=scores,
        mask=mask_bk_dup,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        DUPPED_INDICES
        + idx_b * stride_dupped_indices_b
        + idx_bdst * stride_dupped_indices_bdst
        + idx_bk_dup * stride_dupped_indices_bk,
        value=dupped_indices,
        mask=mask_bk_dup,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        DUPPED_GROUP_SIZE
        + idx_b * stride_dupped_group_size_b
        + idx_bdst * stride_dupped_group_size_bdst
        + idx_bk_dup * stride_dupped_group_size_bk,
        value=dupped_group_sizes,
        mask=mask_bk_dup,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )


@triton.jit
def masking_iteration_draft_cuda_gather(
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    GROUP_SIZES,
    stride_group_sizes_b,
    stride_group_sizes_bdst,
    stride_group_sizes_bk,
    SCORES_FINAL,
    stride_scores_final_b,
    stride_scores_final_bdst,
    stride_scores_final_bk,
    DUPPED_INDICES,
    stride_dupped_indices_b,
    stride_dupped_indices_bdst,
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZE,
    stride_dupped_group_size_b,
    stride_dupped_group_size_bdst,
    stride_dupped_group_size_bk,
    SCORES,
    stride_scores_b,
    stride_scores_bdst,
    stride_scores_bk,
    TOPK_INDICES,
    stride_topk_indices_b,
    stride_topk_indices_bdst,
    stride_topk_indices_bk,
    T_GROUP_SIZE,
    stride_t_group_size_b,
    stride_t_group_size_bdst,
    G: tl.constexpr,
    BK,
    BLOCK_BK: tl.constexpr,
    pid_0=None,
    pid_1=None,
    pid_2=None,
):
    if pid_0 is not None:
        pid_b = pid_2
        pid_bdst = pid_1
        pid_bk = pid_0
    else:
        pid_b = tl.program_id(2)
        pid_bdst = tl.program_id(1)
        pid_bk = tl.program_id(0)

    idx_b = pid_b
    idx_bdst = pid_bdst
    idx_bk = pid_bk * BLOCK_BK + tl.arange(0, BLOCK_BK)
    mask_bk = idx_bk < (BK * G)

    t_group_size = tl.load(
        T_GROUP_SIZE
        + idx_b * stride_t_group_size_b
        + idx_bdst * stride_t_group_size_bdst,
    )
    if t_group_size <= 1.0:
        return

    topk_indices = tl.load(
        TOPK_INDICES
        + idx_b * stride_topk_indices_b
        + idx_bdst * stride_topk_indices_bdst
        + idx_bk * stride_topk_indices_bk,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )

    dupped_indices = tl.load(
        DUPPED_INDICES
        + idx_b * stride_dupped_indices_b
        + idx_bdst * stride_dupped_indices_bdst
        + topk_indices * stride_dupped_indices_bk,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    dupped_group_size = tl.load(
        DUPPED_GROUP_SIZE
        + idx_b * stride_dupped_group_size_b
        + idx_bdst * stride_dupped_group_size_bdst
        + topk_indices * stride_dupped_group_size_bk,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    scores = tl.load(
        SCORES
        + idx_b * stride_scores_b
        + idx_bdst * stride_scores_bdst
        + topk_indices * stride_scores_bk,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )

    tl.store(
        INDICES
        + idx_b * stride_indices_b
        + idx_bdst * stride_indices_bdst
        + idx_bk * stride_indices_bk,
        value=dupped_indices,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        GROUP_SIZES
        + idx_b * stride_group_sizes_b
        + idx_bdst * stride_group_sizes_bdst
        + idx_bk * stride_group_sizes_bk,
        value=dupped_group_size,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        SCORES_FINAL
        + idx_b * stride_scores_final_b
        + idx_bdst * stride_scores_final_bdst
        + idx_bk * stride_scores_final_bk,
        value=scores,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )


@triton.jit
def masking_iteration_draft_cuda_epiloge(
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS,
    stride_ks_b,
    stride_ks_bdst,
    KS_COUNT,
    stride_ks_count_b,
    stride_ks_count_bdst,
    stride_ks_count_g,
    KS_START_END,
    stride_ks_start_end_b,
    stride_ks_start_end_bdst,
    stride_ks_start_end_g,
    BK,
    MAX_TSRC,
    G: tl.constexpr,
    BLOCK_BK: tl.constexpr,
):
    idx_b = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2) * BLOCK_BK + tl.arange(0, BLOCK_BK)

    ks = tl.load(
        KS + idx_b * stride_ks_b + idx_bdst * stride_ks_bdst,
    )
    mask_bk = idx_bk < ks

    indices = tl.load(
        INDICES
        + idx_b * stride_indices_b
        + idx_bdst * stride_indices_bdst
        + idx_bk * stride_indices_bk,
        mask=mask_bk,
        other=0,
    ).to(tl.int32)

    hist = tl.histogram(indices // MAX_TSRC, G)
    hist -= (tl.arange(0, G) == 0).to(tl.int32) * (tl.sum((~mask_bk).to(tl.int32)))

    hist_cumsum = tl.cumsum(hist)

    idx_g = tl.arange(0, G)

    tl.atomic_add(
        KS_COUNT
        + idx_b * stride_ks_count_b
        + idx_bdst * stride_ks_count_bdst
        + idx_g * stride_ks_count_g,
        val=hist,
    )
    tl.atomic_add(
        KS_START_END
        + idx_b * stride_ks_start_end_b
        + idx_bdst * stride_ks_start_end_bdst
        + (idx_g + 1) * stride_ks_start_end_g,
        val=hist_cumsum,
    )


@triton.jit
def masking_iteration_draft_cuda_partial_softmax(
    SCORES,
    stride_scores_b,
    stride_scores_bdst,
    stride_scores_bk,
    DUPPED_INDICES,
    stride_dupped_indices_b,
    stride_dupped_indices_bdst,
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZES,
    stride_dupped_group_sizes_b,
    stride_dupped_group_sizes_bdst,
    stride_dupped_group_sizes_bk,
    PROBS,
    stride_probs_b,
    stride_probs_bdst,
    stride_probs_bk,
    SINK_TOKEN_SIZE,
    MASK_BLOCK_K,
    G: tl.constexpr,
    BK,
    MAX_BSRC,
    BLOCK_SIZE_K,
    BLOCK_SCORE: tl.constexpr,
    pid_0=None,
    pid_1=None,
    CARRYING: tl.constexpr = False,
):
    if pid_0 is None:
        pid_0 = tl.program_id(0)
    if pid_1 is None:
        pid_1 = tl.program_id(1)

    idx_b = pid_1
    idx_bdst = pid_0
    idx_bk = tl.arange(0, BLOCK_SCORE)
    mask_bk = idx_bk < BK

    indices = tl.load(
        DUPPED_INDICES
        + idx_b * stride_dupped_indices_b
        + idx_bdst * stride_dupped_indices_bdst
        + idx_bk * stride_dupped_indices_bk,
        mask=mask_bk,
        other=MAX_BSRC * G,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    group_sizes = tl.load(
        DUPPED_GROUP_SIZES
        + idx_b * stride_dupped_group_sizes_b
        + idx_bdst * stride_dupped_group_sizes_bdst
        + idx_bk * stride_dupped_group_sizes_bk,
        mask=mask_bk,
        other=MAX_BSRC * G,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    groups = indices // MAX_BSRC
    scores = tl.load(
        SCORES
        + idx_b * stride_scores_b
        + idx_bdst * stride_scores_bdst
        + idx_bk * stride_scores_bk,
        mask=mask_bk,
        other=float("-inf"),
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    ).to(tl.bfloat16)

    one = tl.zeros((1,), dtype=tl.bfloat16) + 1
    for i_group in range(G):
        mask_softmax = groups == i_group
        scores_masked = tl.where(mask_softmax, scores, float("-inf"))
        if G == 1:
            scores_softmax = tl.sigmoid(scores_masked.to(tl.float32)).to(
                scores_masked.dtype
            )
        else:
            count = tl.max(mask_softmax.to(tl.int32)).to(tl.float32)
            t = count / (BK * G)
            scores_softmax = tl.softmax(scores_masked * t)
            neg_scores_softmax_sorted = tl.sort(-scores_softmax)
            scores_promote_thresh = -tl.min(
                neg_scores_softmax_sorted
                * (tl.arange(0, BLOCK_SCORE) == (MASK_BLOCK_K * 0.5 * one).to(tl.int32))
            )
            scores_softmax = tl.where(
                scores_softmax >= scores_promote_thresh,
                scores_softmax + 1,
                scores_softmax,
            )
        scores = tl.where(mask_softmax, scores_softmax, scores).to(scores.dtype)

    # scores = tl.where((indices % MAX_BSRC) < tl.cdiv(SINK_TOKEN_SIZE, BLOCK_SIZE_K), 2, scores) # NOTE: sink token will handled by block_sparse_attention
    scores = tl.where(group_sizes == 0, -1, scores)

    tl.store(
        PROBS
        + idx_b * stride_scores_b
        + idx_bdst * stride_scores_bdst
        + idx_bk * stride_scores_bk,
        value=scores,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )


@triton.jit
def masking_iteration_draft_cuda_argsort(
    PROBS,
    stride_probs_b,
    stride_probs_bdst,
    stride_probs_bk,
    IDS,
    stride_ids_b,
    stride_ids_bdst,
    stride_ids_bk,
    T_GROUP_SIZES,
    stride_t_group_size_b,
    stride_t_group_size_bdst,
    BDST,
    BK: tl.constexpr,
    TOP_BK: tl.constexpr,
    BLOCK_BDST: tl.constexpr,
    pid_0=None,
    pid_1=None,
    CARRYING: tl.constexpr = False,
    carried_probs=None,
):
    if pid_0 is None:
        pid_0 = tl.program_id(0)
    if pid_1 is None:
        pid_1 = tl.program_id(1)

    idx_b = pid_1
    idx_bdst = pid_0 * BLOCK_BDST + tl.arange(0, BLOCK_BDST)
    mask_bdst = idx_bdst < BDST
    idx_bk = tl.arange(0, BK)

    t_group_size = tl.load(
        T_GROUP_SIZES
        + idx_b * stride_t_group_size_b
        + idx_bdst * stride_t_group_size_bdst,
        mask=mask_bdst,
        other=1.0,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    if tl.max(t_group_size) < 1.0:
        return

    probs = tl.load(
        PROBS
        + idx_b * stride_probs_b
        + idx_bdst[:, None] * stride_probs_bdst
        + idx_bk[None, :] * stride_probs_bk,
        mask=mask_bdst[:, None],
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    ids = tl.broadcast_to(tl.arange(0, BK)[None, :], (BLOCK_BDST, BK)).to(tl.int32)

    # ids_low, ids_high = tl.split(tl.reshape(ids, TOP_BK, 2))
    # probs_low, probs_high = tl.split(tl.reshape(probs.to(tl.float32), TOP_BK, 2))
    # probs_low, ids_low = tl_argsort(probs_low, ids_low, 0, True)
    # probs_high, ids_high = tl_argsort(probs_high, ids_high, 0, True)
    # tl.store(
    #     IDS +\
    #         idx_b * stride_ids_b +\
    #         idx_bdst[:, None] * stride_ids_bdst +\
    #         tl.arange(0, TOP_BK)[None, :] * stride_ids_bk,
    #     value=tl.where(
    #         probs_low > probs_high,
    #         ids_low,
    #         ids_high,
    #     )[None, :],
    #     mask=mask_bdst[:, None],
    #     cache_modifier=DEFAULT_CACHE_MODIFIER,
    # )

    _, ids = tl_argsort(probs.to(tl.float32), ids, 1, True)
    # ids, _ = tl.split(tl.trans(tl.reshape(ids, 2, TOP_BK), 1, 0))

    tl.store(
        IDS
        + idx_b * stride_ids_b
        + idx_bdst[:, None] * stride_ids_bdst
        + idx_bk[None, :] * stride_ids_bk,
        value=ids,
        mask=(idx_bk < TOP_BK)[None, :] & mask_bdst[:, None],
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )


def masking_iteration_draft_python_epilog(
    indices: Tensor, ks: Tensor, mask_block_k, MAX_TSRC, B, BDST, G
):
    if G > 1:
        ks_count = torch.zeros((B, BDST, G), dtype=torch.int32, device=indices.device)
        ks_start_end = torch.zeros(
            (B, BDST, G + 1), dtype=torch.int32, device=indices.device
        )

        BLOCK_BK = 128
        grid = (B, BDST, triton.cdiv(indices.shape[-1], BLOCK_BK))
        pre_device = torch.get_default_device()
        torch.set_default_device(indices.device)
        masking_iteration_draft_cuda_epiloge[grid](
            indices,
            *indices.stride(),
            ks,
            *ks.stride(),
            ks_count,
            *ks_count.stride(),
            ks_start_end,
            *ks_start_end.stride(),
            mask_block_k,
            MAX_TSRC,
            G,
            BLOCK_BK,
        )
        torch.set_default_device(pre_device)
        # print(indices[0, -1] // TSRC)
        # print(ks_count[0, -1], ks_start_end[0, -1])
        # print(ks_count.float().mean(1).int()[0])
        # if topk_indices is not None:
        #     scores_final = scores\
        #         .gather(index=topk_indices, dim=-1)\
        #         .gather(index=indices_sort_mapping, dim=-1)
        # else:
        #     scores_final = scores[:, :, :indices_sort_mapping.shape[-1]]\
        #         .gather(index=indices_sort_mapping, dim=-1)
    else:
        ks_count = ks[:, :, None]
        ks_start_end = torch.zeros(
            (B, BDST, G + 1), dtype=torch.int32, device=indices.device
        )
        ks_start_end[:, :, -1] = ks
        # if topk_indices is not None:
        #     scores_final = scores\
        #         .gather(index=topk_indices, dim=-1)\
        #         .gather(index=indices_sort_mapping, dim=-1)
        # else:
        #     scores_final = scores[:, :, :indices_sort_mapping.shape[-1]]\
        #         .gather(index=indices_sort_mapping, dim=-1)

    return ks_count, ks_start_end


def get_masking_iteration_draft_cuda_fused_configs():
    autotune_disabled = os.getenv("HIP_DISABLE_AUTOTUNE", "1") == "1"
    if autotune_disabled:
        device_name = torch.cuda.get_device_name()
        defaults = {
            "NVIDIA A100-SXM4-80GB": dict(
                num_warps=4,
                num_stages=2,
                maxnreg=512,
            ),
        }.get(device_name, dict(num_warps=4, num_stages=2, maxnreg=512))
        return [triton.Config({}, **defaults)]
    if os.getenv("HIP_DISABLE_AUTOTUNE_WARNINGS", "0") == "0":
        warnings.warn(
            "triton autotuning is activated. this should be disabled for faster startup. if you want set HIP_DISABLE_AUTOTUNE=1"
        )
    configs = []
    for num_warps in [2, 4, 8]:
        # for num_warps in [4,]:
        for num_stages in [
            2,
        ]:
            # for num_stages in [2]:
            for num_regs in [64, 128, 256, 512]:
                # for num_regs in [256]:
                configs.append(
                    triton.Config(
                        {},
                        num_warps=num_warps,
                        num_stages=num_stages,
                        maxnreg=num_regs,
                    )
                )
    return configs


@triton.jit
def sum_all_diagonal_cuda(
    SCORES,
    stride_scores_n,
    stride_scores_tdst,
    stride_scores_tsrc,
    OUT,
    stride_out_n,
    stride_out_tdst,
    stride_out_tsrc,
    TDST,
    TSRC,
    GROUP_TSRC,
    BLOCK_TDST: tl.constexpr,
):
    idx_n = tl.program_id(2).to(tl.int64)
    idx_bdst = tl.program_id(1).to(tl.int64)
    idx_bsrc = tl.program_id(0).to(tl.int64)

    for i_gsrc in range(GROUP_TSRC):
        idx_tsrc_end = idx_bsrc * GROUP_TSRC + i_gsrc
        idx_tdst_end = tl.minimum(TDST - 1, idx_bdst * BLOCK_TDST)
        idx_out = idx_tsrc_end + TDST - 1 - idx_tdst_end
        mask_out = (idx_out >= 0) & (idx_out < TSRC) & (idx_tsrc_end < TSRC)

        idx_tdst = tl.arange(0, BLOCK_TDST) + idx_bdst * BLOCK_TDST
        mask_tdst = idx_tdst < BLOCK_TDST

        idx_tsrc = idx_tsrc_end - (BLOCK_TDST - 1) + tl.arange(0, BLOCK_TDST)
        mask_tsrc = (idx_tsrc >= 0) & (idx_tsrc < TSRC) & mask_tdst

        scores = tl.load(
            SCORES
            + idx_n * stride_scores_n
            + idx_tdst * stride_scores_tdst
            + idx_tsrc * stride_scores_tsrc,
            mask=mask_tdst & mask_tsrc & mask_out,
            other=0,
        )
        score = tl.sum(scores)

        tl.atomic_add(
            OUT
            + idx_n * stride_out_n
            + 0 * stride_out_tdst
            + idx_out * stride_out_tsrc,
            val=score,
            mask=mask_out,
        )


def sum_all_diagonal(scores: Tensor):
    N, TDST, TSRC = scores.shape
    reduced_score = torch.zeros(
        (N, 1, TSRC),
        dtype=torch.float32,
        device=scores.device,
    )

    GROUP_TSRC = cdiv_python(TSRC, 2048)
    BLOCK_TDST = 128
    grid = (
        triton.cdiv(TSRC, GROUP_TSRC),
        triton.cdiv(TDST, BLOCK_TDST),
        N,
    )
    d = torch.get_default_device()
    torch.set_default_device(scores.device)
    sum_all_diagonal_cuda[grid](
        scores,
        *scores.stride(),
        reduced_score,
        *reduced_score.stride(),
        TDST,
        TSRC,
        GROUP_TSRC,
        BLOCK_TDST,
    )
    torch.set_default_device(d)

    return reduced_score.to(scores.dtype)


@triton.autotune(
    configs=get_masking_iteration_draft_cuda_fused_configs(),
    key=[
        "BLOCK_BK",
        "BLOCK_SIZE_K",
        "BLOCK_SIZE_Q",
        "HID",
        "TDST_NEXT_POWER_OF_2",
    ],
    restore_value=[
        "KEY_ACCESS_LOG",
        "KEY_ACCESS_COUNT",
        "INDICES",
        "KS",
        "GROUP_SIZE",
        "DUPPED_INDICES",
        "DUPPED_GROUP_SIZE",
        "SCORES",
        "SCORES_FINAL",
        "PROBS",
        "TOPK_IDS",
        "T_GROUP_SIZE",
    ],
    warmup=200,
    rep=1000,
)
@triton.jit
def masking_iteration_draft_cuda_fused(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_bh,
    stride_q_g,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    VERTICAL_MASK,
    stride_vertical_mask_n,
    stride_vertical_mask_tsrc,
    DIAGONAL_MASK,
    stride_diagonal_mask_n,
    stride_diagonal_mask_tsrc,
    ACCESS_LOG_STORE_FROM,
    KEY_ACCESS_LOG,
    stride_key_access_log_b,
    stride_key_access_log_bdst,
    stride_key_access_log_t,
    KEY_ACCESS_COUNT,
    stride_key_access_count_b,
    stride_key_access_count_bdst,
    MAX_ACCESS_COUNT,
    BLOCK_ACCESS_LOG,
    stride_block_access_log_b,
    stride_block_access_log_bdst,
    stride_block_access_log_t,
    BLOCK_ACCESS_SCORE,
    stride_block_access_score_b,
    stride_block_access_score_bdst,
    stride_block_access_score_t,
    BLOCK_ACCESS_COUNT,
    stride_block_access_count_b,
    stride_block_access_count_bdst,
    MAX_BLOCK_ACCESS_COUNT,
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS,
    stride_ks_b,
    stride_ks_bdst,
    GROUP_SIZE,
    stride_group_size_b,
    stride_group_size_bdst,
    stride_group_size_bk,
    DUPPED_INDICES,
    stride_dupped_indices_b,
    stride_dupped_indices_bdst,
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZE,
    stride_dupped_group_size_b,
    stride_dupped_group_size_bdst,
    stride_dupped_group_size_bk,
    SCORES,
    stride_scores_b,
    stride_scores_bdst,
    stride_scores_bk,
    SCORES_FINAL,
    stride_scores_final_b,
    stride_scores_final_bdst,
    stride_scores_final_bk,
    SCORES_CACHED: tl.constexpr,
    PROBS,
    stride_probs_b,
    stride_probs_bdst,
    stride_probs_bk,
    TOPK_IDS,
    stride_topk_ids_b,
    stride_topk_ids_bdst,
    stride_topk_ids_bk,
    T_GROUP_SIZE,
    stride_t_group_size_b,
    stride_t_group_size_bdst,
    INDICES_TDST,
    stride_indices_tdst_t,
    mask_k,
    sink_token_size,
    sliding_window_size,
    BH: tl.constexpr,
    G: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    MAX_BDST,
    MAX_BSRC,
    BK: tl.constexpr,
    HID: tl.constexpr,
    RAND_SEED,
    SAMPLE_METHOD: tl.constexpr,
    BRANCH_METHOD: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND,
    stride_q_ind_b,
    stride_q_ind_g,
    stride_q_ind_bdst,
    stride_q_ind_k,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_bsz,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    BLOCK_SCORE: tl.constexpr,
    GROUP_BDST,
    GROUP_BH,
    TDST_NEXT_POWER_OF_2,
    indices_bk_len: tl.constexpr,
    probs_bk_len: tl.constexpr,
):
    # _pid = tl.program_id(0)
    # #(BBH, GDST, GBH, BSZ)
    # _grid_bbh = GROUP_BH
    _grid_gdst = tl.cdiv(MAX_BDST, GROUP_BDST)
    # _grid_gbh = BH // GROUP_BH

    # _pid_bbh = _pid % _grid_bbh
    # _pid_gdst = (_pid // _grid_bbh) % _grid_gdst
    # _pid_gbh = (_pid // (_grid_bbh * _grid_gdst)) % _grid_gbh
    # _pid_bsz = _pid // (_grid_bbh * _grid_gdst * _grid_gbh)

    # # BH
    # _pid_0 = (_pid_bbh + _pid_gbh * GROUP_BH)
    # # BDST / GROUP BDST
    # _pid_1 = _pid_gdst
    # # BSZ
    # _pid_2 = _pid_bsz

    _pid_0 = (
        tl.program_id(0).to(tl.int64) % GROUP_BH
        + tl.program_id(1).to(tl.int64) * GROUP_BH
    )
    _pid_1 = (tl.program_id(0).to(tl.int64) // GROUP_BH) % _grid_gdst
    _pid_2 = tl.program_id(2).to(tl.int64)

    # _pid_0 = _pid % BH
    # _pid_1 = (_pid // BH) % _grid_gdst
    # _pid_2 = _pid // (BH * _grid_gdst)

    # _pid_0 = tl.program_id(0)
    # _pid_1 = tl.program_id(1)
    # _pid_2 = tl.program_id(2)

    pid_1 = _pid_2 * BH + _pid_0

    num_groups = tl.minimum(GROUP_BDST, (MAX_BDST - _pid_1 * GROUP_BDST))

    if num_groups == 1:
        pid_0 = _pid_1 * GROUP_BDST
        idx_b = pid_1
        idx_bdst = pid_0
        max_group_size = tl.load(
            T_GROUP_SIZE
            + idx_b * stride_t_group_size_b
            + idx_bdst * stride_t_group_size_bdst,
        ).to(tl.float32)
        if max_group_size <= 1:
            return

    for i_group in range(num_groups):
        # originally bdst dim, before vectorize head
        pid_0 = _pid_1 * GROUP_BDST + i_group
        idx_b = pid_1
        idx_bdst = pid_0

        max_group_size = tl.load(
            T_GROUP_SIZE
            + idx_b * stride_t_group_size_b
            + idx_bdst * stride_t_group_size_bdst,
        ).to(tl.float32)

        while max_group_size > 1:
            n_program = tl.cdiv(indices_bk_len, BLOCK_BK)
            for i_program in range(n_program):
                masking_iteration_draft_cuda_dup_and_score(
                    Q,
                    stride_q_bsz,
                    stride_q_tdst,
                    stride_q_bh,
                    stride_q_g,
                    stride_q_hid,
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    POS,
                    stride_pos_bsz,
                    stride_pos_tdst,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    VERTICAL_MASK,
                    stride_vertical_mask_n,
                    stride_vertical_mask_tsrc,
                    DIAGONAL_MASK,
                    stride_diagonal_mask_n,
                    stride_diagonal_mask_tsrc,
                    ACCESS_LOG_STORE_FROM,
                    KEY_ACCESS_LOG,
                    stride_key_access_log_b,
                    stride_key_access_log_bdst,
                    stride_key_access_log_t,
                    KEY_ACCESS_COUNT,
                    stride_key_access_count_b,
                    stride_key_access_count_bdst,
                    MAX_ACCESS_COUNT,
                    BLOCK_ACCESS_LOG,
                    stride_block_access_log_b,
                    stride_block_access_log_bdst,
                    stride_block_access_log_t,
                    BLOCK_ACCESS_SCORE,
                    stride_block_access_score_b,
                    stride_block_access_score_bdst,
                    stride_block_access_score_t,
                    BLOCK_ACCESS_COUNT,
                    stride_block_access_count_b,
                    stride_block_access_count_bdst,
                    MAX_BLOCK_ACCESS_COUNT,
                    INDICES,
                    stride_indices_b,
                    stride_indices_bdst,
                    stride_indices_bk,
                    KS,
                    stride_ks_b,
                    stride_ks_bdst,
                    GROUP_SIZE,
                    stride_group_size_b,
                    stride_group_size_bdst,
                    stride_group_size_bk,
                    DUPPED_INDICES,
                    stride_dupped_indices_b,
                    stride_dupped_indices_bdst,
                    stride_dupped_indices_bk,
                    DUPPED_GROUP_SIZE,
                    stride_dupped_group_size_b,
                    stride_dupped_group_size_bdst,
                    stride_dupped_group_size_bk,
                    SCORES,
                    stride_scores_b,
                    stride_scores_bdst,
                    stride_scores_bk,
                    SCORES_FINAL,
                    stride_scores_final_b,
                    stride_scores_final_bdst,
                    stride_scores_final_bk,
                    SCORES_CACHED,
                    T_GROUP_SIZE,
                    stride_t_group_size_b,
                    stride_t_group_size_bdst,
                    INDICES_TDST,
                    stride_indices_tdst_t,
                    mask_k,
                    sliding_window_size,
                    BH,
                    G,
                    MAX_TDST,
                    MAX_TSRC,
                    BK,
                    HID,
                    RAND_SEED,
                    SAMPLE_METHOD,
                    BRANCH_METHOD,
                    KV_HEAD_REPEAT,
                    USING_EXTEND,
                    NEED_APPLY_ROPE,
                    extend_window_size,
                    extend_group_size,
                    USING_SPARQ,
                    SPARQ_HID,
                    Q_IND,
                    stride_q_ind_b,
                    stride_q_ind_g,
                    stride_q_ind_bdst,
                    stride_q_ind_k,
                    # paged attention args template
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    V_CACHE,
                    stride_v_cache_page,
                    stride_v_cache_offset,
                    stride_v_cache_kv_head,
                    stride_v_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_bsz,
                    # offload cache args template
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_METHOD,
                    OFFLOAD_CACHE_BUDGET,
                    OFFLOAD_CACHE_KV_HEAD,
                    OFFLOAD_CACHE_K_TABLES,
                    stride_offload_cache_k_tables_n,
                    stride_offload_cache_k_tables_t,
                    OFFLOAD_CACHE_K_BANKS,
                    stride_offload_cache_k_banks_n,
                    stride_offload_cache_k_banks_page,
                    stride_offload_cache_k_banks_offset,
                    stride_offload_cache_k_banks_hid,
                    OFFLOAD_CACHE_K_BANK_STATS,
                    stride_offload_cache_k_bank_stats_n,
                    stride_offload_cache_k_bank_stats_page,
                    stride_offload_cache_k_bank_stats_k,
                    OFFLOAD_CACHE_COUNTERS,
                    stride_offload_cache_counters_n,
                    stride_offload_cache_counters_k,
                    IS_CAUSAL,
                    BLOCK_SIZE_Q,
                    BLOCK_STRIDE_Q,
                    BLOCK_SIZE_K,
                    BLOCK_STRIDE_K,
                    BLOCK_BK,
                    0,
                    0,
                    pid_0=i_program,
                    pid_1=pid_0,
                    pid_2=pid_1,
                )
            # end for
            tl.debug_barrier()

            # same grid with master (BDST, B)
            masking_iteration_draft_cuda_partial_softmax(
                SCORES,
                stride_scores_b,
                stride_scores_bdst,
                stride_scores_bk,
                DUPPED_INDICES,
                stride_dupped_indices_b,
                stride_dupped_indices_bdst,
                stride_dupped_indices_bk,
                DUPPED_GROUP_SIZE,
                stride_dupped_group_size_b,
                stride_dupped_group_size_bdst,
                stride_dupped_group_size_bk,
                PROBS,
                stride_probs_b,
                stride_probs_bdst,
                stride_probs_bk,
                sink_token_size,
                BK,
                G,
                probs_bk_len,
                MAX_BSRC,
                BLOCK_SIZE_K,
                BLOCK_SCORE,
                pid_0=pid_0,
                pid_1=pid_1,
            )
            tl.debug_barrier()

            # TODO: support score_head_group_size

            # same grid with master (BDST, B)
            masking_iteration_draft_cuda_argsort(
                PROBS,
                stride_probs_b,
                stride_probs_bdst,
                stride_probs_bk,
                TOPK_IDS,
                stride_topk_ids_b,
                stride_topk_ids_bdst,
                stride_topk_ids_bk,
                T_GROUP_SIZE,
                stride_t_group_size_b,
                stride_t_group_size_bdst,
                MAX_BDST,
                probs_bk_len,
                BK * G,
                1,
                pid_0=pid_0,
                pid_1=pid_1,
            )
            tl.debug_barrier()

            # num_program = tl.cdiv(indices_bk_len, BLOCK_BK)
            # for i_program in range(num_program):
            masking_iteration_draft_cuda_gather(
                INDICES,
                stride_indices_b,
                stride_indices_bdst,
                stride_indices_bk,
                GROUP_SIZE,
                stride_group_size_b,
                stride_group_size_bdst,
                stride_group_size_bk,
                SCORES_FINAL,
                stride_scores_final_b,
                stride_scores_final_bdst,
                stride_scores_final_bk,
                DUPPED_INDICES,
                stride_dupped_indices_b,
                stride_dupped_indices_bdst,
                stride_dupped_indices_bk,
                DUPPED_GROUP_SIZE,
                stride_dupped_group_size_b,
                stride_dupped_group_size_bdst,
                stride_dupped_group_size_bk,
                SCORES,
                stride_scores_b,
                stride_scores_bdst,
                stride_scores_bk,
                TOPK_IDS,
                stride_topk_ids_b,
                stride_topk_ids_bdst,
                stride_topk_ids_bk,
                T_GROUP_SIZE,
                stride_t_group_size_b,
                stride_t_group_size_bdst,
                G,
                BK,
                indices_bk_len,
                pid_0=0,
                pid_1=pid_0,
                pid_2=pid_1,
            )

            tl.debug_barrier()

            # SCORES_CACHED = True

            if BRANCH_METHOD == "random":
                max_group_size *= 0.7
            else:
                max_group_size *= 0.5
        tl.store(
            T_GROUP_SIZE
            + idx_b * stride_t_group_size_b
            + idx_bdst * stride_t_group_size_bdst,
            value=max_group_size,
        )
        tl.debug_barrier()


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_BK': 16}, num_warps=1),
#         triton.Config({'BLOCK_BK': 32}, num_warps=1),
#         # triton.Config({'BLOCK_BK': 64}, num_warps=1),
#         # triton.Config({'BLOCK_BK': 128}, num_warps=1),

#         # triton.Config({'BLOCK_BK': 16}, num_warps=2),
#         triton.Config({'BLOCK_BK': 32}, num_warps=2),
#         triton.Config({'BLOCK_BK': 64}, num_warps=2),
#         # triton.Config({'BLOCK_BK': 128}, num_warps=2),

#         # triton.Config({'BLOCK_BK': 16}, num_warps=4),
#         # triton.Config({'BLOCK_BK': 32}, num_warps=4),
#         triton.Config({'BLOCK_BK': 64}, num_warps=4),
#         triton.Config({'BLOCK_BK': 128}, num_warps=4),

#         # triton.Config({'BLOCK_BK': 16}, num_warps=8),
#         # triton.Config({'BLOCK_BK': 32}, num_warps=8),
#         triton.Config({'BLOCK_BK': 64}, num_warps=8),
#         triton.Config({'BLOCK_BK': 128}, num_warps=8),


#         # triton.Config({'BLOCK_BK': 16}, num_warps=16),
#         # triton.Config({'BLOCK_BK': 32}, num_warps=16),
#         triton.Config({'BLOCK_BK': 64}, num_warps=16),
#         triton.Config({'BLOCK_BK': 128}, num_warps=16),
#     ],
#     key=['BLOCK_SIZE_K', 'BLOCK_SIZE_Q'],
#     rep=200,
#     use_cuda_graph=True,
# )
@triton.jit
def masking_iteration_draft_cuda_initialize_score(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_bh,
    stride_q_g,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    VERTICAL_MASK,
    stride_vertical_mask_n,
    stride_vertical_mask_tsrc,
    DIAGONAL_MASK,
    stride_diagonal_mask_n,
    stride_diagonal_mask_tsrc,
    ACCESS_LOG_STORE_FROM,
    KEY_ACCESS_LOG,
    stride_key_access_log_b,
    stride_key_access_log_bdst,
    stride_key_access_log_t,
    KEY_ACCESS_COUNT,
    stride_key_access_count_b,
    stride_key_access_count_bdst,
    MAX_ACCESS_COUNT,
    BLOCK_ACCESS_LOG,
    stride_block_access_log_b,
    stride_block_access_log_bdst,
    stride_block_access_log_t,
    BLOCK_ACCESS_SCORE,
    stride_block_access_score_b,
    stride_block_access_score_bdst,
    stride_block_access_score_t,
    BLOCK_ACCESS_COUNT,
    stride_block_access_count_b,
    stride_block_access_count_bdst,
    MAX_BLOCK_ACCESS_COUNT,
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    SCORES,
    stride_scores_b,
    stride_scores_bdst,
    stride_scores_bk,
    T_GROUP_SIZE,
    stride_t_group_size_b,
    stride_t_group_size_bdst,
    INDICES_TDST,
    stride_indices_tdst_t,
    sliding_window_size,
    indices_bk_len,
    BH: tl.constexpr,
    G: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    HID: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
    MASK_K: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND,
    stride_q_ind_b,
    stride_q_ind_g,
    stride_q_ind_bdst,
    stride_q_ind_k,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_bsz,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    KEY_DUP: tl.constexpr = 1,
):
    pid = tl.program_id(0).to(tl.int64)

    grid_bh = BH
    grid_bk = tl.cdiv(indices_bk_len, BLOCK_BK)
    grid_bdst = tl.cdiv(MAX_TDST, BLOCK_SIZE_Q)

    pid_bh = tl.program_id(0) % BH
    pid_bk = tl.program_id(0) // BH
    pid_bdst = tl.program_id(1).to(tl.int64)
    pid_bsz = tl.program_id(2).to(tl.int64)

    idx_bk = pid_bk * BLOCK_BK + tl.arange(0, BLOCK_BK).to(tl.int64)
    mask_bk = idx_bk < indices_bk_len
    idx_bdst = pid_bdst
    idx_b = pid_bsz * BH + pid_bh

    t_group_size = tl.load(
        T_GROUP_SIZE
        + idx_b * stride_t_group_size_b
        + idx_bdst * stride_t_group_size_bdst,
    )
    if t_group_size <= 1.0:
        return

    idx_tdst = (
        idx_bdst * BLOCK_SIZE_Q
        + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
        + (tl.minimum(MAX_TDST, BLOCK_STRIDE_Q) - 1)
    )
    # idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) + (BLOCK_SIZE_Q - BLOCK_SIZE_Q // BLOCK_STRIDE_Q)
    idx_tdst_no_proj = idx_tdst
    mask_tdst = idx_tdst < MAX_TDST
    if INDICES_TDST is not None:
        idx_tdst = tl.load(
            INDICES_TDST + idx_tdst.to(tl.int64) * stride_indices_tdst_t,
            mask=mask_tdst,
            other=MAX_TDST,
        ).to(tl.int64)

    if IS_CAUSAL:
        pos_tdst = tl.load(
            POS + (idx_b // BH) * stride_pos_bsz + idx_tdst_no_proj * stride_pos_tdst,
            mask=mask_tdst,
            other=0,
        ).to(tl.int64)
    else:
        pos_tdst = tl.full(
            (BLOCK_SIZE_Q // BLOCK_STRIDE_Q,), value=MAX_TSRC, dtype=tl.int64
        )
    TSRC = tl.max(pos_tdst)
    TSRC = tl.maximum(0, TSRC - sliding_window_size)
    BSRC = tl.cdiv(TSRC, BLOCK_SIZE_K)

    indices = tl.load(
        INDICES
        + idx_b * stride_indices_b
        + idx_bdst * stride_indices_bdst
        + idx_bk * stride_indices_bk,
        mask=mask_bk,
        other=0,
    )

    scores = masking_iteration_draft_cuda_dup_and_score_calc_score(
        indices,
        KEY_DUP,
        Q,
        stride_q_bsz,
        stride_q_tdst,
        stride_q_bh,
        stride_q_g,
        stride_q_hid,
        K,
        stride_k_bsz,
        stride_k_tsrc,
        stride_k_head,
        stride_k_hid,
        COS,
        stride_cos_t,
        stride_cos_hid,
        SIN,
        stride_sin_t,
        stride_sin_hid,
        VERTICAL_MASK,
        stride_vertical_mask_n,
        stride_vertical_mask_tsrc,
        DIAGONAL_MASK,
        stride_diagonal_mask_n,
        stride_diagonal_mask_tsrc,
        ACCESS_LOG_STORE_FROM,
        KEY_ACCESS_LOG,
        stride_key_access_log_b,
        stride_key_access_log_bdst,
        stride_key_access_log_t,
        KEY_ACCESS_COUNT,
        stride_key_access_count_b,
        stride_key_access_count_bdst,
        MAX_ACCESS_COUNT,
        BLOCK_ACCESS_LOG,
        stride_block_access_log_b,
        stride_block_access_log_bdst,
        stride_block_access_log_t,
        BLOCK_ACCESS_SCORE,
        stride_block_access_score_b,
        stride_block_access_score_bdst,
        stride_block_access_score_t,
        BLOCK_ACCESS_COUNT,
        stride_block_access_count_b,
        stride_block_access_count_bdst,
        MAX_BLOCK_ACCESS_COUNT,
        idx_b,
        idx_bdst,
        idx_tdst,
        mask_tdst,
        pos_tdst,
        idx_bk,
        mask_bk,
        sliding_window_size,
        BH,
        G,
        MAX_TDST,
        MAX_TSRC,
        HID,
        KV_HEAD_REPEAT,
        MASK_K,
        USING_EXTEND,
        NEED_APPLY_ROPE,
        extend_window_size,
        extend_group_size,
        USING_SPARQ,
        SPARQ_HID,
        Q_IND,
        stride_q_ind_b,
        stride_q_ind_g,
        stride_q_ind_bdst,
        stride_q_ind_k,
        # paged attention args template
        USING_PAGES,
        PAGE_SIZE,
        K_CACHE,
        stride_k_cache_page,
        stride_k_cache_offset,
        stride_k_cache_kv_head,
        stride_k_cache_hid,
        V_CACHE,
        stride_v_cache_page,
        stride_v_cache_offset,
        stride_v_cache_kv_head,
        stride_v_cache_hid,
        BLOCK_TABLE,
        stride_block_table_bsz,
        stride_block_table_page,
        CACHE_SEQ_LENS,
        stride_cache_seq_lens_bsz,
        # offload cache args template
        USING_OFFLOAD_CACHE,
        OFFLOAD_CACHE_METHOD,
        OFFLOAD_CACHE_BUDGET,
        OFFLOAD_CACHE_KV_HEAD,
        OFFLOAD_CACHE_K_TABLES,
        stride_offload_cache_k_tables_n,
        stride_offload_cache_k_tables_t,
        OFFLOAD_CACHE_K_BANKS,
        stride_offload_cache_k_banks_n,
        stride_offload_cache_k_banks_page,
        stride_offload_cache_k_banks_offset,
        stride_offload_cache_k_banks_hid,
        OFFLOAD_CACHE_K_BANK_STATS,
        stride_offload_cache_k_bank_stats_n,
        stride_offload_cache_k_bank_stats_page,
        stride_offload_cache_k_bank_stats_k,
        OFFLOAD_CACHE_COUNTERS,
        stride_offload_cache_counters_n,
        stride_offload_cache_counters_k,
        IS_CAUSAL,
        BLOCK_SIZE_Q,
        BLOCK_STRIDE_Q,
        BLOCK_SIZE_K,
        BLOCK_STRIDE_K,
        BLOCK_BK,
        "max",
    )

    tl.store(
        SCORES
        + idx_b * stride_scores_b
        + idx_bdst * stride_scores_bdst
        + idx_bk * stride_scores_bk,
        mask=mask_bk,
        value=scores,
    )


@nvtx.annotate("masking_iteration_draft")
def masking_iteration_draft(
    q: Tensor,
    k: Optional[Tensor],
    position_ids: Tensor,
    args: "HiPAttentionArgs",
    # seeds
    indices_seed: Optional[Tensor] = None,
    ks_seed: Optional[Tensor] = None,
    scores_seed: Optional[Tensor] = None,
    group_size_seed: Optional[Tensor] = None,
    max_group_size_seed: Optional[float] = None,
    indices_tdst: Optional[Tensor] = None,
):
    assert isinstance(q, Tensor)
    if k is not None:
        assert q.device == k.device
        assert isinstance(k, Tensor)

    if args.rope_cos is not None and args.using_extend:
        assert args.rope_cos.ndim == 2
        assert args.rope_cos.shape[-1] == q.shape[-1]
        assert isinstance(args.rope_cos, Tensor)

    if args.rope_sin is not None and args.using_extend:
        assert args.rope_sin.ndim == 2
        assert args.rope_sin.shape[-1] == q.shape[-1]
        assert isinstance(args.rope_sin, Tensor)
        assert isinstance(args.rope_sin, Tensor)

    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, KV_HEAD, HID = k.shape
    else:
        assert args.k_cache is not None
        N_PAGES, PAGE_SIZE, KV_HEAD, HID = args.k_cache.shape
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD
    N = BSZ * HEAD
    if indices_tdst is not None:
        TDST = len(indices_tdst)
        assert indices_tdst.ndim == 1
        indices_tdst_stride = indices_tdst.stride()
    else:
        indices_tdst_stride = (0,)
    BDST = cdiv_python(TDST, args.block_size_q)
    if k is not None:
        _, TSRC, _, _ = k.shape
        BSRC = cdiv_python(TSRC, args.block_size_k)
        MAX_TSRC = TSRC
        MAX_BSRC = BSRC
    else:
        TSRC = BSRC = None
        MAX_TSRC = N_PAGES * PAGE_SIZE
        MAX_BSRC = cdiv_python(MAX_TSRC, args.block_size_k)

    assert (
        N % args.topk_head_group_size
    ) == 0, "batch * n_head should divisible by head group size"

    # split batch-head dim into head groups
    q = q.view(
        BSZ, -1, HEAD // args.topk_head_group_size, args.topk_head_group_size, HID
    )
    if k is not None:
        k = k.view(BSZ, TSRC, KV_HEAD, HID)

    BSZ, _, BH, G, HID = q.shape
    B = BSZ * BH
    mask_block_k = cdiv_python(args.mask_k, args.block_size_k)

    assert args.block_size_k_group == 1
    if args.block_size_k_group > 1:
        warnings.warn("K grouping is inefficient right now.")
        k_group = k.view(
            BSZ,
            triton.cdiv(TSRC, args.block_size_k_group),
            args.block_size_k_group,
            BH,
            G,
            HID,
        )
        k_group_min = torch.min(k_group, dim=2)
        k_group_max = torch.max(k_group, dim=2)
        k = torch.concat([k_group_min, k_group_max], dim=-1)

    indices = torch.full(
        (
            B,
            cdiv_python(TDST, args.block_size_q),
            # head group is merged as single sequence
            G * mask_block_k,
        ),
        fill_value=(MAX_BSRC + args.block_size_k + args.block_size_q) * G,
        dtype=torch.int32,
        device=q.device,
    )

    ks = torch.zeros(
        (
            B,
            cdiv_python(TDST, args.block_size_q),
        ),
        dtype=torch.int32,
        device=q.device,
    )

    group_sizes = torch.zeros_like(indices)
    t_group_sizes = torch.zeros((B, BDST), dtype=torch.float32, device=q.device)

    if max_group_size_seed is None:
        max_group_strategy = "worst"

        if indices_seed is None:
            # always chunks are evenly distributed. fastest.
            max_group_strategy = "best"

        if k is not None:
            if max_group_strategy == "oracle":
                # > oracle      5.1117  18.4503 sec
                # This is impossible at this point, because t_group_size is initilized by following kernel
                raise NotImplementedError()
                max_group_size = torch.max(t_group_sizes).item()
            elif max_group_strategy == "best":
                # > best case   5.1218  10.3745 sec
                #   (not complete search if you gave seed)
                max_group_size = triton.cdiv(BSRC, mask_block_k)
            elif max_group_strategy == "worst":
                # > worst case  5.1097  17.6545 sec
                #   (always complete search)
                max_group_size = max(1, BSRC - mask_block_k)
            elif max_group_strategy == "greedy":
                # > greedy      5.1202  11.4861 sec
                #   (slightly generous then best stratgy)
                max_group_size = triton.cdiv(BSRC, mask_block_k) * 2
            elif max_group_strategy == "constant":
                # TODO: test this
                max_group_size = min(triton.cdiv(BSRC, args.block_size_k), 8)
            else:
                raise Exception()
        else:
            assert args.k_cache is not None
            max_group_size = None
    else:
        max_group_size = max_group_size_seed

    if max_group_size is not None:
        KEY_ACCESS_LEN = args.mask_k * 2 * math.ceil(math.log2(max_group_size) + 1)
    else:
        KEY_ACCESS_LEN = args.mask_k * 2 * math.ceil(math.log2(MAX_BSRC) + 1)

    access_log_num_blocks_to_keep = min(args.access_log_num_blocks_to_keep, BDST)
    access_log_store_from = BDST - access_log_num_blocks_to_keep
    if args.output_key_access_log:
        key_access_log = torch.full(
            (
                B,
                access_log_num_blocks_to_keep,
                KEY_ACCESS_LEN,
            ),
            dtype=torch.int32,
            # fill_value=torch.iinfo(torch.int32).max,
            device=q.device,
            fill_value=-1,
        )
        key_access_count = torch.zeros(
            (
                B,
                access_log_num_blocks_to_keep,
            ),
            dtype=torch.long,
            device=q.device,
        )
    else:
        key_access_log = None
        key_access_count = None

    BLOCK_ACCESS_LEN = KEY_ACCESS_LEN // (args.block_size_k // args.block_stride_k)
    if args.output_block_access_log:
        block_access_log = torch.full(
            (
                B,
                access_log_num_blocks_to_keep,
                BLOCK_ACCESS_LEN,
            ),
            dtype=torch.int32,
            device=q.device,
            fill_value=-1,
        )
        block_access_score = torch.full(
            (B, access_log_num_blocks_to_keep, BLOCK_ACCESS_LEN),
            device=q.device,
            dtype=torch.bfloat16,
            fill_value=-32000.0,
        )
        block_access_count = torch.zeros(
            (
                B,
                access_log_num_blocks_to_keep,
            ),
            dtype=torch.long,
            device=q.device,
        )
    else:
        block_access_log = None
        block_access_score = None
        block_access_count = None

    assert len(q.stride()) == 5  # BSZ, MAX_TDST, BH, G, HID
    if k is not None:
        assert len(k.stride()) == 4  # BSZ, MAX_TSRC, KV_HEAD, HID
    if args.k_cache is not None:
        assert args.k_cache.ndim == 4  # N_PAGES, PAGE_SIZE, KV_HEAD, HID
        assert args.block_table.ndim == 2  # BSZ, N_PAGES
        assert args.cache_seq_lens.ndim == 1  # BSZ
    assert len(indices.stride()) == 3
    assert len(ks.stride()) == 2
    assert len(group_sizes.stride()) == 3
    assert len(t_group_sizes.stride()) == 2
    if indices_seed is not None:
        assert len(indices_seed.stride()) == 3, indices_seed.shape
        assert len(ks_seed.stride()) == 2
        assert (
            indices_seed.shape == indices.shape
        ), f"{indices_seed.shape} == {indices.shape}"
        assert ks_seed.shape == ks.shape, f"{ks_seed.shape} == {ks.shape}"
        indices_seed = indices_seed // args.block_size_k
    if args.rope_cos is not None:
        assert len(args.rope_cos.stride()) == 2, args.rope_cos.shape
        assert len(args.rope_sin.stride()) == 2, args.rope_cos.shape

    assert args.sample_method in [
        "first",
        "last",
        "center",
        "position",
        "random",
        "oracle",
    ]
    assert position_ids.ndim == 2, position_ids.shape

    vertical_attention_mask = None
    diagonal_attention_mask = None
    adding_snap_kv = (
        args.add_snap_kv
        and (isinstance(q, Tensor) and isinstance(k, Tensor))
        and (q.shape[1] > 1)
    )
    if adding_snap_kv:
        observation_window = args.snap_kv_obs_window
        snap_kv_k = args.snap_kv_vert_k
        snap_kv_kernel_size = args.snap_kv_kernel_size
        diag_kv_k = args.snap_kv_diag_k
        diag_kv_kernel_size = args.snap_kv_kernel_size

        # TODO: fuse this
        obs_q = q.view(q.shape[0], q.shape[1], -1, q.shape[-1])[
            :, -observation_window:
        ]  # NOTE: merge topk-group dim
        obs_k = k.view(k.shape[0], k.shape[1], -1, k.shape[-1])[:, :]
        if HEAD != KV_HEAD:
            obs_k = obs_k.repeat_interleave(HEAD // KV_HEAD, dim=2)
        snap_attn_weights = obs_q.permute(0, 2, 1, 3) @ obs_k.permute(0, 2, 3, 1)
        snap_attn_weights = snap_attn_weights.mean(dim=1)
        # TODO: fuse this

        snap_vote = snap_attn_weights.mean(dim=1, keepdim=True)
        snap_vote[:, :, : args.sink_token_size].fill_(torch.finfo(snap_vote.dtype).min)
        snap_vote[:, :, -args.sliding_window_size - observation_window :].fill_(
            torch.finfo(snap_vote.dtype).min
        )
        snap_kv_kernel_size = max(
            snap_kv_kernel_size, 1 + 2 * args.block_size_k_after_masking
        )
        snap_pool_stride = max(args.block_size_k, args.block_size_k_after_masking)
        snap_pool_vote = F.max_pool1d(
            snap_vote,
            kernel_size=snap_kv_kernel_size,
            stride=snap_pool_stride,
            padding=snap_kv_kernel_size // 2,
        )
        snap_indices = snap_pool_vote.topk(
            min(
                snap_pool_vote.shape[-1],
                snap_kv_k // max(args.block_size_k_after_masking, args.block_size_k),
            ),
            dim=-1,
            sorted=False,
        ).indices.to(torch.int32)
        snap_indices.mul_(snap_pool_stride)
        snap_indices = (
            snap_indices.view(BSZ, 1, -1)
            .repeat_interleave(HEAD, dim=0)
            .expand(BSZ * HEAD, BDST, -1)
        )

        diag_vote = sum_all_diagonal(snap_attn_weights)
        diag_vote[:, :, : args.sink_token_size].fill_(torch.finfo(snap_vote.dtype).min)
        diag_vote[:, :, -args.sliding_window_size - observation_window :].fill_(
            torch.finfo(snap_vote.dtype).min
        )
        diag_pool_stride = max(
            args.block_size_k, args.block_size_k_after_masking, args.block_size_q
        )
        diag_kv_kernel_size = max(diag_kv_kernel_size, 1 + 2 * diag_pool_stride)
        diag_pool_vote = F.max_pool1d(
            diag_vote,
            kernel_size=diag_kv_kernel_size,
            stride=diag_pool_stride,
            padding=diag_kv_kernel_size // 2,
        )
        diag_indices = diag_pool_vote.topk(
            min(
                diag_pool_vote.shape[-1],
                diag_kv_k
                // max(
                    args.block_size_k_after_masking,
                    args.block_size_k,
                    args.block_size_q,
                ),
            ),
            dim=-1,
            sorted=False,
        ).indices.to(torch.int32)
        # BUG: what is happend here? why i have to sub 2?
        diag_indices.sub_(2).mul_(
            diag_pool_stride
        )  # - (diag_kv_kernel_size // diag_pool_stride * diag_pool_stride)
        diag_indices = (
            diag_indices[:, :, :, None].to(torch.int32)
            + (
                torch.arange(
                    0,
                    args.block_size_q,
                    max(args.block_size_k_after_masking, args.block_size_k),
                    device=indices.device,
                )
                - (args.block_size_q // 2)
            ).to(torch.int32)[None, None, None, :]
        ).view(diag_indices.shape[0], -1)
        diag_indices = (
            diag_indices.view(BSZ, 1, -1)
            .repeat_interleave(HEAD, dim=0)
            .expand(BSZ * HEAD, BDST, -1)
        )
        diag_indices = (
            diag_indices
            - torch.flip(
                torch.arange(
                    0,
                    BDST * args.block_size_q,
                    args.block_size_q,
                    device=indices.device,
                    dtype=diag_indices.dtype,
                ),
                dims=(0,),
            )[None, :, None]
        )
        diag_indices.clamp_min_(0)

        if os.getenv("HIP_SNAP_KV_NO_OVERLAP", "0") == "1":
            assert vertical_attention_mask is None
            vertical_attention_mask = torch.ones(
                (BSZ * HEAD, MAX_TSRC), device=q.device, dtype=torch.bool
            )
            vertical_attention_mask.scatter_(
                dim=1,
                index=(
                    snap_indices[:, -1, :, None]
                    + torch.arange(
                        0,
                        max(args.block_size_k, args.block_size_k_after_masking),
                        device=q.device,
                    )[None, None, :]
                ).view(BSZ * HEAD, -1),
                value=0,
            )

            assert diagonal_attention_mask is None
            diagonal_attention_mask = torch.ones(
                (BSZ * HEAD, MAX_TSRC), device=q.device, dtype=torch.bool
            )
            diagonal_attention_mask.scatter_(
                dim=1,
                index=(
                    diag_indices[:, -1, :, None]
                    + torch.arange(
                        0,
                        max(args.block_size_k, args.block_size_k_after_masking),
                        device=q.device,
                    )[None, None, :]
                ).view(BSZ * HEAD, -1),
                value=0,
            )

    # launch kernels
    # print('init in', indices[0, -1, :10])
    # if indices_seed is not None:
    #     print('init ins', indices_seed[0, -1, :10])
    BLOCK_MASK_BLOCK_K = triton.next_power_of_2(mask_block_k)

    assert (args.sink_token_size % args.block_size_k) == 0
    assert (args.sliding_window_size % args.block_size_k) == 0
    if group_size_seed is None:
        grid = (B, BDST, G)
        # print('init grid', grid)
        pre_device = torch.get_default_device()
        torch.set_default_device(indices.device)
        masking_iteration_draft_cuda_initialize[grid](
            indices_seed,
            *(indices_seed.stride() if indices_seed is not None else (0, 0, 0)),
            ks_seed,
            *(ks_seed.stride() if ks_seed is not None else (0, 0)),
            position_ids,
            *position_ids.stride(),
            indices,
            *indices.stride(),
            ks,
            *ks.stride(),
            group_sizes,
            *group_sizes.stride(),
            t_group_sizes,
            *t_group_sizes.stride(),
            args.mask_k,
            args.sliding_window_size,
            args.sink_token_size,
            max_group_size_seed if max_group_size_seed is not None else MAX_TSRC,
            G,
            TDST,
            MAX_TSRC,
            HEAD,
            args.block_size_q,
            args.block_stride_q,
            args.block_size_k,
            args.is_causal,
            BLOCK_MASK_BLOCK_K,
            # num_warps=min(max(cdiv_python(BLOCK_MASK_BLOCK_K, 32), 1), 32),
            num_warps=1,
            num_stages=1,
        )
        torch.set_default_device(pre_device)
    else:
        indices.copy_(indices_seed)
        ks.copy_(ks_seed)
        assert (
            group_sizes.shape == group_size_seed.shape
        ), f"{group_sizes.shape} == {group_size_seed.shape}"
        group_sizes.copy_(group_size_seed)
        t_group_sizes = group_sizes.max(dim=-1)[0].float()
    # print('init in after', indices[0, 0, :10])
    # print('init in after', indices[0, -1, :10])
    # print('init gs after', group_sizes[0, 0, :10])
    # print('init gs after', group_sizes[0, :, 0])
    # print('init ks after', ks[0, :])
    # print('init pos', position_ids[:])

    dupped_indices = torch.empty(
        (B, BDST, indices.shape[-1] * 2),
        dtype=torch.int32,
        device=q.device,
    )
    dupped_group_sizes = torch.empty(
        (B, BDST, indices.shape[-1] * 2),
        dtype=torch.int32,
        device=q.device,
    )
    scores = torch.empty_like(dupped_indices, dtype=torch.bfloat16)
    probs = torch.empty_like(scores)
    if (scores_seed is not None) and args.sample_method == "first":
        scores_final = scores_seed.clone()
    else:
        scores_final = torch.zeros_like(indices, dtype=torch.bfloat16)

        # BLOCK_BK = 128 // block_size_k
        # grid = (triton.cdiv(indices.shape[-1], BLOCK_BK), BDST, B)

        BLOCK_BK = 256 // (args.block_size_k // args.block_stride_k) * G

        if HID == 256:
            BLOCK_BK //= 2

        if args.using_extend:
            BLOCK_BK //= 4

        assert B == BSZ * BH
        grid = (
            BH * triton.cdiv(indices.shape[-1], BLOCK_BK),
            BDST,
            BSZ,
        )

        # BUG: autotune ruin the access log
        # grid = lambda META: (triton.cdiv(indices.shape[-1], META['BLOCK_BK']), BDST, B)
        pre_device = torch.get_default_device()
        torch.set_default_device(q.device)
        masking_iteration_draft_cuda_initialize_score[grid](
            q,
            *q.stride(),
            k,
            *args.safe_stride(k, 4),
            position_ids,
            *position_ids.stride(),
            vertical_attention_mask,
            *args.safe_stride(vertical_attention_mask, 2),
            diagonal_attention_mask,
            *args.safe_stride(diagonal_attention_mask, 2),
            access_log_store_from,
            key_access_log,
            *args.safe_stride(key_access_log, 3),
            key_access_count,
            *args.safe_stride(key_access_count, 2),
            KEY_ACCESS_LEN,
            block_access_log,
            *args.safe_stride(block_access_log, 3),
            block_access_score,
            *args.safe_stride(block_access_score, 3),
            block_access_count,
            *args.safe_stride(block_access_count, 2),
            BLOCK_ACCESS_LEN,
            indices,
            *indices.stride(),
            scores_final,
            *scores_final.stride(),
            t_group_sizes,
            *t_group_sizes.stride(),
            indices_tdst,
            *indices_tdst_stride,
            args.sliding_window_size,
            indices.shape[-1],
            BH,
            G,
            TDST,
            MAX_TSRC,
            HID,
            KV_HEAD_REPEAT,
            args.mask_k,
            *args.args_extend(),
            *args.args_sparq(),
            *args.args_paged_kv_cache(),
            *args.args_offload_cache(is_masking=True),
            args.is_causal,
            *args.args_bq_bsq_bk_bsk(),
            BLOCK_BK,
            num_warps=4,
            num_stages=2,
        )
        torch.set_default_device(pre_device)

        # print('-- after initialize')
        # print(scores.shape, key_access_log.shape, key_access_count.shape)
        # print('access count', key_access_count[0])
        # print('access log', key_access_log[0, -1, :key_access_count[0, -1].item()].tolist())
    scores_cached = args.sample_method in ["first", "last"]
    # scores_cached = False

    BLOCK_BK = 256 // 2 // args.block_size_k
    assert BLOCK_BK > 0
    BLOCK_HID = HID
    assert (HID % BLOCK_HID) == 0

    # print(indices[0, -10])
    # print(ks[0, -10])
    # assert indices[0, -10].shape == torch.unique(indices[0, -10]).shape, f'{indices[0, -10].shape} == {torch.unique(indices[0, -10]).shape}'

    topk_indices = None

    # max_group_size = max_group_size

    topk_indices = torch.empty(
        (probs.shape[0], probs.shape[1], mask_block_k * G),
        device=probs.device,
        dtype=torch.int32,
    )
    BLOCK_SCORE = triton.next_power_of_2(scores.shape[-1])

    using_fused_iteration = True
    if using_fused_iteration:
        assert args.score_head_group_size == 1

        if not scores_cached:
            BLOCK_BK = 128 // (args.block_size_k // args.block_stride_k)
        else:
            BLOCK_BK = 128 // (args.block_size_k // args.block_stride_k) // 2
        # BLOCK_BK = indices.shape[-1]
        # BLOCK_BK = indices.shape[-1] // 4

        # BLOCK_BK = indices.shape[-1] // 4
        if HID == 256:
            BLOCK_BK //= 2

        if args.using_extend:
            BLOCK_BK //= 4

        GROUP_BDST = 1
        GROUP_BH = 1

        assert (BH % GROUP_BH) == 0
        assert B == BSZ * BH

        # grid = (BH, triton.cdiv(BDST, GROUP_BDST), BSZ,)
        # grid = (triton.cdiv(BDST, GROUP_BDST), BSZ, BH,)
        # grid = (B, triton.cdiv(BDST, GROUP_BDST),)

        # grid = (
        #     triton.cdiv(BDST, GROUP_BDST) * BH * BSZ,
        # )

        grid = (GROUP_BH * triton.cdiv(BDST, GROUP_BDST), BH // GROUP_BH, BSZ)

        pre_device = torch.get_default_device()
        torch.set_default_device(q.device)
        masking_iteration_draft_cuda_fused[grid](
            q,
            *q.stride(),
            k,
            *args.safe_stride(k, 4),
            position_ids,
            *position_ids.stride(),
            vertical_attention_mask,
            *args.safe_stride(vertical_attention_mask, 2),
            diagonal_attention_mask,
            *args.safe_stride(diagonal_attention_mask, 2),
            access_log_store_from,
            key_access_log,
            *args.safe_stride(key_access_log, 3),
            key_access_count,
            *args.safe_stride(key_access_count, 2),
            KEY_ACCESS_LEN,
            block_access_log,
            *args.safe_stride(block_access_log, 3),
            block_access_score,
            *args.safe_stride(block_access_score, 3),
            block_access_count,
            *args.safe_stride(block_access_count, 2),
            BLOCK_ACCESS_LEN,
            indices,
            *indices.stride(),
            ks,
            *ks.stride(),
            group_sizes,
            *group_sizes.stride(),
            dupped_indices,
            *dupped_indices.stride(),
            dupped_group_sizes,
            *dupped_group_sizes.stride(),
            scores,
            *scores.stride(),
            scores_final,
            *scores_final.stride(),
            scores_cached,
            probs,
            *probs.stride(),
            topk_indices,
            *topk_indices.stride(),
            t_group_sizes,
            *t_group_sizes.stride(),
            indices_tdst,
            *indices_tdst_stride,
            args.mask_k,
            args.sink_token_size,
            args.sliding_window_size,
            BH,
            G,
            # TDST,
            # TSRC,
            # cdiv_python(TDST, args.block_size_q),
            # cdiv_python(TSRC, args.block_size_k),
            TDST,
            MAX_TSRC,
            cdiv_python(TDST, args.block_size_q),
            MAX_BSRC,
            mask_block_k,
            HID,
            random.randint(0, 1024 * 1024),
            args.sample_method,
            args.branch_method,
            KV_HEAD_REPEAT,
            *args.args_extend(),
            *args.args_sparq(),
            *args.args_paged_kv_cache(),
            *args.args_offload_cache(is_masking=True),
            args.is_causal,
            *args.args_bq_bsq_bk_bsk(),
            BLOCK_BK,
            BLOCK_SCORE,
            GROUP_BDST,
            GROUP_BH,
            TDST_NEXT_POWER_OF_2=triton.next_power_of_2(TDST),
            indices_bk_len=indices.shape[-1],
            probs_bk_len=probs.shape[-1],
            # num_warps=4,
            # num_stages=2,
        )
        torch.set_default_device(pre_device)
    else:
        raise NotImplementedError()
        i_iteration = 0
        while max_group_size > 1:
            BLOCK_BK = 128 // block_size_k
            grid = (
                triton.cdiv(indices.shape[-1], BLOCK_BK),
                BDST,
                B,
            )
            masking_iteration_draft_cuda_dup_and_score[grid](
                q,
                *q.stride(),
                k,
                *k.stride(),
                position_ids,
                *position_ids.stride(),
                rope_cos,
                *(rope_cos.stride() if rope_cos is not None else (0, 0)),
                rope_sin,
                *(rope_sin.stride() if rope_sin is not None else (0, 0)),
                key_access_log,
                *(key_access_log.stride() if key_access_log is not None else (0, 0, 0)),
                key_access_count,
                *(
                    key_access_count.stride()
                    if key_access_count is not None
                    else (0, 0)
                ),
                KEY_ACCESS_LEN,
                block_access_log,
                *args.safe_stride(block_access_log, 3),
                block_access_score,
                *args.safe_stride(block_access_score, 3),
                block_access_count,
                *args.safe_stride(block_access_count, 2),
                BLOCK_ACCESS_LEN,
                indices,
                *indices.stride(),
                ks,
                *ks.stride(),
                group_sizes,
                *group_sizes.stride(),
                dupped_indices,
                *dupped_indices.stride(),
                dupped_group_sizes,
                *dupped_group_sizes.stride(),
                scores,
                *scores.stride(),
                scores_final,
                *scores_final.stride(),
                scores_cached,
                t_group_sizes,
                *t_group_sizes.stride(),
                indices_tdst,
                *indices_tdst_stride,
                mask_k,
                sliding_window_size,
                G,
                TDST,
                TSRC,
                mask_block_k,
                HID,
                random.randint(0, 1024 * 1024),
                sample_method,
                branch_method,
                using_extend,
                self_extend_neighboor_window,
                self_extend_group_size,
                using_sparq,
                sparq_hid,
                sparq_ind,
                *(sparq_ind.stride() if sparq_ind is not None else (0, 0, 0, 0)),
                block_size_q,
                block_stride_q,
                block_size_k,
                block_stride_k,
                BLOCK_BK,
                max_group_size,
                i_iteration,
                num_warps=(2 if scores_cached else 4) * G,
                num_stages=max(1, 4 // G),
            )

            # NOTE: because of softmax, we cannot fuse everything...
            # BLOCK_SCORE = min(1024, mask_block_k * G)
            grid = (BDST, B)
            masking_iteration_draft_cuda_partial_softmax[grid](
                scores,
                *scores.stride(),
                dupped_indices,
                *dupped_indices.stride(),
                dupped_group_sizes,
                *dupped_group_sizes.stride(),
                probs,
                *probs.stride(),
                sink_token_size,
                mask_block_k,
                G,
                scores.shape[-1],
                BSRC,
                block_size_k,
                BLOCK_SCORE,
                num_warps=min(32, BLOCK_SCORE // 32),
            )

            if score_head_group_size > 1:
                assert score_head_group_size <= B
                assert (B % score_head_group_size) == 0
                scores_max = scores.view(
                    B // score_head_group_size,
                    score_head_group_size,
                    BDST,
                    scores.shape[-1],
                ).min(1, keepdim=True)[0]
                scores = scores_max.repeat(1, score_head_group_size, 1, 1).view(
                    -1, scores_max.shape[-2], scores_max.shape[-1]
                )

            # also villan
            BLOCK_BDST = 1
            grid = (
                triton.cdiv(BDST, BLOCK_BDST),
                B,
            )
            masking_iteration_draft_cuda_argsort[grid](
                probs,
                *probs.stride(),
                topk_indices,
                *topk_indices.stride(),
                t_group_sizes,
                *t_group_sizes.stride(),
                BDST,
                probs.shape[-1],
                mask_block_k * G,
                BLOCK_BDST,
                num_warps=min(32, max(1, (probs.shape[-1] * BLOCK_BDST) // 256)),
                num_stages=8,
            )

            BLOCK_BK = indices.shape[-1]
            grid = (
                triton.cdiv(indices.shape[-1], BLOCK_BK),
                BDST,
                B,
            )
            masking_iteration_draft_cuda_gather[grid](
                indices,
                *indices.stride(),
                group_sizes,
                *group_sizes.stride(),
                scores_final,
                *scores_final.stride(),
                dupped_indices,
                *dupped_indices.stride(),
                dupped_group_sizes,
                *dupped_group_sizes.stride(),
                scores,
                *scores.stride(),
                topk_indices,
                *topk_indices.stride(),
                t_group_sizes,
                *t_group_sizes.stride(),
                G,
                mask_block_k,
                BLOCK_BK,
            )

            # indices, indices_sort_mapping = torch.sort(indices, dim=-1, stable=False)
            # scores_final = scores_final\
            #     .gather(index=indices_sort_mapping, dim=-1)
            # group_sizes = group_sizes\
            #     .gather(index=indices_sort_mapping, dim=-1)

            if sample_method in ["first", "last", "center", "half"]:
                scores_cached = True

            if branch_method == "random":
                max_group_size = max_group_size * 0.7
                if max_group_size > 1.0:
                    t_group_sizes.mul_(0.7)
            else:
                max_group_size = max_group_size * 0.5
                if max_group_size > 1.0:
                    t_group_sizes.mul_(0.5)
            i_iteration += 1

    indices.mul_(args.block_size_k)

    if adding_snap_kv:
        # concat and union and update ks
        # this peak memory too much
        indices_out = torch.empty(
            (
                indices.shape[0],
                indices.shape[1],
                indices.shape[2] + snap_indices.shape[2] + diag_indices.shape[2],
            ),
            dtype=torch.int32,
            device=indices.device,
        )
        chunk_count = cdiv_python(
            indices_out.shape[1] * indices_out.shape[2], 2048 * 4096
        )
        chunk_size = cdiv_python(indices.shape[1], chunk_count)
        ks_out = torch.empty_like(ks)
        for i in range(0, indices.shape[1], chunk_size):
            i_start = i
            i_end = min(i_start + chunk_size, indices.shape[1])
            t_indices = (
                torch.cat(
                    [
                        indices[:, i_start:i_end],
                        snap_indices[:, i_start:i_end],
                        diag_indices[:, i_start:i_end],
                    ],
                    dim=-1,
                )
                .sort(dim=-1)
                .values
            )
            t_unique_mask = torch.roll(t_indices, shifts=1, dims=-1) != t_indices
            t_indices = torch.where(t_unique_mask, t_indices, MAX_TSRC * G)
            t_indices = t_indices.sort(dim=-1).values
            t_ks = t_unique_mask.int().sum(-1)
            indices_out[:, i_start:i_end].copy_(t_indices, non_blocking=True)
            ks_out[:, i_start:i_end].copy_(t_ks, non_blocking=True)
        indices = indices_out
        ks = ks_out
        # print('indices shape', indices.shape)

        scores_final = None
    elif args.add_approx_k_window:
        approx_k = cdiv_python(args.approx_k, args.block_size_k)
        approx_k_window = cdiv_python(args.approx_k_window, args.block_size_k)
        _, selected_indices = torch.topk(scores_final, dim=-1, k=approx_k)
        approx_top_k_indices = indices.gather(dim=-1, index=selected_indices)
        approx_top_k_indices = approx_top_k_indices[:, :, :, None] + (
            torch.arange(0, approx_k_window, device=indices.device)
            - approx_k_window // 2
        )
        approx_top_k_indices = approx_top_k_indices.view(
            indices.shape[0], indices.shape[1], -1
        )
        approx_top_k_indices = approx_top_k_indices.clamp_min_(0)
        indices = torch.cat([indices, approx_top_k_indices], dim=-1)

        # union
        indices = indices.sort(dim=-1).values
        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(unique_mask, indices, MAX_TSRC * G)
        indices = indices.sort(dim=-1).values

        ks = unique_mask.int().sum(-1)

        # NOTE: before this sort, indices are sorted by imporatnce of each block
        # indices, indices_sort_mapping = torch.sort(indices, dim=-1, stable=False)

        scores_final = None
        # scores_final = scores_final\
        #     .gather(index=indices_sort_mapping, dim=-1)
    else:
        # NOTE: before this sort, indices are sorted by imporatnce of each block
        indices, indices_sort_mapping = torch.sort(indices, dim=-1, stable=False)

        scores_final = scores_final.gather(index=indices_sort_mapping, dim=-1)

    # scores_final = None

    ks_count, ks_start_end = masking_iteration_draft_python_epilog(
        indices, ks, mask_block_k, MAX_TSRC, B, BDST, G
    )

    # assert indices[0, -10].shape == torch.unique(indices[0, -10]).shape, f'{indices[0, -10].shape} == {torch.unique(indices[0, -10]).shape}'
    # t = indices[0, 16]
    # c = ks[0, 16]
    # tu = torch.unique(t)
    # print(t)
    # print(tu)
    # print(t.shape, tu.shape, c)

    return (
        indices,
        ks,
        ks_count,
        ks_start_end,
        scores_final,
        group_sizes,
        key_access_log,
        key_access_count,
        block_access_log,
        block_access_score,
        block_access_count,
    )


@triton.jit
def block_sparse_attention_cuda_step(
    # QKV
    queries,
    keys,
    keys_rot,
    values,
    # indices
    idx_tsrc,
    mask_tsrc,
    idx_tdst,
    mask_tdst,
    # rolling value
    acc,
    l_i,
    m_i,
    # TDST,
    # TSRC,
    sliding_window_size,
    sink_token_size,
    mask_k,
    EXCLUDE_SLIDING_WINDOW: tl.constexpr,
    HAS_FIRST_TOKEN: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    model_context_length,
    idx_bk,
    pos_tdst,
    idx_hid,
    IS_CAUSAL: tl.constexpr,
    HID: tl.constexpr,
    BLOCK_TQ,
    BLOCK_TK,
    BLOCK_SIZE_K: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr = DEFAULT_EXTEND_BACKEND,
):
    # keys := [BLOCK_HID: hid, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    # queries := [BLOCK_SIZE_Q: tdst, BLOCK_HID: hid]
    # scores := [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]

    # keys = tl.load(
    #     K +\
    #         (idx_n // KV_REPEAT_INTERLEAVE) * stride_k_n +\
    #         idx_tsrc[None, :] * stride_k_tsrc +\
    #         idx_hid[:, None] * stride_k_hid,
    #     mask = mask_tsrc[None, :] & mask_hid[:, None],
    #     other = 0,
    # )

    # queries_max = tl.maximum(1.0, tl.max(tl.abs(queries)).to(tl.float32))
    # keys_max = tl.maximum(1.0, tl.max(tl.abs(keys)).to(tl.float32))
    # queries_scale = (1.0 / queries_max)
    # keys_scale = (1.0 / keys_max)
    # qk = tl.dot(
    #     # (queries * queries_scale).to(queries.dtype),
    #     # (keys * keys_scale).to(keys.dtype),
    #     queries, keys,
    #     allow_tf32=True,
    # ).to(tl.float32) * 1.44269504 # * queries_max * keys_max)

    if USING_EXTEND:
        if EXTEND_BACKEND == "self_extend":
            assert COS is not None
            assert SIN is not None

            # dynamic_group_size = tl.maximum(1.0, tl.math.ceil(tl.max(pos_tdst / 8192))).to(tl.int32)
            dynamic_group_size = extend_group_size

            old_tsrc = idx_tsrc
            mask_tsrc_window = idx_tsrc >= (
                tl.min(tl.where(mask_tdst, (pos_tdst - 1), 987654321))
                - extend_window_size
            )
            new_tsrc = tl.where(
                mask_tsrc_window, old_tsrc, old_tsrc // dynamic_group_size
            )

            keys = keys.trans(1, 0)
            keys = adjust_rope(
                keys,
                old_tsrc,
                new_tsrc,
                mask_tsrc,
                idx_hid,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                BLOCK_TK,
                HID,
                NEED_APPLY_ROPE,
            )
            keys = tl.trans(keys, 1, 0)
            keys = keys * mask_tsrc[None, :]

            old_tdst = pos_tdst - 1
            new_tdst = old_tdst // dynamic_group_size

            queries_grouped = adjust_rope(
                queries,
                old_tdst,
                new_tdst,
                mask_tdst,
                idx_hid,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                BLOCK_TQ,
                HID,
                NEED_APPLY_ROPE,
            )
            queries_grouped = queries_grouped * mask_tdst[:, None]

            t_window = tl.dot(
                (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                    queries.dtype
                ),
                (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                    queries.dtype
                ),
                allow_tf32=True,
            )
            t_grouped = tl.dot(
                (
                    queries_grouped * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))
                ).to(queries_grouped.dtype),
                (keys.to(queries_grouped.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                    queries_grouped.dtype
                ),
                allow_tf32=True,
            )
            qk = (
                tl.where(
                    mask_tsrc_window[None, :],
                    t_window,
                    t_grouped,
                ).to(tl.float32)
                * 1.44269504
            )
        elif (EXTEND_BACKEND == "streaming") or (EXTEND_BACKEND == "dynamic_extend"):
            pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst - 1, 987654321))
            if not NEED_APPLY_ROPE:
                if (
                    (pos_tdst_min >= model_context_length) and EXCLUDE_SLIDING_WINDOW
                ) and True:
                    assert COS is not None
                    assert SIN is not None

                    if HAS_FIRST_TOKEN:
                        old_tdst = pos_tdst - 1
                        new_tdst = tl.minimum(
                            old_tdst, sliding_window_size + mask_k + sink_token_size - 1
                        )

                        queries_adjusted = adjust_rope(
                            queries,
                            old_tdst,
                            new_tdst,
                            mask_tdst,
                            idx_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            BLOCK_TQ,
                            HID,
                            NEED_APPLY_ROPE,
                        )

                        keys_adjusted = keys
                    else:
                        old_tsrc = idx_tsrc
                        new_tsrc = tl.ravel(
                            (idx_bk * BLOCK_SIZE_K)[:, None]
                            + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        )
                        new_tsrc = tl.maximum(
                            0,
                            new_tsrc
                            + pos_tdst_min
                            - sliding_window_size
                            - sink_token_size
                            - mask_k
                            - BLOCK_TQ
                            + 1,
                        )

                        keys_adjusted = keys.trans(1, 0)
                        keys_adjusted = adjust_rope(
                            keys_adjusted.to(queries.dtype),
                            old_tsrc,
                            new_tsrc,
                            mask_tsrc,
                            idx_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            BLOCK_TK,
                            HID,
                            NEED_APPLY_ROPE,
                        )
                        keys_adjusted = tl.trans(keys_adjusted, 1, 0)

                        queries_adjusted = queries
                else:
                    if NEED_APPLY_ROPE:
                        queries = adjust_rope(
                            queries.to(tl.float32),
                            pos_tdst - 1,
                            pos_tdst - 1,
                            mask_tdst,
                            idx_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            BLOCK_TQ,
                            HID,
                            True,
                        ).to(queries.dtype)
                        queries_adjusted = (queries * mask_tdst[:, None]).to(
                            queries.dtype
                        )

                        keys = tl.trans(
                            adjust_rope(
                                tl.trans(keys.to(tl.float32), 1, 0),
                                idx_tsrc,
                                idx_tsrc,
                                mask_tsrc,
                                idx_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
                                BLOCK_TK,
                                HID,
                                True,
                            ),
                            1,
                            0,
                        ).to(keys.dtype)
                        keys_adjusted = (keys * mask_tsrc[None, :]).to(keys.dtype)
            else:
                tl.static_assert(NEED_APPLY_ROPE)
                tl.static_assert(USING_EXTEND)
                # tl.static_assert(not EXCLUDE_SLIDING_WINDOW)

                if EXCLUDE_SLIDING_WINDOW:
                    # new_tsrc = tl.ravel((idx_bk * BLOCK_SIZE_K)[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :])
                    # # new_tsrc = tl.minimum(new_tsrc, sliding_window_size + sink_token_size + mask_k - 1)
                    # # new_tsrc = tl.minimum(new_tsrc, pos_tdst_min + tl.sum(mask_tdst.to(tl.int32)) - 1)
                    # new_tsrc = tl.maximum(0, new_tsrc)

                    pos_tdst_max = pos_tdst_min + tl.sum(mask_tdst.to(tl.int32))

                    if EXTEND_BACKEND == "streaming":
                        # streaming
                        new_tsrc = tl.ravel(
                            (idx_bk * BLOCK_SIZE_K)[:, None]
                            + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        )
                        new_tsrc = tl.maximum(
                            0,
                            new_tsrc
                            + pos_tdst_min
                            - sliding_window_size
                            - sink_token_size
                            - mask_k
                            + 1,
                        )
                        # new_tsrc = idx_tsrc
                    elif EXTEND_BACKEND == "dynamic_extend":
                        # dynamic extend
                        window = model_context_length // 4

                        # new_tsrc = tl.where(
                        #     (idx_tsrc >= (pos_tdst_max - window)) | (pos_tdst_max <= model_context_length),
                        #     idx_tsrc,
                        #     ((idx_tsrc - (pos_tdst_min - model_context_length)) * ((model_context_length - window) / (pos_tdst_min - window))).to(tl.int32) + (pos_tdst_min - model_context_length)
                        # )
                        new_tsrc = tl.where(
                            (idx_tsrc >= (pos_tdst_max - window))
                            | (pos_tdst_max <= model_context_length),
                            idx_tsrc,
                            (
                                (idx_tsrc + window - pos_tdst_min)
                                * (
                                    (model_context_length - window)
                                    / (pos_tdst_min - window)
                                )
                            ).to(tl.int32)
                            + pos_tdst_min
                            - window,
                        )
                        new_tsrc = tl.maximum(
                            pos_tdst_max - model_context_length, new_tsrc
                        )
                    else:
                        raise Exception()
                else:
                    new_tsrc = idx_tsrc
                # new_tsrc = idx_tsrc

                # keys_adjusted = keys.trans(1, 0)
                # keys_adjusted = adjust_rope(
                #     keys_adjusted.to(queries.dtype),
                #     new_tsrc,
                #     new_tsrc,
                #     mask_tsrc,
                #     idx_hid,
                #     COS, stride_cos_t, stride_cos_hid,
                #     SIN, stride_sin_t, stride_sin_hid,
                #     BLOCK_TK,
                #     HID,
                #     True,
                # )
                # keys_adjusted = tl.trans(keys_adjusted, 1, 0)

                keys = keys.to(queries.dtype)
                keys_rot = keys_rot.to(queries.dtype)

                # cos_new = tl.load(
                #     COS +\
                #         new_tsrc[None, :].to(tl.int64) * stride_cos_t +\
                #         idx_hid[:, None] * stride_cos_hid,
                #     mask=mask_tsrc[None, :],
                #     other=0.0,
                # ).to(keys.dtype)
                # sin_new = tl.load(
                #     SIN +\
                #         new_tsrc[None, :].to(tl.int64) * stride_sin_t +\
                #         idx_hid[:, None] * stride_sin_hid,
                #     mask=mask_tsrc[None, :],
                #     other=0.0,
                # ).to(keys.dtype)

                cos_new = tl.load(
                    COS
                    + new_tsrc[None, :].to(tl.int64) * stride_cos_t
                    + (tl.arange(0, HID) % (HID // 2))[:, None] * stride_cos_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                ).to(keys.dtype)
                sin_new = tl.load(
                    SIN
                    + new_tsrc[None, :].to(tl.int64) * stride_sin_t
                    + (tl.arange(0, HID) % (HID // 2))[:, None] * stride_sin_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                ).to(keys.dtype)

                if EXCLUDE_SLIDING_WINDOW:
                    if EXTEND_BACKEND == "dynamic_extend":
                        streaming_tsrc = tl.ravel(
                            (idx_bk * BLOCK_SIZE_K)[:, None]
                            + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        )
                        streaming_tsrc = tl.maximum(
                            0,
                            streaming_tsrc
                            + pos_tdst_min
                            - sliding_window_size
                            - sink_token_size
                            - mask_k
                            + 1,
                        )

                        cos_zero = tl.load(
                            COS
                            + streaming_tsrc[None, :].to(tl.int64) * stride_cos_t
                            + (tl.arange(0, HID) % (HID // 2))[:, None]
                            * stride_cos_hid,
                            # mask=mask_tsrc[None, :],
                            # other=0.0,
                        ).to(keys.dtype)
                        sin_zero = tl.load(
                            SIN
                            + streaming_tsrc[None, :].to(tl.int64) * stride_sin_t
                            + (tl.arange(0, HID) % (HID // 2))[:, None]
                            * stride_sin_hid,
                            # mask=mask_tsrc[None, :],
                            # other=0.0,
                        ).to(keys.dtype)

                        cos_new = (cos_zero * 0.75 + cos_new * 0.25).to(cos_new.dtype)
                        sin_new = (sin_zero * 0.75 + sin_new * 0.25).to(sin_new.dtype)

                # rope_theta = 500000.0
                # inv_freqs = ((tl.arange(0, HID) * 2) % HID)
                # 1.0 / tl.extra.cuda.libdevice.fast_powf(
                #     rope_theta,
                #     ((tl.arange(0, HID) * 2) % HID).to(tl.float32) / HID
                # )
                # freqs = new_tsrc[None, :].to(tl.float32) * inv_freqs[:, None]
                # cos_new = tl.extra.cuda.libdevice.fast_cosf(freqs)
                # sin_new = tl.extra.cuda.libdevice.fast_sinf(freqs)

                # keys_rot = tl.where(
                #     (idx_hid + HID // 2)[:, None] < HID,
                #     -keys_rot,
                #     keys_rot
                # )
                keys_rot = keys_rot * (
                    ((idx_hid + HID // 2)[:, None] < HID) * (-2) + 1
                ).to(keys_rot.dtype)

                keys_adjusted = (keys * cos_new + keys_rot * sin_new).to(keys.dtype)

                # error = tl.sum(tl.abs(keys_adjusted * mask_tsrc[None, :] - keys_adjusted_ * mask_tsrc[None, :]))
                # tl.device_print('err', error)
                # keys_adjusted = keys_adjusted_

                queries_adjusted = queries
                # pass

            qk = tl.dot(
                queries_adjusted
                * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0))).to(queries.dtype),
                keys_adjusted * (1 / tl.sqrt(tl.sqrt(HID * 1.0))).to(queries.dtype),
                out_dtype=tl.float32,
                allow_tf32=True,
            ).to(tl.float32)
            if LOGIT_SOFTCAP is not None:
                qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
            qk = qk * 1.44269504
        elif EXTEND_BACKEND == "dynamic_extend":
            assert COS is not None
            assert SIN is not None

            pos_tdst_min = tl.min(
                tl.where(mask_tdst, tl.maximum(0, pos_tdst - 1), 987654321)
            ) + tl.sum(mask_tdst.to(tl.int32))
            if (pos_tdst_min >= model_context_length) and EXCLUDE_SLIDING_WINDOW:
                old_tdst = pos_tdst - 1
                new_tdst = tl.minimum(model_context_length - 1, old_tdst)
                # new_tdst = old_tdst // 16

                queries = adjust_rope(
                    queries,
                    old_tdst,
                    new_tdst,
                    mask_tdst & (old_tdst != 0),
                    idx_hid,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    BLOCK_TQ,
                    HID,
                    NEED_APPLY_ROPE,
                ).to(queries.dtype)
                queries = (queries * mask_tdst[:, None]).to(queries.dtype)

                if not HAS_FIRST_TOKEN:
                    old_tsrc = idx_tsrc

                    # src_scale = (
                    #     (pos_tdst_min + 1 - sink_token_size - sliding_window_size) /\
                    #     (model_context_length - sink_token_size - sliding_window_size)
                    # )
                    # new_tsrc = (
                    #     (old_tsrc - sink_token_size) / src_scale + sink_token_size
                    # ).to(tl.int32)

                    # new_tsrc = tl.where(
                    #     (old_tsrc - pos_tdst_min + model_context_length - 1) > (model_context_length // 2),
                    #     old_tsrc - pos_tdst_min + model_context_length - 1,
                    #     ((old_tsrc - sink_token_size) * ((model_context_length // 2 - 1 - sink_token_size - sliding_window_size) / (pos_tdst_min - sink_token_size - sliding_window_size))).to(tl.int32) + sink_token_size
                    # )

                    new_tsrc = tl.where(
                        (old_tsrc - pos_tdst_min + model_context_length - 1)
                        > (model_context_length // 2),
                        old_tsrc - pos_tdst_min + model_context_length - 1,
                        (
                            (old_tsrc - sink_token_size)
                            * (
                                (model_context_length // 2)
                                / (pos_tdst_min - model_context_length // 2)
                            )
                        ).to(tl.int32)
                        + sink_token_size,
                        # ((old_tsrc - num_sinks) * (model_context_length / real_pos_tdst_min)).to(tl.int32) + num_sinks
                    )

                    # new_tsrc = old_tsrc // 16

                    keys_adjusted = keys.trans(1, 0)
                    keys_adjusted = adjust_rope(
                        keys_adjusted,
                        old_tsrc,
                        new_tsrc,
                        mask_tsrc & (old_tsrc != 0),
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TK,
                        HID,
                        NEED_APPLY_ROPE,
                    ).to(keys.dtype)
                    keys_adjusted = tl.trans(keys_adjusted, 1, 0).to(keys.dtype)
                    # keys_adjusted = (keys_adjusted * mask_tsrc[None, :]).to(keys.dtype)
                else:
                    keys_adjusted = keys
            else:
                keys_adjusted = keys

            qk = tl.dot(
                (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                    queries.dtype
                ),
                (
                    keys_adjusted.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))
                ).to(queries.dtype),
                out_dtype=tl.float32,
                allow_tf32=True,
            ).to(tl.float32)

            if LOGIT_SOFTCAP is not None:
                qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
            qk = qk * 1.44269504
        else:
            raise Exception()
    else:
        qk = tl.dot(
            (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                queries.dtype
            ),
            (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                queries.dtype
            ),
            out_dtype=tl.float32,
            allow_tf32=True,
        ).to(tl.float32)
        if LOGIT_SOFTCAP is not None:
            qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
        qk = qk * 1.44269504

    # qk_mask = (
    #     ((idx_tdst[:, None] + TSRC - TDST) < (idx_tsrc)[None, :]) |
    #     (~(mask_tdst[:, None] & mask_tsrc[None, :]))
    # )

    if IS_CAUSAL:
        if EXCLUDE_SLIDING_WINDOW:
            qk_mask = (
                ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                | ((pos_tdst - 1)[:, None] < (idx_tsrc + sliding_window_size)[None, :])
                | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
            )
        else:
            qk_mask = (
                ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                | ((pos_tdst - 1)[:, None] >= (idx_tsrc + sliding_window_size)[None, :])
                | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
            )
    else:
        qk_mask = ~(mask_tdst[:, None] & mask_tsrc[None, :])

    # qk = tl.where(
    #     qk_mask,
    #     float('-inf'),
    #     qk
    # )

    # qk += qk_mask * (-1.0e+6)

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])
    qk = qk - m_ij
    # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    p = tl.math.exp2(qk)

    p = tl.where(qk_mask, 0, p)
    # p *= ~qk_mask

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    l_ij = tl.sum(p, axis=1)

    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)
    # tl.device_print('ff', l_ij)
    l_i = (l_i * alpha + l_ij[:, None]).to(l_i.dtype)

    # -- update output accumulator --
    acc = acc * alpha.to(acc.dtype)

    # values = tl.load(
    #     V +\
    #         (idx_n // KV_REPEAT_INTERLEAVE) * stride_v_n +\
    #         idx_tsrc[:, None] * stride_v_tsrc +\
    #         idx_hid[None, :] * stride_v_hid,
    #     mask = mask_tsrc[:, None] & mask_hid[None, :],
    #     other = 0
    # )

    # update acc
    acc += tl.dot(
        p.to(queries.dtype),
        values.to(queries.dtype),
        out_dtype=tl.float32,
        allow_tf32=True,
    ).to(acc.dtype)

    # update m_i and l_i
    m_i = m_ij.to(m_i.dtype)

    return acc, l_i, m_i


def get_block_sparse_attention_configs():
    autotune_disabled = os.getenv("HIP_DISABLE_AUTOTUNE", "1") == "1"
    if autotune_disabled:
        device_name = torch.cuda.get_device_name()
        defaults = {
            "NVIDIA A100-SXM4-80GB": dict(
                num_warps=4,
                num_stages=2,
                maxnreg=256,
            ),
        }.get(device_name, dict(num_warps=4, num_stages=2, maxnreg=256))
        return [triton.Config({}, **defaults)]
    if os.getenv("HIP_DISABLE_AUTOTUNE_WARNINGS", "0") == "0":
        warnings.warn(
            "triton autotuning is activated. this should be disabled for faster startup. if you want set HIP_DISABLE_AUTOTUNE=1"
        )
    configs = []
    # for block_bk in [4, 8, 16, 32]:
    # for block_bk in [16, 32,]:
    for max_nreg in [128, 256, 512]:
        for num_warps in [4, 8]:
            for num_stages in [2, 4]:
                configs.append(
                    triton.Config(
                        {}, num_warps=num_warps, num_stages=num_stages, maxnreg=max_nreg
                    )
                )
    return configs


# def perf_model_block_sparse_attention(**kwargs):
#     block_bk = kwargs['BLOCK_BK']
#     block_k = kwargs['BLOCK_SIZE_K']
#     assert block_k <= 64, 'this will not good idea'
#     if ((block_bk * block_k) <= 64) and ((block_bk * block_k) >= 32):
#         return 0
#     return 999999999 # run might fails


@triton.autotune(
    configs=get_block_sparse_attention_configs(),
    key=[
        "BLOCK_SIZE_K",
        "BLOCK_SIZE_Q",
        "HID",
        "TDST_NEXT_POWER_OF_2",
    ],
    # prune_configs_by={
    #     'perf_model': perf_model_block_sparse_attention,
    #     'top_k': 24,
    # }
)
@triton.jit
def block_sparse_attention_cuda(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    V,
    stride_v_bsz,
    stride_v_tsrc,
    stride_v_head,
    stride_v_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS_START_END,
    stride_ks_start_end_b,
    stride_ks_start_end_bdst,
    stride_ks_start_end_g,
    CONTEXT,
    stride_context_bsz,
    stride_context_tdst,
    stride_context_head,
    stride_context_hid,
    HEAD: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    KV_HEAD_REPEAT: tl.constexpr,
    sliding_window_size: tl.constexpr,
    sink_token_size: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    model_context_length,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_V_TABLES,
    stride_offload_cache_v_tables_n,
    stride_offload_cache_v_tables_t,
    OFFLOAD_CACHE_V_BANKS,
    stride_offload_cache_v_banks_n,
    stride_offload_cache_v_banks_page,
    stride_offload_cache_v_banks_offset,
    stride_offload_cache_v_banks_hid,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    TDST_NEXT_POWER_OF_2,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HID: tl.constexpr,
    # autotuning parameters
    BLOCK_BK: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
):
    pid_bsz = tl.program_id(2).to(tl.int64)
    pid_bdst = tl.program_id(1).to(tl.int64)
    pid_head = tl.program_id(0).to(tl.int64)

    idx_bsz = pid_bsz.to(tl.int64)
    idx_head = pid_head
    idx_n = idx_bsz * HEAD + idx_head
    idx_b = idx_n // G
    idx_g = idx_n % G

    idx_bdst = pid_bdst
    if BLOCK_SIZE_Q < 16:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, 16)
        mask_tdst = (idx_tdst < MAX_TDST) & (tl.arange(0, 16) < BLOCK_SIZE_Q)
    else:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, BLOCK_SIZE_Q)
        mask_tdst = idx_tdst < MAX_TDST
    if IS_CAUSAL:
        pos_tdst = tl.load(
            POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
            mask=mask_tdst,
            other=0,
        )
    else:
        pos_tdst = tl.where(
            mask_tdst, tl.full((BLOCK_SIZE_Q,), value=MAX_TSRC, dtype=tl.int64), 0
        )

    idx_hid = tl.arange(0, HID)

    if BLOCK_SIZE_Q < 16:
        acc = tl.zeros((16, HID), dtype=tl.float32)
        m_i = tl.full((16, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((16, 1), 1.0, dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_Q, HID), dtype=tl.float32)
        m_i = tl.full((BLOCK_SIZE_Q, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_Q, 1), 1.0, dtype=tl.float32)

    range_start = tl.load(
        KS_START_END
        + idx_b * stride_ks_start_end_b
        + idx_bdst * stride_ks_start_end_bdst
        + idx_g * stride_ks_start_end_g
    )
    range_end = tl.load(
        KS_START_END
        + idx_b * stride_ks_start_end_b
        + idx_bdst * stride_ks_start_end_bdst
        + (idx_g + 1) * stride_ks_start_end_g
    )
    if BK <= 0:
        range_start = 0
        range_end = 0

    queries = tl.load(
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0.0,
        # cache_modifier='.cg',
        # eviction_policy='evict_last',
        # volatile=True,
    )
    if queries.dtype == tl.float8e5:
        queries = queries.to(tl.float16)

    if USING_EXTEND and NEED_APPLY_ROPE:
        rope_tdst = pos_tdst - 1
        # rope_tdst = tl.minimum(
        #     rope_tdst,
        #     sliding_window_size + sink_token_size + (range_end - range_start) * BLOCK_SIZE_K - 1
        # )

        # queries = adjust_rope(
        #     queries,
        #     rope_tdst,
        #     rope_tdst,
        #     mask_tdst,
        #     idx_hid,
        #     COS, stride_cos_t, stride_cos_hid,
        #     SIN, stride_sin_t, stride_sin_hid,
        #     BLOCK_SIZE_Q,
        #     HID,
        #     True,
        # ).to(queries.dtype)

        queries_rot = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + ((idx_hid + HID // 2) % HID)[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0.0,
            # cache_modifier='.cg',
            # eviction_policy='evict_last',
            # volatile=True,
        )
        if queries_rot.dtype == tl.float8e5:
            queries_rot = queries_rot.to(tl.float16)

        cos_new = tl.load(
            COS
            + rope_tdst[:, None].to(tl.int64) * stride_cos_t
            + (idx_hid % (HID // 2))[None, :] * stride_cos_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)
        sin_new = tl.load(
            SIN
            + rope_tdst[:, None].to(tl.int64) * stride_sin_t
            + (idx_hid % (HID // 2))[None, :] * stride_sin_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)

        # queries_rot = tl.where(
        #     (idx_hid + HID // 2)[None, :] < HID,
        #     -queries_rot,
        #     queries_rot
        # )
        queries_rot = queries_rot * (
            ((idx_hid + HID // 2)[None, :] < HID) * (-2) + 1
        ).to(queries_rot.dtype)

        queries = (queries * cos_new + queries_rot * sin_new).to(queries.dtype)

    if (BK > 0) and True:
        for i_bk in range(range_start, range_start + (BK * G), BLOCK_BK):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)
            mask_bk = (idx_bk < (range_start + BK * G)) & (idx_bk < range_end)

            if i_bk < range_end:
                idx_tsrc_start = tl.load(
                    INDICES
                    + idx_b * stride_indices_b
                    + idx_bdst * stride_indices_bdst
                    + idx_bk * stride_indices_bk,
                    mask=mask_bk,
                    # other=(MAX_TSRC + 1) * G,
                )
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC * G + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full(
                    (1, BLOCK_SIZE_K), 1, dtype=tl.int1
                )
                mask_tsrc_from_bk = tl.reshape(
                    mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K)
                )
                mask_tsrc = (
                    (idx_tsrc < (MAX_TSRC * (idx_g + 1)))
                    & (idx_tsrc >= (MAX_TSRC * idx_g))
                    & mask_tsrc_from_bk
                )
                idx_tsrc = idx_tsrc % MAX_TSRC
                mask_tsrc = (
                    mask_tsrc
                    & (idx_tsrc < tl.max(pos_tdst))
                    & (idx_tsrc >= sink_token_size)
                )
                # mask_tsrc = True
                # mask_tsrc = idx_tsrc > 0
                # idx_group = idx_tsrc // MAX_TSRC

                # min_tsrc = tl.min(idx_tsrc)

                # if min_tsrc <= tl.max(idx_tdst):
                # idx_n = idx_b * G + idx_group

                keys = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    # offload cache args template
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_METHOD,
                    OFFLOAD_CACHE_BUDGET,
                    OFFLOAD_CACHE_KV_HEAD,
                    False,
                    OFFLOAD_CACHE_K_TABLES,
                    stride_offload_cache_k_tables_n,
                    stride_offload_cache_k_tables_t,
                    OFFLOAD_CACHE_K_BANKS,
                    stride_offload_cache_k_banks_n,
                    stride_offload_cache_k_banks_page,
                    stride_offload_cache_k_banks_offset,
                    stride_offload_cache_k_banks_hid,
                    None,
                    0,
                    0,
                    0,
                    OFFLOAD_CACHE_COUNTERS,
                    stride_offload_cache_counters_n,
                    stride_offload_cache_counters_k,
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid[:, None],
                    mask_tsrc[None, :],
                    BLOCK_SIZE_K,
                )

                if USING_EXTEND and NEED_APPLY_ROPE:
                    keys_rot = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        # offload cache args template
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_METHOD,
                        OFFLOAD_CACHE_BUDGET,
                        OFFLOAD_CACHE_KV_HEAD,
                        False,
                        OFFLOAD_CACHE_K_TABLES,
                        stride_offload_cache_k_tables_n,
                        stride_offload_cache_k_tables_t,
                        OFFLOAD_CACHE_K_BANKS,
                        stride_offload_cache_k_banks_n,
                        stride_offload_cache_k_banks_page,
                        stride_offload_cache_k_banks_offset,
                        stride_offload_cache_k_banks_hid,
                        None,
                        0,
                        0,
                        0,
                        OFFLOAD_CACHE_COUNTERS,
                        stride_offload_cache_counters_n,
                        stride_offload_cache_counters_k,
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        ((idx_hid + HID // 2) % HID)[:, None],
                        mask_tsrc[None, :],
                        BLOCK_SIZE_K,
                    )
                else:
                    keys_rot = None

                values = load_tokens(
                    V,
                    stride_v_bsz,
                    stride_v_tsrc,
                    stride_v_head,
                    stride_v_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    V_CACHE,
                    stride_v_cache_page,
                    stride_v_cache_offset,
                    stride_v_cache_kv_head,
                    stride_v_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    # offload cache args template
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_METHOD,
                    OFFLOAD_CACHE_BUDGET,
                    OFFLOAD_CACHE_KV_HEAD,
                    False,
                    OFFLOAD_CACHE_V_TABLES,
                    stride_offload_cache_v_tables_n,
                    stride_offload_cache_v_tables_t,
                    OFFLOAD_CACHE_V_BANKS,
                    stride_offload_cache_v_banks_n,
                    stride_offload_cache_v_banks_page,
                    stride_offload_cache_v_banks_offset,
                    stride_offload_cache_v_banks_hid,
                    None,
                    0,
                    0,
                    0,
                    OFFLOAD_CACHE_COUNTERS,
                    stride_offload_cache_counters_n,
                    stride_offload_cache_counters_k,
                    idx_bsz,
                    idx_tsrc[:, None],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid[None, :],
                    mask_tsrc[:, None],
                    BLOCK_SIZE_K,
                )

                acc, l_i, m_i = block_sparse_attention_cuda_step(
                    queries,
                    keys,
                    keys_rot,
                    values,
                    idx_tsrc,
                    mask_tsrc,
                    idx_tdst,
                    mask_tdst,
                    acc,
                    l_i,
                    m_i,
                    sliding_window_size,
                    sink_token_size,
                    (range_end - range_start) * BLOCK_SIZE_K,
                    True,
                    False,
                    LOGIT_SOFTCAP,
                    USING_EXTEND,
                    NEED_APPLY_ROPE,
                    extend_window_size,
                    extend_group_size,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    model_context_length,
                    idx_bk + sink_token_size // BLOCK_SIZE_K,
                    pos_tdst,
                    idx_hid,
                    IS_CAUSAL,
                    HID,
                    BLOCK_SIZE_Q,
                    BLOCK_BK * BLOCK_SIZE_K,
                    BLOCK_SIZE_K,
                    EXTEND_BACKEND=EXTEND_BACKEND,
                )
            else:
                pass

    if (sink_token_size > 0) and True:
        for i_tsrc in range(0, sink_token_size, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < tl.minimum(MAX_TSRC, sink_token_size)

            # idx_n = idx_b * G + idx_group
            keys = load_tokens(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head,
                stride_k_hid,
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                # offload cache args template
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_METHOD,
                OFFLOAD_CACHE_BUDGET,
                OFFLOAD_CACHE_KV_HEAD,
                False,
                OFFLOAD_CACHE_K_TABLES,
                stride_offload_cache_k_tables_n,
                stride_offload_cache_k_tables_t,
                OFFLOAD_CACHE_K_BANKS,
                stride_offload_cache_k_banks_n,
                stride_offload_cache_k_banks_page,
                stride_offload_cache_k_banks_offset,
                stride_offload_cache_k_banks_hid,
                None,
                0,
                0,
                0,
                OFFLOAD_CACHE_COUNTERS,
                stride_offload_cache_counters_n,
                stride_offload_cache_counters_k,
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],
                BLOCK_SIZE_K,
            )

            if USING_EXTEND and NEED_APPLY_ROPE:
                keys_rot = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    # offload cache args template
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_METHOD,
                    OFFLOAD_CACHE_BUDGET,
                    OFFLOAD_CACHE_KV_HEAD,
                    False,
                    OFFLOAD_CACHE_K_TABLES,
                    stride_offload_cache_k_tables_n,
                    stride_offload_cache_k_tables_t,
                    OFFLOAD_CACHE_K_BANKS,
                    stride_offload_cache_k_banks_n,
                    stride_offload_cache_k_banks_page,
                    stride_offload_cache_k_banks_offset,
                    stride_offload_cache_k_banks_hid,
                    None,
                    0,
                    0,
                    0,
                    OFFLOAD_CACHE_COUNTERS,
                    stride_offload_cache_counters_n,
                    stride_offload_cache_counters_k,
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    ((idx_hid + HID // 2) % HID)[:, None],
                    mask_tsrc[None, :],
                    BLOCK_SIZE_K,
                )
            else:
                keys_rot = None

            values = load_tokens(
                V,
                stride_v_bsz,
                stride_v_tsrc,
                stride_v_head,
                stride_v_hid,
                USING_PAGES,
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                # offload cache args template
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_METHOD,
                OFFLOAD_CACHE_BUDGET,
                OFFLOAD_CACHE_KV_HEAD,
                False,
                OFFLOAD_CACHE_V_TABLES,
                stride_offload_cache_v_tables_n,
                stride_offload_cache_v_tables_t,
                OFFLOAD_CACHE_V_BANKS,
                stride_offload_cache_v_banks_n,
                stride_offload_cache_v_banks_page,
                stride_offload_cache_v_banks_offset,
                stride_offload_cache_v_banks_hid,
                None,
                0,
                0,
                0,
                OFFLOAD_CACHE_COUNTERS,
                stride_offload_cache_counters_n,
                stride_offload_cache_counters_k,
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],
                BLOCK_SIZE_K,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                keys_rot,
                values,
                idx_tsrc,
                mask_tsrc,
                idx_tdst,
                mask_tdst,
                acc,
                l_i,
                m_i,
                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                True,
                True,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                extend_window_size,
                extend_group_size,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                model_context_length,
                tl.arange(0, BLOCK_BK) + i_tsrc // BLOCK_SIZE_K,
                pos_tdst,
                idx_hid,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    if sliding_window_size > 0:
        CURR_TSRC = tl.max(pos_tdst)
        # CURR_TSRC = (idx_bdst + 1) * BLOCK_SIZE_Q + MAX_TSRC - MAX_TDST
        i_tsrc_range_start = tl.maximum(
            0, CURR_TSRC - sliding_window_size - BLOCK_SIZE_Q
        )
        TSRC_RANGE_STEP: tl.constexpr = BLOCK_BK * BLOCK_SIZE_K
        for i_tsrc in range(i_tsrc_range_start, CURR_TSRC, TSRC_RANGE_STEP):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < MAX_TSRC

            # idx_n = idx_b * G + idx_group
            keys = load_tokens(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head,
                stride_k_hid,
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                # offload cache args template
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_METHOD,
                OFFLOAD_CACHE_BUDGET,
                OFFLOAD_CACHE_KV_HEAD,
                False,
                OFFLOAD_CACHE_K_TABLES,
                stride_offload_cache_k_tables_n,
                stride_offload_cache_k_tables_t,
                OFFLOAD_CACHE_K_BANKS,
                stride_offload_cache_k_banks_n,
                stride_offload_cache_k_banks_page,
                stride_offload_cache_k_banks_offset,
                stride_offload_cache_k_banks_hid,
                None,
                0,
                0,
                0,
                OFFLOAD_CACHE_COUNTERS,
                stride_offload_cache_counters_n,
                stride_offload_cache_counters_k,
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],
                BLOCK_SIZE_K,
            )

            if USING_EXTEND and NEED_APPLY_ROPE:
                keys_rot = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    # offload cache args template
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_METHOD,
                    OFFLOAD_CACHE_BUDGET,
                    OFFLOAD_CACHE_KV_HEAD,
                    False,
                    OFFLOAD_CACHE_K_TABLES,
                    stride_offload_cache_k_tables_n,
                    stride_offload_cache_k_tables_t,
                    OFFLOAD_CACHE_K_BANKS,
                    stride_offload_cache_k_banks_n,
                    stride_offload_cache_k_banks_page,
                    stride_offload_cache_k_banks_offset,
                    stride_offload_cache_k_banks_hid,
                    None,
                    0,
                    0,
                    0,
                    OFFLOAD_CACHE_COUNTERS,
                    stride_offload_cache_counters_n,
                    stride_offload_cache_counters_k,
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    ((idx_hid + HID // 2) % HID)[:, None],
                    mask_tsrc[None, :],
                    BLOCK_SIZE_K,
                )
            else:
                keys_rot = None

            values = load_tokens(
                V,
                stride_v_bsz,
                stride_v_tsrc,
                stride_v_head,
                stride_v_hid,
                USING_PAGES,
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                # offload cache args template
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_METHOD,
                OFFLOAD_CACHE_BUDGET,
                OFFLOAD_CACHE_KV_HEAD,
                False,
                OFFLOAD_CACHE_V_TABLES,
                stride_offload_cache_v_tables_n,
                stride_offload_cache_v_tables_t,
                OFFLOAD_CACHE_V_BANKS,
                stride_offload_cache_v_banks_n,
                stride_offload_cache_v_banks_page,
                stride_offload_cache_v_banks_offset,
                stride_offload_cache_v_banks_hid,
                None,
                0,
                0,
                0,
                OFFLOAD_CACHE_COUNTERS,
                stride_offload_cache_counters_n,
                stride_offload_cache_counters_k,
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],
                BLOCK_SIZE_K,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                keys_rot,
                values,
                idx_tsrc,
                mask_tsrc,
                idx_tdst,
                mask_tdst,
                acc,
                l_i,
                m_i,
                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                False,
                False,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                extend_window_size,
                extend_group_size,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                model_context_length,
                # tl.arange(0, BLOCK_BK) +\
                #     (range_end - range_start) +\
                #     (sink_token_size // BLOCK_SIZE_K) +\
                #     (i_tsrc-i_tsrc_range_start) // BLOCK_SIZE_K,
                tl.arange(0, BLOCK_BK)
                + (i_tsrc - i_tsrc_range_start) // BLOCK_SIZE_K
                + (
                    tl.max(pos_tdst * mask_tdst)
                    - tl.sum(mask_tdst.to(tl.int32))
                    - sliding_window_size
                )
                // BLOCK_SIZE_K,
                pos_tdst,
                idx_hid,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / (tl.where(l_i == 0.0, 1e-20, l_i))

    tl.store(
        CONTEXT
        + idx_bsz * stride_context_bsz
        + idx_tdst[:, None] * stride_context_tdst
        + idx_head * stride_context_head
        + idx_hid[None, :] * stride_context_hid,
        mask=mask_tdst[:, None],
        value=acc.to(CONTEXT.type.element_ty),
        # eviction_policy='evict_first',
        # cache_modifier='.cs', # TODO: uncomment this
        # value = l_i
    )


def block_sparse_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    args: "HiPAttentionArgs",
    EXTEND_BACKEND: str = DEFAULT_EXTEND_BACKEND,
    model_context_length: int = 131072,
):
    if os.getenv("HIP_DEBUG_SA", "0") == "1":
        return block_sparse_attention_pytorch(
            q,
            k,
            v,
            indices,
            ks,
            args,
        )

    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        BSRC = cdiv_python(TSRC, args.block_size_k)
        MAX_TSRC = TSRC
        MAX_BSRC = BSRC
    else:
        NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
        TSRC = None
        BSRC = None
        MAX_TSRC = NUM_PAGE * PAGE_SIZE
        MAX_BSRC = cdiv_python(MAX_TSRC, args.block_size_k)
    N = BSZ * HEAD
    # assert q.shape == k.shape
    BDST = cdiv_python(TDST, args.block_size_q)
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD

    G = args.topk_head_group_size
    B = N // G
    assert (B * G) == N
    BK = indices.shape[-1]  # cdiv_python(args.mask_k, args.block_size_k)

    context = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    # BLOCK_BK = 64 // block_size_k
    # if block_size_k > 4:
    #     BLOCK_BK = 128 // block_size_k
    # elif block_size_k > 8:
    #     BLOCK_BK = 256 // block_size_k
    # BLOCK_BK = 64 // args.block_size_k

    max_block_size = int(os.getenv("SA_BLOCK_SIZE", "32"))
    BLOCK_BK = max_block_size // args.block_size_k
    BLOCK_BK = max(1, min(max_block_size, BLOCK_BK))
    if "SA_BLOCK_BK" in os.environ:
        BLOCK_BK = int(os.environ["SA_BLOCK_BK"])

    assert BLOCK_BK > 0, BLOCK_BK

    # sliding_window_size = min(sliding_window_size, block_size_k * 16)

    if args.rope_cos is not None:
        assert len(args.rope_cos.stride()) == 2
        assert len(args.rope_sin.stride()) == 2

    assert context.ndim == 4
    if ks_start_end is not None:
        assert ks_start_end.ndim == 3
    if indices is not None:
        assert indices.ndim == 3
    assert q.ndim == 4
    if k is not None:
        assert k.ndim == 4
        assert v.ndim == 4
    elif args.using_paged_cache:
        assert args.k_cache.ndim == 4
        assert args.v_cache.ndim == 4
    else:
        raise Exception()
    assert seq_lens.ndim == 2

    grid = (HEAD, BDST, BSZ)
    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    # print(indices.shape, indices[0, -1], ks_start_end[0, -1])
    # if indices.shape[1] == 1:
    #     input()

    block_sparse_attention_cuda[grid](
        q,
        *args.safe_stride(q, 4),
        k,
        *args.safe_stride(k, 4),
        v,
        *args.safe_stride(v, 4),
        seq_lens,
        *args.safe_stride(seq_lens, 2),
        indices,
        *args.safe_stride(indices, 3),
        ks_start_end,
        *args.safe_stride(ks_start_end, 3),
        context,
        *args.safe_stride(context, 4),
        HEAD,
        G,
        BK,
        TDST,
        MAX_TSRC,
        KV_HEAD_REPEAT,
        args.sliding_window_size,
        args.sink_token_size,
        args.logit_softcap,
        *args.args_extend(),
        model_context_length,
        *args.args_paged_kv_cache(),
        *args.args_offload_cache(is_masking=False),
        triton.next_power_of_2(TDST),
        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        HID,
        # 2,
        BLOCK_BK=BLOCK_BK,
        EXTEND_BACKEND=EXTEND_BACKEND,
        # num_warps=4,
        # num_stages=2 if not using_extend else 1,
    )
    torch.set_default_device(pre_device)

    if (
        (os.getenv("HIP_CUMSUM", "0") == "1")
        and isinstance(v, Tensor)
        and q.shape[1] > 1
    ):
        v_cumsum = (
            v.cumsum(dim=1)
            / torch.arange(1, v.shape[1] + 1, device=v.device)[None, :, None, None]
        )
        a = torch.arange(1, v.shape[1] + 1, device=v.device)[None, :, None]
        b = (
            ks.repeat_interleave(args.block_size_q, 1)[:, : v.shape[1]]
            .view(BSZ, HEAD, -1)
            .permute(0, 2, 1)
            * args.block_size_k
        )
        scaler = ((a - b) / a).clamp_min(0)[:, :, :, None].pow(2) * 0.05
        context = (
            context * (1 - scaler)
            + v_cumsum.repeat_interleave(HEAD // KV_HEAD, dim=2) * scaler
        )

    return context


@triton.jit
def to_dense_cuda(
    INDICES,
    stride_indices_n,
    stride_indices_bdst,
    stride_indices_k,
    KS,
    stride_ks_n,
    stride_ks_bdst,
    OUT,
    stride_out_n,
    stride_out_tdst,
    stride_out_tsrc,
    N,
    TDST,
    TSRC,
    BK,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    idx_n = tl.program_id(1)
    idx_bdst = tl.program_id(0)

    for idx_k in range(0, BK):
        ks = tl.load(KS + idx_n * stride_ks_n + idx_bdst * stride_ks_bdst)
        if idx_k < ks:
            idx_trsc = tl.load(
                INDICES
                + idx_n * stride_indices_n
                + idx_bdst * stride_indices_bdst
                + idx_k * stride_indices_k,
            )
            out_tdst = tl.arange(0, BLOCK_SIZE_Q) + idx_bdst * BLOCK_SIZE_Q
            mask_tdst = out_tdst < TDST
            out_tsrc = tl.arange(0, BLOCK_SIZE_K) + idx_trsc
            mask_tsrc = out_tsrc < TSRC
            tl.atomic_add(
                OUT
                + idx_n * stride_out_n
                + out_tdst[:, None] * stride_out_tdst
                + out_tsrc[None, :] * stride_out_tsrc,
                mask=mask_tdst[:, None] & mask_tsrc[None, :],
                val=1,
            )
        else:
            pass
        tl.debug_barrier()


def to_dense_efficient(
    indices: Tensor,
    ks: Tensor,
    N,
    TDST,
    TSRC,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_K,
):
    out = torch.zeros((N, TDST, TSRC), dtype=torch.float32, device=indices.device)

    _, BDST, BK = indices.shape

    grid = (
        BDST,
        N,
    )

    pre_device = torch.get_default_device()
    torch.set_default_device(indices.device)
    to_dense_cuda[grid](
        indices,
        *indices.stride(),
        ks,
        *ks.stride(),
        out,
        *out.stride(),
        N,
        TDST,
        TSRC,
        BK,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
    )
    torch.set_default_device(pre_device)

    return out


def block_sparse_attention_pytorch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    indices: Tensor,
    ks: Tensor,
    args: "HiPAttentionArgs",
):
    BSZ, TDST, HEAD, HID = q.shape
    _, TSRC, HEAD_KV, _ = k.shape
    N, BDST, BK = indices.shape

    if HEAD_KV != HEAD:
        k = k.repeat_interleave(repeats=HEAD // HEAD_KV, dim=2)
        v = v.repeat_interleave(repeats=HEAD // HEAD_KV, dim=2)

    MAX_SCORE_BUDGET = 4096 * 4096
    GROUP_TDST = (
        cdiv_python(cdiv_python(MAX_SCORE_BUDGET, TSRC), args.block_size_q)
        * args.block_size_q
    )
    GROUP_BDST = GROUP_TDST // args.block_size_q

    out = torch.zeros_like(q)

    for i_start_bdst in tqdm.tqdm(
        range(0, BDST, GROUP_BDST), desc="BSA", dynamic_ncols=True, leave=False, delay=3
    ):
        i_start_tdst = i_start_bdst * args.block_size_q
        i_end_tdst = min(TDST, i_start_tdst + GROUP_TDST)

        mask = to_dense_efficient(
            indices[:, i_start_bdst : i_start_bdst + GROUP_BDST],
            ks[:, i_start_bdst : i_start_bdst + GROUP_BDST],
            N,
            i_end_tdst - i_start_tdst,
            TSRC,
            args.block_size_q,
            args.block_size_k,
        )
        mask = mask.bool().view(BSZ, HEAD, -1, TSRC)

        causal_mask = (
            torch.arange(0, TSRC, device=q.device)[None, :]
            <= torch.arange(i_start_tdst, i_end_tdst, device=q.device)[:, None]
        )

        t_q = q[:, i_start_tdst:i_end_tdst]
        score = torch.matmul(
            t_q.contiguous().permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)
        )
        torch.where(mask, score, torch.tensor(-32000.0, device=mask.device), out=score)
        if args.is_causal:
            torch.where(
                causal_mask,
                score,
                torch.tensor(-32000.0, device=mask.device),
                out=score,
            )
        probs = torch.softmax(score, dim=-1)
        del score
        context = probs @ v.permute(0, 2, 1, 3)
        context = context.permute(0, 2, 1, 3)
        out[:, i_start_tdst:i_end_tdst] = context

    return out


@nvtx.annotate("masking_step_loop")
def masking_step_loop(
    q: Tensor, k: Tensor, chunk_offset: int, args: "HiPAttentionArgs"
):
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, _, _ = k.shape
    else:
        TSRC = None
    N = BSZ * HEAD

    # NOTE: this make ppl worse
    # with nvtx.annotate('k_adjust'):
    #     if topk_head_group_size > 1:
    #         k = k - k[:, :2, :].mean(-2, keepdim=True)

    indices_blocks = []
    ks_blocks = []
    ks_count_blocks = []
    ks_start_end_blocks = []
    scores_blocks = []
    key_access_log_blocks = []
    key_access_count_blocks = []
    block_access_log_blocks = []
    block_access_score_blocks = []
    block_access_count_blocks = []
    indices_seed = ks_seed = None
    for i_chunk_tdst in range(0, args.chunk_size, args.block_size_q * args.step_size):
        idx_tdst = (
            torch.arange(
                i_chunk_tdst,
                i_chunk_tdst + args.block_size_q * args.step_size,
                device=q.device,
            )[None, :]
            + torch.arange(
                0,
                TDST,
                args.chunk_size,
                device=q.device,
            )[:, None]
            + chunk_offset
        )
        idx_tdst = idx_tdst % TDST
        idx_tdst = idx_tdst.reshape(-1)
        if args.position_ids is not None:
            pos_tdst = args.position_ids.gather(
                dim=1, index=idx_tdst.unsqueeze(0).expand(BSZ, -1)
            )  # + 1
        else:
            if TSRC is not None:
                pos_tdst = (idx_tdst[None, :] + TSRC - TDST).expand(BSZ, -1) + 1
            else:
                pos_tdst = idx_tdst[None, :] + args.cache_seq_lens[:, None] - TDST + 1
        scores_seed = None
        with nvtx.annotate(
            f"masking_samples(seed={tuple(indices_seed.shape) if indices_seed is not None else None})"
        ):
            for idx_sample in range(args.num_samples):
                with nvtx.annotate(f"masking_iteration_draft(idx_sample={idx_sample})"):
                    if (
                        args.low_res_sample_scale <= 1
                        and args.low_res_oversample_rate <= 1
                    ):
                        (
                            indices,
                            ks,
                            ks_count,
                            ks_start_end,
                            scores,
                            group_sizes,
                            key_access_log,
                            key_access_count,
                            block_access_log,
                            block_access_score,
                            block_access_count,
                        ) = masking_iteration_draft(
                            q,
                            k,
                            position_ids=pos_tdst,
                            indices_seed=indices_seed,
                            ks_seed=ks_seed,
                            scores_seed=scores_seed,
                            indices_tdst=idx_tdst,
                            args=args,
                        )

                        indices_seed = indices
                        ks_seed = ks
                        scores_seed = scores
                        if key_access_log is not None:
                            key_access_log_blocks.append(key_access_log)
                        if key_access_count is not None:
                            key_access_count_blocks.append(key_access_count)
                        if block_access_log is not None:
                            block_access_log_blocks.append(block_access_log)
                        if block_access_score is not None:
                            block_access_score_blocks.append(block_access_score)
                        if block_access_count is not None:
                            block_access_count_blocks.append(block_access_count)
                    else:
                        assert isinstance(args.low_res_sample_scale, int)
                        low_mask_k = args.mask_k * args.low_res_oversample_rate
                        low_block_size_k = (
                            args.block_size_k
                            * args.low_res_oversample_rate
                            * args.low_res_sample_scale
                        )

                        assert args.low_res_sample_scale >= 1
                        assert args.low_res_oversample_rate >= 1
                        assert isinstance(args.low_res_sample_scale, int)
                        assert isinstance(args.low_res_oversample_rate, int)

                        # low_res_oversample_rate == group_size
                        # low_res_sample_scale == num block split

                        # NOTE: following code is for downsample the seed from last step
                        """
                        # need to be num element low_mask_k // low_block_size_k
                        stride = low_res_oversample_rate * low_res_sample_scale
                        assert stride > 1
                        if indices_seed is not None:
                            indices_seed = indices_seed[:, :, ::stride]
                        if scores_seed is not None:
                            scores_seed = scores_seed[:, :, ::stride]

                        if low_res_sample_scale > 1:
                            if ks_seed is not None:
                                ks_seed = torch.ceil(ks_seed / low_res_sample_scale).to(torch.int32)

                        if low_res_oversample_rate > 1:
                            if indices_seed is not None:
                                scores_seed = None
                                indices_seed = indices_seed\
                                    .repeat_interleave(low_res_oversample_rate, dim=-1)\
                                    .view(*indices_seed.shape, 2)
                                indices_seed = indices_seed +\
                                    torch.arange(
                                        0,
                                        low_res_oversample_rate * low_block_size_k,
                                        low_block_size_k,
                                        device=indices_seed.device
                                    )[None, None, None, :]
                                indices_seed = indices_seed.view(
                                    indices_seed.shape[0],
                                    indices_seed.shape[1],
                                    indices_seed.shape[2] * low_res_oversample_rate
                                )
                        """

                        low_res_sample_config = args.clone()
                        low_res_sample_config.mask_k = low_mask_k
                        low_res_sample_config.block_size_k = low_block_size_k
                        low_res_sample_config.block_stride_k = (
                            args.low_res_oversample_block_stride_k
                        )

                        with nvtx.annotate("low_res_sample"):
                            # TODO: reduce initial seeds
                            (
                                indices,
                                ks,
                                ks_count,
                                ks_start_end,
                                scores,
                                group_sizes,
                                key_access_log,
                                key_access_count,
                                block_access_log,
                                block_access_score,
                                block_access_count,
                            ) = masking_iteration_draft(
                                q[:, :, :],
                                k[:, :, :],
                                position_ids=pos_tdst,
                                indices_seed=indices_seed,
                                ks_seed=ks_seed,
                                scores_seed=scores_seed,
                                indices_tdst=idx_tdst,
                                args=low_res_sample_config,
                            )

                            indices_seed = indices
                            ks_seed = ks
                            scores_seed = scores

                            # indices_for_seed = indices
                            # scores_for_seed = scores
                            # ks_for_seed = ks

                            # NOTE: if we recurrent on low res, then upsampling is ignored for few steps
                            if args.num_samples > 1 and idx_sample < (
                                args.num_samples - 1
                            ):
                                continue

                        with nvtx.annotate("sample_division"):
                            if args.low_res_sample_scale > 1:
                                indices = (
                                    indices[:, :, :, None]
                                    + torch.arange(
                                        0,
                                        low_block_size_k,
                                        args.block_size_k
                                        * args.low_res_oversample_rate,
                                        device=indices.device,
                                    )[None, None, None, :]
                                )
                                indices = indices.view(
                                    indices.shape[0], indices.shape[1], -1
                                )
                                ks = ks.mul(args.low_res_sample_scale)
                                group_sizes = torch.repeat_interleave(
                                    group_sizes, args.low_res_sample_scale, dim=-1
                                )

                                # NOTE: block is break down, this is not accurate
                                if scores is not None:
                                    scores = (
                                        scores[:, :, :, None]
                                        .expand(-1, -1, -1, 2)
                                        .contiguous()
                                        .view(scores.shape[0], scores.shape[1], -1)
                                    )

                                ks_count, ks_start_end = (
                                    masking_iteration_draft_python_epilog(
                                        indices,
                                        ks,
                                        cdiv_python(args.mask_k, args.block_size_k),
                                        TSRC,
                                        ks.shape[0],
                                        ks.shape[1],
                                        args.topk_head_group_size,
                                    )
                                )

                        with nvtx.annotate("downsample"):
                            if args.low_res_oversample_rate > 1:
                                init_indices = torch.full_like(
                                    indices,
                                    fill_value=(
                                        cdiv_python(TSRC, args.block_size_k)
                                        + args.block_size_k
                                        + args.block_size_q
                                    )
                                    * args.topk_head_group_size,
                                )
                                init_ks = torch.zeros_like(ks)
                                init_group_sizes = torch.zeros_like(group_sizes)
                                grid = (
                                    N // args.topk_head_group_size,
                                    init_group_sizes.shape[1],
                                    args.topk_head_group_size,
                                )
                                pre_device = torch.get_default_device()
                                torch.set_default_device(pos_tdst.device)
                                masking_iteration_draft_cuda_initialize[grid](
                                    None,
                                    *(0, 0, 0),
                                    None,
                                    *(0, 0),
                                    pos_tdst,
                                    *pos_tdst.stride(),
                                    init_indices,
                                    *init_indices.stride(),
                                    init_ks,
                                    *init_ks.stride(),
                                    init_group_sizes,
                                    *init_group_sizes.stride(),
                                    None,
                                    *(
                                        0,
                                        0,
                                    ),
                                    args.mask_k,
                                    args.block_size_q,
                                    args.block_stride_q,
                                    args.block_size_k,
                                    args.is_causal,
                                    args.sliding_window_size,
                                    args.topk_head_group_size,
                                    len(idx_tdst),
                                    TSRC,
                                    HEAD,
                                    cdiv_python(args.mask_k, args.block_size_k),
                                    # num_warps=min(max(cdiv_python(BLOCK_MASK_BLOCK_K, 32), 1), 32),
                                    num_warps=1,
                                    num_stages=1,
                                )
                                torch.set_default_device(pre_device)

                                # init_indices.mul_(block_size_k)

                                group_sizes_scaled = (
                                    torch.maximum(
                                        group_sizes.float(),
                                        torch.ones_like(group_sizes),
                                    )
                                    * args.low_res_oversample_rate
                                )

                                # print(init_group_sizes[0, idx_tdst[::32] < 1024, :10])
                                # print(group_sizes_scaled[0, idx_tdst[::32] < 1024, :10])

                                mask_tdst = (
                                    pos_tdst[:, :: args.block_size_q] < args.mask_k * 2
                                )
                                print(
                                    mask_tdst.shape,
                                    init_group_sizes.shape,
                                    group_sizes_scaled.shape,
                                )
                                group_sizes = torch.where(
                                    mask_tdst[:, :, None],
                                    init_group_sizes,
                                    group_sizes_scaled,
                                )
                                indices = torch.where(
                                    mask_tdst[:, :, None],
                                    init_indices * args.block_size_k,
                                    indices,
                                )
                                ks = torch.where(
                                    mask_tdst[:, :],
                                    init_ks,
                                    ks,
                                )

                                (
                                    indices,
                                    ks,
                                    ks_count,
                                    ks_start_end,
                                    scores,
                                    group_sizes,
                                    key_access_log,
                                    key_access_count,
                                    block_access_log,
                                    block_access_score,
                                    block_access_count,
                                ) = masking_iteration_draft(
                                    q[:, :, :],
                                    k[:, :, :],
                                    position_ids=pos_tdst,
                                    indices_seed=indices_seed,
                                    ks_seed=ks_seed,
                                    scores_seed=None,
                                    indices_tdst=idx_tdst,
                                    args=args,
                                )

                        # use this indices for cache, if you want to downsample
                        """
                        indices_seed = indices
                        ks_seed = ks
                        scores_seed = scores
                        """

        if not args.traverse_from_last_step:
            indices_seed = ks_seed = None
        # if (chunk_size is not None) and ((((i_chunk_tdst + chunk_offset) // block_size_q + 1) % (chunk_size // block_size_q)) == 0):
        # if ((i_chunk_tdst + 1) % (chunk_size - chunk_offset)) == 0:
        # indices_seed = ks_seed = None

        indices_blocks.append(indices)
        ks_blocks.append(ks)
        ks_count_blocks.append(ks_count)
        ks_start_end_blocks.append(ks_start_end)
        scores_blocks.append(scores)

    if len(indices_blocks) == 1:
        indices = indices_blocks[0]
        ks = ks_blocks[0]
        ks_count = ks_count_blocks[0]
        ks_start_end = ks_start_end_blocks[0]
        scores = scores_blocks[0]
    else:
        indices = torch.cat(indices_blocks, dim=1)
        ks = torch.cat(ks_blocks, dim=1)
        ks_count = torch.cat(ks_count_blocks, dim=1)
        ks_start_end = torch.cat(ks_start_end_blocks, dim=1)
        scores = torch.cat(scores_blocks, dim=1)

    if len(key_access_log_blocks) == 0:
        key_access_log = None
        key_access_count = None
    elif len(key_access_log_blocks) == 1:
        key_access_log = key_access_log_blocks[0]
        key_access_count = key_access_count_blocks[0]
    else:
        key_access_log = torch.cat(key_access_log_blocks, dim=1)
        key_access_count = torch.cat(key_access_count_blocks, dim=1)

    if len(block_access_log_blocks) == 0:
        block_access_log = None
        block_access_score = None
        block_access_count = None
    elif len(block_access_log_blocks) == 1:
        block_access_log = block_access_log_blocks[0]
        block_access_score = block_access_score_blocks[0]
        block_access_count = block_access_count_blocks[0]
    else:
        block_access_log = torch.cat(block_access_log_blocks, dim=1)
        block_access_score = torch.cat(block_access_score_blocks, dim=1)
        block_access_count = torch.cat(block_access_count_blocks, dim=1)

    # print(indices.shape)
    # print(ks.shape)
    # print(ks_count.shape)
    # print(ks_start_end.shape)
    # print(scores.shape)
    # torch.Size([32, 256, 256])
    # torch.Size([32, 256])
    # torch.Size([32, 256, 1])
    # torch.Size([32, 256, 2])
    # torch.Size([32, 256, 256])

    num_chunks = triton.cdiv(TDST, args.chunk_size)

    if num_chunks > 1:

        def permute_3d(x: Tensor):
            N, BDST, K = x.shape
            return (
                x.view(N, triton.cdiv(BDST, num_chunks), num_chunks, K)
                .permute(0, 2, 1, 3)
                .reshape(N, BDST, K)
            )

        indices = permute_3d(indices)
        ks = permute_3d(ks.unsqueeze(-1)).squeeze(-1)
        ks_count = permute_3d(ks_count)
        ks_start_end = permute_3d(ks_start_end)
        scores = permute_3d(scores)

    return (
        indices,
        ks,
        ks_count,
        ks_start_end,
        scores,
        key_access_log,
        key_access_count,
        block_access_log,
        block_access_score,
        block_access_count,
    )


@nvtx.annotate("hip_masking")
def hip_masking(
    q: Tensor,
    k: Optional[Tensor],
    args: "HiPAttentionArgs",
):
    assert args.logit_softcap != 0, "args.logit_softcap should be None if 0"

    if not args.is_causal:
        assert (
            args.sliding_window_size == 0
        ), "if bidirectional, you should disable sliding window"
        assert (
            args.sink_token_size == 0
        ), "if bidirectional, you should disable sink tokens"

    if args.randomize_mask or (os.getenv("HIP_RANDOM_MASK", "0") == "1"):
        warnings.warn("BigBird simulated with HiP kernel")

        BSZ, TDST, HEAD, HID = q.shape
        if k is not None:
            _, _, HEAD_KV, HID = k.shape
        elif args.k_cache is not None:
            _, _, HEAD_KV, HID = args.k_cache.shape
        else:
            raise Exception()

        assert args.topk_head_group_size == 1
        assert args.position_ids is not None

        N = BSZ * HEAD
        BDST = cdiv_python(TDST, args.block_size_q)
        BK = cdiv_python(args.mask_k, args.block_size_k)

        LARGE_INT = 987654321

        indices = torch.rand((N, BDST, BK), dtype=torch.float32, device=q.device)

        seq_lens = (
            args.position_ids[
                :, min(args.block_stride_q - 1, TDST - 1) :: args.block_size_q
            ]
            + args.block_size_q
        )

        if seq_lens.shape != (BSZ, BDST):
            seq_lens = torch.cat(
                [seq_lens, args.position_ids[:, -1:] + args.block_size_q], dim=1
            )

        assert seq_lens.shape == (
            BSZ,
            BDST,
        ), f"{seq_lens.shape} == ({BSZ}, {BDST}), {args.position_ids}, {args.block_size_q}"
        seq_lens = torch.clamp(seq_lens - args.sliding_window_size, 0, LARGE_INT)
        indices = indices * seq_lens.repeat_interleave(repeats=HEAD, dim=0).unsqueeze(
            -1
        )
        indices = indices.long() // args.block_size_k * args.block_size_k
        indices[:, :, 0] = 0  # sink block
        indices = indices.sort(dim=-1).values
        indices = torch.where(
            indices != torch.roll(indices, shifts=1, dims=-1), indices, LARGE_INT
        )
        indices = indices.sort(dim=-1).values

        ks = torch.logical_and(indices >= 0, indices < LARGE_INT).int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros((N, BDST, 2), dtype=torch.int64, device=q.device)
        ks_start_end[:, :, -1] = ks

        if os.getenv("HIP_DEBUG", "0") == "1":
            B, TDST, H, HID = q.shape
            if k is not None:
                _, TSRC, H_KV, _ = k.shape
            else:
                TSRC = torch.max(args.cache_seq_lens).item()
            N = B * H

            def render_mask():
                debug_mask = to_dense(
                    indices.cpu().numpy(),
                    ks.cpu().numpy(),
                    None,
                    cdiv_python(N, args.topk_head_group_size),
                    TDST,
                    TSRC * args.topk_head_group_size,
                    args.block_size_q,
                    args.block_size_k * args.block_size_k_group,
                )[1]
                if args.group_size_q > 1:
                    debug_mask = debug_mask.repeat(axis=0, repeats=args.group_size_q)
                plt.clf()
                plt.figure(figsize=(4 * args.topk_head_group_size, 4))
                plt.imshow(debug_mask)
                plt.tight_layout()
                plt.savefig("dummy.png", dpi=96, bbox_inches="tight")
                print("saved dummy.png")

            render_mask()

        return (
            indices,
            ks,
            ks_count,
            ks_start_end,
            None,
            None,
            None,
            None,
            None,
            args,
        )

    if args.decode_style_masking and (q.shape[1] > 1):
        args = args.clone()
        shift = args.block_size_q
        args.sliding_window_size += shift
        args.decode_style_masking = False

        (
            indices,
            ks,
            ks_count,
            ks_start_end,
            key_access_log,
            key_access_count,
            block_access_log,
            block_access_score,
            block_access_count,
            args,
        ) = hip_masking(
            # TODO(-): apply PCA topk
            q=q,
            k=k,
            args=args,
        )

        args = args.clone()
        args.sliding_window_size -= shift

        indices = torch.roll(indices, shifts=-1, dims=1)
        indices[:, -1, :] = indices[:, -2, :]
        ks = torch.roll(ks, shifts=-1, dims=1)
        ks[:, -1] = ks[:, -2]
        ks_count[:, :, 0] = ks
        ks_start_end[:, :, 1] = ks

        return (
            indices,
            ks,
            ks_count,
            ks_start_end,
            None,
            None,
            None,
            None,
            None,
            args,
        )

    if args.group_size_q > 1:
        q_quant = q

        # TODO args size q handling should be inside of hip_masking
        n, t, h, d = q_quant.shape
        n_groups = cdiv_python(t, args.block_size_q * args.group_size_q)

        to_pad = 0
        if (n_groups * args.block_size_q * args.group_size_q) != t:
            to_pad = n_groups * args.block_size_q * args.group_size_q - t
            q_quant = F.pad(q_quant, pad=(0, 0, 0, 0, to_pad, 0))

        q_quant = q_quant.view(
            n,
            n_groups,
            args.group_size_q,
            args.block_size_q,
            h,
            d,
        )
        q_quant = q_quant[:, :, -1, :, :, :].reshape(
            n, n_groups * args.block_size_q, h, d
        )

        original_position_ids = args.position_ids
        assert original_position_ids is not None
        position_ids = F.pad(
            original_position_ids.unsqueeze(0), pad=(to_pad, 0)
        ).squeeze(0)
        position_ids = position_ids.view(
            n, n_groups, args.group_size_q, args.block_size_q
        )[:, :, -1, :].reshape(n, n_groups * args.block_size_q)
        args = args.clone()
        args.position_ids = position_ids
        original_group_size_q = args.group_size_q
        args.group_size_q = 1  # NOTE(-): perform hip masking as usual

        (
            indices,
            ks,
            ks_count,
            ks_start_end,
            key_access_log,
            key_access_count,
            block_access_log,
            block_access_score,
            block_access_count,
            args,
        ) = hip_masking(
            # TODO(-): apply PCA topk
            q=q_quant,
            k=k,
            args=args,
        )

        args.group_size_q = original_group_size_q

        # args.block_size_k = args.block_size_q
        # indices = (indices // args.block_size_q) * args.block_size_q

        LARGE_INT = 987654321

        assert args.topk_head_group_size == 1
        # repeat the mask
        assert indices.ndim == 3
        assert ks.ndim == 2, ks.shape
        assert ks_count.ndim == 3, ks_count.shape
        assert ks_start_end.ndim == 3, ks_start_end.shape
        if key_access_log is not None:
            raise NotImplementedError()
        if block_access_log is not None:
            raise NotImplementedError()
        indices = torch.repeat_interleave(indices, args.group_size_q, 1)
        # ks = torch.repeat_interleave(ks, args.group_size_q, 1)
        # ks_count = torch.repeat_interleave(ks_count, args.group_size_q, 1)
        # ks_start_end = torch.repeat_interleave(ks_start_end, args.group_size_q, 1)

        indices = torch.repeat_interleave(indices, 2, 2)
        n, t, d = indices.shape
        n_groups = t // (args.group_size_q)
        indices = indices.view(n, n_groups, args.group_size_q, d)
        indices[:, :, :, 0::2] -= (
            args.group_size_q
            - torch.arange(args.group_size_q, device=indices.device)
            - 1
        )[None, None, :, None] * args.block_size_q
        indices = indices.view(n, t, d)
        indices = torch.where(indices >= 0, indices, LARGE_INT)
        indices = torch.sort(indices, dim=-1, stable=False).values
        rolled_indices = torch.roll(indices, shifts=1, dims=-1)
        indices = torch.where(indices != rolled_indices, indices, LARGE_INT)
        indices = torch.sort(indices, dim=-1, stable=False).values

        n_queries = original_position_ids.shape[1]
        indices = indices[:, -n_queries:].contiguous()
        ks = (indices < LARGE_INT).to(torch.int32).sum(-1).contiguous()
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros(
            (ks.shape[0], ks.shape[1], 2), dtype=ks.dtype, device=ks.device
        )
        ks_start_end[:, :, -1] = ks
        args.position_ids = original_position_ids

        if os.getenv("HIP_DEBUG", "0") == "1":
            max_query = 2048

            B, TDST, H, HID = q.shape
            TDST = min(max_query, TDST)
            if k is not None:
                _, TSRC, H_KV, _ = k.shape
            else:
                TSRC = torch.max(args.cache_seq_lens).item()
            N = B * H

            def render_mask():
                debug_mask = to_dense(
                    indices[:, -cdiv_python(TDST, args.block_size_q) :].cpu().numpy(),
                    ks[:, -cdiv_python(TDST, args.block_size_q) :].cpu().numpy(),
                    None,
                    cdiv_python(N, args.topk_head_group_size),
                    TDST,
                    TSRC * args.topk_head_group_size,
                    args.block_size_q,
                    args.block_size_k * args.block_size_k_group,
                )[0]
                plt.clf()
                cv2.imwrite("dummy_prefetch_raw.png", debug_mask * 255)
                print("saved dummy_prefetch_raw.png")
                plt.figure(figsize=(4 * args.topk_head_group_size, 4))
                plt.imshow(debug_mask)
                plt.tight_layout()
                plt.savefig("dummy_prefetch.png", dpi=192, bbox_inches="tight")
                print(
                    "saved dummy_prefetch.png",
                    indices.shape,
                    debug_mask.shape,
                    original_position_ids.shape,
                )

            render_mask()

        return (
            indices,
            ks,
            ks_count,
            ks_start_end,
            key_access_log,
            key_access_count,
            block_access_log,
            block_access_score,
            block_access_count,
            args,
        )

    assert (k is None and args.k_cache is not None) or (
        k is not None and args.k_cache is None
    )
    assert q.ndim == 4
    if k is not None:
        assert k.ndim == 4
    BSZ, TDST, HEAD, HID = q.shape
    G = args.topk_head_group_size
    B = BSZ * HEAD // G
    N = BSZ * HEAD

    args = args.clone()

    assert args.num_unions > 0
    if args.chunk_size is None:
        args.chunk_size = q.shape[1]
    assert args.chunk_size > 0
    assert args.chunk_size >= args.num_unions

    if args.step_size is None:
        args.step_size = cdiv_python(q.shape[1], args.block_size_q)
    assert args.step_size > 0
    assert args.step_size <= cdiv_python(
        q.shape[1], args.block_size_q
    ), f"{args.step_size} <= {cdiv_python(q.shape[1], args.block_size_q)}"

    if args.using_sparq:
        raise Exception("vectorized head not support SparQ")
        BSZ, T, HEAD, D = q.shape
        q_score = q.view(
            BSZ,
            triton.cdiv(T, block_size_q),
            block_size_k,
            HEAD // topk_head_group_size,
            topk_head_group_size,
            D,
        )
        _, sparq_ind = (
            q_score.abs()
            .sum(dim=2)
            .topk(k=sparq_hid, dim=-1, largest=True, sorted=False)
        )
        sparq_ind, _ = torch.sort(sparq_ind, dim=-1)
    else:
        sparq_ind = None

    indices_sampled = []
    ks_sampled = []
    ks_count_sampled = []
    ks_start_end_sampled = []
    scores_sampled = []
    key_access_log_sampled = []
    key_access_count_sampled = []
    block_access_log_sampled = []
    block_access_score_sampled = []
    block_access_count_sampled = []
    for i_chunk_offset in range(0, args.chunk_size, args.chunk_size // args.num_unions):
        (
            indices,
            ks,
            ks_count,
            ks_start_end,
            scores,
            key_access_log,
            key_access_count,
            block_access_log,
            block_access_score,
            block_access_count,
        ) = masking_step_loop(
            q=q,
            k=k,
            chunk_offset=i_chunk_offset,
            args=args,
        )

        indices_sampled.append(indices)
        ks_sampled.append(ks)
        ks_count_sampled.append(ks_count)
        ks_start_end_sampled.append(ks_start_end)
        scores_sampled.append(scores)
        if key_access_log is not None:
            key_access_log_sampled.append(key_access_log)
        if key_access_count is not None:
            key_access_count_sampled.append(key_access_count)
        if block_access_log is not None:
            block_access_log_sampled.append(block_access_log)
        if block_access_score is not None:
            block_access_score_sampled.append(block_access_score)
        if block_access_count is not None:
            block_access_count_sampled.append(block_access_count)

    if len(indices_sampled) > 1:
        ignore_ranage = (
            max(
                cdiv_python(args.mask_k, args.block_size_q),
                cdiv_python(args.chunk_size, args.block_size_q * args.num_unions),
            )
            * 2
        )
        compute_range = cdiv_python(q.shape[1], args.block_size_q) - ignore_ranage

        bcs = args.chunk_size // args.block_size_q
        bcs_step = bcs // args.num_unions
        indices = torch.cat(
            [
                x[:, bcs - bcs_step * ix : x.shape[1] - bcs_step * ix]
                for ix, x in enumerate(indices_sampled)
            ],
            dim=-1,
        )[:, -compute_range:]
        scores = torch.cat(
            [
                x[:, bcs - bcs_step * ix : x.shape[1] - bcs_step * ix]
                for ix, x in enumerate(scores_sampled)
            ],
            dim=-1,
        )[:, -compute_range:]

        indices_to_sorted = torch.argsort(indices, dim=-1)

        indices = indices.gather(dim=-1, index=indices_to_sorted)
        scores = scores.gather(dim=-1, index=indices_to_sorted)

        unique_indices_mask = indices != torch.roll(indices, shifts=(1,), dims=(2,))
        scores.masked_fill_(~unique_indices_mask, float("-inf"))

        scores_to_highest = torch.argsort(scores, dim=-1, descending=True)[
            :,
            :,
            : triton.cdiv((args.mask_k * args.topk_head_group_size), args.block_size_k),
        ]

        indices = indices.gather(dim=-1, index=scores_to_highest)
        scores = scores.gather(dim=-1, index=scores_to_highest)

        top_indices_to_sorted = torch.argsort(indices, dim=-1)

        indices = indices.gather(dim=-1, index=top_indices_to_sorted)
        scores = scores.gather(dim=-1, index=top_indices_to_sorted)

        indices_sampled[0][:, ignore_ranage:, :] = indices

        indices = indices_sampled[0]
        ks = ks_sampled[0]
        # ks_count = ks_count_sampled[0]
        # ks_start_end = ks_start_end_sampled[0]

        BSZ, TDST, H, _ = q.shape
        _, TSRC, _, _ = k.shape
        BDST = triton.cdiv(TDST, args.block_size_q)
        mask_block_k = triton.cdiv(args.mask_k, args.block_size_k)

        ks_count = torch.zeros((B, BDST, G), dtype=torch.int32, device=q.device)
        ks_start_end = torch.zeros((B, BDST, G + 1), dtype=torch.int32, device=q.device)

        BLOCK_BK = 128
        grid = (B, BDST, triton.cdiv(indices.shape[-1], BLOCK_BK))
        pre_device = torch.get_default_device()
        torch.set_default_device(indices.device)
        masking_iteration_draft_cuda_epiloge[grid](
            indices,
            *indices.stride(),
            ks,
            *ks.stride(),
            ks_count,
            *ks_count.stride(),
            ks_start_end,
            *ks_start_end.stride(),
            mask_block_k,
            TSRC,
            G,
            BLOCK_BK,
        )
        torch.set_default_device(pre_device)

        ks = ks_count.sum(-1)

        if len(key_access_log_sampled) > 0:
            key_access_log = torch.cat(key_access_log_sampled, dim=1)
            key_access_count = torch.cat(key_access_count_sampled, dim=1)
        else:
            key_access_log = None
            key_access_count = None

        if len(block_access_log_sampled) > 0:
            block_access_log = torch.cat(block_access_log_sampled, dim=1)
            block_access_score = torch.cat(block_access_score_sampled, dim=1)
            block_access_count = torch.cat(block_access_count_sampled, dim=1)
        else:
            block_access_log = None
            block_access_score = None
            block_access_count = None
    else:
        indices = indices_sampled[0]
        ks = ks_sampled[0]
        ks_count = ks_count_sampled[0]
        ks_start_end = ks_start_end_sampled[0]

        if len(key_access_log_sampled) > 0:
            key_access_log = key_access_log_sampled[0]
            key_access_count = key_access_count_sampled[0]
        else:
            key_access_log = None
            key_access_count = None

        if len(block_access_log_sampled) > 0:
            block_access_log = block_access_log_sampled[0]
            block_access_score = block_access_score_sampled[0]
            block_access_count = block_access_count_sampled[0]
        else:
            block_access_log = None
            block_access_score = None
            block_access_count = None

    if (os.getenv("HIP_DEBUG", "0") == "1") and (
        not torch.cuda.is_current_stream_capturing()
    ):
        max_query = 1024
        B, TDST, H, HID = q.shape
        TDST = min(TDST, max_query)
        if k is not None:
            _, TSRC, H_KV, _ = k.shape
        else:
            TSRC = torch.max(args.cache_seq_lens).item()
        N = B * H

        def render_mask():
            debug_mask = to_dense(
                indices[:, -cdiv_python(TDST, args.block_size_q) :].cpu().numpy(),
                ks[:, -cdiv_python(TDST, args.block_size_q) :].cpu().numpy(),
                None,
                cdiv_python(N, args.topk_head_group_size),
                TDST,
                TSRC * args.topk_head_group_size,
                args.block_size_q,
                args.block_size_k * args.block_size_k_group,
            )[0]
            if args.group_size_q > 1:
                debug_mask = debug_mask.repeat(axis=0, repeats=args.group_size_q)

            cv2.imwrite("dummy_raw.png", debug_mask.astype(np.uint8) * 255)
            print(
                "saved dummy_raw.png",
                indices.shape,
                ks.shape,
                debug_mask.shape,
                q.shape,
                TSRC,
            )
            # input()

            # plt.clf()
            # plt.figure(figsize=(4*args.topk_head_group_size, 4))
            # plt.imshow(debug_mask)
            # plt.tight_layout()
            # plt.savefig('dummy.png', dpi=96, bbox_inches='tight')
            # print('saved dummy.png')

        # render_mask()
        render_mask()
        # if q.shape[1] > 64000:
        # else:
        #     print(q.shape[1])

    if (args.block_size_k_after_masking > 0) and (
        args.block_size_k_after_masking != args.block_size_k
    ):
        warnings.warn(f"block size k after masking {args.block_size_k_after_masking}")
        indices = (
            indices // args.block_size_k_after_masking * args.block_size_k_after_masking
        )

        # indices = indices.sort(dim=-1).values
        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
        indices = indices.sort(dim=-1).values
        # active_mask = unique_mask
        active_mask = indices < (
            args.position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
            + args.block_size_q
        )
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end[:, :, -1] = ks

        args = args.clone()
        args.block_size_k = args.block_size_k_after_masking
        args.block_size_k_after_masking = -1

    HIP_DEBUG_MASK_UNIQUE = os.getenv("HIP_DEBUG_MASK_UNIQUE", "0") == "1"
    if HIP_DEBUG_MASK_UNIQUE:
        # indices = indices.sort(dim=-1).values
        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
        indices = indices.sort(dim=-1).values
        # active_mask = unique_mask
        active_mask = indices < (
            args.position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
            + args.block_size_q
        )
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end[:, :, -1] = ks

    return (
        indices,
        ks,
        ks_count,
        ks_start_end,
        key_access_log,
        key_access_count,
        block_access_log,
        block_access_score,
        block_access_count,
        args,
    )


@nvtx.annotate("hip_attention")
@torch.inference_mode()
def hip_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    args: Optional[HiPAttentionArgs] = None,
    previous_metadata: Optional[HiPAttentionOutputMetadata] = None,
    mask_only: bool = False,
    **kwargs,
) -> Tuple[Tensor, HiPAttentionOutputMetadata]:
    if args is None:
        args = HiPAttentionArgs(**kwargs)

    if not args.is_causal:
        if args.sliding_window_size > 0:
            warnings.warn("sliding_window is not supported for bidirectional yet")
            args = args.clone()
            args.sliding_window_size = 0
        if args.sink_token_size > 0:
            warnings.warn("sink token is not supported for bidirectional yet")
            args = args.clone()
            args.sink_token_size = 0

    if args.num_dense_queries > 0:
        dense_context = flash_attn_func(
            q=q[:, : args.num_dense_queries],
            k=k[:, : args.num_dense_queries],
            v=v[:, : args.num_dense_queries],
            softmax_scale=1,
            causal=True,
        )

        num_sparse_queries = q.shape[1] - args.num_dense_queries
        if num_sparse_queries > 0:
            sparse_args = args.clone()
            sparse_args.num_dense_queries = -1
            sparse_context, metadata = hip_attention(
                q[:, -num_sparse_queries:],
                k,
                v,
                previous_metadata=previous_metadata,
                args=sparse_args,
            )

            return (torch.cat([dense_context, sparse_context], dim=1), metadata)
        else:
            return dense_context, None

    assert q.ndim == 4
    assert k.ndim == 4

    if args.position_ids is None:
        args = args.clone()
        args.position_ids = (
            torch.arange(0, q.shape[1], device=q.device) + k.shape[1] - q.shape[1] + 1
        )[None, :].expand(q.shape[0], -1)

    if previous_metadata is None:
        (
            indices,
            ks,
            ks_count,
            ks_start_end,
            key_access_log,
            key_access_count,
            block_access_log,
            block_access_score,
            block_access_count,
            args,
        ) = hip_masking(
            # TODO(-): apply PCA topk
            q=args.get_q_quant(q),
            k=args.get_k_quant(k),
            args=args,
        )
    else:
        indices = previous_metadata.indices
        ks = previous_metadata.ks
        ks_count = previous_metadata.ks_count
        ks_start_end = previous_metadata.ks_start_end
        key_access_log = previous_metadata.key_access_log
        key_access_count = previous_metadata.key_access_count
        block_access_log = previous_metadata.block_access_log
        block_access_score = previous_metadata.block_access_score
        block_access_count = previous_metadata.block_access_count

    if (os.getenv("HIP_DEBUG", "0") == "1") and (
        not torch.cuda.is_current_stream_capturing()
    ):
        max_query = 1024
        B, TDST, H, HID = q.shape
        TDST = min(TDST, max_query)
        if k is not None:
            _, TSRC, H_KV, _ = k.shape
        else:
            TSRC = torch.max(args.cache_seq_lens).item()
        N = B * H

        def render_mask():
            debug_mask = to_dense(
                indices[:, -cdiv_python(TDST, args.block_size_q) :].cpu().numpy(),
                ks[:, -cdiv_python(TDST, args.block_size_q) :].cpu().numpy(),
                None,
                cdiv_python(N, args.topk_head_group_size),
                TDST,
                TSRC * args.topk_head_group_size,
                args.block_size_q,
                args.block_size_k * args.block_size_k_group,
            )[0]
            if args.group_size_q > 1:
                debug_mask = debug_mask.repeat(axis=0, repeats=args.group_size_q)

            cv2.imwrite("dummy_final_raw.png", debug_mask.astype(np.uint8) * 255)
            print(
                "saved dummy_final_raw.png",
                indices.shape,
                ks.shape,
                debug_mask.shape,
                q.shape,
                TSRC,
            )

            # plt.clf()
            # plt.figure(figsize=(4*args.topk_head_group_size, 4))
            # plt.imshow(debug_mask)
            # plt.tight_layout()
            # plt.savefig('dummy.png', dpi=96, bbox_inches='tight')
            # print('saved dummy.png')

        # render_mask()
        render_mask()
        # if q.shape[1] > 64000:
        # else:
        #     print(q.shape[1])

    metadata = HiPAttentionOutputMetadata(
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        key_access_log=key_access_log,
        key_access_count=key_access_count,
        block_access_log=block_access_log,
        block_access_score=block_access_score,
        block_access_count=block_access_count,
    )

    if (os.getenv("HIP_DEBUG_SKIP_SA", "0") == "1") or mask_only:
        return None, metadata

    context = block_sparse_attention(
        q=q,
        k=k,
        v=v,
        seq_lens=args.position_ids,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
    )

    return context, metadata


@nvtx.annotate("paged_hip_attention")
def paged_hip_attention(
    q: Tensor,
    softmax_scale: float,
    args: HiPAttentionArgs,
    previous_mask_metadata: Optional[HiPAttentionOutputMetadata] = None,
):
    B, TDST, HEAD, HID = q.shape
    assert args.k_cache.shape[-1] == HID
    N_PAGES, PAGE_SIZE, HEAD_KV, HID = args.k_cache.shape
    assert args.v_cache.shape == args.k_cache.shape

    assert args.block_table.shape[0] == B
    assert args.cache_seq_lens.shape[0] == B

    if args.num_dense_queries > 0:
        warnings.warn("paged attention does not support dense queries.")

    if args.k_cache.dtype == torch.float8_e5m2:
        args.k_cache = args.k_cache.view(torch.uint8)
    if args.v_cache.dtype == torch.float8_e5m2:
        args.v_cache = args.v_cache.view(torch.uint8)
    if args.position_ids is None:
        args = args.clone()
        position_ids = (
            torch.arange(0, TDST, device=q.device)[None, :]
            + args.cache_seq_lens[:, None]
            - TDST
            + 1
        )
        args.position_ids = position_ids

    q_dtype = q.dtype
    q = (q * softmax_scale).to(q.dtype)  # .to(torch.float32)

    if previous_mask_metadata is None:
        # print(q.shape, args.json())
        (
            indices,
            ks,
            ks_count,
            ks_start_end,
            key_access_log,
            key_access_count,
            block_access_log,
            block_access_score,
            block_access_count,
            args,
        ) = hip_masking(q=q, k=None, args=args)
    else:
        indices = previous_mask_metadata.indices
        ks = previous_mask_metadata.ks
        ks_count = previous_mask_metadata.ks_count
        ks_start_end = previous_mask_metadata.ks_start_end
        key_access_log = previous_mask_metadata.key_access_log
        key_access_count = previous_mask_metadata.key_access_count
        block_access_log = previous_mask_metadata.block_access_log
        block_access_score = previous_mask_metadata.block_access_score
        block_access_count = previous_mask_metadata.block_access_count

        if args.block_size_k_after_masking > 0:
            args = args.clone()
            args.block_size_k = args.block_size_k_after_masking
            args.block_size_k_after_masking = -1

    context = block_sparse_attention(
        q=q,
        k=None,
        v=None,
        seq_lens=args.position_ids,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
    )

    context = context.to(q_dtype)

    return context, HiPAttentionOutputMetadata(
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        key_access_log=key_access_log,
        key_access_count=key_access_count,
        block_access_log=block_access_log,
        block_access_score=block_access_score,
        block_access_count=block_access_count,
    )


@nvtx.annotate("varlen_hip_attention")
def varlen_hip_attention(
    q: Tensor,
    softmax_scale: float,
    k: Tensor,
    v: Tensor,
    seq_lens: List[int],
    args: HiPAttentionArgs,
):
    q = q * softmax_scale

    outs = []
    total_length = 0
    for seq_len in seq_lens:
        seq_start = total_length
        seq_end = seq_start + seq_len
        total_length = seq_end

        # torch.cuda.synchronize()

        out, _ = hip_attention(
            q=q[seq_start:seq_end].unsqueeze(0),
            k=k[seq_start:seq_end].unsqueeze(0),
            v=v[seq_start:seq_end].unsqueeze(0),
            args=args.clone(),
        )

        # torch.cuda.synchronize()
        # from flash_attn import flash_attn_func
        # out_flash = flash_attn_func(
        #     q=q[seq_start:seq_end].unsqueeze(0),
        #     k=k[seq_start:seq_end].unsqueeze(0),
        #     v=v[seq_start:seq_end].unsqueeze(0),
        #     softmax_scale=1,
        #     causal=True,
        # )
        # print('varlen', seq_start, seq_end, out.shape, F.mse_loss(out, out_flash))

        outs.append(out.squeeze(0))

    return torch.cat(outs, dim=0)


@nvtx.annotate("paged_varlen_hiop_attention")
def paged_varlen_hip_attention(
    q: Tensor,
    softmax_scale: float,
    seq_lens: List[int],
    args: HiPAttentionArgs,
):
    # q = q * softmax_scale

    outs = []
    total_length = 0
    total_page_length = 0
    for idx_batch, seq_len in enumerate(seq_lens):
        seq_start = total_length
        seq_end = seq_start + seq_len

        page_start = total_page_length
        page_end = page_start + cdiv_python(seq_len, args.k_cache.shape[1])

        total_length = seq_len
        total_page_length = page_end

        curr_args = args.clone()
        curr_args.block_table = args.block_table[idx_batch : idx_batch + 1]
        curr_args.cache_seq_lens = args.cache_seq_lens[idx_batch : idx_batch + 1]

        out, _ = paged_hip_attention(
            q=q[seq_start:seq_end].unsqueeze(0),
            softmax_scale=softmax_scale,
            args=curr_args,
        )
        # print('varlen', seq_start, seq_end, out.shape)
        outs.append(out.squeeze(0))

    return torch.cat(outs, dim=0)
