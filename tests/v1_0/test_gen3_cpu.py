import math
import subprocess

handle = subprocess.Popen(
    "nvidia-smi --list-gpus".split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)
handle.wait()
num_gpus = len(handle.stdout.readlines())

import os

os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() // num_gpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count() // num_gpus)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // num_gpus)
os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count() // num_gpus)
import time

import numba
import numpy as np
import triton
from numpy import ndarray


@numba.njit(parallel=True, fastmath=True)
def scan_stage(
    q: ndarray,
    k: ndarray,
    indices_left: ndarray,
    indices_right: ndarray,
    block_size_q: int,
    block_stride_q: int,
    block_size_k: int,
):
    BSZ, TDST, HEAD, HID = q.shape
    _, TSRC, HEAD_KV, HID = k.shape
    _, BDST, _, CHUNK = indices_left.shape
    BCHUNK = int(math.ceil(CHUNK / block_size_k))
    assert indices_left.shape == indices_right.shape

    program_count = BSZ * HEAD * BDST * BCHUNK

    for _pid in numba.prange(program_count):
        idx_bchunk = _pid % BCHUNK
        pid = _pid // BCHUNK
        idx_bdst = pid % BDST
        pid = pid // BDST
        idx_head = pid % HEAD
        pid = pid // HEAD
        idx_bsz = pid

        idx_head_kv = idx_head // (HEAD // HEAD_KV)
        idx_tdst_start = idx_bdst * block_size_q
        idx_tdst = np.arange(
            idx_tdst_start, min(idx_tdst_start + block_size_q, TDST), block_stride_q
        )

        idx_chunk_start = idx_bchunk * block_size_k
        idx_chunk = np.arange(
            idx_chunk_start, min(idx_chunk_start + block_size_k, CHUNK)
        )

        queries = q[idx_bsz, idx_tdst, idx_head, :].astype(np.float32)

        chunk_left = indices_left[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ]
        chunk_right = indices_right[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ]
        while True:
            chunk_center = (chunk_left + chunk_right) // 2
            keys_left = k[
                idx_bsz, (chunk_left + chunk_center) // 2, idx_head_kv, :
            ].astype(np.float32)
            keys_right = k[
                idx_bsz, (chunk_center + chunk_right) // 2, idx_head_kv, :
            ].astype(np.float32)

            scores_left = queries @ keys_left.T
            reduced_left = np.empty((scores_left.shape[1],), dtype=scores_left.dtype)
            for i in range(reduced_left.shape[0]):
                reduced_left[i] = np.amax(scores_left[:, i])
            scores_left = reduced_left
            scores_right = queries @ keys_right.T
            reduced_right = np.empty((scores_right.shape[1],), dtype=scores_left.dtype)
            for i in range(reduced_right.shape[0]):
                reduced_right[i] = np.amax(scores_right[:, i])
            scores_right = reduced_right

            is_left_win = scores_left > scores_right
            chunk_left = np.where(is_left_win, chunk_left, chunk_center)
            chunk_right = np.where(is_left_win, chunk_center, chunk_right)

            chunk_size = chunk_right - chunk_left
            is_done = chunk_size <= 1
            if is_done.all():
                break

        indices_left[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ] = chunk_left
        indices_right[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ] = chunk_right


bsz = 1
tdst = 1
tsrc = 1024 * 1024
head = 32
head_kv = 8
hid = 128
dtype = np.float32

stage_chunk_size = 256
stage_block_size_q = 64
stage_block_stride_q = 4
# NOTE: tune this
block_size_k = 4

q = np.random.randn(bsz, tdst, head, hid).astype(dtype)
k = np.random.randn(bsz, tsrc, head_kv, hid).astype(dtype)

print("init done", flush=True)

elapsed = []
for i in range(100):
    indices_left = np.arange(0, tsrc, stage_chunk_size)[None, None, None, :]
    indices_left = np.tile(
        indices_left, (bsz, triton.cdiv(tdst, stage_block_size_q), head, 1)
    )
    indices_right = indices_left + stage_chunk_size

    t_start = time.monotonic() * 1000
    scan_stage(
        q,
        k,
        indices_left,
        indices_right,
        stage_block_size_q,
        stage_block_stride_q,
        block_size_k,
    )
    t_end = time.monotonic() * 1000

    print(t_end - t_start)
    if i > 3:
        elapsed.append(t_end - t_start)

print("avg", sum(elapsed) / len(elapsed))
