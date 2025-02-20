import math

import numba
import numpy as np
from numpy import ndarray as NdArray


@numba.njit(parallel=True)
def access_log_to_dense(
    key_access_log: NdArray,
    key_access_count: NdArray,
    TSRC,
    KV_HEAD_REPEAT,
):
    B, BDST, K = key_access_log.shape
    out_int = np.zeros((B // KV_HEAD_REPEAT, BDST, TSRC), dtype=np.int32)
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ikv in range(KV_HEAD_REPEAT):
            ib = ibh * KV_HEAD_REPEAT + ikv
            for ibdst in numba.prange(BDST):
                nk = key_access_count[ib, ibdst]
                for ik in range(min(nk, K)):
                    tsrc = key_access_log[ib, ibdst, ik]
                    if tsrc < TSRC:
                        out_int[ibh, ibdst, tsrc] += 1

    return out_int


@numba.njit(parallel=True)
def access_score_log_to_dense(
    key_access_log: NdArray,
    key_access_count: NdArray,
    TSRC,
    KV_HEAD_REPEAT,
    key_access_score: NdArray,
):
    B, BDST, K = key_access_log.shape
    out_fp = np.zeros((B // KV_HEAD_REPEAT, BDST, TSRC), dtype=np.float32) - 98765432
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ikv in range(KV_HEAD_REPEAT):
            ib = ibh * KV_HEAD_REPEAT + ikv
            for ibdst in numba.prange(BDST):
                nk = key_access_count[ib, ibdst]
                for ik in range(min(nk, K)):
                    tsrc = key_access_log[ib, ibdst, ik]
                    if tsrc < TSRC:
                        out_fp[ibh, ibdst, tsrc] = key_access_score[ib, ibdst, ik]
    return out_fp


@numba.njit
def img_reduce(img: NdArray, rh: int, rw: int):
    H, W = img.shape
    RH = H // rh
    RW = W // rw
    out = np.zeros((RH, RW))
    for ih in range(RH):
        for iw in range(RW):
            chunk = img[ih * rh : ih * rh + rh, iw * rw : iw * rw + rw]
            scaler = np.mean(chunk)
            out[ih, iw] = scaler
    return out


@numba.njit
def incr_first_iteration(
    mask: NdArray,
    block_size_q: int,
    mask_k: int,
    block_size_k: int,
    block_stride_k: int,
    sliding_window_size: int,
):
    B, BDST, TSRC = mask.shape
    for ib in range(B):
        for ibdst in range(BDST):
            _mask_k = mask_k * max(1, int(math.log2(ibdst * block_size_q / 8192)))
            for ibk in range(_mask_k // block_size_k):
                tsrc = ((ibdst + 1) * block_size_q - sliding_window_size) * (
                    ibk / (_mask_k // block_size_k)
                )
                tsrc = round(tsrc)
                if tsrc >= 0:
                    tsrc = tsrc - (tsrc % block_size_k)
                    for ioffset in range(
                        block_stride_k - 1, block_size_k, block_stride_k
                    ):
                        mask[ib, ibdst, tsrc + ioffset] += 1


@numba.jit
def numba_softmax(x, temperature):
    # x = np.clip(x + temperature, 0, temperature * 2)
    # return x

    x = x / temperature
    m = np.max(x)
    ex = np.exp(x - m)
    exsum = np.sum(ex)
    return ex / exsum


@numba.njit(parallel=True, fastmath=True, boundscheck=True)
def perform_lru_hot_prefetch_unified(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    mask_k,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
    prefetch_step,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    cache_type_map = np.zeros_like(key_access_map, dtype=np.int32)
    loaded_key_mask = np.zeros_like(key_access_map)

    b = mask_k
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT) * 3
    lru_budget = 0

    # LRU-temperature cache
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_is_prefetch = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.bool_,
    )
    loaded_key_timestamp = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_hit = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.float32,
    )
    loaded_key_missed = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.float32,
    )
    all_key_temperature = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            TSRC,
        ),
        dtype=np.float32,
    )
    all_key_rel_temperature = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            TSRC,
        ),
        dtype=np.float32,
    )
    momentum = 0.95
    decay_momentum = 0.9
    penalty_decay = 0.7

    # LRU prefetch cache
    prefetch_candidate = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    prefetch_candidate_try = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        + 1
    )
    prefetch_candidate_priority = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.float32,
        )
        - 1
    )

    avg_lru_budget = np.zeros((B // KV_HEAD_REPEAT,), dtype=np.int32)

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            seq_len = (ibdst + 1) + block_size_q
            b = mask_k
            s = lru_budget_log_scale
            lru_budget = math.floor(math.log2(seq_len / b + 1) * b * s * KV_HEAD_REPEAT)
            fetch_budget = lru_budget

            # --- begin of handle previous cache miss ---

            # before caching and prefetching, evict not used long time
            # for icache in range(lru_budget):
            #     cached_pointer = loaded_key_list[ibh, icache]
            #     if cached_pointer < 0: continue

            #     is_prefetch = loaded_key_is_prefetch[ibh, icache]
            #     if (loaded_key_missed[ibh, icache] > max(1, loaded_key_hit[ibh, icache]) * 3):
            #         if not is_prefetch:
            #             all_key_temperature[ibh, cached_pointer] *= penalty_decay
            #         else:
            #             all_key_rel_temperature[ibh, seq_len - cached_pointer - 1] *= penalty_decay

            # handle cache misses
            for ikv in range(KV_HEAD_REPEAT):
                last_accessed = key_access_log[ibh * KV_HEAD_REPEAT + ikv, ibdst - 1, :]
                last_accessed_count = key_access_count[
                    ibh * KV_HEAD_REPEAT + ikv, ibdst - 1
                ]

                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count)):
                    current_pointer = last_accessed[ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    in_candidate = False
                    least_priority_val = 999999999
                    victim_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer < 0:
                            victim_idx = icache
                            least_priority_val = -1
                            continue

                        if cached_pointer == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            if loaded_key_missed[ibh, icache] != 0:
                                loaded_key_hit[ibh, icache] = 1
                            else:
                                loaded_key_hit[ibh, icache] += 1
                            loaded_key_missed[ibh, icache] = 0
                            in_cache = True
                            break
                        else:
                            is_prefetch = loaded_key_is_prefetch[ibh, icache]
                            if is_prefetch:
                                temperature = all_key_rel_temperature[
                                    ibh, seq_len - cached_pointer - 1
                                ]
                            else:
                                temperature = all_key_temperature[ibh, cached_pointer]
                            if temperature < least_priority_val:
                                least_priority_val = temperature
                                victim_idx = icache

                    if all_key_temperature[ibh, current_pointer] < 1e-10:
                        all_key_temperature[ibh, current_pointer] = 1
                    else:
                        all_key_temperature[ibh, current_pointer] = (
                            momentum * all_key_temperature[ibh, current_pointer]
                            + (1 - momentum)
                        )

                    current_rel_pointer = seq_len - current_pointer - 1
                    if all_key_rel_temperature[ibh, current_rel_pointer] < 1e-10:
                        all_key_rel_temperature[ibh, current_rel_pointer] = 1
                    else:
                        all_key_rel_temperature[ibh, current_rel_pointer] = (
                            momentum * all_key_rel_temperature[ibh, current_rel_pointer]
                            + (1 - momentum)
                        )

                    # handle candidates
                    for ifetch in range(fetch_budget):
                        candidate_pointer = prefetch_candidate[ibh, ifetch]
                        if candidate_pointer == current_pointer:
                            # push candidate prefetch
                            prefetch_candidate_try[ibh, ifetch] -= 1
                            prefetch_candidate_priority[ibh, ifetch] = ibdst
                            in_candidate = True

                    if (not in_cache) and (not in_candidate):
                        # print(current_pointer + prefetch_step, (seq_len - sliding_window_size - 2 * seq_len / mask_k))
                        candidate_victim_idx = np.argmin(
                            prefetch_candidate_priority[ibh, :fetch_budget]
                        )
                        prefetch_candidate[ibh, candidate_victim_idx] = (
                            current_pointer - prefetch_step
                        )
                        prefetch_candidate_try[ibh, candidate_victim_idx] = (
                            2 * 8 // block_size_q
                        )
                        prefetch_candidate_priority[ibh, candidate_victim_idx] = ibdst

                    will_prefetch = False
                    for ifetch in range(lru_budget):
                        is_prefetch = loaded_key_is_prefetch[ibh, ifetch]
                        current_fetch = loaded_key_list[ibh, ifetch]
                        next_fetch = current_fetch + prefetch_step
                        if (
                            is_prefetch
                            and (current_fetch >= 0)
                            and (next_fetch == current_pointer)
                        ):
                            will_prefetch = True

                    # if victim has cooler, then push to cache
                    victim_pointer = loaded_key_list[ibh, victim_idx]
                    is_victim_prefetch = loaded_key_is_prefetch[ibh, victim_idx]
                    if is_victim_prefetch:
                        victim_temperature = all_key_rel_temperature[
                            ibh, seq_len - victim_pointer - 1
                        ]
                    else:
                        victim_temperature = all_key_temperature[ibh, victim_pointer]
                    if (
                        (not in_cache)
                        and (not will_prefetch)
                        and (
                            (victim_pointer < 0)
                            or (
                                all_key_temperature[ibh, current_pointer]
                                >= victim_temperature
                            )
                        )
                    ):
                        loaded_key_list[ibh, victim_idx] = current_pointer
                        loaded_key_timestamp[ibh, victim_idx] = ibdst
                        loaded_key_is_prefetch[ibh, victim_idx] = False
                        loaded_key_hit[ibh, victim_idx] = 0
                        loaded_key_missed[ibh, victim_idx] = 0

            # kill not used candidates
            for ifetch in range(fetch_budget):
                candidate_pointer = prefetch_candidate[ibh, ifetch]
                if prefetch_candidate_priority[ibh, ifetch] < (
                    ibdst - 16 * 3 // block_size_q
                ):
                    prefetch_candidate[ibh, ifetch] = -1
                    prefetch_candidate_try[ibh, ibdst] = 999999999
                    prefetch_candidate_priority[ibh, ifetch] = -1

            # depending on current step, prefetch
            for ifetch in range(fetch_budget):
                if prefetch_candidate[ibh, ifetch] >= 0:
                    prefetch_candidate[ibh, ifetch] += prefetch_step
            for icache in range(lru_budget):
                if (loaded_key_list[ibh, icache] >= 0) and loaded_key_is_prefetch[
                    ibh, icache
                ]:
                    t = loaded_key_list[ibh, icache] + prefetch_step
                    # while t in loaded_key_list[ibh, :lru_budget]:
                    #     t += prefetch_step
                    loaded_key_list[ibh, icache] = t
            for icache in range(lru_budget):
                if (loaded_key_list[ibh, icache] >= 0) and loaded_key_is_prefetch[
                    ibh, icache
                ]:
                    new_prefetch_pointer = loaded_key_list[ibh, icache]
                    if new_prefetch_pointer in loaded_key_list[ibh, :lru_budget]:
                        victim_idx = np.argmax(
                            (
                                new_prefetch_pointer
                                == loaded_key_list[ibh, :lru_budget]
                            ).astype(np.int32)
                        )
                        loaded_key_list[ibh, victim_idx] = -1

            for ifetch in range(fetch_budget):
                candidate_pointer = prefetch_candidate[ibh, ifetch]
                if candidate_pointer < 0:
                    continue
                if prefetch_candidate_try[ibh, ifetch] > 0:
                    continue

                prefetch_candidate_try[ibh, ifetch] = 99999999
                prefetch_candidate[ibh, ifetch] = -1
                prefetch_candidate_priority[ibh, ifetch] = -1

                if candidate_pointer in loaded_key_timestamp[ibh, :lru_budget]:
                    continue

                victim_idx = -1
                victim_val = 999999
                for icache in range(lru_budget):
                    victim_pointer = loaded_key_list[ibh, icache]
                    if victim_pointer < 0:
                        victim_idx = icache
                        break
                    if loaded_key_is_prefetch[ibh, icache]:
                        t = all_key_rel_temperature[ibh, seq_len - victim_pointer - 1]
                    else:
                        t = all_key_temperature[ibh, victim_pointer]
                    if t < victim_val:
                        victim_idx = icache
                        victim_val = t
                victim_pointer = loaded_key_list[ibh, victim_idx]
                if loaded_key_is_prefetch[ibh, victim_idx]:
                    victim_temp = all_key_rel_temperature[
                        ibh, seq_len - victim_pointer - 1
                    ]
                else:
                    victim_temp = all_key_temperature[ibh, victim_pointer]
                if (victim_pointer < 0) or (
                    all_key_rel_temperature[ibh, seq_len - candidate_pointer - 1]
                    >= victim_temp
                ):
                    loaded_key_timestamp[ibh, victim_idx] = ibdst
                    loaded_key_list[ibh, victim_idx] = candidate_pointer
                    loaded_key_is_prefetch[ibh, victim_idx] = True
                    loaded_key_hit[ibh, victim_idx] = 0
                    loaded_key_missed[ibh, victim_idx] = 0

            # promote normal cache to prefetch
            # for icache in range(lru_budget):
            #     if loaded_key_list[ibh, icache] < 0: continue
            #     if loaded_key_is_prefetch[ibh, icache]: continue
            #     current_pointer = loaded_key_list[ibh, icache]
            #     if all_key_temperature[ibh, current_pointer] < all_key_rel_temperature[ibh, seq_len - (current_pointer - prefetch_step) - 1]:
            #         loaded_key_is_prefetch[ibh, icache] = True

            # --- end of handle previous cache miss ---

            # <--- actual current step's cache bank

            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1
                    if loaded_key_is_prefetch[ibh, icache]:
                        cache_type_map[ibh, ibdst, idx] = 255
                    else:
                        cache_type_map[ibh, ibdst, idx] = 64

            all_key_temperature[ibh] *= decay_momentum
            all_key_rel_temperature[ibh] *= decay_momentum
            loaded_key_missed[ibh] += 1

        avg_lru_budget[ibh] = lru_budget

    return loaded_key_mask, cache_type_map, loaded_key_list, avg_lru_budget.sum()


@numba.njit(parallel=True)
def perform_lru_hot_prefetch(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    mask_k,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
    prefetch_step,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    cache_type_map = np.zeros_like(key_access_map)
    loaded_key_mask = np.zeros_like(key_access_map)

    b = mask_k
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    prefetch_ratio = 0.1
    max_prefetch_budget = math.ceil(max_lru_budget * prefetch_ratio)
    max_lru_budget = math.ceil(max_lru_budget * (1 - prefetch_ratio))

    # LRU-temperature cache
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget + max_prefetch_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_timestamp = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget + max_prefetch_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    all_key_temperature = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            TSRC,
        ),
        dtype=np.float32,
    )
    all_key_rel_temperature = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            TSRC,
        ),
        dtype=np.float32,
    )
    momentum = 0.7
    decay_momentum = 0.95

    # LFU or LRU prefetch cache
    #   candidate priority is frequency
    #   prefetch key priority is frequencey
    prefetch_candidate = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_prefetch_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    prefetch_candidate_try = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_prefetch_budget,
            ),
            dtype=np.int32,
        )
        + 1
    )
    prefetch_candidate_priority = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_prefetch_budget,
            ),
            dtype=np.float32,
        )
        - 1
    )
    prefetch_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    prefetch_key_priority = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = mask_k
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )
            prefetch_budget = math.floor(lru_budget * prefetch_ratio)
            # prefetch_budget = min(prefetch_budget, mask_k * 2)
            lru_budget = lru_budget - prefetch_budget

            all_key_temperature[ibh] *= decay_momentum
            all_key_rel_temperature[ibh] *= decay_momentum

            # --- begin of handle previous cache miss ---

            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    in_candidate = False
                    least_priority_val = 999999999
                    victim_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            all_key_temperature[ibh, current_pointer] = (
                                momentum * all_key_temperature[ibh, current_pointer]
                                + (1 - momentum)
                            )
                            all_key_rel_temperature[ibh, current_pointer] = (
                                momentum * all_key_rel_temperature[ibh, current_pointer]
                                + (1 - momentum)
                            )
                            in_cache = True
                            break
                        else:
                            temperature = all_key_temperature[ibh, cached_pointer]
                            if temperature < least_priority_val:
                                least_priority_val = temperature
                                victim_idx = icache

                    for ifetch in range(prefetch_budget):
                        prefetch_pointer = prefetch_key_list[ibh, ifetch]
                        if prefetch_pointer == current_pointer:
                            prefetch_key_priority[ibh, ifetch] = ibdst
                            # all_key_temperature[ibh, current_pointer] = momentum * all_key_temperature[ibh, current_pointer] + (1 - momentum)
                            in_cache = True
                        candidate_pointer = prefetch_candidate[ibh, ifetch]
                        if candidate_pointer == current_pointer:
                            # push candidate prefetch
                            prefetch_candidate_try[ibh, ifetch] -= 1
                            prefetch_candidate_priority[ibh, ifetch] = ibdst
                            in_candidate = True

                    # need to push cache
                    if not in_cache:
                        # update temperature
                        all_key_temperature[ibh, current_pointer] = (
                            momentum * all_key_temperature[ibh, current_pointer]
                            + (1 - momentum)
                        )
                        # add to prefetch candidate

                    if (not in_cache) and (not in_candidate):
                        least_candidate_priority_val = 999999999
                        candidate_victim_idx = -1
                        for ifetch in range(prefetch_budget):
                            priority = prefetch_candidate_priority[ibh, ifetch]
                            if priority < least_candidate_priority_val:
                                least_candidate_priority_val = priority
                                candidate_victim_idx = ifetch
                        prefetch_candidate[ibh, candidate_victim_idx] = (
                            current_pointer - prefetch_step
                        )
                        prefetch_candidate_try[ibh, candidate_victim_idx] = (
                            1 * 8 // block_size_q
                        )
                        prefetch_candidate_priority[ibh, candidate_victim_idx] = ibdst

                    # if victim has cooler, then push to cache
                    if (not in_cache) and (
                        all_key_temperature[ibh, current_pointer]
                        >= all_key_temperature[ibh, loaded_key_list[ibh, victim_idx]]
                    ):
                        loaded_key_list[ibh, victim_idx] = current_pointer
                        loaded_key_timestamp[ibh, victim_idx] = ibdst

            # before start next step, perform prefetch
            for ifetch in range(prefetch_budget):
                # udpate prefetch entry
                if prefetch_key_list[ibh, ifetch] >= 0:
                    prefetch_key_list[ibh, ifetch] += prefetch_step
                    if prefetch_key_priority[ibh, ifetch] < (
                        ibdst - (16 * 8 // block_size_q)
                    ):
                        prefetch_key_priority[ibh, ifetch] = -1
                        prefetch_key_list[ibh, ifetch] = -1
                # update candiate
                if prefetch_candidate[ibh, ifetch] >= 0:
                    prefetch_candidate[ibh, ifetch] += prefetch_step

            for ifetch in range(prefetch_budget):
                if prefetch_candidate_try[ibh, ifetch] <= 0:
                    candidate_pointer = prefetch_candidate[ibh, ifetch]
                    candidate_priority = prefetch_candidate_priority[ibh, ifetch]

                    prefetch_candidate_try[ibh, ifetch] = 1
                    prefetch_candidate[ibh, ifetch] = -1
                    prefetch_candidate_priority[ibh, ifetch] = -1

                    if (
                        candidate_pointer
                        not in prefetch_key_list[ibh, :prefetch_budget]
                    ):
                        victim_idx = np.argmin(
                            prefetch_key_priority[ibh, :prefetch_budget]
                        )
                        prefetch_key_priority[ibh, victim_idx] = candidate_priority
                        prefetch_key_list[ibh, victim_idx] = candidate_pointer

            # --- end of handle previous cache miss ---

            # <--- actual current step's cache bank

            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1
                    cache_type_map[ibh, ibdst, idx] = 255
            for ifetch in range(prefetch_budget):
                idx = prefetch_key_list[ibh, ifetch]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1
                    cache_type_map[ibh, ibdst, idx] = 64

    return loaded_key_mask, cache_type_map


@numba.njit(parallel=True)
def perform_gd_score(
    key_access_map,
    key_access_log,
    key_access_score,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
    temperature,
    minimum_cost=1000,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    # for output
    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_probs_map = np.zeros_like(key_access_map, dtype=np.float32)

    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_scores = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.float32,
        )
        - 987654321
    )
    loaded_key_h = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.float32,
    )
    loaded_key_h_init = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.float32,
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        last_min_h = 0
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )
            probs = numba_softmax(loaded_key_scores[ibh], temperature)
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    min_h_value = 99999999
                    victim_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_h[ibh, icache] = (
                                last_min_h + probs[icache] + minimum_cost
                            )
                            in_cache = True
                        else:
                            h = loaded_key_h[ibh, icache]
                            if h < min_h_value:
                                min_h_value = h
                                victim_idx = icache

                    last_min_h = min_h_value

                    # else, evict victim
                    if not in_cache:
                        min_h = min_h_value

                        # decrease by L
                        loaded_key_h[ibh, :] -= min_h

                        # enqueue to cache
                        loaded_key_scores[ibh, victim_idx] = key_access_score[
                            ib, ibdst, ik
                        ]
                        probs = numba_softmax(loaded_key_scores[ibh], temperature)

                        new_h = min_h_value + probs[victim_idx] + minimum_cost
                        loaded_key_list[ibh, victim_idx] = current_pointer
                        loaded_key_h[ibh, victim_idx] = new_h
                        loaded_key_h_init[ibh, victim_idx] = new_h
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1
                    loaded_key_probs_map[ibh, ibdst, idx] = probs[icache]

    return loaded_key_mask, loaded_key_probs_map


@numba.njit(parallel=True)
def perform_lru_score(
    key_access_map,
    key_access_log,
    key_access_score,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_score = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.float32,
        )
        - 987654321
    )

    momemtum = 0.9

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )

            loaded_key_score[ibh] *= momemtum

            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_timestamp_val = 999999999.0
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_score[ibh, icache] = momemtum * loaded_key_score[
                                ibh, icache
                            ] + key_access_score[ib, ibdst, ik] * (1 - momemtum)
                            in_cache = True
                            break
                        else:
                            score = loaded_key_score[ibh, icache]
                            if score < least_timestamp_val:
                                least_timestamp_val = score
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        loaded_key_list[ibh, least_timestamp_idx] = current_pointer
                        loaded_key_score[ibh, least_timestamp_idx] = key_access_score[
                            ib, ibdst, ik
                        ]
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lru_hot_score(
    key_access_map,
    key_access_log,
    key_access_score,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_score = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.float32,
        )
        - 987654321
    )
    all_key_cost = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            TSRC,
        ),
        dtype=np.float32,
    )

    momentum = 0.7
    decay_momentum = 0.95

    score_momemtum = 0.8
    score_decay = 0.95

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )

            all_key_cost[ibh] *= decay_momentum
            loaded_key_score[ibh] *= score_decay

            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_hot_val = 0
                    victim_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_score[ibh, icache] = (
                                score_momemtum * loaded_key_score[ibh, icache]
                                + key_access_score[ib, ibdst, ik] * (1 - score_momemtum)
                            )
                            all_key_cost[ibh, current_pointer] = (
                                momentum * all_key_cost[ibh, current_pointer]
                                + (1 - momentum)
                            )
                            in_cache = True
                            break
                        else:
                            is_victim = False
                            score = loaded_key_score[ibh, icache]
                            cost = (
                                all_key_cost[ibh, cached_pointer] * 0.01 + score * 0.99
                            )

                            hot = 1 - cost
                            if hot > least_hot_val:
                                is_victim = True

                            if is_victim:
                                least_hot_val = hot
                                victim_idx = icache
                    # else, evict victim
                    if not in_cache:
                        all_key_cost[ibh, current_pointer] = momentum * all_key_cost[
                            ibh, current_pointer
                        ] + (1 - momentum)
                    if (not in_cache) and (
                        all_key_cost[ibh, current_pointer]
                        >= all_key_cost[ibh, loaded_key_list[ibh, victim_idx]]
                    ):
                        loaded_key_list[ibh, victim_idx] = current_pointer
                        loaded_key_score[ibh, victim_idx] = key_access_score[
                            ib, ibdst, ik
                        ]
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lru_hot(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_timestamp = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    all_key_cost = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            TSRC,
        ),
        dtype=np.float32,
    )
    momentum = 0.7
    decay_momentum = 0.95

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )

            all_key_cost[ibh] *= decay_momentum

            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_timestamp_val = 0
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            all_key_cost[ibh, current_pointer] = (
                                momentum * all_key_cost[ibh, current_pointer]
                                + (1 - momentum)
                            )
                            in_cache = True
                            break
                        else:
                            # timestamp = loaded_key_timestamp[ibh, icache]
                            # dist = ibdst - timestamp
                            cost = all_key_cost[ibh, cached_pointer]
                            score = 1 - cost
                            if score > least_timestamp_val:
                                least_timestamp_val = score
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        all_key_cost[ibh, current_pointer] = momentum * all_key_cost[
                            ibh, current_pointer
                        ] + (1 - momentum)
                    if (
                        not in_cache
                        and all_key_cost[ibh, current_pointer]
                        >= all_key_cost[ibh, loaded_key_list[ibh, least_timestamp_idx]]
                    ):
                        loaded_key_list[ibh, least_timestamp_idx] = current_pointer
                        loaded_key_timestamp[ibh, least_timestamp_idx] = ibdst
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lru(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_timestamp = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_timestamp_val = 999999999
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            in_cache = True
                            break
                        else:
                            if loaded_key_timestamp[ibh, icache] < least_timestamp_val:
                                least_timestamp_val = loaded_key_timestamp[ibh, icache]
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        loaded_key_list[ibh, least_timestamp_idx] = current_pointer
                        loaded_key_timestamp[ibh, least_timestamp_idx] = ibdst
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lru_k(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
    lru_k,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_timestamp = np.zeros(
        (B // KV_HEAD_REPEAT, max_lru_budget, lru_k), dtype=np.int32
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_timestamp_val = 999999999
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_timestamp[ibh, icache, ibdst % lru_k] = ibdst
                            in_cache = True
                            break
                        else:
                            timestamp = loaded_key_timestamp[
                                ibh, icache, (ibdst - 1) % lru_k
                            ]
                            if timestamp < least_timestamp_val:
                                least_timestamp_val = timestamp
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        loaded_key_list[ibh, least_timestamp_idx] = current_pointer
                        loaded_key_timestamp[ibh, least_timestamp_idx, :] = 999999999
                        loaded_key_timestamp[
                            ibh, least_timestamp_idx, ibdst % lru_k
                        ] = ibdst
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lru_tie_break_lre(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_timestamp = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    all_key_last_evicted = np.zeros((B // KV_HEAD_REPEAT, TSRC), dtype=np.int32)

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )
            # if ibh == 0:
            #     print(all_key_last_evicted[ibh, 1024:1034])
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_timestamp_val = 999999999
                    least_timestamp_last_evict = 999999999
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            in_cache = True
                            break
                        else:
                            timestamp = loaded_key_timestamp[ibh, icache]
                            last_evict = all_key_last_evicted[
                                ibh, loaded_key_list[ibh, icache]
                            ]
                            is_victim = False
                            if timestamp < least_timestamp_val:
                                is_victim = True

                            if (timestamp == least_timestamp_val) and (
                                last_evict < least_timestamp_last_evict
                            ):
                                is_victim = True

                            if is_victim:
                                least_timestamp_last_evict = last_evict
                                least_timestamp_val = timestamp
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        all_key_last_evicted[
                            ibh, loaded_key_list[ibh, least_timestamp_idx]
                        ] = ibdst
                        loaded_key_list[ibh, least_timestamp_idx] = current_pointer
                        loaded_key_timestamp[ibh, least_timestamp_idx] = ibdst
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lru_tie_break_lfu(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_timestamp = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_freq = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.int32,
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )

            loaded_key_freq[ibh] -= 1

            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_timestamp_val = 999999999
                    least_timestamp_freq = 999999999
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            loaded_key_freq[ibh, icache] += 1
                            in_cache = True
                            break
                        else:
                            timestamp = loaded_key_timestamp[ibh, icache]
                            is_victim = False
                            if timestamp < least_timestamp_val:
                                is_victim = True
                            elif timestamp - least_timestamp_val < 1:
                                freq = loaded_key_freq[ibh, icache]
                                if freq < least_timestamp_freq:
                                    is_victim = True

                            if is_victim:
                                least_timestamp_val = timestamp
                                least_timestamp_freq = freq
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        loaded_key_list[ibh, least_timestamp_idx] = current_pointer
                        loaded_key_timestamp[ibh, least_timestamp_idx] = ibdst
                        loaded_key_freq[ibh, icache] = 1

            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lfu(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
    decay=False,
):
    LOWER_BOUND_FREQ = -987654321

    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_freq = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        + LOWER_BOUND_FREQ
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                if decay:
                    for icache in range(lru_budget):
                        loaded_key_freq[ibh, icache] = max(
                            LOWER_BOUND_FREQ, loaded_key_freq[ibh, icache] - 1
                        )
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_freq_val = 999999999
                    least_freq_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_freq[ibh, icache] += 1
                            # if in cache, update life
                            in_cache = True
                            break
                        else:
                            if loaded_key_freq[ibh, icache] < least_freq_val:
                                least_freq_val = loaded_key_freq[ibh, icache]
                                least_freq_idx = icache
                    # else, evict victim
                    if not in_cache:
                        loaded_key_list[ibh, least_freq_idx] = current_pointer
                        loaded_key_freq[ibh, least_freq_idx] = 1
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1
            # for icache in range(lru_budget):
            #     loaded_key_freq[ibh, icache] = 0

    return loaded_key_mask


@numba.njit(parallel=True)
def perform_lfu_timestep_aware(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
    window=8,
):
    # within window, perform LFU, when tie, break with LRU
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape

    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_freq = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            window,
            max_lru_budget,
        ),
        dtype=np.float32,
    )
    loaded_key_last_accessed = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(
                math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT
            )

            loaded_key_freq[ibh, ibdst % window] = 0
            # loaded_key_freq[ibh] = np.clip(loaded_key_freq[ibh] * 0.9, 0, 987654321)

            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue

                    in_cache = False
                    least_freq_val = 999999999
                    least_freq_last_accessed = 99999999
                    least_freq_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_freq[ibh, ibdst % window, icache] = 1
                            loaded_key_last_accessed[ibh, icache] = ibdst
                            in_cache = True
                            break
                        else:
                            freq = np.sum(loaded_key_freq[ibh, :, icache])
                            if freq <= least_freq_val:
                                if freq == least_freq_val:
                                    if (
                                        loaded_key_last_accessed[ibh, icache]
                                        < least_freq_last_accessed
                                    ):
                                        least_freq_val = freq
                                        least_freq_last_accessed = (
                                            loaded_key_last_accessed[ibh, icache]
                                        )
                                        least_freq_idx = icache
                                    else:
                                        pass
                                else:
                                    least_freq_val = freq
                                    least_freq_last_accessed = loaded_key_last_accessed[
                                        ibh, icache
                                    ]
                                    least_freq_idx = icache
                    # else, evict victim
                    if not in_cache:
                        loaded_key_list[ibh, least_freq_idx] = current_pointer
                        loaded_key_freq[ibh, :, least_freq_idx] = 0
                        loaded_key_freq[ibh, ibdst % window, least_freq_idx] = 1
                        loaded_key_last_accessed[ibh, icache] = ibdst
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1

    return loaded_key_mask


@numba.njit(parallel=True, fastmath=True)
def perform_lru_heuristic(
    key_access_map,
    key_access_log,
    key_access_count,
    lru_budget_log_scale,
    max_lru_budget,
    KV_HEAD_REPEAT,
    block_size_q=32,
    block_size_k=8,
    sliding_window_size=512,
    perform_heuristic=False,
):
    B, BDST, K = key_access_log.shape

    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_value = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_first_value = (
        np.zeros(
            (
                B // KV_HEAD_REPEAT,
                max_lru_budget,
            ),
            dtype=np.int32,
        )
        - 1
    )
    loaded_key_first_stamp = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.int32,
    )
    loaded_key_importance = np.zeros(
        (
            B // KV_HEAD_REPEAT,
            max_lru_budget,
        ),
        dtype=np.int32,
    )

    for ibh in numba.prange(B // KV_HEAD_REPEAT):  # prange
        for ibdst in range(sliding_window_size // block_size_q, BDST):
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                b = sliding_window_size
                s = lru_budget_log_scale
                lru_budget = round(
                    (math.log2((ibdst * block_size_q + b) / b) * b - b) * s + b
                )

                last_accessed = key_access_log[:, ibdst - 1, :]
                last_accessed_count = key_access_count[:, ibdst - 1]

                # prefetch keys using scaling
                if perform_heuristic:
                    if ibdst > (sliding_window_size // block_size_q):
                        for _icache in range(lru_budget):
                            icache = lru_budget - _icache - 1
                            current_pointer = loaded_key_value[
                                ib // KV_HEAD_REPEAT, icache
                            ]
                            if current_pointer >= 0:
                                first_ibdst = loaded_key_first_stamp[
                                    ib // KV_HEAD_REPEAT, icache
                                ]
                                first_value = loaded_key_first_value[
                                    ib // KV_HEAD_REPEAT, icache
                                ]
                                first_offset = first_value % block_size_k
                                new_position = (
                                    (first_value // block_size_k) / first_ibdst * ibdst
                                )
                                new_position = (
                                    math.ceil(new_position) * block_size_k
                                    + first_offset
                                )

                                if (
                                    new_position
                                    not in loaded_key_value[ib // KV_HEAD_REPEAT]
                                ):
                                    loaded_key_value[ib // KV_HEAD_REPEAT, icache] = (
                                        new_position
                                    )
                                else:
                                    loaded_key_value[ib // KV_HEAD_REPEAT, icache] = (
                                        current_pointer
                                    )
                                    if new_position == current_pointer:
                                        # when keep position
                                        loaded_key_importance[
                                            ib // KV_HEAD_REPEAT, icache
                                        ] -= 0
                                    else:
                                        # when collide
                                        loaded_key_importance[
                                            ib // KV_HEAD_REPEAT, icache
                                        ] -= 1
                # try to add last accessed to LRU cache
                # loaded_key_importance[ib] -= 1 # decay freq if LFU
                for ik in range(min(last_accessed_count[ib], K)):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0:
                        continue
                    in_cache = False
                    least_timestamp_val = 999999999
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        if (
                            loaded_key_value[ib // KV_HEAD_REPEAT, icache]
                            == current_pointer
                        ):
                            loaded_key_importance[ib // KV_HEAD_REPEAT, icache] = ibdst
                            # loaded_key_importance[ib, icache] += 3
                            # if in LRU cache, update life
                            in_cache = True
                        else:
                            if (
                                loaded_key_importance[ib // KV_HEAD_REPEAT, icache]
                                < least_timestamp_val
                            ):
                                least_timestamp_val = loaded_key_importance[
                                    ib // KV_HEAD_REPEAT, icache
                                ]
                                least_timestamp_idx = icache
                    # else, evict victim
                    if perform_heuristic:
                        if not in_cache:
                            new_position = (
                                (current_pointer // block_size_k) / (ibdst - 1) * ibdst
                            )
                            new_position = math.ceil(new_position) * block_size_k + (
                                current_pointer % block_size_k
                            )
                            if (
                                new_position
                                not in loaded_key_value[ib // KV_HEAD_REPEAT, :]
                            ):
                                loaded_key_value[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = new_position
                                loaded_key_first_value[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = current_pointer
                                loaded_key_first_stamp[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = (ibdst - 1)
                                loaded_key_importance[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = ibdst
                            else:
                                for i in range(
                                    len(loaded_key_value[ib // KV_HEAD_REPEAT, :])
                                ):
                                    if (
                                        loaded_key_value[ib // KV_HEAD_REPEAT, i]
                                        == new_position
                                    ):
                                        loaded_key_value[ib // KV_HEAD_REPEAT, i] = (
                                            new_position
                                        )
                                        loaded_key_first_value[
                                            ib // KV_HEAD_REPEAT, i
                                        ] = current_pointer
                                        loaded_key_first_stamp[
                                            ib // KV_HEAD_REPEAT, i
                                        ] = (ibdst - 1)
                                        loaded_key_importance[
                                            ib // KV_HEAD_REPEAT, i
                                        ] = ibdst
                                loaded_key_value[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = current_pointer
                                loaded_key_first_value[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = current_pointer
                                loaded_key_first_stamp[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = (ibdst - 1)
                                loaded_key_importance[
                                    ib // KV_HEAD_REPEAT, least_timestamp_idx
                                ] = ibdst
                    else:
                        if not in_cache:
                            loaded_key_value[
                                ib // KV_HEAD_REPEAT, least_timestamp_idx
                            ] = current_pointer
                            loaded_key_first_value[
                                ib // KV_HEAD_REPEAT, least_timestamp_idx
                            ] = current_pointer
                            loaded_key_first_stamp[
                                ib // KV_HEAD_REPEAT, least_timestamp_idx
                            ] = (ibdst - 1)
                            loaded_key_importance[
                                ib // KV_HEAD_REPEAT, least_timestamp_idx
                            ] = ibdst
                # submit to mask for debug, in realworld, time to fetch
                for icache in range(lru_budget):
                    idx = loaded_key_value[ib // KV_HEAD_REPEAT, icache]
                    if idx >= 0:
                        loaded_key_mask[ib // KV_HEAD_REPEAT, ibdst, idx] = 1

    return loaded_key_mask
