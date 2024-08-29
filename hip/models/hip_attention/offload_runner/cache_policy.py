import gc
import os
if __name__ == '__main__':
    os.environ['HIP_DISABLE_AUTOTUNE'] = '1'

import numba
import numpy as np
from numpy import ndarray as NdArray
from typing import List, Union, Optional, Dict, Tuple
import torch
from torch import Tensor
import math, os, time, tqdm, random
from hip import hip_attention_11, HiPAttentionArgs11
from hip.models.hip_attention.attention1_block_gpu import load_checkouts, to_dense
import matplotlib.pyplot as plt
import cv2

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
def img_reduce(
    img: NdArray,
    rh: int, rw: int
):
    H, W = img.shape
    RH = H // rh
    RW = W // rw
    out = np.zeros((RH, RW))
    for ih in range(RH):
        for iw in range(RW):
            chunk = img[ih * rh: ih * rh + rh, iw * rw: iw * rw + rw]
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
                tsrc = ((ibdst + 1) * block_size_q - sliding_window_size) * (ibk / (_mask_k // block_size_k))
                tsrc = round(tsrc)
                if tsrc >= 0:
                    tsrc = tsrc - (tsrc % block_size_k)
                    for ioffset in range(block_stride_k - 1, block_size_k, block_stride_k):
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

@numba.njit(parallel=True)
def perform_lru_hot_prefetch(
    key_access_map,
    key_access_log,
    key_access_count,
    block_size_q,
    sliding_window_size,
    lru_budget_log_scale,
    KV_HEAD_REPEAT,
    prefetch_step,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape
    
    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)
    
    prefetch_ratio = 0.4
    max_prefetch_budget = math.ceil(max_lru_budget * prefetch_ratio)
    max_lru_budget = math.ceil(max_lru_budget * (1-prefetch_ratio))
    
    loaded_key_mask = np.zeros_like(key_access_map)
    
    # LRU-temperature cache
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_timestamp = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    all_key_temperature = np.zeros((B // KV_HEAD_REPEAT, TSRC,), dtype=np.float32)
    momentum = 0.7
    decay_momentum = 0.95
    
    # LFU or LRU prefetch cache
    #   candidate priority is frequency
    #   prefetch key priority is frequencey
    prefetch_candidate = np.zeros((B // KV_HEAD_REPEAT, max_prefetch_budget,), dtype=np.int32) - 1
    prefetch_candidate_try = np.zeros((B // KV_HEAD_REPEAT, max_prefetch_budget, ), dtype=np.int32) + 1
    prefetch_candidate_priority = np.zeros((B // KV_HEAD_REPEAT, max_prefetch_budget,), dtype=np.float32) - 1
    prefetch_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    prefetch_key_priority = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            prefetch_budget = math.floor(lru_budget * prefetch_ratio)
            lru_budget = math.floor(lru_budget * (1-prefetch_ratio))
            
            all_key_temperature[ibh] *= decay_momentum
            
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
                    in_cache = False
                    in_candidate = False
                    least_priority_val = 999999999
                    victim_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            all_key_temperature[ibh, current_pointer] = momentum * all_key_temperature[ibh, current_pointer] + (1 - momentum)
                            in_cache = True
                            break
                        else:
                            temperature = all_key_temperature[ibh, cached_pointer]
                            if temperature < least_priority_val:
                                least_priority_val = temperature
                                victim_idx = icache
                    
                    if not in_cache:
                        for ifetch in range(prefetch_budget):
                            prefetch_pointer = prefetch_key_list[ibh, ifetch]
                            if prefetch_pointer == current_pointer:
                                prefetch_key_priority[ibh, ifetch] = ibdst
                                in_cache = True
                                break
                            candidate_pointer = prefetch_candidate[ibh, ifetch]
                            if candidate_pointer == current_pointer:
                                # push candidate prefetch
                                prefetch_candidate_try[ibh, ifetch] -= 1
                                prefetch_candidate_priority[ibh, ifetch] = ibdst
                                in_candidate = True
                                break
                    
                    # need to push cache
                    if not in_cache:
                        # update temperature
                        all_key_temperature[ibh, current_pointer] = momentum * all_key_temperature[ibh, current_pointer] + (1 - momentum)
                        # add to prefetch candidate
                    
                    if (not in_cache) and (not in_candidate):
                        least_candidate_priority_val = 999999999
                        candidate_victim_idx = -1
                        for ifetch in range(prefetch_budget):
                            priority = prefetch_candidate_priority[ibh, ifetch]
                            if priority < least_candidate_priority_val:
                                least_candidate_priority_val = priority
                                candidate_victim_idx = ifetch
                        prefetch_candidate[ibh, candidate_victim_idx] = current_pointer
                        prefetch_candidate_try[ibh, candidate_victim_idx] = 3
                        prefetch_candidate_priority[ibh, candidate_victim_idx] = ibdst
                    
                    # if victim has cooler, then push to cache
                    if not in_cache and all_key_temperature[ibh, current_pointer] >= all_key_temperature[ibh, loaded_key_list[ibh, victim_idx]]:
                        loaded_key_list[ibh, victim_idx] = current_pointer
                        loaded_key_timestamp[ibh, victim_idx] = ibdst
            
            # before start step, perform prefetch
            for ifetch in range(prefetch_budget):
                if prefetch_key_list[ibh, ifetch] >= 0:
                    prefetch_key_list[ibh, ifetch] += prefetch_step
                    if prefetch_key_priority[ibh, ifetch] < ibdst - 16:
                        prefetch_key_priority[ibh, ifetch] = -1
                        prefetch_key_list[ibh, ifetch] = -1
                if prefetch_candidate[ibh, ifetch] >= 0:
                    prefetch_candidate[ibh, ifetch] += prefetch_step
            
            for ifetch in range(prefetch_budget):
                if prefetch_candidate_try[ibh, ifetch] <= 0:
                    candidate_pointer = prefetch_candidate[ibh, ifetch]
                    candidate_priority = prefetch_candidate_priority[ibh, ifetch]
                    
                    prefetch_candidate_try[ibh, ifetch] = 1
                    prefetch_candidate[ibh, ifetch] = -1
                    prefetch_candidate_priority[ibh, ifetch] = -1
                    
                    victim_idx = np.argmin(prefetch_key_priority[ibh, :prefetch_budget])
                    prefetch_key_priority[ibh, victim_idx] = candidate_priority
                    prefetch_key_list[ibh, victim_idx] = candidate_pointer
            
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ibh, icache]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1
            for ifetch in range(prefetch_budget):
                idx = prefetch_key_list[ibh, ifetch]
                if idx >= 0:
                    loaded_key_mask[ibh, ibdst, idx] = 1
    
    return loaded_key_mask

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
    minimum_cost = 1000,
):
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape
    
    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)
    
    # for output
    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_probs_map = np.zeros_like(key_access_map, dtype=np.float32)
    
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_scores = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.float32) - 987654321
    loaded_key_h = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.float32)
    loaded_key_h_init = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.float32)
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        last_min_h = 0
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            probs = numba_softmax(loaded_key_scores[ibh], temperature)
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
                    in_cache = False
                    min_h_value = 99999999
                    victim_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_h[ibh, icache] = last_min_h + probs[icache] + minimum_cost
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
                        loaded_key_scores[ibh, victim_idx] = key_access_score[ib, ibdst, ik]
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_score = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.float32) - 987654321
    
    momemtum = 0.9
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            
            loaded_key_score[ibh] *= momemtum
            
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
                    in_cache = False
                    least_timestamp_val = 999999999.0
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_score[ibh, icache] = momemtum * loaded_key_score[ibh, icache] +\
                                key_access_score[ib, ibdst, ik] * (1 - momemtum)
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
                        loaded_key_score[ibh, least_timestamp_idx] = key_access_score[ib, ibdst, ik]
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_score = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.float32) - 987654321
    all_key_cost = np.zeros((B // KV_HEAD_REPEAT, TSRC,), dtype=np.float32)
    
    momentum = 0.7
    decay_momentum = 0.95
    
    score_momemtum = 0.8
    score_decay = 0.95
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            
            all_key_cost[ibh] *= decay_momentum
            loaded_key_score[ibh] *= score_decay
            
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
                    in_cache = False
                    least_hot_val = 0
                    victim_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_score[ibh, icache] = score_momemtum * loaded_key_score[ibh, icache] +\
                                key_access_score[ib, ibdst, ik] * (1 - score_momemtum)
                            all_key_cost[ibh, current_pointer] = momentum * all_key_cost[ibh, current_pointer] + (1 - momentum)
                            in_cache = True
                            break
                        else:
                            is_victim = False
                            score = loaded_key_score[ibh, icache]
                            cost = all_key_cost[ibh, cached_pointer] * 0.01 + score * 0.99
                            
                            hot = (1 - cost)
                            if hot > least_hot_val:
                                is_victim = True
                            
                            if is_victim:
                                least_hot_val = hot
                                victim_idx = icache
                    # else, evict victim
                    if not in_cache:
                        all_key_cost[ibh, current_pointer] = momentum * all_key_cost[ibh, current_pointer] + (1 - momentum)
                    if  (not in_cache) and\
                        (
                            all_key_cost[ibh, current_pointer] >=\
                            all_key_cost[ibh, loaded_key_list[ibh, victim_idx]]
                        ):
                        loaded_key_list[ibh, victim_idx] = current_pointer
                        loaded_key_score[ibh, victim_idx] = key_access_score[ib, ibdst, ik]
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_timestamp = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    all_key_cost = np.zeros((B // KV_HEAD_REPEAT, TSRC,), dtype=np.float32)
    momentum = 0.7
    decay_momentum = 0.95
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            
            all_key_cost[ibh] *= decay_momentum
            
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
                    in_cache = False
                    least_timestamp_val = 0
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        cached_pointer = loaded_key_list[ibh, icache]
                        if cached_pointer == current_pointer:
                            loaded_key_timestamp[ibh, icache] = ibdst
                            all_key_cost[ibh, current_pointer] = momentum * all_key_cost[ibh, current_pointer] + (1 - momentum)
                            in_cache = True
                            break
                        else:
                            # timestamp = loaded_key_timestamp[ibh, icache]
                            # dist = ibdst - timestamp
                            cost = all_key_cost[ibh, cached_pointer]
                            score = (1 - cost)
                            if score > least_timestamp_val:
                                least_timestamp_val = score
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        all_key_cost[ibh, current_pointer] = momentum * all_key_cost[ibh, current_pointer] + (1 - momentum)
                    if not in_cache and all_key_cost[ibh, current_pointer] >= all_key_cost[ibh, loaded_key_list[ibh, least_timestamp_idx]]:
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_timestamp = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_timestamp = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget, lru_k), dtype=np.int32)
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
                    in_cache = False
                    least_timestamp_val = 999999999
                    least_timestamp_idx = -1
                    for icache in range(lru_budget):
                        if loaded_key_list[ibh, icache] == current_pointer:
                            loaded_key_timestamp[ibh, icache, ibdst % lru_k] = ibdst
                            in_cache = True
                            break
                        else:
                            timestamp = loaded_key_timestamp[ibh, icache, (ibdst - 1) % lru_k]
                            if timestamp < least_timestamp_val:
                                least_timestamp_val = timestamp
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        loaded_key_list[ibh, least_timestamp_idx] = current_pointer
                        loaded_key_timestamp[ibh, least_timestamp_idx, :] = 999999999
                        loaded_key_timestamp[ibh, least_timestamp_idx, ibdst % lru_k] = ibdst
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_timestamp = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    all_key_last_evicted = np.zeros((B // KV_HEAD_REPEAT, TSRC), dtype=np.int32)
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            # if ibh == 0:
            #     print(all_key_last_evicted[ibh, 1024:1034])
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
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
                            last_evict = all_key_last_evicted[ibh, loaded_key_list[ibh, icache]]
                            is_victim = False
                            if  (timestamp < least_timestamp_val):
                                is_victim = True
                            
                            if  (timestamp == least_timestamp_val) and\
                                (last_evict < least_timestamp_last_evict):
                                is_victim = True
                            
                            if is_victim:
                                least_timestamp_last_evict = last_evict
                                least_timestamp_val = timestamp
                                least_timestamp_idx = icache
                    # else, evict victim
                    if not in_cache:
                        all_key_last_evicted[ibh, loaded_key_list[ibh, least_timestamp_idx]] = ibdst
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_timestamp = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_freq = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32)
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            
            loaded_key_freq[ibh] -= 1
            
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
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
    decay = False,
):
    LOWER_BOUND_FREQ = -987654321
    
    B, BDST, K = key_access_log.shape
    _, _, TSRC = key_access_map.shape
    
    b = sliding_window_size
    s = lru_budget_log_scale
    max_lru_budget = math.ceil(math.log2(TSRC / b + 1) * b * s * KV_HEAD_REPEAT)
    
    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_freq = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) + LOWER_BOUND_FREQ
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                if decay:
                    for icache in range(lru_budget):
                        loaded_key_freq[ibh, icache] = max(
                            LOWER_BOUND_FREQ,
                            loaded_key_freq[ibh, icache] - 1
                        )
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
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
    loaded_key_list = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_freq = np.zeros((B // KV_HEAD_REPEAT, window, max_lru_budget,), dtype=np.float32)
    loaded_key_last_accessed = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT):
        for ibdst in range(1, BDST):
            b = sliding_window_size
            s = lru_budget_log_scale
            lru_budget = round(math.log2(ibdst * block_size_q / b + 1) * b * s * KV_HEAD_REPEAT)
            
            loaded_key_freq[ibh, ibdst % window] = 0
            # loaded_key_freq[ibh] = np.clip(loaded_key_freq[ibh] * 0.9, 0, 987654321)
            
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                # try to add last accessed to LRU cache
                for ik in range(min(K, last_accessed_count[ib])):
                    current_pointer = last_accessed[ib, ik]
                    if current_pointer < 0: continue
                    
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
                                    if loaded_key_last_accessed[ibh, icache] < least_freq_last_accessed:
                                        least_freq_val = freq
                                        least_freq_last_accessed = loaded_key_last_accessed[ibh, icache]
                                        least_freq_idx = icache
                                    else:
                                        pass
                                else:
                                    least_freq_val = freq
                                    least_freq_last_accessed = loaded_key_last_accessed[ibh, icache]
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
    block_size_q = 32,
    block_size_k = 8,
    sliding_window_size = 512,
    perform_heuristic = False,
):
    B, BDST, K = key_access_log.shape
    
    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_value = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_first_value = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32) - 1
    loaded_key_first_stamp = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32)
    loaded_key_importance = np.zeros((B // KV_HEAD_REPEAT, max_lru_budget,), dtype=np.int32)
    
    for ibh in numba.prange(B // KV_HEAD_REPEAT): #prange
        for ibdst in range(sliding_window_size // block_size_q, BDST):
            for ikv in range(KV_HEAD_REPEAT):
                ib = ibh * KV_HEAD_REPEAT + ikv
                b = sliding_window_size
                s = lru_budget_log_scale
                lru_budget = round((math.log2((ibdst * block_size_q + b) / b) * b - b) * s + b)
                
                last_accessed = key_access_log[:, ibdst-1, :]
                last_accessed_count = key_access_count[:, ibdst-1]
                
                # prefetch keys using scaling
                if perform_heuristic:
                    if ibdst > (sliding_window_size // block_size_q):
                        for _icache in range(lru_budget):
                            icache = lru_budget - _icache - 1
                            current_pointer = loaded_key_value[ib // KV_HEAD_REPEAT, icache]
                            if current_pointer >= 0:
                                first_ibdst = loaded_key_first_stamp[ib // KV_HEAD_REPEAT, icache]
                                first_value = loaded_key_first_value[ib // KV_HEAD_REPEAT, icache]
                                first_offset = first_value % block_size_k
                                new_position = (first_value // block_size_k) / first_ibdst * ibdst
                                new_position = math.ceil(new_position) * block_size_k + first_offset
                                
                                if new_position not in loaded_key_value[ib // KV_HEAD_REPEAT]:
                                    loaded_key_value[ib // KV_HEAD_REPEAT, icache] = new_position
                                else:
                                    loaded_key_value[ib // KV_HEAD_REPEAT, icache] = current_pointer
                                    if new_position == current_pointer:
                                        # when keep position
                                        loaded_key_importance[ib // KV_HEAD_REPEAT, icache] -= 0
                                    else:
                                        # when collide
                                        loaded_key_importance[ib // KV_HEAD_REPEAT, icache] -= 1
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
                        if loaded_key_value[ib // KV_HEAD_REPEAT, icache] == current_pointer:
                            loaded_key_importance[ib // KV_HEAD_REPEAT, icache] = ibdst
                            # loaded_key_importance[ib, icache] += 3
                            # if in LRU cache, update life
                            in_cache = True
                        else:
                            if loaded_key_importance[ib // KV_HEAD_REPEAT, icache] < least_timestamp_val:
                                least_timestamp_val = loaded_key_importance[ib // KV_HEAD_REPEAT, icache]
                                least_timestamp_idx = icache
                    # else, evict victim
                    if perform_heuristic:
                        if not in_cache:
                            new_position = (current_pointer // block_size_k) / (ibdst - 1) * ibdst
                            new_position = math.ceil(new_position) * block_size_k + (current_pointer % block_size_k)
                            if new_position not in loaded_key_value[ib // KV_HEAD_REPEAT, :]:
                                loaded_key_value[ib // KV_HEAD_REPEAT, least_timestamp_idx] = new_position
                                loaded_key_first_value[ib // KV_HEAD_REPEAT, least_timestamp_idx] = current_pointer
                                loaded_key_first_stamp[ib // KV_HEAD_REPEAT, least_timestamp_idx] = ibdst - 1
                                loaded_key_importance[ib // KV_HEAD_REPEAT, least_timestamp_idx] = ibdst
                            else:
                                for i in range(len(loaded_key_value[ib // KV_HEAD_REPEAT, :])):
                                    if loaded_key_value[ib // KV_HEAD_REPEAT, i] == new_position:
                                        loaded_key_value[ib // KV_HEAD_REPEAT, i] = new_position
                                        loaded_key_first_value[ib // KV_HEAD_REPEAT, i] = current_pointer
                                        loaded_key_first_stamp[ib // KV_HEAD_REPEAT, i] = ibdst - 1
                                        loaded_key_importance[ib // KV_HEAD_REPEAT, i] = ibdst
                                loaded_key_value[ib // KV_HEAD_REPEAT, least_timestamp_idx] = current_pointer
                                loaded_key_first_value[ib // KV_HEAD_REPEAT, least_timestamp_idx] = current_pointer
                                loaded_key_first_stamp[ib // KV_HEAD_REPEAT, least_timestamp_idx] = ibdst - 1
                                loaded_key_importance[ib // KV_HEAD_REPEAT, least_timestamp_idx] = ibdst
                    else:
                        if not in_cache:
                            loaded_key_value[ib // KV_HEAD_REPEAT, least_timestamp_idx] = current_pointer
                            loaded_key_first_value[ib // KV_HEAD_REPEAT, least_timestamp_idx] = current_pointer
                            loaded_key_first_stamp[ib // KV_HEAD_REPEAT, least_timestamp_idx] = ibdst - 1
                            loaded_key_importance[ib // KV_HEAD_REPEAT, least_timestamp_idx] = ibdst
                # submit to mask for debug, in realworld, time to fetch
                for icache in range(lru_budget):
                    idx = loaded_key_value[ib // KV_HEAD_REPEAT, icache]
                    if idx >= 0:
                        loaded_key_mask[ib // KV_HEAD_REPEAT, ibdst, idx] = 1
        
    return loaded_key_mask

def main_exp():
    seq_len = 1024 * 128
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=seq_len, 
        return_cos_sin=True, 
        dtype=torch.bfloat16
    )
    
    H = q.shape[0]
    H_KV = k.shape[0]
    
    def reshape(x, HEAD):
        N, T, H = x.shape
        x = x.contiguous()\
            .view(N // HEAD, HEAD, T, H)\
            .permute(0, 2, 1, 3)\
            .contiguous()
        assert x.shape == (N // HEAD, T, HEAD, H)
        assert x.is_contiguous()
        return x

    q = reshape(q, H)
    k = reshape(k, H_KV)
    v = reshape(v, H_KV)
    out = reshape(out, H)
    
    args = HiPAttentionArgs11(
        mask_k=128,
        block_size_k=2,
        block_stride_k=1,
        block_size_q=32,
        block_stride_q=2,
        sliding_window_size=32,
        sink_token_size=4,
        output_key_access_log=False,
        output_block_access_log=True,
    )
    _, metadata = hip_attention_11(
        q, k, v,
        args=args
    )
    
    block_access_log = metadata.block_access_log.cpu()
    block_access_score = metadata.block_access_score.float().cpu()
    block_access_count = metadata.block_access_count.cpu()
    args.block_size_q = args.block_size_q // args.block_size_k
    
    TDST = q.shape[1] // args.block_size_k
    TSRC = k.shape[1] // args.block_size_k
    
    KV_HEAD_REPEAT = H // H_KV
    # KV_HEAD_REPEAT = 1
    # KV_HEAD_REPEAT = H
    
    del metadata, q, k, v, out, cos, sin
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    block_access_map = access_log_to_dense(
        block_access_log.cpu().numpy(),
        block_access_count.cpu().numpy(),
        TSRC,
        KV_HEAD_REPEAT,
    )
    # block_access_score_map = access_score_log_to_dense(
    #     block_access_log.cpu().numpy(),
    #     block_access_count.cpu().numpy(),
    #     TSRC,
    #     KV_HEAD_REPEAT,
    #     block_access_score.cpu().numpy()
    # )
    block_access_mask = np.clip(block_access_map, 0, 1)
    
    def render_recalls():
        recalls = {}
        B, BDST, TSRC = block_access_mask.shape
        mask = block_access_mask.reshape(B // KV_HEAD_REPEAT, KV_HEAD_REPEAT, BDST, TSRC)
        mask = torch.tensor(np.clip(np.sum(mask, axis=1), 0, 1), device=0, dtype=torch.int32)
        for i in tqdm.tqdm(range(args.mask_k // args.block_size_q + 1, BDST), dynamic_ncols=True, desc='recalls', leave=False):
            for j in range(i + 1, BDST):
                if (random.random() > (20 / (BDST - i))) and (j - i) > 16: continue
                pred = mask[:, i, :j * args.block_size_q]
                target = mask[:, j, :j * args.block_size_q]
                match = ((pred == target).to(torch.int32) * target).to(torch.int32)
                num_match = torch.sum(match)
                num_target = torch.sum(target)
                points = recalls.get(j - i, [])
                points.append((num_match / (num_target + 1e-20) * 100).to('cpu', non_blocking=True))
                recalls[j-i] = points
        del mask
        data = list(map(lambda x: list(map(lambda y: y.item(), x[1])), sorted(recalls.items(), key=lambda z: z[0])))
        means = np.array([np.mean(d) for d in data])
        stds = np.array([np.std(d) for d in data])
        xs = np.array(list(recalls.keys()))
        xs.sort()
        # print(xs, means, stds, xs.shape, means.shape, stds.shape)
        plt.clf()
        plt.fill_between(xs, means-stds, means+stds, alpha=0.3, facecolor='green')
        plt.plot(xs, means, color='green')
        plt.xlabel(f'Decode Step Distance')
        plt.ylabel('Key Access Pattern Recall (%)')
        plt.xlim(1, 128)
        plt.ylim(50, 100)
        plt.xscale('log', base=2)
        plt.grid()
        path = 'dummy_access_recalls.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'saved {path}')
    # render_recalls()
    
    def render_access_map_fullres():
        # mat = cv2.applyColorMap(key_access_map[0], cv2.COLORMAP_JET)
        for i in range(block_access_map.shape[0]):
            path = f'dummy_access_map_fullres_{i}.png'
            cv2.imwrite(path, (block_access_map[i] * 255).astype(np.uint8))
            print(f'saved {path}')
    # render_access_map_fullres()
    
    def render_access_score_map_fullres():
        t_min = -10
        t_max = 10
        t = (
            (block_access_score_map - t_min) /\
            (t_max - t_min)
        )
        t = np.clip(t, 0, 1)
        for i in range(t.shape[0]):
            path = f'dummy_access_score_map_fullres_{i}.png'
            cv2.imwrite(path, (t[i] * 255).astype(np.uint8))
            print(f'saved {path}')
    # render_access_score_map_fullres()
    
    # plot key access map
    def render_access_map():
        img = block_access_map[0]
        img = img_reduce(img, 1, args.block_size_q)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.colorbar()
        plt.title(f'avg access count (T={TSRC}, bq={args.block_size_q}, bk={args.block_size_k})')
        plt.tight_layout()
        plt.savefig('dummy_access.png', dpi=96, bbox_inches='tight')
        print('saved dummy_access.png')
    # render_access_map()
    
    def plot_stats(
        name,
        loaded_key_mask,
    ):
        print('-' * 20, name, '-' * 20)
        # calc fetchs
        fetched_key_mask = loaded_key_mask[:, 1:, :] - loaded_key_mask[:, :-1, :]
        fetched_key_mask = np.clip(fetched_key_mask, 0, 1)
        
        # calc misses
        missed_key_mask = np.clip(block_access_mask[:, 1:, :] - loaded_key_mask[:, 1:, :], 0, 1)
        
        cv2.imwrite(f'dummy_{name}_loaded.png', loaded_key_mask[0, 1:, :] * 255)
        cv2.imwrite(f'dummy_{name}_fetched.png', fetched_key_mask[0] * 255)
        cv2.imwrite(f'dummy_{name}_missed.png', missed_key_mask[0] * 255)
        cv2.imwrite(f'dummy_{name}_accessed.png', block_access_mask[0, 1:, :] * 255)
        
        # 0 (black): not loaded
        # 1 (white): loaded but not used
        # 2 (green): cache hit
        # 3 (red): missed
        load_map = cache_map = loaded_key_mask[0, 1:, :]
        access_map = block_access_map[0, 1:, :]
        cache_map = np.where(load_map * access_map, 2, cache_map)
        cache_map = np.where((1 - load_map) * access_map, 3, cache_map)
        colormap = np.array([
            [0, 0, 0],
            [255,255,255],
            [0,255,0],
            [0,0,255],
        ], dtype=np.int64)
        H, W = cache_map.shape
        cache_image = np.take(colormap.reshape(1, 1, 4, 3), cache_map.reshape(H, W, 1, 1), axis=-2)
        cache_image = np.reshape(cache_image, (H, W, 3))
        path = f'dummy_{name}_combined.png'
        cv2.imwrite(path, cache_image)
        print('saved', path)
        
        accessed_key_counts = block_access_mask[:, 1:, :].sum(axis=-1)
        loaded_key_counts = loaded_key_mask[:, 1:, :].sum(axis=-1)
        fetched_key_counts = fetched_key_mask.sum(axis=-1)
        missed_key_counts = missed_key_mask.sum(axis=-1)
        xs = np.arange(args.block_size_q, TDST, args.block_size_q)
        
        plt.figure(figsize=(8, 12))
        plt.plot(xs, loaded_key_counts.T.mean(axis=-1), color='gray')
        plt.plot(xs, fetched_key_counts.T.mean(axis=-1), color='green')
        plt.plot(xs, missed_key_counts.T.mean(axis=-1), color='red')
        plt.plot(xs, fetched_key_counts.T.mean(axis=-1) + missed_key_counts.T.mean(axis=-1), color='orange')
        plt.plot(xs, accessed_key_counts.T.mean(axis=-1), color='blue')
        plt.axhline(TSRC / args.block_stride_k, color='darkgray')
        plt.grid()
        filename = f'dummy_{name}_stats'
        path = f'{filename}.png'
        plt.savefig(path, dpi=96)
        print(f'saved {path}')
        
        accessed_count = accessed_key_counts.T[-1].mean()
        missed_count = missed_key_counts.T[-1].mean()
        print(f'cache hit ratio: {(1 - missed_count / accessed_count) * 100:.4f}')
        
        est_cache_budget = loaded_key_counts.T[-1].mean()
        oracle_cache_budget = accessed_key_counts.T[-1].mean()
        print(f'estimated cache size: {est_cache_budget}, oracle cache size: {oracle_cache_budget}, relative size: {est_cache_budget/oracle_cache_budget:.2f}, sparsity: {est_cache_budget/block_access_map.shape[-1]*100:.2f} %')
        
        fetched_count = fetched_key_counts.T[-1].mean()
        n_layer = 32
        n_kv_head = 8
        n_kv_hid = 128 * args.block_size_k
        fetched_mb = fetched_count * n_layer * n_kv_head * n_kv_hid / (1024 * 1024)
        print(f'fetched tokens: {fetched_count:.1f}, {fetched_mb:.4f} MB, took {fetched_mb / 64:.2f} ms (bsz=1) / {fetched_mb / 64 * 32:.2f} ms (bsz=32) in PCIe 4.0')
        
        missed_count = missed_key_counts.T[-1].mean()
        n_layer = 32
        n_kv_head = 8
        n_kv_hid = 128 * args.block_size_k
        missed_mb = missed_count * n_layer * n_kv_head * n_kv_hid / (1024 * 1024)
        print(f'missed tokens: {missed_count:.1f}, {missed_mb:.4f} MB, took {missed_mb / 64:.2f} ms (bsz=1) / {missed_mb / 64 * 32:.2f} ms (bsz=32) in PCIe 4.0')
    
    def render_lru_hot_prefetch(lru_budget_log_scale=2):
        loaded_key_mask = perform_lru_hot_prefetch(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
            args.block_size_q,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_hot_prefetch_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_hot(1)
    render_lru_hot_prefetch(1.5)
    render_lru_hot_prefetch(2.0)
    render_lru_hot_prefetch(3.0)
    render_lru_hot_prefetch(4.0)
    
    def render_lru_hot(lru_budget_log_scale=2):
        loaded_key_mask = perform_lru_hot(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_hot_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_hot(1)
    render_lru_hot(1.5)
    render_lru_hot(2.0)
    render_lru_hot(3.0)
    render_lru_hot(4.0)
    
    def render_lru(lru_budget_log_scale=2):
        loaded_key_mask = perform_lru(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru(1)
    render_lru(1.5)
    render_lru(2.0)
    render_lru(3.0)
    render_lru(4.0)
    
    def render_gd_score(lru_budget_log_scale=2, temperature=10):
        scores = block_access_score.cpu().numpy()
        probs = torch.softmax((block_access_score / temperature), dim=-1).cpu().numpy()
        loaded_key_mask, loaded_key_probs_map = perform_gd_score(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            scores,
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
            temperature,
        )
        
        
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_gd_score_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_hot(1)
    render_gd_score(1.5)
    render_gd_score(2.0)
    render_gd_score(3.0)
    render_gd_score(4.0)
    
    def render_lru_hot_score(lru_budget_log_scale=2):
        loaded_key_mask = perform_lru_hot_score(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            torch.softmax(block_access_score, dim=-1).cpu().numpy(),
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_hot_score_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_hot(1)
    render_lru_hot_score(1.5)
    render_lru_hot_score(2.0)
    render_lru_hot_score(3.0)
    render_lru_hot_score(4.0)
    
    def render_lru_score(lru_budget_log_scale=2):
        loaded_key_mask = perform_lru_score(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_score.cpu().numpy(),
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_score_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru(1)
    render_lru_score(1.5)
    render_lru_score(2.0)
    render_lru_score(3.0)
    render_lru_score(4.0)
    
    def render_lru_k(lru_budget_log_scale=2, k=4):
        loaded_key_mask = perform_lru_k(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
            k,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_{k}_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_k(1, 2)
    render_lru_k(1.5, 2)
    render_lru_k(2.0, 2)
    render_lru_k(3.0, 2)
    render_lru_k(4.0, 2)
    # render_lru_k(1, 3)
    render_lru_k(1.5, 3)
    render_lru_k(2.0, 3)
    render_lru_k(3.0, 3)
    render_lru_k(4.0, 3)
    # render_lru_k(1)
    render_lru_k(1.5)
    render_lru_k(2.0)
    render_lru_k(3.0)
    render_lru_k(4.0)
    
    def render_lru_tie_break_lre(lru_budget_log_scale=2):
        loaded_key_mask = perform_lru_tie_break_lre(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_tie_break_lre_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_tie_break_lre(1)
    render_lru_tie_break_lre(1.5)
    render_lru_tie_break_lre(2.0)
    render_lru_tie_break_lre(3.0)
    render_lru_tie_break_lre(4.0)
    
    def render_lru_tie_break_lfu(lru_budget_log_scale=2):
        loaded_key_mask = perform_lru_tie_break_lfu(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lru_tie_break_lfu_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_tie_break_lfu(1)
    render_lru_tie_break_lfu(1.5)
    render_lru_tie_break_lfu(2.0)
    
    def render_lfu_timestep_aware(lru_budget_log_scale=2):
        loaded_key_map = perform_lfu_timestep_aware(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_map, 0, 1)
        plot_stats(f'lfu_timestep_{lru_budget_log_scale}', loaded_key_mask)
    # render_lfu_timestep_aware(1)
    render_lfu_timestep_aware(1.5)
    render_lfu_timestep_aware(2)
    
    def render_lfu_decay(lru_budget_log_scale=2):
        loaded_key_mask = perform_lfu(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
            True
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        plot_stats(f'lfu_decay_{lru_budget_log_scale}', loaded_key_mask)
    # render_lfu_decay(1)
    render_lfu_decay(1.5)
    render_lfu_decay(2.0)
    
    def render_lfu(lru_budget_log_scale=2):
        loaded_key_map = perform_lfu(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(), 
            args.block_size_q,
            args.sliding_window_size,
            lru_budget_log_scale,
            KV_HEAD_REPEAT,
        )
        loaded_key_mask = np.clip(loaded_key_map, 0, 1)
        plot_stats(f'lfu_{lru_budget_log_scale}', loaded_key_mask)
    # render_lfu(1)
    render_lfu(1.5)
    render_lfu(2)
    
    def render_lru_heuristic(lru_budget_log_scale=4):
        B, BDST, K = block_access_log.shape
        b = args.sliding_window_size
        s = lru_budget_log_scale
        print('performing heuristic', lru_budget_log_scale, flush=True)
        loaded_key_mask = perform_lru_heuristic(
            block_access_map, 
            block_access_log.cpu().numpy(), 
            block_access_count.cpu().numpy(),
            lru_budget_log_scale, 
            round((math.log2((BDST * args.block_size_q + b) / b) * b - b) * s + b), 
            KV_HEAD_REPEAT,
            args.block_size_q,
            args.block_size_k,
            args.sliding_window_size,
        )
        loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
        print('plot stats', flush=True)
        plot_stats(f'lru_heuristic_{lru_budget_log_scale}', loaded_key_mask)
    # render_lru_heuristic(1)
    # render_lru_heuristic(2)
    # render_lru_heuristic(4)
    # render_lru_heuristic(8)
    # render_lru_heuristic(16)
    
if __name__ == '__main__':
    main_exp()