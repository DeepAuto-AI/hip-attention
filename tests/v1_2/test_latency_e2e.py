import math
import os
from typing import Any, Tuple
import json

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_kvpacked_func
from flash_attn_interface import flash_attn_func as flash_attn3_func
from hip_attn.v1_2 import HiPAttentionConfig

from hip_attn.v1_2.attention_extend import dual_stage_quadratic_hip_attention

# from hip_research.utils.load_checkouts import load_checkouts
import sglang as sgl

import transformers
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, ScanStage
from hip_attn.v1_2.delta.apply_delta import apply_delta
from hip_attn.v1_2.query_sparse_attention import query_sparse_attention


MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

def latency(fn: Any, n_sample: int = 10) -> float:
    elapsed = []
    for i in range(n_sample):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        _ = fn()
        end.record()
        end.synchronize()
        if i > 3:
            elapsed.append(start.elapsed_time(end))
    return float(sum(elapsed) / len(elapsed))


def get_llama31_text(length):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    tokens = torch.arange(131072)

    repeats = math.ceil(length / 131072)
    tokens = tokens.repeat(repeats)[:length]
    string = tokenizer.decode(tokens)
    return string

log_level = "INFO"
tp_size = 2
ep_size = 2

def test_latency_delta():
    hip_config_path="/data/jeff/delta/hip-attention/configs/mixed_landmark_0814_no_extend_qsa.json"
    hip_attention_config_override_json='{"__seq_thresh_fa3": 0}'

    os.environ["BSA_BLOCK_K"] = "64"
    os.environ["BSA_K"] = "64"
    os.environ["BSA_EXACT_K"] = "64"
    os.environ["BSA_WINNER_TREE"] = "False"
    os.environ["HIP_DEBUG_DELTA_QSA"] = "1"

    hip_attention_config = HiPAttentionConfig(
        json_or_path=hip_config_path,
        json_override=hip_attention_config_override_json,
    )

    original_pos = 131072
    extended_pos = 2**19 + 16384
    extend_factor = extended_pos / original_pos

    model = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="auto",
        tp_size=tp_size,
        ep_size=ep_size,
        enable_hip_attention=True,
        attention_backend="hip_attention",
        hip_attention_config=hip_attention_config,
        disable_radix_cache=True,
        context_length=extended_pos,
        log_level=log_level,
        cuda_graph_max_bs=1,
        cuda_graph_bs=[1],
        show_time_cost=True,
        json_model_override_args=(
            '{"rope_scaling":{"rope_type":"yarn","factor":' + f'{extend_factor},'
            '"original_max_position_embeddings":' + f'{original_pos}' + '}}'
        ),
    )

    out = []
    for i in range(15, 20):
        prompt = get_llama31_text(2**i)
        lat = latency(lambda: model.generate(prompt, {"max_new_tokens": 1}))
        out += [(2**i, lat)]
        print(f"delta latency: {lat} for {2**i}")

    with open(f"/data/jeff/delta/latency/delta.jsonl", "w") as f:
        json.dump(out, f)


def test_latency_hip():
    # for running plain hip
    hip_config_path="/data/jeff/delta/hip-attention/configs/rebuttal/llama31_noextend_dense_decode.json"
    hip_attention_config_override_json='{"using_extend": false, "__delta_attention_args": "window_0-diff_2-w_256-dense_decode", "__seq_thresh_fa3": 0}'

    os.environ["HIP_DEBUG_DELTA_QSA"] = "0"

    hip_attention_config = HiPAttentionConfig(
        json_or_path=hip_config_path,
        json_override=hip_attention_config_override_json,
    )

    original_pos = 131072
    extended_pos = 2**19 + 16384
    extend_factor = extended_pos / original_pos

    model = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="auto",
        tp_size=tp_size,
        ep_size=ep_size,
        enable_hip_attention=True,
        attention_backend="hip_attention",
        hip_attention_config=hip_attention_config,
        disable_radix_cache=True,
        context_length=extended_pos,
        log_level=log_level,
        cuda_graph_max_bs=1,
        cuda_graph_bs=[1],
        show_time_cost=True,
        json_model_override_args=(
            '{"rope_scaling":{"rope_type":"yarn","factor":' + f'{extend_factor},'
            '"original_max_position_embeddings":' + f'{original_pos}' + '}}'
        ),
    )

    out = []
    for i in range(15, 20):
        prompt = get_llama31_text(2**i)
        lat = latency(lambda: model.generate(prompt, {"max_new_tokens": 1}))
        out += [(2**i, lat)]
        print(f"hip latency: {lat} for {2**i}")

    with open(f"/data/jeff/delta/latency/hip.jsonl", "w") as f:
        json.dump(out, f)


def test_latency_fa3():

    original_pos = 131072
    extended_pos = 2**19 + 16384
    extend_factor = extended_pos / original_pos

    model = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="auto",
        tp_size=tp_size,
        ep_size=ep_size,
        attention_backend="fa3",
        disable_radix_cache=True,
        context_length=extended_pos,
        log_level=log_level,
        cuda_graph_max_bs=1,
        cuda_graph_bs=[1],
        show_time_cost=True,
        json_model_override_args=(
            '{"rope_scaling":{"rope_type":"yarn","factor":' + f'{extend_factor},'
            '"original_max_position_embeddings":' + f'{original_pos}' + '}}'
        ),
    )

    out = []
    for i in range(15, 20):
        prompt = get_llama31_text(2**i)
        lat = latency(lambda: model.generate(prompt, {"max_new_tokens": 1}))
        out += [(2**i, lat)]
        print(f"fa3 latency: {lat} for {2**i}")

    with open(f"/data/jeff/delta/latency/fa3.jsonl", "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    with torch.no_grad():
        # test_latency_delta()
        # test_latency_hip()
        test_latency_fa3()
