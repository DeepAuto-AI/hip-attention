#!/bin/bash

SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_DEBUG_EXIT_WARMUP=0 \
HIP_DEBUG_CAPTURE_DECORATOR=0 \
HIP_DISABLE_AUTOTUNE=1 \
HIP_DEBUG_BENCH=0 \
SA_BLOCK_SIZE_LANDMARK=256 \
SA_BLOCK_SIZE=128 \
PASSKEY_LEN=126 \
HIP_DEBUG=0 \
HIP_DEBUG_LOGALL=1 \
CUDA_VISIBLE_DEVICES=4,5 \
$(which python) -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp 2 \
    --max-total-tokens 131072 \
    --context-length 131072 \
    --cuda-graph-bs 1 \
    --max-running-req 1 \
    --chunked-prefill-size 131072 \
    --hip-attention-config ./configs/qwen3_1b_norm_const.json \
    --attention-backend flashinfer \
    --port 20000 \
    --enable-hip-attention \
    --port 33330
