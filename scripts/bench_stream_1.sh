#!/bin/bash

export OFFLOAD_PREFETCH=0
export BENCHMARK_RUNNER=1
export CACHE_ENGINE='vllm'
export ATTENTION_BACKEND=$BACKEND
export PROMPT_ATTENTION_BACKEND=$BACKEND
export PAGED_ATTENTION_BACKEND=$BACKEND
export HIP_DENSE_LAYERS=4 
export HIP_K=512
# export CUDA_VISIBLE_DEVICES=0,1
# export HF_HOME=/d1/heejun/cache/hf_home

echo HF_HOME=$HF_HOME
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo BACKEND=$BACKEND
echo PYBIN=$PYBIN
echo TARGET=samples/$FILENAME.md
echo BATCH_SIZE=$BATCH_SIZE
echo HIP_K=$HIP_K
echo HIP_REFRESH_INTERVAL=$HIP_REFRESH_INTERVAL
$PYBIN -b timber/main/model_eval.py \
    --model vllm_qwen14b_gptq \
    --job stream \
    --batch_size $BATCH_SIZE \
    --input samples/$FILENAME.md \
    --stride 32000 \
    --max_tokens 32