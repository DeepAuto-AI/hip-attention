#!/bin/bash

export OFFLOAD_PREFETCH=0 
export BENCHMARK_RUNNER=1 
export CACHE_ENGINE='vllm' 
export ATTENTION_BACKEND='hip' 
export PROMPT_ATTENTION_BACKEND='hip' 
export PAGED_ATTENTION_BACKEND='hip' 
export HIP_DENSE_LAYERS=4 
export HIP_K=512 
export CUDA_VISIBLE_DEVICES=1 
export HF_HOME=/d1/heejun/cache/hf_home
/home/ainl/miniconda3/envs/hip/bin/python -b timber/main/llama_eval.py --model vllm_llama1b --job stream --batch_size 1 --input samples/16k.md --stride 20000 --max_tokens 3