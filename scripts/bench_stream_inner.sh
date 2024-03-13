#!/bin/bash

export METHOD=timber
export BENCHMARK_RUNNER=1
export PROMPT_ATTENTION_BACKEND=$METHOD
export PAGED_ATTENTION_BACKEND=$METHOD
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=./
export HF_HOME=/d1/heejun/.cache/huggingface/transformers
export TRANSFORMERS_CACHE=/d1/heejun/.cache/huggingface/transformers
export FORCE_EAGER=0

/home/ainl/anaconda3/envs/timber/bin/python timber/main/llama_eval.py --model vllm_yi6b --method timber --job stream --stride 32000 --input sample32k.md --batch_size 8 --max_tokens 16