#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python hip/main/llama_eval.py --job mmlu --model llama32k --method hip --block_size_q 32 --block_size_k 2 --k 512 --dense_layers 3
python hip/main/llama_eval.py --job mmlu --model llama13b_32k --method hip --block_size_q 32 --block_size_k 2 --k 512 --dense_layers 4
