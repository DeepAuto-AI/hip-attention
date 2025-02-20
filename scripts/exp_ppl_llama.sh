#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# llama13b_32k
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 1 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 2 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 4 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 8 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 16 --k 128

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 1 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 2 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 4 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 8 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 16 --k 256

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 8 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 16 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 1 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 2 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 4 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 8 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 16 --k 1024

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 32 --block_size_k 32 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 16 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 16 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 16 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 16 --block_size_k 8 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 16 --block_size_k 16 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 8 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 8 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 8 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 8 --block_size_k 8 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 4 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 4 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 4 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama13b_32k --dense_layers 4 --method hip --block_size_q 4 --block_size_k 8 --k 512

# llama32k
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 1 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 2 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 4 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 8 --k 128
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 16 --k 128

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 1 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 2 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 4 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 8 --k 256
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 16 --k 256

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 8 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 16 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 1 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 2 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 4 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 8 --k 1024
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 16 --k 1024

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 32 --block_size_k 32 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 16 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 16 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 16 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 16 --block_size_k 8 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 16 --block_size_k 16 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 8 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 8 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 8 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 8 --block_size_k 8 --k 512

python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 4 --block_size_k 1 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 4 --block_size_k 2 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 4 --block_size_k 4 --k 512
python hip/main/llama_eval.py --job ppl --stride 8192 --dense_queries 0 --model llama32k --dense_layers 3 --method hip --block_size_q 4 --block_size_k 8 --k 512
