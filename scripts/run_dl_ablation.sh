#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seq_len in 4096 8192 12000
do
    for dl in 0 1 2 3 4 5 10 15 20 25 30 32
    do
        python hip/main/model_eval.py --method hip --k 512 --block_size_q 32 --block_size_k 2 --dense_layers $dl --stride $seq_len
    done
done
