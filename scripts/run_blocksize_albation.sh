#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for bq in 32 16 8 4 2 1
do
    for bk in 1 2 4 8
    do
        python hip/main/model_eval.py --method hip --k 512 --block_size_q $bq --block_size_k $bk --dense_layers 3 --stride 12000
    done
done
