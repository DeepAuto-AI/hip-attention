#!/bin/bash

function nsys_bench_fa2 {
    echo method fa2 T=$1
    sudo -E $(which nsys) profile -t cuda --cuda-graph-trace node --stats=false --output report_fa2_512_$1.nsys-rep --force-overwrite=true $(which python)  hip/models/hip_attention/attention1_block_gpu.py --method fa2 --dups $1 --query_size 1 --batch_size 32
}

function nsys_bench_hip {
    echo method hip T=$1
    sudo -E $(which nsys) profile -t cuda --cuda-graph-trace node --stats=false --output report_hip_512_$1.nsys-rep --force-overwrite=true $(which python)  hip/models/hip_attention/attention1_block_gpu.py --k 512 --method hip1.1 --dups $1 --query_size 1 --batch_size 32 --block_size_q 64 --block_stride_q 2 --block_size_k 2 --block_stride_k 1
}

for T in 8 16 32 64 128
do

nsys_bench_fa2 $T
nsys_bench_hip $T

done
