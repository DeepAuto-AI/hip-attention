#!/bin/bash

if [[ "${CUDA_VISIBLE_DEVICES}" == "" ]]; then
    export CUDA_VISIBLE_DEVICES=1
fi
export HIP_DISABLE_AUTOTUNE_WARNINGS=1
export HIP_DEBUG=0 
export HIP_DISABLE_AUTOTUNE=0 

function bench {
    echo "bench prefill=$1 decode=$2 k=$3 T=${4}k"

    python hip/models/hip_attention/attention1_block_gpu.py --method $1 --dups $4 --query_size $(( $4 * 1024 )) --batch_size 1 --k $3 --block_size_q 64 --block_stride_q 2 --block_size_k 2 --block_stride_k 1 --refresh_interval 8 --sample 50

    echo --------------------------------------------------

    export HIP_USING_SNAP_KV=0
    python hip/models/hip_attention/attention1_block_gpu.py --method $2 --dups $4 --query_size 1 --batch_size 32 --k $3 --block_size_q 64 --block_stride_q 2 --block_size_k 2 --block_stride_k 1 --refresh_interval 8
}

function SLLM {
    echo ==================================================
    echo SLLM $1

    bench "streaming" "streaming" $1 $2

    echo ==================================================
}

function fa {
    echo ==================================================
    echo FA2

    bench "fa2" "fa2" 0 $1

    echo ==================================================
}

function AV {
    echo ==================================================
    echo AV $1

    export HIP_USING_SNAP_KV=1
    export HIP_SNAP_KV_VERT_K=$1
    export HIP_SNAP_KV_DIAG_K=$1
    export HIP_SNAP_KV_NO_OVERLAP=0
    export HIP_BK_AFTER_MASK=64
    export GROUP_SIZE_Q=1
    export HIP_NSINK=16
    export HIP_SW=$(( $1 / 2 ))

    bench "hip1.1" "fa2" 16 $2

    echo ==================================================
}

function bigbird {
    echo ==================================================
    echo BigBird $1

    export HIP_USING_SNAP_KV=0
    export HIP_BK_AFTER_MASK=-1
    export GROUP_SIZE_Q=1
    export HIP_RANDOM_MASK=1
    export HIP_NSINK=$(( $1 / 32 ))
    export HIP_SW=$(( $1 / 2 ))

    bench "hip1.1" "hip1.1" $1 $2

    export HIP_RANDOM_MASK=0
    echo ==================================================
}

function hip {
    echo ==================================================
    echo HiP $1

    export HIP_USING_SNAP_KV=0
    export HIP_BK_AFTER_MASK=-1
    export GROUP_SIZE_Q=1
    export HIP_NSINK=$(( $1 / 32 ))
    export HIP_SW=$(( $1 / 2 ))

    bench "hip1.1" "hip1.1" $1 $2
    echo ==================================================
}

function hip_AV {
    echo ==================================================
    echo HiP + AV $1

    export HIP_USING_SNAP_KV=1 
    export HIP_SNAP_KV_VERT_K=$1
    export HIP_SNAP_KV_DIAG_K=$1
    export HIP_SNAP_KV_NO_OVERLAP=0
    export HIP_BK_AFTER_MASK=16
    export GROUP_SIZE_Q=1
    export HIP_NSINK=128
    export HIP_SW=$1

    bench "hip1.1" "hip1.1" $1 $2
    echo ==================================================
}

function bench_ruler {
    AV 1024 128
    AV 2048 128
    AV 4096 128

    bigbird 512 128
    bigbird 1024 128
    bigbird 2048 128
    bigbird 4096 128

    hip 512 128
    hip 1024 128
    hip 2048 128
    hip 4096 128

    hip_AV 1024 128
    hip_AV 2048 128

    fa 128
}

function bench_longbench {
    SLLM 512 32

    bigbird 512 32
    bigbird 1024 32

    AV 1024 32

    hip 512 32
    hip 1024 32
    hip_AV 512 32

    fa 32
}

# =============================================================================

# bench_ruler
bench_longbench