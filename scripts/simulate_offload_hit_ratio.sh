#!/bin/bash

SA_HIT_RATIO=0.99

for MASK_HIT_RATIO in 1.0 0.975 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do
echo "============================================================"
echo " Mask hit ratio = ${MASK_HIT_RATIO}, Sparse attn hit ratio = ${SA_HIT_RATIO}"
echo "============================================================"

CUDA_LAUNCH_BLOCKING=0 HIP_DISABLE_AUTOTUNE=0 python -b \
    hip/models/hip_attention/offload_runner/offload_runner.py \
        --cache_backend uvm \
        --kv_share 1 \
        --method hip \
        --offload-cache \
        --batch_size 8 \
        --sw 512 \
        --max_tokens 256 \
        --input ./samples/32k.md \
        --cache_size 4096 \
        --refresh_interval 8 \
        --simulate-hit-ratio \
        --simulated_mask_hit_ratio $MASK_HIT_RATIO \
        --simulated_sa_hit_ratio $SA_HIT_RATIO

done
