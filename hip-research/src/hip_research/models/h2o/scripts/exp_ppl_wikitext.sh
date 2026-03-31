#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Define strides and method (can be h2o or h2o_stream)
strides=(8192 16384 32768 65536 131072)
echo METHOD=${METHOD}

# Loop over strides and execute the command with each stride
for stride in "${strides[@]}"; do
    echo "Running with stride $stride and method $METHOD"
    python -m hip.main.model_eval --job ppl --stride "$stride" --method "$METHOD" --k 512 --model llama3.1_8b
done
