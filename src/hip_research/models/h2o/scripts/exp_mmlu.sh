#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="llama3.1_8b"

echo MODEL=${MODEL}
echo METHOD=${METHOD}

# python hip/main/llama_eval.py --job mmlu --model llama32k --method hip --block_size_q 32 --block_size_k 2 --k 512 --dense_layers 3
# python hip/main/llama_eval.py --job mmlu --model llama13b_32k --method hip --block_size_q 32 --block_size_k 2 --k 512 --dense_layers 4

# h2o
if [ "${METHOD}" == "h2o" ]; then
echo run h2o
python -m hip.main.model_eval --job mmlu --model llama3.1_8b --method h2o
elif [ "${METHOD}" == "h2o_stream" ]; then
echo run h2o_stream
python -m hip.main.model_eval --job mmlu --model llama3.1_8b --method h2o_stream
fi
