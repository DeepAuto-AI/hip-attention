"""
PAGED_ATTENTION_BACKEND=hip python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-7B-Chat-GPTQ-Int4 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8_e5m2 \
    --dtype half \
    --gpu-memory-utilization 0.8
"""
