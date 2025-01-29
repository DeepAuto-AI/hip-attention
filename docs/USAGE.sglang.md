# Running HiP Attention with SGLang OpenAI server

- [Running HiP Attention with SGLang OpenAI server](#running-hip-attention-with-sglang-openai-server)
  - [`meta-llama/Llama-3.1-8B-Instruct`](#meta-llamallama-31-8b-instruct)
    - [Multi GPU](#multi-gpu)
      - [Local](#local)
      - [Docker](#docker)
  - [`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`](#deepseek-aideepseek-r1-distill-qwen-14b)
    - [Single GPU](#single-gpu)
      - [Local](#local-1)
      - [Docker](#docker-1)

## `meta-llama/Llama-3.1-8B-Instruct`

### Multi GPU

- 2M context length
- With cache offloading
- For cache offloading, KV cache type is `fp8_e5m2`
- Tested model: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- Testwd on 2x A100 40GB
- Tested at: 2025-01-29
- Tested version:
  - `hip-attention`: `a1f2578e0b8d948efdb7df10bad89be0b09c47c6`
  - `sglang`: `0005b7e1e2523e7ed40a5f6a43a62e2306e95c55`

#### Local

```bash
export SRT_PORT=9913
export CUDA_VISIBLE_DEVICES=0,1
export CACHE_SIZE=2048000

SRT_WARMUP_PASSKEY_LENGTH=1024000 \
python -m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
--served-model-name meta-llama/Llama-3.1-8B-Instruct \
--kv-cache-dtype fp8_e5m2 \
--tp-size 2 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--cuda-graph-bs 1 \
--context-length $CACHE_SIZE \
--max-total-tokens $CACHE_SIZE \
--max-running-requests 1 \
--enable-hip-attention \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 8000 \
--hip-max-mask-cache-token-size 128000
```

#### Docker

```bash
export SRT_PORT=9913
export CONTEXT_LENGTH=2048000
export DOCKER_NAME="meta-llama-llama-3-1-8b-instruct"
export SRT_MODEL_PATH="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
export SRT_SERVED_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

docker run --rm --runtime nvidia \
--gpus '"device=0,1"' \
--name $DOCKER_NAME \
-p $SRT_PORT:$SRT_PORT \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HUGGING_FACE_HUB_TOKEN=<secret>" \
--env "SRT_WARMUP_PASSKEY_LENGTH=1024000" \
hip-sglang:latest \
python3 \
-m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype fp8_e5m2 \
--tp-size 2 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--cuda-graph-bs 1 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $CONTEXT_LENGTH \
--max-running-requests 1 \
--enable-hip-attention \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 8000 \
--hip-max-mask-cache-token-size 128000
```

## `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

### Single GPU

- 2M context length
- Cache offloading enabled
- Tested model: `neody/r1-14b-awq`
- Testwd on 1x L40S 48GB
- Tested at: 2025-01-29
- Tested version:
  - `hip-attention`: `a1f2578e0b8d948efdb7df10bad89be0b09c47c6`
  - `sglang`: `0005b7e1e2523e7ed40a5f6a43a62e2306e95c55`

#### Local

```bash
export SRT_PORT=9913
export CONTEXT_LENGTH=1048576
export DOCKER_NAME="deepseek-ai-deepseek-r1-distill-qwen-14b"
export SRT_MODEL_PATH="neody/r1-14b-awq"
export SRT_SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

SRT_WARMUP_PASSKEY_LENGTH=1024000 \
CUDA_VISIBLE_DEVICES=0 \
HIP_HEAD_REDUCE=1 \
SRT_MAX_BATCH=1 \
python -m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--tp-size 1 \
--kv-cache-dtype auto \
--chunked-prefill-size 49152 \
--max-prefill-tokens 49152 \
--stream-interval 1 \
--context-length $CONTEXT_LENGTH \
--enable-hip-attention \
--max-running-requests 1 \
--cuda-graph-bs 1 \
--enable-hip-offload \
--hip-max-mask-cache-token-size 64000 \
--hip-max-sa-cache-token-size 4096 \
--max-total-tokens $CONTEXT_LENGTH \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}'
```

#### Docker

```bash
export SRT_PORT=9913
export CONTEXT_LENGTH=1048576
export DOCKER_NAME="deepseek-ai-deepseek-r1-distill-qwen-14b"
export SRT_MODEL_PATH="neody/r1-14b-awq"
export SRT_SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

docker run --rm --runtime nvidia \
--gpus '"device=0"' \
--name $DOCKER_NAME \
-p $SRT_PORT:$SRT_PORT \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HUGGING_FACE_HUB_TOKEN=<secret>" \
--env "SRT_WARMUP_PASSKEY_LENGTH=1024000" \
--env "HIP_HEAD_REDUCE=1" \
--env "SRT_MAX_BATCH=1" \
hip-sglang:latest \
python3 \
-m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--tp-size 1 \
--kv-cache-dtype auto \
--chunked-prefill-size 49152 \
--max-prefill-tokens 49152 \
--stream-interval 1 \
--context-length $CONTEXT_LENGTH \
--enable-hip-attention \
--max-running-requests 1 \
--cuda-graph-bs 1 \
--enable-hip-offload \
--hip-max-mask-cache-token-size 64000 \
--hip-max-sa-cache-token-size 4096 \
--max-total-tokens $CONTEXT_LENGTH \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}'
```
