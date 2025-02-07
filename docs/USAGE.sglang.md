# Running HiP Attention with SGLang OpenAI server

- [Running HiP Attention with SGLang OpenAI server](#running-hip-attention-with-sglang-openai-server)
  - [`meta-llama/Llama-3.1-8B-Instruct`](#meta-llamallama-31-8b-instruct)
    - [Single GPU (with cache offloading)](#single-gpu-with-cache-offloading)
      - [Local](#local)
    - [Single GPU (WITHOUT cache offloading)](#single-gpu-without-cache-offloading)
      - [Local](#local-1)
    - [Multi GPU (with cache offloading)](#multi-gpu-with-cache-offloading)
      - [Local](#local-2)
      - [Docker](#docker)
  - [`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`](#deepseek-aideepseek-r1-distill-qwen-14b)
    - [Single GPU (with cache offloading)](#single-gpu-with-cache-offloading-1)
      - [Local](#local-3)
      - [Docker](#docker-1)
  - [`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`](#deepseek-aideepseek-r1-distill-qwen-32b)
    - [Multi GPU (with cache offloading)](#multi-gpu-with-cache-offloading-1)
      - [Local](#local-4)
  - [Testing](#testing)

## `meta-llama/Llama-3.1-8B-Instruct`

### Single GPU (with cache offloading)

- 2M context length
- Cache offloading enabled
- For cache offloading, KV cache type is `fp8_e5m2`
- Tested model: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- Testwd GPU: 1x L40S 48GB
- Tested at: 2025-01-29
- Tested version:
  - `hip-attention`: `a1f2578e0b8d948efdb7df10bad89be0b09c47c6`
  - `sglang`: `0005b7e1e2523e7ed40a5f6a43a62e2306e95c55`

#### Local

```bash
export SRT_PORT=9913
export CONTEXT_LENGTH=2048000
export DOCKER_NAME="meta-llama-llama-3-1-8b-instruct"
export SRT_MODEL_PATH="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
export SRT_SERVED_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

SRT_WARMUP_PASSKEY_LENGTH=1024000 \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype fp8_e5m2 \
--tp-size 1 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--cuda-graph-bs 1 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $CONTEXT_LENGTH \
--max-running-requests 1 \
--enable-hip-attention \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 5000 \
--hip-max-mask-cache-token-size 64000
```

### Single GPU (WITHOUT cache offloading)

- 2M context length
- Cache offloading disabled
- Tested model: `meta-llama/Llama-3.1-8B-Instruct`
- Testwd GPU: 1x L40S 48GB
- Tested at: 2025-01-29
- Tested version:
  - `hip-attention`: `a1f2578e0b8d948efdb7df10bad89be0b09c47c6`
  - `sglang`: `0005b7e1e2523e7ed40a5f6a43a62e2306e95c55`

#### Local

```bash
export SRT_PORT=9913
export CONTEXT_LENGTH=1024000
export DOCKER_NAME="meta-llama-llama-3-1-8b-instruct"
export SRT_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
export SRT_SERVED_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

SRT_WARMUP_PASSKEY_LENGTH=1024000 \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype auto \
--tp-size 1 \
--mem-fraction-static 0.8 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $CONTEXT_LENGTH \
--max-running-request 32 \
--enable-hip-attention \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}' \
--allow-auto-truncate
```

### Multi GPU (with cache offloading)

- 2M context length
- With cache offloading
- For cache offloading, KV cache type is `fp8_e5m2`
- Tested model: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- Testwd GPU: 2x A100 40GB
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

### Single GPU (with cache offloading)

- 2M context length
- Cache offloading enabled
- Tested model: `neody/r1-14b-awq`
- Testwd GPU: 1x L40S 48GB
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

## `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

### Multi GPU (with cache offloading)

- 1M context length
- Cache offloading enabled
- Tested model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- Testwd GPU: 4x A100 40GB
- Tested at: 2025-02-07
- Tested version:
  - `hip-attention`: `1f346394bf98c4f53b3484d83c746435038b5b98`
  - `sglang`: `06fafb06657f18103886956995da5ecbdc5f1817`

#### Local

```bash
export SRT_PORT=8921
export HIP_DEBUG_UNION_HEAD=1;
export HIP_HEAD_REDUCE=0;
export CUDA_VISIBLE_DEVICES=0,1,2,3;
export SRT_WARMUP_PASSKEY_LENGTH=1000;
export TOTAL_TOKENS=2097152;
export CONTEXT_LENGTH=1048576;
export SRT_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export SRT_SERVED_MODEL_NAME="deepauto/DeepSeek-R1-Distill-Qwen-32B-1B-Ctx"

$(which python) -m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype fp8_e5m2 \
--tp-size 4 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--stream-interval 1 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $TOTAL_TOKENS \
--max-running-requests 1 \
--cuda-graph-bs 1 \
--enable-hip-attention \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 32768 \
--hip-max-mask-cache-token-size 131072 \
--disable-custom-all-reduce
```

#### Docker

```bash
export SRT_PORT=8921
export HIP_DEBUG_UNION_HEAD=1;
export HIP_HEAD_REDUCE=0;
export SRT_WARMUP_PASSKEY_LENGTH=1000;
export TOTAL_TOKENS=2097152;
export CONTEXT_LENGTH=1048576;
export DOCKER_NAME="deepseek-ai-deepseek-r1-distill-qwen-32b"
export SRT_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export SRT_SERVED_MODEL_NAME="deepauto/DeepSeek-R1-Distill-Qwen-32B-1B-Ctx"

docker run --rm --runtime nvidia \
--gpus '"device=0,1,2,3"' \
--name $DOCKER_NAME \
-p $SRT_PORT:$SRT_PORT \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN={__secret__}" \
--env "HIP_DEBUG_UNION_HEAD=$HIP_DEBUG_UNION_HEAD" \
--env "HIP_HEAD_REDUCE=$HIP_HEAD_REDUCE" \
--env "SRT_WARMUP_PASSKEY_LENGTH=$SRT_WARMUP_PASSKEY_LENGTH" \
--env "TOTAL_TOKENS=$TOTAL_TOKENS" \
--env "CONTEXT_LENGTH=$CONTEXT_LENGTH" \
hip-sglang:1f34639 \
python3 \
-m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype fp8_e5m2 \
--tp-size 4 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--stream-interval 1 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $TOTAL_TOKENS \
--max-running-requests 1 \
--cuda-graph-bs 1 \
--enable-hip-attention \
--hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 32768 \
--hip-max-mask-cache-token-size 131072 \
--disable-custom-all-reduce
```

## Testing

```bash
SRT_PORT=9913 python scripts/test_openai.py
# 1M tokens
SRT_PORT=9913 python scripts/test_openai_long.py
```
****
