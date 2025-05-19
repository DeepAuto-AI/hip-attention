# Running HiP Attention with SGLang OpenAI server

- [Running HiP Attention with SGLang OpenAI server](#running-hip-attention-with-sglang-openai-server)
  - [Prerequisites](#prerequisites)
  - [Testing](#testing)
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
    - [Multi GPU (with cache offloading)](#multi-gpu-with-cache-offloading-1)
      - [Local](#local-4)
      - [Docker](#docker-2)
  - [`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`](#deepseek-aideepseek-r1-distill-qwen-32b)
    - [Multi GPU (with cache offloading)](#multi-gpu-with-cache-offloading-2)
      - [Local](#local-5)
      - [Docker](#docker-3)
  - [`Qwen/QwQ-32B`](#qwenqwq-32b)
    - [Multi GPU (with cache offloading)](#multi-gpu-with-cache-offloading-3)
      - [Local](#local-6)
      - [Docker](#docker-4)
  - [`meta-llama/Llama-3.3-70B-Instruct` with AWQ](#meta-llamallama-33-70b-instruct-with-awq)
    - [Multi GPU (with cache offloading)](#multi-gpu-with-cache-offloading-4)
      - [Local](#local-7)
      - [Docker](#docker-5)
  - [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](#meta-llamallama-4-scout-17b-16e-instruct)
    - [1M Context (without cache offloading)](#1m-context-without-cache-offloading)
      - [Local](#local-8)
      - [Docker](#docker-6)
  - [`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`](#meta-llamallama-4-maverick-17b-128e-instruct-fp8)
    - [1M -\> 2M Context (with cache offloading)](#1m---2m-context-with-cache-offloading)
      - [Local](#local-9)
      - [Docker](#docker-7)
  - [`Qwen/Qwen3-30B-A3B`](#qwenqwen3-30b-a3b)
    - [32K -\> 2M Context (without cache offloading)](#32k---2m-context-without-cache-offloading)
      - [Local](#local-10)
      - [Docker](#docker-8)
    - [32K -\> 5M Context (with cache offloading)](#32k---5m-context-with-cache-offloading)
      - [Local](#local-11)
      - [Docker](#docker-9)

## Prerequisites

- Export the `HF_TOKEN` environment variable.

```bash
export HF_TOKEN=<secret>
```

## Testing

```bash
SRT_PORT=30000 uv run scripts/test_openai.py
# 1M tokens
SRT_PORT=30000 uv run scripts/test_openai_long.py
```

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
uv run -m sglang.launch_server \
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
uv run -m sglang.launch_server \
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
uv run -m sglang.launch_server \
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
python \
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
uv run -m sglang.launch_server \
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
python \
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

### Multi GPU (with cache offloading)

- 1M context length
- Cache offloading enabled
- Tested model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- Testwd GPU: 4x A100 40GB
- Tested at: 2025-02-10
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
export SRT_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
export SRT_SERVED_MODEL_NAME="deepauto/DeepSeek-R1-Distill-Qwen-14B-1B-Ctx"

uv run -m sglang.launch_server \
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
export HIP_DEBUG_UNION_HEAD=0;
export HIP_HEAD_REDUCE=1;
export SRT_WARMUP_PASSKEY_LENGTH=1000;
export TOTAL_TOKENS=2097152;
export CONTEXT_LENGTH=1048576;
export DOCKER_NAME="deepseek-ai-deepseek-r1-distill-qwen-14b"
export SRT_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
export SRT_SERVED_MODEL_NAME="deepauto/DeepSeek-R1-Distill-Qwen-14B-1M-Ctx"

docker run --rm --runtime nvidia \
--gpus '"device=0,1,2,3"' \
--name $DOCKER_NAME \
-p $SRT_PORT:$SRT_PORT \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "HIP_DEBUG_UNION_HEAD=$HIP_DEBUG_UNION_HEAD" \
--env "HIP_HEAD_REDUCE=$HIP_HEAD_REDUCE" \
--env "SRT_WARMUP_PASSKEY_LENGTH=$SRT_WARMUP_PASSKEY_LENGTH" \
--env "TOTAL_TOKENS=$TOTAL_TOKENS" \
--env "CONTEXT_LENGTH=$CONTEXT_LENGTH" \
hip-sglang:1f34639 \
python \
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
export SRT_SERVED_MODEL_NAME="deepauto/deepseek-r1-distill-qwen-32b-1m-ctx"

uv run -m sglang.launch_server \
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
export HIP_DEBUG_UNION_HEAD=0;
export HIP_HEAD_REDUCE=1;
export SRT_WARMUP_PASSKEY_LENGTH=1000;
export TOTAL_TOKENS=2097152;
export CONTEXT_LENGTH=1048576;
export DOCKER_NAME="deepseek-ai-deepseek-r1-distill-qwen-32b"
export SRT_MODEL_PATH="Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ"
export SRT_SERVED_MODEL_NAME="deepauto/deepseek-r1-distill-qwen-32b-1m-ctx"

docker run --rm --runtime nvidia \
--gpus '"device=0,1,2,3"' \
--name $DOCKER_NAME \
-p $SRT_PORT:$SRT_PORT \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "HIP_DEBUG_UNION_HEAD=$HIP_DEBUG_UNION_HEAD" \
--env "HIP_HEAD_REDUCE=$HIP_HEAD_REDUCE" \
--env "SRT_WARMUP_PASSKEY_LENGTH=$SRT_WARMUP_PASSKEY_LENGTH" \
--env "TOTAL_TOKENS=$TOTAL_TOKENS" \
--env "CONTEXT_LENGTH=$CONTEXT_LENGTH" \
hip-sglang:1f34639 \
python \
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

## `Qwen/QwQ-32B`

### Multi GPU (with cache offloading)

- 1M context length
- Cache offloading enabled
- Tested model: `Qwen/QwQ-32B`
- Testwd GPU: 4x A100 40GB
- Tested at: 2025-04-06
- Tested version:
  - `hip-attention`: `600d3b614e6da8dd26c38f91d0245d046a90a046`
  - `sglang`: `cf7158db50590ef4fe98c5b5d17d15946a6eef87`

#### Local

```bash
export SRT_PORT=8921
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONTEXT_LENGTH=1048576
export SRT_WARMUP_PASSKEY_LENGTH=1000000
export CHUNK_SIZE=32768
export SRT_MODEL_PATH="Qwen/QwQ-32B"
export SRT_SERVED_MODEL_NAME="deepauto/qwq-32b-1m-ctx"

uv run -m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype auto \
--tp-size 4 \
--chunked-prefill-size $CHUNK_SIZE \
--max-prefill-tokens $CHUNK_SIZE \
--cuda-graph-bs 1 2 4 8 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $CONTEXT_LENGTH \
--max-running-requests 8 \
--enable-hip-attention \
--hip-attention-config '{"dense_layers": [0,1,2], "mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 3000 \
--hip-max-mask-cache-token-size 32000
```

#### Docker

```bash
export SRT_PORT=8921
export CONTEXT_LENGTH=1048576
export SRT_WARMUP_PASSKEY_LENGTH=1000000
export CHUNK_SIZE=32768
export DOCKER_NAME="qwen-32b-1b-ctx"
export SRT_MODEL_PATH="Qwen/QwQ-32B"
export SRT_SERVED_MODEL_NAME="deepauto/qwq-32b-1m-ctx"

docker run --rm --runtime nvidia \
--gpus '"device=0,1,2,3"' \
--name $DOCKER_NAME \
-p $SRT_PORT:$SRT_PORT \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "SRT_WARMUP_PASSKEY_LENGTH=$SRT_WARMUP_PASSKEY_LENGTH" \
hip-sglang:latest \
python \
-m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype auto \
--tp-size 4 \
--chunked-prefill-size $CHUNK_SIZE \
--max-prefill-tokens $CHUNK_SIZE \
--cuda-graph-bs 1 2 4 8 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $CONTEXT_LENGTH \
--max-running-requests 8 \
--enable-hip-attention \
--hip-attention-config '{"dense_layers": [0,1,2], "mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 3000 \
--hip-max-mask-cache-token-size 32000
```

## `meta-llama/Llama-3.3-70B-Instruct` with AWQ

### Multi GPU (with cache offloading)

- 1M context length
- Cache offloading enabled
- Tested model: `casperhansen/llama-3.3-70b-instruct-awq`
- Testwd GPU: 4x A100 40GB
- Tested at: 2025-04-06
- Tested version:
  - `hip-attention`: `600d3b614e6da8dd26c38f91d0245d046a90a046`
  - `sglang`: `cf7158db50590ef4fe98c5b5d17d15946a6eef87`
#### Local

```bash
export SRT_PORT=8921
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONTEXT_LENGTH=1048576
export SRT_WARMUP_PASSKEY_LENGTH=1000000
export CHUNK_SIZE=32768
export SRT_MODEL_PATH="casperhansen/llama-3.3-70b-instruct-awq"
export SRT_SERVED_MODEL_NAME="deepauto/llama-3.3-70b-instruct-awq-1m-ctx"

uv run -m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype auto \
--tp-size 4 \
--chunked-prefill-size $CHUNK_SIZE \
--max-prefill-tokens $CHUNK_SIZE \
--cuda-graph-bs 1 2 4 8 16 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $CONTEXT_LENGTH \
--max-running-requests 16 \
--enable-hip-attention \
--hip-attention-config '{"dense_layers": [0,1,2], "mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 3000 \
--hip-max-mask-cache-token-size 32000
```

#### Docker

```bash
export SRT_PORT=8921
export CONTEXT_LENGTH=1048576
export SRT_WARMUP_PASSKEY_LENGTH=1000000
export CHUNK_SIZE=32768
export DOCKER_NAME="llama-3-3-70b-instruct-awq-1m-ctx"
export SRT_MODEL_PATH="casperhansen/llama-3.3-70b-instruct-awq"
export SRT_SERVED_MODEL_NAME="deepauto/llama-3.3-70b-instruct-awq-1m-ctx"

docker run --rm --runtime nvidia \
--gpus '"device=0,1,2,3"' \
--name $DOCKER_NAME \
-p $SRT_PORT:$SRT_PORT \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "SRT_WARMUP_PASSKEY_LENGTH=$SRT_WARMUP_PASSKEY_LENGTH" \
hip-sglang:latest \
python \
-m sglang.launch_server \
--host 0.0.0.0 \
--port $SRT_PORT \
--model-path $SRT_MODEL_PATH \
--served-model-name $SRT_SERVED_MODEL_NAME \
--kv-cache-dtype auto \
--tp-size 4 \
--chunked-prefill-size $CHUNK_SIZE \
--max-prefill-tokens $CHUNK_SIZE \
--cuda-graph-bs 1 2 4 8 16 \
--context-length $CONTEXT_LENGTH \
--max-total-tokens $CONTEXT_LENGTH \
--max-running-requests 16 \
--enable-hip-attention \
--hip-attention-config '{"dense_layers": [0,1,2], "mask_refresh_interval": [96, 24, 8]}' \
--enable-hip-offload \
--hip-max-sa-cache-token-size 3000 \
--hip-max-mask-cache-token-size 32000
```

## `meta-llama/Llama-4-Scout-17B-16E-Instruct`

### 1M Context (without cache offloading)

- 1M context length
- Cache offloading disabled
- Tested model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- Testwd GPU: 8x H100 80GB
- Tested at: 2025-05-19
- Tested version:
  - `hip-attention`: `3974c558f79149584847576ad27a7adf32d30be5`
  - `sglang` ([DeepAuto-AI/sglang](https://github.com/DeepAuto-AI/sglang)): `8125e3e029a9f90c7800199d1a9939ffda5fa871`

#### Local

```bash
HIP_CONFIG_PRESET=llama4 \
HIP_DISABLE_FLASHDECODE=0 \
HIP_HEAD_REDUCE=0 \
HIP_DEBUG_FORCE_DENSE_DECODE=1 \
uv run \
--env-file ".env" \
-m sglang.launch_server \
--model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
--served-model-name deepauto/llama-4-scout-1m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 1000000 \
--context-length 1000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--hip-attention-config '{"using_extend": false, "dense_layers": [0, 1, 2, 3]}' \
--attention-backend flashinfer \
--chat-template llama-4
```

#### Docker

```bash
docker run --rm \
--gpus all \
--name deepauto-llama-4-scout-1m-ctx \
-p 30000:30000 \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "HIP_CONFIG_PRESET=llama4" \
--env "HIP_DISABLE_FLASHDECODE=0" \
--env "HIP_HEAD_REDUCE=0" \
--env "HIP_DEBUG_FORCE_DENSE_DECODE=1" \
deepauto/hip-attention:v1.2.5-sglang \
python \
-m sglang.launch_server \
--model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
--served-model-name deepauto/llama-4-scout-1m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 1000000 \
--context-length 1000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--hip-attention-config '{"using_extend": false, "dense_layers": [0, 1, 2, 3]}' \
--attention-backend flashinfer \
--chat-template llama-4
```

## `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`

### 1M -\> 2M Context (with cache offloading)

- 2M context length
- Cache offloading enabled
- Tested model: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- Testwd GPU: 8x H100 80GB
- Tested at: 2025-05-19
- Tested version:
  - `hip-attention`: `3974c558f79149584847576ad27a7adf32d30be5`
  - `sglang` ([DeepAuto-AI/sglang](https://github.com/DeepAuto-AI/sglang)): `8125e3e029a9f90c7800199d1a9939ffda5fa871`

#### Local
```bash
HIP_CONFIG_PRESET=llama4 \
HIP_DISABLE_FLASHDECODE=1 \
HIP_HEAD_REDUCE=0 \
HIP_DEBUG_FORCE_DENSE_DECODE=0 \
uv run \
--env-file ".env" \
-m sglang.launch_server \
--model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
--served-model-name deepauto/llama-4-maverick-2m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 2000000 \
--context-length 2000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--hip-attention-config '{"using_extend": false, "dense_layers": [0, 1, 2, 3]}' \
--attention-backend flashinfer \
--chat-template llama-4 \
--enable-hip-kv-cache-offload \
--hip-max-mask-cache-size 131072 \
--hip-max-sa-cache-size 8192
```

#### Docker

```bash
docker run --rm \
--gpus all \
--name deepauto-llama-4-maverick-2m-ctx \
-p 30000:30000 \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "HIP_CONFIG_PRESET=llama4" \
--env "HIP_DISABLE_FLASHDECODE=0" \
--env "HIP_HEAD_REDUCE=0" \
--env "HIP_DEBUG_FORCE_DENSE_DECODE=0" \
deepauto/hip-attention:v1.2.5-sglang \
python \
-m sglang.launch_server \
--model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
--served-model-name deepauto/llama-4-maverick-2m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 2000000 \
--context-length 2000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--hip-attention-config '{"using_extend": false, "dense_layers": [0, 1, 2, 3]}' \
--attention-backend flashinfer \
--chat-template llama-4 \
--enable-hip-kv-cache-offload \
--hip-max-mask-cache-size 131072 \
--hip-max-sa-cache-size 8192
```

## `Qwen/Qwen3-30B-A3B`

### 32K -\> 2M Context (without cache offloading)

- 2M context length
- Cache offloading disabled
- Tested model: `Qwen/Qwen3-30B-A3B`
- Testwd GPU: 8x H100 80GB
- Tested at: 2025-05-19
- Tested version:
  - `hip-attention`: `3974c558f79149584847576ad27a7adf32d30be5`
  - `sglang` ([DeepAuto-AI/sglang](https://github.com/DeepAuto-AI/sglang)): `8125e3e029a9f90c7800199d1a9939ffda5fa871`

#### Local

```bash
HIP_CONFIG_PRESET=qwen3 \
HIP_DISABLE_FLASHDECODE=0 \
HIP_HEAD_REDUCE=1 \
HIP_DEBUG_FORCE_DENSE_DECODE=1 \
uv run \
--env-file ".env" \
-m sglang.launch_server \
--model-path Qwen/Qwen3-30B-A3B \
--served-model-name deepauto/qwen3-30b-a3b-2m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 2000000 \
--context-length 2000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--attention-backend flashinfer
```

#### Docker

```bash
docker run --rm \
--gpus all \
--name deepauto-qwen3-30b-a3b-2m-ctx \
-p 30000:30000 \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "HIP_CONFIG_PRESET=qwen3" \
--env "HIP_DISABLE_FLASHDECODE=0" \
--env "HIP_HEAD_REDUCE=1" \
--env "HIP_DEBUG_FORCE_DENSE_DECODE=1" \
deepauto/hip-attention:v1.2.5-sglang \
python \
-m sglang.launch_server \
--model-path Qwen/Qwen3-30B-A3B \
--served-model-name deepauto/qwen3-30b-a3b-2m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 2000000 \
--context-length 2000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--attention-backend flashinfer
```

### 32K -\> 5M Context (with cache offloading)

- 5M context length
- Cache offloading enabled
- Tested model: `Qwen/Qwen3-30B-A3B`
- Testwd GPU: 8x H100 80GB
- Tested at: 2025-05-19
- Tested version:
  - `hip-attention`: `3974c558f79149584847576ad27a7adf32d30be5`
  - `sglang` ([DeepAuto-AI/sglang](https://github.com/DeepAuto-AI/sglang)): `8125e3e029a9f90c7800199d1a9939ffda5fa871`

#### Local

```bash
HIP_CONFIG_PRESET=qwen3 \
HIP_DISABLE_FLASHDECODE=0 \
HIP_HEAD_REDUCE=1 \
HIP_DEBUG_FORCE_DENSE_DECODE=1 \
uv run \
--env-file ".env" \
-m sglang.launch_server \
--model-path Qwen/Qwen3-30B-A3B \
--served-model-name deepauto/qwen3-30b-a3b-5m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 5000000 \
--context-length 5000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--attention-backend flashinfer \
--enable-hip-kv-cache-offload \
--hip-max-mask-cache-size 266144 \
--hip-max-sa-cache-size 16384
```

#### Docker

```bash
docker run --rm \
--gpus all \
--name deepauto-qwen3-30b-a3b-5m-ctx \
-p 30000:30000 \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=${HF_TOKEN}" \
--env "HIP_CONFIG_PRESET=qwen3" \
--env "HIP_DISABLE_FLASHDECODE=0" \
--env "HIP_HEAD_REDUCE=1" \
--env "HIP_DEBUG_FORCE_DENSE_DECODE=1" \
deepauto/hip-attention:v1.2.5-sglang \
python \
-m sglang.launch_server \
--model-path Qwen/Qwen3-30B-A3B \
--served-model-name deepauto/qwen3-30b-a3b-5m-ctx \
--host 0.0.0.0 \
--port 30000 \
--tp 8 \
--max-total-tokens 5000000 \
--context-length 5000000 \
--cuda-graph-bs 1 2 4 8 16 24 32 \
--max-running-req 32 \
--chunked-prefill-size 65536 \
--enable-hip-attention \
--attention-backend flashinfer \
--enable-hip-kv-cache-offload \
--hip-max-mask-cache-size 266144 \
--hip-max-sa-cache-size 16384
```
