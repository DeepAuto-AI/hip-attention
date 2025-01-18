![demo gif](docs/demo.gif)

| [**Demo**](docs/demo.mp4) | [**Preprint**](https://arxiv.org/abs/2406.09827) | [**SGlang Integration**](https://github.com/DeepAuto-AI/sglang) | [**vLLM Integration (Deprecated)**](https://github.com/DeepAuto-AI/vllm) |

**HiP Attention** reduces the computational cost of quadratic attention, such as Flash Attention, into sub-quadratic `O(T log T)` in a plug-and-play manner while maintaining original performance using hierarchically pruned sparse attention. We are aiming to support future researchers while maintaining practical efficiency with this project.

## News

- 2025.01.03: Version 1.2 will be released soon. The new version fully supports context extension and better controls pruning hierarchy. It will also have better SGlang support (with proper KV offloading!, [working preview](https://github.com/gmlwns2000/sglang-hip12)).
- 2024.10.05: Version 1.1 is now ready, check `ainl-hip-offload`. KV offloading feature in under alpha state.
- 2024.09.09: Version 1.1 will be released soon. Please refer to the `ainl-hip-attention2` branch for a preview. It will reduce the latency further and improve the accuracy (and this will fix most of the internal bugs of v1.0). It offers many more experimental options for further research (e.g., key access logs, modular design of masking kernel). As discussed in the Appendix, this release will actually have (hopefully) a KV offloading feature, either UVM or a custom cache management algorithm. Also, SGLang will be supported by this release. Please take a look at our company's fork for a preview.

## Usage

After installation, you can access the `hip` package from any project. `hip` is the code name of HiP attention.

## TL;DR

**We provide an OpenAI-compatible API server built with vLLM and HiP attention!** The only thing you need to integrate HiP is replacing the single line of the flash attention call.

```diff
- | from flash_attn import flash_attn_func
- | context = flash_attn_func(q, k, v, sm_scale=1.0, is_causal=True)
+ | from hip import hip_attention
+ | context, metadata = hip_attention(q, k, v)
```

## SGlang

```
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
POPULATION_FILE=none
HIP_EXTEND_CONTEXT_LENGTH=131072
HIP_REFRESH_INTERVAL=4
SRT_DEBUG_DECODE_SPECIAL_TOKENS=1
EXTEND_LEN=512 HIP_EXTEND=1
HIP_DISABLE_AUTOTUNE=1
SRT_ATTENTION_BACKEND=HIP_ATTN
SRT_MAX_BATCH=8
python -m sglang.launch_server \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.6 \
    --tp-size 1 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 8192 \
    --context-length 256000 \
    --port 30000 \
    --enable-p2p-check \
    --disable-cuda-graph
```

The above command requires version 1.2. Set `HIP_EXTEND=0` to use version 1.1.

## API

```py
from torch import Tensor
from typing import Tuple
from hip import (
    hip_attention, 
    HiPAttentionArgs, 
    HiPAttentionOutputMetadata
)

# NOTE: **you have to scale the Q before pass to our kernel**
scale = 1 / (HID ** 0.5)

"""
- q: Tensor[N, TDST, H, HID]
- k: Tensor[N, TSRC, H, HID]
- v: Tensor[N, TSRC, H, HID]
    query, key, value of attention mechanism.

- mask_k: int, 
    same as $k$ in the paper
- block_size_q: int, 
    same as $b_q$ in the paper.
- block_stride_q: int,
    same as $b_{sq}$ in the paper.
- block_size_k: int, 
    same as $b_k$ in the paper.
- block_stride_k: int,
    same as $b_{sk}$ in the paper.

... Please refer HiPAttentionArgs for more details ...
"""

output, _ = hip_attention(
    q=q * scale, # NOTE: **IMPORTANT** You have to scale Q before attention, because we did not implement softmax scaler... 
    k=k,
    v=v,
    mask_k=512,
    block_size_q=64,
    block_stride_q=2,
    block_size_k=2,
    block_stride_k=1,
) # type: Tuple[Tensor[N, TDST, HEAD, HID], HiPAttentionMetadata]

from hip import hip_attention, paged_hip_attention

"""
Paged Attention Supported HiP Attention

This function is already integrated with in provided vLLM patches.
Please look following sections, to utilize the paged attention and 
OpenAI compatible API server with HiP.
"""
output, _ = paged_hip_attention(
    ...
) # type: Tuple[Tensor[N, TDST, H, HID], ...]
```

## How To Install

### How to clone the repository

```bash
git clone {REPO URL}
cd hip-attention
```

### Running Docker

After building the container, run commands below (change `--gpus` and `--tensor-parallel-size` according to your environment):

```bash
docker run --runtime nvidia --rm -it --ipc=host \
    --gpus '"device=0"' \
    -p 8001:8001 \
    -v ~/.cache/huggingface/:/root/.cache/huggingface \
    -e 'ATTENTION_BACKEND=hip' \
    -e 'HIP_K=512' \
    -e 'HIP_REFRESH_INTERVAL=8' \
    -e 'HIP_DENSE_LAYERS=4' \
    hip/vllm-hip-openai:latest \
        --port 8001 \
        --model Qwen/Qwen2-1.5B-Instruct \
        --tensor-parallel-size 1 \
        --kv-cache-dtype fp8_e5m2 \
        --dtype half \
        --gpu-memory-utilization 0.50
```

### How to build Docker

Run commands below:

```bash
cd ../
git clone {REPO URL}
cd vllm
docker build . --build-context hip=../hip-attention --target vllm-openai --tag hip/vllm-hip-openai
```

### Setup without docker

```bash
conda create --name llm python=3.11
conda activate llm

cd hip-attention
pip install -e "."
# Optional for development
pip install -e ".[dev]"

# Optional, depends on your CUDA environment
export CUDACXX=/usr/local/cuda/bin/nvcc
# Dependencies that requires --no-build-isolation
pip install -e ".[no_build_iso]" --no-build-isolation --verbose
# vLLM with OpenAI API support for serving
pip install -e ".[vllm,openai]" --no-build-isolation --verbose
```

### Running without docker
```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ATTENTION_BACKEND=HIP_ATTN \
HIP_K=512 \
HIP_REFRESH_INTERVAL=8 \
HIP_DENSE_LAYERS=4 \
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-1.5B-Instruct \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8_e5m2 \
    --dtype half \
    --gpu-memory-utilization 0.50
```

This command will be deprecated. Check below's SGlang `sglang.launch_server` command.

### vllm + Qwen's Dynamic-NTK

add the following content in Qwen's `config.json`. 

- `seq_length` is the threshold for activating NTK, default 8192 (the same as Qwen).
- `factor` does not affect the logic of dynamic-ntk. It is used by vllm to calculate the maximum input length for model. If it is set to 1, warnings will occur if input is longer than 8192. Setting to 4 may be enough.

```
"rope_scaling": {
    "type": "dynamic-qwen",
    "seq_length": 8192,
    "factor": 4.0
}
```

## Experiments Reproduce

With the following commands, you can reproduce most of our experiments.

### Streaming Demo
```bash
#HiP
CUDA_VISIBLE_DEVICES=0,1 VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_K=512 HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYERS=4 python hip/main/model_eval.py --job stream_demo --model vllm_llama3.1_8b_instruct --stride 32000 --input samples/32k.md --batch_size 3 --max_tokens 512

#vLLM
CUDA_VISIBLE_DEVICES=0,1 VLLM_ATTENTION_BACKEND=FLASH_ATTN python hip/main/model_eval.py --job stream_demo --model vllm_llama3.1_8b_instruct --stride 32000 --input samples/32k.md --batch_size 3 --max_tokens 512
```
### Generation Demo
```bash
VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_K=512 HIP_REFRESH_INTERVAL=8 BENCHMARK_RUNNER=1 HIP_DENSE_LAYERS=4 python hip/main/model_eval.py --model vllm_qwen7b --job stream --method hip --k 512 --block_size_q 32 --block_size_k 2 --input samples/32k.md --max_tokens 128 --stride 32000 --batch_size 4
```

### Interactive Generation Demo
```bash
# NOTE: this demo use eager mode. this must be much slower than ideal speed due to single batch and JIT compilation.
python hip/main/model_eval.py --model llama3.1_8b --job stream --method hip --k 512 --block_size_q 32 --block_size_k 2
```

### Attention Latency Microbenchmarks
```bash
python hip/models/hip_attention/attention1_block_gpu.py --method hip --k 512 --block_size_q 32 --block_size_k 2 --query_size 32 --dups 16 --batch_size 32 --head_size 40 --hidden_size 128 --samples 200

python hip/models/hip_attention/attention1_block_gpu.py --method none --query_size 32 --dups 16 --batch_size 32 --head_size 40 --hidden_size 128 --samples 200

python hip/models/hip_attention/attention1_block_gpu.py --method flash --query_size 32 --dups 16 --batch_size 32 --head_size 40 --hidden_size 128 --samples 200
```

### Wikitext2 Perplexity
```bash
# HiP
python hip/main/model_eval.py --job ppl --method hip --k 512 --block_size_q 32 --block_size_k 2 --overwrite --model llama3.1_8b --stride 8192

# StreamingLLM
python hip/main/model_eval.py --job ppl --method streaming_llm --k 512 --overwrite --model llama3.1_8b --stride 8192

# HyperAttention
python hip/main/model_eval.py --job ppl --method hyper_attention --overwrite --model llama3.1_8b --stride 8192 --dense_layers 6

# vanilla
python hip/main/model_eval.py --job ppl --method none --k 512 --block_size_q 32 --block_size_k 2 --overwrite --model llama3.1_8b --stride 8192
```

### LongBench

Check the codebase in [gmlwns2000/LongBench-hip](https://github.com/gmlwns2000/LongBench-hip)

```bash
# HiP
HIP_K=512 HIP_DENSE_LAYERS=3 HIP_REFRESH_INTERVAL=8 VLLM_ATTENTION_BACKEND=HIP_ATTN CUDA_VISIBLE_DEVICES=0 ATTENTION_METHOD=hip python pred.py --method hip --k 512 --model llama3.1-8b-chat-128k
python eval.py --method hip --k 512 --modl llama3.1-8b-chat-128k

# vLLM, try XFORMERS if FLASH_ATTN not works.
VLLM_ATTENTION_BACKEND=FLASH_ATTN HIP_K=512 ATTENTION_METHOD=none CUDA_VISIBLE_DEVICES=0 python pred.py --model llama3.1-8b-chat-128k --method none --k 512
python eval.py --method none --k 512 --modl llama3.1-8b-chat-128k

# StreamingLLM
HIP_K=512 ATTENTION_METHOD=streaming_llm CUDA_VISIBLE_DEVICES=0 python pred.py --model llama3.1-8b-chat-128k --method streaming_llm --k 512
python eval.py --method streaming_llm --k 512 --modl llama3.1-8b-chat-128k

# use llama3.1-8b-chat-128k for reproduce llama3.1
```

### RULER

Check the codebase in [gmlwns2000/RULER-hip](https://github.com/gmlwns2000/RULER-hip)

```bash
SERVER_PORT=5782 BATCH_SIZE=1 SA_BLOCK_SIZE=64 HIP_DISABLE_AUTOTUNE=1 CUDA_VISIBLE_DEVICES=5 GPUS=1 VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_PREFILL_BQ=64 HIP_PREFILL_BK=2 HIP_SW=1024 HIP_NSINK=128 HIP_K=2048 HIP_USING_SNAP_KV=1 HIP_SNAP_KV_VERT_K=2048 HIP_SNAP_KV_DIAG_K=1024 HIP_BK_AFTER_MASK=16 HIP_RANDOM_MASK=0 HIP_DECODE_ALWAYS_DENSE=1 ./run.sh llama3.1-8b-chat-6 synthetic
```

### BookSum
```bash
CUDA_VISIBLE_DEVICES=0 python hip/main/model_eval.py --model llama13b_32k --job booksum --stride 32000 --max_tokens 256 --method streaming_llm --k 512 --name exp_name --overwrite

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_K=512 HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYERS=4 python hip/main/model_eval.py --model vllm_llama13b_32k --job booksum --stride 32000 --max_tokens 256 --method hip --k 512 --name exp_name --overwrite

CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN python hip/main/model_eval.py --model vllm_llama13b_32k --job booksum --stride 32000 --max_tokens 256 --method none --name exp_name --overwrite
```

## UVM Benchmark (no longer supported, you can mimic this kv offload runner)
```bash
BENCHMARK_RUNNER=1 CACHE_ENGINE='offload_v' VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYERS=4 HIP_K=512 CUDA_VISIBLE_DEVICES=0 python hip/main/model_eval.py --model vllm_qwen14b_gptq --job stream --batch_size 4 --input samples/16k.md --stride 22000 --max_tokens 32
```

## KV Offload Runner
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HIP_DISABLE_AUTOTUNE=1 
python hip/models/hip_attention/offload_runner/offload_runner.py --cache_backend uvm --kv_share 1 --method hip --offload-cache --batch_size 1 --sw 256 --k 512 --max_tokens 256 --input ./samples/32k.md --cache_size 4096 --refresh_interval 8 --offload_cache_method single_level
```

### Nsight-System
```bash
# with su
MODEL=vllm_luxia21.4b BATCH_SIZE=72 BACKEND=hip HIP_REFRESH_INTERVAL=8 /usr/local/cuda-12.2/bin/nsys profile --gpu-metrics-device all --cuda-graph-trace node --python-backtrace=cuda --gpu-metrics-frequency 10000 --output report_hip_luxia -t cuda -n true  ./scripts/bench_stream_1.sh

MODEL=vllm_luxia21.4b BATCH_SIZE=72 BACKEND=vllm HIP_REFRESH_INTERVAL=1 /usr/local/cuda-12.2/bin/nsys profile --gpu-metrics-device all --cuda-graph-trace node --python-backtrace=cuda --gpu-metrics-frequency 10000 --output report_vllm_luxia -t cuda -n true  ./scripts/bench_stream_1.sh
```

## Development Notes

### Developer Commands (RTX 4090)
```bash
BENCHMARK_RUNNER=1 CACHE_ENGINE='offload_v' VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYERS=4 HIP_K=1024 CUDA_VISIBLE_DEVICES=0 python hip/main/model_eval.py --model vllm_qwen14b_gptq --job stream --batch_size 4 --input samples/16k.md --stride 22000 --max_tokens 32

sudo /usr/local/cuda-12.2/bin/ncu --target-processes all -f -o profile ./scripts/bench_stream_1.sh

sudo /usr/local/cuda-12.2/bin/nsys profile -t cuda ./scripts/bench_stream_1.sh

sudo /usr/local/cuda-12.2/bin/nsys profile --gpu-metrics-device all --cuda-graph-trace node --python-backtrace=cuda --gpu-metrics-frequency 50000 --output report_hip_sys_17 -t cuda -n true --env-var FILENAME=16k,PYBIN=`which python`,BACKEND=hip ./scripts/bench_stream_1.sh

lm_eval --model hf --model_args pretrained=togethercomputer/LLaMA-2-7B-32K,load_in_4bit=True,attention_method=streaming_llm,hip_k=512 --tasks arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande,gsm8k --device cuda:0 --batch_size 1 --num_fewshot 5

sudo /usr/local/cuda-12.2/bin/nsys profile --gpu-metrics-device all --cuda-graph-trace node --python-backtrace=cuda --gpu-metrics-frequency 50000 --output report_hip_sys_17 -t cuda -n true ./scripts/bench_stream_1.sh

CUDA_VISIBLE_DEVICES=0,1 HIP_K=512 HIP_DENSE_LAYER=4 HIP_REFRESH_INTERVAL=8 VLLM_ATTENTION_BACKEND=HIP_ATTN python hip/main/model_eval.py --job stream_demo --model vllm_qwen7b --stride 32000 --input samples/32k.md --batch_size 3 --max_tokens 512

CUDA_VISIBLE_DEVICES=0,1 VLLM_ATTENTION_BACKEND=HIP_ATTN python hip/main/model_eval.py --job stream_demo --model vllm_qwen7b --stride 32000 --input samples/32k.md --batch_size 3 --max_tokens 512

python examples/openai_chat_image_stress.py --image-file="images/cherry_blossom.jpg" --model="microsoft/Phi-3-vision-128k-instruct" --endpoint="http://localhost:8888/v1" --token="token-blw7qUu6tFQeO9Ch5LVrIBWN3PEx2isaf4Xp" --num-workers 4 --num-seqs 32

MEASURE_PEAK_MEMORY=0 DISABLE_SAMPLING=1 BENCHMARK_RUNNER=1 VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_K=512 HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYERS=4 CUDA_VISIBLE_DEVICES=0,2 python3 -m vllm.entrypoints.openai.api_server --model microsoft/Phi-3-vision-128k-instruct --download-dir $HF_HOME --tensor-parallel-size 2 --kv-cache-dtype fp8_e5m2 --dtype half --gpu-memory-utilization 0.7 --max-model-len 32000 --max-num-seq 256 --trust-remote-code --image-input-type pixel_values --image-token-id -1 --image-input-shape "1008, 1344" --fake-image-input-shape "1, 16, 3, 336, 336" --image-feature-size 1921 --disable-log-request --max-seq-len-to-capture 32000 --swap-space 4 --port 8888

python examples/openai_chat_image_client.py --image-file="images/cherry_blossom.jpg" --model="microsoft/Phi-3-vision-128k-instruct" --endpoint="http://localhost:8888/v1" --token="token-blw7qUu6tFQeO9Ch5LVrIBWN3PEx2isaf4Xp" --max-tokens 512

VLLM_ATTENTION_BACKEND=HIP_ATTN CUDA_VISIBLE_DEVICES=1 python -c "import vllm; x=vllm.LLM('meta-llama/Meta-Llama-3.1-8B', enforce_eager=True, gpu_memory_utilization=0.7, max_model_len=1024).generate('User: hello, world\nAssistant: '); print(x[0].outputs[0].text)"
```

### Example Training Command
```bash
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --num_processes=4 --main_process_port 29501 hip/trainer/hip_trainer_hf.py --method hip --block_size_q 32 --block_size_k 2 --k 512 --lora_r 256 --dataset openwebtext --dense_layers 4 --name bs16_warmup10_dq2k --dense_queries 2048 --seq_len 32768 --disable_kd --sparsity_reg 0.01 --gradient_accumulation_steps 4 --warmup_steps 10 --model giraffe13b --using_deepspeed
```

## Citation

```
@misc{lee2024_hip_attention,
      title={A Training-free Sub-quadratic Cost Transformer Model Serving Framework With Hierarchically Pruned Attention}, 
      author={Heejun Lee and Geon Park and Youngwan Lee and Jaduk Suh and Jina Kim and Wonyoung Jeong and Bumsik Kim and Hyemin Lee and Myeongjae Jeon and Sung Ju Hwang},
      year={2024},
      eprint={2406.09827},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.09827}, 
}
```
