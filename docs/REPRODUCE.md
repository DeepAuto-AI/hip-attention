# How to reproduce

## v1.2.0 (Gen3; Codename: InfiniteHiP; Under review version)

Following experiment commands are should be done after install `hip` and `sglang`.

### Server Side Setup

```bash
# ----- Env Vars ----- #
export SRT_PORT=32320
export SEQ_LEN=128
export CONTEXT_LENGTH=$(( $SEQ_LEN * 1024 ))

# ----- Server Side ----- #
python -m sglang.launch_server \
    # Adjust this model to match your client side
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 16384 \
    --context-length $CONTEXT_LENGTH \
    --max-total-tokens $CONTEXT_LENGTH \
    --max-running-requests 1 \
    --cuda-graph-bs 1 \
    --enable-hip-attention

    # if running out of GPU memory, you can turn on InfiniteHiP's offloading
    --enable-hip-offload \
    --hip-max-mask-cache-token-size 32000 \ # adjust this
    --hip-max-sa-cache-token-size 5000      # and this to fit your memory
```

### InfiniteBench

```bash
# ----- Env Vars ----- #
# ... addiontal to server env args ...
export EXPERIMENT_NAME=dev
export TASK=longbook_choice_eng

# ----- Server Side ----- #
# ... Check server side setup section ...

# ----- Client Side ----- #
git clone git@github.com:gmlwns2000/InfiniteBench-hip.git
cd InfiniteBench-hip/src

# > Run experiment (generation)
export SGLANG_PORT=$SRT_PORT
export USING_SGLANG=1
export SEQ_LEN=SEQ_LEN

# For Llama 3.1
python eval_llama3.py \
    --task $TASK \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --model_name $EXPERIMENT_NAME;

# For Gemma 2
export IS_GEMMA=1
python eval_llama3.py \
    --task $TASK \
    --model_path google/gemma-2-9b-it \
    --model_name $EXPERIMENT_NAME;

# For EXAONE3 and 3.5
export IS_EXAONE=1
python eval_llama3.py \
    --task $TASK \
    --model_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --model_name $EXPERIMENT_NAME;

# > Run evaluation

# For Llama 3.1
python compute_scores.py --task all --model_name llama3-$SEQ_LEN-$EXPERIMENT_NAME

# For Gemma 2
python compute_scores.py --task all --model_name gemma2-$SEQ_LEN-$EXPERIMENT_NAME

# For Exaone 3 and 3.5
python compute_scores.py --task all --model_name exaone3-$SEQ_LEN-$EXPERIMENT_NAME
```

### LongBench

```bash
# ----- Env Vars ----- #
# ... addiontal to server env args ...
export EXPERIMENT_NAME=dev

# ----- Server Side ----- #
# ... Check server side setup section ...

# ----- Client Side ----- #
git clone git@github.com:gmlwns2000/LongBench-hip.git
cd LongBench-hip

# > Run generation
export IS_INFLLM=1
export SGLANG_ENDPOINT="http://localhost:$SRT_PORT/"
python pred.py --model sglang-256k --name $EXPERIMENT_NAME

# > Run evaluation
python eval.py --model sglang-256k --name $EXPERIMENT_NAME
```

## v1.1.0, v1.0.0 (Gen1 and Gen2; Codename: HiP; ICLR2025, NIPS2025 submitted version)

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

### UVM Benchmark (no longer supported, you can mimic this kv offload runner)
```bash
BENCHMARK_RUNNER=1 CACHE_ENGINE='offload_v' VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_REFRESH_INTERVAL=8 HIP_DENSE_LAYERS=4 HIP_K=512 CUDA_VISIBLE_DEVICES=0 python hip/main/model_eval.py --model vllm_qwen14b_gptq --job stream --batch_size 4 --input samples/16k.md --stride 22000 --max_tokens 32
```

### KV Offload Runner
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
