# TimberAttention

## How to Install

```bash
pip install -e .
```

## Note

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer/timber_trainer.py --batch_size 1 --gradient_accumulation_steps 2 --dataset wikitext2 --lora_r 512 --max_steps 10000 --block_size 4 --k 256

python src/trainer/timber_trainer.py --disable_kd --lora_r 512 --batch_size 1 --block_size 8 --k 512 --init_checkpoint ./saves/dev/llama32k-wikitext103-4096-block8-k512-epoch-00-step-8400.pth --dataset booksum --using_fsdp --max_steps 10000

CUDA_VISIBLE_DEVICES=0 PROMPT_ATTENTION_BACKEND=timber PAGED_ATTENTION_BACKEND=timber BENCHMARK_PAGED_ATTENTION=0 FORCE_SINGLE_LAYER=0 python timber/main/llama_eval.py --model vllm_llama32k --job stream --batch_size 1 --input sample_booksum.md --stride 16000

# for ppl
TIMBER_DEBUG=0 CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=0 python timber/main/llama_eval.py --model llama32k --method timber --dense_queries 0 --k 512 --block_size_q 32 --block_size_k 2 --job ppl --stride 8192

python timber/models/timber_attention/attention1_block_gpu.py --method timber --k 1024 --block_size_q 32 --block_size_k 4 --dups 16 --batch_size 16

HIP_K=256 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROMPT_ATTENTION_BACKEND=vllm PAGED_ATTENTION_BACKEND=vllm python3 benchmarks/benchmark_throughput.py --input-len 1000 --output-len 1000 --model Qwen/Qwen1.5-7B-Chat-GPTQ-Int4 --num-prompts 10 --dtype float16 --kv-cache-dtype fp8_e5m2 --max-model-len 2000

# for vllm dev
HIP_DENSE_LAYERS=3 HIP_K=256 CUDA_VISIBLE_DEVICES=0 python timber/main/llama_eval.py --model vllm_llama1b --job stream --batch_size 4 --input sample4k.md --stride 4096

# 4090 vllm dev
BENCHMARK_RUNNER=1 CACHE_ENGINE='offload_v' ATTENTION_BACKEND='hip' HIP_DENSE_LAYERS=4 HIP_K=1024 CUDA_VISIBLE_DEVICES=0 python timber/main/llama_eval.py --model vllm_qwen14b_gptq --job stream --batch_size 4 --input samples/16k.md --stride 22000 --max_tokens 32

sudo /usr/local/cuda-12.2/bin/ncu --target-processes all -f -o profile ./scripts/bench_stream_1.sh

sudo /usr/local/cuda-12.2/bin/nsys profile -t cuda ./scripts/bench_stream_1.sh

sudo /usr/local/cuda-12.2/bin/nsys profile --gpu-metrics-device all --cuda-graph-trace node --python-backtrace=cuda --gpu-metrics-frequency 50000 --output report_hip_sys -t cuda ./scripts/bench_stream_1.sh
```