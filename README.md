# TimberAttention

## How to Install

```bash
pip install -e .
```

## Note

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer/timber_trainer.py --batch_size 1 --gradient_accumulation_steps 2 --dataset wikitext2 --lora_r 512 --max_steps 10000 --block_size 4 --k 256

python src/trainer/timber_trainer.py --disable_kd --lora_r 512 --batch_size 1 --block_size 8 --k 512 --init_checkpoint ./saves/dev/llama32k-wikitext103-4096-block8-k512-epoch-00-step-8400.pth --dataset booksum --using_fsdp --max_steps 10000

CUDA_VISIBLE_DEVICES=0 PROMPT_ATTENTION_BACKEND=1 PAGED_ATTENTION_BACKEND=timber BENCHMARK_PAGED_ATTENTION=1 FORCE_SINGLE_LAYER=0 python timber/main/llama_eval.py --model vllm_llama32k --job stream --batch_size 1

TIMBER_DEBUG=0 CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=0 python timber/main/llama_eval.py --model llama32k --method timber --dense_queries 0 --k 512 --block_size_q 16 --block_size_k 2 --job ppl --stride 4096

python timber/models/timber_attention/attention1_block_gpu.py --method timber --k 1024 --block_size_q 32 --block_size_k 4 --dups 16 --batch_size 16
```