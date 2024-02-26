# TimberAttention

## How to build Docker

Run commands below:

```bash
git clone <this-repo-url> lightweight-lm
cd lightweight-lm/third_party/vllm-timber
docker build . --build-context timber=../.. --target vllm-openai --tag vllm/vllm-openai
```

## Running Docker

After building the container, run commands below (change `--gpus` and `--tensor-parallel-size` according to your environment):

```bash
docker run --runtime nvidia --rm -it --gpus 0,1,2,3 --ipc=host \
       -v ~/.cache/huggingface/:/root/.cache/huggingface \
       -e 'PAGED_ATTENTION_BACKEND=timber' \
       vllm/vllm-openai \
            --model togethercomputer/LLaMA-2-7B-32K \
            --tensor-parallel-size 4 \
            --kv-cache-dtype fp8_e5m2 \
            --dtype half \
            --gpu-memory-utilization 0.8
```
----

python src/trainer/timber_trainer.py --disable_kd --lora_r 512 --batch_size 1 --block_size 8 --k 512 --init_checkpoint ./saves/dev/llama32k-wikitext103-4096-block8-k512-epoch-00-step-8400.pth --dataset booksum --using_fsdp --max_steps 10000

CUDA_VISIBLE_DEVICES=0 PAGED_ATTENTION_BACKEND=timber BENCHMARK_PAGED_ATTENTION=1 FORCE_SINGLE_LAYER=0 python timber/main/llama_eval.py --model vllm_llama32k --job stream --batch_size 1
