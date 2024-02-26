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
