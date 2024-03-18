# TimberAttention

## How to clone the repository

```bash
git clone <this-repo-url> lightweight-lm
cd lightweight-lm
git submodule update --init --remote --recursive  # pull submodules
````

## How to build Docker

Run commands below:

```bash
cd third_party/vllm-timber
docker build . --build-context timber=../.. --target vllm-openai --tag vllm/vllm-openai
```

## Running Docker

After building the container, run commands below (change `--gpus` and `--tensor-parallel-size` according to your environment):

```bash
docker run --runtime nvidia --rm -it --gpus 0,1,2,3 --ipc=host \
       -v ~/.cache/huggingface/:/root/.cache/huggingface \
       -e 'PAGED_ATTENTION_BACKEND=timber' \
       -e 'PROMPT_ATTENTION_BACKEND=timber' \
       vllm/vllm-openai \
            --model togethercomputer/LLaMA-2-7B-32K \
            --tensor-parallel-size 4 \
            --kv-cache-dtype fp8_e5m2 \
            --dtype half \
            --gpu-memory-utilization 0.8
```
----

## Setup without docker
```bash
conda create --name llm python=3.11
conda activate llm
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install -c conda-forge cupy cuda-version=11.8
cd lightweight-lm
pip install -e .
pip install numba packaging
cd third_party/vllm-timber
pip install -r requirements-build.txt
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-build-isolation --verbose
```

## Running without docker
```bash
PAGED_ATTENTION_BACKEND=timber \  
PROMPT_ATTENTION_BACKEND=timber \
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m vllm.entrypoints.openai.api_server \
--model togethercomputer/LLaMA-2-7B-32K \
--download-dir "/tmp/$(whoami)" \
--tensor-parallel-size 2 \
--kv-cache-dtype fp8_e5m2 \
--dtype half \
--gpu-memory-utilization 0.8
```


## vllm + Qwen's Dynamic-NTK

add the following content in Qwen's `config.json`. 

<<<<<<< HEAD
# for vllm dev
HIP_DENSE_LAYERS=3 HIP_K=256 CUDA_VISIBLE_DEVICES=0 python timber/main/llama_eval.py --model vllm_llama1b --job stream --batch_size 4 --input sample4k.md --stride 4096
=======
- `seq_length` is the threshold for activating NTK, default 8192 (the same as Qwen).
- `factor` does not affect the logic of dynamic-ntk. It is used by vllm to calculate the maximum input length for model. If it is set to 1, warnings will occur if input is longer than 8192. Setting to 4 may be enough.

```
"rope_scaling": {
    "type": "dynamic-qwen",
    "seq_length": 8192,
    "factor": 4.0
}
>>>>>>> geon-dev
```