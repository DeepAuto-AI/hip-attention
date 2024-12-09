import os
import json
import subprocess
import torch

n_gpus = torch.cuda.device_count()
model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
output_file = 'cache/sglang/result.json'
chunked_prefill_size = 8192

def reset_result():
    os.makedirs('cache/sglang', exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)
reset_result()

def run_sample(seq_len: int, batch_size: int, envs: dict):
    reset_result()
    subprocess.call(
        (
            f"python -m sglang.bench_latency --model-path {model} "
            f"--batch {batch_size} --input-len {seq_len * 1024} --output-len {256} --tp-size {n_gpus} "
            f"--mem-fraction-static 0.7 --enable-p2p-check --chunked-prefill-size {chunked_prefill_size} "
            f"--max-prefill-tokens {chunked_prefill_size} --result-filename {output_file}"
        ).split(), 
        env=envs, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                return None, None
            prefill = data['prefill_throughput']
            decode = data['p90_decode_throughput']
            return prefill, decode
    else:
        return None, None

seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
batch_sizes = [1, 8, 32]

seq_lens = [1, 64, 256]
batch_size = [1, 8, 32]

parent_envs = os.environ.copy()
parent_envs.update({
    'POPULATION_FILE':'none',
    'HIP_DISABLE_AUTOTUNE': '1',
    'SRT_MAX_BATCH': '32',
})
test_envs = {
    'HIP G3E': {
        'SRT_ATTENTION_BACKEND': 'HIP_ATTN',
        'HIP_EXTEND': '1',
        'EXTEND_LEN': '512',
    },
    'HIP G2': {
        'SRT_ATTENTION_BACKEND': 'HIP_ATTN',
        'HIP_EXTEND': '0',
        'HIP_K': '1024',
    },
    'SRT': {
        'SRT_ATTENTION_BACKEND': 'SRT',
    },
}

results = {
    'settings': {
        'seq_lens': seq_lens,
        'batch_sizes': batch_sizes,
    }
}
for test_exp_name, test_exp_update in test_envs.items():
    print('='*80)
    envs = parent_envs.copy()
    envs.update(test_exp_update)
    
    print(f'test {test_exp_name}')
    
    data_prefill = []
    data_decode = []
    for batch_size in batch_sizes:
        row_prefill = []
        row_decode = []
        for seq_len in seq_lens:
            prefill_throughput, decode_throughput = run_sample(
                seq_len=seq_len,
                batch_size=batch_size,
                envs=envs,
            )
            row_prefill.append(prefill_throughput)
            row_decode.append(decode_throughput)
            print(f'[exp={test_exp_name}, batch_size={batch_size}, seq_len={seq_len}] prefill={prefill_throughput:.2f}, decode={decode_throughput:.2f}')
        data_prefill.append(row_prefill)
        data_decode.append(row_decode)
    results[test_exp_name] = {
        'prefill': data_prefill,
        'decode': data_decode,
    }