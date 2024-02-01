import subprocess
import os, json

def samples():
    block_size = 4
    query_size = 1
    k = 512
    batch_sizes = {
        1: 128,
        4: 128,
        5: 96,
        6: 80,
        7: 70,
        8: 60,
        9: 48,
        10: 48,
        11: 42,
        12: 40,
        13: 32,
        14: 32,
        16: 32,
    }
    num_samples = 200
    
    batch_size = batch_sizes[1]
    results = {}
    dups = range(1, 17)
    for dup in dups:
        if dup in batch_sizes:
            batch_size = batch_sizes[dup]
            
        subprocess.call([
            'python', 'src/models/tree_attention/attention1_block_gpu.py',
            '--method', 'tree',
            '--block_size', str(block_size),
            '--k', str(k),
            '--query_size', str(query_size),
            '--dups', str(dup),
            '--batch_size', str(batch_size),
            '--samples', str(num_samples),
        ])
        with open('./cache/attention1_block_gpu/result.json', 'r') as f:
            latency_tree = json.load(f)['mean']
        
        subprocess.call([
            'python', 'src/models/tree_attention/attention1_block_gpu.py',
            '--method', 'none',
            '--block_size', str(block_size),
            '--k', str(k),
            '--query_size', str(query_size),
            '--dups', str(dup),
            '--batch_size', str(batch_size),
            '--samples', str(num_samples),
        ])
        with open('./cache/attention1_block_gpu/result.json', 'r') as f:
            latency_base = json.load(f)['mean']
        
        seq_len = dup * 1024
        results[f's{seq_len}'] = {
            'latency_tree': latency_tree,
            'latency_base': latency_base,
            'batch_size': batch_size,
            'query_size': query_size,
            'seq_len': seq_len,
            'dups': dup,
            'k': k,
            'speedup': latency_base / latency_tree,
        }
    
    os.makedirs('./saves/seqlen_speed_report', exist_ok=True)
    path = './saves/seqlen_speed_report/result.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('dumped', path)

def plot():
    path = './saves/seqlen_speed_report/result.json'
    with open(path, 'r') as f:
        data = json.load(f)

def main():
    samples()
    plot()

if __name__ == '__main__':
    main()