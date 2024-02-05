import subprocess
import os, json

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

dups = range(1, 17)

def samples():
    block_size = 16
    block_size_k = 2
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
    
    batch_size = max(list(batch_sizes.values()))
    results = {}
    for dup in dups:
        if dup in batch_sizes:
            batch_size = batch_sizes[dup]
            
        subprocess.call([
            'python', 'src/models/tree_attention/attention1_block_gpu.py',
            '--method', 'tree',
            '--block_size_q', str(block_size),
            '--block_size_k', str(block_size_k),
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
    
    xs = []
    ys_base = []
    ys_tree = []
    ys_speedup = []
    
    for dup in dups:
        entry = data[f's{dup * 1024}']
        xs.append(entry['seq_len'])
        ys_base.append(entry['latency_base'] / entry['batch_size'] * 1000)
        ys_tree.append(entry['latency_tree'] / entry['batch_size'] * 1000)
        ys_speedup.append(entry['speedup'])
    
    plt.figure(figsize=(5,4))
    
    sns.lineplot(x=xs, y=ys_base, label='baseline')
    sns.lineplot(x=xs, y=ys_tree, label='timber')
    plt.legend()
    plt.title('Single Query Latency')
    plt.xlabel('Seq. Length')
    plt.ylabel('Latency (us)')
    plt.xlim(2048, 16*1024)
    
    fig_path = './saves/seqlen_speed_report/plot_seqlen_latency'
    plt.savefig(fig_path + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(fig_path + '.pdf', dpi=200, bbox_inches='tight')
    print(f'saved {fig_path}.png')
    
    plt.figure(figsize=(5,4))
    
    plt.title('Single Query Speedup (k=512, b=4)')
    sns.lineplot(x=xs, y=ys_speedup, label='speedup')
    plt.xlabel('Seq. Length')
    plt.ylabel('Speedup')
    plt.xlim(2048, 16*1024)
    
    fig_path = './saves/seqlen_speed_report/plot_seqlen_speedup'
    plt.savefig(fig_path + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(fig_path + '.pdf', dpi=200, bbox_inches='tight')
    print(f'saved {fig_path}.png')

def main():
    samples()
    plot()

if __name__ == '__main__':
    main()