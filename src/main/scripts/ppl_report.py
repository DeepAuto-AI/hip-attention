import json
import math
import sys, subprocess, os, itertools, tqdm
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import pypareto

os.environ['PYTHONPATH'] = './'

block_sizes = [1, 2, 4, 8, 16]
ks = [128, 256, 512, 1024]

def samples():
    results = {}
    for block_size, k in tqdm.tqdm(
        list(itertools.product(block_sizes, ks)), 
        desc='exam', dynamic_ncols=True
    ):
        print(f'ppl measure b{block_size}, k{k}')
        subprocess.call([
            'python', 'src/main/llama_eval.py', 
            '--method', 'tree',
            '--stride', '4096',
            '--block_size', str(block_size),
            '--k', str(k),
        ])
        with open('./cache/llama_eval/ppl.json', 'r') as f:
            ppl = json.load(f)['ppl']
        print(f'ppl measured {ppl} (b{block_size}, k{k})')
        results[f'b{block_size}_k{k}'] = {
            'block_size': block_size,
            'k': k,
            'num_blocks': math.ceil(k / block_size),
            'ppl': ppl,
        }
    
    os.makedirs('./saves/ppl_report', exist_ok=True)
    with open('./saves/ppl_report/report.json', 'w') as f:
        json.dump(results, f, indent=2)

def by_value(a, b):
    if isinstance(a, (tuple, list)):
        return pypareto.Domination.EQUAL

    if a > b:
        return pypareto.Domination.GREATER
    elif a < b:
        return pypareto.Domination.LESS
    else:
        return pypareto.Domination.EQUAL

def plots():
    baseline_ppl = 5.59
    
    with open('./saves/ppl_report/report.json', 'r') as f:
        data = json.load(f)
    
    entries = list(data.values())
    xs = [] # num blocks
    ys = [] # ppl
    
    for entry in entries:
        xs.append(entry['num_blocks'])
        ys.append(entry['ppl'])
    
    pts = list(zip(xs, ys, map(lambda x: (x,), range(len(xs)))))
    chain = pypareto.Comparison(by_value, pypareto.MaxMinList(pypareto.MaxMin.MIN, pypareto.MaxMin.MIN, pypareto.MaxMin.MIN)).as_chain()
    pts = chain.split_by_pareto(pts)[0]
    xs_front = [pt[0] for pt in pts]
    ys_front = [pt[1] for pt in pts]
    idxs_front = [pt[2][0] for pt in pts]
    
    plt.figure(figsize=(5, 4))
    
    sns.lineplot(x=xs_front, y=ys_front)
    for idx in range(len(idxs_front)):
        plt.annotate(
            f'k:{entries[idxs_front[idx]]["k"]}, b:{entries[idxs_front[idx]]["block_size"]}', 
            (xs_front[idx], ys_front[idx]),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=9,
        )
    
    plt.axhline(baseline_ppl, color='#555', linestyle='--', linewidth=1)
    sns.scatterplot(x=xs, y=ys)
    
    plt.title('Perplexity / Num. Blocks')
    plt.xlabel('Num. Blocks')
    plt.ylabel('PPL. (w/o train)')
    
    plt.savefig('./saves/ppl_report/plot_ppl_report.png', dpi=200, bbox_inches='tight')
    plt.savefig('./saves/ppl_report/plot_ppl_report.pdf', dpi=200, bbox_inches='tight')
    print('saved', './saves/ppl_report/plot_ppl_report.png')

def main():
    # samples()
    plots()

if __name__ == '__main__':
    main()