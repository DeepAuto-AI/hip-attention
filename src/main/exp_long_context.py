import copy
import itertools
import os, sys, subprocess, json
import time

"""
__LOAD_PREFIX=dev5
DYNAMIC_K=128
QUERY_SKIPS=1
LOAD_AFTER_RESIZE=1
__CONTEXT=4096 
__STRIDE=4096 
python -m src.trainer.perlin_trainer \
    --model opt-125m \
    --method perlin \
    --dataset wikitext2 \
    --k 128 \
    --predictor-length 96 \
    --performer-nb-feature-factor 8.0 \
    --context-output-method mix \
    --load-checkpoint auto \
    --eval
"""

load_prefix = 'dev5'
k = 128
pw = 96

dynamic_ks = [96, 104, 112, 120, 128]
# query_skips = [1, 2, 4, 8, 16]
query_skips = [16, 8, 4, 2, 1]

def long_sleep(sec):
    last_tick = time.time()
    elapsed = 0
    while elapsed < sec:
        time.sleep(0.1)
        elapsed += time.time() - last_tick
        last_tick = time.time()
        print(f'sleeping ({elapsed:.1f} sec)\r', end='', flush=True)
    print()

def samples():
    from .benchmark_opt_ablation import exam_config, BenchConfig
    
    options = itertools.product(query_skips, dynamic_ks)
    data = {}
    for qskip, dks in options:
        envs = copy.deepcopy(os.environ)
        envs.update({
            '__LOAD_PREFIX': load_prefix,
            'DYNAMIC_K': str(int(dks)),
            'QUERY_SKIPS': str(int(qskip)),
            'LOAD_AFTER_RESIZE': '1',
            '__CONTEXT': '4096',
            '__STRIDE': '4096',
            # 'CUDA_VISIBLE_DEVICES': '0',
        })
        cmd = \
            f'python -m src.trainer.perlin_trainer '\
            f'--model opt-125m '\
            f'--method perlin '\
            f'--dataset wikitext2 '\
            f'--k {int(k)} '\
            f'--predictor-length {int(pw)} '\
            f'--performer-nb-feature-factor 8.0 '\
            f'--context-output-method mix '\
            f'--load-checkpoint auto '\
            f'--eval'
        subprocess.call(cmd.split(' '), env=envs)
        with open('./cache/perlin_trainer/last_ppl.txt', 'r') as f:
            text = f.read()
            text = text.strip().replace('\n', '')
            ppl = float(text)
        os.environ.update({
            'DYNAMIC_K': str(int(dks)),
            'QUERY_SKIPS': str(int(qskip)),
        })
        long_sleep(60)
        latency, mem = exam_config(BenchConfig(
            method='perlin',
            seq_len=4096,
            k=k,
            w=pw,
            trace=False,
            causal=True
        ))
        # long_sleep(120)
        latency = latency * 1000
        mem = mem / (1024 ** 2)
        sample = {
            'mem': mem,
            'latency': latency,
            'query_skip': qskip,
            'dynamic_k': dks,
            'ppl': ppl,
        }
        print(sample)
        data[f'{dks},{qskip}'] = sample
    
    os.makedirs('./plots/exp_long_context', exist_ok=True)
    with open('./plots/exp_long_context/data.json', 'w') as f:
        json.dump(data, f, indent=2)

r"""
\begin{table}[h]
\caption{...}
\label{table.baseline.glue}
\begin{center}
\begin{tabular}{l|c|ccccccc}
\toprule
Metric &\textcolor{gray}{Vanilla}& \textbf{SEA (Ours)}&Cosformer&Reformer&Sinkhorn&Synthesizer&Performer\\
\midrule
Accuracy&\textcolor{gray}{84.1}&84.0& 82.7& 82.5& 81.9& 75.5& 74.7\\
Memory (MB) & \textcolor{gray}{9.00} & 17.17& 10.88& 88.36& 9.39& 8.25& 14.76\\
Latency ($\mu$s)& \textcolor{gray}{238} & 701 & 242& 900& 152& 181& 320\\
\bottomrule
\end{tabular}
\end{center}
\end{table}
"""
def wrap_color(v, vmin, vmax, cmin=(64, 212, 19), cmax=(227, 25, 72)):
    #\textcolor[RGB]{62,114,196} {Medium Blue}
    x = min(1, max((v-vmin) / (vmax-vmin), 0))
    c = [int(t[0] * x + t[1] * (1-x)) for t in zip(cmax, cmin)]
    return f'\\textbf{{\\textcolor[RGB]{{{c[0]}, {c[1]}, {c[2]}}}{{{v:.2f}}}}}'

def render_table():
    with open('./plots/exp_long_context/data.json', 'r') as f:
        data = json.load(f)
    
    print(data)
    
    print('-'*80)
    print(r'\begin{table}[h]')
    print(r'\label{table.exp_long_context}')
    print(
        r'\caption{The trade off on long context experiment on Wikitext2 using post training compression techniques: '
        r'Query skipping and dynamic k control. Each entry of table shows \textcode{PPL(ms/MB)}. '
        r'Each values are colored with green and red. Better values are more green, and worse values are more red. }'
    )
    print(r'\begin{center}')
    print(r'\resizebox{1.0\linewidth}{!}{')
    print(r'\begin{tabular}{r|c|c|c|c|c}')
    print(r'\toprule')
    print(f'\\backslashbox{{\\tiny Query Skips}}{{\\tiny Dynamic-k}} & {" & ".join([str(x) for x in dynamic_ks])} \\\\')
    print(r'\midrule')
    for iq, qskip in enumerate(query_skips):
        line = f'{qskip}'
        for ds in dynamic_ks:
            sample = data[f'{ds},{qskip}']
            line += f' & {wrap_color(sample["ppl"], 21.5, 24.0)} ({{\\small {wrap_color(sample["latency"], 14, 21)} / {wrap_color(sample["mem"], 450, 480)}}})'
        line += '\\\\'
        print(line)
        if iq < (len(query_skips) - 1):
            print(r'\midrule')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'}')
    print(r'\end{center}')
    print(r'\end{table}')
    print('-'*80)
    
    print()
    print()
    
    print('-'*80)
    print('|' + '|'.join(['   '] + [str(x) for x in dynamic_ks]) + '|')
    print('---'.join(['|',] * (len(dynamic_ks) + 2)))
    for qskip in query_skips:
        entries = [f'$\\textbf{{{qskip}}}$']
        for ds in dynamic_ks:
            sample = data[f'{ds},{qskip}']
            entries.append(f'${wrap_color(sample["ppl"], 21.5, 24.0)}$(${wrap_color(sample["latency"], 14, 21)}$/${wrap_color(sample["mem"], 450, 480)}$)')
        print('|' + '|'.join(entries) + '|')
    print('-'*80)

def main():
    # samples()
    render_table()
    
if __name__ == '__main__':
    main()