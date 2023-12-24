import json
import math
import os
from ..benchmark_bert import BASELINES

path = './plots/main/benchmark_bert/data.json'
with open(path, 'r') as f:
    data = json.load(f)
print(data)

assert len(BASELINES) == len(data['vram_baseline']), f"{len(BASELINES)} == {len(data['vram_baseline'])}"

METHOD2NAME = {
    'none': 'Vanilla',
    'performer': 'Performer',
    'sinkhorn': 'Sinkhorn',
    'cosformer': 'Cosformer',
    'reformer': 'Reformer',
    'scatterbrain': 'Scatterbrain',
    'synthesizer': 'Synthesizer',
    'flash': 'FlashAttention',
}

ts = data['ts'] + ['Avg.']
ks = data['ks']

just_width = max([len(s) for s in BASELINES])
def format(lst):
    return [(f'{i:.2f}' if not math.isnan(i) else 'OOM').rjust(just_width) for i in lst]
def avg(lst):
    return sum(lst) / len(lst)

def render_table(key_baseline, key_perlin):
    cell_data = []
    for ib, name in enumerate(BASELINES):
        cell_data.append([name]+data[key_baseline][ib]+[avg(data[key_baseline][ib])])
        if name == 'none':
            cell_data.append('\\midrule\n')
    N_XFORMERS = (len(BASELINES)-1)
    cell_data[-N_XFORMERS:] = sorted(cell_data[-N_XFORMERS:], key=lambda row: row[-1])
    cell_data.append('\\midrule\n')
    for ik, k in enumerate(ks):
        cell_data.append([f'Ours (k={int(k)})'] + data[key_perlin][ik]+[avg(data[key_perlin][ik])])

    # print(cell_data)

    table_cells = "|".join(['c']*(len(ts)+1))
    table_header = " & ".join(["".rjust(just_width)]+[(f'{i:,}' if not isinstance(i, str) else i).rjust(just_width) for i in ts])
    table_data = ""
    for row in cell_data:
        if isinstance(row, list):
            name, row_data = row[0], row[1:]
            name = METHOD2NAME.get(name, name)
            table_data += ' & '.join([name.rjust(just_width)]+format(row_data))
            table_data += '\\\\\n'
        elif isinstance(row, str):
            table_data += row
        else:
            raise Exception()

    table = \
    f"\\begin{{table}}[t]\n"\
    f"\\caption{{{key_baseline}-{key_perlin}}}\n"\
    f"\\label{{table.benchmark_bert.{key_baseline}-{key_perlin}}}\n"\
    f"\\begin{{center}}\n"\
    f"\\begin{{tabular}}{{{table_cells}}}\n"\
    f"{table_header}\\\\\n"\
    f"\\midrule\n"\
    f"{table_data}"\
    f"\\end{{tabular}}\n"\
    f"\\end{{center}}\n"\
    f"\\end{{table}}"\
    
    print('-'*80)
    print()
    print(table)
    print()
    print('-'*80)

render_table('latencies_baseline', 'latencies_perlin')
render_table('vram_baseline', 'vram_perlin')