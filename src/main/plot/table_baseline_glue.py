# BigBird removed due to implementation error. The value is dummy

data_mnli = {
    'SEA (Ours)': 84.6,
    'Cosformer': 82.7,
    'Reformer': 82.3,
    # 'B.B': 80.0,
    'ScatterBrain': 80.4,
    # 'Long-': 78.8,
    'Sinkhorn': 81.9,
    'Synthesizer': 75.5,
    'Performer': 74.66,
}
data_cola = {
    'Ours': 82.1,
    'Re-': 81.7,
    # 'B.B': 80.0,
    'S.B': 79.5,
    'Long-': 78.8,
    'Sink.': 76.0,
    'Per-': 74.0,
    'Synth.': 74.0,
    # 'Cos-': 62.2,
}
data_mrpc = {
    'Ours': 82.1,
    'Re-': 81.7,
    # 'B.B': 80.0,
    'S.B': 79.5,
    'Long-': 78.8,
    'Sink.': 76.0,
    'Per-': 74.0,
    'Synth.': 74.0,
    # 'Cos-': 62.2,
}
data_sst2 = {
    'Ours': 82.1,
    'Re-': 81.7,
    # 'B.B': 80.0,
    'S.B': 79.5,
    'Long-': 78.8,
    'Sink.': 76.0,
    'Per-': 74.0,
    'Synth.': 74.0,
    # 'Cos-': 62.2,
}

data = {
    'MNLI': data_mnli,
    # 'CoLA': data_cola,
    # 'MRPC': data_mrpc,
    # 'SST2': data_sst2,
}

my_key = 'Ours'

col_keys = list(data_mnli.keys())

cell_data = []
for ds in data.keys():
    cell_data.append([f'{ds} (Acc.)'] + [data[ds][c] for c in col_keys])
    # cell_data.append([f'{ds} (Rel. Acc.)'] + [data[ds][c] / data[ds][my_key] * 100 for c in col_keys])
just_width = 13
just_width_header = 20
def format(lst):
    return [f'{i:.1f}'.rjust(just_width) for i in lst]
table_data = ""
for row in cell_data:
    if isinstance(row, list):
        name, row_data = row[0], row[1:]
        table_data += '&'.join([name.rjust(just_width_header)]+format(row_data))
        table_data += '\\\\\n'
    elif isinstance(row, str):
        table_data += row
    else:
        raise Exception()


table_cells = "|".join(['c'] * len(cell_data[0]))
table_header = "&".join(["".rjust(just_width_header)] + [k.rjust(just_width) for k in col_keys])
table = \
f"\\begin{{table}}[h]\n"\
f"\\caption{{Comparison with GLUE dataset among linear attention methods. Trained with same number of optimizer steps.}}\n"\
f"\\label{{table.baseline.glue}}\n"\
f"\\begin{{center}}\n"\
f"\\begin{{tabular}}{{{table_cells}}}\n"\
f"{table_header}\\\\\n"\
f"\\hline\n"\
f"{table_data}"\
f"\\end{{tabular}}\n"\
f"\\end{{center}}\n"\
f"\\end{{table}}"

print(table)