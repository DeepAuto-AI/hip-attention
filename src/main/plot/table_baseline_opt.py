data = {
    'opt-125m': {
        'none': 25.16,
        'ours': 25.88,
        'reformer': 73.747,
        'performer': 49.791,
    },
    'opt-350m': {
        'none': 18.64,
        'ours': 19.99,
        'reformer': 65.25,
        'performer': 36.573,
    },
    'opt-1.3b': {
        'none': 13.29,
        'ours': 0.0,
        'reformer': 46.73,
        'performer': 30.81,
    },
    # 'opt-2.7b': {
    #     'none': 12.0,
    #     'ours': 0.0,
    #     'reformer': 0.0,
    #     'performer': 0.0,
    # },
    # 'opt-6.7b': {
    #     'none': 10.0,
    #     'ours': 0.0,
    #     'reformer': 0.0,
    #     'performer': 0.0,
    # }
}

aliases = {
    'opt-125m': '125M',
    'opt-350m': '350M',
    'opt-1.3b': '1.3B',
    'opt-2.7b': '2.7B',
    'opt-6.7b': '6.7B',
    'none': 'None',
    'ours': 'Ours (k=64)',
    'reformer': 'Reformer',
    'performer': 'Performer',
}
col_keys = ['opt-125m', 'opt-350m', 'opt-1.3b', ]
row_keys = ['none', 'ours', 'reformer', 'performer']

cell_data = []
for row_name in row_keys:
    cell_data.append([row_name] + [data[c][row_name] for c in col_keys])
just_width = 13
just_width_header = 10
def format(lst):
    return [f'{i:.1f}'.rjust(just_width) for i in lst]
table_data = ""
for row in cell_data:
    if isinstance(row, list):
        name, row_data = row[0], row[1:]
        table_data += '&'.join([aliases[name].rjust(just_width_header)]+format(row_data))
        table_data += '\\\\\n'
    elif isinstance(row, str):
        table_data += row
    else:
        raise Exception()

table_cells = "|".join(['c'] * len(cell_data[0]))
table_header = "&".join(["".rjust(just_width_header)] + [aliases[k].rjust(just_width) for k in col_keys])
table = \
f"\\begin{{table}}[h]\n"\
f"\\caption{{opt benchmark}}\n"\
f"\\label{{table.baseline.opt}}\n"\
f"\\begin{{center}}\n"\
f"\\begin{{tabular}}{{{table_cells}}}\n"\
f"{table_header}\\\\\n"\
f"\\hline\n"\
f"{table_data}"\
f"\\end{{tabular}}\n"\
f"\\end{{center}}\n"\
f"\\end{{table}}"

print(table)