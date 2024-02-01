import os
import json
from src.main.jobs.mmlu import MMLU_SUBJECTS

def main():
    os.makedirs('./saves/mmlu_report', exist_ok=True)
    
    configs = ['none', 'tree_b4_k512']
    header = ['config',] + MMLU_SUBJECTS + ['avg',]
    lines = [','.join(header)]
    seq_len = {}
    data_len = {}
    for config in configs:
        row = [config]
        acc_len = acc_sum = 0
        for subject in MMLU_SUBJECTS:
            json_path = f'./saves/llama_eval/mmlu/{subject}_{config}.json'
            with open(json_path, 'r') as f:
                data = json.load(f)
                row.append(str(data['accuracy']))
                acc_sum += data['accuracy']
                acc_len += 1
                seq_len[subject] = data['avg_seq_len']
                data_len[subject] = len(data['results'])
        row.append(str(acc_sum / acc_len))
        lines.append(','.join(row))
    seq_len = ['seq_len'] + [str(seq_len[subject]) for subject in MMLU_SUBJECTS] + [str(sum(seq_len.values()) / len(seq_len))]
    data_len = ['data_len'] + [str(data_len[subject]) for subject in MMLU_SUBJECTS] + [str(sum(data_len.values()) / len(data_len))]
    lines.append(','.join(seq_len))
    lines.append(','.join(data_len))
    csv = '\n'.join(lines)

    with open('./saves/mmlu_report/report.csv', 'w') as f:
        f.write(csv)
    print('dumped ./saves/mmlu_report/report.csv')

if __name__ == '__main__':
    main()