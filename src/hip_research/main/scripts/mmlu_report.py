import argparse
import json
import os

from hip_research.main.jobs.mmlu import MMLU_SUBJECTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs="+", required=True)
    args = parser.parse_args()
    os.makedirs("./saves/mmlu_report", exist_ok=True)

    header = (
        [
            "config",
        ]
        + MMLU_SUBJECTS
        + [
            "avg",
        ]
    )
    lines = [",".join(header)]
    seq_len = {}
    data_len = {}
    for config in args.configs:
        row = [repr(config)]
        acc_len = acc_sum = 0
        for subject in MMLU_SUBJECTS:
            json_path = f"./saves/llama_eval/mmlu/{config}/{subject}.json"
            if not os.path.exists(json_path):
                row.append("N/A")
                print(f"not found: {json_path}. skipping")
                continue
            with open(json_path, "r") as f:
                data = json.load(f)
                row.append(str(data["accuracy"]))
                acc_sum += data["accuracy"]
                acc_len += 1
                seq_len[subject] = data["avg_seq_len"]
                data_len[subject] = len(data["results"])
        row.append(str(acc_sum / acc_len))
        lines.append(",".join(row))
    seq_len = (
        ["seq_len"]
        + [str(seq_len[subject]) for subject in MMLU_SUBJECTS]
        + [str(sum(seq_len.values()) / len(seq_len))]
    )
    data_len = (
        ["data_len"]
        + [str(data_len[subject]) for subject in MMLU_SUBJECTS]
        + [str(sum(data_len.values()) / len(data_len))]
    )
    lines.append(",".join(seq_len))
    lines.append(",".join(data_len))
    csv = "\n".join(lines)

    with open("./saves/mmlu_report/report.csv", "w") as f:
        f.write(csv)
    print("dumped ./saves/mmlu_report/report.csv")


if __name__ == "__main__":
    main()
