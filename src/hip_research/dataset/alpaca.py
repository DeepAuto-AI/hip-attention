import os
import random

import datasets
import torch
import tqdm
import transformers
from torch.utils.data import Dataset


class AlpacaDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()

        self.n_questions = 20
        self.tokenizer = tokenizer

        dataset = datasets.load_dataset("tatsu-lab/alpaca")["train"]
        self.entries = []
        for entry in dataset:
            if len(entry["input"]) > 0:
                prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f"{entry['instruction']}\n\n{entry['input']}",
                        },
                        {"role": "assistant", "content": entry["output"]},
                    ],
                    tokenize=False,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": entry["instruction"]},
                        {"role": "assistant", "content": entry["output"]},
                    ],
                    tokenize=False,
                )
            self.entries.append(prompt)

    def __len__(self):
        return len(self.entries) // self.n_questions

    def __getitem__(self, index):
        ids = self.tokenizer(self.entries[index], return_tensors="pt").input_ids
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100

        return ids[0], labels[0]


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K"
    )
    ds = AlpacaDataset(tokenizer)
    for ids, labels in ds:
        assert ids.shape == labels.shape
        print(ids.shape)
