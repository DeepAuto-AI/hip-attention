import os
import random

import datasets
import torch
import tqdm
import transformers
from torch.utils.data import Dataset


class LmsysChatDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset("lmsys/lmsys-chat-1m")["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        prompt = tokenizer.apply_chat_template(
            self.dataset[index]["conversation"],
            tokenize=False,
        )
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100

        return ids[0], labels[0]


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K"
    )
    ds = LmsysChatDataset(tokenizer)
    for ids, labels in ds:
        assert ids.shape == labels.shape
        print(ids.shape)
        print(ids)
