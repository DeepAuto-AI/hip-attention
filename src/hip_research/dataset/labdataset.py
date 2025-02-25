"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model

"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch
import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LabDataset(Dataset):
    """Mini version of WikiText2."""

    def __init__(
        self,
        data_dir: Path = "./cache/wikitext2",
        block_size: int = 4096,
        download: bool = True,
        tokenizer: AutoTokenizer = None,
        dataset: str = "wikitext103",
    ) -> None:
        super().__init__()

        self.dataset = dataset
        os.makedirs("./cache/wikitext2", exist_ok=True)
        cache_path = "./cache/wikitext2/tokenized.pth"
        if self.dataset != "wikitext103":
            cache_path = f"./cache/wikitext2/tokenized_{self.dataset}.pth"
        if os.path.exists(cache_path):
            data = torch.load(cache_path)
        else:
            self.path = Path(data_dir) / "wikitext-2.txt"
            if download:
                self.download(self.path)
            document = tokenize(self.path)
            print("tokenizing")
            lines = []
            for line in tqdm.tqdm(document, dynamic_ncols=True, leave=False):
                data = tokenizer(line, return_tensors="pt").input_ids.view(-1)
                lines.append(data)
            data = torch.cat(lines)
            torch.save(data, cache_path)
            print("tokenized")

        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) // self.block_size - 1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        inputs = self.data[start:end]
        target = self.data[(start + 1) : (end + 1)]  # noqa: E203
        return inputs, target

    def download(self, destination: Path) -> None:
        os.makedirs(destination.parent, exist_ok=True)

        dataset = self.dataset
        if dataset == "wikitext2":
            url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
            if os.path.exists(destination):
                return
            data = requests.get(url).text
        elif dataset == "wikitext103":
            from datasets import load_dataset

            test = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            data = "\n\n".join(test["text"])
        else:
            raise Exception()

        with open(destination, "w") as f:
            f.write(data)


class Dictionary:
    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


def tokenize(path: Path) -> Tuple[Tensor, Dictionary]:
    dictionary = Dictionary()

    assert os.path.exists(path)
    # Add words to the dictionary
    lines = []
    with open(path, encoding="utf8") as f:
        for line in f:
            lines.append(line + "\n")
    return lines
    #         words = line.split()
    #         for word in words:
    #             dictionary.add_word(word)

    # # Tokenize file content
    # with open(path, encoding="utf8") as f:
    #     idss: List[Tensor] = []
    #     for line in f:
    #         words = line.split() + ["<eos>"]
    #         ids: List[int] = []
    #         for word in words:
    #             ids.append(dictionary.word2idx[word])
    #         idss.append(torch.tensor(ids).type(torch.int64))

    # return torch.cat(idss), dictionary
