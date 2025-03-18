import random

import datasets
from torch.utils.data import Dataset
from tqdm import tqdm


class OpenWebTextDataset(Dataset):
    def __init__(self, tokenizer, stride):
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset(
            "Skylion007/openwebtext", trust_remote_code=True
        )["train"]
        self.window_size = 30
        self.stride = stride

    def __len__(self):
        return len(self.dataset) // self.window_size

    def __getitem__(self, idx):
        text = []
        for i in range(self.window_size):
            entry = self.dataset[idx * self.window_size + i]
            text.append(entry["text"])
        random.shuffle(text)
        ids = self.tokenizer(
            "\n\n".join(text),
            return_tensors="pt",
            truncation=True,
            max_length=self.stride,
        ).input_ids
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100
        return ids[0], labels[0]


if __name__ == "__main__":
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K"
    )
    ds = OpenWebTextDataset(tokenizer, 32000)

    lengths = []
    for ids, labels in tqdm(ds, total=len(ds)):
        lengths.append(ids.shape[-1])
        print("max", max(lengths))
