import random

import torch.multiprocessing as mp
import transformers
from torchtext.datasets import WikiText103


class FilteredWikitext:
    def __init__(self, min_length=50):
        self.data = None
        self.length = 0
        self.min_length = min_length
        for i in self:
            self.length += 1

    def __iter__(self):
        self.data = iter(WikiText103(split="train", root="./cache/wikitext103"))
        return self

    def __next__(self):
        line = ""
        while len(line) < self.min_length:
            line = next(self.data)
        return line

    def __len__(self):
        return self.length


class WikitextBatchLoader:
    def __init__(self, batch_size):
        data = FilteredWikitext()
        self.bank = []
        for i in data:
            self.bank.append(i)
        del data
        self.tokenizer = None
        self.batch_size = batch_size
        self.index = 0
        self.queue = mp.Queue(maxsize=64)
        procs = []
        self.num_workers = 2
        for i in range(self.num_workers):
            proc = mp.Process(target=self.worker_main, daemon=True)
            proc.start()
            procs.append(proc)
        self.procs = procs

    def worker_main(self):
        print("WikitextBatchLoader: worker_main")
        while True:
            item = self.random_batch()
            self.queue.put(item)

    def random_sample(self):
        # mimic GLUE
        line = self.bank[random.randint(0, len(self.bank) - 1)].strip()
        line2 = self.bank[random.randint(0, len(self.bank) - 1)].strip()

        # masking
        def masking_line(line):
            spl = line.split()
            for i in range(len(spl)):
                if random.random() < 0.15:
                    if random.random() < 0.8:
                        spl[i] = "[MASK]"
                    else:
                        spl[i] = spl[random.randint(0, len(spl) - 1)]
            line = " ".join(spl)
            return line

        line = masking_line(line)
        line2 = masking_line(line2)
        # random cut
        if random.random() < 0.65:  # need to re experiment
            spl = line.split()
            if len(spl) > 10:
                spl = spl[: random.randint(10, len(spl))]
            line = " ".join(spl)
        # mimic sep
        if random.random() < 0.75:
            spl = line.split()
            sep_idx = random.randint(0, len(spl) - 1)
            spl.insert(sep_idx, "[SEP]")
            if random.random() < 0.5:
                spl2 = line2.split()
                spl2_patch_len = min(len(spl) - sep_idx - 1, len(spl2))
                spl[sep_idx + 1 : min(sep_idx + 1 + spl2_patch_len, len(spl))] = spl2[
                    :spl2_patch_len
                ]
            line = " ".join(spl)

        # end sep
        if random.random() < 0.75:
            line = line + "[SEP]"
        # mimic cls
        if random.random() < 0.75:
            line = "[CLS]" + line
        return line

    def random_batch(self):
        lines = [self.random_sample() for i in range(self.batch_size)]
        if self.tokenizer is None:
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained(
                "bert-base-uncased"
            )
        result = self.tokenizer(
            lines, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        item = {
            "input_ids": result.input_ids,
            "attention_mask": result.attention_mask,
        }
        return item

    def __iter__(self):
        self.index = 0
        return self

    def ___next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        self.index += 1
        return self.queue.get()

    def __next__(self):
        return self.___next__()

    def __len__(self):
        return len(self.bank) // self.batch_size


if __name__ == "__main__":
    data = WikitextBatchLoader(16)
    for i in range(50):
        print(data.random_sample())
