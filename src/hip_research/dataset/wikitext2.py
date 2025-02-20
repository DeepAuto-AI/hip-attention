import gc
import math
import multiprocessing as mp
import os

import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class ChunkedIterator:
    def __init__(self, iterator, chunk_size, cutoff=None):
        self.iterator = iterator
        self.chunk_size = chunk_size
        self.eof = False
        self.ichunk = 0
        self.cutoff = cutoff

    def __iter__(self):
        return self

    def __next__(self):
        if self.eof:
            raise StopIteration

        chunk = []
        end_of_file = False
        while not end_of_file:
            try:
                line = next(self.iterator)["text"]
                chunk.append(line)
                if self.ichunk * self.chunk_size > self.cutoff:
                    raise StopIteration
            except StopIteration:
                end_of_file = True
                self.eof = True
            if len(chunk) >= self.chunk_size or end_of_file:
                # print('read', self.ichunk, self.cutoff, self.ichunk * self.chunk_size > self.cutoff)
                self.ichunk += 1
                return chunk


class Wikitext2Dataset(Dataset):
    def __init__(
        self, subset, tokenizer, stride=2048, max_length=None, strided_indexing=None
    ):
        super().__init__()

        self.tokenizer = tokenizer
        if subset == "valid":
            subset = "validation"
        if subset in ["validation", "test"] and strided_indexing is None:
            strided_indexing = True
        self.strided_indexing = strided_indexing

        os.makedirs("./cache/wikitext", exist_ok=True)
        dataset = "wikitext2"
        if os.environ.get("FORCE_OPENWEBTEXT", "0") == "1":
            print("FORCELY USE OPENWEBTEXT!")
            dataset = "openwebtext"
        cache_path = f"./cache/wikitext/{dataset}-{subset}.pth"
        if os.path.exists(cache_path):
            self.encodings = torch.load(cache_path)
            print("cache size", self.encodings.shape)
        else:
            cutoff_dataset = 5000000  # 5M document
            if dataset == "openwebtext":
                chunk_size = 50
                if subset == "train":
                    cutoff_dataset = 500000  # 500k document
                    data = load_dataset("Skylion007/openwebtext", split="train[:99%]")
                else:
                    cutoff_dataset = 2000  # 2k document
                    data = load_dataset("Skylion007/openwebtext", split="train[99%:]")
                print("OPENWEBTEXT loaded")
            else:
                chunk_size = 50 * 1000
                data = load_dataset("wikitext", "wikitext-2-raw-v1", split=subset)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # self.encodings = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids

            # num_lines = len(data['text'])
            # num_chunks = math.ceil(num_lines / chunk_size)
            # encodings = []
            # print('nchunk', num_chunks, flush=True)
            # for ichunk in tqdm.tqdm(range(num_chunks), disable=num_chunks < 2, leave=False, dynamic_ncols=True):
            #     chunk = data['text'][ichunk*chunk_size:min(num_lines, (ichunk+1)*chunk_size)]
            #     print('a', flush=True)
            #     flatten_text = "\n\n".join(chunk)
            #     print('b', flush=True)
            #     if ichunk == 0:
            #         flatten_text = '</s>' + flatten_text
            #     print('c', flush=True)
            #     chunk_encodings = tokenizer(flatten_text, return_tensors='pt', add_special_tokens=True).input_ids
            #     print(chunk_encodings.shape, flush=True)
            #     encodings.append(chunk_encodings)
            #     gc.collect()
            #     print('d', flush=True)

            # data_iter = iter(data)
            # chunk = []
            # encodings = []
            # ichunk = 0
            # num_tokens = 0
            # end_of_file = False
            # with tqdm.tqdm(leave=False, dynamic_ncols=True) as pbar:
            #     while not end_of_file:
            #         pbar.update(1)
            #         try:
            #             line = next(data_iter)['text']
            #             chunk.append(line)
            #         except StopIteration:
            #             end_of_file = True
            #         if len(chunk) >= chunk_size or end_of_file:
            #             # print('a', flush=True)
            #             flatten_text = "\n\n".join(chunk)
            #             # print('b', flush=True)
            #             if ichunk == 0:
            #                 flatten_text = '</s>' + flatten_text
            #             # print('c', flush=True)
            #             chunk_encodings = tokenizer(flatten_text, return_tensors='pt', add_special_tokens=False).input_ids
            #             # print(chunk_encodings.shape, flush=True)
            #             encodings.append(chunk_encodings)
            #             num_tokens += chunk_encodings.shape[1]
            #             pbar.set_description(f'Tokens: {num_tokens}')
            #             if (ichunk % 10) == 0:
            #                 gc.collect()
            #             # print('d', flush=True)
            #             ichunk += 1

            encodings = []
            encodings_size = 0
            chunked_iterator = ChunkedIterator(
                iter(data), chunk_size, cutoff=cutoff_dataset
            )
            with (
                mp.Pool(mp.cpu_count() - 1) as pool,
                tqdm.tqdm(
                    pool.imap(self.get_encodings, chunked_iterator, chunksize=8),
                    dynamic_ncols=True,
                    total=math.ceil(cutoff_dataset / chunk_size),
                ) as pbar,
            ):
                for chunk_encodings in pbar:
                    encodings_size += chunk_encodings.shape[1]
                    pbar.set_description(f"tokens: {encodings_size}")
                    encodings.append(torch.tensor(chunk_encodings))

            self.encodings = torch.cat(encodings, dim=1)

            torch.save(self.encodings, cache_path)
        self.seq_len = self.encodings.size(1)
        print(f"{self.seq_len} tokens loaded")
        # self.seq_len = self.encodings.input_ids.size(1)
        self.stride = stride
        self.max_length = max_length
        self.check_last_shape = subset == "train"
        self.last_shape = None

    def get_encodings(self, chunk):
        flatten_text = "\n\n".join(chunk)
        chunk_encodings = self.tokenizer(
            flatten_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        return chunk_encodings.numpy()

    def __len__(self):
        if self.strided_indexing:
            # drop last by default
            return max(math.floor(self.seq_len / self.stride), 1)
        else:
            # return self.seq_len - self.stride * 2
            return self.seq_len - self.stride

    def __getitem__(self, idx):
        max_length = self.max_length
        assert max_length > 0

        if not self.strided_indexing:
            # idx = idx + self.stride
            begin_loc = idx
        else:
            begin_loc = idx * self.stride

        end_loc = min(begin_loc + max_length, self.seq_len)
        trg_len = end_loc - min(begin_loc - self.stride + max_length, self.seq_len)

        input_ids = self.encodings[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if self.check_last_shape:
            if self.last_shape is not None:
                assert self.last_shape == input_ids.shape
            self.last_shape = input_ids.shape

        return {
            "input_ids": input_ids[0],
            "labels": target_ids[0],
            "trg_len": torch.tensor(trg_len),
        }


def get_dataloader(
    subset, tokenizer, batch_size=1, max_length=None, local_rank=0, world_size=1
):
    assert max_length is not None
    ds = Wikitext2Dataset(subset, tokenizer, stride=max_length, max_length=max_length)
    use_shuffle = subset == "train"

    if world_size > 1:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
            sampler=DistributedSampler(
                dataset=ds,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=use_shuffle,
            ),
        )
    else:
        return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=use_shuffle)


if __name__ == "__main__":
    import transformers

    t = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m")
    # loader = get_dataloader('train', t, batch_size=1, max_length=768)
    loader = get_dataloader("valid", t, batch_size=1, max_length=768)

    for batch in tqdm.tqdm(loader):
        # print([(k, v.shape) for k, v in batch.items()])
        pass
