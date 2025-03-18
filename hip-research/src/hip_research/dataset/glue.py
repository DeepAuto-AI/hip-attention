import os
import random

import torch
from datasets import load_dataset

TASK_TO_VALID = {
    "cola": "validation",
    "mnli": "validation_matched",
    "mrpc": "test",
    "qnli": "validation",
    "qqp": "validation",
    "rte": "validation",
    "sst2": "validation",
    "stsb": "validation",
    "wnli": "validation",
    "bert": "validation",
}

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_dataloader(subset, tokenizer, batch_size, split="train", encode_batch_size=384):
    if subset == "bert":
        subset = "cola"  # return dummy set

    dataset = load_dataset(
        "glue",
        subset,
        split=split,
        cache_dir=os.environ.get("HF_DATASETS_CACHE", "./cache/datasets"),
    )

    sentence1_key, sentence2_key = TASK_TO_KEYS[subset]

    def encode(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, max_length=256, truncation=True)
        # result = tokenizer(*args, padding="max_length", max_length=512, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    if split.startswith("train"):  # shuffle when train set
        dataset = dataset.sort("label")
        dataset = dataset.shuffle(seed=random.randint(0, 10000))
    dataset = dataset.map(
        lambda examples: {"labels": examples["label"]},
        batched=True,
        batch_size=encode_batch_size,
    )
    dataset = dataset.map(encode, batched=True, batch_size=encode_batch_size)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
    )
    return dataloader
