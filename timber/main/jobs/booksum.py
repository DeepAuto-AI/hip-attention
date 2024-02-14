import json
import os
import pathlib

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from timber.dataset.booksum import BookSumDataset
from timber.utils import seed, get_bench
from torch.utils.data import Subset

import subprocess
import logging


def gen_summary(args, model, tokenizer, device, idx, item, out_dir, gen_tokens=512):
    PROMPT_ALWAYS_FLASH = os.environ.get('PROMPT_ALWAYS_FLASH', '0') == '1'
    LAST_DENSE = os.environ.get('LAST_DENSE', '0') == '1'

    inputs, completion = item

    if (out_dir / f"out_{idx}.txt").exists():
        with open(out_dir / f"out_{idx}.txt", 'r') as f:
            return f.read()

    tokenizer.truncation_side = 'left'

    assert hasattr(model, 'config')
    assert hasattr(model.config, 'max_position_embeddings')

    inputs = inputs[..., inputs.shape[-1] - (model.config.max_position_embeddings - gen_tokens):]

    seq_len = inputs.shape[-1]
    print(f"seq_len: {seq_len}")

    if PROMPT_ALWAYS_FLASH:
        for m in model.modules():
            if hasattr(m, 'attention_method'):
                m.tree_dense_queries = seq_len - 1
    if LAST_DENSE:
        for m in model.modules():
            if hasattr(m, 'attention_method'):
                m.tree_last_dense_queries = -1

    output = model.generate(
        inputs=inputs.unsqueeze(0).cuda(),
        attention_mask=torch.ones((1, inputs.shape[-1]), dtype=torch.long, device='cuda'),
        max_new_tokens=gen_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    output = tokenizer.decode(
        output[0][seq_len:].data.cpu(),
        skip_special_tokens=True,
    )
    tqdm.write(f"{idx} Summary:\t{(output[:200],)}[...]\n\n")

    with open(out_dir / f"out_{idx}.txt", 'w') as f:
        f.write(output)

    return output


def install_rogue():
    logger = logging.getLogger()

    ROUGE_HOME = os.environ.get('ROUGE_HOME', "cache/ROUGE-1.5.5")
    if "ROUGE_HOME" not in os.environ:
        logger.info("ROUGE_HOME not set, using default location %s", ROUGE_HOME)

    if not os.path.exists(ROUGE_HOME):
        logger.info("ROUGE_HOME=%s not a directory.", ROUGE_HOME)
        try:
            logger.info("Installing rouge Perl script to {ROUGE_HOME} - this will take a few seconds")
            subprocess.run(
                ["curl", "-L", "https://github.com/Yale-LILY/SummEval/tarball/7e4330d", "-o", "project.tar.gz", "-s"])
            subprocess.run(["tar", "-xzf", "project.tar.gz"])
            subprocess.run(["mv", "Yale-LILY-SummEval-7e4330d/evaluation/summ_eval/ROUGE-1.5.5/", ROUGE_HOME])
            subprocess.run(["rm", "project.tar.gz"])
            subprocess.run(["rm", "-rf", "Yale-LILY-SummEval-7e4330d/"])
        except subprocess.CalledProcessError as err:
            logger.error(
                "Failed to install the rouge Perl script; please install manually and set the ROUGE_HOME environment variable.")
            raise err

    return ROUGE_HOME


@torch.no_grad()
def job_booksum(args, model, tokenizer, device):
    seed()

    dataset = BookSumDataset(
        tokenizer=tokenizer,
        for_eval=True,
    )
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.05)
    test_dataset = Subset(dataset, test_idx)

    outputs = []

    out_dir = pathlib.Path(f"saves/llama_eval/booksum/{args.model}_{args.method}_bq{args.block_size_q}_bk{args.block_size_k}_k{args.k}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path("saves/llama_eval/booksum/reference").mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(tqdm(test_dataset, dynamic_ncols=True, leave=True, desc="booksum")):
        output = gen_summary(args, model, tokenizer, device, idx, item, out_dir)
        outputs.append(output)

        with open(f"saves/llama_eval/booksum/reference/ref_{idx}.txt", 'w') as f:
            inputs, completion = item
            f.write(tokenizer.decode(completion, skip_special_tokens=True))

    rouge_dir = install_rogue()

    from pyrouge import Rouge155

    r = Rouge155(rouge_dir=rouge_dir)
    r.system_dir = out_dir  # "system" is the one we want to measure
    r.model_dir = "saves/llama_eval/booksum/reference"  # "model" is the gold standard (i.e. human summaries)
    r.system_filename_pattern = r'out_(\d+)\.txt'
    r.model_filename_pattern = r'ref_#ID#\.txt'

    output = r.convert_and_evaluate()
    print("R: Recall, P: Precision, F: F1 score")
    print(output)
    output_dict = r.output_to_dict(output)
    with open(out_dir / "rouge_scores.json", 'w') as f:
        json.dump(output_dict, f, indent=2)
