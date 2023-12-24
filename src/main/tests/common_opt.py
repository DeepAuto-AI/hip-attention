import gc
import time
from typing import Tuple
import torch
import cv2
import os
import argparse

import tqdm

from ...utils import batch_to
from ...models import perlin_opt
from ...models import perlin_attention
from ...trainer.perlin_trainer import OptTrainer, add_perlin_model_options, parse_perlin_model_options
from transformers import OPTForCausalLM, AutoTokenizer

def init_opt(
    dataset = 'wikitext2',
    checkpoint_path = None,
    **kwargs
):
    trainer = OptTrainer(
        subset=dataset,
        disable_compile=True,
        **kwargs,
    )
    trainer.device = 'cuda'
    if checkpoint_path is None:
        checkpoint_path = trainer.checkpoint_path()
    if os.path.exists(checkpoint_path):
        trainer.load(path=checkpoint_path)
    else:
        print('checkpoint not exists', checkpoint_path)
    
    model = trainer.model.to(trainer.device).eval() # type: perlin_opt.OPTForCausalLM
    tokenizer = trainer.tokenizer
    
    return trainer, model, tokenizer

def init(skip_init_loaders=False, checkpoint_path=None) -> Tuple[OptTrainer, perlin_opt.OPTForCausalLM, AutoTokenizer]:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--checkpoint', type=str, default=checkpoint_path)
    parser.add_argument('--model', type=str, default='opt')
    parser.add_argument('--max-seq-len', type=int, default=768)
    add_perlin_model_options(parser)
    
    args = parser.parse_args()
    print(args)
    
    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'model': args.model,
        'dataset': args.dataset,
        'checkpoint_path': args.checkpoint,
        'max_seq_len': args.max_seq_len,
        'skip_init_loaders': skip_init_loaders,
    })
    
    return init_opt(**kwargs)