"""
Check floating point precision error on FP16 FP32 BF16

Usage: python -m src.main.tests.test_perlin_opt_precision --k 64 --predictor-length 256

NOTE: this test script is not maintained anymore.
"""

import torch
import cv2
import os
import argparse

import tqdm
import torch.nn.functional as F

from ...utils import batch_to, get_bench
from ...models import perlin_opt
from ...trainer.perlin_trainer import OptTrainer, add_perlin_model_options, parse_perlin_model_options

bench = get_bench()
bench.activate_temp_buffers = True

from ..visualize.common import (
    gather_fixed_batch,
    process_batch_index,
)

def sample_and_reset():
    samples = bench.buffers
    bench.buffers = {}
    return samples

def compare_smaples(*samples_dicts):
    buffer_names = samples_dicts[0].keys()
    for name in buffer_names:
        print('-'*80)
        print(f'compare: {name} ', end='')
        report = ""
        max_loss = 0
        for i in range(len(samples_dicts[0][name])):
            report += f' - id {i}\n'
            samples = [buffer[name][i] for buffer in samples_dicts]
            report += '    '
            for s in samples:
                report += f'{s.dtype}, '
            report += '\n'
            n = len(samples)
            for j in range(n):
                report += f'    '
                for k in range(n):
                    x = samples[j].double()
                    y = samples[k].double()
                    x = torch.clamp(x, -32000, 32000)
                    y = torch.clamp(y, -32000, 32000)
                    loss = F.mse_loss(x, y)
                    max_loss = max(loss, max_loss)
                    report += f'{f"{loss:.6f}".rjust(14)}, '
                report += '\n'
        if max_loss > 0:
            print()
            print(report)
        else:
            print(f'passed ({max_loss:.8f})')

def main(
    dataset = 'wikitext2',
    checkpoint_path = None,
    evaluate = False,
    **kwargs
):
    trainer = OptTrainer(
        model='opt',
        subset=dataset,
        **kwargs,
    )
    # if os.path.exists(checkpoint_path if checkpoint_path is not None else trainer.checkpoint_path()):
    #     trainer.load(path=checkpoint_path)
    
    batch = gather_fixed_batch(trainer.valid_loader, 1)
    batch = batch_to(batch, trainer.device)
    del batch['trg_len']
    
    teacher = trainer.base_model
    student = trainer.model
    
    teacher.eval()
    student.eval()
    
    i = 0
    mini_batch = {k: v[i:i+1] for k, v in batch.items()}
    
    with torch.no_grad(), torch.autocast('cuda', torch.float16):
        teacher(**mini_batch)
        mini_batch['teacher'] = teacher
        student(**mini_batch)
    
    samples_fp16 = sample_and_reset()
    print('fp16 sampled')
        
    with torch.no_grad(), torch.autocast('cuda', torch.float32):
        teacher(**mini_batch)
        mini_batch['teacher'] = teacher
        student(**mini_batch)
    
    samples_fp32 = sample_and_reset()
    print('fp32 sampled')
    
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        teacher(**mini_batch)
        mini_batch['teacher'] = teacher
        student(**mini_batch)
    
    samples_bf16 = sample_and_reset()
    print('bf16 sampled')
    
    compare_smaples(samples_fp16, samples_fp32, samples_bf16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--max-seq-len', type=int, default=2048)
    add_perlin_model_options(parser)
    
    args = parser.parse_args()
    print(args)
    
    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'dataset': args.dataset,
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate,
        'max_seq_len': args.max_seq_len,
    })
    
    main(**kwargs)