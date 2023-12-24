"""
Dump attention estimation intermediates

Usage: python -m src.main.tests.test_perlin_est_atten

NOTE: this test script is outdated, may malfunctioning
"""

from ...utils import get_bench
from ..visualize.glue import main as visualize_main
from ..visualize.glue import add_perlin_model_options, parse_perlin_model_options
import argparse, os, torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

if __name__ == '__main__':
    # mp.set_start_method('spawn')

    bench = get_bench()
    bench.activate_temp_buffers = True

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--subset', type=str, default='mnli')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    add_perlin_model_options(parser)

    args = parser.parse_args()
    print(args)

    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'subset': args.subset,
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate
    })

    visualize_main(**kwargs)

    index_layer = 1

    v_for_atten = bench.get_temp_buffer('v_for_atten', index_layer)
    performer_context_layer = bench.get_temp_buffer('performer_context_layer', index_layer)
    estimated_attention_score = bench.get_temp_buffer('estimated_attention_score', index_layer)
    estimated_attention_probs = bench.get_temp_buffer('estimated_attention_probs', index_layer)
    estimated_attention_probs_for_output = bench.get_temp_buffer('estimated_attention_probs_for_output', index_layer)
    partial_attention_mask_before_interp = bench.get_temp_buffer('partial_attention_mask_before_interp', index_layer)
    partial_attention_mask = bench.get_temp_buffer('partial_attention_mask', index_layer)
    attention_mask = bench.get_temp_buffer('attention_mask', index_layer)
    attention_probs_truth = bench.get_temp_buffer('attention_probs_truth', index_layer)
    # attention_probs_truth_m = bench.get_temp_buffer('attention_probs_truth_m', index_layer)
    t_attention_predictor = bench.get_temp_buffer('t_attention_predictor', index_layer)

    def imsave(img: torch.Tensor, path):
        plt.clf()
        plt.imshow(img.cpu().numpy())
        plt.colorbar()
        plt.savefig(path, dpi=300)
        print(f'saved {path}')

    root = './saves/tests/test_perlin_est_atten/'
    os.makedirs(root, exist_ok=True)

    index_batch = 2
    index_head = 0

    imsave(v_for_atten[index_batch,index_head], os.path.join(root, 'v_atten.png'))
    imsave(performer_context_layer[index_batch,index_head], os.path.join(root, 'perf_cont.png'))
    imsave(estimated_attention_score[index_batch,index_head], os.path.join(root, 'est_score.png'))
    imsave(estimated_attention_probs[index_batch,index_head], os.path.join(root, 'est.png'))
    imsave(estimated_attention_probs_for_output[index_batch,index_head], os.path.join(root, 'est_interp.png'))
    imsave(
        (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)) /\
        (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)).sum(dim=-1, keepdim=True), os.path.join(root, 'est_interp_norm.png'))
    imsave(partial_attention_mask_before_interp[index_batch,index_head], os.path.join(root, 'part.png'))
    imsave(partial_attention_mask[index_batch,index_head], os.path.join(root, 'part_interp.png'))
    imsave(attention_probs_truth[index_batch,index_head], os.path.join(root, 'attention_probs_truth.png'))
    # imsave(attention_probs_truth_m[index_batch,index_head], os.path.join(root, 'attention_probs_truth_m.png'))
    imsave(t_attention_predictor[index_batch,index_head], os.path.join(root, 't_attention_predictor.png'))