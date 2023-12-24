import torch
import cv2
import os
import argparse

import tqdm

from ...utils import batch_to
from ...models import perlin_opt
from ...trainer.perlin_trainer import OptTrainer, add_perlin_model_options, parse_perlin_model_options

from .common import (
    gather_fixed_batch,
    process_batch_index,
)
from . import common

def main(
    dataset = 'wikitext2',
    model = 'opt',
    checkpoint_path = None,
    evaluate = False,
    **kwargs
):
    trainer = OptTrainer(
        model=model,
        subset=dataset,
        **kwargs,
    )
    trainer.load(path=checkpoint_path)
    
    if evaluate:
        class OnEvaluateStep:
            def __init__(self):
                self.k_sum = self.k_count = 0
            
            def __call__(self):
                student = trainer.model # type: perlin_opt.OPTForCausalLM
                for layer in student.model.decoder.layers:
                    layer = layer #type: perlin_opt.OPTDecoderLayer
                    output = layer.self_attn.last_perlin_output
                    partial_mask = (output.partial_attention_mask > -1).float()
                    N, H, T, T = partial_mask.shape
                    avg_k = partial_mask.sum()
                    self.k_sum += avg_k.item()
                    self.k_count += N*H*T
            
            def calc(self): return self.k_sum / self.k_count
        
        callback = OnEvaluateStep()
        
        print(f'PPL: {trainer.evaluate(on_step=callback, quite=True)}, k: {callback.calc():.2f}')
    
    batch = gather_fixed_batch(trainer.valid_loader, 5)
    batch = batch_to(batch, trainer.device)
    del batch['trg_len']
    
    teacher = trainer.base_model
    student = trainer.model
    
    for module in student.modules():
        if isinstance(module, perlin_opt.OPTAttention):
            module.checkout_intermediates = True
            module.checkout_perlin_output = True
            module.swap_out_device = torch.device('cuda')
    
    teacher.eval()
    student.eval()
    
    attentions = []
    
    for i in tqdm.tqdm(range(len(batch['input_ids'])), dynamic_ncols=True, desc='sample'):
        mini_attentions = []
        mini_batch = {k: v[i:i+1] for k, v in batch.items()}
        
        with torch.no_grad(), torch.autocast('cuda', torch.float16):
            teacher(**mini_batch)
            mini_batch['teacher'] = teacher
            student(**mini_batch)
        
        for module in student.modules():
            if isinstance(module, perlin_opt.OPTAttention):
                teacher_attn = torch.softmax(module.teacher_attention_scores, dim=-1).cpu()
                estimated_attn = module.last_perlin_output.estimated_attention_probs.cpu()
                dense_attn = module.last_perlin_output.dense_attention_probs.cpu()
                partial_attn = module.last_perlin_output.partial_attention_probs.cpu()
                mini_attentions.append({
                    'teacher_attn': teacher_attn,
                    'estimated_attn': estimated_attn,
                    'dense_attn': dense_attn,
                    'partial_attn': partial_attn,
                })
        
        if len(attentions) == 0:
            attentions = mini_attentions
        else:
            for i in range(len(attentions)):
                attentions[i] = {
                    k: torch.cat([v, mini_attentions[i][k]]) for k, v in attentions[i].items()
                }
    
    num_layers = len(teacher.model.decoder.layers)
    os.makedirs(f"./plots/visualize_opt", exist_ok=True)
    for i in range(len(batch['input_ids'])):
        token_length = batch['input_ids'].shape[-1]
        # token_length = batch['input_ids'].shape[-1]
        common.POOL = 8
        img = process_batch_index(attentions, i, token_length, gs=[0.2,0.2,0.2,0.2])
        layer_dir = f"./plots/visualize_opt/{dataset}_{i}"
        os.makedirs(layer_dir, exist_ok=True)
        assert (img.shape[0] % num_layers) == 0, f"{img.shape}"
        for j in range(num_layers):
            img_layer = img[j*(img.shape[0] // num_layers):(j+1)*(img.shape[0] // num_layers)]
            path = os.path.join(layer_dir, f'l{j}.png')
            cv2.imwrite(path, img_layer)
            print('processed', path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--max-seq-len', type=int, default=2048)
    parser.add_argument('--model', type=str, default='opt')
    add_perlin_model_options(parser)
    
    args = parser.parse_args()
    print(args)
    
    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'dataset': args.dataset,
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate,
        'max_seq_len': args.max_seq_len,
        'model': args.model
    })
    
    main(**kwargs)