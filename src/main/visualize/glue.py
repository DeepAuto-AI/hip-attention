import torch
import cv2
import os
import argparse

from ...utils import batch_to
from ...models import perlin_bert
from ...trainer.perlin_trainer import GlueTrainer, add_perlin_model_options, parse_perlin_model_options

from .common import (
    gather_fixed_batch,
    process_batch_index,
)

def main(
    subset = 'mnli',
    checkpoint_path = None,
    evaluate = False,
    **kwargs
):
    trainer = GlueTrainer(
        subset=subset,
        **kwargs
    )
    trainer.load(path=checkpoint_path)
    
    batch = gather_fixed_batch(trainer.valid_loader, 10)
    batch = batch_to(batch, trainer.device)
    
    teacher = trainer.base_model
    bert = trainer.model
    
    teacher.eval()
    bert.eval()
    
    with torch.no_grad():
        teacher(**batch)
        batch['teacher'] = teacher
        bert(**batch)
    
    attentions = []
    for module in bert.modules():
        if isinstance(module, perlin_bert.BertSelfAttention):
            teacher_attn = module.teacher_attention_prob
            estimated_attn = module.last_perlin_estimated_probs
            dense_attn = module.last_perlin_dense_probs
            partial_attn = module.last_perlin_partial_probs
            attentions.append({
                'teacher_attn': teacher_attn.cpu(),
                'estimated_attn': estimated_attn.cpu(),
                'dense_attn': dense_attn.cpu(),
                'partial_attn': partial_attn.cpu(),
            })
    
    os.makedirs(f"./plots/visualize_glue", exist_ok=True)
    for i in range(len(batch['input_ids'])):
        token_length = int(batch['attention_mask'][i].sum().item())
        # token_length = batch['input_ids'].shape[-1]
        img = process_batch_index(attentions, i, token_length)
        path = f"./plots/visualize_glue/{i}.png"
        cv2.imwrite(path, img)
        print('processed', path)
    
    if evaluate:
        print('accuracy', trainer.evaluate())

if __name__ == '__main__':
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
    
    main(**kwargs)