import torch
import cv2
import os
import argparse

import tqdm

from ...utils import batch_to, get_bench
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
    
    if os.environ.get('CPU', '0') == '1':
        trainer.device = 'cpu'
        trainer.model.to('cpu')
        trainer.base_model.to('cpu')
    
    if evaluate:
        class OnEvaluateStep:
            def __init__(self):
                self.k_sum = self.k_count = 0
            
            def __call__(self):
                student = trainer.model # type: perlin_opt.OPTForCausalLM
                # for layer in student.model.decoder.layers:
                #     layer = layer #type: perlin_opt.OPTDecoderLayer
                #     output = layer.self_attn.last_perlin_output
                #     partial_mask = (output.partial_attention_mask > -1).float()
                #     N, H, T, T = partial_mask.shape
                #     avg_k = partial_mask.sum()
                #     self.k_sum += avg_k.item()
                #     self.k_count += N*H*T
            
            def calc(self): 
                return 0
                # return self.k_sum / self.k_count
        
        callback = OnEvaluateStep()
        
        print(f'PPL: {trainer.evaluate(on_step=callback, quite=True)}, k: {callback.calc():.2f}')
    
    batch = gather_fixed_batch(trainer.valid_loader, 1)
    batch = batch_to(batch, trainer.device)
    del batch['trg_len']
    
    teacher = trainer.base_model
    student = trainer.model
    
    for module in student.modules():
        if isinstance(module, perlin_opt.OPTAttention):
            module.checkout_intermediates = True
            module.checkout_perlin_output = True
            # module.swap_out_device = torch.device('cuda')
    
    teacher.eval()
    student.eval()
    
    get_bench().activate_temp_buffers = True
    
    for i in tqdm.tqdm(range(len(batch['input_ids'])), dynamic_ncols=True, desc='sample'):
        mini_batch = {k: v[i:i+1] for k, v in batch.items()}
        
        with torch.no_grad(), torch.autocast('cuda', torch.float16):
            teacher(**mini_batch)
            mini_batch['teacher'] = teacher
            student(**mini_batch)
    
    os.makedirs(f"./plots/visualize_opt_perf_cnn", exist_ok=True)
    performer_value = get_bench().get_temp_buffer('estimated_attention_score_dec_row', 0)
    estimated_probs = get_bench().get_temp_buffer('estimated_attention_probs', 0)
    torch.save({'value': performer_value, 'probs': estimated_probs}, './plots/visualize_opt_perf_cnn/sample.pth')

def sample_llm():
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

def render_visuals():
    data = torch.load('./plots/visualize_opt_perf_cnn/sample.pth', map_location='cpu')
    SAMPLE_HEIGHT = 768
    x = data['value'][0, 4, 128:128+SAMPLE_HEIGHT].numpy()
    y = data['probs'][0, 0, 128:128+SAMPLE_HEIGHT].numpy()
    print(x.shape, y.shape)
    
    HEIGHT = 96
    
    x = cv2.resize(common.convert_to_colormap(x), dsize=(y.shape[1], HEIGHT), interpolation=cv2.INTER_NEAREST)
    y = cv2.resize(common.convert_to_colormap(y), dsize=(y.shape[1], HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    x = cv2.resize(x, dsize=None, fy=8.0, fx=8.0, interpolation=cv2.INTER_NEAREST)
    y = cv2.resize(y, dsize=None, fy=8.0, fx=8.0, interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite('./plots/visualize_opt_perf_cnn/x.png', x)
    cv2.imwrite('./plots/visualize_opt_perf_cnn/y.png', y)

if __name__ == '__main__':
    # sample_llm()
    render_visuals()