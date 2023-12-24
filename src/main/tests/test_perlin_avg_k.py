"""
Check the avearge $k$ (not $k_m$) on given dataset.
Usage: python -m src.main.tests.test_perlin_avg_k --subset mnli --k 7
"""

from ...models import perlin_bert
from transformers import AutoConfig, AutoTokenizer
from ...dataset.glue import get_dataloader, TASK_TO_VALID
from ...utils import batch_to
import torch, tqdm, argparse
from ...trainer.perlin_trainer import add_perlin_model_options, parse_perlin_model_options
from ...trainer.perlin_trainer import GlueTrainer

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
    
    trainer.base_model.eval()
    trainer.model.eval()
    
    for module in trainer.model.modules():
        if hasattr(module, 'benchmarking'):
            module.benchmarking = False
    
    batch_size = 16
    encode_batch_size = 384
    valid_loader = get_dataloader(
        trainer.subset, 
        trainer.tokenizer, 
        batch_size, 
        TASK_TO_VALID[trainer.subset], 
        encode_batch_size
    )
    
    acc_sum = 0
    acc_count = 0
    k_sum = 0
    k_count = 0
    with tqdm.tqdm(valid_loader, dynamic_ncols=True) as pbar:
        for batch in pbar:
            batch = batch_to(batch, trainer.device)
            with torch.no_grad(), torch.autocast('cuda', torch.float32):
                batch['output_attentions'] = True
                batch['output_hidden_states'] = True
                trainer.base_model(**batch)
                batch['teacher'] = trainer.base_model
                output = trainer.model(**batch)
                acc_sum += (torch.argmax(output.logits, dim=-1) == batch['labels']).float().sum().item()
                acc_count += len(batch['labels'])
            
            for layer in trainer.model.bert.encoder.layer:
                layer = layer # type: perlin_bert.BertLayer
                last_partial_probs = layer.attention.self.last_perlin_partial_probs
                if last_partial_probs.is_sparse:
                    last_partial_probs = last_partial_probs.to_dense()
                    last_partial_probs = (1-last_partial_probs) * -10000
                H = 12
                N = len(batch['labels'])
                T, T_1 = last_partial_probs.shape[-2:]
                assert T == T_1
                elem_alive = (last_partial_probs.abs() > 1e-8).float().view(N, -1).sum(dim=-1)
                token_length = batch['attention_mask'].sum(-1)
                k = (elem_alive / token_length / H).sum()
                k_sum += k.item()
                k_count += N
            pbar.set_description(f'k:{k_sum/(k_count+1e-8):.2f} acc:{acc_sum/acc_count:.4f}')
    print(f'dataset average k = {k_sum/(k_count+1e-8):.3f}, dataset accuracy = {acc_sum/acc_count*100:.2f} %')

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