"""
Simple script for measuring memory. Just for testing purpose

Usage: python -m src.main.tests.test_perlin_opt_vram --k 64 --predictor-length 256
"""

from ...models import perlin_attention, perlin_opt
from .common_opt import init
import torch

FP_MIN = torch.finfo(torch.float32).min

trainer, model, tokenizer = init(skip_init_loaders=True, checkpoint_path='asdf')
device = trainer.device

layer = model.model.decoder.layers[0] # type: perlin_opt.OPTDecoderLayer
layer_self = layer.self_attn.perlin_self_attention
if hasattr(layer_self, '_attention_unwrap') and layer_self._attention_unwrap != None:
    layer = layer_self._attention_unwrap # type: perlin_attention.PerlinAttention
else:
    layer = layer_self.attention # type: perlin_attention.PerlinAttention

seq_len = 2048
q = torch.randn((1, 12, seq_len, 64), device=device)
k = q.clone()
v = q.clone()
attention_mask = (torch.arange(seq_len).view(1, seq_len) > torch.arange(seq_len).view(seq_len, 1)) * FP_MIN
attention_mask = attention_mask.to(device).view(1, 1, seq_len, seq_len)
attention_scores_truth = torch.randn((1, 12, seq_len, seq_len), device=device)
context_layer_truth = torch.randn((1, 12, seq_len, 64*12), device=device)
start_mem = torch.cuda.max_memory_allocated()
def measure():
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        y = layer(
            q=q, k=k, v=v,
            q_for_atten=q, k_for_atten=k, v_for_atten=v,
            q_for_score=q, k_for_score=k,
            attention_mask=attention_mask,
            attention_scores_truth=attention_scores_truth,
            context_layer_truth=context_layer_truth,
        )
    mem = torch.cuda.max_memory_allocated() - start_mem
    return y, mem / 1024 / 1024
y, mem = measure()
print('mem took', mem)