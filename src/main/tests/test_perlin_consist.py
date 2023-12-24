"""
Check consistency between optimized implementation (for testing) and non optimized implementation (for training)

Usage: python -m src.main.tests.test_perlin_consist

NOTE: this test script is outdated. if you want to test causal, please use .test_perlin_opt_consist
"""

import os, tqdm, gc
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
from transformers import logging
logging.set_verbosity_error()

import torch, time, random, json
from transformers import AutoConfig
from ...models.hf_bert import BertLayer
from ...models.hf_bert import BertModel as TeacherBertModel
from ...models import perlin_bert
from ...models.perlin_attention import PerlinSelfAttention
from ...models.perlin_bert import BertModel, BertSelfAttention
from ...models.perlin_attention.config import PerlinAttentionConfig, register_default_config
from ...utils import seed, get_bench
import torch.nn.functional as F

get_bench().activate_temp_buffers = True
seed()

device = torch.device('cuda')

config = AutoConfig.from_pretrained('bert-base-uncased')
config.max_position_embeddings = 4096

register_default_config(PerlinAttentionConfig(
    performer_nb_factor=8,
    lora_enabled=False,
    lora_in_approx_enabled=False,
    partial_attention_scaler=True,
    k_flatten=True,
))

perlin = BertModel(config).to(device).eval()
for module in perlin.modules():
    if isinstance(module, BertSelfAttention):
        module.attention_method = 'perlin'

def set_benchmark(model, v):
    for module in model.modules():
        if hasattr(module, 'benchmarking'):
            module.benchmarking = v

BSIZE = 4
SEQ_LEN = 32
BENCH_PRECISION = torch.float32

input_ids = torch.randint(0, 10000, (BSIZE, SEQ_LEN)).to(device)
attention_mask = torch.ones((BSIZE, SEQ_LEN)).to(device)
for i in range(attention_mask.shape[0]):
    attention_mask[i, random.randint(16, attention_mask.shape[1]-1):] = 1
    # attention_mask[i, random.randint(16, attention_mask.shape[1]-1):] = 0
hidden_states = torch.randn((BSIZE, SEQ_LEN, config.hidden_size), device=device, dtype=BENCH_PRECISION)
attention_mask_expand = attention_mask.view(BSIZE, 1, 1, -1).contiguous()
attention_mask_expand = (1-attention_mask_expand)*(-32000)

layer = perlin.encoder.layer[0] # type: BertLayer
attention = layer.attention.self # type: BertSelfAttention
attention.teacher_attention_prob = torch.rand((BSIZE, 12, SEQ_LEN, SEQ_LEN), device=device)
attention.teacher_attention_score = torch.rand((BSIZE, 12, SEQ_LEN, SEQ_LEN), device=device)
attention.teacher_context_layer = torch.rand((BSIZE, SEQ_LEN, config.hidden_size), device=device)

with torch.no_grad():
    output_b0_0 = layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)[0]
    partial_attention_mask_b0 = (get_bench().get_temp_buffer('partial_attention_mask') > -1).float()
    attention_matrix_b0 = get_bench().get_temp_buffer('attention_matrix')
    q_for_score_b0 = get_bench().get_temp_buffer('q_for_score')
    k_for_score_b0 = get_bench().get_temp_buffer('k_for_score')
    partial_attention_scores_b0 = get_bench().get_temp_buffer('partial_attention_scores')

set_benchmark(perlin, True)
with torch.no_grad():
    output_b1_0 = layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)[0]
    partial_attention_mask_b1 = get_bench().get_temp_buffer('partial_attention_mask').to_dense().view(BSIZE, 12, SEQ_LEN, SEQ_LEN)
    attention_matrix_b1 = get_bench().get_temp_buffer('attention_matrix').to_dense().view(BSIZE, 12, SEQ_LEN, SEQ_LEN)
    q_for_score_b1 = get_bench().get_temp_buffer('q_for_score')
    k_for_score_b1 = get_bench().get_temp_buffer('k_for_score')
    partial_attention_scores_b1 = get_bench().get_temp_buffer('partial_attention_scores').to_dense().view(BSIZE, 12, SEQ_LEN, SEQ_LEN)

def exam(name, a, b, thresh=1e-5):
    loss = F.mse_loss(a, b, reduction='sum')
    if loss > thresh:
        os.makedirs('./saves/tests/test_perlin_consist/', exist_ok=True)
        path = f'./saves/tests/test_perlin_consist/{name}.pth'
        torch.save({
            'a':a, 'b':b, 'name': name
        }, path)
        print(f'{name} is not matched, dump to {path}. loss {loss}')
    else:
        print(f'{name} is matched. loss {loss}')

exam('partial_attention_mask', partial_attention_mask_b0, partial_attention_mask_b1)
exam('q_for_score', q_for_score_b0, q_for_score_b1)
exam('k_for_score', k_for_score_b0, k_for_score_b1)
exam('partial_attention_scores', partial_attention_scores_b0, partial_attention_scores_b1)
exam('output', output_b0_0, output_b1_0)