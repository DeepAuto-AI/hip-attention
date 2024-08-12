from hip.models.hip_attention.attention1_block_gpu import hip_attention as hip_attention_10
from hip.models.hip_attention.attention1_block_gpu import paged_hip_attention as paged_hip_attention_10
from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec import hip_attention as hip_attention_11
from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec import hip_attention_paged_kv_cache as paged_hip_attention_11
from hip.models.attention import custom_attention
from hip.models.modeling_llama import LlamaForCausalLM as HipLlamaForCausalLM

hip_attention = hip_attention_11
paged_hip_attention = paged_hip_attention_11

__version__ = '1.1.0'

__all__ = [
    # version 1.0
    'hip_attention_10',
    'paged_hip_attention_10',
    
    # version 1.1
    'hip_attention_11',
    'paged_hip_attention_11',
    
    # general purpose APIs, up to date for stable API
    'hip_attention',
    'paged_hip_attention',
    'custom_attention',
    'HipLlamaForCausalLM',
]