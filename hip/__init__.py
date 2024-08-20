# version 1.0
from hip.models.hip_attention.attention1_block_gpu \
    import hip_attention as hip_attention_10
from hip.models.hip_attention.attention1_block_gpu \
    import paged_hip_attention as paged_hip_attention_10

# version 1.1
from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec \
    import hip_attention as hip_attention_11
from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec \
    import paged_hip_attention as paged_hip_attention_11
from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec \
    import varlen_hip_attention as varlen_hip_attention_11
from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec \
    import paged_varlen_hip_attention as paged_varlen_hip_attention_11
from hip.models.hip_attention.attention2_draft_causal_batch_gpu_fused_vec \
    import HiPAttentionArgs as HiPAttentionArgs11

# api
from hip.models.attention \
    import custom_attention
from hip.models.modeling_llama \
    import LlamaForCausalLM as HipLlamaForCausalLM

hip_attention = hip_attention_11
paged_hip_attention = paged_hip_attention_11
varlen_hip_attention = varlen_hip_attention_11
paged_varlen_hip_attention = paged_varlen_hip_attention_11
HiPAttentionArgs = HiPAttentionArgs11

__version__ = '1.1.0'

__all__ = [
    # version 1.0
    'hip_attention_10',
    'paged_hip_attention_10',
    
    # version 1.1
    'hip_attention_11',
    'paged_hip_attention_11',
    'varlen_hip_attention_11',
    'paged_varlen_hip_attention_11',
    'HiPAttentionArgs11',
    
    # general purpose APIs, up to date for stable API
    'hip_attention',
    'paged_hip_attention',
    'varlen_hip_attention',
    'paged_varlen_hip_attention'
    'HiPAttentionArgs',
    
    'custom_attention',
    'HipLlamaForCausalLM',
]