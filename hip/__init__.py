# version 1.0
from hip.models.hip_attention.attention1_block_gpu \
    import hip_attention as hip_attention_10
from hip.models.hip_attention.attention1_block_gpu \
    import paged_hip_attention as paged_hip_attention_10

# version 1.1
from hip.models.hip_attention.attention2_draft_prefetch \
    import hip_attention as hip_attention_11
from hip.models.hip_attention.attention2_draft_prefetch \
    import paged_hip_attention as paged_hip_attention_11
from hip.models.hip_attention.attention2_draft_prefetch \
    import varlen_hip_attention as varlen_hip_attention_11
from hip.models.hip_attention.attention2_draft_prefetch \
    import paged_varlen_hip_attention as paged_varlen_hip_attention_11
from hip.models.hip_attention.attention2_draft_prefetch \
    import HiPAttentionArgs as HiPAttentionArgs11
from hip.models.hip_attention.attention2_draft_prefetch \
    import HiPAttentionOutputMetadata as HiPAttentionOutputMetadata11

# version 1.2
from hip.models.hip_attention.gen3.attention_extend \
    import dual_stage_quadratic_hip_attention as hip_attention_12
from hip.models.hip_attention.gen3.attention_metadata \
    import HiPAttentionArgs as HiPAttentionArgs12
from hip.models.hip_attention.gen3.attention_metadata \
    import HiPAttentionOutputMetadata as HiPAttentionOutputMetadata12
from hip.models.hip_attention.gen3.attention_metadata \
    import HiPAttentionCacheAccessStatistics as HiPAttentionCacheAccessStatistics12
from hip.models.hip_attention.gen3.attention_metadata \
    import HiPOffloadCache as HiPOffloadCache12
from hip.models.hip_attention.gen3.attention_metadata \
    import ScanStage as ScanStage12

# api
from hip.models.attention \
    import custom_attention
from hip.models.modeling_llama \
    import LlamaForCausalLM as HipLlamaForCausalLM

# NOTE: V1.1 is used for main API (for temporary)
hip_attention = hip_attention_11
paged_hip_attention = paged_hip_attention_11
varlen_hip_attention = varlen_hip_attention_11
paged_varlen_hip_attention = paged_varlen_hip_attention_11
HiPAttentionArgs = HiPAttentionArgs11
HiPAttentionOutputMetadata = HiPAttentionOutputMetadata11

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
    'HiPAttentionOutputMetadata11',

    # version 1.2
    'hip_attention_12',
    'HiPAttentionArgs12',
    'HiPAttentionOutputMetadata12',
    'HiPAttentionCacheAccessStatistics12',
    'HiPOffloadCache12',
    'ScanStage12',
    
    # general purpose APIs, up to date for stable API
    'hip_attention',
    'paged_hip_attention',
    'varlen_hip_attention',
    'paged_varlen_hip_attention'
    'HiPAttentionArgs',
    'HiPAttentionOutputMetadata',
    
    'custom_attention',
    'HipLlamaForCausalLM',
]