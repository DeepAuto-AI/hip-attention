# version 1.0
from hip_attn.models.modeling_llama import LlamaForCausalLM as HipLlamaForCausalLM

# api
from hip_attn.utils.attention import custom_attention
from hip_attn.v1_0.attention1_block_gpu import hip_attention as hip_attention_10
from hip_attn.v1_0.attention1_block_gpu import (
    paged_hip_attention as paged_hip_attention_10,
)

# version 1.1
from hip_attn.v1_1.attention2_draft_prefetch import (
    HiPAttentionArgs as HiPAttentionArgs11,
)
from hip_attn.v1_1.attention2_draft_prefetch import (
    HiPAttentionOutputMetadata as HiPAttentionOutputMetadata11,
)
from hip_attn.v1_1.attention2_draft_prefetch import hip_attention as hip_attention_11
from hip_attn.v1_1.attention2_draft_prefetch import (
    paged_hip_attention as paged_hip_attention_11,
)
from hip_attn.v1_1.attention2_draft_prefetch import (
    paged_varlen_hip_attention as paged_varlen_hip_attention_11,
)
from hip_attn.v1_1.attention2_draft_prefetch import (
    varlen_hip_attention as varlen_hip_attention_11,
)

# version 1.2
from hip_attn.v1_2.attention_extend import (
    dual_stage_quadratic_hip_attention as hip_attention_12,
)
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs as HiPAttentionArgs12
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionCacheAccessStatistics as HiPAttentionCacheAccessStatistics12,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionOutputMetadata as HiPAttentionOutputMetadata12,
)
from hip_attn.v1_2.attention_metadata import ScanStage as ScanStage12
from hip_attn.v1_2.uvm_gpu_cache import GPUCache as GPUCache12
from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache as HiPOffloadCache12
from hip_attn.v1_2.uvm_gpu_cache import UVMCache as UVMCache12

hip_attention = hip_attention_12
HiPAttentionArgs = HiPAttentionArgs12
HiPAttentionOutputMetadata = HiPAttentionOutputMetadata12

__all__ = [
    # version 1.0
    "hip_attention_10",
    "paged_hip_attention_10",
    # version 1.1
    "hip_attention_11",
    "paged_hip_attention_11",
    "varlen_hip_attention_11",
    "paged_varlen_hip_attention_11",
    "HiPAttentionArgs11",
    "HiPAttentionOutputMetadata11",
    # version 1.2
    "hip_attention_12",
    "HiPAttentionArgs12",
    "HiPAttentionOutputMetadata12",
    "HiPAttentionCacheAccessStatistics12",
    "ScanStage12",
    "HiPOffloadCache12",
    "GPUCache12",
    "UVMCache12",
    # general purpose APIs, up to date for stable API
    "hip_attention",
    "HiPAttentionArgs",
    "HiPAttentionOutputMetadata",
    "custom_attention",
    "HipLlamaForCausalLM",
]
