from dataclasses import dataclass
from torch import Tensor
from typing import Optional

@dataclass
class HiPAttentionOutputMetadata:
    indices: Tensor
    ks: Tensor
    ks_count: Tensor
    ks_start_end: Tensor