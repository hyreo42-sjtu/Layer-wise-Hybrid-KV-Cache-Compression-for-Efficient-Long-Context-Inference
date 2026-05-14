from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from .base import attention_entropy
from .pyramidkv import PyramidKVPress, select_keep_indices, snapkv_importance


@dataclass
class AdaptivePyramidKVPress(PyramidKVPress):
    budget_min: float = 0.2
    budget_max: float = 0.8
    warmup_steps: int = 128
    entropy_ema: Dict[int, float] = field(default_factory=dict)
    name: str = "adaptive_pyramidkv"

    def reset(self) -> None:
        self.entropy_ema.clear()

    def update_entropy(self, layer_idx: int, attention: Optional[torch.Tensor]) -> None:
        ent = attention_entropy(attention)
        if ent is None:
            return
        if layer_idx not in self.entropy_ema:
            self.entropy_ema[layer_idx] = ent
        else:
            self.entropy_ema[layer_idx] = 0.9 * self.entropy_ema[layer_idx] + 0.1 * ent

    def adaptive_ratio(self, layer_idx: int) -> float:
        if not self.entropy_ema:
            return self.budget_ratio
        values = list(self.entropy_ema.values())
        h_min = min(values)
        h_max = max(values)
        h = self.entropy_ema.get(layer_idx, sum(values) / len(values))
        norm_h = (h - h_min) / (h_max - h_min + 1e-8)
        return self.budget_min + (self.budget_max - self.budget_min) * norm_h

    def layer_budget(self, seq_len: int, layer_idx: int, num_layers: int) -> int:
        ratio = self.adaptive_ratio(layer_idx)
        return max(2, min(seq_len, int(round(seq_len * ratio))))

    def compress_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: Optional[torch.Tensor],
        layer_idx: int,
        num_layers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.update_entropy(layer_idx, attention)
        seq_len = key.shape[-2]
        budget = self.layer_budget(seq_len, layer_idx, num_layers)
        if budget >= seq_len:
            return key, value
        scores = snapkv_importance(attention, seq_len, self.score_window, self.kernel_size, key.device)
        keep_idx = select_keep_indices(scores, seq_len, budget, self.sink_size, self.recent_size, key.device)
        return key.index_select(dim=-2, index=keep_idx).contiguous(), value.index_select(dim=-2, index=keep_idx).contiguous()
