from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from .base import KVPress, gather_kv


@dataclass
class KvpressSnapKVPress(KVPress):
    budget_ratio: float = 0.3
    window_size: int = 128
    kernel_size: int = 5
    name: str = "snapkv"

    def __post_init__(self) -> None:
        from kvpress.presses.snapkv_press import SnapKVPress

        self.press = SnapKVPress(
            compression_ratio=1.0 - self.budget_ratio,
            window_size=self.window_size,
            kernel_size=self.kernel_size,
        )

    def compress_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: Optional[torch.Tensor],
        layer_idx: int,
        num_layers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = key.shape[-2]
        keep = max(2, min(seq_len, int(round(seq_len * self.budget_ratio))))
        if attention is None or keep >= seq_len or seq_len <= self.window_size:
            return key, value
        dummy_module = SimpleNamespace(
            config=SimpleNamespace(
                num_attention_heads=attention.shape[1],
                num_key_value_heads=key.shape[1],
            ),
            head_dim=key.shape[-1],
        )
        hidden_states = torch.empty(key.shape[0], seq_len, key.shape[1] * key.shape[-1], device=key.device, dtype=key.dtype)
        scores = self.press.score(dummy_module, hidden_states, key, value, attention, {})
        scores = scores.mean(dim=(0, 1))
        keep_idx = scores.topk(keep).indices.sort().values.to(key.device)
        return gather_kv(key, value, keep_idx)
