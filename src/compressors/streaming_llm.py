from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .base import KVPress, attention_scores, gather_kv, unique_sorted


@dataclass
class StreamingLLMPress(KVPress):
    sink_size: int = 4
    window_size: int = 512
    use_snapkv_enhance: bool = False
    top_k: int = 128
    name: str = "streamingllm"

    def compress_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: Optional[torch.Tensor],
        layer_idx: int,
        num_layers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = key.shape[-2]
        if seq_len <= self.sink_size + self.window_size:
            return key, value
        device = key.device
        sink_end = min(self.sink_size, seq_len)
        sink_idx = torch.arange(0, sink_end, device=device, dtype=torch.long)
        window_start = max(sink_end, seq_len - self.window_size)
        window_idx = torch.arange(window_start, seq_len, device=device, dtype=torch.long)
        if self.use_snapkv_enhance and window_idx.numel() > self.top_k:
            scores = attention_scores(attention, seq_len, device)
            window_scores = scores[window_idx]
            selected = torch.topk(window_scores, k=min(self.top_k, window_scores.numel())).indices
            window_idx = window_idx[selected]
        keep_idx = unique_sorted(torch.cat([sink_idx, window_idx]))
        return gather_kv(key, value, keep_idx)
