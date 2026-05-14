from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .adaptive_pyramidkv import AdaptivePyramidKVPress
from .base import KVPress
from .streaming_llm import StreamingLLMPress


@dataclass
class HybridCompressor(KVPress):
    num_layers: int
    shallow_press: StreamingLLMPress
    deep_press: AdaptivePyramidKVPress
    split_layer: Optional[int] = None
    name: str = "hybrid"

    def __post_init__(self) -> None:
        if self.split_layer is None:
            self.split_layer = self.num_layers // 2

    def get_compressor(self, layer_idx: int) -> KVPress:
        if layer_idx < int(self.split_layer):
            return self.shallow_press
        return self.deep_press

    def compress_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: Optional[torch.Tensor],
        layer_idx: int,
        num_layers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_compressor(layer_idx).compress_layer(key, value, attention, layer_idx, num_layers)

    def reset(self) -> None:
        self.shallow_press.reset()
        self.deep_press.reset()
