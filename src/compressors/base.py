from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

PastKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class KVPress(ABC):
    name: str = "base"

    @abstractmethod
    def compress_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: Optional[torch.Tensor],
        layer_idx: int,
        num_layers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


def attention_scores(attention: Optional[torch.Tensor], seq_len: int, device: torch.device) -> torch.Tensor:
    if attention is None:
        return torch.arange(seq_len, device=device, dtype=torch.float32)
    attn = attention.detach().float()
    if attn.ndim == 4:
        scores = attn[..., -1:, :seq_len].mean(dim=(0, 1, 2))
    elif attn.ndim == 3:
        scores = attn[..., -1:, :seq_len].mean(dim=(0, 1))
    else:
        scores = attn.reshape(-1, seq_len).mean(dim=0)
    return scores.to(device)


def attention_entropy(attention: Optional[torch.Tensor]) -> Optional[float]:
    if attention is None:
        return None
    probs = attention.detach().float().clamp_min(1e-8)
    ent = -(probs * probs.log()).sum(dim=-1).mean()
    return float(ent.item())


def unique_sorted(indices: torch.Tensor) -> torch.Tensor:
    return indices.unique().sort().values


def gather_kv(key: torch.Tensor, value: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return key.index_select(dim=-2, index=indices).contiguous(), value.index_select(dim=-2, index=indices).contiguous()
