from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import KVPress, gather_kv, unique_sorted


def layer_budgets(seq_len: int, num_layers: int, ratio: float) -> List[int]:
    if ratio >= 0.999:
        return [seq_len] * num_layers
    avg_budget = max(2, min(seq_len, int(round(seq_len * ratio))))
    min_budget = min(avg_budget, min(64, max(2, seq_len // 32)))
    max_budget = min(seq_len, 2 * avg_budget - min_budget)
    if num_layers == 1:
        return [avg_budget]
    step = (max_budget - min_budget) / (num_layers - 1)
    budgets = [max(2, min(seq_len, int(round(max_budget - layer_idx * step)))) for layer_idx in range(num_layers)]
    delta = avg_budget * num_layers - sum(budgets)
    budgets[-1] = max(2, min(seq_len, budgets[-1] + delta))
    return budgets


def snapkv_importance(attention: Optional[torch.Tensor], seq_len: int, score_window: int, kernel_size: int, device: torch.device) -> torch.Tensor:
    if attention is None:
        return torch.arange(seq_len, device=device, dtype=torch.float32)
    attn = attention.detach().float()
    window = min(score_window, max(1, seq_len - 1))
    history_len = seq_len - window
    if history_len <= 0:
        return torch.ones(seq_len, dtype=attn.dtype, device=device)
    scores = attn[..., -window:, :history_len].mean(dim=-2)
    if kernel_size > 1:
        scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        scores = scores[..., :history_len]
    max_score = scores.max().item()
    scores = F.pad(scores, (0, window), value=max_score)
    return scores.mean(dim=(0, 1)).to(device)


def select_keep_indices(scores: torch.Tensor, seq_len: int, budget: int, sink_size: int, recent_size: int, device: torch.device) -> torch.Tensor:
    budget = min(seq_len, max(2, budget))
    if budget >= seq_len:
        return torch.arange(seq_len, device=device)
    sink_idx = torch.arange(0, min(sink_size, seq_len), device=device, dtype=torch.long)
    recent_start = max(sink_idx.numel(), seq_len - min(recent_size, max(1, budget - sink_idx.numel())))
    recent_idx = torch.arange(recent_start, seq_len, device=device, dtype=torch.long)
    fixed = unique_sorted(torch.cat([sink_idx, recent_idx]))
    remaining = budget - fixed.numel()
    if remaining <= 0:
        return fixed[-budget:].sort().values
    candidate_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    candidate_mask[fixed] = False
    candidate_idx = candidate_mask.nonzero(as_tuple=False).flatten()
    candidate_scores = scores.to(device)[candidate_idx]
    top = torch.topk(candidate_scores, k=min(remaining, candidate_scores.numel())).indices
    return unique_sorted(torch.cat([fixed, candidate_idx[top]]))


@dataclass
class PyramidKVPress(KVPress):
    budget_ratio: float = 0.3
    sink_size: int = 4
    recent_size: int = 128
    score_window: int = 128
    kernel_size: int = 5
    name: str = "pyramidkv"

    def layer_budget(self, seq_len: int, layer_idx: int, num_layers: int) -> int:
        return layer_budgets(seq_len, num_layers, self.budget_ratio)[layer_idx]

    def compress_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: Optional[torch.Tensor],
        layer_idx: int,
        num_layers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = key.shape[-2]
        budget = self.layer_budget(seq_len, layer_idx, num_layers)
        if budget >= seq_len:
            return key, value
        scores = snapkv_importance(attention, seq_len, self.score_window, self.kernel_size, key.device)
        keep_idx = select_keep_indices(scores, seq_len, budget, self.sink_size, self.recent_size, key.device)
        return gather_kv(key, value, keep_idx)
