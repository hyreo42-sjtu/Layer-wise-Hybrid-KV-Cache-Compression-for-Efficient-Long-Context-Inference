from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

PastKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def patch_gpt_neox_layerwise_attention_mask() -> None:
    try:
        import transformers.models.gpt_neox.modeling_gpt_neox as modeling_gpt_neox
    except Exception:
        return
    if getattr(modeling_gpt_neox.eager_attention_forward, "_layerwise_patched", False):
        return
    original_forward = modeling_gpt_neox.eager_attention_forward

    def patched_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if attention_mask is not None:
            key_len = key.shape[-2]
            mask_len = attention_mask.shape[-1]
            if mask_len > key_len:
                attention_mask = attention_mask[..., -key_len:]
            elif mask_len < key_len:
                attention_mask = F.pad(attention_mask, (key_len - mask_len, 0), value=0.0)
        return original_forward(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

    patched_eager_attention_forward._layerwise_patched = True
    modeling_gpt_neox.eager_attention_forward = patched_eager_attention_forward


def cache_to_tuple(past_key_values) -> PastKV:
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        if not hasattr(legacy_cache, "layers"):
            return legacy_cache
    if hasattr(past_key_values, "layers"):
        return tuple(
            (layer.keys, layer.values)
            for layer in past_key_values.layers
            if getattr(layer, "keys", None) is not None and getattr(layer, "values", None) is not None
        )
    return past_key_values


def tuple_to_cache(past_key_values: PastKV):
    return DynamicCache(past_key_values)


def kv_cache_memory_mb(past_key_values) -> float:
    past_tuple = cache_to_tuple(past_key_values)
    total_bytes = 0
    for key, value in past_tuple:
        total_bytes += key.numel() * key.element_size()
        total_bytes += value.numel() * value.element_size()
    return total_bytes / 1024 / 1024
