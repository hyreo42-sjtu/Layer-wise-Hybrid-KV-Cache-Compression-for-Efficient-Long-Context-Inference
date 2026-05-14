import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import List

import torch


@dataclass
class Metrics:
    dataset: str
    method: str
    run_id: int
    input_length: int
    generated_tokens: int
    ppl: float
    ttft_sec: float
    tpot_ms: float
    throughput_tok_s: float
    peak_memory_mb: float
    kv_cache_memory_mb: float


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


def save_results(rows: List[Metrics], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = [asdict(row) for row in rows]
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "results.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
