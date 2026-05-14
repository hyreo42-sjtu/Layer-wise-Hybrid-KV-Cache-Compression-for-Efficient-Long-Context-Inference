import os
import urllib.request
from typing import Iterable, List

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def iter_texts(dataset_name: str) -> Iterable[str]:
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        for row in dataset:
            text = row.get("text", "")
            if text and text.strip():
                yield text
    elif dataset_name == "pg19":
        try:
            dataset = load_dataset("deepmind/pg19", split="validation", streaming=True)
            for row in dataset:
                text = row.get("text", "")
                if text and text.strip():
                    yield text
        except Exception:
            yield from iter_pg19_official_validation()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def iter_pg19_official_validation() -> Iterable[str]:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "pg19_official")
    os.makedirs(cache_dir, exist_ok=True)
    split_file = hf_hub_download("deepmind/pg19", "data/validation_files.txt", repo_type="dataset")
    with open(split_file, encoding="utf-8") as f:
        validation_files = [line.strip() for line in f if line.strip()]
    for relative_path in validation_files:
        local_path = os.path.join(cache_dir, relative_path.replace("/", "_"))
        if not os.path.exists(local_path):
            url = f"https://storage.googleapis.com/deepmind-gutenberg/{relative_path}"
            urllib.request.urlretrieve(url, local_path)
        with open(local_path, encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            yield text


def build_token_sample(tokenizer, dataset_name: str, total_tokens: int) -> torch.Tensor:
    pieces: List[str] = []
    for text in iter_texts(dataset_name):
        pieces.append(text)
        ids = tokenizer("\n\n".join(pieces), return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.numel() >= total_tokens:
            return ids[:total_tokens].unsqueeze(0)
    raise RuntimeError(f"Dataset {dataset_name} did not provide enough tokens for {total_tokens} tokens.")
