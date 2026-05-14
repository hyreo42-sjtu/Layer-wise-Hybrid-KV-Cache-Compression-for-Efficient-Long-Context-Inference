import argparse
import csv
import gc
import math
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.compressors import AdaptivePyramidKVPress, HybridCompressor, KVPress, KvpressSnapKVPress, PyramidKVPress, StreamingLLMPress
from src.utils.cache import cache_to_tuple, kv_cache_memory_mb, patch_gpt_neox_layerwise_attention_mask, tuple_to_cache
from src.utils.data import build_token_sample
from src.utils.metrics import Metrics, peak_memory_mb, reset_peak_memory, save_results, synchronize_if_needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise Hybrid KV Cache Compression Experiment")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "pg19"], choices=["wikitext", "pg19"])
    parser.add_argument("--methods", nargs="+", default=["baseline", "streaming", "streaming_snapkv", "snapkv", "pyramid_0.3", "pyramid_0.5", "mix_a", "mix_b"])
    parser.add_argument("--input_length", type=int, default=2048)
    parser.add_argument("--generate_length", type=int, default=256)
    parser.add_argument("--ppl_eval_tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=384)
    parser.add_argument("--hybrid_split_layer", type=int, default=8)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--score_window", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["flash_attention_2", "eager", "sdpa"])
    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def dtype_from_arg(dtype_arg: str):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device, need_attentions: bool):
    attn_impl = args.attn_implementation
    if need_attentions and attn_impl == "flash_attention_2":
        attn_impl = "eager"
        print("output_attentions is required for compression; falling back from flash_attention_2 to eager attention.")
    if attn_impl == "eager":
        patch_gpt_neox_layerwise_attention_mask()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_arg(args.dtype),
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def compress_past(past_key_values, attentions, compressor: Optional[KVPress]):
    if compressor is None:
        return past_key_values
    past_tuple = cache_to_tuple(past_key_values)
    num_layers = len(past_tuple)
    compressed = []
    for layer_idx, (key, value) in enumerate(past_tuple):
        attention = attentions[layer_idx] if attentions is not None else None
        new_key, new_value = compressor.compress_layer(key, value, attention, layer_idx, num_layers)
        compressed.append((new_key, new_value))
    return tuple_to_cache(tuple(compressed))


def prefill(model, input_ids: torch.Tensor, compressor: Optional[KVPress]):
    need_attn = compressor is not None
    if compressor is not None:
        compressor.reset()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True, output_attentions=need_attn, return_dict=True)
    past = compress_past(outputs.past_key_values, outputs.attentions if need_attn else None, compressor)
    return outputs.logits, past, kv_cache_memory_mb(past)


def generate_and_measure(model, input_ids: torch.Tensor, compressor: Optional[KVPress], generate_length: int, device: torch.device):
    reset_peak_memory(device)
    synchronize_if_needed(device)
    start = time.perf_counter()
    logits, past, kv_mb = prefill(model, input_ids, compressor)
    cur = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    synchronize_if_needed(device)
    first_done = time.perf_counter()
    generated = [cur]
    next_position = input_ids.shape[1]
    for _ in range(generate_length - 1):
        with torch.inference_mode():
            position_ids = torch.tensor([[next_position]], device=device, dtype=torch.long)
            outputs = model(input_ids=cur, past_key_values=past, position_ids=position_ids, use_cache=True, return_dict=True)
        past = outputs.past_key_values
        cur = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur)
        next_position += 1
    synchronize_if_needed(device)
    end = time.perf_counter()
    ttft = first_done - start
    tpot_ms = (end - first_done) / max(1, generate_length - 1) * 1000
    throughput = generate_length / max(1e-9, end - start)
    return torch.cat(generated, dim=1), ttft, tpot_ms, throughput, peak_memory_mb(device), kv_mb


def evaluate_ppl(model, input_ids: torch.Tensor, compressor: Optional[KVPress], eval_tokens: int, device: torch.device) -> float:
    context = input_ids[:, :-eval_tokens]
    targets = input_ids[:, -eval_tokens:]
    logits, past, _ = prefill(model, context, compressor)
    losses = []
    prev_logits = logits[:, -1, :]
    for i in range(eval_tokens):
        target = targets[:, i]
        losses.append(F.cross_entropy(prev_logits.float(), target, reduction="none"))
        with torch.inference_mode():
            position_ids = torch.tensor([[context.shape[1] + i]], device=device, dtype=torch.long)
            outputs = model(input_ids=target.view(1, 1), past_key_values=past, position_ids=position_ids, use_cache=True, return_dict=True)
        past = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :]
    return float(math.exp(torch.cat(losses).mean().item()))


def build_compressor(method: str, num_layers: int, args: argparse.Namespace) -> Tuple[str, Optional[KVPress]]:
    if method == "baseline":
        return "baseline", None
    if method == "streaming":
        return "streaming", StreamingLLMPress(args.sink_size, args.window_size, False, args.top_k)
    if method == "streaming_snapkv":
        return "streaming_snapkv", StreamingLLMPress(args.sink_size, args.window_size, True, args.top_k)
    if method == "snapkv":
        return "snapkv", KvpressSnapKVPress(0.3, args.score_window, args.kernel_size)
    if method.startswith("adaptive_pyramid_"):
        ratio = float(method.rsplit("_", 1)[1])
        return method, AdaptivePyramidKVPress(ratio, args.sink_size, args.recent_size, args.score_window, args.kernel_size)
    if method.startswith("pyramid_"):
        ratio = float(method.split("_", 1)[1])
        return method, PyramidKVPress(ratio, args.sink_size, args.recent_size, args.score_window, args.kernel_size)
    if method == "mix_a":
        method_name = f"mix_a_split{args.hybrid_split_layer}" if args.hybrid_split_layer is not None else "mix_a"
        return method_name, HybridCompressor(
            num_layers,
            StreamingLLMPress(args.sink_size, args.window_size, False, args.top_k),
            AdaptivePyramidKVPress(0.3, args.sink_size, args.recent_size, args.score_window, args.kernel_size),
            args.hybrid_split_layer,
        )
    if method == "mix_b":
        return "mix_b", HybridCompressor(
            num_layers,
            StreamingLLMPress(args.sink_size, args.window_size, True, args.top_k),
            AdaptivePyramidKVPress(0.3, args.sink_size, args.recent_size, args.score_window, args.kernel_size),
            args.hybrid_split_layer,
        )
    raise ValueError(f"Unknown method: {method}")


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_existing_results(output_dir: str) -> List[Metrics]:
    path = os.path.join(output_dir, "results.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                Metrics(
                    row["dataset"],
                    row["method"],
                    int(row["run_id"]),
                    int(row["input_length"]),
                    int(row["generated_tokens"]),
                    float(row["ppl"]),
                    float(row["ttft_sec"]),
                    float(row["tpot_ms"]),
                    float(row["throughput_tok_s"]),
                    float(row["peak_memory_mb"]),
                    float(row["kv_cache_memory_mb"]),
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    need_attentions = any(method != "baseline" for method in args.methods)
    model, tokenizer = load_model_and_tokenizer(args, device, need_attentions)
    rows: List[Metrics] = load_existing_results(args.output_dir)
    total_tokens = args.input_length + max(args.ppl_eval_tokens, 1)
    for dataset in args.datasets:
        input_ids = build_token_sample(tokenizer, dataset, total_tokens).to(device)
        gen_input = input_ids[:, : args.input_length]
        ppl_input = input_ids[:, : args.input_length + args.ppl_eval_tokens]
        for method in args.methods:
            method_name, compressor = build_compressor(method, model.config.num_hidden_layers, args)
            for run_id in range(args.runs):
                cleanup(device)
                print(f"Running dataset={dataset} method={method_name} run={run_id}")
                ppl = evaluate_ppl(model, ppl_input, compressor, args.ppl_eval_tokens, device)
                _, ttft, tpot, throughput, peak_mb, kv_mb = generate_and_measure(model, gen_input, compressor, args.generate_length, device)
                rows.append(Metrics(dataset, method_name, run_id, args.input_length, args.generate_length, ppl, ttft, tpot, throughput, peak_mb, kv_mb))
                save_results(rows, args.output_dir)
                print(f"PPL={ppl:.4f} TTFT={ttft:.4f}s TPOT={tpot:.2f}ms Throughput={throughput:.2f} tok/s Peak={peak_mb:.1f}MB KV={kv_mb:.1f}MB")
    save_results(rows, args.output_dir)


if __name__ == "__main__":
    main()
