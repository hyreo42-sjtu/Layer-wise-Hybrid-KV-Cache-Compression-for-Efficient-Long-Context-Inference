# Layer-wise Hybrid KV Cache Compression for Efficient Long-Context Inference

本项目实现了语言模型高效推理的小组部分：实现了 Adaptive PyramidKV 和基于 StreamingLLM with SnapKV Enhanced + AdaptiveKV 的混合压缩策略，并在 Pythia-70M 上比较了 Dense Baseline、StreamingLLM、SnapKV、PyramidKV、Adaptive PyramidKV 以及层间混合压缩策略的 PPL、速度和 KV cache memory。

## 方法

本项目的核心目标是在长上下文推理中减少 decode 阶段需要维护和读取的 KV cache，同时尽量保持语言建模质量。所有压缩方法均在 prefill 后对 `past_key_values` 进行压缩，随后使用压缩后的 KV cache 进入逐 token decode。

### Dense Baseline

Dense Baseline 不对 KV cache 做任何压缩，所有层都完整保留输入序列的 key/value 张量。该方法质量最高，作为 PPL、速度和 KV cache memory 的参照上界，但在长上下文场景中 KV cache 会随输入长度和生成长度线性增长。

### StreamingLLM

StreamingLLM 基于“attention sink + recent window”的观察：模型在长上下文推理时通常会持续关注开头少量 sink token，同时最近 token 对下一步预测最直接。因此该方法保留：

- **Sink tokens**：序列开头固定数量 token，用于维持稳定注意力锚点。
- **Recent window**：序列末尾最近窗口 token，用于保留局部上下文。

该方法实现简单、压缩率高，但会直接丢弃中间历史 token，因此当任务依赖远距离信息时 PPL 可能升高。

### StreamingLLM + SnapKV Enhance

StreamingLLM + SnapKV Enhance 在 StreamingLLM 的基础上引入 SnapKV 风格的重要性筛选。除固定 sink token 外，它不再简单保留整个最近窗口，而是在候选窗口中根据 attention score 选择 top-k token。这样可以在相同或更低 KV cache 预算下保留对当前预测更重要的历史 token。

该方法压缩最激进，在实验中 KV cache 可降至约 3.1 MB，但 wikitext 上 PPL 明显上升，说明过小的 top-k 会造成信息损失。

### SnapKV

SnapKV 使用最近 query 对历史 key 的 attention 分布估计 token 重要性，并保留高分 token。本项目直接调用 `kvpress` 库中的 `SnapKVPress` 获取标准 SnapKV 打分逻辑，再封装为兼容 GPT-NeoX `past_key_values` 的 compressor。

SnapKV 的优势是可以在固定压缩率下保留更具语义贡献的 token；实验中 wikitext PPL 优于固定预算 PyramidKV 0.3，说明注意力打分比单纯按层分配预算更细粒度。

### PyramidKV

PyramidKV 利用不同层对上下文信息需求不同的特点进行 layer-wise budget 分配。通常低层更偏向局部和表层模式，高层更偏向抽象语义，因此 PyramidKV 按层形成金字塔式预算：部分层保留更多 token，部分层保留更少 token，并结合 SnapKV 风格窗口 attention score 选择保留位置。

本项目实现了 `pyramid_0.3` 和 `pyramid_0.5` 两种预算比例，用于观察压缩强度变化对 PPL、速度和 KV cache memory 的影响。

### Adaptive PyramidKV

Adaptive PyramidKV 是本项目的创新方法之一。它在 PyramidKV 的 layer-wise 思路上进一步引入 attention entropy，动态衡量不同层当前 attention 分布的不确定性：

- **低熵层**：attention 更集中，说明少量 token 已能解释主要上下文信息，因此可以分配较小预算。
- **高熵层**：attention 更分散，说明模型可能需要更广泛的上下文信息，因此应保留更多 KV。

实现上，每层维护 attention entropy 的指数滑动统计，并将其归一化映射到 `budget_min` 到 `budget_max` 的预算区间。相比固定 PyramidKV，Adaptive PyramidKV 不再使用静态层预算，而是根据实际样本和层状态自适应分配 KV cache。实验中它在 wikitext 上以约 9.6 MB KV cache 取得 44.6455 PPL，相比固定 `pyramid_0.3` 同时减少 KV cache 并改善 PPL，体现了动态预算的优势。

### Mix-A 

Mix-A 是基础 hybrid 策略：浅层使用 StreamingLLM，深层使用 Adaptive PyramidKV。其设计动机是浅层表示更偏局部模式，可以采用简单的 sink + recent 策略强压缩；深层对语义整合更敏感，因此使用 Adaptive PyramidKV 保留更重要的全局信息。

### Mix-B

Mix-B 是本项目重点优化的 Hybrid 方法。它沿用“浅层强压缩、深层自适应保真”的层间分工，但进一步增强浅层压缩质量：

- **浅层**：使用 StreamingLLM + SnapKV Enhance。保留 sink token，并在候选窗口内通过 attention score 选择 top-k 重要 token，而不是机械保留固定窗口。
- **深层**：使用 Adaptive PyramidKV。根据各层 attention entropy 自适应分配预算，让语义层在需要时保留更多上下文。
- **层切分**：默认 `hybrid_split_layer=8`，即 24 层 Pythia-70M 中前 8 层采用浅层策略，后 16 层采用深层策略。
- **浅层 top-k**：默认 `top_k=384`，该值来自参数搜索，在 PPL 与 KV cache 压缩之间取得较好折中。

Mix-B 的核心思想是避免“一刀切”压缩：浅层通过更激进但带打分的 token 选择降低 KV cache，深层通过 Adaptive PyramidKV 保护语义信息。实验中 Mix-B 将 KV cache 从 baseline 的 48.0 MB 降至约 9.1 MB，在 pg19 上 PPL 仍接近 baseline，体现了较好的 PPL-KV cache 折中。

## 文件结构

```text
src/
  compressors/
    base.py
    streaming_llm.py
    snapkv.py
    pyramidkv.py
    adaptive_pyramidkv.py
    hybrid.py
  utils/
    cache.py
    data.py
    metrics.py
  run.py
run_hybrid_experiment.py
requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

本项目默认使用 `eager` attention 运行所有方法，以保证 baseline 与压缩方法处于相同 attention backend 下进行公平比较。对于需要 attention score 的压缩方法，prefill 阶段必须设置 `output_attentions=True`；当前 HuggingFace/FlashAttention-2 路径通常不返回完整 attention matrix，因此这类压缩方法不能直接使用 FlashAttention-2 获取打分。若仅单独测试 baseline 的极限速度，可以手动指定 `--attn_implementation flash_attention_2`，但该结果不应与压缩方法的 eager 结果直接比较。

## 运行方式

快速 smoke test：

```bash
python run_hybrid_experiment.py --datasets wikitext --methods baseline mix_a mix_b --input_length 512 --generate_length 32 --ppl_eval_tokens 32 --runs 1 --dtype float32 --attn_implementation eager
```

标准实验：

```bash
python run_hybrid_experiment.py --datasets wikitext pg19 --methods baseline streaming streaming_snapkv snapkv pyramid_0.3 pyramid_0.5 mix_a mix_b --input_length 2048 --generate_length 256 --ppl_eval_tokens 256 --runs 3
```

Mix-B 默认使用参数搜索得到的最优配置：

```bash
python run_hybrid_experiment.py --datasets wikitext pg19 --methods mix_b --input_length 2048 --generate_length 256 --ppl_eval_tokens 256 --runs 3
```

结果自动保存到：

```text
result/results.csv
result/results.json
```

## 指标

- **PPL**：困惑度。
- **TTFT**：从输入结束到第一个 token 输出的时间。
- **TPOT**：生成第 2 个到最后一个 token 的平均耗时。
- **Throughput**：总生成 token 数 / 总生成时间。
- **Peak Memory**：`torch.cuda.max_memory_allocated()` 记录的峰值显存。
- **KV Cache Memory**：压缩后传入 decode 阶段的 KV cache 实际张量大小。

## 实现说明

本项目为了兼容 `EleutherAI/pythia-70m` 的 GPT-NeoX 结构，沿用了个人部分中的运行时 patch，使不同层 KV cache 长度不一致时可以裁剪 causal mask，避免 HuggingFace 默认统一 mask 导致维度不匹配。

压缩流程采用 prefill 后压缩 `past_key_values`，再进入 autoregressive decode。对于需要 attention score 的方法，prefill 阶段设置 `output_attentions=True`。

为避免 baseline 独享 FlashAttention-2 加速造成速度指标不公平，脚本默认 `--attn_implementation eager`。如果用户显式指定 `flash_attention_2` 且运行压缩方法，脚本会自动回退到 `eager`，因为压缩算法需要 attention score 进行 token 重要性估计。

## 正式实验结果

实验设置：

- **模型**：`EleutherAI/pythia-70m`
- **数据集**：wikitext、pg19
- **输入长度**：2048
- **生成长度**：256
- **PPL 评估 token 数**：256
- **运行次数**：每个设置运行 3 次并取平均值
- **Attention backend**：统一使用 `eager`，避免 baseline 独享 FlashAttention-2 加速
- **运行命令**：

```bash
python run_hybrid_experiment.py --datasets wikitext pg19 --methods baseline streaming streaming_snapkv snapkv pyramid_0.3 pyramid_0.5 mix_a mix_b --input_length 2048 --generate_length 256 --ppl_eval_tokens 256 --runs 3 --dtype float32 --attn_implementation eager --output_dir result
```

结果文件保存于：

```text
result/results.csv
```

| Dataset | Method | PPL | TTFT (s) | TPOT (ms/token) | Throughput (tok/s) | Peak Memory (MB) | KV Cache (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext | baseline | 41.5484 | 0.0673 | 4.1753 | 226.21 | 730.6 | 48.0 |
| wikitext | streaming | 46.2643 | 0.0725 | 4.4867 | 210.42 | 1517.2 | 12.1 |
| wikitext | streaming_snapkv | 54.7621 | 0.0766 | 4.5309 | 207.86 | 1494.0 | 3.1 |
| wikitext | snapkv | 43.5751 | 0.0752 | 4.5235 | 208.38 | 1520.2 | 14.4 |
| wikitext | pyramid_0.3 | 45.0902 | 0.0783 | 4.5877 | 205.13 | 1518.1 | 14.4 |
| wikitext | pyramid_0.5 | 44.3109 | 0.0765 | 4.7292 | 199.84 | 1536.9 | 24.0 |
| wikitext | adaptive_pyramid_0.3 | 44.6455 | 0.1056 | 4.3853 | 209.19 | 1879.8 | 9.6 |
| wikitext | mix_a | 45.4775 | 0.0980 | 4.5604 | 203.05 | 1884.5 | 10.9 |
| wikitext | mix_b | 46.7319 | 0.0757 | 4.4124 | 213.22 | 1506.0 | 9.1 |
| pg19 | baseline | 45.6332 | 0.0697 | 4.3410 | 217.56 | 730.6 | 48.0 |
| pg19 | streaming | 45.3261 | 0.0736 | 4.4725 | 210.88 | 1517.2 | 12.1 |
| pg19 | streaming_snapkv | 46.0509 | 0.0778 | 4.5705 | 205.92 | 1494.0 | 3.1 |
| pg19 | snapkv | 45.3778 | 0.0755 | 4.5293 | 208.08 | 1520.2 | 14.4 |
| pg19 | pyramid_0.3 | 45.2712 | 0.0761 | 4.5541 | 206.97 | 1518.1 | 14.4 |
| pg19 | pyramid_0.5 | 45.3884 | 0.0767 | 4.5531 | 206.83 | 1536.9 | 24.0 |
| pg19 | adaptive_pyramid_0.3 | 45.6301 | 0.1074 | 4.4090 | 207.86 | 1879.8 | 9.6 |
| pg19 | mix_a | 45.5579 | 0.0975 | 4.5278 | 204.51 | 1884.5 | 10.9 |
| pg19 | mix_b | 45.3677 | 0.0784 | 4.3863 | 213.92 | 1506.0 | 9.1 |

### 结果分析

- **KV cache 压缩效果明显**：baseline 的 KV cache 为 48.0 MB；StreamingLLM 降至约 12.1 MB；StreamingLLM + SnapKV Enhance 降至约 3.1 MB；Adaptive PyramidKV 降至约 9.6 MB；优化后的 Mix-B 降至约 9.1 MB。说明各压缩方法均能显著降低 decode 阶段需要维护的 KV 张量规模。
- **Adaptive PyramidKV 体现动态预算优势**：在 wikitext 上，固定 `pyramid_0.3` 的 PPL 为 45.0902，KV cache 为 14.4 MB；Adaptive PyramidKV 0.3 的 PPL 改善到 44.6455，同时 KV cache 降至 9.6 MB。这说明基于 attention entropy 的预算分配可以避免固定比例预算的冗余，在部分层减少无效 KV 保留，同时为更不确定的层保留足够上下文。
- **Hybrid 方法具有 PPL-KV cache 折中优势**：Mix-B 的 KV cache 约为 9.1 MB，仅为 baseline 的约 19%，同时 pg19 PPL 为 45.3677，优于 baseline 的 45.6332。相比 StreamingLLM + SnapKV Enhance，Mix-B 用适度增加的 KV cache 显著改善了质量；相比 PyramidKV / SnapKV，Mix-B 进一步降低 KV cache。这正是 Hybrid 设计的主要优势：通过浅层强压缩和深层自适应保真，在生成质量和KV Cache之间取得更平衡的点。
- **Hybrid 方法在部分压缩方法中速度较优**：在 pg19 数据集上，Mix-B 的 TPOT 和 Throughput 优于多数压缩方法，说明 Hybrid 策略在降低 KV cache 的同时没有引入明显 decode 阶段负担。但重新使用统一 `eager` attention 重跑 baseline 后，baseline 的 TPOT 和 Throughput 仍略优于 Mix-B，因此不能简单认为当前工程环境下 Hybrid 已取得稳定端到端速度优势。
- **不同数据集表现存在差异**：wikitext 上 Mix-B 的 PPL 高于 baseline，说明该数据集对被压缩掉的上下文信息更敏感；pg19 上多数压缩方法 PPL 与 baseline 接近，说明长文档数据中局部延续性和冗余上下文更强，KV cache 压缩更容易保持质量。
- **端到端速度优势不明显**：虽然 KV cache memory 明显下降，但 TPOT 和 Throughput 相比 baseline 没有稳定提升。主要原因包括：`pythia-70m` 参数规模较小，单步 decode 的瓶颈不完全在 KV cache 读取；当前实现基于 HuggingFace eager 推理和 Python 逐 token 循环，调度开销较高；压缩方法需要 `output_attentions=True` 在 prefill 阶段获取 attention score，会引入额外计算和显存开销；实验运行在较短生成长度下，KV cache 压缩带来的理论收益尚未充分放大。
- **峰值显存偏高不等于 KV cache 变大**：压缩方法的 Peak Memory 高于 baseline，主要是因为 prefill 阶段为了计算 token 重要性需要保存 attention 矩阵。表中的 KV Cache Memory 才表示进入 decode 阶段后实际保留的 KV 张量大小。因此本项目的主要收益体现在 decode KV cache 占用，而不是当前工程实现下的峰值显存。

## Mix-B 参数搜索与长生成补充实验

为了优化混合策略，进一步对 Mix-B 的浅层 SnapKV-style `top_k` 和层切分位置 `hybrid_split_layer` 进行参数搜索，并加入单独的 Adaptive PyramidKV 方法作为对照。

参数搜索设置：

```text
input_length = 2048
generate_length = 256
ppl_eval_tokens = 256
runs = 3
top_k ∈ {128, 256, 384}
hybrid_split_layer ∈ {8, 12, 16}
```

结果文件保存于：

```text
result_grid_2048_256_v2/results.csv
```

### 2048/256 参数搜索结果

| Dataset | Method | PPL | TTFT (s) | TPOT (ms/token) | Throughput (tok/s) | Peak Memory (MB) | KV Cache (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext | baseline | 41.5484 | 0.0686 | 4.3329 | 218.32 | 730.6 | 48.0 |
| wikitext | adaptive_pyramid_0.3 | 44.6455 | 0.1056 | 4.3853 | 209.19 | 1879.8 | 9.6 |
| wikitext | mix_b_top128_split8 | 54.7621 | 0.0776 | 4.3686 | 214.87 | 1494.0 | 3.1 |
| wikitext | mix_b_top256_split8 | 49.7104 | 0.0944 | 4.4161 | 209.96 | 1500.0 | 6.1 |
| wikitext | mix_b_top384_split8 | 46.7319 | 0.0757 | 4.4124 | 213.22 | 1506.0 | 9.1 |
| pg19 | baseline | 45.6332 | 0.0678 | 4.3472 | 217.64 | 730.6 | 48.0 |
| pg19 | adaptive_pyramid_0.3 | 45.6301 | 0.1074 | 4.4090 | 207.86 | 1879.8 | 9.6 |
| pg19 | mix_b_top128_split8 | 46.0509 | 0.0769 | 4.3472 | 215.98 | 1494.0 | 3.1 |
| pg19 | mix_b_top256_split8 | 45.7585 | 0.0797 | 4.3495 | 215.34 | 1500.0 | 6.1 |
| pg19 | mix_b_top384_split8 | 45.3677 | 0.0784 | 4.3863 | 213.92 | 1506.0 | 9.1 |

完整参数搜索中，`split_layer=8/12/16` 对最终指标影响较小，主要差异来自浅层 `top_k`。随着 `top_k` 从 128 增大到 384，KV cache 从约 3.1 MB 增加到 9.1 MB，但 wikitext PPL 从 54.7621 明显改善到 46.7319，说明原始 Mix-B 的浅层筛选过于激进。综合质量和压缩率，后续长生成实验采用 `top_k=384, hybrid_split_layer=8` 作为优化后的 Mix-B 配置。
