我需要完成一个语言模型高效推理的小组项目 coding 部分。我们之前已经手动实现了 PyramidKV 的基础版本。现在需要在此基础上实现一个**混合压缩框架**，具体要求如下。

## 1. 总体目标
构建一个可运行的推理评测脚本，实现：
- 底层统一使用 FlashAttention-2 加速（仅需模型加载参数设置）。
- 层间混合策略：浅层（前 L/2 层）使用 **StreamingLLM**，深层（后 L/2 层）使用 **Adaptive PyramidKV**（基于注意力熵动态调整压缩预算）。
- 在 StreamingLLM 的滑动窗口内增加 **SnapKV 风格的注意力分数筛选**（增强版），即窗口内保留最重要的 K 个 token + 固定 sink token。

## 2. 已有代码说明
- 我已有的 PyramidKV 实现：`pyramidkv.py` 中提供了一个类 `PyramidKVPress`，其核心方法 `compress(kv_cache, attention_mask, ...)` 接受当前层的 KV cache 和注意力分数，返回压缩后的 cache。预算分配为固定曲线（例如每层保留比例随层数线性下降）。
- 我还没有 StreamingLLM 和 SnapKV 的实现。

请生成完整的代码，包括：
- 实现 `StreamingLLMPress`（含窗口内 SnapKV 增强选项）
- 实现 `AdaptivePyramidKVPress`（继承或修改已有的 PyramidKVPress，增加基于注意力熵的动态预算分配）
- 实现混合调度器 `HybridCompressor`，根据层索引选择使用哪个 press
- 实现评测脚本 `run_hybrid_experiment.py`，加载 Pythia-70M，加载 pg-19 和 wikitext 数据，运行 baseline、单独方法、混合方法并输出指标（PPL, TTFT, TPOT, Throughput, 显存）

## 3. 技术细节

### 3.1 模型与基础配置
- 模型：`EleutherAI/pythia-70m`（共 24 层，hidden size=512，num_heads=8）
- 设备：GPU（CUDA），dtype=float16
- FlashAttention-2：加载模型时使用 `attn_implementation="flash_attention_2"`，需要 `transformers>=4.35` 和 `flash-attn` 库。

### 3.2 StreamingLLMPress（含 SnapKV 增强）
- 参数：`sink_size=4`, `window_size=512`, `use_snapkv_enhance=False`, `top_k=128`（当 enhance 启用时，在窗口内保留 attention 得分最高的 top_k 个 token）
- 逻辑：
  - 始终保留前 `sink_size` 个 token 的 KV。
  - 保留最近 `window_size` 个 token 的 KV（若未增强）。
  - 若增强：从最近 `window_size` 个 token 中，根据**累积注意力分数**（或当前 query 的注意力分数）选择得分最高的 `top_k` 个 token 保留。实现简单方式：在生成每个 token 时，获取当前层当前 head 对窗口内所有 token 的注意力分数，对所有 head 求和或平均，然后取 top_k 索引。为了性能，可以每隔几步计算一次（但为了代码简洁，允许每步都计算）。
- 注意：需要维护一个环形缓冲区或列表来存储 KV。

### 3.3 Adaptive PyramidKVPress
- 继承自你已有的 `PyramidKVPress`，但修改预算分配方式。
- 预热阶段：处理前 `warmup_steps=128` 个 token 时，记录每一层（仅深层，即后 L/2 层）的注意力熵。对于每一层，计算所有 head 的平均注意力分布熵：`H = -sum(p * log(p+1e-8))`，其中 p 是注意力 softmax 后的概率。
- 在预热结束后，为每一深层计算目标保留比例：
  - `budget_min = 0.2`, `budget_max = 0.8`
  - 归一化熵值：`norm_H = (H - H_min) / (H_max - H_min + 1e-8)`
  - `budget = budget_min + (budget_max - budget_min) * norm_H`
- 注意：PyramidKV 原本需要知道总的序列长度和 budget 来决定保留多少个 token。可在每次压缩时根据当前序列长度 `L` 计算保留数量 `keep = max(1, int(L * budget))`。
- 压缩时根据注意力分数（或简单根据位置？PyramidKV 原版使用最近的位置保留？请按 PyramidKV 论文方式：保留最近的重要 token 和 sink？）—— 为了简化，可以使用保留注意力分数最高的 token（类似 SnapKV）的位置，也可以保留最近的 token 与高分 token 混合。建议：保留 budget 数量的 token，其中 80% 为最近的重要 token（按注意力分数），20% 为全局高分 token。但可先实现简单版本：只保留注意力分数最高的 `keep` 个 token（不包括 sink？需要确定是否保留 sink）。由于深层通常有 sink 现象，建议始终保留前 4 个 token 再加上高分 token，这样总数不超过 keep+4。具体逻辑可以：从全序列中选出 top-(keep-4) 个高分 token，再加上前 4 个 sink token。

### 3.4 混合调度器 HybridCompressor
- 输入：模型层数 `num_layers=24`，浅层索引 0~11，深层索引 12~23。
- 定义 `get_compressor(layer_idx)`：
  - 若 layer_idx < 12：返回 StreamingLLMPress 实例（可配置参数）
  - 否则：返回 AdaptivePyramidKVPress 实例
- 在推理的每个 forward 之后，需要获取该层的 KV cache 和注意力分数（如果可能从模型内部获取）。由于标准 transformers 不直接暴露每层的注意力分数，我们需要 **hook** 或 **修改模型 forward**。简单方案：使用 `transformers` 的 `output_attentions=True` 来获取所有层的注意力概率（shape: [batch, num_heads, seq_len, seq_len]），然后在每层生成后对 KV cache 进行压缩。
- 或者，更简洁但可能略低效的方法是：在生成每个 token 后，重新计算一次当前层的注意力分数（仅对当前 query 和当前 cache 中的 keys 做点积）。但为了性能，建议使用 `output_attentions` 一次获取所有层的注意力，然后每层压缩。

### 3.5 评测脚本
- 实现函数 `evaluate(model, dataloader, compressor, max_length, gen_length)` 返回 PPL, TTFT, TPOT, throughput, peak_memory。
- PPL 计算：对前缀部分计算 loss（使用模型前向，不生成），然后取 exp(mean loss)。对于生成部分，可以计算生成 token 的交叉熵。
- TTFT：从输入结束到第一个 token 输出的时间。
- TPOT：从第一个 token 之后，每个 token 的平均生成时间（毫秒）。
- Throughput：生成的总 token 数 / 总生成时间（包含 TTFT）。
- 显存：使用 `torch.cuda.max_memory_allocated()` 记录峰值。
- 运行对比：
  - Dense Baseline（无压缩，但开启 FlashAttention）
  - StreamingLLM (标准，window=512)
  - PyramidKV (固定预算 0.3, 0.5)
  - SnapKV (如果有现成实现，否则跳过)
  - Mix-A（双层混合，浅层 StreamingLLM 标准，深层 Adaptive PyramidKV）
  - Mix-B（双层混合 + 浅层 StreamingLLM 增强版，即窗口内 SnapKV 筛选）
- 输出 CSV 或打印表格。

## 4. 文件结构要求
- `src/`
  - `compressors/`
    - `base.py` (抽象基类)
    - `streaming_llm.py` (实现 StreamingLLMPress)
    - `pyramidkv.py` (原有的 PyramidKV 实现，保留)
    - `adaptive_pyramidkv.py` (新，继承或修改)
    - `hybrid.py` (混合调度器)
  - `utils/` (数据加载、指标计算)
  - `run.py` (主入口)
- `requirements.txt` (包含 `torch, transformers, datasets, accelerate, flash-attn`)
- `README.md` (简要运行说明)

## 5. 其他要求
- 代码必须能直接在 GPU 上运行，处理 pg-19 长文本时内存不爆炸（使用生成时增量压缩）。
- 注意边界情况：序列长度小于窗口大小时不压缩。
- 提供清晰的注释和日志输出（如每层压缩后 cache 大小）。
- 生成完成后，自动保存结果到 `results/` 文件夹（JSON 或 CSV）。