# KV Cache

KV Cache（Key-Value 缓存）保存已经过 Attention 投影的 token 的 Key 与 Value，供自回归生成的后续步骤复用。它是**推理数据结构**，不是一种改变 Attention 数学语义的算子：对当前 query，仍计算

```text
O_t = Softmax(Q_t K_{0:t}ᵀ / √D + mask) V_{0:t}
```

区别在于，`K_{0:t-1}` 与 `V_{0:t-1}` 已在过去的步骤得到，不应再次经过 K/V 投影。

## 生命周期：Prefill、Decode 与释放

```text
Prefill（prompt，S 个 token）
  hidden[0:S] ── QKV projection ──► K[0:S], V[0:S] ──► append cache

Decode（第 t 个新 token；逐层执行）
  hidden[t] ── QKV projection ──► Q_t, K_t, V_t
                          └──────► append K_t, V_t
  Q_t × cached K[0:t]ᵀ ──► softmax ──► × cached V[0:t]

request 完成 / 被取消 ──► 回收其 page；共享前缀仅在引用计数归零后回收
```

- **Prefill**：一次处理整个提示词；K/V 可以成批写入 Cache，但当前 prompt 内的 attention 仍须按 causal mask 计算，不能因为 K/V 已产生就跳过 attention。
- **Decode**：每层每步只新增一个（或小批）token 的 K/V。Q 很短，读取历史 K/V 的流量随上下文线性增长，常成为 token/s 上限。
- **训练**：标准 teacher-forcing 训练对整段序列并行计算，通常没有跨 step 复用需求，主流框架也通常在训练时关闭 `use_cache` 以节省显存。不要把这表述为“KV Cache 天然不能反传”：这是标准训练路径的工程选择，而非对所有带缓存/重计算训练算法的普遍定理。

## 为什么只保存 K/V，而通常不保存历史 Q

在因果 self-attention 的第 `t` 步，输出只需要当前的 `Q_t` 去匹配所有历史 K，并用同位置 V 聚合。过去 token 的 `Q_i` 不再作为未来步骤的输入，因此缓存它通常不能避免后续计算；过去的 K/V 则会被每个未来 query 重复读取。Cross-attention 是一个例外的生命周期：encoder memory 的 K/V 可在 decoder 生成开始前投影一次，在各 Decode step 间只读复用，见 [[npu/implement/Encoder、Decoder 与交叉注意力]]。

## 容量与带宽模型

设 `L` 为层数、`B` 为并发请求数、`T` 为当前每请求缓存长度、`N_kv` 为 KV head 数、`D` 为 head dimension、`b` 为每元素字节数，则不含对齐、page 表和元数据的原始容量约为：

```text
KV bytes = 2 × L × B × T × N_kv × D × b
```

`2` 分别对应 K 和 V。容量随 `T` 和 `B` 线性增长；单 token Decode 的历史读取量也近似随 `T` 增长。GQA/MQA 通过令 `N_kv < N_q` 直接降低这两项成本；KV 量化降低 `b`，但要为 scale、反量化和端到端精度预留预算。

注意：不开 Cache 并非“指数级”重复计算。其代价取决于实现和生成长度：每一步都要重新投影越来越长的前缀，且会反复做前缀 attention；总量会迅速恶化，但应按具体的 projection 与 attention 复杂度分析，而不是笼统称为指数增长。

## NPU 布局与调度

逻辑索引常写为 `cache[layer][request][K|V][kv_head][position][D]`；物理实现再按固定 token page 切分：

```text
layer → request 的 page table → K 或 V → KV head → token-in-page → D tile
```

- **追加**：新 token 的 K/V 写入其逻辑末尾 page；写完成事件必须先于本 step 对该 K/V 的读取。
- **读取**：score 阶段流式读取 K page，`P×V` 阶段流式读取 V page；对长序列应和 [[npu/implement/实现/Attention 硬件实现|online softmax]] 配合，避免物化完整 score。
- **分页**：把逻辑连续 token 映射到非连续的物理块，支持请求增长、释放和 continuous batching，而无需搬移整段 Cache。PagedAttention 的核心收益正是降低碎片并支持按块共享，而不是让 Attention 可以忽略 page table。
- **布局选择**：K、V 分离通常利于两个流式阶段各自连续读取；K/V 交错可能有利于小块追加。选择应基于 DMA 突发长度、SRAM bank 冲突和实际 kernel 测量。
- **位置编码一致性**：若使用 [[RoPE]]，Cache 中 K 是旋转后还是旋转前必须固定；两种布局的读取路径不可混用。

## 验证清单

1. 增量 Decode 的 logits 是否与同精度的全序列 causal Prefill 对齐（允许既定归约/量化容差）？
2. 新 K/V 的写事件是否在本层 attention 读取前完成？不同 request 的 page table、长度和 causal 边界是否隔离？
3. page 回收或共享前缀 copy-on-write 时，是否存在仍在飞行的 DMA / kernel 引用？
4. GQA 的 `q_head → kv_head` 映射和缓存 head 数是否与导出模型一致？

## 可核验资料

- [Hugging Face Transformers — Caching](https://huggingface.co/docs/transformers/cache_explanation)：自回归生成中缓存的用途、cache layer 与不同 cache 策略。
- [Kwon et al., 2023 — PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：按 page 管理 KV Cache、降低碎片和跨请求共享的设计动机。
- [[npu/implement/QKV 投影、布局与缓存]]：本库中的投影形状、GQA/MQA 和 Cache 写入顺序。
