# QKV 投影、布局与缓存

Q、K、V（Query、Key、Value）不是三类独立硬件单元，而是同一 hidden state 经三组线性权重投影得到的三个张量。NPU 的重点是复用输入 `X`、减少 QKV 中间结果的搬运，并使随后 Attention 与 [[npu/operator/transformer common/KV Cache|KV Cache]] 的访问连续。

## 数学语义与形状

设输入为 `X[B, S, H]`：`B` 为 batch，`S` 为序列长度，`H` 为 hidden size。以 `N_h` 个 query head、每 head 维度 `D` 为例，通常 `H = N_h × D`：

```text
Q = X × Wq + bq    Q[B, S, N_h, D]
K = X × Wk + bk    K[B, S, N_kv, D]
V = X × Wv + bv    V[B, S, N_kv, D]
```

- **Q** 表示当前位置要检索什么；它参与 `QKᵀ` 的左操作数。
- **K** 表示每个位置可被匹配的索引；它参与 `QKᵀ` 的右操作数。
- **V** 是与 K 同位置的内容；Attention 概率最终对它加权。
- 标准多头注意力（MHA）中 `N_kv = N_h`；分组查询注意力（GQA）中多个 Q head 共享一个 KV head；多查询注意力（MQA）则 `N_kv = 1`。后两者主要减少 Decode 中 KV Cache 的容量和读带宽。

投影后常将张量重排成 `[B, N_head, S, D]`，令每个 head 的 token 维连续，从而让 `QKᵀ` 与 `P×V` 对应 [[npu/operator/tensor operator/GEMM|GEMM]] 的规则 tile。这个 reshape/transpose 属于 [[npu/operator/tensor operator/Layout Transform|Layout Transform]]，最好和投影输出或 Attention 读取融合，避免完整中间张量写回 DRAM。

## 合并 QKV GEMM

推理中常按输出通道拼接权重：

```text
Wqkv = concat(Wq, Wk, Wv, axis=output_channel)
[Q | K | V] = X × Wqkv + bqkv
```

对于 MHA，`Wqkv` 的形状可看作 `[H, 3H]`；GQA/MQA 中 K/V 的输出通道较少，拼接宽度为 `H + 2 × N_kv × D`。一次较宽的 GEMM 有三项收益：

1. `X` tile 只从 SRAM 读取一次，且同一权重搬运与 PE 阵列调度更规则。
2. 共享同一输出 tile 的累加、bias 和量化/类型转换路径，减少命令和同步开销。
3. 输出可以在片上立即切分为 Q、K、V，接续 [[npu/operator/transformer common/RoPE|RoPE]]、Cache append 或 Attention kernel。

拆成三次 GEMM 有时仍合理：例如硬件阵列不能高效处理极宽 N 维、Q 与 K/V 精度不同，或跨注意力的 K/V 已经预计算。选择依据应是实际 tile 利用率、SRAM 容量和片外访存量，而不是仅看算术次数。

## RoPE、mask 与写入顺序

对使用 RoPE 的 decoder-only 模型，Q 和 K 在投影后、score 前按当前位置旋转；V 不旋转。增量 Decode 最常见的正确顺序是：

```text
X → QKV GEMM → split / head layout → RoPE(Q, K)
  → K、V 追加到本层 KV Cache → Q 与已缓存 K/V 做 Attention
```

这样 Cache 中的 K 已是旋转后的表示，后续 query 可直接读取。若设计选择缓存旋转前 K，则每次读取 K 时必须以其原始 position 重新旋转；两种策略不能混用。对 decoder self-attention，新 token 的 K/V 必须先纳入本次可见范围，随后通过 causal mask 禁止访问未来位置。

## Prefill 与 Decode 的数据流

| 阶段 | QKV 产生方式 | K/V 去向 | 主要硬件关注点 |
| --- | --- | --- | --- |
| Prefill | `S` 个 token 成批投影，M 维较大 | 当前块参与 Attention，并按 token 追加 Cache | 合并 GEMM 吞吐、Q/K/V tile 驻留、RoPE 融合 |
| Decode | 每个请求每层通常只投影 1 个新 token | 新 K/V 追加 Cache；历史 K/V 从 Cache 读出 | 小 GEMM 利用率、Cache append 延迟、KV 读带宽 |

Prefill 中不应仅因已经生成了 K/V 就跳过当前 prompt 内的 Attention；它仍需按 mask 计算每个 query 对当前及历史 token 的上下文。Decode 中则禁止重算历史 token 的 K/V，这正是 [[npu/operator/transformer common/KV Cache|KV Cache]] 的价值。

## 推荐的 K/V Cache 布局

逻辑上 Cache 可写为 `cache[layer][request][kv_head][position][D]`。物理布局通常再按 page 和 block 切分：

```text
layer → request/page → K or V → kv_head → token-in-page → D tile
```

- **layer 分离**：每层只读写自己的 K/V，地址计算和生命周期清晰。
- **K/V 分开或交错**：分开有利于 `QKᵀ` / `P×V` 分阶段流式读取；小 token block 交错则有利于一次 DMA 追加同位置 K、V。应按内存突发粒度实测选择。
- **D 连续且对齐**：一个 head 的 D 维应与向量宽度、GEMM `K_t` 对齐；不足部分用 mask 或零填充，不能参与 score/softmax 语义。
- **分页地址表**：长序列和 continuous batching 下，逻辑 position 到物理 page 的映射由调度器维护，使请求可增长、回收并避免大块搬移。

容量可粗略估算为 `2 × L × T × N_kv × D × bytes_per_element`，其中 `2` 表示 K 和 V，`L` 为层数，`T` 为已缓存 token 数。它解释了为什么 GQA/MQA、KV 量化和 Paged KV 对 Decode 吞吐非常重要。

## 实现检查点

1. `Wqkv` 的输出通道顺序、bias 切分和 head reshape 是否与模型导出图一致？
2. RoPE 的 position、pair 维度、sin/cos 精度，以及写入 Cache 前后的位置是否一致？
3. GQA 的 Q-to-KV head 映射是否正确，例如 `kv_head = q_head / group_size`？
4. Cache append 与读旧 page 是否有依赖事件，避免读到未写完的新 K/V？
5. K 的转置/pack 是否在 score kernel 内消化，避免为 `Kᵀ` 额外物化一份全长 Cache？

Attention 的分块计算见 [[Attention 硬件实现]]；Encoder、Decoder 和 Cross-Attention 对 Q/K/V 来源的差异见 [[Encoder、Decoder 与交叉注意力]]。
