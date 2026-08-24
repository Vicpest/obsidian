# Attention 硬件实现

缩放点积 Attention 为 `O = Softmax(QKᵀ / √d + mask)V`。它组合了 [[npu/operator/tensor operator/GEMM|GEMM]]、[[npu/operator/transformer common/Softmax|Softmax]]、[[npu/operator/tensor operator/Reduction|Reduction]] 和大量数据搬运。

## 基本数据路径

```text
X ── QKV GEMM ──► Q, K, V ──► [[npu/operator/transformer common/RoPE|RoPE]] (Q/K)
                                  │
                    Q tile × K tileᵀ
                                  ▼
                       score tile + scale + mask
                                  ▼
                  row max ─► exp ─► row sum ─► normalize
                                  ▼
                         probability tile × V tile
                                  ▼
                              output tile O
```

`QKᵀ` 与 `PV` 都可使用矩阵引擎；scale、mask、max、指数和归一化适合向量/SFU 与归约引擎。

## 分块与 online softmax

若直接物化 score，存储量为 `S × S`，长序列会迅速超过 SRAM。分块实现针对一个 Q tile 依次读取多个 K/V tile，并维护每行状态：

- `m`：截至当前 K tile 的行最大值；
- `l`：按最大值重标定后的指数和；
- `O`：按相同重标定系数维护的未归一化输出。

读完所有 K/V tile 后以 `O / l` 得到最终输出。这个流程与 FlashAttention 的 IO-aware 思路相同：中间 score 不写回 HBM/DRAM。

## Prefill 与 Decode 的区别

| 维度 | Prefill | Decode |
| --- | --- | --- |
| Q 长度 | 大 | 通常为 1 或很小 |
| K/V 来源 | 当前输入 tile | 当前 token 加历史 [[npu/operator/transformer common/KV Cache|KV Cache]] |
| 资源重点 | 两次 GEMM 和 Softmax 吞吐 | KV 读带宽、请求并发和小矩阵效率 |
| 常见优化 | Q/K/V 分块、融合、head 并行 | Paged KV、GQA/MQA、continuous batching |

## Mask 与数值语义

- **causal mask**：位置 `j > i` 的 score 不得参与 softmax；实现可在 exp 前设为负无穷或明确置零。
- **padding mask**：屏蔽 batch 内无效 token，必须与 row max / sum 的归约范围一致。
- **精度**：score 与 softmax 累加一般使用高于 Q/K 输入的精度；[[npu/operator/others/Exp|Exp]] 的近似误差应在端到端精度预算内。

## KV Cache 数据布局

K/V 可按 layer、KV head、sequence page、head dimension 分块。布局要让当前 Q tile 的 K/V 读取连续，避免每个 token 产生零散 DRAM 请求。详细调度与带宽计算见 [[存储层级、调度与性能分析]]。
