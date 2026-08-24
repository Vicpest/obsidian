# Transformer 推理工作流程

Transformer block 可概括为两段残差子层：Attention 和 MLP。对于第 `l` 层，常见的 Pre-Norm 形式为：

```text
h1 = h + Attention(Norm1(h))
h2 = h1 + MLP(Norm2(h1))
```

其中 Attention 使用 [[npu/operator/transformer common/Attention|Attention]]，MLP 通常是两次 [[npu/operator/tensor operator/GEMM|GEMM]] 与 [[npu/operator/tensor operator/Activation|GELU / SiLU / SwiGLU]] 的组合。

## 单层执行顺序

1. **输入读取与 Norm**：DMA 将 hidden-state tile 读入片上 SRAM；向量引擎计算 [[npu/operator/transformer common/RMSNorm|RMSNorm]] 或 [[npu/operator/transformer common/LayerNorm|LayerNorm]]。
2. **QKV 投影**：矩阵引擎执行 `XWq`、`XWk`、`XWv`。实现上可合并为一次输出通道更宽的 GEMM，减少 X 的重复读取。
3. **位置编码与注意力**：对 Q/K 施加 [[npu/operator/transformer common/RoPE|RoPE]]；计算 `QKᵀ`、[[npu/operator/transformer common/Softmax|Softmax]]、`PV`。历史 K/V 从 [[npu/operator/transformer common/KV Cache|KV Cache]] 读取。
4. **输出投影与残差**：Attention 输出经 `Wo` 的 GEMM 后，与原 hidden state 逐元素相加。
5. **MLP**：第二次 Norm 后执行上投影 GEMM；经过激活/门控；再执行下投影 GEMM 并做残差相加。
6. **写回与层间交接**：结果可保留在 SRAM 供下一层读取；容量不足时写回 DRAM/HBM。

## Prefill 与 Decode

| 阶段 | 输入序列长度 | Attention 特征 | 常见瓶颈 | 实现侧重点 |
| --- | --- | --- | --- | --- |
| Prefill | 一次处理提示词的多个 token | Q、K、V 均为矩阵，计算量高 | GEMM 算力 | 大 tile、阵列利用率、分块 attention |
| Decode | 每步新增一个或少量 token | Q 很短，K/V 来自历史缓存 | KV Cache 带宽与小 GEMM 利用率 | KV 分页、连续 batching、低延迟调度 |

Decode 不能重复计算历史 K/V；每层仅将新 token 的 K/V 追加到缓存。这是 [[npu/operator/transformer common/KV Cache|KV Cache]] 的核心价值。

## 常见融合边界

- `bias + residual + RMSNorm`：共用一次输入/输出访存。
- `QKV GEMM + layout transform + RoPE`：减少中间 Q/K/V 的写回。
- `QKᵀ + mask + Softmax + PV`：使用 streaming / online softmax，避免落地完整 score 矩阵。
- `GEMM + bias + activation`：MLP 的常见融合块。

下一步见 [[NPU 总体硬件架构]]，了解这些阶段实际落在哪些资源上。
