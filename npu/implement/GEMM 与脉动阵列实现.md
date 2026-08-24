# GEMM 与脉动阵列实现

[[npu/operator/tensor operator/GEMM|GEMM]] 是 Transformer 中占算力大头的算子。其基本形式为 `C[M,N] = A[M,K] × B[K,N]`，其中每个输出元素是 K 项 [[npu/operator/basic/MAC|MAC]] 的归约。

## Tile 映射

设计算阵列为 `P_M × P_N` 个 [[npu/operator/others/PE|PE]]。编译器把矩阵切分为 `M_t × K_t`、`K_t × N_t` 和 `M_t × N_t` tile：

```text
for m_tile, n_tile:
    C_tile = 0
    for k_tile:
        load A[m_tile, k_tile] and B[k_tile, n_tile] to SRAM
        C_tile += A_tile × B_tile       # PE 阵列执行
    post-process C_tile and store
```

`K_t` 的循环次数决定同一个输出 tile 的累加长度。若 C_tile 放得下 SRAM，应在完成所有 K tile 前避免写回片外内存。

## 常见数据流

| 数据流 | 保持不动的数据 | 适用特点 |
| --- | --- | --- |
| Weight-stationary | 权重 B | 多个激活 tile 复用同一权重，适合推理线性层 |
| Output-stationary | 部分和 C | 减少 partial sum 移动，适合长 K 累加 |
| Row-stationary / systolic | 操作数随阵列流动 | 规则布线、局部通信，适合二维 PE 阵列 |

脉动阵列中 A、B 从阵列边界输入，每周期向相邻 PE 转发，PE 在操作数相遇时完成 MAC。阵列需要填充与排空周期；小矩阵或不规则形状会降低利用率。

## PE 内的运算路径

```text
A / B operand → multiply → accumulator → optional bias / scale → output
                    │              │
                    └─ integer multiplier may use [[npu/operator/basic/CSA|CSA]]
                                   and a final carry-propagate adder
```

对于低比特整数，乘积累加后常执行 bias、激活和 [[npu/operator/tensor operator/Quantization|Quantization]]。对于 BF16/FP8，还需处理舍入、溢出和规格化。

## Transformer 中的形状特征

- **QKV / 输出投影**：大而规则的 GEMM，适合高阵列利用率；QKV 合并可提升激活复用。
- **MLP 上投影/下投影**：常是最重的计算部分，输出通道宽，适合 weight-stationary。
- **Decode 小 GEMM**：M 近似为 batch size，常小于阵列边长；需使用多请求 continuous batching 或将多个 head/group 合并。

## 实现检查点

1. tile 是否能在 SRAM 中同时容纳 A、B、C 及双缓冲？
2. accumulator 位宽是否覆盖 `K × max(a×b)`？
3. K 尾块、M/N 尾块采用 mask 还是 padding，是否降低阵列利用率？
4. 后处理能否和写回融合，避免额外的 [[npu/operator/tensor operator/Layout Transform|Layout Transform]]？

Attention 中两次 GEMM 的特殊访存问题见 [[Attention 硬件实现]]。
