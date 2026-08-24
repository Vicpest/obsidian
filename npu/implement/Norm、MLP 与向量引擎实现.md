# Norm、MLP 与向量引擎实现

Transformer 中不适合完全交给矩阵阵列的算子，通常由 SIMD/vector engine、归约树和特殊函数单元（SFU）处理。这些算子虽然 FLOPs 较少，却常因额外读写而影响端到端延迟。

## RMSNorm / LayerNorm

[[npu/operator/transformer common/RMSNorm|RMSNorm]] 的硬件流程：

```text
x tile → square → sum reduction → divide by hidden size → + ε
      → [[npu/operator/others/RSQRT|RSQRT]] → multiply x → multiply γ → y
```

[[npu/operator/transformer common/LayerNorm|LayerNorm]] 还需要计算均值并执行 `x - mean`。常将求和与平方和放在同一次读取中，以得到均值和方差所需统计量。

实现要点：

- 归约树的累加精度应高于输入，避免长 hidden dimension 带来明显误差。
- [[npu/operator/others/RSQRT|RSQRT]] 常用 [[npu/operator/basic/LUT|LUT]] 初值加迭代；[[npu/operator/basic/LZD|LZD]] / 移位器用于规格化。
- 将 residual add、Norm 与后续 QKV/MLP 的输入布局协同处理，降低写回次数。

## MLP 与门控激活

标准 MLP 为 `W2 · activation(W1x + b1) + b2`。SwiGLU 形式常见于 LLM：

```text
u, g = split(W_up x)
y = W_down · (u × SiLU(g))
```

其中两次投影是 [[npu/operator/tensor operator/GEMM|GEMM]]，`split`、SiLU、逐元素乘法和 bias 是向量操作。高效实现会将上投影结果在 SRAM 中按 tile 处理，立刻执行 [[npu/operator/tensor operator/Activation|Activation]] 与门控，再供下投影消费。

## 向量引擎的典型能力

| 能力 | 实现任务 |
| --- | --- |
| SIMD FMA / ALU | bias、residual、scale、RoPE、门控 |
| 比较与选择 | ReLU、mask、量化饱和 |
| 归约树 | mean、variance、softmax max/sum |
| SFU / LUT | exp、rsqrt、reciprocal、GELU/SiLU 近似 |
| 类型转换 | BF16/FP16/INT8/INT4 的转换与舍入 |

## 融合策略

- `residual + RMSNorm`：读入 x 与 residual，直接输出已规格化 tile。
- `up GEMM + bias + SiLU + gate`：上投影输出不落地 DRAM。
- `down GEMM + residual + quantize`：将最后的 vector 后处理置于矩阵引擎 writeback 路径。

关键原则是：优先减少中间张量的片外访问，其次才是减少几条 ALU 指令。
