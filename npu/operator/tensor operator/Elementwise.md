# Elementwise

Elementwise（逐元素）算子对同位置元素独立执行加、减、乘、除、最大/最小、选择或逻辑运算。

## 功能

它支持广播（broadcast）和掩码（mask）语义，是张量图中最常见的向量计算类型。

## 在 NPU 里面哪里会出现 Elementwise

1. 残差连接的 `x + residual`、偏置相加和门控乘法。
2. [[RMSNorm]]、[[LayerNorm]] 的缩放、平移和 epsilon 相加。
3. [[RoPE]] 的成对旋转，以及 [[Quantization]] 的 scale / zero-point 操作。

## 实现要点

- 通常映射到 SIMD/vector 单元，并与相邻算子融合以减少中间张量读写。
- 广播维度、饱和规则和舍入模式要与框架语义保持一致。
