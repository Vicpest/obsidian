# Softmax

Softmax 将向量映射为和为 1 的非负分布：`pᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))`。

## 功能

先减去最大值可避免指数溢出；随后计算 [[Exp]]、指数和归约，并用 [[Reciprocal]] 或除法完成归一化。

## 在 NPU 里面哪里会出现 Softmax

1. [[Attention]] 中，将每个 query 对应的 score 行转换为注意力概率。
2. 分类模型的 logits 概率化，以及采样前的 token 分布处理。

## 实现要点

- Max 与 Sum 都是 [[Reduction]]；前者使用 [[Comparator]]，后者需足够宽的累加精度。
- causal/padding mask 需在最大值与指数计算前按框架语义施加。
- 对长序列，online softmax 可在流式读取 K/V 时维护 running max 和 running sum，避免物化完整 score 矩阵。

## 参考

[[参考资料#ONNX 算子规范]]。
