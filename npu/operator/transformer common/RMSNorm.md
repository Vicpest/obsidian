# RMSNorm

RMSNorm（Root Mean Square Normalization）按特征向量的均方根进行缩放：`y = x / √(mean(x²) + ε) · γ`。

## 功能

与 [[LayerNorm]] 相比，RMSNorm 不减均值，通常减少一组归约与逐元素操作，因此是许多 LLM 的常用归一化层。

## 在 NPU 里面哪里会出现 RMSNorm

1. LLaMA 等 Transformer 的 attention 和 MLP 之前/之后的归一化。
2. 解码阶段每层的向量处理路径。

## 实现要点

- 先用 [[Reduction]] 计算平方和，再用 [[RSQRT]] 和 [[Elementwise]] 完成缩放。
- 适合与残差相加、量化/反量化融合，减少对外部内存的往返。

## 参考

[[参考资料#ONNX 算子规范]]。
