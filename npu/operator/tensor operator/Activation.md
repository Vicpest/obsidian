# Activation

Activation（激活函数）对张量施加非线性映射，打破纯线性网络的表达限制。

## 功能

常见函数包括 ReLU、LeakyReLU、GELU、SiLU/Swish、Sigmoid 和 Tanh。GELU 与 SiLU 是 Transformer 和现代视觉模型中最常见的选择。

## 在 NPU 里面哪里会出现 Activation

1. FFN / MLP 中的 `GELU(Wx+b)`。
2. GLU、SwiGLU 等门控 MLP 中的 SiLU 或 Sigmoid。
3. CNN、检测和分割网络中的 ReLU 类激活。

## 实现要点

- 可用 [[LUT]]、分段多项式或定点近似实现；范围归约和 [[Quantization]] 决定误差。
- 常与前后的 [[Elementwise]]、偏置相加融合，避免写回中间张量。
