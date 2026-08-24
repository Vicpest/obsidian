# Reciprocal

Reciprocal（倒数）计算或近似 `1/x`。

## 功能

常以 [[LUT]] 给出初值，再执行 Newton-Raphson 等迭代：`yₙ₊₁ = yₙ(2 - xyₙ)`。迭代主体主要由乘法和 [[Elementwise]] 减法组成。

## 在 NPU 里面哪里会出现 Reciprocal

1. [[Softmax]] 用 `1 / Σ exp(x)` 完成概率归一化。
2. [[LayerNorm]] 的除以标准差，以及通用除法转换。
3. 动态量化和缩放因子计算。

## 实现要点

- 对零、负数、无穷和 NaN 的处理应与输入数值格式定义一致。
- 初值精度与迭代次数共同决定延迟、面积和最终误差。
