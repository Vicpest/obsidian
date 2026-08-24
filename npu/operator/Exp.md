# Exp

Exp（指数）计算或近似 `e^x`，是 [[Softmax]] 的关键非线性。

## 功能

典型定点实现将 `x` 做范围归约为 `x = k·ln(2) + r`，以 `2^k × e^r` 重建结果；`e^r` 可通过 [[LUT]]、多项式或分段近似获得。

## 在 NPU 里面哪里会出现 Exp

1. [[Softmax]] 的指数通路。
2. Sigmoid、GELU 等 [[Activation]] 的近似实现。

## 实现要点

- [[LZD]]、[[Barrel Shifter]] 常用于中间值规格化和移位缩放。
- 必须定义输入截断范围、溢出/下溢处理和近似误差预算。
