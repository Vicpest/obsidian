# RSQRT

RSQRT = Reciprocal Square Root，倒数平方根，计算或近似 `1 / √x`。
$\frac{1}{\}$

## 功能

可使用 LUT 初值加 Newton-Raphson 迭代，或采用专用平方根/除法单元。相较先求平方根再求倒数，直接 RSQRT 往往更适合归一化。

## 在 NPU 里面哪里会出现 RSQRT

1. [[RMSNorm]] 的 `1 / √mean(x² + ε)`。
2. [[LayerNorm]] 的 `1 / √(variance + ε)`。
3. 向量归一化、距离计算和部分传统视觉算子。

## 实现要点

- 输入须为非负；epsilon 的加入既避免除零，也影响低幅值数据的数值稳定性。
- [[LZD]] 与 [[Barrel Shifter]] 用于将输入调整到适合查表/迭代的规格化范围。
