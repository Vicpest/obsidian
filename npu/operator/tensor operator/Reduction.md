# Reduction

Reduction（归约）沿指定轴把多个元素合并为一个元素，常见操作为 Sum、Max、Mean 和 Sum of Squares。

## 功能

硬件一般使用树形加法或比较网络归约；长向量可先在各 [[PE]] / vector lane 局部归约，再进行跨 lane 合并。

## 在 NPU 里面哪里会出现 Reduction

1. [[Softmax]] 的最大值和指数和。
2. [[RMSNorm]] 的平方均值，[[LayerNorm]] 的均值与方差。
3. 全局/平均池化、损失统计和量化校准。

## 实现要点

- 浮点归约的加法顺序会影响舍入误差；并行树形实现必须定义可接受的数值容差。
- 累加器位宽和中间精度往往比输入精度更高。
