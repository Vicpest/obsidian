# PE

PE = Processing Element，处理单元，是计算阵列中重复实例化的基本节点。

## 功能

PE 通常执行一次或多次 [[MAC]]，保存局部累加结果，并将激活、权重或部分和按阵列节拍转发给相邻节点。在脉动阵列中，多个 PE 协同完成 [[GEMM]]。

## 在 NPU 里面哪里会出现 PE

1. QKV、Attention 输出投影和 FFN 的矩阵乘法阵列。
2. 卷积转换为矩阵乘法后的计算阵列。
3. 面向低比特推理的 INT8/INT4 dot-product 阵列。

## 实现要点

- PE 的累加位宽须覆盖 K 维长度和输入精度；输出通常再进入 [[Quantization]] 或 [[Elementwise]] 单元。
- 阵列利用率不仅取决于算力，也受 [[Layout Transform]]、片上存储带宽和数据复用影响。
