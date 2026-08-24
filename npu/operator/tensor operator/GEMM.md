# GEMM

GEMM = General Matrix Multiplication，通用矩阵乘法，常表示为 `C = αAB + βC`；批量场景也可使用 batched MatMul。

## 功能

它将大量 [[MAC]] 规则地组织为矩阵乘法，并通过分块（tiling）复用片上缓存中的 A、B 和 C 子块。

## 在 NPU 里面哪里会出现 GEMM

1. Transformer 的 Q、K、V、输出投影及两层 FFN 线性变换。
2. [[Attention]] 的 score 计算 `QKᵀ` 和 value 聚合 `P×V`。
3. 将 [[Convolution]] 变换为 im2col 或等价矩阵形式后的主计算。

## 实现要点

- M、N、K 分块需匹配 [[PE]] 阵列形状和 SRAM 容量。
- 对量化模型，输入、权重和累加器的精度以及 [[Quantization]] 重标定位置决定精度与带宽。

## 参考

[[参考资料#ONNX 算子规范]]。
