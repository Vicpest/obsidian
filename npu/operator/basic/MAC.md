# MAC

MAC = Multiply-Accumulate，乘加器。

## 功能

计算 `acc_next = acc + a × b`。点积、[[GEMM]]、[[Convolution]] 和许多统计量归约都可分解为 MAC 序列。

## 在 NPU 里面哪里会出现 MAC

1. [[PE]] 的主计算流水线。
2. Transformer 的线性层、`QKᵀ` 和 `P×V`。
3. [[RMSNorm]]、[[LayerNorm]] 的平方和、均值和方差统计。

## 实现要点

- 整数 MAC 要为乘积和 K 次累加预留足够的 accumulator 位宽。
- 浮点 MAC 需要指数对齐、尾数运算与 [[LZD]] 驱动的规格化；低比特路径常接 [[Quantization]]。
