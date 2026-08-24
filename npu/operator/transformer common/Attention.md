# Attention

Attention 是 Transformer 的上下文聚合算子。缩放点积注意力可写为 `Softmax(QKᵀ / √d + mask)V`。

## 功能

它先将输入投影为 Q、K、V，再以 Q 与 K 的相似度得到权重，最后对 V 加权求和。多头注意力把通道划为多个 head 并行计算。

## 在 NPU 里面哪里会出现 Attention

1. Encoder、decoder 与交叉注意力层。
2. LLM 预填充（prefill）和增量解码（decode）的主导算子；两者分别更偏计算受限与带宽受限。

## 实现要点

- 核心由两次 [[GEMM]] 和一次 [[Softmax]] 构成，Q/K 常先执行 [[RoPE]]。
- 以分块方式融合 `QKᵀ`、softmax 和 `PV` 可避免物化巨大的 score 矩阵；这是 FlashAttention 类实现的关键思想。
- 解码时通过 [[KV Cache]] 读取历史 K/V；mask、head 映射和精度策略需要明确。

## 参考

[[参考资料#ONNX 算子规范]]；[[参考资料#Attention 实现资料]]。
