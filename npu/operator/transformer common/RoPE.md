# RoPE

RoPE = Rotary Positional Embedding，旋转位置编码。它将 query/key 的相邻维度对按位置相关角度旋转。

## 功能

对二维分量 `(x₀, x₁)`，按位置角度 `θ` 变换为 `(x₀cosθ - x₁sinθ, x₀sinθ + x₁cosθ)`。旋转后的 Q/K 点积可表达相对位置信息。

## 在 NPU 里面哪里会出现 RoPE

1. Decoder-only LLM 的 Q 和 K 投影之后、[[Attention]] score 计算之前。
2. 预填充和增量解码路径；K 通常以已旋转表示写入 [[KV Cache]]。

## 实现要点

- Sin/Cos 系数可由 [[LUT]]、预计算表或运行时计算提供。
- 旋转主要是成对乘加，可映射到 vector [[Elementwise]] FMA；布局和 position 索引需与 batch、head、sequence 维度匹配。
