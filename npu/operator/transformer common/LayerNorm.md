# LayerNorm

LayerNorm 对指定特征维度做均值-方差归一化：`y = (x - mean(x)) / √(var(x) + ε) · γ + β`。

## 功能

它保持每个样本独立，适合序列模型。相较 [[RMSNorm]]，额外需要均值计算和中心化步骤。

## 在 NPU 里面哪里会出现 LayerNorm

1. BERT、ViT 和许多 Transformer block 的 Pre-Norm / Post-Norm。
2. 视觉、语音和多模态模型的特征归一化。

## 实现要点

- [[Reduction]] 产生均值与方差，[[RSQRT]] 产生逆标准差，最后由 [[Elementwise]] 做缩放和平移。
- 使用 `E[x²] - E[x]²` 计算方差时要考虑消去误差；累加精度往往应高于输入。

## 参考

[[参考资料#ONNX 算子规范]]。
