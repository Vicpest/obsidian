# Convolution

卷积对输入特征图的局部窗口和卷积核做乘加，生成输出特征图。实现可分为`GEMM` 、`Direct`、 `Winograd` 、`FFT`四类卷积操作。
## 功能

二维卷积的核心是空间滑窗上的多通道点积；可直接映射为专用滑窗数据流，也可转化为 [[GEMM]]。

## 在 NPU 里面哪里会出现 Convolution

1. CNN 与视觉分类、检测、分割网络。
2. Vision Transformer 的卷积 stem、下采样或混合骨干网络。
3. 传统多模态前端的视觉/音频特征提取。

## 实现要点

- 重点在输入滑窗复用、权重复用和输出累加；核心计算仍由 [[MAC]] / [[PE]] 完成。
- Padding、stride、dilation 和数据布局变换常由 [[Layout Transform]] 协助处理。
