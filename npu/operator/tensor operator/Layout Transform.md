# Layout Transform

Layout Transform（布局变换）包括 Transpose、Reshape、Slice、Concat、Split、Gather 和张量分块等不以大量算术为主的操作。

## 功能

它改变张量的视图、维度顺序或内存排布，使数据符合后续计算引擎的 tile、向量宽度和连续访存要求。

## 在 NPU 里面哪里会出现 Layout Transform

1. QKV 拆分/合并，BHSD、BSHD 等 Attention 布局转换。
2. [[KV Cache]] 的追加、分页、读取和 batch 重排。
3. [[Convolution]] 的 NCHW/NHWC 转换及 GEMM 分块装载。

## 实现要点

- Reshape 在连续内存条件下可为零拷贝视图；Transpose 通常需要真实的数据搬移。
- 布局变换可能受内存带宽限制，应与 [[GEMM]]、[[Attention]] 融合或使用 DMA/片上 SRAM 双缓冲隐藏开销。
