# NPU 算子分类

本目录按 NPU 硬件实现与模型计算图两个层次整理。前者是 [[PE]] 内的数字数据通路单元，后者是编译器调度到矩阵、向量和归约引擎的张量算子。

## 基础数据通路单元

| 算子 | 核心功能 | NPU 典型位置 |
| --- | --- | --- |
| [[CSA]] | 将三个操作数压缩为和、进位两行 | [[MAC]] 乘法器的部分积压缩树 |
| [[LZD]] | 统计前导零，给出规格化移位量 | 浮点/定点缩放、[[Exp]]、[[RSQRT]] |
| [[MAC]] | 计算 `a × b + acc` | [[PE]]、GEMM、卷积、归约 |
| [[Barrel Shifter]] | 单周期可变距离移位 | 对齐、规格化和量化缩放 |
| [[Comparator]] | 比较并生成选择控制 | Max、裁剪、[[Softmax]] 最大值归约 |
| [[LUT]] | 用查表近似非线性函数 | [[Exp]]、激活函数、倒数初值 |

## 通用张量算子

| 算子 | 核心功能 | NPU 典型位置 |
| --- | --- | --- |
| [[GEMM]] | 矩阵乘法/批量矩阵乘法 | 线性层、卷积 lowering、Attention |
| [[Convolution]] | 滑动窗口乘加 | CNN、视觉编码器 |
| [[Elementwise]] | 逐元素算术、比较和选择 | 残差、偏置、门控、量化 |
| [[Reduction]] | 沿一个或多个轴求和、最大值等 | Norm、[[Softmax]]、池化 |
| [[Activation]] | ReLU、GELU、SiLU 等非线性 | MLP、门控网络、视觉网络 |
| [[Quantization]] | 浮点与 INT8/INT4/FP8 表示转换 | 权重量化、激活量化、输出重标定 |
| [[Layout Transform]] | Transpose、Reshape、Concat、Slice | 张量分块、算子衔接和 KV Cache |

## Transformer 常用复合算子

| 算子 | 核心功能 | 主要依赖 |
| --- | --- | --- |
| [[Attention]] | Q、K、V 的加权上下文计算 | [[GEMM]]、[[Softmax]]、[[RoPE]]、[[KV Cache]] |
| [[Softmax]] | 指数归一化为概率分布 | [[Reduction]]、[[Comparator]]、[[Exp]]、[[Reciprocal]] |
| [[RMSNorm]] | 按均方根缩放特征 | [[Reduction]]、[[RSQRT]]、[[Elementwise]] |
| [[LayerNorm]] | 按均值与方差归一化特征 | [[Reduction]]、[[RSQRT]]、[[Elementwise]] |
| [[RoPE]] | 对 Q/K 施加位置旋转编码 | [[Elementwise]]、Sin/Cos 表或 [[LUT]] |
| [[KV Cache]] | 追加、读取历史 Key/Value | [[Layout Transform]]、片上 SRAM / DRAM DMA |

## 典型数据流

- MLP：[[RMSNorm]] → [[GEMM]] → [[Activation]] → [[GEMM]] → [[Elementwise]]（残差）。
- Attention：[[RMSNorm]] → [[GEMM]]（QKV）→ [[RoPE]] → [[Attention]] → [[GEMM]]。
- Attention 内部：`QKᵀ`（[[GEMM]]）→ [[Softmax]] → `P×V`（[[GEMM]]），增量解码时读取 [[KV Cache]]。

页面中的“参考”均指向 [[参考资料]]，其中收录了联网核对过的 ONNX 官方算子规范与硬件实现资料。
