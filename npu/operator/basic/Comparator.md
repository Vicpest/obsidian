# Comparator

Comparator（比较器）判断两个数的大小或相等关系，并产生选择、掩码或饱和控制。

## 功能

输出 `a > b`、`a = b` 或 `a < b`。向量处理器常将比较结果形成 predication mask，再由 [[Elementwise]] select 完成逐元素选择。

## 在 NPU 里面哪里会出现 Comparator

1. [[Softmax]] 的按行最大值归约，先减去最大值以提高数值稳定性。
2. 激活函数、ReLU、量化裁剪的阈值判断。
3. Top-k、采样和注意力 mask 处理。

## 实现要点

- 浮点比较需正确处理 NaN、正负零和非规格化数；低精度路径应保持与软件语义一致。
