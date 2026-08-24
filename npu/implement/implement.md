# Transformer 工作流程与 NPU 硬件实现

本目录说明 Transformer 推理如何从计算图映射为 NPU 上的矩阵、向量、归约和存储访问。算子定义与底层数据通路单元见 [[npu/operator/operator|NPU 算子分类]]。

## 阅读路径

| 页面 | 关注的问题 | 对应内容 |
| --- | --- | --- |
| [[Transformer 推理工作流程]] | 一个 token / 一个 batch 依次经过什么计算 | Prefill、Decode、单层数据流 |
| [[NPU 总体硬件架构]] | 哪些硬件模块承载这些计算 | 控制器、计算引擎、存储层级、DMA |
| [[GEMM 与脉动阵列实现]] | 线性层如何高效执行 | Tile、PE 阵列、数据流、累加与量化 |
| [[Attention 硬件实现]] | QKV、Softmax、KV Cache 如何协作 | 分块、online softmax、Prefill / Decode |
| [[Norm、MLP 与向量引擎实现]] | 非 GEMM 算子如何映射 | RMSNorm、激活、SwiGLU、融合 |
| [[存储层级、调度与性能分析]] | 为什么会带宽受限、如何调度 | SRAM、DRAM、DMA、Roofline、流水 |

## 端到端映射


Token / hidden state
        │
        ▼
[[npu/operator/transformer common/RMSNorm|RMSNorm]] / [[npu/operator/transformer common/LayerNorm|LayerNorm]] ──► 向量引擎 + 归约树
        │
        ▼
QKV [[npu/operator/tensor operator/GEMM|GEMM]] ────────────────► PE 阵列 + 本地 SRAM
        │
        ▼
[[npu/operator/transformer common/RoPE|RoPE]] + [[npu/operator/transformer common/Attention|Attention]] ────► 向量引擎 + GEMM 阵列 + [[npu/operator/transformer common/KV Cache|KV Cache]]
        │
        ▼
输出投影 [[npu/operator/tensor operator/GEMM|GEMM]] + 残差 ────► PE 阵列 + 向量引擎
        │
        ▼
Norm → MLP [[npu/operator/tensor operator/GEMM|GEMM]] → [[npu/operator/tensor operator/Activation|Activation]] → GEMM → 残差
        │
        ▼
下一 Transformer 层 / logits

## 设计边界

- 本文以推理为主；训练还需要反向传播、优化器状态和更高的片外带宽。
- "算子" 是计算图语义，"硬件单元" 是实现资源：一个 [[npu/operator/tensor operator/GEMM|GEMM]] 可占用许多 [[npu/operator/others/PE|PE]]，而一个融合 kernel 可覆盖多个算子。
- 具体实现会随精度（INT8 / INT4 / FP8 / BF16）、阵列大小、SRAM 容量和目标时钟而变化。
