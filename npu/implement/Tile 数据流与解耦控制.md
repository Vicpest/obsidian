# Tile 数据流与解耦控制

面向 Transformer 的 NPU 可把计算阵列组织为多个重复的 tile。每个 tile 将**低带宽控制**和**高带宽数据面**分开：控制核心/任务管理器负责发起和同步，矩阵、向量、归约单元与本地 SRAM 负责持续的数据流执行。这样既方便扩展 tile 数，也使 DMA、NoC 和双缓冲成为显式的可调度资源。

> [!warning] 来源边界：本文的“Redwood”命名、具体 tile 规模、接口宽度、FPGA/ASIC 性能、面积和功耗数值来自 `reading/AINPU.md`。截至 2026-09-02，对公开 arXiv 的 “Redwood Transformer inference accelerator” / “Redwood tile accelerator” 检索未找到可对应的一手公开条目，因此这些专有实现细节**未作为已联网验证事实转写**。下文只保留可独立理解的通用设计模式；如补充论文 DOI、项目主页或代码仓库，应据其更新本页的案例部分。

## 通用 tile 组织

```text
Host / runtime
  └─ global command processor / scheduler
       ├─ global DMA：DRAM/HBM ↔ shared SRAM / tile 边界
       ├─ NoC：tile ↔ tile、DMA ↔ tile
       └─ tile[i]
            ├─ control core：运行短内核/提交命令
            ├─ task manager：依赖、事件、循环、完成通知
            └─ data plane
                 ├─ local DMA + local SRAM / CMEM
                 ├─ matrix engine：GEMM/GEMV
                 ├─ vector engine：RoPE、Norm、activation、layout
                 └─ reduction/SFU：max/sum、exp、rsqrt
```

这与 [[NPU 总体硬件架构]] 的职责划分一致；区别是把共享资源进一步落到 tile 级，以便编译器显式安排数据在哪里驻留、何时被搬运、由哪个 tile 消费。

## 为什么控制面与数据面要解耦

控制路径的命令密度远低于矩阵/向量数据流。将其分开可带来：

- **低开销控制**：控制核心提交 DMA、矩阵、向量和同步任务后可等待事件/中断，不应参与逐 MAC 调度。
- **可替换数据面**：矩阵阵列、SFU 或 DMA 规格调整时，控制 ISA/编程模型不必与每个微结构强耦合。
- **可见的依赖图**：任务 ID、fence/event 和队列表达“load → compute → store”与跨 tile 依赖；这比隐式轮询更适合双缓冲和多请求并行。
- **能耗管理机会**：在长数据面 kernel 运行期，空闲控制逻辑可时钟门控；实际节能幅度必须以工艺、活动因子和门控策略测量。

## 数据流与同步模式

一个可复用的 tile kernel 可按如下顺序构造：

```text
1. DMA prefetch 下一 A/B/KV tile 到 ping buffer
2. 等待 buffer 就绪；矩阵/向量/SFU 消费当前 buffer
3. 将结果留在 local SRAM，或经 NoC 发送给下游 tile
4. 结果写回/消费完成事件 → 允许该 buffer 被下一次 DMA 覆盖
```

对于 Attention，K/V page 可由 DMA 流式搬入本地 SRAM，矩阵引擎计算 score，向量/归约单元更新 online-softmax 状态，再读取 V tile 累积输出；完整 score 不应落到外存。流程见 [[实现/Attention 硬件实现]]。对于 MLP/QKV，权重和激活 tile 的复用、MAC 阵列数据流见 [[实现/GEMM 与脉动阵列实现]]。

跨 tile 的消息/事件应只表达数据依赖和资源释放，例如“page 已到达”“partial sum 已写完”“下游 buffer 可接收”。它们不能替代容量规划：若多个 tile 争用同一 NoC 链路、SRAM bank 或 DRAM 通道，仍需由 runtime 规避拥塞。

## Transformer 映射建议

| 工作负载 | tile 内主要资源 | tile 间 / 全局工作 | 关键依赖 |
| --- | --- | --- | --- |
| Prefill QKV/MLP | matrix engine + local SRAM | 权重预取、按 head/输出通道分片 | A/B tile 就绪、partial sum 完成 |
| Prefill attention | matrix + vector/reduction | K/V tile 广播或分片 | online-softmax 状态顺序合并 |
| Decode attention | vector + matrix，历史 K/V 流 | page table、连续 batching、KV DMA | 新 K/V append 先于读取 |
| MoE | matrix engine | router、token regroup、All-to-All | expert capacity 与通信完成 |

对于 Decode，不应把“每个 token 一个 tile”误解为最佳方案：单请求的小 GEMM 很难填满阵列。更常见做法是将多个 request 的同层 Decode 聚合，同时保留各自的 [[npu/operator/transformer common/KV Cache|KV Cache]] 页表、长度和 causal 边界。

## 设计检查清单

1. tile SRAM 是否能同时容纳当前/下一 ping-pong tile、累加器和必要的 vector 临时量？
2. 每条 DMA/NoC 传输是否有生产与消费事件，且不存在覆盖仍在读的 buffer？
3. command/task 队列是否足以描述循环、条件依赖与跨 tile fan-out，而不需要控制核心忙等？
4. Roofline 中的可用带宽是否按最窄链路和压缩/元数据后的真实字节计算？
5. 小 shape、尾 tile、异构 request 是否有分桶/回退路径？

性能定量建模与双缓冲原则见 [[存储层级、调度与性能分析]]。
