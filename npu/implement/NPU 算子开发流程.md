# NPU 算子开发流程

NPU 算子开发不是把一个数学公式直接翻译成 RTL。一个可交付的算子需要同时闭环：**语义正确、数值误差受控、张量布局可搬运、SRAM 容纳并持续供数、硬件/指令接口可调度、性能与功耗达到目标、异常路径可验证**。这里的“算子”既可以是 [[npu/operator/tensor operator/GEMM|GEMM]]、[[npu/operator/transformer common/RMSNorm|RMSNorm]] 等图算子，也可以是融合后的 kernel，如 `RMSNorm + QKV` 或 online-softmax Attention。

```text
模型语义 / ONNX / 框架图
          │
          ▼
  1. 算子契约与参考实现
          │
          ▼
  2. 融合边界、tile 与数据布局
          │
          ▼
  3. 数据流、SRAM 与 DMA/NoC 规划
          │
          ▼
  4. 微架构、指令描述符与调度
          │
          ▼
  5. 功能/数值/周期验证
          │
          ▼
  6. 性能、面积、功耗分析与迭代
```

## 1. 定义算子契约

先固定外部可观察语义；不要先从 PE 数量或某个现有 SRAM 宏倒推算法。算子规格至少应包含：

| 项目 | 需要明确的内容 |
| --- | --- |
| 输入/输出 | tensor 名称、shape、动态维度范围、stride/layout、对齐要求、是否允许原地覆盖 |
| 数值语义 | 数学公式、广播/reshape/reduction 轴、mask、边界值、NaN/Inf/denorm 策略 |
| 数据类型 | 输入/权重/累加/输出精度，量化 scale/zero-point 的位置与舍入、饱和规则 |
| 正确性准则 | 位精确或误差界，比较指标（max error、relative error、cosine similarity 等）和容忍度 |
| 状态与副作用 | KV Cache append、随机数状态、页表、in-place buffer、错误/poison 传播 |
| 性能目标 | Prefill token/s、Decode TPOT、单次延迟、吞吐、可接受 workspace 和带宽预算 |

为该契约编写可执行的 CPU/高精度 reference model。它应覆盖随机数据与结构化边界：零值、极大/极小值、非 tile 整数倍长度、mask 边界、不同 batch/head 和别名输入。reference model 是后续编译器仿真、RTL testbench 与硬件结果比较的唯一语义基线。

## 2. 确定融合边界与可实现性

以减少中间张量的读写为目标划分 kernel，而不是机械地“一图算子对应一个硬件任务”。例如 `bias + activation` 通常可融合到 GEMM epilogue；Attention 应通过 online softmax 避免物化完整 score 矩阵。融合同时受以下边界约束：

- 中间结果是否能在寄存器或 SRAM 中驻留；不能驻留时，融合可能造成长依赖和资源饥饿。
- producer 与 consumer 的 layout、精度和 tile 是否兼容；为融合新增 transpose/pack 有时得不偿失。
- 两端是否争用同一矩阵引擎、向量引擎、SFU、SRAM bank 或 DMA 通道。
- 编译器是否能表达该融合并提供 fallback；动态 shape、稀疏、mask 和不支持的数据类型通常需要拆分路径。

输出一份 kernel 划分表，列出每个 kernel 的输入、输出、融合算子、使用引擎、临时 workspace 和回退条件。

## 3. 建立 tile、布局与数据流

将逻辑张量映射为可持续执行的 tile。以 `C[M,N] = A[M,K] × B[K,N]` 为例，选择 `M_t、N_t、K_t`，并定义 A/B/C 在内存中的 layout、pack 方式、尾块 padding/mask 和地址公式。矩阵阵列的映射见 [[GEMM 与脉动阵列实现]]；Attention 的分块和 online 状态见 [[Attention 硬件实现]]。

每个 tile 至少回答：

1. 哪些数据在本 kernel 内复用，复用次数是多少？
2. Weight、IFMAP、PSUM、OFMAP 分别驻留在哪一级 SRAM？功能含义见 [[NPU 常用 SRAM 功能类型]]。
3. 哪些数据随 K/reduction 循环流动，哪些留在 PE/register/PSUM SRAM？
4. tile 边界如何处理：padding、mask、剩余通道、因果 mask 或 page 边界？
5. 生产者何时释放 buffer，消费者何时可以读取？

常见数据流包括 weight-stationary、output-stationary 和 row/input-stationary。选择的依据是实际复用、SRAM 带宽和累加位宽，不是名称本身。尤其应避免高位宽 PSUM 在每次 K 累加后往返片外内存。

## 4. 进行 SRAM、DMA 与并发资源预算

在写 RTL 前计算容量和每周期访问量。对双缓冲 GEMM 的一阶容量估计可写为：

```text
S_required ≈ n_A × M_t × K_t × b_A
           + n_B × K_t × N_t × b_B
           + n_C × M_t × N_t × b_PSUM
           + S_metadata
```

其中 `n_A/n_B/n_C` 是是否双缓冲以及是否分离读写 buffer 带来的副本数，`b_*` 为每元素字节数；PSUM 通常使用更宽的 `b_PSUM`。实际还需加入 bank 对齐、ECC/check bits、descriptor、FIFO 和预留容量。

同时列出每一个周期的读写请求：

| 资源 | 典型问题 | 处理方式 |
| --- | --- | --- |
| Weight SRAM | 是否能向所有 PE/lane 提供所需权重？ | broadcast、复制只读 bank、pack 或预取 |
| IFMAP SRAM | 滑窗/stride 或矩阵行列读取是否造成 bank conflict？ | 交错 banking、line buffer、layout transform |
| PSUM SRAM | read-modify-write 是否阻塞 MAC？ | PE 内累加、分 bank、merge buffer 或更高端口数 |
| OFMAP SRAM | 上游写与下游读是否并发？ | ping-pong、1R1W、事件/fence |
| DMA / NoC | load/store 能否被计算隐藏？ | 双缓冲、burst 合并、QoS、分阶段预取 |

若计算所需的 operand bit/cycle 超过 SRAM/NoC 可提供的有效带宽，阵列峰值吞吐不可达；应减小并行度、调整 tile/layout、增加 bank，或改变数据流。存储层级和 Roofline 分析见 [[存储层级、调度与性能分析]]。

## 5. 定义微架构与软件接口

将 tile 的动作拆成可流水的阶段，例如 `DMA load → layout/pack → compute → reduce/epilogue → store`，并明确 ready/valid、buffer ownership 和错误处理。硬件接口不应只传一个“开始”信号；kernel descriptor 至少应可描述：

- 输入/输出基地址、stride、shape 与 tile 参数；
- 数据类型、量化参数、layout/pack 模式、mask/causal 边界；
- SRAM bank 或 workspace 分配、双缓冲槽位和 DMA 长度；
- 依赖 event、完成 event、异常状态和 profiling counter 地址。

对于固定功能单元，这些字段可映射为寄存器或命令队列；对于可编程 NPU，则映射为 ISA 指令、微码或 kernel launch descriptor。编译器负责选择 kernel variant、分配 tile/bank、插入 DMA 与 event；runtime 负责处理动态 shape、页表、资源不足和 fallback。控制与数据面分离方式见 [[Tile 数据流与解耦控制]]。

## 6. 分层验证

验证应从语义逐层收紧，不能只跑一个端到端模型后看最终 logits。

| 层次 | 检查内容 |
| --- | --- |
| Reference / kernel simulator | shape、layout、mask、量化、融合公式与误差界；随机和边界输入对照 reference |
| 编译器/指令仿真 | descriptor 编码、地址生成、tile 边界、DMA 长度、event 顺序、fallback 选择 |
| RTL 单元级 | 算术单元、pack/unpack、SFU 近似、bank 仲裁、FIFO、异常状态机 |
| RTL kernel 级 | tile 结果、连续 tile 流水、backpressure、读写同址、尾块、性能计数器 |
| 系统级 | 多核/多请求争用、DRAM 延迟、KV page 边界、reset/timeout、错误恢复与重放 |

数值验证应分别比较未量化理想结果、软件量化结果与 RTL/硬件结果，才能定位误差是来自量化规格、近似函数还是实现 bug。若有 ECC/ABFT，需做单错、不可纠正错误、poison 和 replay 注入，相关机制见 [[ABFT：检2纠1的逐周期实现]]。

## 7. 评估性能并迭代

至少同时报告功能正确、clean-path 性能和资源代价：

1. **延迟与吞吐**：首 tile latency、initiation interval、单核/全芯片 TOPS、Prefill token/s、Decode TPOT、P50/P99。
2. **利用率**：PE、vector/SFU、SRAM bank、DMA/NoC 利用率，以及 stall 原因（数据未到、bank conflict、尾块、依赖）。
3. **存储流量**：DRAM/HBM、共享 SRAM、局部 SRAM 的 read/write byte 数；确认融合确实减少了中间结果落地。
4. **实现代价**：面积、关键路径、最高频率、动态/静态功耗；不要只用 MAC 数衡量代价。
5. **鲁棒性**：动态 shape、长序列、小 batch、稀疏/非稀疏回退、异常恢复下的正确性和尾延迟。

若阵列利用率低，先用 trace 判断是 tile 太小、带宽不足、bank conflict 还是依赖串行，再修改相应层次。盲目加 PE 或增加 SRAM 容量通常不能消除 layout、端口或 DMA 调度瓶颈。

## 8. 算子交付清单

- 算子契约、reference model 和数值容忍度。
- kernel 划分、融合/回退条件、支持矩阵（shape、dtype、layout）。
- tile/layout/地址生成文档，以及 Weight、IFMAP、PSUM、OFMAP 的容量与带宽预算。
- descriptor/寄存器或 ISA 定义、事件与错误语义。
- 编译器 lowering、调度与 runtime workspace 管理方案。
- 单元、kernel、系统级测试及覆盖的边界/异常用例。
- 性能、面积、功耗、带宽和利用率报告，附基线与测试配置。

完成这些交付物后，算子才能从“公式或 RTL 模块”变成可由编译器稳定调用、可在真实模型中定位性能问题的 NPU kernel。
