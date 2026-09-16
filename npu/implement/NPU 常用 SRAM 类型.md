# NPU 常用 SRAM 类型

NPU 中的 SRAM 通常用于权重、激活、partial sum、KV tile、查找表和控制状态。所谓“SRAM 类型”可能指端口组织、物理实现或系统用途；三者不能混为一谈。例如，“activation buffer”描述存放什么数据，“1R1W”描述每周期能做什么访问，“banked SRAM”描述如何用多个宏提高并行度。

## 1. 按端口组织分类

端口能力决定同一 SRAM 宏在一个周期内可接受的访问组合。下表中的能力是逻辑上限，实际还受 SRAM compiler、时钟频率、读延迟和同地址冲突规则约束。

| 类型 | 每周期典型能力 | 优点 | 代价与限制 | NPU 常见用途 |
| --- | --- | --- | --- | --- |
| 单端口 SRAM（1P / 1RW） | 读或写一次 | 面积、功耗和布线代价最低 | 读写互斥，DMA、计算与 scrub 容易争用 | 只读占主导的权重 tile、低带宽暂存、容量优先的 buffer |
| 简单双端口 SRAM（SDP / 1R1W） | 一个读端口加一个写端口 | 可同时生产和消费，适合流式流水 | 通常不能两读或两写；同地址读写语义需确认 | ping-pong buffer、FIFO、激活/结果流、KV 追加与读取 |
| 真双端口 SRAM（TDP / 2RW） | 两个端口各自可读或写 | 调度灵活，可服务两个访问源 | bit-cell、外围、验证和功耗代价更高；同地址冲突复杂 | 共享 scratchpad、DMA 与计算并发、前台访问与后台 scrub |
| 多端口 SRAM / Register File | 多读、多写，如 2R1W、4R2W | 单周期向多个 lane/PE 广播或收集数据 | 端口数增加会快速放大面积、布线和动态功耗，容量通常较小 | 向量寄存器、PE operand file、累加器、调度表和小型 metadata |
| 伪多端口 SRAM | 用复制、banking、多泵或仲裁模拟多端口 | 可用普通宏获得更高逻辑并发 | 复制消耗容量；多泵提高频率/功耗；banking 可能冲突 | 大容量高带宽 scratchpad、多个读消费者、低成本多端口接口 |

### 1.1 1P：容量与能效优先

1P SRAM 只有一个读写端口，一个周期只能完成一次读或一次写。它适合访问方向稳定、可由调度器分时复用的存储。例如权重先由 DMA 写入，计算阶段再连续读取。若 DMA 写入下一 tile 与阵列读取当前 tile 必须重叠，通常使用两个 1P bank 做 ping-pong，而不是直接升级为更昂贵的双端口宏。

### 1.2 1R1W：流式生产者—消费者

1R1W SRAM 有独立读、写端口，适合一侧持续填充、另一侧持续消费。典型例子是 DMA 写 activation tile、向量或矩阵引擎同时读取另一地址，以及 Decode 中追加新 K/V 的同时读取历史 K/V。

1R1W 不等于任意双端口：它不能自然支持两个读请求或两个写请求。同一周期读写同一地址时，输出可能是旧值、新值或未定义，必须依据宏规格在 RTL 中加入 bypass、stall 或禁止条件。

### 1.3 2RW：灵活但昂贵

2RW 的两个端口都能独立读写，适合访问方向动态变化的共享存储。它可以让计算引擎和 DMA、两个计算引擎，或前台请求和 ECC scrub 并行工作。但两个端口访问同一地址、尤其同时写入时，需要明确优先级和冲突行为。

如果工作负载大部分时间只需要“一读一写”，2RW 的额外灵活性未必能转化为性能，应先比较 1R1W、双缓冲 1P 和多 bank 1P 的面积、频率与有效带宽。

### 1.4 多端口 Register File：小容量、近计算

PE 和 vector lane 常要求同周期读取多个操作数并写回结果，因此更适合多端口 register file、寄存器阵列或 latch-based memory。它们在逻辑功能上也可表现为 SRAM，但容量、端口和物理实现与大容量 compiled SRAM 不同。随着端口数增加，mux、decoder、wordline、bitline 和布线成本迅速上升，因此通常只放最热的数据，而不承担大容量 tile buffer。

## 2. Banked SRAM 不是一种新端口宏

Banking 是用多个独立 SRAM 宏构成一个逻辑地址空间。若有 `B` 个 1P bank 且请求均匀落到不同 bank，理论上每周期最多完成 `B` 次访问；多个请求命中同一 bank 时仍需仲裁或停顿。

```text
logical address
      │
      ├─ bank select = hash/address bits
      └─ row address
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
  1P bank0 1P bank1 ... 1P bank B-1
     └───────┼────────┘
             ▼
       response crossbar
```

因此，`B × 1P` 不能无条件等价为一个 `B` 端口 SRAM：前者受地址映射、bank conflict、crossbar 和负载均衡影响。NPU 常用以下映射降低冲突：

- 按连续 word 交错，使宽向量的相邻元素分散到不同 bank。
- 按 channel、head、tile 行/列映射，使并行 lane 的访问落到不同 bank。
- 使用 XOR/hash 缓解固定 stride 冲突，但会增加地址生成和验证复杂度。
- 为广播型权重复制只读副本，以容量换取多个独立读端口。

多 bank 的带宽、ECC、scrub 与失效降级问题见 [[大容量多 Bank SRAM 容错设计]]。

## 3. 按物理实现分类

端口组织是架构设计最常用的分类；在 ASIC 物理实现中，还会按 bitcell 和 compiler 优化目标区分宏类型：

| 实现 | 特点 | NPU 中的适用场景 |
| --- | --- | --- |
| 6T SRAM | 标准高密度 bitcell，读写共享内部节点；容量效率通常较好 | 大多数常规 weight/activation scratchpad，具体电压和频率能力以 compiler 为准 |
| 8T / 10T 等读解耦 SRAM | 读路径与存储节点进一步隔离，通常有利于低电压稳定性或独立读端口 | 低电压、高可靠或特定多端口需求；面积通常高于 6T |
| High-density 宏 | 优先容量密度与漏电，速度可能较低 | 大容量共享 SRAM、较冷的权重或 KV staging 区 |
| High-speed 宏 | 加强外围电路和驱动以提高频率 | 紧邻 PE、每周期供数的较小 buffer |
| Register file / latch-based memory | 针对浅深度、宽 word 或多端口优化 | operand、accumulator、vector register 和控制状态 |

“6T、8T、10T”只说明 bitcell 拓扑的一部分，不能直接推出用户可见端口数、宏面积或访问周期。同一工艺的可用组合由 memory compiler 决定；NPU RTL 通常先确定容量、宽度、端口与时序，再由物理设计选择可实现的宏。FPGA 中则映射到 block RAM、UltraRAM 或分布式 RAM，无需也不能由 RTL 设计者选择 6T/8T bitcell。

## 4. 按 NPU 功能划分

功能名不能直接决定 SRAM 端口，仍需根据数据流计算每周期读写次数。

| 逻辑存储 | 访问特征 | 常见实现选择 |
| --- | --- | --- |
| Weight buffer | DMA 成块写入，计算阶段连续读或广播，读多写少 | ping-pong 1P、多 bank 1P；多个消费者时复制只读 bank |
| Activation buffer | 生产者写、下游算子读，常需算子间流水 | 1R1W、ping-pong 1P 或 banked 1P |
| Partial-sum buffer | 高频读—修改—写，位宽通常高于输入 | 1R1W/2RW、banked accumulator SRAM；近 PE 时使用 register file |
| KV buffer / KV tile cache | 每步追加少量新 K/V，Attention 读取大量历史数据 | 1R1W 或多 bank 1P；容量不足时作为片外 KV Cache 的 tile staging buffer |
| Line buffer / sliding-window buffer | 固定速率写入与窗口式多点读取 | 1R1W、多个 bank 或 shift-register/latch 结构 |
| LUT / coefficient memory | 初始化后只读、容量小 | ROM、1P SRAM 或寄存器；是否使用 SRAM 取决于是否需要在线更新 |
| Metadata / queue / page table | 容量小、随机访问、可能多读写 | 小型多端口 RF、复制 RAM 或 1R1W SRAM，并加强 ECC/复制保护 |

## 5. 选型时必须确认的宏属性

仅写“使用双端口 SRAM”不足以完成接口设计。至少需要从目标 PDK 的 memory compiler 或 FPGA memory primitive 中确认：

1. **端口组合**：1RW、1R1W、2RW，是否允许同周期访问同一地址。
2. **读语义**：同步读还是异步读，read latency 是多少，输出是否带寄存器。
3. **read-during-write 行为**：返回旧数据、返回新数据、保持输出或未定义。
4. **写粒度**：整 word、byte enable、bit mask；部分写是否触发 ECC read-modify-write。
5. **吞吐与时序**：端口 initiation interval、最高频率，以及 bank/crossbar 是否进入关键路径。
6. **低功耗能力**：clock gating、light/deep sleep、retention 和唤醒周期。
7. **测试与修复**：MBIST、spare row/column、BISR，以及测试模式对端口的占用。
8. **可靠性支持**：parity/ECC 位是否与数据同宏读出，是否支持 scrub 和错误注入。

同步 compiled SRAM 常见为“地址在周期 `t` 寄存、数据在 `t+1` 或更晚返回”，不能按组合数组建模后再假定综合工具会自动得到相同宏。FPGA 上的 block RAM / UltraRAM 等资源也有各自的端口、read-first/write-first 和输出寄存器限制，应以器件手册为准。

## 6. NPU 选型方法

先从 kernel 的周期级访问需求出发，而不是先选宏名称：

```text
每周期需要的读/写次数
        │
        ├─ 能否按地址分散？ ──► banked 1P / 1R1W
        ├─ 能否按阶段分离？ ──► ping-pong 1P
        ├─ 是否只需多个读？ ──► 复制只读副本
        └─ 是否必须随机并发？ ─► 2RW / multi-port RF
```

然后同时检查容量、端口冲突、load-to-use latency、宏面积、动态功耗和布线。对矩阵阵列，峰值 SRAM 供数带宽至少要匹配每周期消费的 operand bit 数；对小 batch Decode，还要优先检查单次读取延迟和 KV 访问冲突。最终选择应以目标 SRAM compiler 报告、综合布局布线和真实访问 trace 为依据。
