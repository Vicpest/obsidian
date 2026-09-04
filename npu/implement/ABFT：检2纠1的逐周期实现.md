# ABFT：检 2 纠 1 的逐周期实现

ABFT 把 checksum 冗余嵌入矩阵算法。本文将“检 2 纠 1”限定为 SEC-DED：最多定位并纠正 1 个受保护错误，最多检测 2 个错误但不纠正两个错误。要对任意故障模式成立，等价线性码最小距离至少为 4；普通单错纠正码不足以保证双错不被误纠正。

## 1. 校验结构

对输出 tile C[M,N] 维护行 parity、列 parity 和 tile-wide overall parity。行/列 syndrome 定位单错 (r,c)，overall parity 区分奇数/偶数错误。对数值 word 错误，建议 XOR parity 与算术 checksum 组合。

加权 checksum：

    p0 = Σ x[i]
    p1 = Σ (i+1)x[i]
    p2 = Σ (i+1)^2 x[i]

若位置 k 有加性错误 e，则 s0=e、s1=(k+1)e、s2=(k+1)^2e。s1/s0 定位、s0 修正，s2·s0=s1^2 验证单错一致性；两个错误通常破坏该关系而报告 double-error。仅用 p0、p1 不能宣称任意故障下 SEC-DED，必须说明码距、模数、溢出和故障模型。

矩阵 ABFT 可用 A_aug=[A;1ᵀA;wᵀA]、B_aug=[B,B1,Bw] 预测 C 的行/列 checksum；工程实现通常在 DMA/归约旁路生成。

## 2. 脉动阵列逐周期

设 P×Q 阵列、K_t 归约长度，每 PE 每周期一次 MAC，阵列填充/排空 T_fill≈P+Q−2，checksum reducer 延迟 T_red：

|               周期 | 数据面                           | ABFT 面                       | 时序问题                            |
| ---------------: | ----------------------------- | ---------------------------- | ------------------------------- |
|          0…K_t−1 | A/B 流入，PE 累加 partial sum      | 输入 checksum 与输出 parity 同步累加  | checksum 不得阻塞 operand injection |
| K_t…K_t+T_fill−1 | 最后片段排空，C tile 完成              | 行/列校验寄存器接收结果                 | 最后 PE 稳定后才能采样                   |
|       K_t+T_fill | C 进入 commit buffer            | 锁存 syndrome 输入               | 建立 commit fence                 |
|        +1…+T_red | 下一 tile 在 ping-pong buffer 计算 | 归约、比较、单错/双错分类                | reducer 可能争用 SRAM/NoC           |
|         +T_red+1 | 正常写回或异常旁路                     | clean 提交；单错输出位置/幅值；双错 poison | 异常不能污染 writeback                |
| +T_red+2…+T_corr | 单错 word 读改写                   | 更新校验并发 corrected event       | T_corr 取决于定位、读改写和重校验            |

例：P=Q=16、K_t=16、T_fill=30、T_red=2，则 GEMM tile 产生延迟 46 cycles。稳态无错时 2-cycle 检测可与下一 tile 重叠；单错定位 1 cycle、输出读改写 1 cycle，约增加 2 个 commit 周期。未提交 partial sum 出错时，安全策略是丢弃并重算，代价接近再执行 46 cycles；双错只置 poison 并 replay。上述是参数化模板，不是固定 NPU 规格。

## 3. 故障位置与边界

| 故障 | 结果 | 动作 |
| --- | --- | --- |
| C 输出寄存器/SRAM word | 单个输出错误 | syndrome 定位后读改写 |
| 单 PE partial sum 翻转 | 可能扩散为一个 C 元素 | 未传播可局部清除，否则重算 |
| A/B/权重 word | 沿 K 维扩散到多个 C | 输入 checksum 定位 block，从可靠副本重取并重算 |
| DMA/NoC word | buffer 中坏 word | 链路 ECC/CRC 先重传，ABFT 端到端复核 |
| 两个同时错误 | 不能安全单点修正 | 仅检测、隔离、replay |

ABFT 发现的是算法关系被破坏，不自动知道物理故障位置；输入错误不能套用输出单点修正公式。

## 4. 关键路径和异常时序

1. 加权 checksum 的乘法不要塞进 MAC 同周期；用预存权重、两级 pipeline 或旁路 reducer。
2. 行/列归约用平衡树、carry-save tree 和寄存器切片；T_red 可增加但应与下一 tile 重叠。
3. s1/s0 除法放在 commit sideband；用倒数近似、查表或范围检查，避免拉长高频主路径。
4. correction queue 满时才对上游 backpressure；无错路径必须旁路。
5. 双错进入 replay/fail-stop，并记录 tile、layer、request、syndrome、重放次数。
6. 浮点舍入、量化截断和近似 SFU 会产生正常 checksum 偏差；需按相同舍入路径生成预测值并做 fault injection。ABFT 不能替代 SRAM ECC 或 NoC CRC。

## 5. NPU 落点

    DMA A/B/权重 → checksum/CRC → tile SRAM → P×Q MAC
                                          ↓
                               row/column checksum
                                          ↓
                               syndrome / SEC-DED
                 clean→commit; 1 error→correct→recheck
                 2 errors→poison→replay/reload

SRAM/ECC 提供固定延迟的物理层保护，ABFT 提供跨 PE、DMA 和算子边界的端到端检查，runtime replay 处理双错和不可纠正故障；编译器应把 checksum tile、commit fence、replay buffer 作为显式资源。

当 tile SRAM 扩展为大容量多 bank 结构时，还要覆盖 ECC 无法单独处理的 bank/decoder 故障、scrub 端口争用和 degraded-bank remap。这些机制以及对吞吐、延迟和面积的分项建模见 [[大容量多 Bank SRAM 容错设计]]。

## 6. 输入数据流接口：推荐分层

如果同时考虑 A/B/权重/KV 的输入数据流，推荐采用“内存映射搬运 + ready/valid 流 + 独立控制”的三层接口，而不是让 MAC 阵列直接暴露 DRAM 总线。

### 6.1 片外：AXI4-MM 或等价 DMA 接口

DMA 从 DRAM/HBM 读出 tile，完成 burst、地址转换、ECC/CRC 检查和 page-table 解析。descriptor 至少应包含：base/page address、byte length、tensor kind（A/B/weight/K/V）、layer/request/tile ID、shape/stride/K-step、checksum seed 或 expected checksum、epoch/version 和 poison 标志。AXI4-MM 只负责可靠搬运和突发效率，不把 ABFT 定位逻辑塞进 AXI 地址/响应通道。

### 6.2 片上：ready/valid data + sideband

从 DMA/NoC 到 tile SRAM、再到矩阵阵列，使用 ready/valid 流。每个 beat 除 W-bit packed data 外，携带：

```text
in_keep      : 尾 beat 的 byte/element mask
in_last      : tile 或 K-step 结束
in_kind      : A | B | K | V
in_coord     : layer, request, tile, row/col, k_step
in_checksum  : p0/p1/p2 或 parity fragment
in_epoch     : replay/version 标识
in_poison    : 上游已检测到不可用数据
```

只有 `in_valid && in_ready` 同时为 1 的周期才消耗一个 beat；`in_ready=0` 时发送端必须保持 data 和全部 sideband 不变。checksum accumulator 也只能在握手周期更新，否则一次 backpressure 就会造成校验漂移。

推荐 A 流和 B 流使用两个独立通道；需要严格对齐时，用相同 coord/k_step 做 join，并在 join buffer 检查 epoch、kind 和 tile ID。

### 6.3 控制/状态接口

用 AXI4-Lite、CSR 或 command queue 配置 tile shape、checksum 模式/位宽/容差、expected-checksum 地址、commit fence、replay buffer、最大重放次数；通过 event/interrupt 报告 clean、corrected、double-error、poison 和 timeout。不要用 data stream 的 last 代替任务完成事件：last 只表示流段结束。

### 6.4 握手与异常时序

```text
cycle t    : DMA valid=1，发送 A/B beat 与 checksum fragment
cycle t+1  : SRAM ready=1，完成握手；checksum accumulator 更新
cycle t+2  : ready=0（bank 冲突/阵列反压）；发送端保持 beat 不变
cycle t+3  : ready=1；同一 beat 再次握手，checksum 只更新一次
cycle ...  : last 握手后锁存 tile checksum，启动 T_red 检测
```

DMA/ECC/CRC 发现错误时优先重传；ABFT 发现输入 checksum 不一致时置 poison、禁止 commit、从可靠副本 replay；输出 syndrome 只有在 coord/epoch 仍匹配且错误位于可读改写范围时才局部纠正，否则重算。双错或校验器异常应按 request 隔离，避免阻塞无关 request。

### 6.5 最小可行接口

```text
DRAM/HBM ── AXI4-MM DMA ──► NoC/Stream (ready-valid + ABFT sideband)
                                      ├─► tile SRAM / join buffer
                                      └─► checksum reducer + MAC array
CSR/command queue ───────────────► descriptor、fence、replay、interrupt
```

RTL 原型可先实现 A-stream、B-stream 两个 ready/valid 通道；统一 stream_meta（tile_id、k_step、epoch、kind、keep、last、poison）；checksum sideband（parity 或 p0/p1/p2）；descriptor CSR；以及 clean/corrected/double_error/replay 状态事件。这样从 AXI4-MM 换成 CHI 或其他 NoC/DMA 协议时，阵列侧接口无需重写。

## 7. 资料

- [Huang & Abraham (1984), Algorithm-Based Fault Tolerance for Matrix Operations](https://doi.org/10.1109/TC.1984.1676475)
- [Performance evaluation of checksum-based ABFT (DFTVS 2001)](https://doi.org/10.1109/DFTVS.2001.966800)
- [Selective Checksum based On-line Error Correction for RRAM based Matrix Operations (2020)](https://doi.org/10.1109/VTS48691.2020.9107606)
- [Hamming code / SEC-DED 概述](https://en.wikipedia.org/wiki/Hamming_code)

精确 T_red、T_corr 和 replay 延迟必须结合阵列尺寸、checksum 位宽、SRAM 端口、时钟和 RTL pipeline 实测。
