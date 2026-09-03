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

| 周期 | 数据面 | ABFT 面 | 时序问题 |
| ---: | --- | --- | --- |
| 0…K_t−1 | A/B 流入，PE 累加 partial sum | 输入 checksum 与输出 parity 同步累加 | checksum 不得阻塞 operand injection |
| K_t…K_t+T_fill−1 | 最后片段排空，C tile 完成 | 行/列校验寄存器接收结果 | 最后 PE 稳定后才能采样 |
| K_t+T_fill | C 进入 commit buffer | 锁存 syndrome 输入 | 建立 commit fence |
| +1…+T_red | 下一 tile 在 ping-pong buffer 计算 | 归约、比较、单错/双错分类 | reducer 可能争用 SRAM/NoC |
| +T_red+1 | 正常写回或异常旁路 | clean 提交；单错输出位置/幅值；双错 poison | 异常不能污染 writeback |
| +T_red+2…+T_corr | 单错 word 读改写 | 更新校验并发 corrected event | T_corr 取决于定位、读改写和重校验 |

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

## 6. 资料

- [Huang & Abraham (1984), Algorithm-Based Fault Tolerance for Matrix Operations](https://doi.org/10.1109/TC.1984.1676475)
- [Performance evaluation of checksum-based ABFT (DFTVS 2001)](https://doi.org/10.1109/DFTVS.2001.966800)
- [Selective Checksum based On-line Error Correction for RRAM based Matrix Operations (2020)](https://doi.org/10.1109/VTS48691.2020.9107606)
- [Hamming code / SEC-DED 概述](https://en.wikipedia.org/wiki/Hamming_code)

精确 T_red、T_corr 和 replay 延迟必须结合阵列尺寸、checksum 位宽、SRAM 端口、时钟和 RTL pipeline 实测。
