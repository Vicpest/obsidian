# 大容量多 Bank SRAM 容错设计

NPU 片上 SRAM 通常同时承载权重 tile、激活、partial sum 和 K/V tile。当容量扩展到多个 bank 时，容错不能只理解为“给每个 word 加 ECC”：数据位、校验位、decoder、sense amplifier、bank 控制器和共享互连都可能故障，而 bank 冲突、scrub 和修复访问还会与计算流量争用端口。因此应分别定义故障覆盖、无错稳态带宽以及异常恢复代价。

## 1. 组织与故障边界

设 SRAM 由 `B` 个 bank 组成，每 bank 深度为 `D` word，数据宽度为 `W` bit，一次访问的有效数据宽度为 `W_payload`。bank 可独立访问时，理想峰值带宽为：

```text
BW_raw = f_clk × W_payload × N_acc
BW_useful = BW_raw × η_bank × η_protocol × (1 - ρ_FT)
```

`N_acc` 是所有 bank 每周期可并行完成的总访问数；例如 `B` 个各可完成一次访问的单端口 bank 在理想条件下有 `N_acc=B`。`η_bank` 反映 bank conflict 和负载不均，`η_protocol` 包括握手、仲裁和尾 beat 利用率，`ρ_FT` 是 scrub、诊断、修复和重放实际占用的端口周期比例。ECC 位宽本身不一定降低 payload/cycle；只要数据与校验位在同一次宏访问中读出，主要代价是宏宽、编解码组合逻辑和可能的 pipeline stage。

需要明确以下故障类型：

| 故障类型 | 典型来源 | 仅用 word-level SEC-DED 的结果 | 需要的补充机制 |
| --- | --- | --- | --- |
| 单 bit 瞬态翻转 | 辐射、噪声、低电压运行 | 可定位并纠正 | scrub 消除潜伏错误 |
| 同一码字多 bit 翻转 | 物理相邻 cell 被同一事件影响 | 可检测任意双 bit 错误但不能纠正；三 bit 及以上不保证检测 | bit interleaving、DEC/TED 或 chipkill-like symbol code |
| 固定 cell/row 故障 | 老化、制造缺陷 | 反复 corrected event，终将耗尽码距 | spare row/column、BISR、行退役 |
| 整个 bank 或端口故障 | decoder、sense amp、时钟/电源岛故障 | 无法保证可访问 | spare bank、bank remap、降级容量模式 |
| 地址/控制路径故障 | 地址 decoder、仲裁器、mux | 可能读到另一个合法码字，ECC 不报错 | 地址 parity/CRC、端到端 tag、控制器复制或 fail-stop |
| ECC 逻辑/校验位故障 | encoder、decoder、check-bit array | 可能误纠正或静默数据破坏 | decoder 自检、parity、周期性测试、端到端 ABFT |

物理 bit interleaving 应使同一局部扰动影响的相邻 cell 落入不同 ECC codeword。否则，逻辑上的 SEC-DED 会因为物理多 bit 故障聚集到一个码字而失去 SEC 能力。

## 2. 分层容错架构

```text
DMA / NoC + CRC
       │ request: bank, row, epoch, tensor/tag
       ▼
address hash / bank remap ──► spare-bank table
       │
       ├──► bank 0: data + ECC, spare rows, scrub state
       ├──► bank 1: data + ECC, spare rows, scrub state
       ├──► ...
       └──► bank B-1 / spare bank
                         │
                         ▼
             ECC decode / correct / poison
                 │              │
          clean or corrected       UE / timeout
                 │              └──► replay, reload or isolation
                 ▼
             compute engine ──► ABFT / commit fence
```

### 2.1 码字级 ECC

- **SEC-DED** 适合常规读写路径：单 bit 纠正，双 bit 检测，无错和可纠正读返回相同的 payload 宽度。
- 对 `W` 位数据，Hamming parity 位数 `r` 满足 `2^r >= W + r + 1`；增加 1 位 overall parity 后，理论校验位为 `r + 1`。实际宏还需计入 byte-write 粒度、对齐、预留位和测试位。
- 部分写入若无独立小粒度 ECC，需执行 `read -> correct -> merge -> encode -> write`。这是多 bank NPU 中容易被忽略的端口占用与写后读冒险来源。
- 不可纠正错误（UE）必须返回 `poison` 并阻止结果 commit，不能只增加错误计数器后继续计算。

### 2.2 行、列与 bank 级修复

制造测试可用 MBIST 扫描故障，BIRA 选择 spare row/column，BISR 在启动时从 eFuse/OTP 加载 remap 表。运行时对同一行的 corrected-error 计数，超过阈值后执行在线行退役：

1. 阻止目标行的新访问，等待 outstanding request 排空。
2. 逐 word 读出并纠正，将数据复制到 spare row。
3. 原子更新 remap 表和 `epoch`，让旧 epoch 的在途响应失效。
4. 重新读取、校验，然后释放 bank。如果源行出现 UE，改从 DRAM/HBM 或 replay buffer 重载。

当 decoder、sense amplifier 或共享控制器故障时，单行修复不足。可配置 `S_b` 个 spare bank，或退化到 `B-S_failed` 个 bank 的容量/带宽模式。bank remap 表应放在受保护的小型复制存储中，否则它本身会成为单点故障。

### 2.3 Scrubbing 与异常恢复

- **Patrol scrub** 以低优先级遍历所有码字，对单错执行读-纠-写，缩短两次独立翻转在同一码字中累积的时间窗。
- **Demand scrub** 在正常读命中发现可纠正错误后回写；回写可放入低优先级 correction queue，避免立即占用前台端口。
- scrub 地址应与当前 remap 表和 power state 一致，并在 bank 时钟/电源关闭时暂停；不应读取已退役行或使 bank 无法进入低功耗状态。
- UE 或 bank timeout 按数据可恢复性处理：可从 DRAM/HBM 重载的权重/激活 tile 直接 replay；未 commit 的 partial sum 丢弃并重算；无可靠副本的状态必须上报 fail-stop。

## 3. 访问时序与语义

建议把 ECC decoder 放在 SRAM 输出与 response queue 之间，使纠正后数据才能进入 PE。典型 pipeline 为：

```text
cycle t    : request 握手，bank select / remap lookup
cycle t+1  : SRAM data + check bits 读出
cycle t+2  : syndrome / SEC-DED 分类，clean/corrected data 进 response queue
cycle t+3+ : 可选 demand-scrub 回写；UE 则发 poison/replay event
```

若 syndrome 组合路径不满足频率，增加一级寄存会让单次读取延迟增加 1 cycle，但只要 request/response 均可流水，无 bank conflict 时仍可保持每 bank 每周期一个 word 的稳态吞吐。因此必须分开报告：

- **load-to-use latency**：从 request 握手到纠正数据可用的周期数。
- **initiation interval (II)**：同一 bank 接收新请求的最小间隔；吞吐受 II 而不是单次延迟直接决定。
- **recovery latency**：从 corrected event/UE 发生到回写、重载或 replay 完成的时间。

ECC 未决议前不得进行架构可见 commit。如果为隐藏 ECC latency 而让 speculative data 进入计算阵列，就必须保留 epoch 和 replay 能力，保证错误数据不会穿过 [[ABFT：检2纠1的逐周期实现|commit fence]]。

## 4. 吞吐率、延迟与面积代价

### 4.1 吞吐率与端口争用

容错对吞吐的主要影响不是 parity bit 数，而是额外访问是否占用前台端口。对单端口 bank，若每 `T_scrub` 个前台周期执行一次 scrub read，其中比例 `p_CE` 需再回写，近似有：

```text
ρ_scrub ≈ (1 + p_CE) / T_scrub
η_FT ≈ max(0, 1 - ρ_scrub - ρ_repair - ρ_replay)
```

`η_FT` 只是各类后台流量与前台流量共享端口、且前台持续饱和时的一阶近似；独立 scrub 端口或大量 idle slot 会使实际损失更小，bank conflict 和队列拥塞则可使尾延迟更大。可用以下方法减轻吞吐损失：

- 只在 bank idle slot 发射 patrol scrub，并为前台 request 保留绝对优先级；代价是高负载时 scrub 间隔可能失控，需有最大延后上限。
- 使用独立 ECC/scrub 端口或真双口 SRAM；前台吞吐更稳定，但存储 cell、外围电路、布线和动态功耗增长明显。
- 将各 bank 的 scrub phase 错开，避免同时占用 NoC/response queue；它不减少总工作量，但可降低瞬时带宽峰值与尾延迟。
- 对 partial write 设置 merge buffer，将同一码字的多个 byte write 合并为一次 read-modify-write。

整 bank 退役后，若无 spare bank，可用并行 bank 数从 `B` 降到 `B-F`。在访问能均匀重映射且数据可并行性充足时，带宽上限近似按 `(B-F)/B` 缩放；对固定 bank-stride 映射，实际损失可能更大，因为重映射会引入新的 conflict 和跨 bank 数据搬移。

### 4.2 延迟代价

| 路径 | 延迟组成 | 对 NPU 的影响 |
| --- | --- | --- |
| clean read | `T_array + T_ECC + T_queue` | `T_ECC` 若被 pipeline 则增加 load-to-use，但可保持 II=1 |
| corrected read | clean read + correction select，可选异步回写 | 数据可与 clean read 同时返回；queue 满时才反压 |
| partial write | `read + correct + merge + encode + write` | 若无 merge buffer，单端口 bank 至少占用两次 array access |
| row retirement | quiesce + `D_row` 次读/写 + remap + verify | 局部 bank 暂停；应让其他 bank/request 继续运行 |
| UE replay | detect + poison propagation + reload + kernel replay | 可能跨越 DMA 和整个 tile，尾延迟远大于 clean path |

在 Prefill 中，额外一级 ECC pipeline 通常可被大 tile 的稳态计算遮蔽；在 batch 较小的 Decode 中，依赖链短且 K/V 读取占主导，load-to-use 与 replay 会更直接反映到 TPOT 和尾延迟。

### 4.3 面积与容量代价

用 bit 数可先得到一个不包含 layout 的下界：

```text
A_data_bits  ∝ B × D × W
A_ECC_bits   ∝ B × D × C
R_ECC_bits   = C / W
R_spare_bank = S_b / B
R_spare_row  = N_spare_row / D
```

`C` 是每码字校验位数。对 64-bit payload 的 SEC-DED，理论上需 8 个 check bit，bit-cell 容量增量为 12.5%；对 128-bit payload，需 9 个 check bit，理论增量约 7.0%。这些只是校验位比例，不是 SRAM subsystem 的总面积百分比。

更完整的面积模型为：

```text
A_total = A_data_macro + A_check_macro + A_ECC_logic
        + A_spare + A_remap + A_MBIST/BISR + A_queue/telemetry
```

以下因素使总面积不能由 `C/W` 直接推导：

- 增宽后的 macro 可能跨过 compiler 宽度档位，导致列 mux、sense amp 和外围电路非线性增长。
- 细粒度码字有利于多 bit 隔离和 partial write，但每 payload bit 的 check-bit 比例、decoder 数量和布线更高。
- spare bank 保护大粒度故障，但容量与外围电路几乎整 bank 复制；spare row 对零散 cell/row 缺陷更高效，但不能修复共享 decoder 或端口故障。
- 双端口 SRAM 可隔离 scrub/修复与前台流量，但其代价不能按“多一组控制逻辑”计算，必须用目标 PDK 的 SRAM compiler 报告评估。

### 4.4 参数化算例

以 `B=16`、每 bank 每周期读取 1 个 256-bit word、`f_clk=1 GHz` 为例，原始读带宽上限为 `512 GB/s`。假设 `η_bank=0.85`、`η_protocol=0.95`、`ρ_FT=0.01`，则有效带宽估算为：

```text
BW_useful = 512 × 0.85 × 0.95 × 0.99 ≈ 409.3 GB/s
```

256-bit payload 的 SEC-DED 需 9 个 Hamming parity bit 加 1 个 overall parity bit，因此 check-bit 容量下界为 `10/256 ≈ 3.91%`。配置 1 个 spare bank 的原始 bank 容量开销为 `1/16=6.25%`；若 spare bank 也使用同样 ECC，则相对未保护的 16-bank data-bit 基线，存储 bit 总增量下界为 `17×266/(16×256)-1 ≈ 10.4%`。这仍不包含 codec、BISR、仲裁和宏形状引起的面积增量。

若 ECC decoder 增加 1 级流水，clean load-to-use 可增加 1 cycle，但在各 bank 独立流水且无额外端口占用时，仍可保持 `II=1` 和原始 `512 GB/s` 吞吐上限。因此这一配置的主要 clean-path 代价是单次访问延迟和面积，而不是必然的 payload/cycle 下降；实际吞吐损失来自 bank conflict、scrub/修复争用和降频。

## 5. 按 NPU 数据类型选择策略

| 数据 | 生命周期/可恢复性 | 建议保护 | 主要代价考量 |
| --- | --- | --- | --- |
| 权重 tile | 只读为主，可从 DRAM/HBM 重载 | SEC-DED + UE reload；高密度区后台 scrub | 可以接受较长恢复，但不应频繁重载拖垮带宽 |
| 激活/Q/K/V tile | 短生命周期，通常可重放 DMA 或前驱算子 | SEC-DED + poison/epoch + replay | Decode 的 load-to-use 和尾延迟敏感 |
| partial sum | 未 commit 前可重算，数值影响大 | SEC-DED，必要时更强 ECC；ABFT 端到端复核 | 重算 tile 的周期和能耗较高 |
| 长驻留 runtime metadata | 页表、remap、queue 状态，难以从数据面重建 | ECC + 复制/lockstep + version/CRC | 容量小，应优先提高覆盖率而非追求最低 bit 开销 |

一个实用基线是：所有数据 bank 使用 SEC-DED 和 bit interleaving；每 bank 配少量 spare rows，全局配 spare bank 或支持降级映射；scrub 采用可限流的低优先级队列；所有 UE 通过 poison/epoch 连接 runtime replay；partial sum 再由 [[ABFT：检2纠1的逐周期实现|ABFT]] 在 commit 前复核。若媒介多 bit upset 概率或安全目标更高，再对关键 bank 采用更强码，而不必对所有容量统一升级。

## 6. 评估与验证

不应只报告“ECC 面积开销”。至少需要对比 `No protection`、`SEC-DED only`、`ECC + scrub`、`ECC + row/bank repair + replay` 四类配置，并给出：

1. **可靠性**：按 bit/word/row/bank/address/control 注错的 SDC、DUE、纠正率、误纠正率和覆盖率；注错分布应与声称的故障模型一致。
2. **性能**：clean-read latency、II、每 bank/aggregate GB/s、bank-conflict rate、scrub 端口占用率、corrected-read latency、UE replay 的平均值和 P99，以及 Prefill token/s、Decode TPOT。
3. **面积与时序**：data/check-bit macro、ECC codec、spare、remap/BISR、queue/telemetry 分项面积，还要报告最大频率与 ECC/remap 是否进入关键路径。
4. **功耗/能耗**：clean read/write、scrub read-correct-write、row migration 和 replay 的每次能量，以及实际工作负载下的总能量。
5. **容量与降级**：ECC/spare 后的 usable capacity，退役 1、2、... bank 时的有效容量、带宽、bank conflict 和模型可运行性。

工作负载测试应包含连续大 tile、小 batch Decode、高 partial-write 比例、制造测试/启动修复和高负载下 scrub deadline 五类情况。最终结论应建立在目标 SRAM compiler、综合/布局布线和周期精确模拟上，不能只用 check-bit 比例估算面积，或只用无错平均值代表恢复延迟。

## 7. 与 NPU 其他层的关系

- SRAM ECC 保护存储码字，但不能完整覆盖错地址、错 tile 或 PE 计算错误；这些需要 [[ABFT：检2纠1的逐周期实现|ABFT]]、CRC 和 commit/replay 共同处理。
- bank 数、地址 hash、端口数和 scrub 配额会改变 [[存储层级、调度与性能分析|Roofline 中的有效带宽]]，因此编译器和 runtime 应能读取 degraded-bank mask 与当前带宽模式。
- 大容量 SRAM 应优先分 bank 隔离故障域和恢复阻塞；但 bank 越多，remap/hash、仲裁、校验返回和物理布线越复杂，不能将 bank 数等同于有效带宽。
