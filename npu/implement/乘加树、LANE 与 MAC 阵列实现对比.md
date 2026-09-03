# 乘加树、LANE 与 MAC 阵列实现对比

NPU 中的“乘加单元”不只有一种组织方式。应把 **单个 MAC 串行累加、乘加树（parallel multipliers + adder tree）、LANE 级 SIMD/vector、二维 MAC/脉动阵列** 看成不同的并行粒度，而不是互相替代的同义词。选择取决于 M/N/K 形状、输入流复用、片上 SRAM 带宽、归约延迟和是否需要持续吞吐。

## 1. 四种组织方式

### 1.1 MAC 串行链

一个 accumulator 每周期接收一个乘积：

```text
acc[t+1] = acc[t] + a[t] × b[t]
```

面积小、控制简单、适合 GEMV 尾块和低吞吐场景；但一个 K 维点积需要 K 个周期，反馈依赖把时钟关键路径和吞吐绑定在 accumulator 上。可复制多条 MAC 链形成 lanes，但每条链仍有 K 周期归约。

### 1.2 乘加树（dot-product tree）

先并行生成 K 个乘积，再用平衡树归约：

```text
K multipliers → K/2 adders → K/4 adders → … → 1 sum
```

未流水化的组合深度约为 `1 multiplier + ceil(log₂K) adder levels`；完全流水化后可每周期接收一个新向量，但首个结果延迟约为 `T_mul + ceil(log₂K)·T_add + register stages`。它消除了 MAC feedback，适合固定 K 的 dot product、卷积窗口、RMSNorm/Softmax reduction；代价是并行乘法器和加法器数量多，布线/功耗/寄存器面积随 K 增长。

### 1.3 LANE 级 SIMD/vector

将数据划分为 L 条 lane，每条 lane 有乘法器、ALU、局部累加器和寄存器文件；lane 间通过 cross-lane reduction、shuffle 或 permute 合并：

```text
lane0: a0×b0 + acc0 ┐
lane1: a1×b1 + acc1 ├─► cross-lane reduce ─► dot / elementwise result
...                  ┘
```

LANE 级单元在同一硬件上兼顾 elementwise、RoPE、Norm、量化和小 GEMV。对不满 L 的尾向量需 mask；不同 request 的 Decode token 可以打包到 lanes，但每个 lane 仍要保留独立长度、page table 和 causal 边界。

### 1.4 二维 MAC/脉动阵列

用 P×Q 个 PE 并行计算多个 C[m,n]，A/B 在阵列边界注入并在 PE 间转发，partial sum 保留在 PE 或输出驻留 buffer。它通过 weight/output/row-stationary 数据流复用操作数，适合 Transformer 的大 GEMM、QKV 和 MLP。代价是阵列填充/排空、尾块利用率、SRAM/NoC 带宽和映射复杂度。

## 2. 周期、面积和带宽对比

| 组织 | 每个点积的主要延迟 | 稳态吞吐 | 片上数据复用 | 面积/功耗 | 适合 NPU 工作负载 |
| --- | --- | --- | --- | --- | --- |
| 单 MAC 串行链 | K cycles（反馈累加） | 低；复制链后线性增加 | 低到中 | 最小；反馈和时钟容易收敛 | 小 GEMV、尾块、控制/校验 |
| 乘加树 | `T_mul+log₂K·T_add`；流水后可 1 vector/cycle | 高（固定 K 时） | 中；需并行输入 K 个元素 | 乘法器/加法器多，树布线和寄存器功耗高 | dot product、卷积窗口、Norm/归约 |
| LANE SIMD/vector | 向量长度/L + cross-lane reduce | 高；受 lane mask 和归约限制 | 中；寄存器/SRAM 可复用 | 可扩展，控制和 shuffle 复杂度中等 | RoPE、Norm、activation、quantization、小 GEMV |
| P×Q MAC 阵列 | `K_t + P + Q − 2` 级填充/排空近似 | 大 GEMM 时最高 | 高；权重/激活/partial sum 可驻留 | PE、SRAM、NoC 面积大；大阵列时钟/布线难 | QKV、Wo、MLP、Prefill GEMM |

表中周期是结构级估算；实际还需加入 DMA、bank conflict、量化和 commit/replay 延迟。

## 3. 与输入数据流的接口含义

### 3.1 乘加树需要“定长向量帧”

树的 K 个乘法器必须在同一 frame 对齐输入。接口应携带 `frame_id、k_base、vector_len、keep/mask、last`；缺一个元素不能靠 ready/valid 自动“跳过”，而要用 mask 或 padding。适合 DMA 一次搬入完整 K tile，或者在 SRAM 中先聚合。

### 3.2 LANE 需要 lane mask 和独立上下文

建议每个 beat 携带：

```text
lane_data[L]
lane_mask[L]
request_id[L]
sequence_len[L]
page_id[L]
```

在 continuous batching 的 Decode 中，lane0/lane1 可能属于不同请求，不能共享一个 causal 边界或 KV page table。

### 3.3 MAC 阵列需要二维坐标和 K-step

阵列输入接口重点是 `tile_id、row、col、k_step、epoch`，并要求 A/B 流在 join buffer 中对齐。阵列可以接受连续流，但必须处理 `T_fill`、尾块和双缓冲；详见 [[GEMM 与脉动阵列实现]] 与 [[Tile 数据流与解耦控制]]。

## 4. 如何组合，而不是单选

实际 NPU 通常采用异构组合：

```text
大 GEMM       → P×Q MAC / systolic array
固定 K dot    → 乘加树或 tensor-dot 单元
Norm/Softmax  → LANE SIMD + reduction tree
RoPE/activation→ LANE SIMD/SFU
尾块/校验     → 少量 MAC 链 + checksum reducer
```

乘加树也可作为 MAC 阵列的局部 PE：每个 PE 内并行处理 4/8/16 个低比特乘积，再把局部和送入阵列级 accumulator；这能减少 K-step 次数，但会增加 PE 内部面积和局部 SRAM 端口需求。

## 5. 选择准则

1. **M、N 很大且规则**：优先 MAC/脉动阵列；阵列复用能摊薄填充和控制开销。
2. **K 固定且较小**：乘加树可用较低的归约延迟完成 dot product；应流水化树，避免组合深度成为关键路径。
3. **M≈1 的 Decode/GEMV**：LANE SIMD 或多条 MAC 链通常比整块二维阵列更容易保持利用率；通过 continuous batching 填充 lanes。
4. **Norm/Softmax/Reduce**：LANE + reduction tree 优于把所有操作映射到矩阵阵列；需要注意 exp/rsqrt 等 SFU 延迟。
5. **稀疏/动态 mask**：LANE 和树要有 mask/compaction；固定 MAC 阵列更适合 block/规则稀疏，任意 gather 会损失复用。
6. **ABFT**：checksum accumulator 可放在 LANE/树旁路；不要把 checksum feedback 插入每个 MAC 的高频主路径，详见 [[ABFT：检2纠1的逐周期实现]]。

## 6. 与 ABFT 的结合

- 乘加树：可在树的每一级附加 parity 或 carry-save checksum；树完成后在 `log₂K` 级同步生成 syndrome。
- LANE：每 lane 维护局部 `p0/p1/p2`，cross-lane reduction 后比较 expected checksum；lane mask 必须同时作用于数据和 checksum。
- MAC 阵列：在输入 DMA 和输出 commit buffer 维护行/列 checksum；异常时按 tile replay，避免逐 PE 复制昂贵纠错器。

## 7. 可核验资料

- [Jouppi et al. (2017), In-Datacenter Performance Analysis of a Tensor Processing Unit](https://doi.org/10.1145/3079856.3080246)：TPU 脉动矩阵乘阵列及其系统级数据流背景。
- [Huang & Abraham (1984), Algorithm-Based Fault Tolerance for Matrix Operations](https://doi.org/10.1109/TC.1984.1676475)：矩阵 checksum 与算法级冗余。
- [A Hardware Accelerator for Computing an Exact Dot Product (2017)](https://doi.org/10.1109/ARITH.2017.38)：并行精确 dot-product 与归约数据通路。
- [A 300mV 494GOPS/W Reconfigurable 4-Way SIMD Vector Processing Accelerator (ISSCC 2009)](https://doi.org/10.1109/ISSCC.2009.4977407)：低功耗 SIMD lane 组织案例。

现有 [[NPU 总体硬件架构]] 中的 Matrix engine、Vector engine、Reduction/SFU，正对应上述三类互补计算资源；不要用单一 MAC 阵列替代所有归约、向量和特殊函数路径。
