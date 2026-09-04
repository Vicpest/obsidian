这次建议按**算法贡献**来划分，而不是按电路、架构和系统层次划分。三个方向应分别解决：

> **如何避开故障、如何发现故障、如何恢复故障。**

并且每个方向都要针对 LLM 的算子结构和自回归推理特征，不能只是把 CNN 容错方法直接移植过来。

# 方向一：面向 LLM 的故障感知 PE 映射与数据流优化

## 核心问题

在 PE 故障图、时序风险或可靠性预算已知的情况下，如何重新安排：

- QKV 投影；
- FFN 的上下投影；
- \(QK^T\) 和 \(AV\)；
- Attention Head；
- Token；
- 权重、激活和 MAC 顺序；

使高风险计算尽量避开高风险 PE。

现有工作已经研究了故障感知权重映射、显著性映射和时序关键输入模式，但主要对象是 CNN 或普通 Transformer。[Han19, Oze22, Zha23]

## 可研究的算法

建立联合优化问题：

\[
\min_{\pi,\sigma,\rho}
\alpha L_{\mathrm{fault}}
+
\beta L_{\mathrm{timing}}
+
\gamma C_{\mathrm{runtime}}
\]

其中：

- \(\pi\)：权重或 Token 到 PE 的映射；
- \(\sigma\)：数据流和 MAC 执行顺序；
- \(\rho\)：不同算子的保护等级；
- \(L_{\mathrm{fault}}\)：故障引起的模型精度或困惑度损失；
- \(L_{\mathrm{timing}}\)：时序错误风险；
- \(C_{\mathrm{runtime}}\)：性能和能耗开销。

具体可以研究：

- Attention Head 到 PE 阵列的可靠性约束映射；
- Prefill 和 Decode 阶段不同的 PE 调度；
- 对高重要性 Token、权重通道和累加路径进行保护性映射；
- 根据故障 PE 位置调整 QKV 和 FFN 的 Tile 划分；
- 结合 MAC 顺序重排，减少长进位链和关键输入模式；
- 对永久故障和 timing error 同时进行映射优化。

## 关键创新

不要做成单纯的“故障敏感性分析”，而要形成：

> **故障感知的 LLM 编译映射和数据流优化算法。**

最终输出应该是新的 Mapper、Scheduler 或 Dataflow Optimizer。

## Q2 潜力条件

需要证明：

- 比普通映射和 SalvageDNN 类方法更好；
- 支持动态序列长度和 Prefill–Decode；
- 同时评估 PE 故障位置和 LLM 输出质量；
- 有大规模搜索、启发式算法或优化理论；
- 有真实 NPU 执行周期、面积、能耗或 FPGA 数据。

---

# 方向二：面向 LLM 算子链的自适应算法级错误检测

## 核心问题

LLM 的 Attention 并不是单一 GEMM，而是：

\[
QK^T
\rightarrow
\operatorname{Softmax}
\rightarrow
AV
\]

如果分别检查每个 GEMM，可能带来较大开销，也无法完整覆盖 Softmax。

已有 Flash-ABFT 和 FT-Transformer 开始进行 Attention 的端到端保护，但仍可以进一步研究适用于 NPU PE 阵列、动态 Tile 和混合精度的统一检测方法。[Tit25, Dai25]

## 可研究的算法

设计分层自适应 ABFT：

### 算子级校验

保护：

- QKV 投影；
- FFN GEMM；
- \(QK^T\)；
- \(AV\)。

可以采用：

- Strided checksum；
- Tile checksum；
- 模校验；
- 稀疏或低秩校验；
- 混合精度误差阈值。

### Attention 端到端校验

将 Softmax 的归一化项纳入校验关系，例如：

\[
\operatorname{Check}(O_i)
=
\frac{
\sum_j \exp(s_{ij})\sum_d V_{j,d}
}{
\sum_j \exp(s_{ij})
}
\]

这样可以跨越 \(QK^T\)、Softmax 和 \(AV\) 进行一致性验证。

### PE 级定位

在 Tile 级错误被发现后，通过：

- 行列残差；
- 代数签名；
- 时间签名；
- 可编程测试向量；

进一步判断具体故障 PE、寄存器或累加器。

## 关键创新

建议把重点放在：

> **同一个检测框架同时支持 QKV、FFN、Attention 和混合精度，并能从算子级错误进一步定位到 PE。**

这比单独设计一个 Attention checksum 更有论文价值。

## Q2 潜力条件

需要具备：

- 数学推导和检测条件；
- 单比特、多比特和 timing error 实验；
- 动态 Tile 和不同序列长度；
- INT8、FP8、FP16 误差阈值处理；
- 检测覆盖率、定位准确率、误报率和开销；
- 与传统 ABFT、Flash-ABFT、DMR/TMR 的比较。[Liu23b, Tit25, Dai25]

---

# 方向三：面向自回归生成的故障严重度感知恢复与误差补偿

## 核心问题

LLM 推理不一定需要恢复每一个错误数值。关键是判断：

- 哪些错误会导致后续 Token 持续错误；
- 哪些错误可以直接忽略；
- 哪些错误需要重新执行；
- 哪些错误可以通过补偿或掩蔽处理。

这与普通 CNN 的一次性分类不同。LLM 中错误可能通过：

- 后续 Layer；
- Attention；
- KV 状态；
- 自回归 Token 生成；

不断传播。

## 可研究的算法

建立错误严重度评分：

\[
S_{\mathrm{err}}
=
w_1\left\lvert \Delta_{\mathrm{checksum}}\right\rvert
+
w_2\Delta_{\mathrm{activation}}
+
w_3P_{\mathrm{prop}}
+
w_4I_{\mathrm{token}}
\]

其中：

- \(\Delta_{\mathrm{checksum}}\)：校验残差；
- \(\Delta_{\mathrm{activation}}\)：激活偏差；
- \(P_{\mathrm{prop}}\)：错误传播风险；
- \(I_{\mathrm{token}}\)：当前 Token 的重要性。

根据严重度选择恢复方式：

| 错误级别 | 恢复方式 |
|---|---|
| 轻微错误 | 直接忽略、裁剪或置零 |
| 中等错误 | 前向补偿、激活修正或局部近似恢复 |
| 严重错误 | Tile 级重执行 |
| 持续性 PE 故障 | 禁用 PE 并重映射任务 |
| 关键 Token 错误 | Token 级回滚或重新生成 |

前向补偿可以估计 PE 错误造成的部分和差值，并在后续累加中抵消；ApproxABFT 已经展示了根据误差大小选择是否恢复的基本思路。[Liu21b, Xue23c]

## LLM 特有的研究点

需要分别考虑：

- Prefill 阶段的大 Tile 重执行；
- Decode 阶段的 Token 级恢复；
- 不同 Attention Head 的错误传播差异；
- 输出投影层的严格恢复；
- KV 相关计算的状态一致性；
- 生成质量而不是单纯数值误差。

## 关键创新

> **将 PE 错误恢复目标从“数值完全正确”转化为“生成结果质量最优”，并根据错误严重度动态选择重执行、补偿或掩蔽。**

这不是简单的错误掩蔽，而是一个面向 LLM 生成质量的自适应恢复策略。

## Q2 潜力条件

需要评估：

- 困惑度；
- Token 错误率；
- 生成质量；
- 下游任务准确率；
- 首 Token 延迟；
- 每 Token 延迟；
- 重执行次数；
- 故障率变化下的吞吐量；
- 与精确重计算、ApproxABFT 和普通 masking 的比较。

---

# 三个方向的边界

| 方向 | 主要解决的问题 | 典型输出 |
|---|---|---|
| 方向一 | 如何让重要计算避开高风险 PE | 故障感知 Mapper 和 Dataflow Optimizer |
| 方向二 | 如何发现和定位错误 | LLM 算子级 ABFT 与 PE 定位算法 |
| 方向三 | 如何以最低代价恢复结果 | 严重度感知的重执行、补偿和掩蔽算法 |

## 推荐的三个题目

1. **Reliability-Aware PE Mapping and Dataflow Optimization for LLM NPUs**
2. **Adaptive End-to-End Algorithm-Based Fault Detection for LLM Operator Pipelines**
3. **Fault-Severity-Aware Error Compensation and Selective Recovery for Autoregressive LLM Inference**

这三个方向分别偏向：

- **映射与数据流算法**；
- **检测与定位算法**；
- **恢复与误差补偿算法**。

其中，方向二最容易形成严格的算法理论和硬件验证，方向三最能体现大模型特色，方向一最适合研究 PE 映射、调度和可靠性优化。

可以。我按前面三个细分方向，整理两类指标：

1. **文献指标**：年份、期刊/会议、工作区引用数；
2. **技术指标**：故障模型、算子、方法、开销、容错效果。

引用数为工作区当前返回值，约截至 **2026 年 9 月 3 日**。不同论文的故障模型和实验条件不同，指标不能直接横向排名。

# 方向一：故障感知 PE 映射与数据流优化

| 文献 | 主要对象 | 核心方法 | 关键技术指标 | 引用数 |
|---|---|---|---|---:|
| [Zha18] | 脉动阵列、永久 MAC 故障 | FAP、FAP+T，故障感知剪枝和再训练 | 最高约 50% MAC 故障率下仍可运行；AlexNet 精度损失约 8%；旁路面积约 9%；运行时无明显性能损失 | 156 |
| [Han19] | CNN 加速器永久故障 | SalvageDNN，按权重/神经元显著性进行故障感知映射 | 不需要大规模重新训练；通过偏置校准补偿；安全旁路 PE 面积约增加 25% | 41 |
| [Oze22] | 卷积和全连接层、WS/IS/OS 数据流 | 硬件感知训练、BN 校准、对角权重平移 | MLP 在 50% PE 旁路率下精度损失不超过 0.61%；ResNet-32 在 10% 旁路率下损失约 1.60%；8-bit PE 面积开销约 7.09%，功耗约 5.36% | 12 |
| [Zha23] | 卷积 MAC、时序错误 | READ，按输入模式重排 MAC 顺序 | 平均 TER 降低约 7.8 倍，部分层最高约 37.9 倍；1024 通道 LUT 小于 2 KB；无精度损失 | — |
| [Pap24] | 脉动阵列数据流 | 比较 WS、OS 等数据流对故障传播的影响 | 重点指标是错误传播范围、故障位置敏感性和数据流鲁棒性 | 2 |

## 对该方向的启示

已有工作主要验证了：

- 故障 PE 不一定要直接禁用；
- 权重显著性和数据流会改变故障影响；
- MAC 顺序可以影响 timing error。

但现有结果大多基于 CNN 或普通 DNN。面向 LLM 仍缺少：

- Prefill 和 Decode 阶段差异；
- QKV、FFN、Attention Head 级映射；
- 动态序列长度；
- Token 位置和自回归传播；
- INT8、FP8、FP16 混合精度下的可靠性映射。

因此，你可以把指标扩展为：

\[
\text{PPL损失}
+
\text{PE利用率}
+
\text{映射开销}
+
\text{故障下吞吐量}
\]

---

# 方向二：面向 LLM 算子链的自适应 ABFT

| 文献 | 保护算子 | 核心方法 | 关键技术指标 | 引用数 |
|---|---|---|---|---:|
| [Liu23b] | Transformer GEMM、QKV、MLP | ALBERTA，选择性层保护和 checksum | 单比特翻转下错误覆盖率超过 99%；计算开销小于 0.2%；内存开销小于 0.01%；平均校正开销小于 2% | 10 |
| [Xue23d] | ViT 的 GEMM、FC、Softmax、GELU | LB-ABFT 加范围保护 | 在高 BER 下，混合方法可以大幅恢复精度；LB-ABFT 开销控制在约 2%以内；Softmax/GELU 使用低成本范围检查 | 37 |
| [Tit25] | 完整 Attention | Flash-ABFT，跨越 \(QK^T\)、Softmax 和 \(AV\) 的单次校验 | 错误检测率约 96.94%–98.87%；面积开销约 5.3%；能耗开销低于 1.9%；基本不增加额外周期 | 4 |
| [Dai25] | Attention、Softmax、Linear Module | FT-Transformer，Strided ABFT 加 SNVR | 平均容错开销约 13.9%；相对传统方法加速约 3.69–7.56 倍；GPT-2 中检测开销约 4.7%，校正开销约 9.1% | 26 |
| [Hua26] | 量化 NPU 的 Tile 级矩阵计算 | TR-ABFT，Tile 级 checksum 和 INT8 模校验 | 16×16 阵列上额外计算开销约 6.37%–24.61%；验证了 ResNet、YOLOv8 和 RT-DETR | — |
| [Zha20] | CNN 卷积 | FT-CNN，多级块、行、列校验和 | 支持卷积软错误和多种卷积实现；误差检测和纠正开销约 4%–8% | 116 |

## 对该方向的启示

当前文献的主要指标包括：

- 错误检测率；
- 错误覆盖率；
- 误报率和漏报率；
- 计算开销；
- 内存开销；
- 校正延迟；
- 对 Softmax、GELU 等非线性算子的覆盖程度。

对于你的 LLM-NPU 方向，Q2 论文最好进一步增加：

- PE 级定位准确率；
- 动态 Tile 支持；
- Prefill 和 Decode 的检测开销；
- FP8/INT8 下的数值容差；
- QKV、Attention、FFN 的统一检测；
- 多比特错误和 timing error；
- 不只是“发现错误”，还要判断错误来自哪个 PE 或哪一类运算。

最有价值的指标可以设计为：

\[
\text{Detection Coverage}
+
\text{PE Localization Accuracy}
+
\text{Mixed-Precision Robustness}
+
\text{LLM Quality Loss}
\]

---

# 方向三：错误严重度感知恢复与前向补偿

| 文献 | 核心方法 | 恢复机制 | 关键技术指标 | 引用数 |
|---|---|---|---|---:|
| [Xue23c] | ApproxABFT | 根据误差阈值决定忽略、定位或恢复 | 相比经典 ABFT，计算开销降低约 25.92%–81.62%；模型精度提升约 2.63%–72.56%；支持分块保护 | 10 |
| [Liu21b] | Forward Error Compensation | 用影子触发器检测错误，并估计部分和误差，在后续周期补偿 | 不暂停流水线；验证了 ResNet50 和 VGG16；主要面向主动故障注入和瞬态错误 | 3 |
| [Bur22] | MOZART+ | 在线测试故障 PE，并将其输出置零或屏蔽 | 单个 PE 故障时精度损失小于 3%，未保护时损失约 15%–33%；测试逻辑面积不超过 8% | 24 |
| [Cha26] | ADPR、ATPR | 精确 PE 与近似 PE 比较；错误时用近似结果替换 | ADPR 额外开销约为传统 DMR 的 38%–53%；ATPR 使用两个近似 PE 进行共识判断；不保证精确恢复 | — |
| [Jia24] | PerFT-N | 故障线程重链接和任务重映射 | 面积开销约 2.7%，功耗约 3.6%；极端情况下约 98.5% PE 故障率仍可运行，但延迟显著增加 | 2 |
| [Gao25] | Detect-and-Replace | 通过校验差值定位故障 PE，并切换到备份 PE | 支持单 PE 软错误的定位与替换；主要假设检测窗口内只有一个 PE 故障 | 6 |

## 对该方向的启示

已有恢复方法主要关注：

- 是否触发恢复；
- 恢复计算开销；
- 精度损失；
- 是否需要暂停流水线；
- 是否允许近似结果。

面向 LLM 可以进一步研究：

- Token 级错误严重度；
- Attention Head 级传播风险；
- Prefill 和 Decode 的不同恢复策略；
- 生成质量而不是单次分类准确率；
- 错误对后续 Token 的累积影响；
- 结合重执行、前向补偿和掩蔽的多级恢复。

建议使用以下指标：

| 指标 | 含义 |
|---|---|
| PPL degradation | 困惑度增加 |
| Token error rate | 错误 Token 比例 |
| Recovery success rate | 恢复成功比例 |
| Recovery latency | 单次恢复延迟 |
| TTFT | 首 Token 延迟 |
| TPOT | 每 Token 解码延迟 |
| Throughput loss | 故障下吞吐量下降 |
| Energy per token | 每 Token 能耗 |
| Error accumulation depth | 错误持续传播的 Token 数量 |

# 三个方向的指标重点

| 方向 | 最重要的技术指标 | 主要短板 |
|---|---|---|
| 故障感知映射与数据流 | PPL 损失、故障下 PE 利用率、吞吐量、映射时间、TER | 现有工作多集中于 CNN，LLM 的 Prefill–Decode 特征不足 |
| 自适应 ABFT | 检测覆盖率、PE 定位率、误报率、计算/内存开销 | 很多方法能检测算子错误，但不能精确定位 PE |
| 严重度感知恢复 | PPL、错误 Token 率、恢复成功率、TTFT、TPOT、能耗 | 现有工作较少评估自回归错误累积和生成质量 |

## 对你课题最有价值的指标组合

如果要形成面向 LLM 的 Q2 论文，建议每个方向至少包含：

1. **故障指标**：bit flip、stuck-at、timing error、多 PE 故障；
2. **硬件指标**：面积、功耗、频率、存储和控制开销；
3. **算法指标**：检测率、定位率、恢复率、误报率；
4. **LLM 指标**：PPL、Token error rate、TTFT、TPOT、生成质量；
5. **对比基线**：无保护、ABFT、DMR/TMR、masking、re-execution 和已有 Transformer 容错方法。

最明显的研究空缺是：

> 现有方法通常只优化“数值正确性”或“分类准确率”，而面向 LLM 的 NPU 容错还需要同时优化 **PE 故障定位、生成质量、自回归错误累积和每 Token 性能**。
