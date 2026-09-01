# Encoder、Decoder 与交叉注意力

Transformer 的 Encoder 与 Decoder 都由 Attention、残差、Norm 和 MLP 组成；区别首先体现在 Q/K/V 的来源和 mask，继而决定 NPU 的数据复用、KV Cache 生命周期与调度方式。基础单层流程见 [[Transformer 推理工作流程]]，QKV 生成与布局见 [[QKV 投影、布局与缓存]]。

## 三类注意力的计算图

| 位置 | Q 来源 | K/V 来源 | mask | 推理期缓存与特点 |
| --- | --- | --- | --- | --- |
| Encoder self-attention | 同一 encoder hidden state | 同一 encoder hidden state | padding / 可选局部 mask；通常双向可见 | 一次处理完整输入；可缓存编码结果供多个解码步骤复用 |
| Decoder masked self-attention | decoder 当前 hidden state | decoder 当前及历史 hidden state | causal mask，位置 `j > i` 不可见 | 每层持续追加 [[npu/operator/transformer common/KV Cache|KV Cache]]，Decode 带宽敏感 |
| Decoder cross-attention | decoder hidden state | encoder 输出（memory） | encoder padding mask，通常不使用 causal mask | encoder K/V 在一次生成任务中固定，可预投影并跨 Decode step 复用 |

三者的核心公式都可以写为 `Softmax(QKᵀ / √D + mask)V`。因此矩阵、向量和归约硬件相同；不同的是产生/读取 Q、K、V 的 buffer、可复用的时间范围和 mask 语义。

## Encoder block

常见 Pre-Norm Encoder block 为：

```text
u = x + SelfAttention(Norm1(x))
y = u + MLP(Norm2(u))
```

Encoder self-attention 的 `Q/K/V` 都来自同一个 token block，且 token 通常可双向互相看见。对于输入 `[B, S_enc, H]`，NPU 可连续完成 QKV GEMM、head layout、`QKᵀ`、[[npu/operator/transformer common/Softmax|Softmax]] 和 `P×V`；长序列仍应使用分块 / online softmax，避免保存 `S_enc²` 个 score。

推理时 Encoder 不会像自回归 Decoder 那样每 token 追加历史状态。若一个 encoder 输出要服务多个 decoder step、多个 beam 或多个候选答案，保留其最终 memory 及已投影的 cross-attention K/V 通常比重复投影更划算。视觉 Transformer 的 patch token 也可按这一类映射，只是常带二维位置编码或窗口/稀疏 attention。

## Decoder-only block（LLM 常见）

Decoder-only LLM 的 block 一般是 masked self-attention 加 MLP：

```text
u = x + MaskedSelfAttention(Norm1(x))
y = u + MLP(Norm2(u))
```

对于当前第 `t` 个 token，Q 只查询 `0…t` 的 K/V。Prefill 将提示词作为矩阵处理，主要追求 [[npu/operator/tensor operator/GEMM|GEMM]] 吞吐；Decode 每一步只生成少量 Q/K/V，却要读取长度近似为 `t` 的历史 K/V，常转为内存带宽受限。

```text
每层 Decode：
hidden(t) → Norm → QKV projection → RoPE(Q,K)
         → append K(t), V(t) to layer cache
         → Q(t) × K(0…t)ᵀ → softmax(causal) → P × V(0…t)
         → Wo / residual / MLP → 下一层
```

硬件调度应把不同请求的 Decode token 做连续批处理，提高小 GEMM 的阵列利用率；但每个请求的 page table、有效长度和 causal 边界必须分别传给 kernel。不能把不同请求的 Cache 直接拼成一条可互相注意的序列。

## Encoder–Decoder Transformer

翻译、摘要等 seq2seq 模型常在每个 Decoder block 中放置两段 attention：masked self-attention 后接 cross-attention。

```text
u = x + MaskedSelfAttention(Norm1(x))
v = u + CrossAttention(Norm2(u), encoder_memory)
y = v + MLP(Norm3(v))
```

Cross-attention 的 `Q = uWq` 来自当前 Decoder 状态，`K = memoryWk`、`V = memoryWv` 来自固定的 Encoder 输出。它与 self-attention 的关键实现差异如下：

- Encoder memory 完成后，可一次性预计算每层 cross-attention 的 K/V；Decode step 只计算新 Q 和读取这些静态 K/V。
- cross-attention 的长度是 `S_dec × S_enc`，而不是 decoder 自身的 `S_dec × S_dec`；mask 屏蔽的是 encoder padding 或跨模态无效位置。
- Beam search 中不同 beam 通常共享同一 encoder memory 与 cross K/V；它们的 decoder self-attention Cache 则必须各自维护或采用 copy-on-write 共享前缀。

如果模型采用 multi-source、检索增强或视觉语言结构，只要各 memory source 的 K/V 来源不同，NPU 就需保留独立的长度、mask、page/address table 与同步事件；不可把它们当作普通 decoder self-attention Cache。

## NPU 资源与同步

```text
Encoder input ──► QKV / Attention / MLP ──► encoder memory ──┐
                                                              ├─► Cross K/V preprojection/cache
Decoder token ──► self QKV ──► self KV append ──► self Attention ─┤
                                                                  ▼
                                                     cross Q + encoder K/V
                                                                  ▼
                                                       output projection / MLP
```

- QKV、输出投影和 MLP 映射到矩阵引擎；RoPE、残差、Norm 映射到向量/归约引擎；softmax 使用归约与 SFU。硬件职责见 [[NPU 总体硬件架构]]。
- Encoder memory 与 decoder self KV 的生命周期不同：前者在一个生成任务/请求内只读，后者每层每步写一次、读很多次。应使用不同 buffer 管理与 DMA 预取策略。
- Cross K/V 若能放入片上 SRAM，可按 Q tile 多次复用；放不下时，应按 encoder sequence 分块流式读取并采用 online softmax，而不是物化完整 attention score。

## 验证用例

1. **Encoder padding**：改变 padding token 的值，非 padding 输出不应改变。
2. **Decoder causal**：改变未来 token，当前 token 的输出不应改变；改变历史 token，应允许输出改变。
3. **Cross source 隔离**：改变 encoder memory 会影响 cross-attention 输出；不应改变 decoder self KV 的内容。
4. **Prefill/Decode 一致性**：在相同权重、position 和数值精度下，逐 token Decode 的输出应与全序列 causal Prefill 在对应位置一致（允许量化与归约顺序造成的既定容差）。

有关 Attention 的在线 softmax、mask 施加位置和 Cache 带宽，参见 [[Attention 硬件实现]] 与 [[存储层级、调度与性能分析]]。
