# KV Cache

KV Cache（Key-Value 缓存）保存已处理 token 的 Key 与 Value，供自回归解码中的后续 token 复用。

## 功能

每层每个新 token 产生的 K/V 会追加到缓存。下一个 token 的 Q 直接与历史 K 计算 score，并以对应 V 加权，无需重新计算历史 token 的 K/V。

## 在 NPU 里面哪里会出现 KV Cache

1. LLM 增量解码的每一层 [[Attention]]。
2. 长上下文推理中的分页 KV 管理、连续批处理和多请求复用。

## 实现要点

- 它本质上受容量与带宽约束：小缓存可置于片上 SRAM，长上下文通常置于 HBM/DRAM。
- 分页、量化、分组查询注意力（GQA）和 [[Layout Transform]] 决定实际访存效率。
- 若使用 [[RoPE]]，须明确缓存的是旋转前还是旋转后的 K，并与读出路径保持一致。
