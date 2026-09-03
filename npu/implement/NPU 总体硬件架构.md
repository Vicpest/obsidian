# NPU 总体硬件架构

面向 Transformer 的 NPU 通常由计算、片上存储、数据搬运和控制四部分组成。不同厂商命名不同，但职责相近。

```text
                 ┌────────────── Host / Runtime ──────────────┐
                 │ 图编译、tile 切分、命令队列、同步事件       │
                 └─────────────────────┬───────────────────────┘
                                       ▼
┌─────────────── NPU ─────────────────────────────────────────────────┐
│  Command processor / scheduler                                       │
│       │                 │                         │                 │
│       ▼                 ▼                         ▼                 │
│  DMA / NoC ─────► Shared SRAM / L2 ◄─────── DMA / NoC               │
│       │                 │                         │                 │
│       ▼                 ▼                         ▼                 │
│  Matrix engine       Vector engine           Reduction / SFU         │
│  PE array / MAC      elementwise, RoPE       exp, rsqrt, reciprocal  │
│       │                 │                         │                 │
│       └──────────────► accumulator / SRAM ◄──────┘                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
                         DRAM / HBM / KV Cache
```

## 模块职责

| 模块                | 主要职责                | 映射算子                                      |
| ----------------- | ------------------- | ----------------------------------------- |
| Matrix engine     | 大规模乘加与输出累加          | [[GEMM]]、[[Convolution]]                  |
| Vector engine     | 逐元素运算、格式转换、寄存器级数据重排 | [[Elementwise]]、[[RoPE]]、[[Quantization]] |
| Reduction / SFU   | Sum/Max 归约及复杂数值函数   | [[Reduction]]、[[Softmax]]、[[RSQRT]]       |
| SRAM / cache      | 提供高带宽 tile 缓冲与数据复用  | 激活、权重 tile、partial sum、K/V tile           |
| DMA / NoC         | 片外/片上数据搬运与多核分发      | 权重预取、结果写回、KV 读写                           |
| Command processor | 执行编译器发出的指令、同步和事件    | tile 循环、双缓冲、核间调度                          |

## 数据类型与累加精度

- **输入/权重**：INT8、INT4、FP8、BF16 等决定 MAC 吞吐和带宽。
- **累加器**：通常比乘数更宽，例如 INT8 点积常累加到 INT32；低精度浮点常以 FP16/FP32 累加。
- **输出重标定**：写回前由 [[npu/operator/tensor operator/Quantization|Quantization]]、[[npu/operator/basic/LZD|LZD]] 与移位/舍入路径处理。

矩阵引擎的详细映射见 [[GEMM 与脉动阵列实现]]；数据搬运与性能权衡见 [[存储层级、调度与性能分析]]。

## 从模块图到 tile 阵列

上述模块可以组织为多个相同或近似相同的 compute tile：tile 内有本地 SRAM、矩阵/向量/归约资源和轻量任务控制；tile 间由 NoC 交换数据和事件，边缘/全局 DMA 连接外部内存。控制面只提交任务与依赖，数据面持续运行 GEMM、attention 或向量 kernel。这样的分层有利于扩展阵列规模、安排双缓冲和降低控制对高带宽路径的干扰，详见 [[Tile 数据流与解耦控制]]。

如果设计声称支持稀疏或 MoE，还要在模块图之外说明压缩格式、metadata/索引路径、动态路由和通信调度；仅增加“sparse TOPS”并不能说明端到端收益，见 [[稀疏计算与条件执行]]。
若 NPU 需要算法级容错，应在 DMA、tile SRAM、PE 阵列和 commit path 之间增加 checksum/syndrome sideband，并把单错纠正、双错检测和 replay 作为显式异常路径，详见 [[ABFT：检2纠1的逐周期实现]]。
