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
