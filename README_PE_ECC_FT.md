# PE 计算与 SRAM 容错子系统

本目录现在包含一个可直接挂到系统 APB 总线上的 PE 容错核：

- `PeFaultToleranceAdapter.sv`：计算侧 ABFT 巡检、单点错误定位、修复与统计。
- `Ecc_Sram_Apb_Wrapper.v`：12×128-bit 本地 SRAM，逻辑码字为 128-bit 数据加 24-bit BCH 校验码。
- `PeEccFtCore.sv`：把上述两个容错单元放进同一个 PE APB 地址空间，并负责 APB 响应复用。
- `uvm/`：完整的 APB UVM agent、driver、monitor、coverage、scoreboard、sequence、test 和仿真顶层。

这里仍然只实现“容错部分”，不包含正常 PE 的 ALU/MAC/FMA 等执行流水线。现有计算容错块运行确定性的 4×4 矩阵乘 ABFT 巡检，用于验证行/列校验综合征、单点定位和修复机制。接入真实 PE 时，可以保留顶层、APB 寄存器和错误统计路径，再把 `PeAbftMatrixPatrol` 的确定性输入替换为真实计算核提供的结果与算法校验和。

## 顶层接口

集成时实例化 `PeEccFtCore`。主要接口如下：

- `clk/rstn`：计算容错域时钟和低有效复位。
- `Pclk/Prstn`：APB 与 ECC SRAM 域时钟和低有效复位。
- `Psel/Penable/Pwrite/Paddr/Pwdata/Pstrb/Pprot`：32-bit APB4 从接口。
- `Prdata/Pready/Pslverr`：APB4 返回通道。
- `core_id`：PE 编号，低 6 位参与 APB 地址译码。
- `computeout_done`：每次 ABFT 巡检结束时在 `clk` 域产生一个周期脉冲。
- `local_*`：保留原 `PeFaultToleranceAdapter` 的本地 cast/merge 网络接口，便于放回原系统。

ECC SRAM 当前使用 `Pclk`，因此 SRAM APB 命令本身没有额外的异步桥。计算 ABFT 状态通过原适配器中的 toggle/snapshot CDC 机制同步到 `Pclk` 域。

## APB 地址空间

每个 PE 的基地址为：

```text
PE_BASE = {PE_APB_PREFIX[10:0], core_id[5:0], 15'h0000}
```

默认 `PE_APB_PREFIX = 11'b1010_0000_000`。例如 `core_id=3` 时：

```text
PE_BASE   = 0xA001_8000
ABFT_BASE = 0xA001_E000
SRAM_BASE = 0xA001_F000
```

### 计算 ABFT 寄存器

| 相对 `PE_BASE` | 名称 | 访问 | 说明 |
|---:|---|:---:|---|
| `0x6000` | `PATROL_REPORT` | R | core、纠错状态、错误位置和巡检序号的压缩报告 |
| `0x6004` | `PATROL_CNT` | R | 已完成的 ABFT 巡检次数 |
| `0x6008` | `STATUS` | R | bit2 busy、bit1 corrected sticky、bit0 uncorrectable sticky |
| `0x600C` | `ABFT_CORR_CNT` | R | 已纠正计算错误数 |
| `0x6010` | `ABFT_UNCORR_CNT` | R | 不可纠正计算错误数 |
| `0x6014` | `ABFT_STATUS` | R | bit31 corrected、bit30 uncorrectable、row/column、低 16 位 syndrome |
| `0x6018` | `ABFT_LOCATION` | R | syndrome、row 和 column |
| `0x601C` | `ABFT_DIGEST` | R | 修复后矩阵摘要；当前为 `C[0][0]` |
| `0x6020` | `CONTROL` | R/W | bit0 注入一个计算错误并立即巡检；bit1 周期巡检；bit8 清状态 |
| `0x6024` | `PATROL_PERIOD` | R/W | 周期巡检间隔，写 0 自动转换为 1 |
| `0x6028` | `CORE_ID` | R | 当前 16-bit core ID |

`CONTROL.bit0` 与 `bit8` 是基于写事件产生的 toggle 命令；软件应通过一次寄存器写触发操作，不应依赖该位自动清零。

### ECC SRAM 寄存器

下表偏移相对 `SRAM_BASE = PE_BASE + 0x7000`。原 `Ecc_Sram_Apb_Wrapper.v` 的寄存器语义保持不变。

| 偏移 | 名称 | 访问 | 说明 |
|---:|---|:---:|---|
| `0x000` | `CONTROL` | R/W | bit0：使能后台 scrub |
| `0x004` | `STATUS` | R | bit0 RDATA_VALID、bit1 corrected、bit2 uncorrectable |
| `0x008` | `SRAM_ADDR` | R/W | SRAM 行地址，合法范围 0–11 |
| `0x00C` | `WRITE_MASK` | R/W | 16-bit 字节写掩码 |
| `0x010`–`0x01C` | `WDATA0`–`WDATA3` | R/W | 128-bit 写数据暂存，低字在 `WDATA0` |
| `0x020`–`0x02C` | `RDATA0`–`RDATA3` | R | 128-bit 读数据；无有效结果时返回 `PSLVERR` |
| `0x030` | `COMMAND` | W | bit0 READ_START；bit1 WRITE_START；命令访问会插入等待周期直到完成 |
| `0x034` | `ERR_STATUS` | R/W1C | bit0 corrected pending；bit1 uncorrectable pending |
| `0x038` | `ERR_ADDR` | R | 最近错误地址 |
| `0x03C` | `CORR_ERR_CNT` | R | 16-bit 可纠正错误计数 |
| `0x040` | `UNCORR_ERR_CNT` | R | 16-bit 不可纠正错误计数 |
| `0x044` | `ERR_CLEAR` | W | 写 bit0 清两个错误计数器，不清 pending 位 |
| `0x048` | `SCRUB_STATUS` | R/W1C | scrub active、wrap pending 和当前地址 |
| `0x04C` | `INJ_ADDR` | R/W | 故障注入目标行 |
| `0x050`–`0x060` | `INJ_MASK0`–`INJ_MASK4` | R/W | 152-bit 码字 XOR 注入掩码 |
| `0x064` | `INJ_COMMAND` | W | bit0 启动注入；访问会等待注入完成 |
| `0x068` | `INJ_STATUS` | R/W1C | bit0 injection-done pending |
| `0x06C` | `WDATA_VALID` | R | 写数据各字节是否已完成暂存 |

写 SRAM 的推荐软件流程：

1. 写 `SRAM_ADDR`。
2. 写 `WRITE_MASK`。
3. 写需要更新的 `WDATA0`–`WDATA3` 字节。
4. 向 `COMMAND` 写 `0x2`，并等待 APB `PREADY`。

读 SRAM 的推荐流程：

1. 写 `SRAM_ADDR`。
2. 向 `COMMAND` 写 `0x1`，并等待 APB `PREADY`。
3. 读 `STATUS`，再读 `RDATA0`–`RDATA3`。
4. 根据 `STATUS[2:1]`、`ERR_STATUS` 和错误计数器判断 ECC 结果。

## UVM 验证结构

`uvm/pe_ecc_ft_uvm_pkg.sv` 中包含：

- `pe_ft_apb_item`：APB transaction。
- `pe_ft_apb_driver`：支持 ECC 长命令 wait-state 的 APB master driver。
- `pe_ft_apb_monitor`：在 APB 接受边沿采样，并收集访问方向、地址窗口和错误响应覆盖率。
- `pe_ft_apb_agent`：active APB agent。
- `pe_ecc_ft_scoreboard`：检查顶层地址译码并统计 APB transaction。
- `pe_ecc_ft_base_seq`：提供寄存器读写、SRAM 128-bit 读写和 152-bit 故障注入 helper。
- `pe_ecc_ft_smoke_seq`：端到端容错测试。
- `pe_ecc_ft_test`：默认 UVM test。

默认 UVM sequence 覆盖：

1. ABFT/core ID 与顶层非法地址响应。
2. 计算结果单点加性故障注入、row/column 定位、syndrome=7 修复和结果摘要检查。
3. ECC SRAM 全字写入与无错误读取。
4. SRAM 单 bit 故障注入、纠正、返回原数据、计数和 pending 状态检查。
5. SRAM 三 bit 故障注入、不可纠正检测、计数和 pending 状态检查。
6. 测试结束前用相同 XOR mask 恢复被污染的 SRAM 行。

## 运行方法

当前机器安装了 Icarus Verilog，但没有安装 UVM class library。因此已实际运行 RTL 冒烟测试；UVM 文件需要在项目使用的 Questa、VCS 或 Xcelium 环境运行。

RTL 冒烟测试：

```bash
iverilog -g2012 -Wall -s tb_PeEccFtCore_smoke \
  -o /tmp/pe_ecc_ft_smoke.vvp -f pe_ecc_ft_smoke.f
vvp /tmp/pe_ecc_ft_smoke.vvp
```

VCS（使用自带 UVM 1.2）：

```bash
vcs -full64 -sverilog -ntb_opts uvm-1.2 \
  -f uvm/uvm_filelist.f -top tb_pe_ecc_ft_uvm -o simv
./simv +UVM_TESTNAME=pe_ecc_ft_test
```

Questa（`UVM_HOME` 指向 UVM 1.2 源码目录）：

```bash
vlib work
vlog -sv +incdir+$UVM_HOME/src $UVM_HOME/src/uvm_pkg.sv \
  -f uvm/uvm_filelist.f
vsim -c tb_pe_ecc_ft_uvm \
  +UVM_TESTNAME=pe_ecc_ft_test -do "run -all; quit -f"
```

Xcelium：

```bash
xrun -64bit -uvm -f uvm/uvm_filelist.f \
  -top tb_pe_ecc_ft_uvm +UVM_TESTNAME=pe_ecc_ft_test
```

## 文件清单

- `pe_ecc_ft_rtl.f`：仅新顶层所需 RTL。
- `pe_ecc_ft_smoke.f`：RTL 与非 UVM 冒烟测试。
- `uvm/uvm_filelist.f`：RTL、UVM interface/package 和 UVM top；UVM 库由仿真器命令提供。
- `tb_PeEccFtCore_smoke.sv`：无需 UVM 库的端到端 sanity test。

接入 SoC 时，系统 APB interconnect 只需把属于该 PE 的访问送到 `PeEccFtCore.Psel`；顶层会进一步把 `0x6000` 和 `0x7000` 两个子窗口分别送给计算 ABFT 与 ECC SRAM。其他被选中的地址会以 `PREADY=1, PSLVERR=1` 结束，方便软件和 UVM 快速发现映射错误。
