# FlashGPU-Sim 矩阵乘示例的 tile 大小和 HBM 排布

## 范围与结论

这里分析的是仓库官方示例 `tutorials/triton-gemm/gemm.py`，以及 `tutorials/triton-gemm/run.sh` 回放的**仓库附带 trace**。附带 trace 的矩阵规模为 `M=2560, N=64, K=2560`，输入和输出都是 FP16。它选中的 Triton 配置为 `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64`，`GROUP_SIZE_M=8`、`num_stages=4`、`num_warps=4`；启动配置是 40 个 CTA，每个 CTA 128 个线程。源码列有其他 autotune 候选配置，因此“64×64×64”特指这份已录制、可直接回放的 trace，并非每次重新 autotune 都固定如此。依据：[附带 trace 说明](tutorials/triton-gemm/trace/README.md#L3-L15)、[autotune 候选](tutorials/triton-gemm/gemm.py#L51-L89)、[回放入口](tutorials/triton-gemm/run.sh#L23-L34)、[实际启动参数](tutorials/triton-gemm/trace/launchers/kernel_tma_gemm_launch1_harness.cu#L454-L460)。

每个 CTA 负责一个 `C` 的 **64×64 输出 tile**。每次 K 循环读取一个 `A` 的 **64×64 tile** 和一个已预转置 `B` 的 **64×64 tile**，计算 `A_tile @ B_tile.T`，累计进 FP32 的 64×64 累加器；最后转为 FP16，写回 `C`。每个矩阵 tile 含 4096 个 FP16 元素，即 **8192 字节（8 KiB）**。一次 K 迭代的两块输入合计 16 KiB；`K/64=40` 次迭代形成一个完整输出 tile。这里的 CTA tile 与底层 `mma.sync.aligned.m16n8k16` **指令形状**不是同一个概念。依据：[描述符、偏移和计算循环](tutorials/triton-gemm/gemm.py#L119-L163)、[PTX MMA 指令](tutorials/triton-gemm/trace/launchers/kernel_tma_gemm_launch1_kernel.ptx#L528-L531)。

## HBM 中的逻辑排布

下面的地址是以各张矩阵的设备指针为基址的**全局线性字节地址**。`a`、`b` 都要求 contiguous；源码把原始 `(K,N)` 的 `b_original` 转置并复制成 contiguous 的 `(N,K)`，`c` 新建为 `(M,N)`。TMA 描述符对三张矩阵分别使用 `shape=[M,K] / [N,K] / [M,N]` 和 `strides=[K,1] / [K,1] / [N,1]`。因此它们在全局内存中都是各自存储形状下的行主序，并没有把每个 tile 预先打包成一整块。依据：[数据创建](tutorials/triton-gemm/gemm.py#L270-L285)、[描述符](tutorials/triton-gemm/gemm.py#L119-L139)、[连续性检查及输出创建](tutorials/triton-gemm/gemm.py#L179-L189)。

设输出 tile 的起点是 `m0=pid_m×64`、`n0=pid_n×64`，第 `t` 次 K 迭代的起点是 `k0=t×64`；`i,j` 均在 `[0,63]`。FP16 元素占 2 字节。令 `A0`、`B0`、`C0` 为三个不同的设备基址：

| tile 内元素 | 全局内存地址 | 相邻行起点距离 | tile 在 HBM 中的样子 |
| --- | --- | ---: | --- |
| `A_tile[i,j]=A[m0+i,k0+j]` | `A0 + 2×((m0+i)×2560 + k0+j)` | 5120 B | 每行 64 个连续 FP16，即 128 B；64 行分布在 64 个相距 5120 B 的区段 |
| `B_tile[i,j]=B[n0+i,k0+j]` | `B0 + 2×((n0+i)×2560 + k0+j)` | 5120 B | 同样是 64 个相距 5120 B 的 128 B 区段；这里的 `i` 是输出的 N 坐标 |
| `C_tile[i,j]=C[m0+i,n0+j]` | `C0 + 2×((m0+i)×64 + n0+j)` | 128 B | 每行 128 B，行与行紧邻；整个 64×64 tile 连续占 8192 B |

对附带 trace，`N=64`，所以 `pid_n` 只有 0；`M/64=40`，故共有 40 个输出 CTA。每个 CTA 沿 K 方向走 40 次。固定一行 A 或 B 时，随着 `t` 从 0 到 39，40 段 128 B 正好拼成该行完整的 5120 B。以 `m0=n0=k0=0` 为例，A、B tile 的第 0 行是相应矩阵基址起的 `[0,127]` 字节，第 1 行是 `[5120,5247]` 字节；C tile 的第 0 行是 `[0,127]`，第 1 行紧接着是 `[128,255]`。这些示例偏移均相对于**各自**的矩阵基址，不能跨 A/B/C 比较绝对位置。依据：[tile 偏移与加载/写回](tutorials/triton-gemm/gemm.py#L141-L163)、[录制参数及输入大小](tutorials/triton-gemm/trace/launchers/kernel_tma_gemm_launch1_harness.cu#L422-L436)。

TMA 在全局内存侧按描述符的全局行跨度逐行形成请求：模拟器在最内维取连续元素，在外层维加 `globalStrides`，再把连续区段切分为至多 128 B、按 128 B 边界对齐的请求。对于本例、在基址满足相应对齐的通常情况下，一个输入 tile 可看作每行一个 128 B 请求；`C` 的逻辑地址仍是连续的。**请求粒度不是 tile 的存储格式**。依据：[TMA 遍历与地址计算](src/gpgpu-sim/flash/tma.cc#L2035-L2129)、[请求切分](src/gpgpu-sim/flash/tma.cc#L2002-L2033)。

## HBM 通道与 shared memory 的区别

回放默认使用 `SM120_RTX5090` 配置；配置给出 **16 个内存通道、每通道 8 个子分区**，并启用 `gpgpu_memory_partition_indexing=2`（IPOLY）。地址映射还有 `dramid@8;...` 规则。模拟器先从地址解码通道、bank 等字段，再用从地址第 8 位起的信息做 IPOLY 子分区散列，最终以 `sub_partition/8` 得到通道。因而上述 A、B 的 64 个行区段虽然有规则的**线性地址间距**，不能据此声称它们固定落在某一个 HBM 通道、按 CTA 或按 tile 独占某个通道；具体通道还取决于设备基址和地址映射。依据：[运行配置](tutorials/triton-gemm/run.sh#L11-L12)、[HBM/分区配置](configs/SM120_RTX5090/gpgpusim.config#L23-L27)、[映射配置](configs/SM120_RTX5090/gpgpusim.config#L151-L158)、[映射规则](configs/SM120_RTX5090/gpgpusim.config#L190-L196)、[IPOLY 实现](src/gpgpu-sim/addrdec.cc#L149-L224)。

录制 PTX 的三个 TMA 描述符均设置 `swizzle_mode=3`，对应 `TMA_SWIZZLE_128B`。在模拟器实现里，TMA 从全局内存按线性地址读出，然后把数据按 swizzle 写入 **shared memory**；写回时反向读取 shared memory，再写到原来的全局线性地址。因此 **128 B swizzle 描述的是片上 shared memory 的 tile 排布，不是 A、B、C 在 HBM 中的行主序被重排**。依据：[PTX 描述符设置](tutorials/triton-gemm/trace/launchers/kernel_tma_gemm_launch1_kernel.ptx#L131-L140)、[swizzle 枚举](src/gpgpu-sim/flash/tensormap.h#L28-L38)、[TMA 读写路径](src/gpgpu-sim/flash/tma.cc#L2249-L2318)。

## 例子：假设有 6 个 HBM，首个 A tile 分到哪里？

**先约定“6 个 HBM”的含义。** 模拟器的参数是内存**通道**数，并没有 HBM 堆叠对象。按仓库现有实验采用的换算假设——每个 HBM 堆叠对应 16 个通道——这里令 H100 风格配置的 `-gpgpu_n_mem=96`、每通道 2 个子分区，把通道 `0–15` 记作 HBM 0，`16–31` 记作 HBM 1，依此类推直到 `80–95` 为 HBM 5。这只是给通道分组的**示意命名**；当前 H100 配置文件实际启用的是 208 个通道，默认 GEMM 回放的 RTX 5090 配置则是 16 个通道。依据：[仓库的 16 通道/HBM 假设](HBM_80_TO_160_IMPACT_ANALYSIS.md#L3-L7)、[H100 当前通道及子分区数](configs/SM90_H100/gpgpusim.config#L23-L29)、[默认回放配置](tutorials/triton-gemm/run.sh#L11-L12)。

取 `m0=0, k0=0` 的第一个 `A` 输入 tile，**假设**它的设备基址 `A0=0xC00000000`（模拟器动态显存堆的起始地址；实际运行中若有其他分配，`A0` 要以运行时指针为准）。该 tile 第 `i` 行的 128 B 数据位于 `A0+5120i` 到 `A0+5120i+127`，`i=0…63`。它不是先切成六份再连续放进六个 HBM，而是每一行的全局地址经地址解码和 IPOLY 散列后选出通道。依据：[动态显存起始地址](src/abstract_hardware_model.h#L344)、[动态分配起点](src/abstract_hardware_model.cc#L184-L191)、[A 的 TMA 描述符](tutorials/triton-gemm/gemm.py#L119-L125)。

本例沿用 H100 配置的 `dramid@9`、`gpgpu_memory_partition_indexing=2`、`gpgpu_ipoly_non_power2_balanced=2`。对一个请求的起始字节地址 `a`，源码可概括为：先取 `q=floor(a/512)`、`decoded_channel=q mod 96`、`h=floor(q/96)`；从解码后的 bank 取低位，与通道组成初始子分区 `2×decoded_channel+(bank&1)`；计算 `v=ipoly_hash(h, 初始子分区, 1024)`；再令 `sub_partition=floor(v×192/1024)`、`channel=floor(sub_partition/2)`、`HBM=floor(channel/16)`。这是**本假设配置**的规则，不是简单的 `行号 mod 6`。依据：[H100 地址/散列配置](configs/SM90_H100/gpgpusim.config#L210-L216)、[非 2 的幂通道配置](configs/SM90_H100/gpgpusim.config#L269-L274)、[地址拆分、IPOLY 与范围缩减](src/gpgpu-sim/addrdec.cc#L128-L224)、[IPOLY(1024) 公式](src/gpgpu-sim/hashing.cc#L161-L188)。

| A tile 行号 | 本行起始地址 | 映射的通道 | 归入的 HBM |
| ---: | --- | ---: | ---: |
| 0 | `0xC00000000` | 25 | 1 |
| 1 | `0xC00001400` | 24 | 1 |
| 2 | `0xC00002800` | 29 | 1 |
| 3 | `0xC00003C00` | 27 | 1 |
| 4 | `0xC00005000` | 30 | 1 |
| 5 | `0xC00006400` | 34 | 2 |
| 6 | `0xC00007800` | 34 | 2 |
| 7 | `0xC00008C00` | 38 | 2 |

例如第 0 行：`a=0xC00000000` 给出 `q=0x6000000`、`decoded_channel=0`、`h=0x100000`、`bank&1=0`；IPOLY 算得 `v=277`，于是 `sub_partition=floor(277×192/1024)=51`、`channel=floor(51/2)=25`，按上述 16 通道分组落在 HBM 1。

按同样规则算完 64 行，得到这块 8 KiB tile 在六组通道上的分布：

| HBM（假设的堆叠编号） | 通道编号范围 | 本 tile 的行数 | 对应数据量 |
| ---: | --- | ---: | ---: |
| 0 | 0–15 | 0 | 0 B |
| 1 | 16–31 | 30 | 3840 B |
| 2 | 32–47 | 34 | 4352 B |
| 3 | 48–63 | 0 | 0 B |
| 4 | 64–79 | 0 | 0 B |
| 5 | 80–95 | 0 | 0 B |

例如 HBM 1 收到第 `0–4`、`10–12`、`14` 等行；HBM 2 收到第 `5–9`、`13`、`15–19` 等行。**有 6 个 HBM 不意味着单个 tile 必须均分到 6 个 HBM。** 上表也不能直接当成实际 HBM 流量：L1/L2 命中、跨 CTA 的 B 重用等会影响最终到 DRAM 的请求。换一个 `A0`、tile 坐标、通道数或散列配置，表中的分布都可能变化。

如果只是为了理解一种“人为轮流放置”的简化模型，可以把 64 行依次分给 `HBM(i mod 6)`：HBM 0–3 各 11 行（1408 B），HBM 4–5 各 10 行（1280 B）。**这只是教学示意，并不是本仓库的 IPOLY 地址映射结果。**
