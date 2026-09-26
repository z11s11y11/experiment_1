# 160 个 HBM 通道、50 MiB L2 能否运行官方矩阵乘示例？

## 判断

**从内存配置来看，可以作为矩阵乘仿真的候选配置；但仅修改这两个值，不能直接保证官方示例会使用它们并运行成功。**

本文将“官方矩阵乘示例”理解为仓库 [`RUN_GEMM.md`](RUN_GEMM.md) 中的 [`tutorials/triton-gemm/`](tutorials/triton-gemm/)。该示例默认使用 RTX 5090 配置，而前面讨论的 80 通道基线来自 H100。必须区分默认运行方式与选择自定义 H100 模型的运行方式。

## 两处配置本身是否合理？

若以 H100 配置为基础，保留其他字段，并采用下列参数：

```text
-gpgpu_n_mem 160
-gpgpu_n_sub_partition_per_mchannel 2
-gpgpu_cache:dl2 S:128:128:10,L:B:m:L:P,A:512:96,32:0,64
```

则会得到 160 个内存控制器、320 个 L2／内存子分区，总 L2 数据容量为 `160 × 2 × 128 × 128 × 10 = 50 MiB`。`S:128:128:10` 只是配置的前半段，实际使用时应保留逗号之后的策略和队列字段。

代码中没有看到这组数量要求修改矩阵乘内核的约束：L2 路数按整数解析，缓存行数按集合数乘路数计算（[`gpu-cache.h:574`](src/gpgpu-sim/gpu-cache.h#L574)、[`gpu-cache.h:780`](src/gpgpu-sim/gpu-cache.h#L780)）；子分区数量按参数计算（[`gpu-sim.h:312`](src/gpgpu-sim/gpu-sim.h#L312)）；当前 H100 的非 2 的幂 IPOLY 映射使用计算后的目标数量（[`addrdec.cc:190`](src/gpgpu-sim/addrdec.cc#L190)）；内置 NoC 使用该数量创建端点（[`gpu-sim.cc:1286`](src/gpgpu-sim/gpu-sim.cc#L1286)）。所以这两处改动本身没有发现明确的初始化障碍。

## 官方脚本默认不会读取修改后的 H100 配置

[`tutorials/triton-gemm/run.sh:11`](tutorials/triton-gemm/run.sh#L11)使用：

```bash
CONFIG_NAME="${PERF_SIM_CONFIG:-SM120_RTX5090}"
```

因此直接执行 `./run.sh` 时，若未设置 `PERF_SIM_CONFIG`，依然选择 `SM120_RTX5090`。修改 `configs/SM90_H100/gpgpusim.config` 不会改变这个默认选择。

脚本第 40 行会将所选配置目录复制到实际运行目录（[`run.sh:40`](tutorials/triton-gemm/run.sh#L40)）。后续若已将 H100 配置按上述方法调整，可通过现有环境变量选择它，无需修改脚本：

```bash
cd tutorials/triton-gemm
PERF_SIM_CONFIG=SM90_H100 ./run.sh
```

也可以用 `PERF_SIM_CONFIG` 指定 `configs/` 下另存的自定义配置目录名。这是待执行的方法说明，本次没有修改配置或运行此命令。不要只修改 `run/tracking/launchers/gpgpusim.config` 后再运行脚本，因为脚本会重新复制配置，覆盖同名文件。

## 还存在示例架构匹配问题

随附回放明确以 `sm_120a` 为目标，见 [`trace/README.md`](tutorials/triton-gemm/trace/README.md#L6)及 [`kernel_tma_gemm_launch1_kernel.ptx:6`](tutorials/triton-gemm/trace/launchers/kernel_tma_gemm_launch1_kernel.ptx#L6)。其 Makefile 根据 PTX 的 `.target` 选择编译和打包架构，而不会因 `PERF_SIM_CONFIG=SM90_H100` 自动重新生成 H100 内核（[`launch1_Makefile`](tutorials/triton-gemm/trace/launchers/kernel_tma_gemm_launch1_Makefile)）。

本次检查可见内核使用 TMA、TensorMap 和 `mma.sync.m16n8k16`，不能仅凭 `.target sm_120a` 就断言模拟器在 H100 配置下必然失败；但也不能把未经验证的跨架构回放当成已经确认能运行的 H100 示例。若研究的是 H100 扩展 HBM 的性能，最好使用匹配 H100 的录制内核和输入；使用现有随附内核时，应明确这是用 H100 性能配置回放 RTX 5090 目标内核的实验。

## 运行成功与性能提升是两件事

运行还需要已编译好的模拟器、正确的模拟器 CUDA 库和支持示例编译架构的 CUDA Toolkit。脚本会检查模拟器库、编译 launcher 并验证库路径（[`run.sh:54`](tutorials/triton-gemm/run.sh#L54)、[`run.sh:61`](tutorials/triton-gemm/run.sh#L61)、[`run.sh:69`](tutorials/triton-gemm/run.sh#L69)）。已有输入和参考输出可用于离线回放，不需要物理 GPU。

最终是否跑通应以日志中同时出现 `Validation PASSED` 和 `gpu_tot_sim_cycle` 为准，这也是官方脚本的检查条件（[`run.sh:87`](tutorials/triton-gemm/run.sh#L87)）。本文是静态代码判断，**没有执行修改后的配置，因此没有声称它已经通过验证**。

此外，默认示例只有 40 个 CTA（[`trace/README.md`](tutorials/triton-gemm/trace/README.md#L7)），无法仅凭这一个用例判断 160 个 HBM 通道是否充分利用；总 L2 容量保持 50 MiB 时，L2 片数仍从 160 增至 320，路数也从 20 降至 10，性能变化包含缓存组织变化，不能全部归因于 HBM 带宽增加。

**简要结论：两项修改在当前 H100 内存模型中是合理的；运行官方示例还必须选择实际使用的配置，并验证随附 `sm_120a` 内核的回放。默认 `./run.sh` 不会使用你修改的 H100 配置，也不能保证性能翻倍。**
