# FlashGPU Sim 中 HBM 数量配置的位置

## 结论

模拟器用 `-gpgpu_n_mem` 设置**内存通道／内存控制器（memory partition）数量**。对于使用 HBM 的 GPU 配置，应在对应的 `gpgpusim.config` 中查找这个参数。它不是“HBM 物理堆叠（stack）数量”的独立配置项；本仓库没有发现专门设置 HBM stack 数量的参数。

当前 H100 配置位于 [`configs/SM90_H100/gpgpusim.config`](configs/SM90_H100/gpgpusim.config#L26)：第 26 行 `-gpgpu_n_mem 80`，即模拟 **80 个内存通道／分区**。第 27 行 `-gpgpu_n_sub_partition_per_mchannel 2` 设置每个通道的子分区数，因此这里共有 **160 个内存子分区**。

## HBM 相关配置文件

| 配置 | 通道数设置位置 | `-gpgpu_n_mem` 值 | 备注 |
| --- | --- | ---: | --- |
| H100 | [`configs/SM90_H100/gpgpusim.config:26`](configs/SM90_H100/gpgpusim.config#L26) | 80 | 当前顶层配置；第 27 行每通道 2 个子分区 |
| TITAN V（tested） | [`configs/tested-cfgs/SM7_TITANV/gpgpusim.config:34`](configs/tested-cfgs/SM7_TITANV/gpgpusim.config#L34) | 24 | [第 171 行](configs/tested-cfgs/SM7_TITANV/gpgpusim.config#L171)注释写明 3 个 HBM stack、24 个通道 |
| GV100（tested） | [`configs/tested-cfgs/SM7_GV100/gpgpusim.config:66`](configs/tested-cfgs/SM7_GV100/gpgpusim.config#L66) | 32 | [第 201 行](configs/tested-cfgs/SM7_GV100/gpgpusim.config#L201)的注释仍写“3 stacks, 24 channels”，与实际参数不一致 |
| QV100（tested） | [`configs/tested-cfgs/SM7_QV100/gpgpusim.config:66`](configs/tested-cfgs/SM7_QV100/gpgpusim.config#L66) | 32 | [第 201 行](configs/tested-cfgs/SM7_QV100/gpgpusim.config#L201)的注释仍写“3 stacks, 24 channels”，与实际参数不一致 |
| P100（deprecated） | [`configs/deprecated-cfgs/SM6_P100/gpgpusim.config:17`](configs/deprecated-cfgs/SM6_P100/gpgpusim.config#L17) | 32 | [第 115 行](configs/deprecated-cfgs/SM6_P100/gpgpusim.config#L115)标注 HBM 为 32 个通道 |
| TITAN V（deprecated） | [`configs/deprecated-cfgs/SM7_TITANV/gpgpusim.config:23`](configs/deprecated-cfgs/SM7_TITANV/gpgpusim.config#L23) | 24 | [第 124 行](configs/deprecated-cfgs/SM7_TITANV/gpgpusim.config#L124)注释写 32 个通道，与实际参数不一致 |

表中以实际 `-gpgpu_n_mem` 参数为准；注释里的 stack 数不能直接当成模拟器中的独立配置值。

## 参数在源码中的位置

- [`src/gpgpu-sim/gpu-sim.cc:282`](src/gpgpu-sim/gpu-sim.cc#L282)：注册 `-gpgpu_n_mem`，写入变量 `m_n_mem`；源码描述为 GPU 中的 memory modules（例如 memory controllers）。
- [`src/gpgpu-sim/gpu-sim.h:312`](src/gpgpu-sim/gpu-sim.h#L312)：以 `m_n_mem × m_n_sub_partition_per_memory_channel` 计算内存子分区总数。
- [`src/gpgpu-sim/gpu-sim.cc:1273`](src/gpgpu-sim/gpu-sim.cc#L1273)：按 `m_n_mem` 创建内存分区单元，说明它控制模拟器实例化的分区数量。
- [`configs/README.md:92`](configs/README.md#L92)：配置文档将 `-gpgpu_n_mem` 定义为 memory channels 的数量。

与通道数量相邻的 `-gpgpu_n_mem_per_ctrlr` 表示**每个内存控制器的内存芯片数**（见 [`src/gpgpu-sim/gpu-sim.cc:288`](src/gpgpu-sim/gpu-sim.cc#L288)），不表示 HBM stack 数量。
