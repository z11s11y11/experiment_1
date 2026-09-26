# FlashGPU Sim：H100 从 80 增至 160 个 HBM 通道的影响

## 结论

以仓库中的 [`configs/SM90_H100/gpgpusim.config`](configs/SM90_H100/gpgpusim.config) 为准，仅将第 26 行 `-gpgpu_n_mem 80` 设为 160，模拟器就会创建 **160 个内存分区／控制器**。第 27 行保持每通道 2 个子分区时，L2／内存子分区与片上互连的内存端点会从 **160 增至 320**。地址映射代码也会使用新的数量，不需要为“让新增通道收到请求”手工指定 160 个通道 ID。

但这**不等于**模拟器有一个显式的“10 个 HBM stack”模型。`5 × 16 = 80` 和 `10 × 16 = 160` 可以作为目标硬件的换算假设；本配置及其源码只以通道／分区为单位，并未定义每 16 个通道属于同一个 stack，也未自动体现新增 stack 的物理布线、功耗或距离。

| 你的判断 | 阅读源码后的结论 |
| --- | --- |
| 80 改 160，Memory controller 数量翻倍 | **成立**。`m_n_mem` 决定 `memory_partition_unit` 的创建次数，每个分区持有 DRAM 模型；见 [`gpu-sim.cc:1270`](src/gpgpu-sim/gpu-sim.cc#L1270)及[`l2cache.cc:226`](src/gpgpu-sim/l2cache.cc#L226)。 |
| HBM 峰值存取带宽翻倍 | **作为 DRAM 端理论峰值成立**：若每通道总线宽度、时钟、时序及可用性保持不变，独立通道数翻倍，合计理论吞吐上限翻倍。实际 GPU 程序带宽还受 SM 请求速率、L2 命中、NoC、冲突及地址分布限制，不能保证翻倍。单通道参数见[配置第 31、208–215 行](configs/SM90_H100/gpgpusim.config#L208)。 |
| HBM 容量自动翻倍 | **在当前模拟器中不成立**。通道数变化没有联动显存容量参数：`cudaDeviceProp.totalGlobalMem` 被写死为 2 GiB；`cudaMemGetInfo` 是固定返回值占位实现；`gpu_malloc` 只递增地址，未按 `m_n_mem` 检查容量。见[`cuda_runtime_api.cc:228`](libcuda/cuda_runtime_api.cc#L228)、[`cuda_runtime_api.cc:3116`](libcuda/cuda_runtime_api.cc#L3116)和[`cuda-sim.cc:761`](src/cuda-sim/cuda-sim.cc#L761)。因此不能用这个改动来声称容量已在软件接口或容量限制中翻倍。 |

## NoC：连通性会自动扩展，性能模型需要重新评估

H100 使用内置 local crossbar（[配置第 184 行](configs/SM90_H100/gpgpusim.config#L184)），不是目录里另存的 `config_ampere_islip.icnt` 网络文件。初始化时，模拟器把 `m_n_mem_sub_partition` 传给 `icnt_create`（[`gpu-sim.cc:1285`](src/gpgpu-sim/gpu-sim.cc#L1285)）；crossbar 用传入的内存端点数创建输入、输出缓冲和仲裁状态（[`local_interconnect.cc:234`](src/gpgpu-sim/local_interconnect.cc#L234)）。因此**保持当前 `-network_mode 2` 时，为连接 320 个端点不必同步修改 NoC 的节点数配置**。

但 NoC 的[`输入／输出缓冲、子网、仲裁设置`](configs/SM90_H100/gpgpusim.config#L184)不会因为通道翻倍而自动重新标定。SM 端仍为 132 个集群（[配置第 24 行](configs/SM90_H100/gpgpusim.config#L24)）；新增输出端口也不能使注入请求速率自动翻倍。若研究目标是“实际可达到的 GPU 带宽”，应检查互连拥塞与请求／回复吞吐，并根据目标硬件重估这些参数。若改用 `network_mode 1` 的 Intersim 网络，拓扑文件及端点映射另需核对；这个条件**不适用于当前 H100 配置**。

## 片上缓存：L2 数量和总容量会自动翻倍

内存配置按 `m_n_mem × m_n_sub_partition_per_memory_channel` 计算子分区总数（[`gpu-sim.h:312`](src/gpgpu-sim/gpu-sim.h#L312)），而每个子分区在启用 L2 时各建一个 L2 cache（[`l2cache.cc:570`](src/gpgpu-sim/l2cache.cc#L570)）。H100 目前每片是 `128 sets × 128 B × 20 ways = 320 KiB`（[配置第 163–166 行](configs/SM90_H100/gpgpusim.config#L163)）：原本 `160 × 320 KiB = 50 MiB`；改为 160 个通道后，将变为 `320 × 320 KiB = 100 MiB`。

所以**是否同步修改 L2 参数取决于目标**：若只想把 HBM 通道／控制器翻倍、保持 H100 的 50 MiB 总 L2，就需要相应缩小每片容量；若也希望 L2 容量翻倍，当前参数可保持。此处只是模型推算，没有修改配置。每 SM 的 L1／共享内存配置不会因 HBM 通道数变化而自动改变，也没有因新增 HBM 通道就必须修改它们的代码约束。

H100 的两组粗粒度 L2 区域由 `-gpgpu_l2_partition_count 2` 定义（[配置第 197 行](configs/SM90_H100/gpgpusim.config#L197)）。区域归属按当前**总子分区数**计算（[`l2cache.cc:915`](src/gpgpu-sim/l2cache.cc#L915)），因此 320 个子分区仍可自动分成两组；不过这仍是原 H100 的两区域延迟模型，不会自动变成反映 10 个 HBM stack 布局的新拓扑。

## 地址交织：会映射到新数量，但均衡度不能直接保证

地址映射初始化接收通道数和每通道子分区数（[`gpu-sim.h:316`](src/gpgpu-sim/gpu-sim.h#L316)）。H100 配置使用 `dramid@9`、`-gpgpu_memory_partition_indexing 2`（IPOLY）、`-gpgpu_ipoly_non_power2_balanced 2`（[配置第 169、212、269 行](configs/SM90_H100/gpgpusim.config#L169)）。因为 **160 不是 2 的幂**，解码器继续走非 2 的幂通道路径：先用 `addr >> 9` 对当前 `m_n_channel` 取模（[`addrdec.cc:127`](src/gpgpu-sim/addrdec.cc#L127)），再由 IPOLY 在固定 1024 个虚拟子分区中散列，并按当前 `m_n_channel × m_n_sub_partition_in_channel` 缩减到 **320 个目标**（[`addrdec.cc:190`](src/gpgpu-sim/addrdec.cc#L190)）。所以**地址交织会自动覆盖新的目标编号范围**；无需另写一个 160 通道的地址表。

需要留意两点。第一，这个 IPOLY 路径会改变地址到通道／L2 片的对应关系；增加通道不会保证原有数据在相同通道，更不代表它模拟了“每 16 通道一个 stack”的分组。第二，1024 到 320 的整数范围缩减以及特定工作负载的访问步长，都可能影响分布；源码能证明映射可生成 320 个目标，**不能证明任意负载都均匀交织或带宽翻倍**。配置末尾“最终 160 个目标”的[注释](configs/SM90_H100/gpgpusim.config#L265)描述的是旧数值；实际目标数由运行时参数计算。

## 建模建议

若目标是一个“只增加 HBM stack、保持其他片上资源不变”的对照模型，通道数改为 160 后，重点应同步核对 **L2 总容量是否要保持 50 MiB**，并检查 NoC 和 L2 延迟／吞吐是否仍代表目标硬件。地址路由及当前内置 NoC 的端点数会自动扩展；显存容量及 CUDA 查询值不会自动扩展。上述判断来自静态代码阅读，未运行修改后的配置或带宽实验。
