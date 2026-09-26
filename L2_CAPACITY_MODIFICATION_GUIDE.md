# FlashGPU Sim：L2 容量修改方法

## 修改位置

H100 的 L2 配置位于 [`configs/SM90_H100/gpgpusim.config` 第 166 行](configs/SM90_H100/gpgpusim.config#L166)：

```text
-gpgpu_cache:dl2 S:128:128:20,L:B:m:L:P,A:512:96,32:0,64
```

实际仿真读取的是工作负载运行目录中的 `gpgpusim.config`，见 [`configs/README.md`](configs/README.md#L3)。因此，后续应用修改时应确认修改的是本次运行实际加载的配置；仅修改仓库模板不会自动更新已经复制出去的配置。

## 容量字段与公式

配置开头格式为 `S:<集合数>:<缓存行字节数>:<路数>`，解析代码见 [`src/gpgpu-sim/gpu-cache.h:574`](src/gpgpu-sim/gpu-cache.h#L574)。当前各字段如下：

| 字段 | 当前值 | 含义 |
| --- | ---: | --- |
| `S` | - | Sector cache 类型，不是容量数值 |
| `nsets` | 128 | 每个 L2 子分区的集合数 |
| `bsize` | 128 | 每条缓存行的字节数 |
| `assoc` | 20 | 每个集合的路数 |

```text
每个 L2 子分区容量 = nsets × bsize × assoc
L2 子分区数量 = gpgpu_n_mem × gpgpu_n_sub_partition_per_mchannel
总 L2 容量 = 上述两项的乘积
```

依据：[`gpu-cache.h:831`](src/gpgpu-sim/gpu-cache.h#L831)按集合数、行大小和路数计算容量；[`gpu-sim.h:312`](src/gpgpu-sim/gpu-sim.h#L312)计算子分区数量；[`l2cache.cc:589`](src/gpgpu-sim/l2cache.cc#L589)为每个子分区创建 L2 cache。

当前 H100 为 `80 × 2 = 160` 个子分区，每片 `128 × 128 × 20 = 327680 B = 320 KiB`，总计 **50 MiB**。这里计算的是缓存数据容量，不含 tag 等元数据。

## 160 个通道时保持总 L2 为 50 MiB

假设后续将 `-gpgpu_n_mem` 设为 160，并保持 `-gpgpu_n_sub_partition_per_mchannel 2`，则共有 320 个子分区。要保持总容量，每片应缩小至 160 KiB。

一种直接的方法是将第 166 行的路数从 **20 改为 10**：

```text
-gpgpu_cache:dl2 S:128:128:10,L:B:m:L:P,A:512:96,32:0,64
```

此时 `160 × 2 × 128 × 128 × 10 = 52428800 B = 50 MiB`。集合数、缓存行大小以及配置字符串后面的策略、MSHR、队列和数据端口宽度字段均保持原值。注意：路数减半也会改变冲突行为；总容量相同不代表缓存性能相同。

如果希望保持 20 路相联，也可将集合数从 **128 改为 64**：

```text
-gpgpu_cache:dl2 S:64:128:20,L:B:m:L:P,A:512:96,32:0,64
```

同样得到 50 MiB，但集合索引和地址冲突行为会变化。当前使用的 `P`（IPOLY）集合索引支持 64 个集合，见 [`gpu-cache.cc:126`](src/gpgpu-sim/gpu-cache.cc#L126)及 [`hashing.cc:69`](src/gpgpu-sim/hashing.cc#L69)。不要通过改变缓存行大小来单纯调容量，那还会改变请求和缓存行组织。

| 通道数 | 每通道子分区 | 每片配置 | 总 L2 |
| ---: | ---: | --- | ---: |
| 80 | 2 | `S:128:128:20` | 50 MiB |
| 160 | 2 | `S:128:128:20` | 100 MiB |
| 160 | 2 | `S:128:128:10` | 50 MiB |
| 160 | 2 | `S:64:128:20` | 50 MiB |

## 方法的边界

这些调整只控制每片及总 L2 的容量，不能保持原来的 160 个 L2 片数量：当前模型中，L2 片数与内存子分区数量绑定。320 个片仍会带来更多独立缓存、MSHR 和队列实例，所以“总容量保持 50 MiB”不等于“整个 L2 组织完全保持不变”。

不需要修改 C++ 源码中的默认字符串；参数注册位置 [`gpu-sim.cc:273`](src/gpgpu-sim/gpu-sim.cc#L273)只是读取配置并提供缺省值。也不要用 `-gpgpu_l2_partition_count` 调容量，它控制的是粗粒度区域划分。

本文件仅记录源码阅读结果和待应用的配置方法，未修改源码或现有配置，也未运行上述配置的仿真实验。
