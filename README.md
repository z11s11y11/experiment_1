成功在本地编译运行的flashgpu sim源码，后面的修改可以基于此baseline

首次拉下本仓库需要执行的命令如下：

## 1. 进入仓库并设置 CUDA 环境

本机的 `/usr/local/cuda` 当前指向 CUDA 12.8。如果你在另一台机器运行，请把 `CUDA_INSTALL_PATH` 改为该机器的 CUDA Toolkit 根目录。

```bash
cd /home/zhangshuyi/MyExperiment/FlashGPU-Sim-Baseline/FlashGPU-Sim
export CUDA_INSTALL_PATH=/usr/local/cuda
"$CUDA_INSTALL_PATH/bin/nvcc" --version
source setup_environment
```

确认 `nvcc` 显示的版本至少为 12.8，并且环境脚本输出 `setup_environment succeeded`。以后打开新的终端编译或运行时，需要重新执行本节的 `export` 和 `source` 命令。

## 2. 编译模拟器

仍在仓库根目录执行：

```bash
make -j "$(nproc)"
test -f "lib/$GPGPUSIM_CONFIG/libcudart.so" && echo "模拟器编译产物存在"
```

本仓库是为了验证改变GPU的HBM数量带来的影响

## 修改位置

H100 的 L2 配置位于 [`configs/SM90_H100/gpgpusim.config` 第 166 行](configs/SM90_H100/gpgpusim.config#L166)：

```text
-gpgpu_cache:dl2 S:128:128:20,L:B:m:L:P,A:512:96,32:0,64
```

以仓库中的 [`configs/SM90_H100/gpgpusim.config`](configs/SM90_H100/gpgpusim.config) 为准，仅将第 26 行 `-gpgpu_n_mem 80` 设为 160，模拟器就会创建 **160 个内存分区／控制器**。

一个HBM对应16个分区

批量扫描HBM参数可以用
```
./sweep_gemm_n_mem.sh
```