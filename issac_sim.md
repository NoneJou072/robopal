## CUDA INSTALL

### 信息查看
查看 Nidia 驱动信息：
```bash
cat /proc/driver/nvidia/version
```
```
NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.161.08  Tue Mar  5 22:42:15 UTC 2024
GCC version:  gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.2)
```

CUDA 版本对应-[https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

### 安装 CUDA

CUDA 下载网址-[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

添加环境变量
```bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```
```bash
source ~/.bashrc
```

### 卸载 CUDA

进入安装目录下的`bin`文件夹中
```bash
cd /usr/local/cuda-12.2/bin
```

勾选要卸载的版本，选择done

## Issac Sim 安装报错：

ERROR 信息：
```
PhysX error: Could not load libcuda.so: libcuda.so: cannot open shared object file: No such file or directory , 
FILE /buildAgent/work/eb2f45c4acc808a0/physx/source/physx/src/gpu/PxPhysXGpuModuleLoader.cpp, LINE 200
```

1. 检查 CUDA 是否正确安装
```bash
nvcc -V
```

2. 检查 CUDA 环境变量是否正确添加
```bash
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```

3. 尝试更换驱动版本
`Software & Updates -> Additional Drivers`


## 参考文章
1. [【保姆级教程】个人深度学习工作站配置指南](https://zhuanlan.zhihu.com/p/336429888)
2. [Ubuntu 20.04 安装 CUDA Toolkit 的三种方式](https://www.cnblogs.com/klchang/p/14353384.html)
3. [Isaac Gym安装及使用教程](https://zhuanlan.zhihu.com/p/618778210)
