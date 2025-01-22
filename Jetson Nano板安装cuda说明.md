

在 Jetson Nano 上已经安装好 Ubuntu 系统的情况下，您可以通过命令行直接安装 JetPack 相关的软件包。以下是详细步骤：

------

### **步骤 1: 更新系统**

确保系统软件包是最新的：

```bash
sudo apt update
sudo apt upgrade -y
```

------

### **步骤 2: 添加 NVIDIA 软件源**

Jetson Nano 的 JetPack SDK 通过 NVIDIA 提供的官方软件源安装。

1. 确保设备已经联网（通过以太网或 Wi-Fi）。

2. 更新 APT 源并添加 NVIDIA 软件源：

   ```bash
   sudo apt-add-repository universe
   sudo apt-add-repository multiverse
   sudo apt-add-repository restricted
   sudo apt update
   ```

------

### **步骤 3: 安装 JetPack SDK**

直接安装 `nvidia-jetpack` 包，JetPack 会自动安装 CUDA、cuDNN、TensorRT 等组件：

```bash
sudo apt install nvidia-jetpack
```

------

### **配置环境变量**

### **找到 CUDA 的安装路径**

JetPack 通常将 CUDA 安装在 `/usr/local/cuda`，可以通过以下命令确认：

```bash
ls /usr/local/
```

如果有 `cuda` 或类似 `cuda-xx.x` 的目录（如 `cuda-11.4`），说明 CUDA 已安装。

将 CUDA 的 `bin` 路径添加到系统的 `PATH` 中：

```bash
echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```



### **步骤 4: 验证安装**

1. 检查 CUDA 是否安装成功：

   ```bash
   nvcc --version
   ```

   如果正确安装，应该显示 CUDA 版本信息。

2. 测试 TensorRT 是否正常工作：

   ```bash
   dpkg -l | grep nvidia
   ```

   确保 `nvidia-cuda`, `nvidia-tensorrt` 等组件已列出。

------

### **步骤 5: 测试 GPU 加速**

运行以下命令测试 GPU 的计算能力：

1. **运行 CUDA 示例程序**： NVIDIA JetPack 提供了一些示例程序用于测试：

   ```bash
   cd /usr/local/cuda/samples/1_Utilities/deviceQuery
   sudo make
   ./deviceQuery
   ```

   输出中应显示 GPU 的详细信息，如 CUDA 核心数量、计算能力等。

2. **测试 TensorRT 推理**： JetPack 包含 TensorRT 的测试工具，可以运行以下命令测试推理功能：

   ```bash
   /usr/src/tensorrt/bin/trtexec --help
   ```

------

### **额外步骤**

#### 安装 Python 开发环境

如果需要在 Python 中开发 AI 应用，可以安装以下包：

```bash
sudo apt install python3-pip
pip3 install numpy scipy matplotlib opencv-python tensorflow torch torchvision
```

#### 安装 OpenCV

虽然 JetPack 中自带 OpenCV，但可能需要自己编译以支持更多功能：

```bash
sudo apt install libopencv-dev python3-opencv
```

------



sudo tegrastats







### **使用 PyTorch 检测 GPU**

1. **安装 PyTorch**： JetPack 通常默认安装了 PyTorch。如果没有，可以通过以下命令安装：

   ```
   bash
   
   
   复制代码
   pip3 install torch torchvision
   ```

2. **运行测试代码**： 创建文件 `gpu_test.py`，内容如下：

   ```
   python复制代码import torch
   
   if torch.cuda.is_available():
       print(torch.cuda.is_available())
       print("Device Name:", torch.cuda.get_device_name(0))
   else:
       print("CUDA is not available.")
   ```

   执行：

   ```
   bash
   
   
   复制代码
   python3 gpu_test.py
   ```

   如果显示 `CUDA is available!` 且打印了 GPU 的设备名称，说明 PyTorch 能正常使用 GPU。

### **注意事项**

1. **JetPack 版本与硬件支持**： 确保安装的 JetPack 版本与 Jetson Nano 硬件版本兼容。可以参考 NVIDIA 官方的 JetPack 兼容表：[JetPack Documentation](https://developer.nvidia.com/embedded/jetpack).
2. **存储需求**： JetPack SDK 安装需要较大的存储空间。如果空间不足，可以使用 USB 或扩展存储。
3. **网络连接**： 下载和安装 JetPack SDK 需要稳定的网络连接。



在 Jetson Nano 上编译 `torchvision` 是一个常见的做法，特别是因为某些预构建的二进制文件可能不完全适配 Jetson 的 ARM 架构和 CUDA 环境。你可以通过以下步骤自行编译 `torchvision`。

### 前提条件

确保已经安装了 PyTorch，并且能够正常使用 GPU。如果你还没有安装 PyTorch，可以参考官方文档或通过 `pip` 安装：

```bash

pip install torch
```

然后，安装一些必需的依赖项，使用以下命令：

```
bash复制代码sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev python3-pip libopenblas-dev libomp-dev libopencv-dev
```

### 1. 克隆 `torchvision` 源代码

首先，你需要从 GitHub 克隆 `torchvision` 的源代码：

```bash

git clone --recursive https://github.com/pytorch/vision
git clone --branch release/0.15 https://github.com/pytorch/vision.git
cd vision
```

### 2. 设置编译环境

`torchvision` 使用 `setup.py` 文件来构建。你需要确保环境中配置了 CUDA 和 PyTorch。在 Jetson Nano 上，默认的 CUDA 和 PyTorch 配置应该已经设置好。

如果你安装了不同版本的 PyTorch 或 CUDA，你可以查看 `torchvision` 兼容的版本并调整安装命令。

### 3. 编译 `torchvision`

运行以下命令来安装 `torchvision`：

```bash

sudo python -E setup.py install
```

这个命令会开始编译 `torchvision`，并且会使用系统中已安装的 CUDA 来加速编译。整个过程可能会需要一段时间，具体取决于你的设备性能。

### 4. 验证安装

安装完成后，验证是否成功安装：

```
bash


复制代码
python3 -c "import torchvision; print(torchvision.__version__)"
```

这将打印出已安装的 `torchvision` 版本。如果没有报错，说明安装成功。

### 5. 注意事项

- **CUDA 版本**：在 Jetson Nano 上，CUDA 版本通常与 PyTorch 和 `torchvision` 版本紧密相关。确保你安装的 PyTorch 和 `torchvision` 版本支持当前的 CUDA 版本。
- **编译过程中的错误**：如果你在编译过程中遇到问题，可以根据错误信息调整依赖项或编译选项。常见的问题包括缺少某些库（如 OpenCV），需要根据错误信息安装缺失的依赖项。

### 6. 额外优化

由于 Jetson Nano 的硬件资源有限，你可以通过以下方式优化编译过程：

- **减少并行度**：如果你遇到内存问题，可以减少并行编译的线程数。例如，使用 `make -j2` 来限制并行编译的线程数。
- **交叉编译**：如果 Jetson Nano 的编译速度较慢，可以考虑在更强大的机器上交叉编译 `torchvision`，然后将编译好的包传输到 Jetson Nano 上。

这样，你就能够在 Jetson Nano 上手动编译和安装 `torchvision`，以便充分利用 GPU 加速。
