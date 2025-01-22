import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置要使用的 GPU 编号

import torch

print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

import torch
print(torch.__version__)  # 显示 PyTorch 版本
print(torch.cuda.is_available())  # 检查是否支持 CUDA
print(torch.version.cuda)  # 显示 PyTorch 编译时支持的 CUDA 版本
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#https://download.pytorch.org/whl/torch/