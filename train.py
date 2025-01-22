from ultralytics import YOLO
import torch

# 检查是否有可用的GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 配置训练参数
train_params = {
    "data": "dataset1/coco.yaml",  # 数据集配置文件路径
    "epochs": 100,                # 训练的轮数
    "imgsz": 640,                 # 输入图像大小
    "batch": 8,                   # 批量大小
    "device": device,             # 使用的设备
    "save_period": 50,            # 每隔多少轮保存一次权重
    "project": "runs/train",      # 训练日志保存的路径
    "name": "exp1",               # 当前实验的名称
    "resume": False               # 是否从上次中断处继续训练
}

# 开始训练
print("Starting training...")
model.train(**train_params)

# 训练完成后进行验证
print("Training complete. Starting validation...")
metrics = model.val()  # 执行验证

# 提取并输出评估指标
precision = metrics.box.map50   # Precision at IoU=0.5
recall = metrics.box.map50_95  # Recall at IoU=0.5:0.95
f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # 防止除以零

print(f"Validation metrics:")
print(f"  Precision (mAP@0.5): {precision:.4f}")
print(f"  Recall (mAP@0.5:0.95): {recall:.4f}")
print(f"  F1 Score: {f1_score:.4f}")
