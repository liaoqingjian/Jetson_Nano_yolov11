from ultralytics import YOLO
 
# 加载你的模型
model = YOLO("hat.pt")
 
# 导出模型
model.export(format="onnx")  # creates 'yolov8n.onnx'
 
# 模型推理
onnx_model = YOLO("hat.onnx")
 
# 选择你的图片、视频
results = onnx_model("hat.mp4")
