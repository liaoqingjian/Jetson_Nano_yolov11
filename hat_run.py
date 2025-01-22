from ultralytics import YOLO
import cv2
import time
import numpy as np

# 加载模型
model = YOLO("model/hat.pt")  # 替换为你的模型路径

# 打开视频文件（或者使用摄像头：cv2.VideoCapture(0)）
video_path = "hat.mp4"  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和大小
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 输出视频设置（保存检测结果）
out = cv2.VideoWriter(
    "output_hat.mp4",  # 替换为你的输出视频路径
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

# 用于生成不同类别的颜色（通过哈希值）
def get_class_color(class_id):
    np.random.seed(class_id)  # 保证每个类别的颜色一致
    return tuple(np.random.randint(0, 255, 3).tolist())

# 视频帧循环
frame_count = 0
start_time = time.time()  # 记录开始时间
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("视频读取完毕或发生错误")
        break

    # 使用模型对当前帧进行检测
    results = model(frame, conf=0.5)  # 可调整 conf 参数以更改置信度阈值

    # 提取目标信息
    for result in results[0].boxes:
        # 提取边界框坐标
        box = result.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # 提取类别和置信度
        class_id = int(result.cls[0])
        confidence = float(result.conf[0])
        class_name = model.names[class_id]  # 获取类别名称

        # 获取类别颜色
        color = get_class_color(class_id)

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制类别和置信度
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # 计算并显示帧率
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps_display = frame_count / elapsed_time
    cv2.putText(
        frame,
        f"FPS: {fps_display:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 显示检测结果
    cv2.imshow("YOLO Detection", frame)

    # 保存检测结果到输出视频
    out.write(frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

