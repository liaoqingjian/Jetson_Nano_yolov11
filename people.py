from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
def gstreamer_pipeline(
    sensor_id=1,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
class PeopleDetection:
    def __init__(self):
        # 加载模型
        self.model = YOLO("model/people.pt")
        self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        self.frame_count = 0
        self.start_time = time.time()
        
        # 获取视频尺寸
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置监控区域
        self.setup_region()

    def setup_region(self):
        center_x, center_y = self.width // 2, self.height // 2
        region_width, region_height = 200, 200
        self.region_x1 = center_x - region_width // 2
        self.region_y1 = center_y - region_height // 2
        self.region_x2 = center_x + region_width // 2
        self.region_y2 = center_y + region_height // 2

    def get_class_color(self, class_id):
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

    def is_in_region(self, x1, y1, x2, y2):
        return not (x2 < self.region_x1 or x1 > self.region_x2 or 
                   y2 < self.region_y1 or y1 > self.region_y2)

    def put_chinese_text(self, img, text, position, font_size, color):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("msyh.ttc", font_size)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # 使用模型对当前帧进行检测
        results = self.model(frame, conf=0.5)

        # 提取目标信息
        for result in results[0].boxes:
            # 提取边界框坐标
            box = result.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # 提取类别和置信度
            class_id = int(result.cls[0])
            confidence = float(result.conf[0])
            class_name = self.model.names[class_id]

            if class_name == "person":
                color = self.get_class_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if self.is_in_region(x1, y1, x2, y2):
                    frame = self.put_chinese_text(frame, "区域入侵!", (10, 50), 40, (0, 0, 255))
                    cv2.rectangle(frame, (self.region_x1, self.region_y1),
                                (self.region_x2, self.region_y2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (self.region_x1, self.region_y1),
                                (self.region_x2, self.region_y2), (0, 255, 0), 2)

        # 计算并显示帧率
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps_display = self.frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    def __del__(self):
        self.cap.release()
