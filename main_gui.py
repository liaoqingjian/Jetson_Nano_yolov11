import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from fall import FallDetection
from fire import FireDetection
from hat import HatDetection
from people import PeopleDetection

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("安全检测系统")
        
        # 设置初始窗口大小
        self.root.geometry("1280x720")
        
        # 创建标签页
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, expand=True, fill='both')
        
        # 创建四个标签页面
        self.fall_frame = ttk.Frame(self.notebook)
        self.fire_frame = ttk.Frame(self.notebook)
        self.hat_frame = ttk.Frame(self.notebook)
        self.people_frame = ttk.Frame(self.notebook)
        
        # 将页面添加到notebook
        self.notebook.add(self.fall_frame, text="人员倒地检测")
        self.notebook.add(self.fire_frame, text="烟火检测")
        self.notebook.add(self.hat_frame, text="安全帽检测")
        self.notebook.add(self.people_frame, text="区域入侵")
        
        # 初始化检测器
        self.fall_detector = FallDetection()
        self.fire_detector = FireDetection()
        self.hat_detector = HatDetection()
        self.people_detector = PeopleDetection()
        
        # 设置视频显示和控制按钮
        self.setup_fall_page()
        self.setup_fire_page()
        self.setup_hat_page()
        self.setup_people_page()
        
        # 视频状态标志
        self.is_fall_running = False
        self.is_fire_running = False
        self.is_hat_running = False
        self.is_people_running = False
        
        # 创建黑色背景图像
        self.create_black_background()
        
        # 绑定窗口大小改变事件
        self.root.bind('<Configure>', self.on_window_resize)

    def create_black_background(self):
        # 创建黑色背景图像
        self.black_background = np.zeros((480, 640, 3), dtype=np.uint8)
        self.display_black_background()

    def display_black_background(self):
        # 在所有视频标签上显示黑色背景
        black_img = Image.fromarray(self.black_background)
        black_photo = ImageTk.PhotoImage(image=black_img)
        
        self.fall_video_label.configure(image=black_photo)
        self.fall_video_label.image = black_photo
        
        self.fire_video_label.configure(image=black_photo)
        self.fire_video_label.image = black_photo
        
        self.hat_video_label.configure(image=black_photo)
        self.hat_video_label.image = black_photo
        
        self.people_video_label.configure(image=black_photo)
        self.people_video_label.image = black_photo

    def on_window_resize(self, event):
        # 获取当前窗口大小
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # 更新黑色背景尺寸
        self.black_background = np.zeros((window_height-100, window_width-50, 3), dtype=np.uint8)
        
        # 如果没有运行检测，则更新黑色背景显示
        if not self.is_fall_running:
            self.display_black_background()

    def setup_fall_page(self):
        # 视频显示区域
        self.fall_video_label = tk.Label(self.fall_frame)
        self.fall_video_label.pack(pady=10, expand=True, fill='both')
        
        # 控制按钮
        self.fall_start_btn = tk.Button(self.fall_frame, text="启动检测", command=self.toggle_fall_detection)
        self.fall_start_btn.pack(pady=5)

    def setup_fire_page(self):
        self.fire_video_label = tk.Label(self.fire_frame)
        self.fire_video_label.pack(pady=10, expand=True, fill='both')
        
        self.fire_start_btn = tk.Button(self.fire_frame, text="启动检测", command=self.toggle_fire_detection)
        self.fire_start_btn.pack(pady=5)

    def setup_hat_page(self):
        self.hat_video_label = tk.Label(self.hat_frame)
        self.hat_video_label.pack(pady=10, expand=True, fill='both')
        
        self.hat_start_btn = tk.Button(self.hat_frame, text="启动检测", command=self.toggle_hat_detection)
        self.hat_start_btn.pack(pady=5)

    def setup_people_page(self):
        self.people_video_label = tk.Label(self.people_frame)
        self.people_video_label.pack(pady=10, expand=True, fill='both')
        
        self.people_start_btn = tk.Button(self.people_frame, text="启动检测", command=self.toggle_people_detection)
        self.people_start_btn.pack(pady=5)

    def update_video(self, frame, video_label):
        if frame is not None:
            # 获取当前窗口大小
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            # 调整图像大小以适应窗口
            frame = cv2.resize(frame, (window_width-50, window_height-100))
            
            # 转换图像格式用于显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            video_label.configure(image=frame)
            video_label.image = frame

    def toggle_fall_detection(self):
        if not self.is_fall_running:
            self.is_fall_running = True
            self.fall_start_btn.configure(text="停止检测")
            self.process_fall_video()
        else:
            self.is_fall_running = False
            self.fall_start_btn.configure(text="启动检测")
            self.display_black_background()

    def toggle_fire_detection(self):
        if not self.is_fire_running:
            self.is_fire_running = True
            self.fire_start_btn.configure(text="停止检测")
            self.process_fire_video()
        else:
            self.is_fire_running = False
            self.fire_start_btn.configure(text="启动检测")
            self.display_black_background()

    def toggle_hat_detection(self):
        if not self.is_hat_running:
            self.is_hat_running = True
            self.hat_start_btn.configure(text="停止检测")
            self.process_hat_video()
        else:
            self.is_hat_running = False
            self.hat_start_btn.configure(text="启动检测")
            self.display_black_background()

    def toggle_people_detection(self):
        if not self.is_people_running:
            self.is_people_running = True
            self.people_start_btn.configure(text="停止检测")
            self.process_people_video()
        else:
            self.is_people_running = False
            self.people_start_btn.configure(text="启动检测")
            self.display_black_background()

    def process_fall_video(self):
        if self.is_fall_running:
            frame = self.fall_detector.process_frame()
            self.update_video(frame, self.fall_video_label)
            self.root.after(10, self.process_fall_video)

    def process_fire_video(self):
        if self.is_fire_running:
            frame = self.fire_detector.process_frame()
            self.update_video(frame, self.fire_video_label)
            self.root.after(10, self.process_fire_video)

    def process_hat_video(self):
        if self.is_hat_running:
            frame = self.hat_detector.process_frame()
            self.update_video(frame, self.hat_video_label)
            self.root.after(10, self.process_hat_video)

    def process_people_video(self):
        if self.is_people_running:
            frame = self.people_detector.process_frame()
            self.update_video(frame, self.people_video_label)
            self.root.after(10, self.process_people_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop() 