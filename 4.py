import cv2

def main():
    # 打开RTSP流
    rtsp_url = "rtsp://localhost:8554/stream"
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("无法连接到 RTSP 流。")
        return

    while True:
        # 从视频流读取帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取视频帧。")
            break
        
        # 显示视频帧
        cv2.imshow('RTSP Stream', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
