import cv2
import os

def select_hoop_roi(video_path, cam_name):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法读取视频: {video_path}，请检查路径。")
        return

    # --- 新增：定位到第 3 秒 ---
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frame_id = int(fps * 3)         # 计算第 3 秒对应的帧索引
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id) # 跳转到该帧
    
    # 读取跳转后的那一帧
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ 视频长度不足 3 秒或无法读取该帧。")
        cap.release()
        return
        
    print(f"\n👉 正在打开 {cam_name} 第 3 秒的画面...")
    print("操作说明：")
    print("1. 用鼠标左键在画面中拖拽画一个矩形框。")
    print("2. 按【空格】或【回车】确认。")
    print("3. 按【C】取消。")

    window_title = f"Select ROI for {cam_name} (at 3s)"
    roi = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    cap.release() # 记得释放资源
    
    x, y, w, h = roi
    if w > 0 and h > 0:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        print(f"✅ {cam_name} 的 ROI 坐标获取成功！")
        print(f"({x1}, {y1}, {x2}, {y2})\n")
    else:
        print(f"⚠️ 未选择有效区域。\n")

if __name__ == "__main__":
    base_dir = ".." 
    video_cam1 = os.path.join(base_dir, "data", "apidis", "camera3_from_1h50m_to_end.mp4")
    select_hoop_roi(video_cam1, "CAM5")