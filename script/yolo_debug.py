import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import deque  # 导入用于保存轨迹的队列

# ==========================================
# 参数配置
# ==========================================
START_SEC = 85  
END_SEC = 95    

# CAM5_HOOP_ROI = (265, 585, 338, 648)
# OUTPUT_VIDEO = "debug_cam5_hybrid_trajectory.mp4"

CAM3_HOOP_ROI = (1266, 568, 1333, 626)
OUTPUT_VIDEO = "debug_cam3_hybrid_trajectory.mp4"


ALPHA = 1.0       
BETA = 0         
MOTION_THRESHOLD = 100.0  

# 轨迹设置
trajectory = deque(maxlen=30) # 保存最近30帧的球心位置

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    # video_path = os.path.abspath(os.path.join(base_path, "..", "data", "apidis", "camera5_from_1h50m_to_end.mp4"))
    video_path = os.path.abspath(os.path.join(base_path, "..", "data", "apidis", "camera3_from_1h50m_to_end.mp4"))


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 找不到视频文件: {video_path}")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, START_SEC * 1000)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("⏳ 正在初始化 YOLOv8m (imgsz=1280) 与背景减除器...")
    model = YOLO('yolov8m.pt')
    
    # 1. 篮筐局部动作检测（保留您原有的高灵敏度配置）
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)
    
    # 2. 🌟 新增：全屏背景去黑化（针对 YOLO 预处理，调高 history 使背景更稳定）
    fgbg_full = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel_full = np.ones((3,3), np.uint8)

    total_frames = int((END_SEC - START_SEC) * fps)

    with tqdm(total=total_frames, desc="处理中") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) > END_SEC:
                break
            
            pbar.update(1)

            # --- A. 图像增强 ---
            enhanced = cv2.convertScaleAbs(frame, alpha=ALPHA, beta=BETA)

            # --- B. 运动检测 (ROI 区域，判定进球网花) ---
            # rx1, ry1, rx2, ry2 = CAM5_HOOP_ROI
            rx1, ry1, rx2, ry2 = CAM3_HOOP_ROI

            roi_zone = enhanced[ry1:ry2, rx1:rx2]
            fgmask = fgbg.apply(roi_zone)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            motion_percent = (np.sum(fgmask == 255) / fgmask.size) * 100

            # --- 🌟 新增：B2. 全局背景去除 (专门喂给 YOLO 的纯净画面) ---
            fgmask_full = fgbg_full.apply(enhanced)
            # 开运算：过滤掉观众席闪烁之类的小噪点
            fgmask_full = cv2.morphologyEx(fgmask_full, cv2.MORPH_OPEN, kernel_full)
            # 按位与：静止背景直接变成纯黑 (0,0,0)，只有在动的物体保留原始彩色像素
            motion_only_frame = cv2.bitwise_and(enhanced, enhanced, mask=fgmask_full)

            # --- C. YOLO 检测 ---
            # 🌟 注意：这里传入的是去除了背景的 motion_only_frame，而不是 enhanced
            results = model(motion_only_frame, classes=[32], conf=0.02, imgsz=1280, max_det=2, verbose=False)

            # --- D. 可视化绘制 (仍在 enhanced 原图上画，保证输出视频观感正常) ---
            # 1. 画出蓝框 ROI
            roi_color = (0, 255, 0) if motion_percent > MOTION_THRESHOLD else (255, 255, 0)
            cv2.rectangle(enhanced, (rx1, ry1), (rx2, ry2), roi_color, 2)
            
            # 2. 处理 YOLO 框与轨迹点
            ball_this_frame = False
            for result in results:
                for box in result.boxes:
                    ball_this_frame = True
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                    
                    # 记录中心点用于画轨迹
                    trajectory.append((bcx, bcy))
                    
                    # 画当前帧的红框
                    cv2.rectangle(enhanced, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.circle(enhanced, (bcx, bcy), 5, (0, 0, 255), -1)

            # 3. 绘制轨迹线 (黄色)
            for i in range(1, len(trajectory)):
                if trajectory[i-1] is None or trajectory[i] is None:
                    continue
                cv2.line(enhanced, trajectory[i-1], trajectory[i], (0, 255, 255), 3)

            # 4. 实时数据显示面板
            cv2.putText(enhanced, f"Motion: {motion_percent:.1f}%", (rx1, ry1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
            
            if motion_percent > MOTION_THRESHOLD:
                cv2.putText(enhanced, "TRIGGER!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            # 提示当前 YOLO 状态
            status_color = (0, 255, 255) if ball_this_frame else (100, 100, 100)
            cv2.putText(enhanced, "YOLO Tracking", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # 5. 右上角“动作掩码”小窗
            mask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            mask_res = cv2.resize(mask_bgr, (200, 150))
            enhanced[20:170, width-220:width-20] = mask_res
            cv2.putText(enhanced, "ROI Motion View", (width-220, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            # 🌟 可选：左上角再加一个全局动目标去黑化后的小窗，方便您直观看到喂给 YOLO 的到底是什么图
            motion_res = cv2.resize(motion_only_frame, (200, 150))
            enhanced[20:170, 20:220] = motion_res
            cv2.putText(enhanced, "YOLO Input (Blacked)", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            out.write(enhanced)

    cap.release()
    out.release()
    print(f"\n✅ 轨迹+动作混合调试视频已生成: {os.path.abspath(OUTPUT_VIDEO)}")