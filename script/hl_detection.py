import cv2
import re
from ultralytics import YOLO
import os
from tqdm import tqdm
import numpy as np

# ==========================================
# 0. 全局配置参数 (根据你的要求)
# ==========================================
CAM5_HOOP_ROI = (265, 585, 338, 648)
CAM3_HOOP_ROI = (1266, 568, 1333, 626)
ALPHA = 2.5       
BETA = 0         
MOTION_THRESHOLD = 4.0  # 预留参数

# ==========================================
# 1. 解析 SRT 标注文件
# ==========================================
def time_to_seconds(time_str):
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def parse_srt_events(srt_text):
    events = []
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n\d+\n|$)', re.DOTALL)
    matches = pattern.findall(srt_text)
    for match in matches:
        start_time = time_to_seconds(match[0])
        end_time = time_to_seconds(match[1])
        text = match[2].strip()
        text = re.sub(r'\\s*', '', text).replace('\n', '')
        if "投篮" in text or "三分球" in text or "罚球" in text:
            is_made = "(进)" in text
            events.append({
                'start': start_time, 'end': end_time, 'text': text,
                'is_made': is_made, 'detected': False
            })
    return events

# ==========================================
# 2. 单个摄像头的 YOLO 进球检测
# ==========================================
def detect_goals_single_cam(video_path, hoop_roi, model, cam_name, fps_stride=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 无法打开视频文件: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    detected_times = []
    ball_history = []
    frame_count = 0
    
    with tqdm(total=total_frames, desc=f"处理 {cam_name}", unit="帧") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            pbar.update(1)
            
            if frame_count % fps_stride != 0:
                continue
            
            # --- 图像增强 (根据 ALPHA, BETA) ---
            enhanced = cv2.convertScaleAbs(frame, alpha=ALPHA, beta=BETA)
            
            current_video_sec = frame_count / fps
            
            # --- YOLO 推理 (根据你的参数: conf=0.02, imgsz=1920) ---
            results = model(enhanced, classes=[32], conf=0.02, imgsz=1920, verbose=False) 
            
            ball_detected_in_hoop = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # 检查球是否在指定的 ROI 内
                    in_hoop = (hoop_roi[0] < cx < hoop_roi[2] and hoop_roi[1] < cy < hoop_roi[3])
                    
                    if in_hoop:
                        ball_history.append((current_video_sec, cy))
                        ball_detected_in_hoop = True
                        
                        # 判定下落逻辑
                        if len(ball_history) >= 3:
                            prev_y = ball_history[-3][1]
                            if cy - prev_y > 10: # 下落阈值
                                detected_times.append(current_video_sec)
                                ball_history.clear() 
                                # 冷却 3 秒
                                skip_frames = int(fps * 3)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + skip_frames)
                                frame_count += skip_frames
                                pbar.update(skip_frames)
                                break 
                            
            if not ball_detected_in_hoop and len(ball_history) > 0:
                if current_video_sec - ball_history[-1][0] > 1.0:
                    ball_history.clear()
                    
    cap.release()
    return detected_times

# ==========================================
# 3. 核心统筹与结果评估
# ==========================================
def evaluate_multi_cam_detections(srt_text, cam5_path, cam3_path):
    VIDEO_START_OFFSET_SEC = 1 * 3600 + 50 * 60 # 1h50m
    
    print("⏳ 正在加载 YOLOv8 模型...")
    model = YOLO('yolov8m.pt') 
    
    print("⏳ 正在解析字幕文件...")
    events = parse_srt_events(srt_text)
    
    # 分别检测 Cam 5 和 Cam 3
    cam5_times = detect_goals_single_cam(cam5_path, CAM5_HOOP_ROI, model, "CAM5")
    cam3_times = detect_goals_single_cam(cam3_path, CAM3_HOOP_ROI, model, "CAM3")
    
    combined_video_times = cam5_times + cam3_times
    absolute_detected_times = sorted([t + VIDEO_START_OFFSET_SEC for t in combined_video_times])
    
    print("\n" + "="*60)
    print("📊 Cam5 & Cam3 双机位进球检测报告")
    print("="*60)
    
    for event in events:
        window_start, window_end = event['start'] - 2.0, event['end'] + 2.0
        for detected_time in absolute_detected_times:
            if window_start <= detected_time <= window_end:
                event['detected'] = True
                break
                
        start_str = f"{int(event['start']//3600):02d}:{int((event['start']%3600)//60):02d}:{int(event['start']%60):02d}"
        
        if event['is_made']:
            status = "✅ 检测成功" if event['detected'] else "❌ 漏检"
            print(f"[{start_str}] {status} | 实际进球: {event['text']}")
        else:
            status = "⚠️ 误报" if event['detected'] else "✅ 正确忽略"
            print(f"[{start_str}] {status} | 实际未进: {event['text']}")

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    srt_file_path = os.path.abspath(os.path.join(base_path, "..", "data", "apidis", "camera_full_game_all_events.srt"))
    # 更新为 cam5 和 cam3 的视频路径
    video_cam5 = os.path.abspath(os.path.join(base_path, "..", "data", "apidis", "camera5_from_1h50m_to_end.mp4"))
    video_cam3 = os.path.abspath(os.path.join(base_path, "..", "data", "apidis", "camera3_from_1h50m_to_end.mp4"))

    print(f"📄 正在读取字幕文件: {srt_file_path}")
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as file:
            srt_content = file.read()
    except:
        with open(srt_file_path, 'r', encoding='gbk') as file:
            srt_content = file.read()

    evaluate_multi_cam_detections(srt_content, video_cam5, video_cam3)