import cv2
import re
from ultralytics import YOLO
import os
from tqdm import tqdm
import numpy as np

# ==========================================
# 0. 全局配置参数
# ==========================================
CAM5_HOOP_ROI = (265, 585, 338, 648)
CAM3_HOOP_ROI = (1266, 568, 1333, 626)
ALPHA = 2.5       
BETA = 0         

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
        text = re.sub(r'\s*', '', text).replace('\n', '')
        if "投篮" in text or "三分球" in text or "罚球" in text:
            is_made = "(进)" in text
            events.append({
                'start': start_time, 'end': end_time, 'text': text,
                'is_made': is_made, 'detected': False
            })
    return events

# ==========================================
# 2. 单个摄像头的背景去黑化 + YOLO 进球检测
# ==========================================
def detect_goals_single_cam(video_path, hoop_roi, model, cam_name, fps_stride=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 无法打开视频文件: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化全局背景减除器 (专门用于 YOLO 预处理)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = np.ones((3,3), np.uint8)
        
    detected_times = []
    # ball_history = []
    frame_count = 0
    cooldown_frames = 0  # 使用冷却计数器替代卡人的 cap.set()
    
    with tqdm(total=total_frames, desc=f"处理 {cam_name}", unit="帧") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            pbar.update(1)
            
            # --- 图像增强 ---
            enhanced = cv2.convertScaleAbs(frame, alpha=ALPHA, beta=BETA)
            
            # 冷却期间只更新背景模型，不进行推理
            if cooldown_frames > 0:
                cooldown_frames -= 1
                fgbg.apply(enhanced) 
                continue
            
            if frame_count % fps_stride != 0:
                # 即使抽帧跳过，也保持背景模型的连续性
                fgbg.apply(enhanced)
                continue
            
            current_video_sec = frame_count / fps

            if current_video_sec > 5 * 60:
                break
            
            # --- 背景去黑化处理 ---
            fgmask = fgbg.apply(enhanced)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            motion_only_frame = cv2.bitwise_and(enhanced, enhanced, mask=fgmask)
            
            # --- YOLO 推理 (新增 max_det=2 防止 NMS 卡死) ---
            results = model(motion_only_frame, classes=[32], conf=0.02, imgsz=1280, max_det=2, verbose=False) 
            
            # ball_detected_in_hoop = False
            # for result in results:
            #     for box in result.boxes:
            #         x1, y1, x2, y2 = map(int, box.xyxy[0])
            #         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
            #         # 检查球是否在指定的 ROI 内
            #         in_hoop = (hoop_roi[0] < cx < hoop_roi[2] and hoop_roi[1] < cy < hoop_roi[3])
                    
            #         if in_hoop:
            #             ball_history.append((current_video_sec, cy))
            #             ball_detected_in_hoop = True
                        
            #             # 判定下落逻辑
            #             if len(ball_history) >= 3:
            #                 prev_y = ball_history[-3][1]
            #                 if cy - prev_y > 10: # 下落阈值
            #                     detected_times.append(current_video_sec)
            #                     ball_history.clear() 
                                
                                # # 实时输出检测结果 (使用 tqdm.write 避免打断进度条)
                                # time_str = f"{int(current_video_sec//60):02d}:{int(current_video_sec%60):02d}"
                                # tqdm.write(f"🏀 [实时发现] {cam_name} 捕捉到进球下落动作！(视频时间 {time_str})")
                                
                                # # 触发 3 秒冷却
                                # cooldown_frames = int(fps * 3)
                                # break 
            ball_detected_in_hoop = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # 检查球是否在指定的 ROI 内
                    in_hoop = (hoop_roi[0] < cx < hoop_roi[2] and hoop_roi[1] < cy < hoop_roi[3])

                    if in_hoop:
                        ball_detected_in_hoop = True
                        detected_times.append(current_video_sec)

                        # 实时输出检测结果
                        time_str = f"{int(current_video_sec//60):02d}:{int(current_video_sec%60):02d}"
                        tqdm.write(f"🏀 [实时发现] {cam_name} 捕捉到进球动作！(视频时间 {time_str})")

                        # 触发 3 秒冷却，避免同一次进球重复多次
                        cooldown_frames = int(fps * 3)
                        break
                if ball_detected_in_hoop:
                    break
                            
            # if not ball_detected_in_hoop and len(ball_history) > 0:
            #     if current_video_sec - ball_history[-1][0] > 1.0:
            #         ball_history.clear()
                    
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
    # combined_video_times = cam5_times
    absolute_detected_times = sorted([t + VIDEO_START_OFFSET_SEC for t in combined_video_times])
    
    print("\n" + "="*60)
    print("📊 双机位进球检测报告 (背景去黑化版)")
    print("="*60)
    
    matched_detection_indices = set()
    
    # 第一步：事件比对
    for event in events:
        window_start, window_end = event['start'] - 2.0, event['end'] + 2.0
        for i, detected_time in enumerate(absolute_detected_times):
            if window_start <= detected_time <= window_end:
                event['detected'] = True
                matched_detection_indices.add(i)
                break
                
        start_str = f"{int(event['start']//3600):02d}:{int((event['start']%3600)//60):02d}:{int(event['start']%60):02d}"
        
        if event['is_made']:
            status = "✅ 检测成功" if event['detected'] else "❌ 漏检"
            print(f"[{start_str}] {status} | 实际进球: {event['text']}")
        else:
            status = "⚠️ 误报" if event['detected'] else "✅ 正确忽略"
            print(f"[{start_str}] {status} | 实际未进: {event['text']}")

    # 第二步：计算准确率指标
    TP = 0  # True Positives: 实际进球，且检测到了
    FN = 0  # False Negatives: 实际进球，但没检测到
    FP_event = 0 # False Positives (类型1): 实际没进，但在对应的动作时间点误报了进球
    TN_event = 0 # True Negatives: 🌟实际没进，代码也正确地没有报进球 (正确忽略)
    
    for event in events:
        if event['is_made']:
            if event['detected']: TP += 1
            else: FN += 1
        else:
            if event['detected']: FP_event += 1
            else: TN_event += 1  # 🌟 加上这一行统计 TN
            
    # False Positives (类型2): 凭空误报 (时间点外凭空报进球)
    FP_ghost = len(absolute_detected_times) - len(matched_detection_indices)
    total_FP = FP_event + FP_ghost
    total_actual_goals = TP + FN
    total_events = len(events) # 🌟 SRT里标注的所有投篮事件总数

    precision = (TP / (TP + total_FP)) * 100 if (TP + total_FP) > 0 else 0.0
    recall = (TP / total_actual_goals) * 100 if total_actual_goals > 0 else 0.0
    
    # 🌟 新增：计算综合准确率 (Accuracy)
    # 逻辑：(判断正确的进球 + 判断正确的未进) / (总投篮事件数 + 幽灵误报)
    accuracy = ((TP + TN_event) / (total_events + FP_ghost)) * 100 if (total_events + FP_ghost) > 0 else 0.0

    print("\n" + "="*60)
    print("📈 最终准确率统计 (Accuracy Metrics)")
    print("="*60)
    print(f"总计投篮事件数 (Total Events): {total_events}")
    print(f"实际进球数 (Ground Truth): {total_actual_goals}")
    print("-" * 60)
    print(f"有效检出进球 (TP): {TP}")
    print(f"正确忽略未进 (TN): {TN_event}")
    print(f"漏检进球 (FN): {FN}")
    print(f"事件内误报 (FP-Event): {FP_event} (没投进但报了进球)")
    print(f"凭空误报 (FP-Ghost): {FP_ghost} (时间点外凭空报进球)")
    print("-" * 60)
    print(f"🎯 查准率 (Precision): {precision:.2f}% (检测到的进球中，有多少是真的)")
    print(f"🔍 查全率 (Recall): {recall:.2f}% (实际的进球中，成功找出了多少)")
    print(f"🏆 综合准确率 (Accuracy): {accuracy:.2f}% (整体判断完全正确的比例)")
    print("="*60)
            
    # False Positives (类型2): 凭空误报 (代码检测到了进球，但那个时间点SRT里根本没有投篮动作)
    FP_ghost = len(absolute_detected_times) - len(matched_detection_indices)
    total_FP = FP_event + FP_ghost
    total_actual_goals = TP + FN

    precision = (TP / (TP + total_FP)) * 100 if (TP + total_FP) > 0 else 0.0
    recall = (TP / total_actual_goals) * 100 if total_actual_goals > 0 else 0.0

    print("\n" + "="*60)
    print("📈 最终准确率统计 (Accuracy Metrics)")
    print("="*60)
    print(f"总计实际进球数 (Ground Truth): {total_actual_goals}")
    print(f"有效检出数 (True Positives): {TP}")
    print(f"漏检数 (False Negatives): {FN}")
    print(f"事件内误报 (False Positives - Event): {FP_event} (没投进但报了进球)")
    print(f"凭空误报 (False Positives - Ghost): {FP_ghost} (时间点外凭空报进球)")
    print("-" * 60)
    print(f"🎯 查准率 (Precision): {precision:.2f}% (检测到的进球中，有多少是真的)")
    print(f"🔍 查全率 (Recall): {recall:.2f}% (实际的进球中，成功找出了多少)")
    print("="*60)

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    srt_file_path = os.path.abspath(os.path.join(base_path, "..", "data", "apidis", "camera_full_game_all_events.srt"))
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