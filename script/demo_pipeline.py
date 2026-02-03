import cv2
import torch
import numpy as np
import json
import librosa
import os
from models import MultiFrameResNet, SimpleAudioCNN

# --- 配置参数 ---
MODEL_DIR = "/home/SONY/s7000043396/Downloads/demo/model"
VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, "video_model_best.pth")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_model_best.pth")
VIDEO_PATH = "/home/SONY/s7000043396/Downloads/demo/video/Virginia Tech vs. Syracuse Condensed Game ｜ 2025-26 ACC Men's Basketball [E4Wza6tUOWY].webm"
OUTPUT_DIR = "/home/SONY/s7000043396/Downloads/demo/output"
OUTPUT_FILENAME = "highlight_result.json"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# 采样设置
FPS_TARGET = 10         # 降采样到 10 FPS
WINDOW_SEC = 5          # 窗口长度 5 秒
STRIDE_SEC = 2.5        # 步长 2.5 秒 (重叠 50%)
FRAMES_PER_WINDOW = FPS_TARGET * WINDOW_SEC
IMG_SIZE = 224          # ResNet 输入尺寸

# 类别定义 (对应模型的输出索引)
CLASSES = ["Background", "Goal", "Dunk", "Fastbreak", "Celebration"]

def preprocess_frame(frame):
    """转灰度 -> Resize -> 归一化"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized

def extract_audio_segment(y, sr, start_sec, duration_sec):
    """截取对应时间的音频并转为梅尔频谱图"""
    # 计算样本索引
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    
    # 填充处理 (如果越界)
    if end_sample > len(y):
        audio_chunk = np.pad(y[start_sample:], (0, end_sample - len(y)))
    else:
        audio_chunk = y[start_sample:end_sample]
        
    # 生成 Mel Spectrogram
    mels = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    # Resize 到 CNN 需要的尺寸 (224x224)
    mels_resized = cv2.resize(mels_db, (IMG_SIZE, IMG_SIZE))
    # 归一化
    mels_norm = (mels_resized - mels_resized.min()) / (mels_resized.max() - mels_resized.min() + 1e-6)
    return mels_norm

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating Output Directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)


    print(f"Loading Models...")
    # 实例化模型 (实际使用时这里应该加载 .pth 权重文件)
    # model_video = MultiFrameResNet(num_input_frames=FRAMES_PER_WINDOW, num_classes=len(CLASSES))
    # model_video.load_state_dict(torch.load("video_best.pth")) 
    
    # 这里用未训练的模型演示 pipeline
    model_video = MultiFrameResNet(num_input_frames=FRAMES_PER_WINDOW, num_classes=len(CLASSES))
    model_audio = SimpleAudioCNN(num_classes=len(CLASSES))
    
    model_video.eval()
    model_audio.eval()

    print(f"Processing Video: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print("Error: Video file not found!")
        return

    # 1. 加载音频 (一次性加载，快)
    print("Loading Audio stream (this may take a moment)...")
    y, sr = librosa.load(VIDEO_PATH, sr=16000) # 降低采样率加速
    
    # 2. 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames_original / original_fps
    
    frame_interval = int(original_fps / FPS_TARGET)
    
    buffer_frames = []
    events = []
    
    current_sec = 0.0
    frame_count = 0
    
    print("Starting Inference Loop...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 降采样：只处理特定间隔的帧
        if frame_count % frame_interval == 0:
            proc_frame = preprocess_frame(frame)
            buffer_frames.append(proc_frame)
            
            # 当缓冲区满 (攒够5秒/50帧)
            if len(buffer_frames) == FRAMES_PER_WINDOW:
                # --- 准备输入数据 ---
                # Video Input: [1, 50, 224, 224]
                video_tensor = torch.FloatTensor(np.array(buffer_frames)).unsqueeze(0)
                
                # Audio Input: [1, 1, 224, 224]
                audio_img = extract_audio_segment(y, sr, current_sec, WINDOW_SEC)
                audio_tensor = torch.FloatTensor(audio_img).unsqueeze(0).unsqueeze(0)
                
                # --- 模型推理 ---
                with torch.no_grad():
                    vid_probs = model_video(video_tensor) # Tensor like [[0.1, 0.8...]]
                    aud_probs = model_audio(audio_tensor)
                
                # --- 简单的融合逻辑 (Average) ---
                final_probs = (vid_probs + aud_probs) / 2.0
                max_score, pred_idx = torch.max(final_probs, 1)
                pred_label = CLASSES[pred_idx.item()]
                
                # --- 记录结果 (模拟：只记录置信度 > 0.4 且非背景的) ---
                # 注意：因为没训练，现在结果是随机的，为了演示JSON生成，我们不做严格过滤
                if pred_label != "Background": 
                    event = {
                        "id": len(events) + 1,
                        "timestamp_start": f"{current_sec:.2f}",
                        "timestamp_end": f"{current_sec + WINDOW_SEC:.2f}",
                        "primary_label": pred_label,
                        "confidence": float(f"{max_score.item():.2f}"),
                        "source_analysis": {
                            "video_score": float(f"{vid_probs[0][pred_idx].item():.2f}"),
                            "audio_score": float(f"{aud_probs[0][pred_idx].item():.2f}")
                        }
                    }
                    events.append(event)
                    print(f"[{current_sec:.1f}s] Detected: {pred_label} (Conf: {max_score.item():.2f})")

                # 滑动窗口：移除前 stride 帧 (步长移动)
                stride_frames = int(STRIDE_SEC * FPS_TARGET)
                buffer_frames = buffer_frames[stride_frames:]
                current_sec += STRIDE_SEC
                
        frame_count += 1

    cap.release()
    
    # 3. 写入 JSON
    output_data = {
        "video_path": VIDEO_PATH,
        "total_duration": duration,
        "model_info": "ResNet18_Grayscale_Stacked + CNN_Audio_Mel",
        "events": events
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Done! Result saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()