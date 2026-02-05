import numpy as np
import json
import os
import cv2
import torch
from PIL import Image
from datetime import timedelta

# ==========================================
# 1. 模型加载区 (根据硬件选择)
# ==========================================

# --- [Mac M3 Pro 方案] SigLIP (Google SOTA, 优于 CLIP) ---
from transformers import AutoProcessor, AutoModel
MODEL_ID = "google/siglip-so400m-patch14-384"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 正在使用 Mac 加速设备: {device}")

# 加载模型
model = AutoModel.from_pretrained(MODEL_ID).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- [CUDA / Linux 方案] InternVideo2.5 (仅供参考，取消注释使用) ---
# """
# import torch
# from transformers import AutoModel, AutoProcessor
# MODEL_ID = "OpenGVLab/InternVideo2-5-1B"
# device = "cuda"
# model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
# # 注意: InternVideo 需要特定的预处理函数，通常需 clone 官方 repo
# """

# ==========================================
# 2. 配置路径
# ==========================================
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
MOSAIC_VIDEO = os.path.join(BASE_DIR, "mosaic_preview_720p.mp4")
FEATURE_FILE = os.path.join(BASE_DIR, "Mosaic_preview_features.npy")
METADATA_FILE = os.path.join(BASE_DIR, "feature_metadata.json")

def extract_features():
    print(f"--- 开始提取特征 (Model: {MODEL_ID}) ---")
    
    cap = cv2.VideoCapture(MOSAIC_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    
    features_list = []
    metadata = []
    
    print(f"视频总时长: {duration} 秒. 采样率: 1 fps")

    for t in range(duration):
        # 精准跳转到第 t 秒
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret: break
        
        # BGR -> RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # ==========================================
        # 推理逻辑 (Mac Active)
        # ==========================================
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            # SigLIP 提取图像特征
            image_features = model.get_image_features(**inputs)
            # 归一化 (便于后续计算余弦相似度)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        features_list.append(image_features.cpu().numpy())

        # ==========================================
        # 推理逻辑 (CUDA Option - Commented)
        # ==========================================
        # inputs = preprocess_internvideo(frame).to(device)
        # with torch.no_grad():
        #     feat = model.encode_video(inputs)
        #     features_list.append(feat.cpu().numpy())

        # 构建元数据
        metadata.append({
            "timestamp_sec": t,
            "ltc": f"14:00:{t:02d}:00", # 模拟 2026 工业级 LTC
            "grid_layout": "4x2"
        })
        
        if t % 10 == 0:
            print(f"已处理 {t}/{duration} 秒...")

    cap.release()
    
    # 保存结果
    if features_list:
        final_matrix = np.vstack(features_list)
        np.save(FEATURE_FILE, final_matrix)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ 特征提取完成! 矩阵形状: {final_matrix.shape}")
    else:
        print("❌ 错误: 未提取到任何特征")

if __name__ == "__main__":
    extract_features()