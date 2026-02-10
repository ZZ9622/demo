import numpy as np
import json
import os
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from datetime import timedelta

# --- 配置 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
MOSAIC_VIDEO = os.path.join(BASE_DIR, "mosaic_preview.mp4")
FEATURE_FILE = os.path.join(BASE_DIR, "Mosaic_preview_features.npy")
METADATA_FILE = os.path.join(BASE_DIR, "feature_metadata.json")

# 模型：Google SigLIP (目前 Transformer Vision Encoder 的 SOTA)
MODEL_ID = "google/siglip-so400m-patch14-384"

def extract_features():
    print(f"loading SigLIP model: {MODEL_ID}...")
    
    # 加载模型到 GPU，使用 BF16 精度 (5090 最佳甜点精度)
    device = "cuda"
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    cap = cv2.VideoCapture(MOSAIC_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)
    
    features_list = []
    metadata = []
    
    print(f"video duration: {duration}s | sampling rate: 1 FPS | starting fast extraction...")

    for t in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret: break
        
        # OpenCV (BGR) -> PIL (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 预处理
        inputs = processor(images=image, return_tensors="pt").to(device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            # 提取特征
            image_features = model.get_image_features(**inputs)
            # 归一化 (关键步骤)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        features_list.append(image_features.cpu().float().numpy()) # 转回 float32 存盘
        
        # 构建元数据索引
        metadata.append({
            "timestamp_sec": t,
            "ltc": (f"14:00:{t//60:02d}:{t%60:02d}:00"), # 模拟时间码
            "frame_idx": int(t * fps)
        })
        
        if t % 10 == 0:
            print(f"processed {t}/{duration} seconds...", end="\r")

    cap.release()
    
    # 保存
    final_matrix = np.vstack(features_list)
    np.save(FEATURE_FILE, final_matrix)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nfeatures extracted: {final_matrix.shape}")

if __name__ == "__main__":
    extract_features()