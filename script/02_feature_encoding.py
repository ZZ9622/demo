import numpy as np
import json
import os
import torch  # 本地部署时开启
from transformers import AutoModel  # 本地部署时开启

# --- 配置 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
MOSAIC_VIDEO = os.path.join(BASE_DIR, "mosaic_preview_720p.mp4")
FEAT_DIM = 1024 

def stage1_feature_encoding():
    print("start to encode the mosaic preview video using InternVideo2.5...")
    
    # [本地部署接口]
    model = AutoModel.from_pretrained("OpenGVLab/InternVideo2-5-1B", trust_remote_code=True).cuda()
    features = model.encode_video(MOSAIC_VIDEO)
    
    # [当前模拟逻辑]
    duration = 4 
    feature_matrix = np.random.randn(duration, FEAT_DIM).astype(np.float32)
    feature_matrix /= np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    
    # 输出 1: .npy 矩阵
    np.save(os.path.join(BASE_DIR, "Mosaic_preview_features.npy"), feature_matrix)
    
    # 输出 2: feature_metadata.json (建立 LTC 索引)
    metadata = []
    for t in range(duration):
        metadata.append({
            "timestamp_sec": t,
            "ltc": f"14:00:0{t}:00", # 模拟同步 LTC
            "grid_layout": "4x2"
        })
    
    with open(os.path.join(BASE_DIR, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"feature library and metadata index generated.")

if __name__ == "__main__":
    stage1_feature_encoding()