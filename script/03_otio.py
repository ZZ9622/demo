import numpy as np
import json
import os
import opentimelineio as otio

# --- 配置 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
DATA_DIR = "/home/SONY/s7000043396/Downloads/demo/data/demo-data"

def stage2_interaction_and_decision(user_prompt):
    print(f"start to perform the semantic retrieval and directing decision | instruction: {user_prompt}")
    
    # 1. 加载 02 生成的特征和元数据
    features = np.load(os.path.join(BASE_DIR, "Mosaic_preview_features.npy"))
    with open(os.path.join(BASE_DIR, "feature_metadata.json"), "r") as f:
        metadata = json.load(f)
    with open(os.path.join(BASE_DIR, "Camera_Layout.json"), "r") as f:
        layout = json.load(f)

    # 2. [VideoChat-Flash 检索逻辑]
    # 模拟检索：计算文本向量与特征相似度
    scores = np.random.uniform(0.7, 0.98, len(features))
    candidates = [metadata[i] for i, s in enumerate(scores) if s > 0.85]
    
    with open(os.path.join(BASE_DIR, "highlight_candidates.json"), "w") as f:
        json.dump(candidates, f, indent=4)

    # 3. [vLLM-Omni 决策逻辑]
    timeline = otio.schema.Timeline(name="AI_Directed_Edit")
    track = otio.schema.Track(name="Main_Sequence")
    timeline.tracks.append(track)

    for cand in candidates:
        # 模拟导播在 8 个机位中选择最好的
        best_cam_idx = 5  # 假设 vLLM-Omni 选中了 int1_cam_51.mp4
        file_name = layout["mapping"][best_cam_idx]["file"]
        
        media_ref = otio.schema.ExternalReference(target_url=os.path.join(DATA_DIR, file_name))
        clip = otio.schema.Clip(
            name=f"Highlight_{file_name}",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(cand['timestamp_sec'] * 30, 30),
                duration=otio.opentime.RationalTime(30, 30) # 1秒片段
            )
        )
        track.append(clip)

    otio.adapters.write_to_file(timeline, os.path.join(BASE_DIR, "timeline.otio"))
    print(f"editing decisions completed, OTIO updated.")

if __name__ == "__main__":
    stage2_interaction_and_decision("Find the slam dunk and block shots.")