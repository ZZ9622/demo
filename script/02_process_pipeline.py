import numpy as np
import json
import opentimelineio as otio
import os

BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
DATA_DIR = "/home/SONY/s7000043396/Downloads/demo/data/demo-data"

def run_pipeline():
    # 1. 加载 8 机位布局
    with open(os.path.join(BASE_DIR, "Camera_Layout.json"), 'r') as f:
        layout = json.load(f)
    video_list = [item['file'] for item in layout['mapping']]
    
    print(f"found {len(video_list)} cameras input.")
    
    # 2. 模拟特征提取
    duration = 4 
    features = np.random.rand(duration, 1024).astype(np.float32)
    np.save(os.path.join(BASE_DIR, "Mosaic_preview_features.npy"), features)
    
    # 3. 模拟 vLLM-Omni 决策 (针对 8 个机位进行抽样决策)
    # 比如：前2秒选第1个视频 (def2_cam_00)，后2秒选最后一个视频 (def2_cam_73)
    decisions = [
        {"start": 0, "dur": 2, "cam_idx": 0}, 
        {"start": 2, "dur": 2, "cam_idx": 7}  
    ]

    # 4. 生成 OTIO
    timeline = otio.schema.Timeline(name="AI_8_Cam_Edit")
    track = otio.schema.Track(name="Main_Sequence")
    timeline.tracks.append(track)

    fps = 30
    for d in decisions:
        file_name = video_list[d['cam_idx']]
        full_path = os.path.join(DATA_DIR, file_name)
        
        media_ref = otio.schema.ExternalReference(target_url=full_path)
        clip = otio.schema.Clip(
            name=f"Action_{file_name}",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(d['start'] * fps, fps),
                duration=otio.opentime.RationalTime(d['dur'] * fps, fps)
            )
        )
        track.append(clip)

    otio_path = os.path.join(BASE_DIR, "timeline.otio")
    otio.adapters.write_to_file(timeline, otio_path)
    print(f"8 cameras pipeline finished")

if __name__ == "__main__":
    run_pipeline()