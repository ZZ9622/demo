import subprocess
import json
import os
import glob

# --- 配置路径 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo"
DATA_DIR = os.path.join(BASE_DIR, "data/demo-data")
SCRIPT_DIR = os.path.join(BASE_DIR, "script")
OUTPUT_MOSAIC = os.path.join(SCRIPT_DIR, "mosaic_preview.mp4")
LAYOUT_FILE = os.path.join(SCRIPT_DIR, "Camera_Layout.json")

def create_mosaic():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    
    # 1. 获取所有 MP4 文件并排序
    videos = sorted(glob.glob(os.path.join(DATA_DIR, "*.mp4")))[:8]
    if len(videos) < 8:
        print("error: less than 8 videos")
        return

    print(f"found {len(videos)} videos, building 4x2 mosaic...")

    # 2. 生成 FFmpeg 复杂滤镜命令 (xstack 4x2 布局)
    # 5090 性能极强，我们直接使用 h264_nvenc 硬件编码
    inputs = ""
    for v in videos:
        inputs += f"-i {v} "
    
    filter_complex = (
        "[0:v][1:v][2:v][3:v]"
        "[4:v][5:v][6:v][7:v]"
        "xstack=inputs=8:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w4_h0|w4+w5_h0|w4+w5+w6_h0"
        "[v]"
    )

    cmd = (
        f"ffmpeg {inputs} -filter_complex \"{filter_complex}\" "
        f"-map \"[v]\" -c:v h264_nvenc -preset p7 -cq 19 -y {OUTPUT_MOSAIC}"
    )

    print("executing GPU hardware acceleration mosaic...")
    subprocess.run(cmd, shell=True, check=True)

    # 3. 生成物理映射 JSON
    layout_data = {"layout": "4x2", "mapping": []}
    for idx, vid_path in enumerate(videos):
        layout_data["mapping"].append({
            "camera_id": idx,
            "file": os.path.basename(vid_path),
            "row": 0 if idx < 4 else 1,
            "col": idx % 4
        })
    
    with open(LAYOUT_FILE, "w") as f:
        json.dump(layout_data, f, indent=4)
    
    print(f"mosaic completed: {OUTPUT_MOSAIC}")
    print(f"mapping table generated: {LAYOUT_FILE}")

if __name__ == "__main__":
    create_mosaic()