import subprocess
import json
import os
import glob

# --- 配置路径 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo"
DATA_DIR = os.path.join(BASE_DIR, "data/demo-data")
SCRIPT_DIR = os.path.join(BASE_DIR, "script")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_MOSAIC = os.path.join(OUTPUT_DIR, "mosaic_preview.mp4")
LAYOUT_FILE = os.path.join(OUTPUT_DIR, "Camera_Layout.json")

def create_mosaic():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(DATA_DIR, "*.mp4")))[:8]
    
    if len(videos) < 8:
        print(f"error: only found {len(videos)} videos")
        return

    print(f"✅ found {len(videos)} videos, starting to create mosaic...")

    # 构建输入参数
    inputs = " ".join([f"-i {v}" for v in videos])
    
    # 核心修改：缩放每路视频至 960x540，然后拼接
    scale_filters = "".join([f"[{i}:v]scale=960:540[v{i}];" for i in range(8)])
    stack_layout = "xstack=inputs=8:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w4_h0|w4+w5_h0|w4+w5+w6_h0"
    
    filter_complex = f"{scale_filters}[v0][v1][v2][v3][v4][v5][v6][v7]{stack_layout}[outv]"

    # 使用 NVENC 加速
    cmd = (
        f"ffmpeg -hide_banner {inputs} -filter_complex \"{filter_complex}\" "
        f"-map \"[outv]\" -c:v h264_nvenc -preset p4 -cq 24 -y {OUTPUT_MOSAIC}"
    )

    print("🚀 executing GPU accelerated mosaic creation...")
    subprocess.run(cmd, shell=True, check=True)

    # 生成映射文件
    layout_data = {"layout": "4x2", "mapping": [{"camera_id": i, "file": os.path.basename(v)} for i, v in enumerate(videos)]}
    with open(LAYOUT_FILE, "w") as f:
        json.dump(layout_data, f, indent=4)
    print(f"✨ mosaic preview video generated: {OUTPUT_MOSAIC}")

if __name__ == "__main__":
    create_mosaic()