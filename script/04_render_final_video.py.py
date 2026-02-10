import opentimelineio as otio
import subprocess
import os

# --- 配置 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
OTIO_PATH = os.path.join(BASE_DIR, "timeline.otio")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "final_highlight_5090.mp4")

def render():
    print("--- starting physical rendering (NVENC acceleration) ---")
    timeline = otio.adapters.read_from_file(OTIO_PATH)
    video_track = timeline.tracks[0]
    
    inputs = ""
    filter_complex = ""
    count = 0
    
    for item in video_track:
        if isinstance(item, otio.schema.Clip):
            path = item.media_reference.target_url
            start = item.source_range.start_time.to_seconds()
            dur = item.source_range.duration.to_seconds()
            
            inputs += f" -ss {start} -t {dur} -i {path}"
            # 缩放至 1080p 统一规格，防止 4K/8K 混合报错
            filter_complex += f"[{count}:v]scale=1920:1080,setpts=PTS-STARTPTS[v{count}];"
            count += 1
            
    concat_part = "".join([f"[v{i}]" for i in range(count)])
    filter_complex += f"{concat_part}concat=n={count}:v=1:a=0[outv]"
    
    # 使用 h264_nvenc 进行极速编码
    cmd = (
        f"ffmpeg {inputs} -filter_complex \"{filter_complex}\" "
        f"-map \"[outv]\" -c:v h264_nvenc -preset p7 -cq 20 -y {OUTPUT_VIDEO}"
    )
    
    print(f"executing command: {cmd}")
    subprocess.run(cmd, shell=True)
    print(f"rendering completed: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    render()