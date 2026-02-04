import opentimelineio as otio
import subprocess
import os

# 路径配置
OTIO_PATH = "/home/SONY/s7000043396/Downloads/demo/script/timeline.otio"
OUTPUT_VIDEO = "/home/SONY/s7000043396/Downloads/demo/script/final_highlight.mp4"

def render_from_otio():
    timeline = otio.adapters.read_from_file(OTIO_PATH)
    video_track = timeline.tracks[0]
    
    filter_complex = ""
    inputs = ""
    count = 0
    
    print("start to parse the video clips from OTIO decision table...")
    
    for item in video_track:
        if isinstance(item, otio.schema.Clip):
            path = item.media_reference.target_url
            start = item.source_range.start_time.to_seconds()
            duration = item.source_range.duration.to_seconds()
            
            # 构建 FFmpeg 命令片段：精准裁剪每个片段
            inputs += f" -ss {start} -t {duration} -i {path}"
            filter_complex += f"[{count}:v]setpts=PTS-STARTPTS[v{count}];"
            count += 1
    
    # 合并所有片段
    concat_filter = "".join([f"[v{i}]" for i in range(count)])
    filter_complex += f"{concat_filter}concat=n={count}:v=1:a=0[outv]"
    
    cmd = f"ffmpeg {inputs} -filter_complex \"{filter_complex}\" -map \"[outv]\" -c:v libx264 -preset fast {OUTPUT_VIDEO} -y"
    
    print(f"execute the final rendering: {cmd}")
    subprocess.run(cmd, shell=True)
    print(f"final rendering finished: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    render_from_otio()