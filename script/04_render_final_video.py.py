import opentimelineio as otio
import subprocess
import os

# --- 路径配置 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
OTIO_PATH = os.path.join(BASE_DIR, "timeline.otio")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "final_highlight.mp4")

def render_highlights():
    if not os.path.exists(OTIO_PATH):
        print(f"error: cannot find OTIO file {OTIO_PATH}")
        return

    # 1. 加载 OTIO 剪辑决策表
    timeline = otio.adapters.read_from_file(OTIO_PATH)
    # 获取第一个视频轨道
    video_track = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video][0]
    
    inputs = ""
    filter_complex = ""
    clip_count = 0
    
    print("start to parse the AI editing decisions...")
    
    # 2. 遍历轨道上的每个 Clip，构建 FFmpeg 命令
    for item in video_track:
        if isinstance(item, otio.schema.Clip):
            # 获取素材路径和时间范围
            media_path = item.media_reference.target_url
            # OTIO 中的时间是 RationalTime 对象，需转为秒
            start_sec = item.source_range.start_time.to_seconds()
            dur_sec = item.source_range.duration.to_seconds()
            
            print(f"clip {clip_count}: from {os.path.basename(media_path)} | start: {start_sec}s | duration: {dur_sec}s")
            
            # -ss 放在 -i 前面可以利用 FFmpeg 的快速寻求功能
            inputs += f" -ss {start_sec} -t {dur_sec} -i {media_path}"
            
            # 构建滤镜链：对每个输入流进行归一化（防止原始视频分辨率不一致导致合并失败）
            filter_complex += f"[{clip_count}:v]scale=1920:1080,setpts=PTS-STARTPTS[v{clip_count}];"
            clip_count += 1
    
    if clip_count == 0:
        print("no valid clip found.")
        return

    # 3. 合并所有片段 (Concat)
    concat_inputs = "".join([f"[v{i}]" for i in range(clip_count)])
    filter_complex += f"{concat_inputs}concat=n={clip_count}:v=1:a=0[outv]"
    
    # 最终命令：只导出视频流 (-a=0 指不处理音频，若需音频可改为 a=1)
    ffmpeg_cmd = (
        f"ffmpeg {inputs} -filter_complex \"{filter_complex}\" "
        f"-map \"[outv]\" -c:v libx264 -preset fast -crf 23 {OUTPUT_VIDEO} -y"
    )
    
    print("\nstart to launch the physical rendering...")
    # print(f"执行命令: {ffmpeg_cmd}") # 调试用
    
    try:
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
        print(f"\nphysical rendering finished: {OUTPUT_VIDEO}")
    except subprocess.CalledProcessError as e:
        print(f"\nphysical rendering failed: {e}")

if __name__ == "__main__":
    render_highlights()