import os
import subprocess
import opentimelineio as otio

# --- 路径配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
OTIO_PATH = os.path.join(OUTPUT_DIR, "timeline.otio")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_highlight_5090.mp4")

def render_from_otio(otio_file):
    """
    直接读取 OTIO 决策文件并调用 FFmpeg 渲染
    """
    if not os.path.exists(otio_file):
        print(f"❌ error: decision file {otio_file} not found, please run 03 script first.")
        return

    # 1. 加载 OTIO
    timeline = otio.adapters.read_from_file(otio_file)
    # 获取 Main 轨道上的所有 Clip
    clips = [item for item in timeline.tracks[0] if isinstance(item, otio.schema.Clip)]
    
    print(f"🎬 detected {len(clips)} AI editing segments, starting rendering...")

    # 2. 构建 FFmpeg 命令
    # 我们将使用 filter_complex 确保每一帧都精确对齐
    input_args = []
    filter_nodes = ""
    
    for i, clip in enumerate(clips):
        # 提取 OTIO 中的决策数据
        media_path = clip.media_reference.target_url
        start_sec = clip.source_range.start_time.to_seconds()
        duration_sec = clip.source_range.duration.to_seconds()
        
        # 为每个片段添加输入：-ss (开始时间) -t (持续时间)
        input_args.extend(["-ss", str(start_sec), "-t", str(duration_sec), "-i", media_path])
        
        # 构建滤镜链标签，例如 [0:v][1:v]...
        filter_nodes += f"[{i}:v]"

    # 拼接滤镜逻辑
    filter_nodes += f"concat=n={len(clips)}:v=1:a=0[outv]"

    # 3. 组合最终命令 (利用 RTX 5090 的 NVENC 加速)
    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_nodes,
        "-map", "[outv]",
        "-c:v", "h264_nvenc", # 5090 硬件加速
        "-preset", "p4",      # 高质量预设
        "-cq", "20",          # 恒定质量
        "-vsync", "cfr",      # 强制恒定帧率
        "-r", "25",           # 目标帧率
        OUTPUT_VIDEO
    ]

    print("🚀 rendering video with NVENC hardware acceleration...")
    
    try:
        # 执行命令
        subprocess.run(cmd, check=True)
        print(f"✨ rendering successful! video saved to: {OUTPUT_VIDEO}")
    except subprocess.CalledProcessError as e:
        print(f"❌ error: rendering failed: {e}")

if __name__ == "__main__":
    # 直接运行，不再需要手动输入 example_decisions
    render_from_otio(OTIO_PATH)