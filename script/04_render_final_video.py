import subprocess
import os

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data/demo-data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_highlight_5090.mp4")

def render_video(decision_results):
    """
    decision_results 应该是类似这样的列表: 
    [{'start': 0, 'end': 2, 'cam': 'def2_cam_00.mp4'}, ...]
    """
    
    # 创建一个 FFmpeg concat 临时文件
    concat_file = os.path.join(OUTPUT_DIR, "concat_list.txt")
    
    with open(concat_file, "w") as f:
        for clip in decision_results:
            video_path = os.path.join(DATA_DIR, clip['cam'])
            # 使用 inpoint 和 outpoint 精确控制时长
            f.write(f"file '{video_path}'\n")
            f.write(f"inpoint {clip['start']}\n")
            f.write(f"outpoint {clip['end']}\n")

    print("🎬 using NVENC hardware acceleration to render the final video...")
    
    # 核心命令：使用 -f concat 和 -safe 0
    # 注意：添加 -vsync cfr 确保帧率恒定，防止时长翻倍
    cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {concat_file} "
        f"-c:v h264_nvenc -preset p4 -cq 20 -vsync cfr -r 25 " 
        f"{OUTPUT_VIDEO}"
    )
    
    subprocess.run(cmd, shell=True, check=True)
    print(f"✅ rendering completed! output path: {OUTPUT_VIDEO}")

# 模拟 AI 决策后的逻辑 (请确保这部分与 03_ai_decision 的输出对接)
if __name__ == "__main__":
    # 假设 AI 选了 cam_00 跑全场，或者每 3 秒换一个机位
    example_decisions = [
        {'start': 0, 'end': 3, 'cam': 'def2_cam_00.mp4'},
        {'start': 3, 'end': 6, 'cam': 'def2_cam_46.mp4'},
        {'start': 6, 'end': 9, 'cam': 'def2_cam_73.mp4'}
    ]
    render_video(example_decisions)