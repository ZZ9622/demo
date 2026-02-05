import numpy as np
import json
import os
import opentimelineio as otio
import cv2
import re

# ==========================================
# 1. 模型加载区 (Mac MLX vs CUDA vLLM)
# ==========================================

# --- [Mac M3 Pro 方案] MLX (Apple Silicon 专用加速) ---
# 自动下载并加载 4-bit 量化版 Qwen2-VL，显存仅需 ~6GB
from mlx_vlm import load, generate
from mlx_vlm.utils import load_image
MODEL_PATH = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
print(f"🚀 正在加载 Mac MLX 模型: {MODEL_PATH}")
model, processor = load(MODEL_PATH, trust_remote_code=True)

# --- [CUDA / Linux 方案] vLLM (服务器级超快推理) ---
# """
# from vllm import LLM, SamplingParams
# MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
# print("正在加载 CUDA vLLM 模型...")
# model = LLM(model=MODEL_PATH, quantization=None, enforce_eager=True)
# sampling_params = SamplingParams(temperature=0.1, max_tokens=50)
# """

# ==========================================
# 2. 配置与工具函数
# ==========================================
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
DATA_DIR = "/home/SONY/s7000043396/Downloads/demo/data/demo-data"
MOSAIC_VIDEO = os.path.join(BASE_DIR, "mosaic_preview_720p.mp4")

def get_best_camera_ai(timestamp_sec, user_prompt):
    """
    核心决策函数：截取该秒的画面 -> 送入 VL 大模型 -> 解析最佳机位
    """
    # 1. 从视频中截取一帧图片
    cap = cv2.VideoCapture(MOSAIC_VIDEO)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return 0 # Fallback
    
    temp_img_path = os.path.join(BASE_DIR, "temp_inference.jpg")
    cv2.imwrite(temp_img_path, frame)

    # 2. 构建多模态 Prompt (教会 AI 怎么看 4x2 布局)
    prompt_text = f"""
    Below is a mosaic view of 8 cameras covering a basketball game.
    Layout:
    [Cam 0] [Cam 1] [Cam 2] [Cam 3]
    [Cam 4] [Cam 5] [Cam 6] [Cam 7]

    User Request: "{user_prompt}"

    Task: Identify which SINGLE camera ID (0-7) captures the action described best.
    Criteria: Clear view, no obstruction, good composition.
    Output: Return ONLY the camera ID number (e.g., 5). Do not explain.
    """

    # ==========================================
    # 推理逻辑 (Mac Active)
    # ==========================================
    response = generate(
        model, 
        processor, 
        image=temp_img_path, 
        prompt=prompt_text, 
        max_tokens=10, 
        verbose=False
    )
    
    # ==========================================
    # 推理逻辑 (CUDA Option - Commented)
    # ==========================================
    # """
    # inputs = [{"prompt": prompt_text, "multi_modal_data": {"image": frame}}]
    # outputs = model.generate(inputs, sampling_params)
    # response = outputs[0].outputs[0].text
    # """

    # 3. 解析结果 (提取数字)
    try:
        # 使用正则找数字
        best_cam = int(re.search(r'\d+', response).group())
        return max(0, min(7, best_cam)) # 限制在 0-7
    except:
        print(f"⚠️ AI 输出无法解析: '{response}'，默认使用机位 0")
        return 0

def run_pipeline(user_prompt):
    print(f"--- 收到导播指令: {user_prompt} ---")
    
    # 加载布局
    with open(os.path.join(BASE_DIR, "Camera_Layout.json"), "r") as f:
        layout = json.load(f)
        
    # [模拟检索层] 假设 VideoChat 已经帮我们找到了几个关键时间点
    # 实际项目中，这里应该先用 02 的特征去搜索 features.npy
    # 这里为了演示 03 的核心能力，我们硬编码了两个精彩时刻
    candidate_timestamps = [2, 10] # 假设第2秒和第10秒有精彩镜头

    # 创建 OTIO 时间轴
    timeline = otio.schema.Timeline(name="AI_Mac_Edit")
    track = otio.schema.Track(name="Main_Sequence")
    timeline.tracks.append(track)
    
    for ts in candidate_timestamps:
        print(f"\n🔍 正在分析第 {ts} 秒的画面...")
        
        # 调用 AI 决策
        best_cam_idx = get_best_camera_ai(ts, user_prompt)
        print(f"🤖 AI 决定选用: 机位 {best_cam_idx}")
        
        # 写入剪辑表
        file_name = layout["mapping"][best_cam_idx]["file"]
        media_ref = otio.schema.ExternalReference(target_url=os.path.join(DATA_DIR, file_name))
        
        clip = otio.schema.Clip(
            name=f"Cut_{ts}s_Cam{best_cam_idx}",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(ts * 30, 30),
                duration=otio.opentime.RationalTime(30, 30) # 剪辑 1 秒
            )
        )
        track.append(clip)

    # 保存结果
    output_path = os.path.join(BASE_DIR, "timeline.otio")
    otio.adapters.write_to_file(timeline, output_path)
    print(f"\n✅ 剪辑决策表已生成: {output_path}")
    print("下一步: 运行 04_render_final_video.py 合成视频")

if __name__ == "__main__":
    # 示例：找一个扣篮动作
    run_pipeline("Find the clearest view of the player slam dunking.")