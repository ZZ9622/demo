#!/Volumes/T7/venv/bin/python
"""
analyze_local_m3.py
-------------------
MacBook M3 Pro 适配版本
- 使用 Apple MPS 加速（Metal Performance Shaders）
- 直接读取本地视频文件，无需网络下载
- 检测 trigger_time 前6秒 + 后4秒内发生的事件
- 视频: camera6_h264.mp4  /  摄像头类型: rightbaseline
"""

import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import logging as hf_logging

# ── 屏蔽冗余日志 ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ── 路径与参数配置 ─────────────────────────────────────────────────────────────
VIDEO_PATH   = Path("/Volumes/T7/demo/script/camera6_h264.mp4")
MODEL_ID     = "Qwen/Qwen2.5-VL-3B-Instruct"    # 3B，CPU 推理（绕开 MPS 4GB 单张量硬限制）
CAMERA_TYPE  = "rightbaseline"                  # camera6 对应的视角名称

# trigger_time: 视频中需要分析的"事件触发时间点"（秒）
trigger_time  = 6654          # ← 可修改为任意你想检测的时间戳
PRE_SECONDS   = 6             # trigger_time 之前取几秒
POST_SECONDS  = 4             # trigger_time 之后取几秒
SAMPLE_FPS    = 2             # 视频模式下每秒采样帧数传给模型（2fps × 10s = 20帧）
MAX_SIDE      = 448           # 送入模型前将长边限制在此像素（降低视觉编码器内存）


# ── 设备选择：模型权重加载用 CPU，彻底绕开 MPS 4 GB 单张量硬限 ────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("ℹ️  检测到 Apple MPS，但 Vision Encoder 会触发 >4 GB 单张量，改用 CPU 推理")
        print("   M3 Pro 统一内存（18/36 GB）无此限制，速度依然可接受")
    else:
        print("ℹ️  使用 CPU 推理")
    return torch.device("cpu")


# ── 从本地视频提取指定时间段，返回 numpy 视频数组 ──────────────────────────────
def extract_video_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 2,
) -> tuple[np.ndarray, float]:
    """
    提取 [start_sec, end_sec] 区间内的帧，返回:
      - clip: np.ndarray, shape (T, H, W, 3), uint8, RGB
      - native_fps: 原始帧率
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps

    print(f"📹 视频信息: {native_fps:.1f} fps | {total_frames} 帧 | 时长 {duration_sec:.1f}s")

    start_sec = max(0.0, start_sec)
    end_sec   = min(duration_sec, end_sec)

    step_frames = max(1, int(round(native_fps / sample_fps)))
    start_frame = int(start_sec * native_fps)
    end_frame   = int(end_sec   * native_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    frames: list[np.ndarray] = []

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % step_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 缩小长边
            h, w = rgb.shape[:2]
            if max(w, h) > MAX_SIDE:
                scale = MAX_SIDE / max(w, h)
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        frame_idx += 1

    cap.release()
    clip = np.stack(frames, axis=0)  # (T, H, W, 3)
    print(f"✅ 视频片段提取完成: {clip.shape[0]} 帧 @ {sample_fps} fps, {start_sec:.1f}s – {end_sec:.1f}s")
    return clip, native_fps


# ── 加载 Qwen2.5-VL 模型（MPS 适配）────────────────────────────────────────────
def load_model(device: torch.device):
    hf_logging.set_verbosity_error()

    print(f"\n🔄 加载模型: {MODEL_ID}  (device: {device})")
    t0 = time.time()

    # CPU 推理：float32，不用 device_map
    dtype = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()

    # min/max_pixels 限制 Qwen2.5-VL 视觉编码器的动态分辨率
    # 默认 max 约 1280*28*28 ≈ 1M px，M3 Pro MPS 单次 <4GB，需严格压缩
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=64  * 28 * 28,   # ~50K px
        max_pixels=256 * 28 * 28,   # ~200K px  (约 448×448)
    )
    print(f"✅ 模型加载完成，耗时 {time.time() - t0:.1f}s")
    return model, processor


# ── 使用 Qwen2.5-VL 以视频模式分析片段 ──────────────────────────────────────
def analyze_video_clip(
    model,
    processor,
    clip: np.ndarray,
    device: torch.device,
    trigger_sec: float,
    start_sec: float,
    end_sec: float,
    camera_type: str,
) -> str:
    """
    以 video 模式（而非多图模式）传入连续帧，让模型理解时序动作。
    clip: np.ndarray (T, H, W, 3) uint8 RGB
    """
    if clip.shape[0] == 0:
        return "⚠️  没有提取到任何帧，无法分析。"

    pre  = trigger_sec - start_sec   # trigger 前多少秒
    post = end_sec - trigger_sec     # trigger 后多少秒

    prompt = (
        f"这是一段来自篮球比赛 {camera_type} 机位的视频，"
        f"时长约 {end_sec - start_sec:.0f} 秒（{start_sec:.1f}s – {end_sec:.1f}s）。\n"
        f"其中 {trigger_sec:.1f}s 是系统标记的'事件触发时间点'，"
        f"它位于视频片段约 {pre:.0f} 秒处（前 {pre:.0f}s / 后 {post:.0f}s）。\n\n"
        "请用中文，以【前→中→后】三段式描述这段视频：\n"
        "【前】触发时间点之前：球在哪里？球员在做什么准备动作？\n"
        "【中】触发时间点附近：发生了什么核心动作（投篮出手、传球、犯规等）？球的轨迹如何？\n"
        "【后】触发时间点之后：动作结果如何（进球/未进/界外/犯规）？球权如何转移？\n\n"
        "注意：只描述你在视频中实际看到的内容，不要猜测或补全。"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "video"},   # 视频占位符，与 processor(videos=[clip]) 对应
            {"type": "text", "text": prompt},
        ]
    }]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 传入视频数组：list of (T, H, W, 3) numpy arrays
    inputs = processor(
        text=[text_prompt],
        videos=[clip],
        fps=SAMPLE_FPS,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"\n🎬 Qwen2.5-VL 视频模式分析 {clip.shape[0]} 帧（{start_sec:.1f}s–{end_sec:.1f}s），请稍候…")
    t0 = time.time()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"✅ 推理完成，耗时 {time.time() - t0:.1f}s")
    return result


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  视频事件分析  |  MacBook M3 Pro 版")
    print("=" * 60)
    print(f"  视频文件   : {VIDEO_PATH}")
    print(f"  触发时间点 : {trigger_time}s")
    print(f"  分析窗口   : [{trigger_time - PRE_SECONDS}s, {trigger_time + POST_SECONDS}s]")
    print(f"  采样帧率   : {SAMPLE_FPS} fps")
    print("=" * 60)

    # 0. 检查视频文件
    if not VIDEO_PATH.exists():
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
        sys.exit(1)

    # 1. 选择设备
    device = get_device()

    # 2. 提取视频片段（numpy array）
    start_sec = float(trigger_time - PRE_SECONDS)
    end_sec   = float(trigger_time + POST_SECONDS)
    clip, _ = extract_video_clip(VIDEO_PATH, start_sec, end_sec, sample_fps=SAMPLE_FPS)

    if clip.shape[0] == 0:
        print("❌ 未能提取到任何帧，请检查 trigger_time 是否在视频时长范围内。")
        sys.exit(1)

    # 3. 加载模型
    model, processor = load_model(device)

    # 4. 视频模式分析
    analysis = analyze_video_clip(
        model, processor, clip, device,
        trigger_sec=float(trigger_time),
        start_sec=start_sec,
        end_sec=end_sec,
        camera_type=CAMERA_TYPE,
    )

    # 5. 输出结果
    print("\n" + "=" * 60)
    print(f"  分析结果  |  CAM6 ({CAMERA_TYPE})")
    print(f"  时间窗口  : {start_sec:.1f}s – {end_sec:.1f}s  (trigger: {trigger_time}s)")
    print("=" * 60)
    print(analysis)
    print("=" * 60)

    # 6. 保存结果到文件
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / f"analysis_cam6_t{trigger_time}.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"视频文件  : {VIDEO_PATH}\n")
        f.write(f"触发时间点: {trigger_time}s\n")
        f.write(f"分析窗口  : {start_sec:.1f}s – {end_sec:.1f}s\n")
        f.write(f"摄像头类型: {CAMERA_TYPE}\n")
        f.write(f"采样帧率  : {SAMPLE_FPS} fps\n")
        f.write(f"提取帧数  : {clip.shape[0]}\n")
        f.write("\n" + "=" * 60 + "\n分析结果:\n" + "=" * 60 + "\n")
        f.write(analysis + "\n")
    print(f"\n💾 分析结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
