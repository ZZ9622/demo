#!/usr/bin/env python3
"""
analyze_qwen25vl3b_ubuntu.py
----------------------------
Analyze a local basketball video clip around a trigger_time using Qwen2.5-VL.
Runs on HP ZBook Power G9 with Ubuntu 24.04.4 LTS.
Optimized for Intel i7-12700H (20 cores) and 32GB RAM.
- Window: trigger_time - 6s  ->  trigger_time + 4s
- Video:  camera6_h264.mp4  (rightbaseline camera)
"""

import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import logging as hf_logging
hf_logging.disable_progress_bar()

import argparse

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["OMP_NUM_THREADS"] = "20"
hf_logging.set_verbosity_error()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-VL-3B-Instruct"
CAMERA_TYPE  = "rightbaseline"

# VIDEO_PATH   = Path("./data/apidis/camera6_h264.mp4")
# trigger_time = 6654   # event timestamp to analyze (seconds)
# PRE_SECONDS  = 6      # seconds before trigger_time
# POST_SECONDS = 4      # seconds after trigger_time

VIDEO_PATH   = Path("/home/SONY/s7000043358/Sports_HL/demo/data/apidis/camera6_from_1h50m_to_end_89_97.mp4")
CAMERA_TYPE  = "rightbaseline"   # 如果你想区分摄像机，可以改成 "camera3"，否则可以保持不变

trigger_time = 93    # 触发时间（秒）
PRE_SECONDS  = 4     # 触发时间前 5 秒
POST_SECONDS = 4     # 触发时间后 5 秒

SAMPLE_FPS   = 4      # Conservative for memory efficiency
MAX_SIDE     = 192    # Reduced resolution for memory efficiency


# ── Video extraction ──────────────────────────────────────────────────────────
def extract_clip(video_path: Path, start_sec: float, end_sec: float) -> np.ndarray:
    """Return a (T, H, W, 3) uint8 RGB array sampled at SAMPLE_FPS."""
    print(f"  Opening video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps
    print(f"  Video specs: {native_fps:.1f} fps | {total_frames} frames | {duration:.1f}s")

    start_sec = max(0.0, start_sec)
    end_sec   = min(duration, end_sec)
    step      = max(1, int(round(native_fps / SAMPLE_FPS)))
    start_f   = int(start_sec * native_fps)
    end_f     = int(end_sec   * native_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames, idx = [], start_f

    while idx <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_f) % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            frames.append(rgb)
        idx += 1

    cap.release()
    clip = np.stack(frames, axis=0)
    print(f"  ✓ Extracted {clip.shape[0]} frames @ {SAMPLE_FPS} fps")
    print(f"  Time range: {start_sec:.1f}s – {end_sec:.1f}s | Resolution: {clip.shape[2]}×{clip.shape[1]}")
    return clip


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    # Check GPU memory if CUDA is available
    device = "cpu"  # Default to CPU for stability
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Detected GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {gpu_mem:.1f} GB")
        
        # Only use GPU if it has enough memory (at least 8GB recommended for 3B model)
        if gpu_mem >= 6.0:
            device = "cuda"
            print("  Using GPU acceleration")
        else:
            print("  GPU memory insufficient, falling back to CPU")
    else:
        print("  No GPU detected, using CPU")
    
    print(f"  Target device: {device.upper()}")
    print(f"  Model: {MODEL_ID}")
    print("  Loading model and processor...")
    
    t0 = time.time()
    
    # Use CPU-only configuration for stability
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     MODEL_ID, 
    #     torch_dtype=torch.float32,  # Always use float32 for compatibility
    #     device_map=None,  # Load to CPU first
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True
    # ).eval()

    # 如果 GPU 可用并且显存足够，直接用半精度加载到 GPU
    if device == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,      # 或 bfloat16，如果你的 GPU 支持
            device_map="cuda",              # 直接放到 GPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,      # CPU 用 float32
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
    
    # Move to device if GPU is suitable
    # if device == "cuda":
    #     try:
    #         model = model.to(device)
    #         torch.cuda.empty_cache()
    #     except RuntimeError as e:
    #         print(f"  GPU loading failed: {e}")
    #         print("  Falling back to CPU")
    #         model = model.to("cpu")
    #         device = "cpu"
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=64  * 28 * 28,   # ~50K px
        max_pixels=256 * 28 * 28,   # Reduced for memory efficiency
        trust_remote_code=True
    )
    
    load_time = time.time() - t0
    print(f"  ✓ Model loaded successfully in {load_time:.1f}s")
    
    # Display model info
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    actual_device = next(model.parameters()).device
    print(f"  Model parameters: {num_params:.1f}B")
    print(f"  Actual device: {actual_device}")

    # 在 load_model 函数中，返回 model 前添加一行：
    print("  Compiling model (this may take a minute)...")
    model = torch.compile(model)        
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────
def analyze(model, processor, clip: np.ndarray, start_sec: float, end_sec: float) -> str:
    pre  = trigger_time - start_sec
    post = end_sec - trigger_time

    # prompt = (
    #     f"This is a {end_sec - start_sec:.0f}-second basketball video clip "
    #     f"from the {CAMERA_TYPE} camera ({start_sec:.1f}s – {end_sec:.1f}s).\n"
    #     f"The system-marked event trigger_time is {trigger_time}s, "
    #     f"which is {pre:.0f}s into this clip (pre={pre:.0f}s / post={post:.0f}s).\n\n"
    #     "Describe what you actually see in the video using this structure:\n"
    #     "[Before] What is happening before trigger_time? Where is the ball? What are the players doing?\n"
    #     "[At] What is the key action at/around trigger_time? (shot attempt, pass, foul, etc.) What is the ball trajectory?\n"
    #     "[After] What is the outcome? (scored / missed / out of bounds / foul) How does possession change?\n\n"
    #     "Only describe what is visible in the video. Do not guess or infer."
    # )

    # prompt = (
    #     f"This is a {end_sec - start_sec:.0f}-second basketball video clip "
    #     f"from the {CAMERA_TYPE} camera ({start_sec:.1f}s – {end_sec:.1f}s).\n\n"
    #     "Watch the entire clip and, based only on the visual content, decide what you consider to be the main basketball event "
    #     "(such as a shot attempt, foul, turnover, steal, rebound, out of bounds, etc.).\n"
    #     "Then describe the video in three parts using this structure:\n"
    #     "[Before] What is happening earlier in the clip before the main event? Where is the ball? What are the players doing and how is the play developing?\n"
    #     "[During] What is the key action around the main event? Who is involved, what exactly happens, and what is the ball trajectory?\n"
    #     "[After] What is the outcome of that main event? (scored / missed / foul / turnover / out of bounds, etc.) How does possession or game state change afterwards?\n\n"
    #     "Only describe what is visible in the video. Do not guess information that cannot be seen."
    # )

    # 这里的 start_sec 和 end_sec 是你视频片段的绝对时间（例如 89 和 97）
    # duration 就是 8.0 秒
    

    duration = end_sec - start_sec
    # prompt = """
    # You are a video observation system.

    # 任务：
    # 对视频中的确实进球的投篮，用自然语言描述：

    # 1. 投篮的准备的动作
    # 2. 出手瞬间的动作细节
    # 3. 投篮结果（命中/未中/被盖）
    # 4. 出手瞬间的投篮的人和防守的人的人员空间站位

    # 要求：
    # 不要补充视频里无法直接观测到的内容

    # 输出形式：
    # [Before]:投篮的准备动作\n
    # [During]:出手瞬间的动作细节,出手瞬间的人员空间站位\n
    # [After]:投篮结果\n
    # """


    prompt = f"""
    你现在是一个视频时间定位助手，需要帮助下游的 TFVTG 模型在一段篮球比赛视频中找到【一次投篮得分事件】对应的时间片段。

    给定的是一段已经剪好的短视频（时长约 {duration:.1f} 秒，包含一次进球投篮），请你只根据画面内容完成下面任务：

    1. 把这次进球投篮拆成若干个子事件（sub-queries），每个子事件只包含一个简单的可见动作或状态
    2. 对于每个子事件，用 2~3 句不同的英文描述同一个画面（可以是同义改写），方便 TFVTG 做文本-视频匹配。
    3. 这些子事件应该覆盖整次投篮过程的关键画面，从准备、起跳、出手、空中飞行到进筐，以及防守者的位置关系。
    4. 如果画面里有多个明显阶段，可以拆成 1-3 个 sub-query；如果只有一个很简单的进球动作，可以只保留 1 个 sub-query。

    请严格按照下面的 JSON 格式输出（不要多余说明，不要加 Markdown，不要用```json包裹，只输出 JSON），请在每个描述里包含投篮人员在场上的位置描述：

    {{
    "reasoning": "用一小段话解释你是如何拆分这些子事件的（可以是中文）。",
    "query_json": [
        {{
        "sub_query_id": 1,
        "descriptions": [
            "子事件1的描述版本A（英文，自然语句）",
            "子事件1的描述版本B（同义改写）"
        ]
        }},
        {{
        "sub_query_id": 2,
        "descriptions": [
            "子事件2的描述版本A",
            "子事件2的描述版本B"
        ]
        }}
        // 如果需要，可以继续添加 sub_query_id 3,4,5...
    ]
    }}

    要求：
    - 只描述视频中能直接看到的内容，不要猜测比分、球员名字等不可见信息。
    - 子事件之间要符合时间顺序（sub_query_id 越大，时间越往后）。
    - JSON 必须是合法的，可被 Python json.loads 直接解析。
    """


    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": prompt},
    ]}]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], videos=[clip], fps=SAMPLE_FPS, padding=True, return_tensors="pt")
    
    device = next(model.parameters()).device
    
    # Clear GPU cache before inference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"  Processing {clip.shape[0]} frames at {SAMPLE_FPS} fps...")
    print(f"  Device: {device}")
    print("  Running inference...")
    
    try:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        t0 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=600,  # Reduced for memory efficiency
                do_sample=False, 
                temperature=0.3, 
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=False  # Disable cache to save memory
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        result  = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        inference_time = time.time() - t0
        print(f"  ✓ Inference completed in {inference_time:.1f}s")
        print(f"  Generated {len(result.split())} words")
        
        # Clean up
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  GPU memory error: {e}")
        print("  Try reducing video resolution or frame count")
        raise
    except Exception as e:
        print(f"  Inference error: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH),
                        help="要分析的视频路径")
    parser.add_argument("--trigger", type=float, default=trigger_time,
                        help="触发时间（秒）")
    parser.add_argument("--pre", type=float, default=PRE_SECONDS,
                        help="触发前窗口（秒）")
    parser.add_argument("--post", type=float, default=POST_SECONDS,
                        help="触发后窗口（秒）")
    return parser.parse_args()
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 用命令行参数覆盖默认配置（如果没传就用文件里原来的默认值）
    global VIDEO_PATH, trigger_time, PRE_SECONDS, POST_SECONDS
    VIDEO_PATH   = Path(args.video)
    trigger_time = args.trigger
    PRE_SECONDS  = args.pre
    POST_SECONDS = args.post
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()  # 记录开始时间    
    
    print("=" * 60)
    print("  Basketball Video Analysis - Ubuntu Version")
    print("=" * 60)
    print(f"  Video  : {VIDEO_PATH.name}")
    print(f"  Trigger: {trigger_time}s  |  Window: [{trigger_time - PRE_SECONDS}s, {trigger_time + POST_SECONDS}s]")
    print(f"  Config : {SAMPLE_FPS} fps sampling, {MAX_SIDE}px max resolution")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        print(f"ERROR: video not found: {VIDEO_PATH}")
        sys.exit(1)

    # 在整体 pipeline 中，视频已经在外部（pipeline.py）按 trigger_time 前后裁剪成短片段。
    # 这里不再根据 trigger_time 做二次裁剪，而是直接分析整段传入的视频。
    start_sec = 0.0
    end_sec   = 999999.0

    print("\n[1/3] Extracting video clip (full input video, no extra cut)...")
    clip = extract_clip(VIDEO_PATH, start_sec, end_sec)
    if clip.shape[0] == 0:
        print("ERROR: no frames extracted. Check trigger_time is within video duration.")
        sys.exit(1)

    print("\n[2/3] Loading Qwen2.5-VL model...")
    model, processor = load_model()
    
    print("\n[3/3] Analyzing video content...")
    result = analyze(model, processor, clip, start_sec, end_sec)

    print("\n" + "=" * 60)
    print("  ANALYSIS RESULT")
    print("=" * 60)
    print(f"  Camera: {CAMERA_TYPE} | Time window: {start_sec:.1f}s – {end_sec:.1f}s")
    print(f"  Trigger: {trigger_time}s | Frames analyzed: {clip.shape[0]}")
    print("=" * 60)
    print(result)
    print("=" * 60)

    print("\nSaving results...")
    out_dir  = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"analysis_ubuntu_cam6_t{trigger_time}.txt"
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"video      : {VIDEO_PATH}\n")
        f.write(f"trigger    : {trigger_time}s\n")
        f.write(f"model      : {MODEL_ID}\n")
        f.write(f"window     : {start_sec:.1f}s – {end_sec:.1f}s\n")
        f.write(f"camera     : {CAMERA_TYPE}\n")
        f.write(f"sample_fps : {SAMPLE_FPS}\n")
        f.write(f"frames     : {clip.shape[0]}\n")
        f.write("\n" + "=" * 60 + "\nResult:\n" + "=" * 60 + "\n")
        f.write(result + "\n")
    
    print(f"✓ Analysis complete! Results saved to: {out_file}")
    print(f"  File size: {out_file.stat().st_size} bytes")

    total_time = time.time() - t0
    print(f"\n总耗时: {total_time:.1f} 秒")    


if __name__ == "__main__":
    main()