#!/Volumes/T7/venv/bin/python
"""
analyze_local_m3.py
-------------------
Analyze a local basketball video clip around a trigger_time using Qwen2.5-VL.
Runs on MacBook M3 Pro via CPU (avoids MPS 4 GB single-tensor hard limit).
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

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
hf_logging.set_verbosity_error()

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH   = Path("/Volumes/T7/demo/script/camera6_h264.mp4")
MODEL_ID     = "Qwen/Qwen2.5-VL-3B-Instruct"
CAMERA_TYPE  = "rightbaseline"

trigger_time = 6654   # event timestamp to analyze (seconds)
PRE_SECONDS  = 6      # seconds before trigger_time
POST_SECONDS = 4      # seconds after trigger_time
SAMPLE_FPS   = 2      # frames per second passed to the model
MAX_SIDE     = 448    # resize longest edge to this before encoding

# ── Video extraction ──────────────────────────────────────────────────────────
def extract_clip(video_path: Path, start_sec: float, end_sec: float) -> np.ndarray:
    """Return a (T, H, W, 3) uint8 RGB array sampled at SAMPLE_FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps
    print(f"Video: {native_fps:.1f} fps | {total_frames} frames | {duration:.1f}s")

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
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        idx += 1

    cap.release()
    clip = np.stack(frames, axis=0)
    print(f"Extracted {clip.shape[0]} frames @ {SAMPLE_FPS} fps  [{start_sec:.1f}s – {end_sec:.1f}s]")
    return clip


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    print(f"\nLoading {MODEL_ID} on CPU ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=True
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=64  * 28 * 28,   # ~50K px
        max_pixels=256 * 28 * 28,   # ~200K px  (≈ 448×448)
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────
def analyze(model, processor, clip: np.ndarray, start_sec: float, end_sec: float) -> str:
    pre  = trigger_time - start_sec
    post = end_sec - trigger_time

    prompt = (
        f"This is a {end_sec - start_sec:.0f}-second basketball video clip "
        f"from the {CAMERA_TYPE} camera ({start_sec:.1f}s – {end_sec:.1f}s).\n"
        f"The system-marked event trigger_time is {trigger_time}s, "
        f"which is {pre:.0f}s into this clip (pre={pre:.0f}s / post={post:.0f}s).\n\n"
        "Describe what you actually see in the video using this structure:\n"
        "[Before] What is happening before trigger_time? Where is the ball? What are the players doing?\n"
        "[At] What is the key action at/around trigger_time? (shot attempt, pass, foul, etc.) What is the ball trajectory?\n"
        "[After] What is the outcome? (scored / missed / out of bounds / foul) How does possession change?\n\n"
        "Only describe what is visible in the video. Do not guess or infer."
    )

    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": prompt},
    ]}]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], videos=[clip], fps=SAMPLE_FPS, padding=True, return_tensors="pt")

    print(f"\nRunning inference on {clip.shape[0]} frames ...")
    t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=600, do_sample=False, temperature=None, top_p=None)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    result  = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Done in {time.time() - t0:.1f}s")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Video  : {VIDEO_PATH.name}")
    print(f"  Trigger: {trigger_time}s  |  Window: [{trigger_time - PRE_SECONDS}s, {trigger_time + POST_SECONDS}s]")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        print(f"ERROR: video not found: {VIDEO_PATH}")
        sys.exit(1)

    start_sec = float(trigger_time - PRE_SECONDS)
    end_sec   = float(trigger_time + POST_SECONDS)

    clip = extract_clip(VIDEO_PATH, start_sec, end_sec)
    if clip.shape[0] == 0:
        print("ERROR: no frames extracted. Check trigger_time is within video duration.")
        sys.exit(1)

    model, processor = load_model()
    result = analyze(model, processor, clip, start_sec, end_sec)

    print("\n" + "=" * 60)
    print(f"  Result  |  CAM6 ({CAMERA_TYPE})  |  {start_sec:.1f}s – {end_sec:.1f}s")
    print("=" * 60)
    print(result)
    print("=" * 60)

    out_dir  = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"analysis_cam6_t{trigger_time}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"video      : {VIDEO_PATH}\n")
        f.write(f"trigger    : {trigger_time}s\n")
        f.write(f"window     : {start_sec:.1f}s – {end_sec:.1f}s\n")
        f.write(f"camera     : {CAMERA_TYPE}\n")
        f.write(f"sample_fps : {SAMPLE_FPS}\n")
        f.write(f"frames     : {clip.shape[0]}\n")
        f.write("\n" + "=" * 60 + "\nResult:\n" + "=" * 60 + "\n")
        f.write(result + "\n")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()

