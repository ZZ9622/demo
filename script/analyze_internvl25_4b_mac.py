#!/Volumes/T7/venv/bin/python
"""
analyze_internvl25_4b_mac.py
-----------------------------
Analyze a local basketball video clip around a trigger_time.
Model  : OpenGVLab/InternVL2_5-4B
Device : CPU  (avoids MPS 4 GB single-tensor hard limit on M3 Pro)
Window : trigger_time - 6s  ->  trigger_time + 4s
Video  : camera6_h264.mp4   (rightbaseline camera)
"""

import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 模型下载/加载缓存存到 T7 硬盘（需在 import transformers 之前设置）
os.environ["HF_HOME"] = "/Volumes/T7/hf_cache"

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH   = Path("/Volumes/T7/demo/script/camera6_h264.mp4")
MODEL_ID     = "OpenGVLab/InternVL2_5-4B"
CAMERA_TYPE  = "rightbaseline"

trigger_time = 6654   # event timestamp to analyze (seconds)
PRE_SECONDS  = 6      # seconds before trigger_time
POST_SECONDS = 4      # seconds after trigger_time
NUM_FRAMES   = 10     # total frames to sample from the window
INPUT_SIZE   = 448    # InternVL standard tile size

# ImageNet normalization (required by InternVL vision encoder)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ── Image preprocessing (InternVL standard pipeline) ─────────────────────────
def build_transform():
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert one PIL frame to a (1, 3, 448, 448) float32 tensor."""
    return build_transform()(pil_img).unsqueeze(0)


# ── Video frame extraction ────────────────────────────────────────────────────
def extract_frames(video_path: Path, start_sec: float, end_sec: float, n: int) -> list[Image.Image]:
    """
    Uniformly sample n frames from [start_sec, end_sec] in the video.
    Returns a list of PIL.Image (RGB).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps
    print(f"Video: {native_fps:.1f} fps | {total_frames} frames | {duration:.1f}s")

    start_sec = max(0.0, start_sec)
    end_sec   = min(duration, end_sec)

    start_f = int(start_sec * native_fps)
    end_f   = int(end_sec   * native_fps)
    indices = np.linspace(start_f, end_f, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    print(f"Extracted {len(frames)} frames  [{start_sec:.1f}s – {end_sec:.1f}s]")
    return frames


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print(f"\nLoading {MODEL_ID} on CPU ...")
    t0 = time.time()

    # low_cpu_mem_usage=False 避免 meta tensor 路径，否则 InternVL 在 __init__ 里 .item() 会报错
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,         # bfloat16 / float16 not reliable on CPU
        low_cpu_mem_usage=False,
        use_flash_attn=False,        # flash-attn requires CUDA
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────
def analyze(model, tokenizer, frames: list[Image.Image], start_sec: float, end_sec: float) -> str:
    """
    Build InternVL-style video prompt and run model.chat().
    Each frame is sent as a separate <image> in 'Frame{i}: <image>' format.
    """
    # Stack all frame tensors: (N, 3, 448, 448)
    pixel_values = torch.cat([pil_to_tensor(f) for f in frames], dim=0)
    num_patches_list = [1] * len(frames)  # 1 tile per frame (max_num=1 for video)

    pre  = trigger_time - start_sec
    post = end_sec - trigger_time

    # InternVL video prompt format: "Frame1: <image>\nFrame2: <image>\n...{question}"
    frame_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(frames))])
    question = frame_prefix + (
        f"These are {len(frames)} uniformly sampled frames from the {CAMERA_TYPE} camera, "
        f"covering {start_sec:.1f}s – {end_sec:.1f}s ({end_sec - start_sec:.0f} seconds total).\n"
        f"The system-marked trigger_time is {trigger_time}s, "
        f"which is {pre:.0f}s into this clip (pre={pre:.0f}s / post={post:.0f}s).\n\n"
        "Describe what you see using this structure:\n"
        "[Before] What is happening before trigger_time? Where is the ball? What are the players doing?\n"
        "[At] What is the key action at/around trigger_time? (shot, pass, foul, etc.) Describe the ball trajectory.\n"
        "[After] What is the outcome? (scored / missed / out of bounds / foul) How does possession change?\n\n"
        "Only describe what is visible in the frames. Do not guess or infer."
    )

    generation_config = dict(max_new_tokens=600, do_sample=False)

    print(f"\nRunning inference on {len(frames)} frames ...")
    t0 = time.time()

    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )

    print(f"Done in {time.time() - t0:.1f}s")
    return response


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Model  : {MODEL_ID}")
    print(f"  Video  : {VIDEO_PATH.name}")
    print(f"  Trigger: {trigger_time}s  |  Window: [{trigger_time - PRE_SECONDS}s, {trigger_time + POST_SECONDS}s]")
    print(f"  Frames : {NUM_FRAMES}")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        print(f"ERROR: video not found: {VIDEO_PATH}")
        sys.exit(1)

    start_sec = float(trigger_time - PRE_SECONDS)
    end_sec   = float(trigger_time + POST_SECONDS)

    frames = extract_frames(VIDEO_PATH, start_sec, end_sec, NUM_FRAMES)
    if not frames:
        print("ERROR: no frames extracted.")
        sys.exit(1)

    model, tokenizer = load_model()
    result = analyze(model, tokenizer, frames, start_sec, end_sec)

    print("\n" + "=" * 60)
    print(f"  Result  |  CAM6 ({CAMERA_TYPE})  |  {start_sec:.1f}s – {end_sec:.1f}s")
    print("=" * 60)
    print(result)
    print("=" * 60)

    out_dir  = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"internvl_cam6_t{trigger_time}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"model      : {MODEL_ID}\n")
        f.write(f"video      : {VIDEO_PATH}\n")
        f.write(f"trigger    : {trigger_time}s\n")
        f.write(f"window     : {start_sec:.1f}s – {end_sec:.1f}s\n")
        f.write(f"camera     : {CAMERA_TYPE}\n")
        f.write(f"frames     : {len(frames)}\n")
        f.write("\n" + "=" * 60 + "\nResult:\n" + "=" * 60 + "\n")
        f.write(result + "\n")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
