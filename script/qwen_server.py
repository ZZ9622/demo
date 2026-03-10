#!/usr/bin/env python3
# qwen_server.py

import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import logging as hf_logging

# 关闭多余的警告和进度条
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
hf_logging.set_verbosity_error()

# ==========================================
# 1. 全局配置与常驻内存变量
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# 先将其设为 None，等 FastAPI 启动时再把真实的权重挂载上来
MODEL = None
PROCESSOR = None

# 视频处理参数（采样率和分辨率控制）
SAMPLE_FPS = 4
MAX_SIDE = 768  

# 初始化 FastAPI 应用
app = FastAPI(title="Qwen2.5-VL Backend Server")

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    # Check GPU memory if CUDA is available
    device = "cpu"  # Default to CPU for stability
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Detected GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {gpu_mem:.1f} GB")
        
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
    
    # 如果 GPU 可用并且显存足够，直接用半精度加载到 GPU
    if device == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,      
            device_map="cuda",              
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,      
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=64  * 28 * 28,   
        max_pixels=256 * 28 * 28,   
        trust_remote_code=True
    )
    
    load_time = time.time() - t0
    print(f"  ✓ Model loaded successfully in {load_time:.1f}s")
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    actual_device = next(model.parameters()).device
    print(f"  Model parameters: {num_params:.1f}B")
    print(f"  Actual device: {actual_device}")

    print("  Compiling model (this may take a minute)...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"  Skip compile: {e}")
        
    return model, processor

# ── Video extraction (已精简：直接读取完整视频) ──────────────────────────────
def extract_clip(video_path: Path) -> np.ndarray:
    """读取传进来的短视频的所有帧，按 SAMPLE_FPS 抽帧，无需再按时间裁剪。"""
    print(f"  Opening video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps if native_fps > 0 else 0
    print(f"  Video specs: {native_fps:.1f} fps | {total_frames} frames | {duration:.1f}s")

    # 计算抽帧步长
    step = max(1, int(round(native_fps / SAMPLE_FPS)))
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 按步长抽帧
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # 缩放逻辑
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            frames.append(rgb)
        idx += 1

    cap.release()
    clip = np.stack(frames, axis=0) if frames else np.empty((0, MAX_SIDE, MAX_SIDE, 3), dtype=np.uint8)
    
    print(f"  ✓ Extracted {clip.shape[0]} frames @ {SAMPLE_FPS} fps")
    if clip.shape[0] > 0:
        print(f"  Resolution: {clip.shape[2]}×{clip.shape[1]}")
    return clip

# ── Inference (已精简：移除无用时间计算) ──────────────────────────────────────
def analyze(model, processor, clip: np.ndarray) -> str:
    prompt = """
    You are a video grounding assistant.

    任务：
    对视频中的确实进球的投篮，用自然语言描述画面信息。

    要求：
    只描述动作，不要添加无法直接识别的细节

    输出形式：
    [Before]:描述投篮的准备动作\n
    [During]:描述出手瞬间的动作和人员空间站位\n(扣篮/投篮，在三分线内/上/外)
    [After]:描述投篮结果\n
    """

    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": prompt},
    ]}]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], videos=[clip], fps=SAMPLE_FPS, padding=True, return_tensors="pt")
    
    device = next(model.parameters()).device
    
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
                max_new_tokens=600,  
                do_sample=False, 
                temperature=0.3, 
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=False  
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        result  = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        inference_time = time.time() - t0
        print(f"  ✓ Inference completed in {inference_time:.1f}s")
        print(f"  Generated {len(result.split())} words")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  GPU memory error: {e}")
        raise
    except Exception as e:
        print(f"  Inference error: {e}")
        raise

# ── API 服务端点 ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global MODEL, PROCESSOR
    print("🚀 正在启动 Qwen 服务，加载模型（仅加载一次）...")
    MODEL, PROCESSOR = load_model()
    print("✅ 模型加载完毕，服务就绪！")

# 剔除掉了 start_sec 和 end_sec
class VideoRequest(BaseModel):
    video_path: str

@app.post("/analyze")
async def analyze_video(req: VideoRequest):
    try:
        t0 = time.time()
        print(f"收到请求: {req.video_path}")
        
        # 只传路径，提取完整视频特征
        clip = extract_clip(Path(req.video_path))
        
        if clip.shape[0] == 0:
            raise ValueError("无法从该视频中提取到任何帧！")
        
        # 调用分析时也仅传 clip
        result_text = analyze(MODEL, PROCESSOR, clip)
        
        # 显式清理缓存
        torch.cuda.empty_cache() 
        
        return {"status": "success", "result": result_text, "time_cost": time.time() - t0}
    except Exception as e:
        # 这里把错误打印在服务端终端里，方便排错
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # 开启热重载模式！
    # 注意：开启 reload=True 时，第一个参数必须是字符串 "文件名:应用名"
    uvicorn.run(
        "qwen_server:app",  # 假设你的文件名为 qwen_server.py
        host="127.0.0.1", 
        port=8000,
        reload=True         # 【核心修改】开启代码热更新
    )