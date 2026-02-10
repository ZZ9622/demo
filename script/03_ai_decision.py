import numpy as np
import json
import os
import cv2
import torch
import opentimelineio as otio
from transformers import AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

# --- 配置 ---
BASE_DIR = "/home/SONY/s7000043396/Downloads/demo/script"
DATA_DIR = os.path.join(BASE_DIR, "../data/demo-data")
MOSAIC_VIDEO = os.path.join(BASE_DIR, "mosaic_preview.mp4")

# 模型配置
SIGLIP_ID = "google/siglip-so400m-patch14-384"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct" 

def load_models():
    print("loading dual model system...")
    
    # 1. 文本搜索模型 (SigLIP)
    search_model = AutoModel.from_pretrained(SIGLIP_ID, torch_dtype=torch.bfloat16).to("cuda")
    search_processor = AutoProcessor.from_pretrained(SIGLIP_ID)
    
    # 2. 视觉推理模型 (Qwen2.5-VL)
    # 使用 Flash Attention 2 加速，自动分配显存
    director_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    director_processor = AutoProcessor.from_pretrained(QWEN_ID)
    
    return search_model, search_processor, director_model, director_processor

def find_timestamps(text_query, features, search_model, search_processor, metadata, top_k=3):
    """语义搜索：找出最符合 Query 的时间点"""
    print(f"searching for: '{text_query}'")
    inputs = search_processor(text=[text_query], return_tensors="pt", padding="max_length").to("cuda", dtype=torch.bfloat16)
    
    with torch.no_grad():
        text_emb = search_model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        
    # 计算相似度 (Cosine Similarity)
    # features 形状 [T, D], text_emb 形状 [1, D]
    feats_tensor = torch.tensor(features).to("cuda", dtype=torch.bfloat16)
    similarity = (feats_tensor @ text_emb.T).squeeze()
    
    # 取 Top K
    values, indices = torch.topk(similarity, top_k)
    results = []
    
    # 过滤相邻太近的时间点 (去重)
    last_idx = -999
    for idx in indices.cpu().numpy():
        idx = int(idx)
        if abs(idx - last_idx) > 2: # 至少间隔2秒
            results.append(metadata[idx])
            last_idx = idx
            
    return results

def ask_ai_director(director_model, director_processor, frame_img, prompt_goal):
    """视觉推理：让 Qwen 看图选机位"""
    
    prompt = f"""You are a professional sports broadcast director. 
    Below is a 4x2 grid view of 8 cameras monitoring a basketball game.
    Layout:
    [Cam 0] [Cam 1] [Cam 2] [Cam 3]
    [Cam 4] [Cam 5] [Cam 6] [Cam 7]
    
    Goal: Pick the single BEST camera ID (0-7) that shows: "{prompt_goal}".
    Criteria: Choose the clearest view, best angle, and least obstruction.
    
    Output ONLY the number (0-7).
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = director_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = director_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = director_model.generate(**inputs, max_new_tokens=10)
        
    output_text = director_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 提取数字
    import re
    match = re.search(r'\d+', output_text)
    if match:
        return int(match.group())
    return 0 # 默认 fallback

def main_pipeline():
    # 1. 准备数据
    features = np.load(os.path.join(BASE_DIR, "Mosaic_preview_features.npy"))
    with open(os.path.join(BASE_DIR, "feature_metadata.json"), "r") as f:
        metadata = json.load(f)
    with open(os.path.join(BASE_DIR, "Camera_Layout.json"), "r") as f:
        layout = json.load(f)
        
    # 2. 加载模型
    s_model, s_proc, d_model, d_proc = load_models()
    
    # 3. 定义你想找的高光时刻
    user_prompts = [
        "A player performing a slam dunk",  # 找扣篮
        "Players fighting for a rebound",   # 找篮板
        "A close-up view of the players"    # 找特写
    ]
    
    timeline = otio.schema.Timeline(name="RTX5090_AI_Edit")
    track = otio.schema.Track(name="Main", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)
    
    cap = cv2.VideoCapture(MOSAIC_VIDEO)

    for p_text in user_prompts:
        print(f"\nprocessing instruction: {p_text}")
        
        # A. 找时间 (Search)
        timestamps = find_timestamps(p_text, features, s_model, s_proc, metadata)
        
        for ts_data in timestamps:
            sec = ts_data['timestamp_sec']
            print(f"  -> locked time: {sec} seconds (LTC: {ts_data['ltc']})")
            
            # B. 截取该秒的画面
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret: continue
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # C. 选机位 (Reasoning)
            best_cam = ask_ai_director(d_model, d_proc, frame_pil, p_text)
            best_cam = max(0, min(7, best_cam)) # 安全限制
            
            print(f"  -> AI director selected camera: {best_cam}")
            
            # D. 添加到剪辑表
            filename = layout["mapping"][best_cam]["file"]
            media_ref = otio.schema.ExternalReference(target_url=os.path.join(DATA_DIR, filename))
            
            clip = otio.schema.Clip(
                name=f"Event_{best_cam}",
                media_reference=media_ref,
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(sec * 30, 30),
                    duration=otio.opentime.RationalTime(60, 30) # 剪辑 2 秒
                )
            )
            track.append(clip)
            
    cap.release()
    otio.adapters.write_to_file(timeline, os.path.join(BASE_DIR, "timeline.otio"))
    print("\nOTIO editing decision table generated: timeline.otio")

if __name__ == "__main__":
    main_pipeline()