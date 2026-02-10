import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoModel, 
    SiglipImageProcessor, # 尽管没直接用，但保持引入以防万一
    SiglipTokenizer, 
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor
)
import opentimelineio as otio

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
VIDEO_DIR = os.path.join(PROJECT_DIR, "data/demo-data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_PATH = os.path.join(OUTPUT_DIR, "Mosaic_preview_features.npy")
MOSAIC_PREVIEW_DIR = os.path.join(OUTPUT_DIR, "mosaic_previews")  # 由 02 脚本写入
OUTPUT_OTIO = os.path.join(OUTPUT_DIR, "timeline.otio")

# --- 模型 ID ---
SIGLIP_ID = "google/siglip-so400m-patch14-384"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def load_models():
    print("🚀 [RTX 5090] loading dual model system...")
    
    # 1. 加载 SigLIP (避开 AutoProcessor Bug)
    s_model = AutoModel.from_pretrained(SIGLIP_ID, torch_dtype=torch.bfloat16).to("cuda")
    s_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_ID)
    
    # 2. 加载 Qwen2.5-VL (使用 SDPA 确保 5090 兼容性)
    # d_proc 用于处理 Qwen 的图像和文本输入
    d_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa",
        device_map="auto"
    )
    d_proc = AutoProcessor.from_pretrained(QWEN_ID)
    
    return s_model, s_tokenizer, d_model, d_proc

def find_highlight_moments(text_query, features, s_model, s_tokenizer):
    """语义搜索：找出最符合描述的秒数"""
    print(f"🔍 semantic search keyword: '{text_query}'")
    
    inputs = s_tokenizer([text_query], return_tensors="pt", padding="max_length").to("cuda")
    
    with torch.no_grad():
        outputs = s_model.get_text_features(**inputs)
        # 兼容性处理：提取 Pooler Output
        text_emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

    # 关键点：将特征转为 bfloat16 以匹配 5090 上的模型输出
    features_tensor = torch.from_numpy(features).to("cuda", dtype=torch.bfloat16)
    similarities = (features_tensor @ text_emb.T).squeeze(1)
    
    # 获取得分最高的前 K 个秒数索引
    # 这里我们只取得分最高的一个秒数，作为核心高光点
    _, index = torch.topk(similarities, k=1)
    return index.cpu().item() # 返回单个最佳秒数

def find_best_cam_with_qwen(mosaic_image_path, d_model, d_proc):
    if not os.path.exists(mosaic_image_path):
        print(f"❌ error: mosaic image {mosaic_image_path} not found")
        return "def2_cam_00.mp4"
        
    print(f"🧐 Qwen2.5-VL is analyzing: {mosaic_image_path}...")
    image = Image.open(mosaic_image_path).convert("RGB")
    
    prompt_text = (
        "In this 2x4 grid of camera views, which camera number provides the best close-up view "
        "of the slam dunk? Reply with ONLY the number (0-7)."
    )
    
    # 严格遵循 Transformers 要求的格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 1. 生成模板文本
    text = d_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 使用 Processor 同时处理图像和文本 (5090 加速关键)
    inputs = d_proc(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(d_model.device)

    # 3. 生成决策
    with torch.no_grad():
        # 注意：Qwen2.5-VL 生成时不需要手动传 input_ids，直接传 inputs 展开即可
        generated_ids = d_model.generate(**inputs, max_new_tokens=20)
        
        # 只需要获取新生成的 token
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = d_proc.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    print(f"🤖 Qwen 回答: {response_text}")

    # 4. 解析结果 (逻辑保持不变)
    cam_map = {
        "0": "def2_cam_00.mp4", "1": "def2_cam_08.mp4", "2": "def2_cam_15.mp4", "3": "def2_cam_23.mp4",
        "4": "def2_cam_45.mp4", "5": "def2_cam_51.mp4", "6": "def2_cam_66.mp4", "7": "def2_cam_73.mp4"
    }
    
    import re
    match = re.search(r"(\d)", response_text)
    if match:
        cam_key = match.group(1)
        if cam_key in cam_map:
            return cam_map[cam_key]
    
    return "def2_cam_00.mp4"

def create_clean_otio(decisions, output_path, fps=25.0):
    """生成标准 OTIO 时间轴，确保时长严格为 9 秒"""
    timeline = otio.schema.Timeline(name="RTX5090_AI_Edit")
    track = otio.schema.Track(name="Main", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)

    for d in decisions:
        # 计算帧数区间
        start_frame = d['start'] * fps
        duration_frames = (d['end'] - d['start']) * fps
        
        media_ref = otio.schema.ExternalReference(
            target_url=os.path.abspath(os.path.join(VIDEO_DIR, d['cam']))
        )
        
        clip = otio.schema.Clip(
            name=f"Segment_{d['start']}s",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(start_frame, fps),
                duration=otio.opentime.RationalTime(duration_frames, fps)
            )
        )
        track.append(clip)

    otio.adapters.write_to_file(timeline, output_path)

def main():
    if not os.path.exists(FEATURE_PATH):
        print("❌ error: feature file video_features.npy not found, please run 02 script first")
        return
    
    # 确保 mosaic_previews 目录存在
    if not os.path.exists(MOSAIC_PREVIEW_DIR):
        print(f"❌ error: mosaic preview directory {MOSAIC_PREVIEW_DIR} not found, please ensure 02 script has generated these preview images.")
        return

    # 1. 初始化
    features = np.load(FEATURE_PATH)
    s_model, s_tokenizer, d_model, d_proc = load_models()

    # 2. 语义搜索：找到“扣篮”发生的核心高光秒数
    query = "A player performing a slam dunk"
    # 这里我们只取语义搜索得分最高的“一秒”，作为 AI 重点关注的高光时间点
    highlight_second = find_highlight_moments(query, features, s_model, s_tokenizer)
    print(f"✅ SigLIP detected core highlight seconds: {highlight_second}")

    # 3. 导演决策逻辑：将 9 秒划分为 1 秒一个的区间进行 AI 决策
    # 这种“坑位”逻辑保证了视频时长绝对不会翻倍
    total_duration_seconds = 9
    final_decisions = []
    
    for current_second in range(total_duration_seconds):
        mosaic_image_file = os.path.join(MOSAIC_PREVIEW_DIR, f"mosaic_preview_{current_second:04d}.png")
        
        # 默认机位
        best_cam = "def2_cam_00.mp4" 

        # 核心 AI 决策：
        # 如果当前秒是核心高光秒，或者是在其附近的一秒，
        # 则调用 Qwen2.5-VL 来分析并选择最佳机位。
        # 否则（非高光时刻），使用默认全景机位。
        if abs(current_second - highlight_second) <= 1: # 高光前后各一秒也进行分析
            print(f"✨ entering AI director mode: analyzing highlight/nearby scene at {current_second} second...")
            best_cam = find_best_cam_with_qwen(mosaic_image_file, d_model, d_proc)
        else:
            print(f"Skip AI analysis for second {current_second}, using default cam: {best_cam}")

        final_decisions.append({
            "start": current_second,
            "end": current_second + 1, # 每段时长1秒
            "cam": best_cam
        })

    # 4. 导出结果
    print(f"\n💾 exporting OTIO timeline to: {OUTPUT_OTIO}")
    create_clean_otio(final_decisions, OUTPUT_OTIO)

if __name__ == "__main__":
    main()