import json
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import requests
import urllib.parse
from huggingface_hub import hf_hub_download

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
from transformers import logging
logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CAMERAS_FILE = OUTPUT_DIR / "camerasurls.json"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

CAMERA_TYPES = {
    1: "leftbaseline", 2: "leftsideline", 3: "rightoverhead", 4: "leftsidemiddle",
    5: "leftoverhead", 6: "rightbaseline", 7: "lefthigh",
}
trigger_time = 6654

# 加载模型    
def load_model():
    print("loading model ... Qwen2.5-VL")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"model loaded: {MODEL_ID}")
    return model, processor

# 获取摄像头URL
def get_camera_url(camera_id: int):
    with open(CAMERAS_FILE, "r") as f:
        cameras = json.load(f)
    
    for camera in cameras:
        if camera["id"] == camera_id:
            return camera["url"]
    raise ValueError(f"未找到摄像头ID: {camera_id}")

# 从远程下载指定时间段的视频片段
def download_video_segment(video_url: str, start_time: int, duration: int, output_path: str):
    """
    下载指定时间段的视频片段，支持Hugging Face数据集
    Args:
        video_url: 远程视频URL
        start_time: 开始时间（秒）
        duration: 持续时间（秒）
        output_path: 输出文件路径
    """
    print(f"正在从远程视频下载片段: {start_time}s - {start_time + duration}s")
    print(f"源URL: {video_url}")
    
    # 检查是否为Hugging Face数据集URL
    if "huggingface.co/datasets" in video_url:
        print("检测到Hugging Face数据集，使用Hub下载...")
        
        # 解析Hugging Face URL
        # 格式: https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}
        parts = video_url.split('/')
        repo_id = f"{parts[4]}/{parts[5]}"  # 例如: SportsHL-Team/Sports_highlight_generation
        filename = '/'.join(parts[8:])  # 例如: apidis/camera6/camera6_full_stitched_interpolated.mp4
        
        print(f"数据集ID: {repo_id}")
        print(f"文件名: {filename}")
        
        try:
            # 下载完整视频文件到临时位置
            temp_full_video = output_path.replace('.mp4', '.full.mp4')
            print("正在从Hugging Face下载完整视频文件...")
            
            # 获取token
            token = os.environ.get('HF_TOKEN')
            if token:
                print("使用HF_TOKEN进行认证...")
            
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                cache_dir=None  # 使用默认缓存
            )
            
            print(f"文件已下载到: {downloaded_file}")
            
            # 使用ffmpeg提取指定时间段
            cmd = [
                'ffmpeg',
                '-i', downloaded_file,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',
                output_path
            ]
            
            print("正在提取视频片段...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"视频片段提取完成: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Hugging Face下载失败: {e}")
            print("尝试直接使用ffmpeg...")
    
    # 对于非Hugging Face URL或HF下载失败的情况，使用原来的ffmpeg方法
    cmd = [
        'ffmpeg',
        '-i', video_url,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # 使用流复制，速度更快
        '-avoid_negative_ts', 'make_zero',
        '-y',  # 覆盖输出文件
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"视频片段下载完成: {output_path}")
    return output_path

# 下载视频片段并返回文件信息
def download_video_segment_for_analysis(video_url: str, trigger_time: int, time_range: int = 10):
    """
    从视频URL下载trigger_time前后指定时间范围的片段
    Args:
        video_url: 视频URL
        trigger_time: 触发时间点（秒）
        time_range: trigger_time前后的时间范围（秒），默认10秒
    Returns:
        tuple: (video_path, fps, cleanup_function) 
               视频路径, 帧率, 清理函数
    """
    start_time = max(0, trigger_time - time_range)  # 确保不小于0
    duration = time_range * 2  # 前后各time_range秒，总计2*time_range秒
    
    print(f"正在下载视频片段，trigger_time: {trigger_time}s, 范围: {start_time}s - {start_time + duration}s")
    
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp(prefix="video_analysis_")
    print(f"创建临时文件夹: {temp_dir}")
    
    # 生成临时视频文件名
    import urllib.parse
    original_filename = urllib.parse.urlparse(video_url).path.split('/')[-1]
    if not original_filename or original_filename == '/':
        original_filename = "video.mp4"
    
    # 确保文件名有.mp4扩展名
    if not original_filename.endswith('.mp4'):
        original_filename += '.mp4'
        
    temp_video_path = os.path.join(temp_dir, f"segment_{original_filename}")
    
    # 下载指定时间段的视频片段
    download_video_segment(video_url, start_time, duration, temp_video_path)
    
    # 获取视频FPS信息
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频片段: {temp_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()
    
    print(f"视频片段信息 - FPS: {fps:.2f}, 总帧数: {total_frames}, 时长: {video_duration:.2f}s")
    
    # 定义清理函数
    def cleanup():
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"已删除临时文件夹: {temp_dir}")
        except Exception as e:
            print(f"警告：删除临时文件夹失败: {e}")
    
    return temp_video_path, fps, cleanup

# 使用模型分析视频内容
def analyze_video_content(model, processor, video_path: str, camera_type: str, fps: float = 1.0):
    """
    使用Qwen2.5-VL模型分析视频内容
    Args:
        model: Qwen2.5-VL模型
        processor: 模型处理器
        video_path: 视频文件路径
        camera_type: 摄像头类型
        fps: 帧率采样参数
    """
    print(f"正在分析 {camera_type} 摄像头的视频内容...")
    
    # 构建提示词
    prompt = f"""请分析这个来自{camera_type}摄像头的视频片段（trigger_time前后10秒）：

1. 描述视频中发生的主要事件和动作
2. 识别视频中的关键对象（人、球、设备等）
3. 分析场景的整体情况和背景
4. 如果是体育场景，请描述比赛状况和关键时刻
5. 评估这个时间段是否包含重要或精彩的内容

请用中文详细描述你看到的内容。"""

    # 使用视频消息格式，直接传入视频文件路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # 处理输入 - 使用兼容的API
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # 生成回答
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def main():
    """
    主函数 - 直接分析CAM6视频
    """
    print("开始视频内容分析系统...")
    
    camera_id = 6
    print(f"\n=== 开始分析摄像头 CAM{camera_id} ===")
    
    try:
        # 1. 加载模型
        model, processor = load_model()
        
        # 2. 获取CAM6的URL
        camera_url = get_camera_url(camera_id)
        camera_type = CAMERA_TYPES.get(camera_id, f"camera{camera_id}")
        
        print(f"摄像头类型: {camera_type}")
        print(f"视频URL: {camera_url}")
        
        # 3. 下载视频片段 (trigger_time 前后10秒)
        video_path, video_fps, cleanup_func = download_video_segment_for_analysis(
            camera_url, trigger_time, time_range=10
        )
        
        # 4. 分析视频内容
        analysis_result = analyze_video_content(
            model, processor, video_path, camera_type, fps=1.0
        )
        
        # 5. 输出结果
        print(f"\n=== CAM{camera_id} ({camera_type}) 分析结果 ===")
        print(f"时间段: {trigger_time - 10}s - {trigger_time + 10}s (trigger_time: {trigger_time}s)")
        print(f"原始视频FPS: {video_fps:.2f}")
        print("\n视频内容分析:")
        print("-" * 50)
        print(analysis_result)
        print("-" * 50)
        
        # 6. 清理临时文件
        cleanup_func()
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 确保清理临时文件
        try:
            if 'cleanup_func' in locals():
                cleanup_func()
        except:
            pass
    
    print("\n分析完成！")
if __name__ == "__main__":
    main()