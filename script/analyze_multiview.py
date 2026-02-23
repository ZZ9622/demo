import json
import os
import cv2
import numpy as np
import subprocess
import tempfile
from PIL import Image
import requests
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CAMERAS_FILE = os.path.join(OUTPUT_DIR, "camerasurls.json")


<<<<<<< HEAD
<<<<<<< HEAD
# Qwen2-VL-7B-Instruct 模型配置
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

=======
# Qwen2-VL-2B-Instruct 模型配置
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
>>>>>>> 1c4320d (add yolo hl detection file)
=======
# Qwen2-VL-2B-Instruct 模型配置
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
=======
# Qwen2-VL-7B-Instruct 模型配置
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)

CAMERA_TYPES = {
    1: "baseline", 2: "sideline", 3: "sideline", 4: "baseline",
    5: "mobile", 6: "mobile", 7: "overhead"
}

class MultiViewAnalyzer:
    def __init__(self):
        with open(CAMERAS_FILE, 'r') as f:
            self.cameras = json.load(f)
        self.temp_dir = tempfile.mkdtemp()
        
<<<<<<< HEAD
<<<<<<< HEAD
        # 初始化Qwen2-VL-7B-Instruct模型
        try:
            print("🔄 正在加载Qwen2-VL-7B-Instruct模型...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="sdpa"  # 强制使用 PyTorch 内置优化，跳过 FlashAttention 检测
            )
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.qwen_available = True
            print("✅ Qwen2-VL-7B-Instruct模型已加载")
=======
=======
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
        # 初始化Qwen2-VL-2B-Instruct模型
        try:
            print("🔄 正在加载Qwen2-VL-2B-Instruct模型...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype="auto",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.qwen_available = True
            print("✅ Qwen2-VL-2B-Instruct模型已加载")
<<<<<<< HEAD
>>>>>>> 1c4320d (add yolo hl detection file)
=======
=======
        # 初始化Qwen2-VL-7B-Instruct模型
        try:
            print("🔄 正在加载Qwen2-VL-7B-Instruct模型...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="sdpa"  # 强制使用 PyTorch 内置优化，跳过 FlashAttention 检测
            )
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.qwen_available = True
            print("✅ Qwen2-VL-7B-Instruct模型已加载")
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
        except Exception as e:
            self.model = None
            self.processor = None
            self.qwen_available = False
            print(f"❌ 模型加载失败: {str(e)}")
    
    def extract_video_segment(self, camera_id, start_time, end_time):
        camera = next(c for c in self.cameras if c['id'] == camera_id)
        output_file = os.path.join(self.temp_dir, f"cam_{camera_id}_{start_time}.mp4")
        
<<<<<<< HEAD
<<<<<<< HEAD
        # 不需要HF_TOKEN，因为这些是公开数据集
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
=======
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-headers', f'Authorization: Bearer {HF_TOKEN}',
>>>>>>> 1c4320d (add yolo hl detection file)
=======
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-headers', f'Authorization: Bearer {HF_TOKEN}',
=======
        # 不需要HF_TOKEN，因为这些是公开数据集
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
            '-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '2',
            '-ss', str(start_time), '-i', camera['url'],
            '-t', str(end_time - start_time), '-c', 'copy', output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            return output_file
        except:
            return None
    
    def extract_key_frames(self, video_path, num_frames=3):
        if not video_path:
            return []
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (480, 270))
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb_frame))
        
        cap.release()
        return frames
    
    def analyze_camera_with_qwen(self, camera_id, video_path):
<<<<<<< HEAD
<<<<<<< HEAD
        """使用Qwen2-VL-7B-Instruct分析单个摄像头的视频片段"""
        if not self.qwen_available or not video_path or not self.model:
            return "Qwen2-VL-7B-Instruct不可用或视频文件无效"
=======
=======
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
        """使用Qwen2-VL-2B-Instruct分析单个摄像头的视频片段"""
        if not self.qwen_available or not video_path or not self.model:
            return "Qwen2-VL-2B-Instruct不可用或视频文件无效"
        
        # 提取关键帧
        frames = self.extract_key_frames(video_path, num_frames=1)
        if not frames:
            return "无法提取视频帧"
<<<<<<< HEAD
>>>>>>> 1c4320d (add yolo hl detection file)
=======
=======
        """使用Qwen2-VL-7B-Instruct分析单个摄像头的视频片段"""
        if not self.qwen_available or not video_path or not self.model:
            return "Qwen2-VL-7B-Instruct不可用或视频文件无效"
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
        
        camera_type = CAMERA_TYPES.get(camera_id, "unknown")
        
        try:
<<<<<<< HEAD
<<<<<<< HEAD
            # 提取视频帧作为图像列表
            frames = self.extract_key_frames(video_path, num_frames=8)
            if not frames:
                return "无法提取视频帧"
            
            # 构建Qwen2-VL的视频消息格式
=======
=======
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
            # 使用第一帧进行分析
            image = frames[0]
            
            # 按照Qwen2-VL格式构建消息
<<<<<<< HEAD
>>>>>>> 1c4320d (add yolo hl detection file)
=======
=======
            # 提取视频帧作为图像列表
            frames = self.extract_key_frames(video_path, num_frames=8)
            if not frames:
                return "无法提取视频帧"
            
            # 构建Qwen2-VL的视频消息格式
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
            messages = [
                {
                    "role": "user",
                    "content": [
<<<<<<< HEAD
<<<<<<< HEAD
                        {
                            "type": "video",
                            "video": [frame for frame in frames],  # 传入PIL图像列表
                            "fps": 1.0,
                        },
                        {
                            "type": "text", 
                            "text": f"分析这个{camera_type}机位的篮球比赛视频片段。请详细描述你看到的内容，包括：球员位置、动作、球的位置、是否有投篮或得分等关键事件。"
                        }
                    ],
=======
                        {"type": "image"},
                        {"type": "text", "text": f"分析这个{camera_type}机位的篮球比赛画面。请详细描述你看到的内容，包括：球员位置、动作、球的位置、是否有投篮或得分等关键事件。"}
                    ]
>>>>>>> 1c4320d (add yolo hl detection file)
=======
                        {"type": "image"},
                        {"type": "text", "text": f"分析这个{camera_type}机位的篮球比赛画面。请详细描述你看到的内容，包括：球员位置、动作、球的位置、是否有投篮或得分等关键事件。"}
                    ]
=======
                        {
                            "type": "video",
                            "video": [frame for frame in frames],  # 传入PIL图像列表
                            "fps": 1.0,
                        },
                        {
                            "type": "text", 
                            "text": f"分析这个{camera_type}机位的篮球比赛视频片段。请详细描述你看到的内容，包括：球员位置、动作、球的位置、是否有投篮或得分等关键事件。"
                        }
                    ],
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
                }
            ]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 处理输入
            inputs = self.processor(
                text=[text],
<<<<<<< HEAD
<<<<<<< HEAD
                images=image_inputs,
=======
                images=[image],
>>>>>>> 1c4320d (add yolo hl detection file)
=======
                images=[image],
=======
                images=image_inputs,
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # 生成分析结果
            generated_ids = self.model.generate(**inputs, max_new_tokens=200)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # 解码输出
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
<<<<<<< HEAD
<<<<<<< HEAD
            analysis = output_text[0] if output_text else f"{camera_type}机位视频分析完成"
=======
            analysis = output_text[0] if output_text else f"{camera_type}机位分析完成"
>>>>>>> 1c4320d (add yolo hl detection file)
=======
            analysis = output_text[0] if output_text else f"{camera_type}机位分析完成"
=======
            analysis = output_text[0] if output_text else f"{camera_type}机位视频分析完成"
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
            return analysis
                
        except Exception as e:
            return f"Qwen分析错误: {str(e)}"
    
    def analyze_trigger(self, trigger_time):
        print(f"分析时间: {trigger_time}s")
        
        video_start = trigger_time - 5
        video_end = trigger_time + 5
        
        camera_analyses = {}
        
        # 分析每个摄像头
<<<<<<< HEAD
<<<<<<< HEAD
        for camera_id in [1]:  # 先测试camera1，避免内存问题
=======
        for camera_id in [1]:  # 仅测试camera1
>>>>>>> 1c4320d (add yolo hl detection file)
=======
        for camera_id in [1]:  # 仅测试camera1
=======
        for camera_id in [1]:  # 先测试camera1，避免内存问题
>>>>>>> 4c1675a7d1d902d33dc07b1c34bff4e74108f624
>>>>>>> ea28d10 (chore: remove large weights and sensitive token)
            video_path = self.extract_video_segment(camera_id, video_start, video_end)
            
            if video_path:
                analysis = self.analyze_camera_with_qwen(camera_id, video_path)
                camera_analyses[f"camera_{camera_id}"] = {
                    "camera_id": camera_id,
                    "camera_type": CAMERA_TYPES.get(camera_id, "unknown"),
                    "analysis": analysis
                }
                print(f"摄像头 {camera_id}: 完成")
            else:
                camera_analyses[f"camera_{camera_id}"] = {
                    "camera_id": camera_id,
                    "camera_type": CAMERA_TYPES.get(camera_id, "unknown"),
                    "analysis": "视频提取失败"
                }
        
        return {
            "trigger_time": trigger_time,
            "analysis_period": {"start": video_start, "end": video_end},
            "camera_analyses": camera_analyses
        }
    
    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir)

def main():
    trigger_time = 6654  # 01:50:54
    analyzer = MultiViewAnalyzer()
    
    results = analyzer.analyze_trigger(trigger_time)
    
    output_file = os.path.join(OUTPUT_DIR, "multiview_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果保存到: {output_file}")
    analyzer.cleanup()

if __name__ == "__main__":
    main()