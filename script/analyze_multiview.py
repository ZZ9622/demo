"""多机位视频片段分析"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CAMERAS_FILE = OUTPUT_DIR / "camerasurls.json"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

CAMERA_TYPES = {
    1: "baseline", 2: "sideline", 3: "sideline", 4: "baseline",
    5: "mobile", 6: "mobile", 7: "overhead",
}


def _get_hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class MultiViewAnalyzer:
    def __init__(self):
        with open(CAMERAS_FILE, "r", encoding="utf-8") as f:
            self.cameras = json.load(f)
        self.temp_dir = tempfile.mkdtemp()
        self.model = None
        self.processor = None
        self.device = _get_device()
        self.available = False

        try:
            import warnings
            import os
            # 完全抑制模型加载警告
            warnings.filterwarnings("ignore")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            
            token = _get_hf_token()
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            model_kwargs = {
                "torch_dtype": dtype,
                "token": token,
                "low_cpu_mem_usage": True,
            }
            
            # 针对 RTX A1000 4GB 显存优化：使用 CPU 卸载
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"  # 自动分配，允许 CPU 卸载
                model_kwargs["max_memory"] = {0: "3GB", "cpu": "16GB"}  # 限制 GPU 使用
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    pass
            else:
                model_kwargs["device_map"] = "cpu"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
                
                # 使用快速处理器，明确设置以避免警告
                self.processor = AutoProcessor.from_pretrained(
                    MODEL_ID, 
                    token=token,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1280 * 28 * 28,
                    use_fast=True  # 明确启用快速处理器
                )
            self.available = True
        except Exception:
            pass

    def extract_video_segment(self, camera_id, start_time, end_time):
        camera = next(c for c in self.cameras if c["id"] == camera_id)
        out = os.path.join(self.temp_dir, f"cam_{camera_id}_{start_time}.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        
        if "huggingface.co" in camera["url"] and _get_hf_token():
            cmd += ["-headers", f"Authorization: Bearer {_get_hf_token()}"]
        
        cmd += ["-ss", str(start_time), "-i", camera["url"], "-t", str(end_time - start_time), "-c", "copy", out]
        
        try:
            subprocess.run(cmd, check=True)
            return out
        except subprocess.CalledProcessError:
            return None

    def extract_key_frames(self, video_path, num_frames=1):
        if not video_path:
            return []
        
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        for i in np.linspace(0, n - 1, num_frames, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames

    def analyze_frame(self, camera_id, image: Image.Image) -> str:
        if not self.available:
            return "模型不可用"
        
        camera_type = CAMERA_TYPES.get(camera_id, "unknown")
        prompt = f"分析这个{camera_type}机位的篮球比赛画面。请简要描述：球员位置、动作、球的位置、是否有投篮或得分等关键信息。"
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            else:
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            text_out = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            return text_out[0].strip() if text_out[0] else f"{camera_type}机位分析完成"
        except Exception as e:
            return f"分析错误: {e}"

    def analyze_trigger(self, trigger_time: int):
        start, end = trigger_time - 5, trigger_time + 5
        results = {}
        
        for camera_id in [1]:
            video_path = self.extract_video_segment(camera_id, start, end)
            frames = self.extract_key_frames(video_path) if video_path else []
            
            analysis = self.analyze_frame(camera_id, frames[0]) if frames else "视频处理失败"
            
            results[f"camera_{camera_id}"] = {
                "camera_id": camera_id,
                "camera_type": CAMERA_TYPES.get(camera_id, "unknown"),
                "analysis": analysis,
            }
        
        return {
            "trigger_time": trigger_time,
            "analysis_period": {"start": start, "end": end},
            "camera_analyses": results,
        }

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    trigger_time = 6654
    analyzer = MultiViewAnalyzer()
    results = analyzer.analyze_trigger(trigger_time)
    
    with open(OUTPUT_DIR / "multiview_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    analyzer.cleanup()
    print(f"分析完成，结果保存到 {OUTPUT_DIR / 'multiview_analysis.json'}")


if __name__ == "__main__":
    main()
