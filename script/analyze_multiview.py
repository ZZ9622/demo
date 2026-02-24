"""
多机位视频片段分析：截取视频 → 抽帧 → 用视觉语言模型生成描述 → 写 multiview_analysis.json
使用 Qwen2-VL-2B-Instruct，与 transformers 原生兼容，无需打补丁。
依赖: pip install transformers qwen-vl-utils
"""
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
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CAMERAS_FILE = OUTPUT_DIR / "camerasurls.json"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

CAMERA_TYPES = {
    1: "baseline", 2: "sideline", 3: "sideline", 4: "baseline",
    5: "mobile", 6: "mobile", 7: "overhead",
}


def _get_hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
            print("🔄 正在加载 Qwen2-VL-2B 模型...")
            token = _get_hf_token()
            dtype = torch.float32 if self.device == "cpu" else (torch.float16 if self.device == "mps" else "auto")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                token=token,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
            self.available = True
            print(f"✅ 模型已加载 (设备: {self.device})")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def extract_video_segment(self, camera_id, start_time, end_time):
        camera = next(c for c in self.cameras if c["id"] == camera_id)
        out = os.path.join(self.temp_dir, f"cam_{camera_id}_{start_time}.mp4")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2",
        ]
        if "huggingface.co" in camera["url"]:
            t = _get_hf_token()
            if t:
                cmd += ["-headers", f"Authorization: Bearer {t}"]
        cmd += [
            "-ss", str(start_time), "-i", camera["url"],
            "-t", str(end_time - start_time), "-c", "copy", out,
        ]
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
        if n > 0:
            for i in np.linspace(0, n - 1, num_frames, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if ok:
                    frame = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def analyze_frame(self, camera_id, image: Image.Image) -> str:
        if not self.available or self.model is None or self.processor is None:
            return "模型不可用"
        camera_type = CAMERA_TYPES.get(camera_id, "unknown")
        prompt = f"分析这个{camera_type}机位的篮球比赛画面。请简要描述：球员位置、动作、球的位置、是否有投篮或得分等关键信息。"
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if process_vision_info is not None:
                image_list, video_list = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_list,
                    videos=video_list,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                )
            inputs = inputs.to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=256)
            start = inputs.input_ids.shape[1]
            gen = out[:, start:]
            text_out = self.processor.batch_decode(
                gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return (text_out[0] or f"{camera_type}机位分析完成").strip()
        except Exception as e:
            return f"分析错误: {e}"

    def analyze_trigger(self, trigger_time: int):
        print(f"分析时间: {trigger_time}s")
        start = trigger_time - 5
        end = trigger_time + 5
        results = {}
        for camera_id in [1]:
            video_path = self.extract_video_segment(camera_id, start, end)
            if not video_path:
                results[f"camera_{camera_id}"] = {
                    "camera_id": camera_id,
                    "camera_type": CAMERA_TYPES.get(camera_id, "unknown"),
                    "analysis": "视频提取失败",
                }
                continue
            frames = self.extract_key_frames(video_path, num_frames=1)
            if not frames:
                results[f"camera_{camera_id}"] = {
                    "camera_id": camera_id,
                    "camera_type": CAMERA_TYPES.get(camera_id, "unknown"),
                    "analysis": "无法提取视频帧",
                }
                continue
            analysis = self.analyze_frame(camera_id, frames[0])
            results[f"camera_{camera_id}"] = {
                "camera_id": camera_id,
                "camera_type": CAMERA_TYPES.get(camera_id, "unknown"),
                "analysis": analysis,
            }
            print(f"摄像头 {camera_id}: 完成")
        return {
            "trigger_time": trigger_time,
            "analysis_period": {"start": start, "end": end},
            "camera_analyses": results,
        }

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    analyzer = MultiViewAnalyzer()
    results = analyzer.analyze_trigger(6654)
    out_path = OUTPUT_DIR / "multiview_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果保存到: {out_path}")
    analyzer.cleanup()


if __name__ == "__main__":
    main()
