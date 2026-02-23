import json
import os
import cv2
import numpy as np
import subprocess
import tempfile
import google.genai as genai
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CAMERAS_FILE = os.path.join(OUTPUT_DIR, "camerasurls.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"  # 使用最新可用的模型
HF_TOKEN = "xx"  # Hugging Face token

CAMERA_TYPES = {
    1: "baseline", 2: "sideline", 3: "sideline", 4: "baseline",
    5: "mobile", 6: "mobile", 7: "overhead"
}

class MultiViewAnalyzer:
    def __init__(self):
        with open(CAMERAS_FILE, 'r') as f:
            self.cameras = json.load(f)
        self.temp_dir = tempfile.mkdtemp()
        
        if GEMINI_API_KEY:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                self.gemini_available = True
                print("✅ Gemini客户端已初始化")
            except Exception as e:
                self.gemini_available = False
                print(f"❌ Gemini初始化失败: {e}")
        else:
            self.gemini_available = False
    
    def extract_video_segment(self, camera_id, start_time, end_time):
        camera = next(c for c in self.cameras if c['id'] == camera_id)
        output_file = os.path.join(self.temp_dir, f"cam_{camera_id}_{start_time}.mp4")
        
        # 使用Hugging Face token进行认证
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-headers', f'Authorization: Bearer {HF_TOKEN}',
            '-ss', str(start_time), '-i', camera['url'],
            '-t', str(end_time - start_time), '-c', 'copy', output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            return output_file
        except Exception as e:
            print(f"视频提取失败 (摄像头 {camera_id}): {e}")
            return None
    
    def analyze_basket_deformation(self, video_path):
        if not video_path:
            return False, 0.0
        
        cap = cv2.VideoCapture(video_path)
        motion_scores = []
        ret, prev_frame = cap.read()
        
        if ret:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, gray)
                motion_scores.append(np.mean(diff))
                prev_gray = gray
        
        cap.release()
        
        if motion_scores:
            max_motion = max(motion_scores)
            avg_motion = np.mean(motion_scores)
            is_basket_hit = max_motion > avg_motion * 2.5
            confidence = min(max_motion / (avg_motion + 1e-6), 1.0)
            return is_basket_hit, confidence
        
        return False, 0.0
    
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
    
    def analyze_shooting_action_with_llm(self, video_path):
        frames = self.extract_key_frames(video_path)
        if not frames:
            return False, 0.0, "无法提取视频帧"
        
        if self.gemini_available:
            return self.analyze_with_gemini(frames)
        else:
            return False, 0.5, "Gemini不可用"
    
    def analyze_with_gemini(self, frames):
        try:
            import base64
            import io
            
            # 简化的提示
            prompt = "分析这些篮球截图，是否包含投篮动作？回答'是'或'否'。"
            
            # 准备消息内容
            parts = [{"text": prompt}]
            
            # 添加图片 (限制为前3帧以避免API限制)
            for frame in frames[:3]:
                img_bytes = io.BytesIO()
                frame.save(img_bytes, format='JPEG', quality=70)
                img_data = base64.b64encode(img_bytes.getvalue()).decode()
                
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg", 
                        "data": img_data
                    }
                })
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": parts}]
            )
            
            # 解析响应
            text = response.candidates[0].content.parts[0].text
            text_lower = text.lower()
            
            if "是" in text_lower or "yes" in text_lower or "投篮" in text_lower:
                return True, 0.8, text
            else:
                return False, 0.6, text
                
        except Exception as e:
            return False, 0.5, f"Gemini API错误: {str(e)}"
    
    
    def analyze_with_gemini_video(self, video_path):
        """使用Gemini分析整个视频片段"""
        if not self.gemini_available or not video_path:
            return False, 0.0, "Gemini不可用或视频文件无效"
        
        try:
            # 提取关键帧 (更多帧以覆盖10秒片段)
            frames = self.extract_key_frames(video_path, num_frames=8)
            if not frames:
                return False, 0.0, "无法提取视频帧"
            
            import base64
            import io
            
            prompt = """分析这个篮球比赛视频片段（约10秒），判断是否包含投篮得分事件。

请重点观察：
1. 是否有球员投篮动作
2. 篮球是否进入篮筐
3. 是否有明显的得分庆祝或反应

如果检测到投篮得分事件，请估算事件在视频中的大致时间位置（开始到结束的相对时间）。

请回答：是/否，并说明原因。如果是投篮得分，请估算事件持续时间。"""
            
            # 准备消息内容
            parts = [{"text": prompt}]
            
            # 添加视频帧
            for frame in frames:
                img_bytes = io.BytesIO()
                frame.save(img_bytes, format='JPEG', quality=70)
                img_data = base64.b64encode(img_bytes.getvalue()).decode()
                
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg", 
                        "data": img_data
                    }
                })
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": parts}]
            )
            
            # 解析响应
            text = response.candidates[0].content.parts[0].text
            text_lower = text.lower()
            
            # 判断是否检测到投篮得分
            if any(word in text_lower for word in ["是", "yes", "投篮", "得分", "进球", "命中"]):
                return True, 0.8, text
            else:
                return False, 0.3, text
                
        except Exception as e:
            return False, 0.0, f"Gemini分析错误: {str(e)}"
    
    def analyze_trigger(self, trigger_time):
        print(f"分析时间码前后±5秒: {trigger_time}s")
        
        # 提取触发时间前后5秒的视频片段
        video_start = trigger_time - 5
        video_end = trigger_time + 5
        
        print(f"提取视频片段: {video_start}s - {video_end}s")
        
        # 使用侧边线机位（更适合观察投篮动作）
        video_path = self.extract_video_segment(2, video_start, video_end)
        
        if not video_path:
            print("❌ 视频提取失败")
            return {
                "trigger_time": trigger_time,
                "event_detected": False,
                "event_start": None,
                "event_end": None,
                "analysis": "视频提取失败"
            }
        
        print("🤖 Gemini分析中...")
        is_event, confidence, analysis = self.analyze_with_gemini_video(video_path)
        
        print(f"分析结果: {'✅ 检测到投篮得分' if is_event else '❌ 未检测到投篮得分'}")
        print(f"置信度: {confidence:.2f}")
        
        if is_event:
            # 如果检测到事件，设置事件时间范围
            event_start = video_start + 1  # 稍微缩小范围
            event_end = video_end - 1
            
            return {
                "trigger_time": trigger_time,
                "event_detected": True,
                "event_start": event_start,
                "event_end": event_end,
                "confidence": confidence,
                "analysis": analysis
            }
        else:
            return {
                "trigger_time": trigger_time,
                "event_detected": False,
                "event_start": None,
                "event_end": None,
                "confidence": confidence,
                "analysis": analysis
            }
    
    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir)

def main():
    trigger_time = 6654  # 01:50:54
    analyzer = MultiViewAnalyzer()
    
    results = analyzer.analyze_trigger(trigger_time)
    
    output_file = os.path.join(OUTPUT_DIR, "analysis_result.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if results["event_detected"]:
        print(f"事件检测: ✅")
        print(f"START: {results['event_start']}s")
        print(f"END: {results['event_end']}s")
    else:
        print(f"事件检测: ❌")
    
    analyzer.cleanup()

if __name__ == "__main__":
    main()