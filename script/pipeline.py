#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline.py  (run in conda env: lavis)

Step A (qwen env): run Qwen analysis script to generate analysis_ubuntu_cam6_t*.txt
Step B (lavis env): read [During] from latest analysis file
Step C (lavis env): TFVTG (calc_scores + localize) over cam1..cam7 clips, rank by raw_score_max, then cut video

Run:
  conda run -n lavis python demo/script/pipeline.py
or:
  conda activate lavis && python demo/script/pipeline.py
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# --- Path setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Sports_HL
DEMO_ROOT = PROJECT_ROOT / "demo"
OUTPUT_DIR = DEMO_ROOT / "output"
DATA_DIR = DEMO_ROOT / "data" / "apidis"

# --- TFVTG imports (must be available in lavis env) ---
sys.path.append(str(PROJECT_ROOT))
from TFVTG.feature_extraction import get_visual_features
from TFVTG.vlm_localizer import localize, calc_scores

# # --- moviepy (in lavis env) ---
# from moviepy import VideoFileClip, concatenate_videoclips

# def cut_video_segment(src_path: str, out_path: str, start_sec: float, end_sec: float):
#     clip = VideoFileClip(src_path).subclipped(start_sec, end_sec)
#     # 这里只是生成一个中间片段给 TFVTG 用，可以用较小码率降低体积
#     clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
#     clip.close()


def cut_video_segment_ffmpeg(src_path: str, out_path: str, start_sec: float, end_sec: float):
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if end_sec <= start_sec:
        raise ValueError(f"invalid range: start={start_sec}, end={end_sec}")

    duration = end_sec - start_sec

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", src_path,
        "-c:v", "libx264",
        "-preset", "superfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        # "-c:a", "aac",
        # "-b:a", "128k",
        "-an",
        "-movflags", "+faststart",
        "-reset_timestamps", "1",
        # "-avoid_negative_ts", "make_zero", # 确保时间戳严格从0开始    
        out_path,
    ]
    print(" ".join(cmd))
    # subprocess.run(cmd, check=True)  # 先别吞输出，方便看到 ffmpeg 警告/报错
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# def run_qwen_analysis(video_path: str, trigger_time: float, pre_sec: float, post_sec: float):
#     """
#     用 qwen 环境执行 analyze_qwen25vl3b_ubuntu.py。
#     该脚本会自己把结果写到 demo/output/analysis_ubuntu_cam6_t{trigger}.txt
#     """
#     script = DEMO_ROOT / "script" / "analyze_qwen25vl3b_ubuntu.py"
#     if not script.exists():
#         raise FileNotFoundError(f"找不到 Qwen 脚本: {script}")

#     my_env = os.environ.copy()
#     for var in ["LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"]:
#         if var in my_env:
#             del my_env[var]

#     cmd = [
#         "conda", "run", "-n", "hf", "python", "demo/script/analyze_qwen25vl3b_ubuntu.py",
#         "--video", video_path,
#         "--trigger", str(trigger_time),
#         "--pre", str(pre_sec),
#         "--post", str(post_sec),
#     ]
#     print("\n=== [A] 运行 Qwen 分析（conda env: qwen）===")
#     print(" ".join(cmd))
#     subprocess.run(cmd, check=True, env=my_env)


import requests
import json

def run_qwen_analysis(video_path: str, trigger_time: float, pre_sec: float, post_sec: float):
    """
    不再启动子进程，而是向常驻的 Qwen 服务发送 HTTP 请求。
    """
    print("\n=== [A] 向 Qwen Server 请求分析 ===")
    url = "http://127.0.0.1:8000/analyze"
    payload = {
        "video_path": str(video_path),
        "start_sec": 0.0,
        "end_sec": pre_sec + post_sec # 这里的视频已经是被裁切好的临时文件了
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            result_text = data["result"]
            print(f"✅ 分析成功 (耗时 {data['time_cost']:.2f}s):\n{result_text}")
            
            # 把结果写回原来的 txt 文件，保持后续逻辑兼容
            out_file = OUTPUT_DIR / f"analysis_ubuntu_cam6_t{trigger_time}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("[During] " + result_text + "\n")
        else:
            raise RuntimeError(f"Qwen 分析出错: {data['message']}")
            
    except Exception as e:
        raise RuntimeError(f"无法连接到 Qwen 服务，请确保 qwen_server.py 正在运行。错误: {e}")


def find_latest_analysis_file() -> Path:
    """
    找 demo/output 下最新的 analysis_ubuntu_cam6_t*.txt
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(OUTPUT_DIR.glob("analysis_ubuntu_cam6_t*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"没找到分析文件: {OUTPUT_DIR}/analysis_ubuntu_cam6_t*.txt")
    return files[0]


def load_during_text(analysis_path: Path) -> str:
    during = None
    with analysis_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[During]"):
                during = line.split("]", 1)[1].strip()
                break
    if not during:
        raise ValueError(f"未在分析文件中找到 [During]: {analysis_path}")
    return during

# def load_during_text(analysis_path: Path) -> str:
#     """
#     从分析文件中读取 Result 段落的正文（不再要求 [During] 前缀）。
#     """
#     in_result = False
#     lines: list[str] = []

#     with analysis_path.open("r", encoding="utf-8") as f:
#         for raw in f:
#             line = raw.rstrip("\n")
#             # 找到 “Result:” 标题下面的正文
#             if not in_result:
#                 if line.strip() == "Result:":
#                     in_result = True
#                 continue

#             # Result 段落后如果有新的分隔线或空行，可以在这里判断结束
#             if line.strip().startswith("====") and lines:
#                 break

#             if line.strip():
#                 lines.append(line.strip())

#     text = " ".join(lines).strip()
#     if not text:
#         raise ValueError(f"未在分析文件中找到 Result 正文: {analysis_path}")
#     return text

def extract_tfvtg_features(video_path: str, fps: float = 4.0) -> Tuple[np.ndarray, float]:
    feats = get_visual_features(
        video_path=video_path,
        fps=fps,
        stride=None,
        max_duration=None,
        batch_size=128,
    )
    if feats is None or feats.shape[0] == 0:
        raise RuntimeError("特征为空")
    duration_sec = feats.shape[0] / fps  # 近似
    return feats, duration_sec

# ========================== visualize scores start ==========================
import matplotlib.pyplot as plt
import numpy as np

def visualize_tfvtg_scores(scores, duration_sec, out_path: str):
    """
    scores: torch.Tensor, 形状 [1, T]
    duration_sec: 这段视频的总时长（秒），就是 extract_tfvtg_features 返回的那个 dur
    """
    # 1) 转成 numpy
    s = scores[0].detach().cpu().numpy()   # 形状 [T]
    T = len(s)

    # 2) 为每个 time step 生成对应的时间戳（相对于 seg_video 的 0 秒）
    times = np.linspace(0, duration_sec, T, endpoint=False)

    # 3) 画图
    plt.figure(figsize=(10, 4))
    plt.plot(times, s)
    plt.xlabel("Time (s)")
    plt.ylabel("Similarity")
    # plt.axvline(start, color="red", linestyle="--", label="start")
    # plt.axvline(end, color="green", linestyle="--", label="end")
    plt.legend()
    plt.title("TFVTG text-video similarity over time")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
# ========================== visualize scores end ==========================
    
def score_and_localize_during(
    feats: np.ndarray,
    duration_sec: float,
    during_text: str,
    base_stride: int = 20,
    max_stride_factor: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    返回：raw_score_max, start, end, local_conf(仅参考)
    - raw_score_max 来自 calc_scores：跨 cam 可比
    - start/end 来自 localize：定位片段
    """
    scores = calc_scores(feats, [during_text])
    raw_score_max = float(scores.detach().cpu().numpy()[0].max())
    # ========================== visualize scores start ==========================
    from pathlib import Path
    out_img = OUTPUT_DIR / "tfvtg_similarity.png"
    visualize_tfvtg_scores(scores, duration_sec, str(out_img))
    print(f"相似度曲线已保存: {out_img}")
    # ========================== visualize scores end ==========================

    T = feats.shape[0]
    stride = min(base_stride, max(3, T // 2))
    max_stride = max(int(T * max_stride_factor), stride * 2)

    ans = localize(
        video_feature=feats,
        duration=duration_sec,
        query_json=[{"descriptions": [during_text]}],
        stride=stride,
        max_stride=max_stride,
    )
    if not ans or "response" not in ans[0] or not ans[0]["response"]:
        return raw_score_max, 0.0, 0.0, 0.0

    seg = ans[0]["response"][0]
    return raw_score_max, float(seg["start"]), float(seg["end"]), float(seg["confidence"])


def run_grounding(video_path: str, trigger_time: float, pre_seconds: float, post_seconds: float):
    """
    执行完整的 Grounding 流程
    """
    # 0) 先在 lavis 环境中，把原始长视频裁成以 trigger 为中心的短片段
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    seg_start = trigger_time - pre_seconds
    seg_end   = trigger_time + post_seconds
    # 确保开始时间不为负数
    seg_start = max(0, seg_start)
    
    seg_video = str(OUTPUT_DIR / "tmp_trigger_segment.mp4")

    print(f"\n=== [0] 裁剪原始视频: {seg_start}s -> {seg_end}s ===")
    # cut_video_segment(video_path, seg_video, seg_start, seg_end)
    cut_video_segment_ffmpeg(video_path, seg_video, seg_start, seg_end)

    # 1) Qwen 分析（qwen env）—— 对同一个「已裁剪」视频做文字分析
    run_qwen_analysis(seg_video, trigger_time, pre_seconds, post_seconds)

    # 2) 找最新 analysis 文件并读 During
    analysis_path = find_latest_analysis_file()
    during_text = load_during_text(analysis_path)

    print("\n=== [B] 读取 Qwen 输出（latest analysis）===")
    print(f"analysis: {analysis_path}")
    print(f"[During] {during_text}")

    # 3) TFVTG 对同一个「已裁剪」视频做 During 匹配（lavis env）
    print("\n=== [C] TFVTG During 匹配（conda env: lavis）===")

    # 复用前面裁剪好的 seg_video
    input_video = seg_video

    print(f"-> processing: {input_video}")
    feats, dur = extract_tfvtg_features(input_video, fps=4.0)
    raw_score, start, end, local_conf = score_and_localize_during(feats, dur, during_text)

    print("\nTFVTG 结果：")
    print(f"raw={raw_score:.4f} | local={local_conf:.4f} | {start:.2f}s ~ {end:.2f}s")

    if end <= start:
        # 如果定位失败，默认输出整个裁剪后的片段
        print("警告：定位时间段异常，输出原始裁剪片段。")
        start, end = 0, dur

    # 4) 剪辑输出（lavis env）
    out_mp4 = OUTPUT_DIR / "pipeline_during_best.mp4"
    print("\n=== [D] 剪辑输出（单视频）===")
    print(f"output={out_mp4}")

    # clip = VideoFileClip(input_video).subclipped(start, end)
    # final = concatenate_videoclips([clip], method="compose")
    # final.write_videofile(str(out_mp4), codec="libx264", audio_codec="aac")
    # clip.close()
    # final.close()

    cut_video_segment_ffmpeg(input_video, str(out_mp4), start, end)


if __name__ == "__main__":
    # 在这里定义你的参数
    video_to_process = str(DATA_DIR / "camera6_from_1h50m_to_end.mp4")
    t_time = 93.0
    pre = 5.0
    post = 5.0

    # 传入函数
    run_grounding(
        video_path=video_to_process,
        trigger_time=t_time,
        pre_seconds=pre,
        post_seconds=post
    )