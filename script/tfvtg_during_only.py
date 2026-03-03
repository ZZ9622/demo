#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tfvtg_during_only.py  (方案 A 版)

根据 analysis_ubuntu_cam6_t93.txt 的 [During] 描述，
对每个机位：
  - 用 TFVTG.vlm_localizer.calc_scores 得到“原始相似度序列”，用 max(raw_score) 作为该机位总分
  - 用 TFVTG.vlm_localizer.localize 只负责给出时间段 [start, end]
跨机位用 raw_score_max 排序，选得分最高的机位剪辑出 During 片段。
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import sys

# 让 Python 能 import 到 Sports_HL/TFVTG
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /home/SONY/s7000043358/Sports_HL
sys.path.append(str(PROJECT_ROOT))

from TFVTG.feature_extraction import get_visual_features
from TFVTG.vlm_localizer import localize, calc_scores

from moviepy import VideoFileClip, concatenate_videoclips


# ==============================
# 1. 只取 [During] 文本
# ==============================
def load_during_text(analysis_path: Path) -> str:
    during = None
    if not analysis_path.exists():
        raise FileNotFoundError(f"分析文件不存在: {analysis_path}")

    with analysis_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[During]"):
                during = line.split("]", 1)[1].strip()
                break

    if not during:
        raise ValueError(f"未在分析文件中找到 [During] 行: {analysis_path}")
    return during


# ==============================
# 2. 提取单机位特征
# ==============================
def extract_tfvtg_features(video_path: str, fps: float = 3.0) -> Tuple[np.ndarray, float]:
    feats = get_visual_features(    # BLIP2提取视频特征 输出形状：[T, num_tokens, D]
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


# ==============================
# 3. 单机位：原始相似度 + 时间段
# ==============================
def score_and_segment_one_cam(
    feats: np.ndarray,
    duration_sec: float,
    during_text: str,
    base_stride: int = 20,
    max_stride_factor: float = 0.5,
) -> Tuple[float, dict]:
    """
    返回：
        raw_score_max: 该机位在整个时间轴上的原始最大相似度（可用于跨 cam 排序）
        seg: localize 给出的时间段（只取 top1）
    """
    # 3.1 原始相似度：calc_scores（不做 per-video 归一化）
    # scores shape: [1, T']，越大越相似
    scores = calc_scores(feats, [during_text])  # torch.Tensor
    scores_np = scores.detach().cpu().numpy()[0]
    raw_score_max = float(scores_np.max())

    # 3.2 用 localize 只做“时间段定位”：[start, end]
    T = feats.shape[0]
    stride = min(base_stride, max(3, T // 2))
    max_stride = int(T * max_stride_factor)
    max_stride = max(max_stride, stride * 2)

    query_json = [{"descriptions": [during_text]}]  # 只有 1 个 subevent

    ans = localize(
        video_feature=feats,
        duration=duration_sec,
        query_json=query_json,
        stride=stride,
        max_stride=max_stride,
    )

    if not ans or "response" not in ans[0] or not ans[0]["response"]:
        seg = {"start": 0.0, "end": 0.0, "confidence": 0.0}
    else:
        # 对 only-During 的场景，response[0] 就是我们要的片段
        seg = ans[0]["response"][0]

    return raw_score_max, seg


# ==============================
# 4. 主流程
# ==============================
def main():
    script_dir = Path(__file__).resolve().parent
    demo_root = script_dir.parent  # /demo

    analysis_path = demo_root / "output" / "analysis_ubuntu_cam6_t93.txt"
    during_text = load_during_text(analysis_path)

    print("=" * 60)
    print(" TFVTG During-only 多机位匹配（使用 calc_scores 原始相似度） ")
    print("=" * 60)
    print(f"分析文件: {analysis_path}")
    print(f"[During] {during_text}")

    data_dir = demo_root / "data" / "apidis"
    video_paths: Dict[str, str] = {
        "cam_1": str(data_dir / "camera1_from_1h50m_to_end_89_97.mp4"),
        "cam_2": str(data_dir / "camera2_from_1h50m_to_end_89_97.mp4"),
        "cam_3": str(data_dir / "camera3_from_1h50m_to_end_89_97.mp4"),
        "cam_4": str(data_dir / "camera4_from_1h50m_to_end_89_97.mp4"),
        "cam_5": str(data_dir / "camera5_from_1h50m_to_end_89_97.mp4"),
        "cam_6": str(data_dir / "camera6_from_1h50m_to_end_89_97.mp4"),
        "cam_7": str(data_dir / "camera7_from_1h50m_to_end_89_97.mp4"),
    }

    results: List[dict] = []

    print("\n[1] 提取特征 + 计算原始相似度 + During 定位（每个机位）...")
    for cam, vpath in video_paths.items():
        if not os.path.exists(vpath):
            print(f"  [跳过] {cam}: 找不到 {vpath}")
            continue

        print(f"  -> {cam}")
        try:
            feats, dur = extract_tfvtg_features(vpath, fps=3.0)
            raw_score_max, seg = score_and_segment_one_cam(
                feats=feats,
                duration_sec=dur,
                during_text=during_text,
                base_stride=20,
                max_stride_factor=0.5,
            )
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            conf_local = float(seg.get("confidence", 0.0))

            res = {
                "cam": cam,
                "video_path": vpath,
                "start": start,
                "end": end,
                # 用 raw_score_max 做“跨 cam 可比的分数”
                "score_raw": raw_score_max,
                # 仅做参考：localize 内部归一化后的 confidence
                "score_local": conf_local,
            }
            results.append(res)
            print(
                f"     raw_score_max={res['score_raw']:.4f} | "
                f"local_conf={res['score_local']:.4f} | "
                f"{res['start']:.2f}s ~ {res['end']:.2f}s"
            )
        except Exception as e:
            print(f"     [失败] {e}")

    if not results:
        print("\n[!] 没有任何机位得到有效结果。")
        return

    # 按 raw_score_max 排序（跨机位真正可比的分数）
    results.sort(key=lambda x: x["score_raw"], reverse=True)

    print("\n[2] 全机位 During 匹配 TOP5（按 raw_score_max 排序）：")
    for i, r in enumerate(results[:5], start=1):
        print(
            f"  TOP{i} | {r['cam']} | "
            f"raw_score={r['score_raw']:.4f} | "
            f"local_conf={r['score_local']:.4f} | "
            f"{r['start']:.2f}s ~ {r['end']:.2f}s"
        )

    best = results[0]
    if best["end"] <= best["start"]:
        print("\n[!] 最优结果时间区间异常，无法剪辑。")
        return

    print("\n[3] 剪辑最优 During 片段（按 raw_score_max 最高的机位）...")
    output_path = demo_root / "output" / "final_highlight_tfvtg_during_only.mp4"

    clip = VideoFileClip(best["video_path"]).subclipped(best["start"], best["end"])
    final_clip = concatenate_videoclips([clip], method="compose")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    clip.close()
    final_clip.close()

    print(f"\n✅ 已输出: {output_path}")
    print(
        f"✅ 最优机位: {best['cam']} | "
        f"raw_score={best['score_raw']:.4f} | local_conf={best['score_local']:.4f}"
    )


if __name__ == "__main__":
    main()