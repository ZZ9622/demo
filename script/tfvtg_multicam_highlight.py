#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tfvtg_multicam_highlight.py

功能：
1. 从 Qwen 分析结果文件（analysis_ubuntu_cam6_t93.txt）中读取 [Before]/[During]/[After] 的文本描述
   ——只用文字，不用时间信息。
2. 对每个机位视频，使用 TFVTG 中的 VLM（BLIP2）+ temporal grounding 模块 (vlm_localizer.localize)
   做时序定位，得到三个阶段（Before / During / After）的候选片段及置信度。
3. 打印每个机位在三个阶段上的相似度（置信度），作为“与描述相似度”的结果。
4. **按你的要求选择多机位高光：**
   - 三个事件时间上有先后顺序（Before → During → After）；
   - 三段时间尽量连续、间隔不大；
   - 三个阶段尽量来自不同机位；
   - 用 moviepy 剪辑输出最终高光视频。

注意：
- 时序定位与相似度打分完全依赖 TFVTG 内的函数：
    - TFVTG.feature_extraction.get_visual_features
    - TFVTG.vlm_localizer.localize
- 剪辑部分使用 moviepy，如果环境中没有，请先安装：
    pip install moviepy
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 为了能直接 import TFVTG 下的模块，这里把工程根目录加入 sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /home/SONY/s7000043358/Sports_HL
sys.path.append(str(PROJECT_ROOT))

# === 引入 TFVTG 提供的函数 ===
from TFVTG.feature_extraction import get_visual_features
from TFVTG.vlm_localizer import localize

# === 剪辑用（不属于 TFVTG，纯粹为了生成视频） ===
from moviepy import VideoFileClip, concatenate_videoclips


# ==============================
# 1. 读取 Qwen 分析文本
# ==============================
def load_qwen_sub_events_from_analysis(analysis_path: Path) -> List[str]:
    """
    从 analysis_ubuntu_cam6_t93.txt 中解析出三个子事件描述：
    [Before] ... -> index 0
    [During] ... -> index 1
    [After]  ... -> index 2
    只关心中括号后的那段文字，不使用任何时间信息。
    """
    before, during, after = None, None, None

    if not analysis_path.exists():
        raise FileNotFoundError(f"分析文件不存在: {analysis_path}")

    with analysis_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[Before]"):
                before = line.split("]", 1)[1].strip()
            elif line.startswith("[During]"):
                during = line.split("]", 1)[1].strip()
            elif line.startswith("[After]"):
                after = line.split("]", 1)[1].strip()

    events = [before, during, after]
    if any(e is None for e in events):
        raise ValueError(f"分析文件格式不完整，无法找到 Before/During/After: {analysis_path}")

    return events


# ==============================
# 2. TFVTG 特征提取（多机位）
# ==============================
def extract_tfvtg_features_for_cameras(
    video_paths: Dict[str, str],
    fps: float = 3.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    使用 TFVTG.feature_extraction.get_visual_features 提取每个机位的视频特征。

    返回：
        cam_features: {cam_name: feature_array}，shape ~ (T, num_tokens, D)
        cam_durations: {cam_name: duration_in_seconds}，用 T / fps 近似
    """
    cam_features: Dict[str, np.ndarray] = {}
    cam_durations: Dict[str, float] = {}

    print("\n[1] 使用 TFVTG 提取多机位视频特征 (BLIP2)...")

    for cam_name, path in video_paths.items():
        if not os.path.exists(path):
            print(f"   [跳过] {cam_name}: 找不到视频文件 {path}")
            continue

        print(f"   -> 处理 {cam_name}: {path}")
        try:
            feats = get_visual_features(
                video_path=path,
                fps=fps,
                stride=None,
                max_duration=None,
                batch_size=128,
            )
        except Exception as e:
            print(f"   [警告] {cam_name}: 特征提取失败: {e}")
            continue

        if feats is None or feats.shape[0] == 0:
            print(f"   [警告] {cam_name}: 特征为空，跳过。")
            continue

        T = feats.shape[0]
        duration_sec = T / fps  # 用采样帧数和 fps 近似时长
        cam_features[cam_name] = feats
        cam_durations[cam_name] = duration_sec

        print(f"      特征维度: {feats.shape}, 近似时长: {duration_sec:.2f}s")

    return cam_features, cam_durations


# ==============================
# 3. TFVTG 时序定位
# ==============================
def run_tfvtg_localization_for_cameras(
    cam_features: Dict[str, np.ndarray],
    cam_durations: Dict[str, float],
    sub_event_texts: List[str],
    base_stride: int = 20,
    max_stride_factor: float = 0.5,
):
    """
    对每个机位跑一次 TFVTG 的 localize：
        - query_json = [{'descriptions': [Before_text, During_text, After_text]}]
        - 得到 ans[0]['response']，里面有多个候选片段，每个包含 start/end/confidence 等。

    返回：
        cam_responses: {cam_name: response_list}
            response_list = ans[0]['response']，每个元素是一个 dict：
            {
                'start': float,
                'static_start': float,
                'end': float,
                'confidence': float
            }
    """
    cam_responses: Dict[str, List[dict]] = {}

    print("\n[2] 使用 TFVTG.vlm_localizer.localize 做零样本时序定位...")

    for cam_name, feats in cam_features.items():
        duration = cam_durations[cam_name]
        T = feats.shape[0]

        stride = min(base_stride, max(3, T // 2))
        max_stride = int(T * max_stride_factor)
        max_stride = max(max_stride, stride * 2)

        query_json = [{
            "descriptions": sub_event_texts  # [Before_text, During_text, After_text]
        }]

        print(f"\n   -> 机位 {cam_name}: duration≈{duration:.2f}s, T={T}, stride={stride}, max_stride={max_stride}")
        try:
            ans = localize(
                video_feature=feats,
                duration=duration,
                query_json=query_json,
                stride=stride,
                max_stride=max_stride,
            )
        except Exception as e:
            print(f"      [警告] localize 失败: {e}")
            continue

        if not ans or "response" not in ans[0]:
            print("      [警告] 未获得有效 response。")
            continue

        cam_responses[cam_name] = ans[0]["response"]
        print(f"      得到 {len(ans[0]['response'])} 个候选片段。")

    return cam_responses


# ==============================
# 4. 打印相似度结果（需求 1）
# ==============================
def print_similarity_scores(cam_responses: Dict[str, List[dict]]):
    """
    把 TFVTG 返回的 response 解释成“与描述的相似度”并输出。

    简单假设（与 TFVTG 原 evaluate 逻辑一致）：
        - response[0] -> subevent 1 (Before)
        - response[1] -> subevent 2 (During)
        - response[2] -> subevent 3 (After)
    """
    event_names = ["Before", "During", "After"]

    if not cam_responses:
        print("\n[!] 没有任何机位的 TFVTG 定位结果。")
        return

    print("\n[3] 各机位基于 TFVTG 的相似度结果（仅看描述，不看时间标签）：")
    print("=" * 80)

    for cam_name, resp in cam_responses.items():
        print(f"\n机位: {cam_name}")
        if len(resp) < 3:
            print(f"  [警告] response 数量 < 3，仅有 {len(resp)} 段，可能无法完整对应 Before/During/After。")

        for idx, event_name in enumerate(event_names):
            if idx >= len(resp):
                print(f"  {event_name:<7}: (无候选片段)")
                continue
            r = resp[idx]
            conf = r.get("confidence", 0.0)
            start = r.get("start", 0.0)
            end = r.get("end", 0.0)
            print(
                f"  {event_name:<7} | "
                f"score={conf:.4f} | "
                f"start={start:.2f}s, end={end:.2f}s"
            )

    print("\n说明：这里的 score 来自 TFVTG 的置信度，可以视作“该机位在对应阶段与文字描述的匹配程度”。")


# ==============================
# 5. 选多机位最佳时间线（满足你 3 个条件）
# ==============================
def select_best_multicam_timeline(
    cam_responses: Dict[str, List[dict]],
    video_paths: Dict[str, str],
    max_gap: float = 2.0,        # 阶段之间允许的最大空隙（秒），略放宽一点
    max_overlap: float = 1.0,    # 允许事件之间重叠的最长时间（秒）
):
    """
    从所有机位的 TFVTG response 中，选出一个“多机位组合时间线”，满足：
    1）Before -> During -> After 有先后顺序；
    2）三段时间尽量连续，间隔不大（由 max_gap 控制），允许 <= max_overlap 的重叠；
    3）三个阶段来自不同机位（cam 不相同）；
    4）During 事件权重更高。
    """
    event_names = ["Before", "During", "After"]

    # 1. 收集候选池
    candidates = {ev: [] for ev in event_names}

    for cam_name, resp in cam_responses.items():
        for idx, ev in enumerate(event_names):
            if idx >= len(resp):
                continue
            seg = resp[idx]
            candidates[ev].append({
                "event": ev,
                "cam": cam_name,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "confidence": float(seg.get("confidence", 0.0)),
                "video_path": video_paths.get(cam_name, ""),
            })

    # 如果某个阶段没有候选，直接失败
    for ev in event_names:
        if not candidates[ev]:
            print(f"\n[!] 阶段 {ev} 没有任何候选片段，无法构建完整时间线。")
            return []

    # 每个阶段按 score 排序，只保留前 top_k，减小组合数量
    top_k = 5
    for ev in event_names:
        candidates[ev].sort(key=lambda x: x["confidence"], reverse=True)
        candidates[ev] = candidates[ev][:top_k]

    print("\n[4] 在多机位候选中搜索满足顺序 & 连续性 & 不同机位的最佳组合...")

    best_combo = None
    best_score = -1e9

    # 权重：During 更高
    w_before = 1.0
    w_during = 2.0
    w_after = 1.0
    gap_penalty = 0.3  # 间隔惩罚系数（可以按效果再调）

    for b in candidates["Before"]:
        for d in candidates["During"]:
            for a in candidates["After"]:
                # 3）三个阶段来自不同机位
                if len({b["cam"], d["cam"], a["cam"]}) < 3:
                    continue

                # 1）时间先后顺序：Before.start <= During.start <= After.start
                if not (b["start"] <= d["start"] <= a["start"]):
                    continue

                # 持续时间必须正常
                if not (b["end"] > b["start"] and d["end"] > d["start"] and a["end"] > a["start"]):
                    continue

                # 计算 Before-During 的间隔 & 重叠
                gap_bd = max(0.0, d["start"] - b["end"])
                overlap_bd = max(0.0, min(b["end"], d["end"]) - max(b["start"], d["start"]))

                # 计算 During-After 的间隔 & 重叠
                gap_da = max(0.0, a["start"] - d["end"])
                overlap_da = max(0.0, min(d["end"], a["end"]) - max(d["start"], a["start"]))

                # 间隔不要太大
                if gap_bd > max_gap or gap_da > max_gap:
                    continue

                # 重叠不超过 max_overlap（允许一定重叠）
                if overlap_bd > max_overlap or overlap_da > max_overlap:
                    continue

                # 组合打分：Before/After 权重 1.0，During 权重 2.0，减去间隔惩罚
                score = (
                    w_before * b["confidence"]
                    + w_during * d["confidence"]
                    + w_after * a["confidence"]
                    - gap_penalty * (gap_bd + gap_da)
                )

                if score > best_score:
                    best_score = score
                    best_combo = (b, d, a)

    if best_combo is None:
        print("\n[!] 在当前约束下（顺序+重叠<=1s+不同机位）没有找到合法组合。")
        return []

    b, d, a = best_combo
    print("\n[4] 找到的最佳多机位组合：")
    for seg in [b, d, a]:
        print(
            f"  {seg['event']:<7} | 机位 {seg['cam']:<6} | "
            f"{seg['start']:.2f}s ~ {seg['end']:.2f}s | score={seg['confidence']:.4f}"
        )

    # 返回按事件顺序排列的 timeline
    timeline = []
    for ev in event_names:
        if ev == "Before":
            timeline.append(b)
        elif ev == "During":
            timeline.append(d)
        else:
            timeline.append(a)
    return timeline

# ==============================
# 6. 剪辑导出视频
# ==============================
def edit_and_export_video(
    timeline: List[dict],
    output_path: Path,
):
    """
    根据 timeline（3 段多机位片段）剪辑并导出一个高光视频。

    规则：
    - 如果 Before / After 与 During 有重叠，按“During 为准”：
        - Before 只保留 [Before.start, min(Before.end, During.start)]；
        - After  只保留 [max(After.start, During.end), After.end]；
    - 如果裁剪后长度 <= 0，则跳过该段。
    """
    if not timeline:
        print("\n[!] timeline 为空，无法剪辑。")
        return

    print("\n[5] 开始剪辑多机位高光视频...")

    # 找到 During 段，作为时间上的锚点
    during_seg = None
    for seg in timeline:
        if seg.get("event") == "During":
            during_seg = seg
            break
    if during_seg is None:
        print("[!] timeline 中没有标记为 'During' 的片段，无法按 During 对齐。")
        return

    ds, de = during_seg["start"], during_seg["end"]

    adjusted_segments: List[dict] = []
    for seg in timeline:
        ev = seg.get("event")
        s, e = seg["start"], seg["end"]

        # 按照“以 During 为准”调整 Before / After 的边界
        if ev == "Before":
            # Before 不能穿过 During.start
            if e > ds:
                e = min(e, ds)
        elif ev == "After":
            # After 不能早于 During.end
            if s < de:
                s = max(s, de)

        if e <= s:
            print(f"  [跳过] {ev} 段裁剪后时间区间异常 ({s:.2f} >= {e:.2f})")
            continue

        new_seg = dict(seg)
        new_seg["start"] = s
        new_seg["end"] = e
        adjusted_segments.append(new_seg)

    if not adjusted_segments:
        print("\n[!] 所有片段裁剪后都无效，无法剪辑。")
        return

    clips = []
    for seg in adjusted_segments:
        cam = seg["cam"]
        vpath = seg["video_path"]
        s = seg["start"]
        e = seg["end"]

        if not vpath or not os.path.exists(vpath):
            print(f"  [跳过] 机位 {cam}: 找不到视频文件 {vpath}")
            continue

        print(f"   -> {seg['event']:<7} | 机位 {cam} | {s:.2f}s ~ {e:.2f}s | {vpath}")
        # 按你说的，用 subclipped（你当前版本里就是这么用的）
        clip = VideoFileClip(vpath).subclipped(s, e)
        clips.append(clip)

    if not clips:
        print("\n[!] 没有有效片段可供拼接。")
        return

    final_clip = concatenate_videoclips(clips, method="compose")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    print(f"\n✅ 多机位高光剪辑完成，已导出至: {output_path}")

    for c in clips:
        c.close()
    final_clip.close()

# ==============================
# 7. 主入口
# ==============================
def main():
    script_dir = Path(__file__).resolve().parent
    demo_root = script_dir.parent  # /demo
    analysis_path = demo_root / "output" / "analysis_ubuntu_cam6_t93.txt"

    print("=" * 60)
    print(" 基于 TFVTG 的多机位相似度计算与高光剪辑 ")
    print("=" * 60)
    print(f"分析文件: {analysis_path}")

    # 1) 解析 Qwen 分析文本
    sub_event_texts = load_qwen_sub_events_from_analysis(analysis_path)
    print("\n[0] Qwen 文本描述（只看内容，不看时间标签）：")
    print(f"  [Before] {sub_event_texts[0]}")
    print(f"  [During] {sub_event_texts[1]}")
    print(f"  [After ] {sub_event_texts[2]}")

    # 2) 多机位视频路径（你可以按需增减机位）
    data_dir = demo_root / "data" / "apidis"
    video_paths = {
        "cam_1": str(data_dir / "camera1_from_1h50m_to_end_89_97.mp4"),
        "cam_2": str(data_dir / "camera2_from_1h50m_to_end_89_97.mp4"),
        "cam_3": str(data_dir / "camera3_from_1h50m_to_end_89_97.mp4"),
        "cam_4": str(data_dir / "camera4_from_1h50m_to_end_89_97.mp4"),
        "cam_5": str(data_dir / "camera5_from_1h50m_to_end_89_97.mp4"),
        "cam_6": str(data_dir / "camera6_from_1h50m_to_end_89_97.mp4"),
        "cam_7": str(data_dir / "camera7_from_1h50m_to_end_89_97.mp4")
        # 如有其他机位，可以继续添加：
        # "cam_1": str(data_dir / "camera1_from_1h50m_to_end_89_97.mp4"),
        # ...
    }

    # 3) TFVTG 特征提取
    cam_features, cam_durations = extract_tfvtg_features_for_cameras(
        video_paths=video_paths,
        fps=3.0,
    )

    if not cam_features:
        print("\n[!] 未成功提取任何机位特征，流程结束。")
        return

    # 4) TFVTG 时序定位 + 相似度打分
    cam_responses = run_tfvtg_localization_for_cameras(
        cam_features=cam_features,
        cam_durations=cam_durations,
        sub_event_texts=sub_event_texts,
        base_stride=20,
        max_stride_factor=0.5,
    )

    # 需求 1：输出各机位相似度结果
    print_similarity_scores(cam_responses)

    if not cam_responses:
        print("\n[!] 没有任何有效 TFVTG 定位结果，无法做组合与剪辑。")
        return

    # 5) 需求 2 & 3：选出满足“顺序 + 紧邻 + 不同机位”的最佳组合并剪辑
    timeline = select_best_multicam_timeline(
        cam_responses=cam_responses,
        video_paths=video_paths,
        max_gap=1.0,        # 可调：阶段之间最多间隔 1 秒
        max_overlap=0.3, # 可调：最多允许 0.3 秒的重叠
    )

    if not timeline:
        print("\n[!] 未选出有效的多机位组合，无法剪辑。")
        return

    output_path = demo_root / "output" / "final_highlight_tfvtg_ordered.mp4"
    edit_and_export_video(timeline, output_path)


if __name__ == "__main__":
    main()