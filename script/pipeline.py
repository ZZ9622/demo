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

import json
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
from TFVTG.llm_prompting import select_proposal, filter_and_integrate

# --- moviepy (in lavis env) ---
from moviepy import VideoFileClip, concatenate_videoclips

def cut_video_segment(src_path: str, out_path: str, start_sec: float, end_sec: float):
    clip = VideoFileClip(src_path).subclipped(start_sec, end_sec)
    # 这里只是生成一个中间片段给 TFVTG 用，可以用较小码率降低体积
    clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
    clip.close()

def run_qwen_analysis(video_path: str, trigger_time: float, pre_sec: float, post_sec: float):
    """
    用 qwen 环境执行 analyze_qwen25vl3b_ubuntu.py。
    该脚本会自己把结果写到 demo/output/analysis_ubuntu_cam6_t{trigger}.txt
    """
    script = DEMO_ROOT / "script" / "analyze_qwen25vl3b_ubuntu.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 Qwen 脚本: {script}")

    my_env = os.environ.copy()
    for var in ["LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"]:
        if var in my_env:
            del my_env[var]

    cmd = [
        "conda", "run", "-n", "hf", "python", "demo/script/analyze_qwen25vl3b_ubuntu.py",
        "--video", video_path,
        "--trigger", str(trigger_time),
        "--pre", str(pre_sec),
        "--post", str(post_sec),
    ]
    print("\n=== [A] 运行 Qwen 分析（conda env: qwen）===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=my_env)


def find_latest_analysis_file() -> Path:
    """
    找 demo/output 下最新的 analysis_ubuntu_cam6_t*.txt
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(OUTPUT_DIR.glob("analysis_ubuntu_cam6_t*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"没找到分析文件: {OUTPUT_DIR}/analysis_ubuntu_cam6_t*.txt")
    return files[0]


def load_qwen_json(analysis_path: Path) -> Dict:
    """
    从 analysis_ubuntu_cam6_t*.txt 中解析出 Qwen 输出的 JSON。
    该 JSON 出现在 Result 段落之后，且不包含 Markdown 代码块外壳。
    如果模型偶尔输出 ``` 或 ```json，这里会自动剥离。
    """
    lines: List[str] = []
    in_result = False
    with analysis_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not in_result:
                if line.strip() == "Result:":
                    in_result = True
                continue
            # 跳过再次出现的分隔线
            if line.strip().startswith("===="):
                continue
            if line.strip():
                lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        raise ValueError(f"未在分析文件中找到 Result JSON: {analysis_path}")

    # 如果不小心带了 ``` 或 ```json，先剥掉
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Result JSON 解析失败: {analysis_path}, err={e}\n{text}")

    if not isinstance(data, dict):
        raise ValueError(f"Result JSON 顶层不是对象: {analysis_path}")
    if "query_json" not in data or not isinstance(data["query_json"], list):
        raise ValueError(f"Result JSON 缺少 query_json 字段: {analysis_path}")
    return data

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

def load_query_json(analysis_path: Path) -> list[dict]:
    """
    从 analysis_ubuntu_cam6_t*.txt 中解析出 Qwen 输出的 query_json。
    假设文件格式大致为：
        ... 头部信息 ...
        Result:
        ============================================
        { ... 这里是一整段 JSON ... }
    """
    lines = []
    in_result = False
    with analysis_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not in_result:
                if line.strip() == "Result:":
                    in_result = True
                continue
            # 跳过分隔线
            if line.strip().startswith("===="):
                continue
            if line.strip():
                lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError(f"未在分析文件中找到 Result JSON: {analysis_path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Result JSON 解析失败: {analysis_path}, err={e}\n{text}")
    if "query_json" not in data or not isinstance(data["query_json"], list):
        raise ValueError(f"Result JSON 中缺少 query_json 字段: {analysis_path}")
    return data["query_json"]

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
    descriptions: List[str],
    base_stride: int = 20,
    max_stride_factor: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    返回：raw_score_max, start, end, local_conf(仅参考)
    - raw_score_max 来自 calc_scores：跨 cam 可比
    - start/end 来自 localize：定位片段
    """
    if not descriptions:
        return 0.0, 0.0, 0.0, 0.0

    scores = calc_scores(feats, descriptions)
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
        query_json=[{"descriptions": descriptions}],
        stride=stride,
        max_stride=max_stride,
    )
    if not ans or "response" not in ans[0] or not ans[0]["response"]:
        return raw_score_max, 0.0, 0.0, 0.0

    seg = ans[0]["response"][0]
    return raw_score_max, float(seg["start"]), float(seg["end"]), float(seg["confidence"])


def main():
    # 0) 先在 lavis 环境中，把原始长视频裁成以 trigger 为中心的短片段
    if not os.path.exists(USER_VIDEO):
        raise FileNotFoundError(f"视频不存在: {USER_VIDEO}")

    seg_start = TRIGGER_TIME - PRE_SECONDS
    seg_end   = TRIGGER_TIME + POST_SECONDS
    seg_video = str(OUTPUT_DIR / "tmp_trigger_segment.mp4")

    cut_video_segment(USER_VIDEO, seg_video, seg_start, seg_end)

    # 1) Qwen 分析（qwen env）—— 对同一个「已裁剪」视频做文字分析
    #    Qwen 脚本内部不再做基于 trigger 的二次裁剪
    run_qwen_analysis(seg_video, TRIGGER_TIME, PRE_SECONDS, POST_SECONDS)

    # # 2) 找最新 analysis 文件并读 Qwen JSON 输出
    # analysis_path = find_latest_analysis_file()
    # qwen_data = load_qwen_json(analysis_path)
    # query_items = qwen_data.get("query_json", [])
    # relationship = qwen_data.get("relationship", "single-query")

    # print("\n=== [B] 读取 Qwen 输出（latest analysis）===")
    # print(f"analysis: {analysis_path}")
    # print("Qwen JSON:")
    # print(json.dumps(qwen_data, ensure_ascii=False, indent=2))

    # # 3) TFVTG 对同一个「已裁剪」视频做多 query 匹配（lavis env）
    # print("\n=== [C] TFVTG 多 query 匹配（conda env: lavis）===")

    # # 复用前面裁剪好的 seg_video，确保 TFVTG 匹配的也是 trigger 前后的视频片段
    # video_path = seg_video

    # print(f"-> single_video: {video_path}")
    # feats, dur = extract_tfvtg_features(video_path, fps=5.0)

    # # 根据 Qwen 输出构造 TFVTG 的 query_json 列表
    # tfvtg_query_json: List[Dict] = []
    # for item in query_items:
    #     descs = item.get("descriptions") or []
    #     if not isinstance(descs, list) or not descs:
    #         continue
    #     tfvtg_query_json.append({"descriptions": descs})

    # if not tfvtg_query_json:
    #     raise RuntimeError(f"Qwen JSON 中没有有效的 descriptions: {analysis_path}")

    # # 使用 TFVTG 的 localize 进行多 query 定位
    # T = feats.shape[0]
    # base_stride = 20
    # max_stride_factor = 0.5
    # stride = min(base_stride, max(3, T // 2))
    # max_stride = max(int(T * max_stride_factor), stride * 2)

    # ans = localize(
    #     video_feature=feats,
    #     duration=dur,
    #     query_json=tfvtg_query_json,
    #     stride=stride,
    #     max_stride=max_stride,
    # )

    # # 将所有 query 的候选片段打平成 proposals: [start, end, confidence]
    # proposals: List[List[float]] = []
    # for query_ans in ans:
    #     resp = query_ans.get("response", [])
    #     for r in resp:
    #         proposals.append([
    #             float(r.get("start", 0.0)),
    #             float(r.get("end", 0.0)),
    #             float(r.get("confidence", 0.0)),
    #         ])

    # if not proposals:
    #     raise RuntimeError("TFVTG 未返回任何候选片段")

    # proposals_np = np.array(proposals, dtype=float)
    # # 使用 TFVTG 提供的 select_proposal 选择最优候选
    # ranked = select_proposal(proposals_np)
    # best = ranked[0]
    # start, end, local_conf = float(best[0]), float(best[1]), float(best[2])

    # # 额外计算一个 raw_score_max（使用所有 descriptions）用于日志
    # all_descs: List[str] = []
    # for item in query_items:
    #     descs = item.get("descriptions") or []
    #     if isinstance(descs, list):
    #         all_descs.extend(descs)
    # if not all_descs:
    #     all_descs = ["basketball player takes a jump shot and the ball goes into the basket"]

    # scores = calc_scores(feats, all_descs)
    # raw_score = float(scores.detach().cpu().numpy()[0].max())

    # print("\nTFVTG 结果（多 query）：")
    # print(f"raw={raw_score:.4f} | local={local_conf:.4f} | {start:.2f}s ~ {end:.2f}s")

    # if end <= start:
    #     raise RuntimeError(f"时间段异常：start={start}, end={end}")

    # # 4) 剪辑输出（lavis env）
    # out_mp4 = OUTPUT_DIR / "pipeline_during_best.mp4"
    # print("\n=== [D] 剪辑输出（单视频）===")
    # print(f"video={video_path} raw={raw_score:.4f}")
    # print(f"output={out_mp4}")

    # clip = VideoFileClip(video_path).subclipped(start, end)
    # final = concatenate_videoclips([clip], method="compose")
    # final.write_videofile(str(out_mp4), codec="libx264", audio_codec="aac")
    # clip.close()
    # final.close()

    # 2) 找最新 analysis 文件并读 Qwen JSON 输出
    analysis_path = find_latest_analysis_file()
    query_json = load_query_json(analysis_path)

    print("\n=== [B] 读取 Qwen 输出（latest analysis）===")
    print(f"analysis: {analysis_path}")
    print("query_json:")
    for item in query_json:
        print(item)

    # 3) TFVTG 对同一个「已裁剪」视频做多 subtask 匹配（lavis env）
    print("\n=== [C] TFVTG Subtask 匹配（conda env: lavis）===")

    # 复用前面裁剪好的 seg_video，确保 TFVTG 匹配的也是 trigger 前后的视频片段
    video_path = seg_video

    print(f"-> single_video: {video_path}")
    feats, dur = extract_tfvtg_features(video_path, fps=4.0)

    # 遍历每个 sub_query，选 raw_score 最大的
    best = None
    for item in query_json:
        descs = item.get("descriptions") or []
        if not isinstance(descs, list) or not descs:
            continue

        sub_id = item.get("sub_query_id", -1)
        print(f"\n--- TFVTG 子查询 sub_query_id={sub_id} ---")
        raw_score, s, e, local_conf = score_and_localize_during(feats, dur, descs)
        print(f"  raw={raw_score:.4f} | local={local_conf:.4f} | {s:.2f}s ~ {e:.2f}s")

        if e <= s:
            continue

        if (best is None) or (raw_score > best["raw"]):
            best = {
                "raw": raw_score,
                "start": s,
                "end": e,
                "conf": local_conf,
                "sub_id": sub_id,
                "descs": descs,
            }

    if best is None:
        raise RuntimeError("所有 sub_query 都未能找到合法时间段，无法剪辑。")

    start = best["start"]
    end = best["end"]
    raw_score = best["raw"]
    local_conf = best["conf"]

    print("\nTFVTG 最优子查询结果：")
    print(f"sub_query_id={best['sub_id']}")
    print("descriptions:")
    for d in best["descs"]:
        print("  -", d)
    print(f"raw={raw_score:.4f} | local={local_conf:.4f} | {start:.2f}s ~ {end:.2f}s")

    if end <= start:
        raise RuntimeError(f"时间段异常：start={start}, end={end}")

    # 4) 剪辑输出（lavis env）——下面保持原逻辑不变
    out_mp4 = OUTPUT_DIR / "pipeline_during_best.mp4"
    print("\n=== [D] 剪辑输出（单视频）===")
    print(f"video={video_path} raw={raw_score:.4f}")
    print(f"output={out_mp4}")

    clip = VideoFileClip(video_path).subclipped(start, end)
    final = concatenate_videoclips([clip], method="compose")
    final.write_videofile(str(out_mp4), codec="libx264", audio_codec="aac")
    clip.close()
    final.close()


if __name__ == "__main__":
    # --- User config: 同一个视频 + 触发时间 + 前后范围 ---
    USER_VIDEO   = str(DATA_DIR / "camera6_from_1h50m_to_end.mp4")  # 这里换成你真正想用的单个视频
    TRIGGER_TIME = 93.0   # 触发时间（秒）
    PRE_SECONDS  = 8.0    # 触发前窗口
    POST_SECONDS = 8.0    # 触发后窗口
    main()