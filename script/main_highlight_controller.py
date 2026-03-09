from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# 确保能导入 pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Sports_HL
DEMO_ROOT = PROJECT_ROOT / "demo"
DATA_DIR = DEMO_ROOT / "data" / "apidis"

sys.path.append(str(PROJECT_ROOT / "demo" / "script"))
from pipeline import run_grounding  # type: ignore

# ====================== 全局配置 ======================

# YOLO 模型权重路径（如已放在默认路径，可直接用 "yolov8n.pt"）
YOLO_WEIGHTS: str = "yolov8n.pt"

# Grounding 时在 trigger 前后取多少秒的视频片段
PRE_SECONDS: float = 8.0
POST_SECONDS: float = 8.0

# 全局 YOLO 模型对象，程序启动时加载一次
YOLO_MODEL: Optional[YOLO] = None


def load_yolo_model() -> YOLO:
    """
    加载 YOLOv8n 模型，只在进程启动时调用一次。
    """
    global YOLO_MODEL
    if YOLO_MODEL is None:
        try:
            YOLO_MODEL = YOLO(YOLO_WEIGHTS)
        except Exception as e:
            raise RuntimeError(f"加载 YOLO 模型失败: {e}")
    return YOLO_MODEL


def read_frame_at_time(video_path: str, time_sec: float) -> Tuple[np.ndarray, float]:
    """
    从离线视频中读取指定时间点的一帧图像。

    :param video_path: 视频文件路径
    :param time_sec: 触发时间（秒）
    :return: (frame, duration_sec)，frame 为 BGR 图像，duration_sec 为视频总时长（秒）
    :raises ValueError: 时间越界或读取失败时抛出
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"视频属性异常: fps={fps}, frame_count={frame_count}")

        duration_sec: float = frame_count / fps

        if time_sec < 0 or time_sec > duration_sec:
            raise ValueError(
                f"trigger_time 超出视频长度: trigger={time_sec:.3f}s, duration={duration_sec:.3f}s"
            )

        # 计算应当读取的帧索引，注意边界
        frame_idx = int(round(time_sec * fps))
        frame_idx = min(max(frame_idx, 0), int(frame_count) - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"在时间 {time_sec:.3f}s (frame={frame_idx}) 读取帧失败")

        return frame, duration_sec
    finally:
        # 确保资源释放
        cap.release()


def count_persons_in_frame(frame: np.ndarray, model: YOLO) -> int:
    """
    使用 YOLO 模型对单帧图像进行推理，只统计 person 类别的数量。

    :param frame: BGR 图像 (H, W, 3)
    :param model: 已加载的 YOLO 模型
    :return: person 数量
    """
    try:
        results = model(frame)
    except Exception as e:
        raise RuntimeError(f"YOLO 推理失败: {e}")

    if not results:
        return 0

    result = results[0]
    if result.boxes is None or result.boxes.cls is None:
        return 0

    cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    # YOLOv8 中 person 的类别 ID 为 0
    person_count = int((cls_ids == 0).sum())
    return person_count


def process_highlight(trigger_time: float, cam1_path: str, cam6_path: str) -> None:
    """
    基于 trigger_time，对 cam1 / cam6 两路视频做帧数比较并调用 Grounding。

    流程：
      1. 从两路视频在 trigger_time 处截取单帧。
      2. 使用 YOLOv8n 分别统计两帧中的 person 数量。
      3. 比较人数，选择人数较多的机位（平局则默认选择 cam1）。
      4. 调用 pipeline.py 中的 run_grounding，对选中机位完整视频做 Grounding。

    :param trigger_time: 触发时间（秒）
    :param cam1_path: 左半场视频路径
    :param cam6_path: 右半场视频路径
    """
    model = load_yolo_model()

    # 1. 读取两路视频在 trigger_time 的帧
    frame_cam1: np.ndarray
    frame_cam6: np.ndarray

    try:
        frame_cam1, dur1 = read_frame_at_time(cam1_path, trigger_time)
    except Exception as e:
        raise RuntimeError(f"读取 cam1 帧失败: {e}")

    try:
        frame_cam6, dur6 = read_frame_at_time(cam6_path, trigger_time)
    except Exception as e:
        raise RuntimeError(f"读取 cam6 帧失败: {e}")

    # 2. 分别统计 person 数量
    try:
        cnt_cam1 = count_persons_in_frame(frame_cam1, model)
        cnt_cam6 = count_persons_in_frame(frame_cam6, model)
    except Exception as e:
        raise RuntimeError(f"YOLO 人数统计失败: {e}")

    print(f"[INFO] trigger_time={trigger_time:.3f}s, cam1_persons={cnt_cam1}, cam6_persons={cnt_cam6}")

    # 3. 决策逻辑：人数多的半场，如果平局，选 cam1
    if cnt_cam6 > cnt_cam1:
        selected_cam = "cam6"
        selected_video_path = cam6_path
    else:
        selected_cam = "cam1"
        selected_video_path = cam1_path

    print(f"[INFO] 选择机位: {selected_cam} (video={selected_video_path})")

    # 4. 调用 Grounding（注意：这里仍然基于完整视频和 trigger_time）
    #    run_grounding 内部会按照 pre/post 秒裁剪，并调用 Qwen + TFVTG。
    try:
        run_grounding(
            video_path=selected_video_path,
            trigger_time=trigger_time,
            pre_seconds=PRE_SECONDS,
            post_seconds=POST_SECONDS,
        )
    except Exception as e:
        raise RuntimeError(f"调用 run_grounding 失败: {e}")


def main() -> None:
    """
    简单示例入口：手动设置一个 trigger_time 测试流程。
    """
    # 示例：你给出的 cam1 路径
    cam1 = str(DATA_DIR / "camera1_from_1h50m_to_end.mp4")
    # 假设 cam6 文件名类似（请根据实际文件名修改）
    cam6 = str(DATA_DIR / "camera6_from_1h50m_to_end.mp4")

    trigger_time: float = 93.0  # 单位：秒，可按需修改或由外部事件传入

    try:
        process_highlight(trigger_time, cam1, cam6)
    except Exception as e:
        # 顶层捕获错误，防止程序无提示崩溃
        print(f"[ERROR] 高光处理失败: {e}")


if __name__ == "__main__":
    main()