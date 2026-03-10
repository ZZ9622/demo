import os
import cv2
import tempfile
from typing import List, Tuple

from vllm import LLM, SamplingParams

# ==========================
# 参数配置
# ==========================
# 你当前工程的根目录是 demo，脚本放在 script 目录下
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "apidis", "camera3_from_1h50m_to_end.mp4")

# 只分析 85s ~ 95s
START_SEC = 85.0
END_SEC = 95.0

# 从视频中抽帧的目标 FPS（越低越快）
SAMPLE_FPS = 3.0

# 选择一个支持图像输入的 vLLM 模型，这里用 Qwen2-VL 作为示例
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"


# ==========================
# 工具函数：抽帧
# ==========================
def extract_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    tmp_dir: str,
) -> List[Tuple[float, str]]:
    """
    从视频中截取 [start_sec, end_sec] 区间的帧，并以 sample_fps 采样，
    保存到 tmp_dir，返回 (时间戳, 图像路径) 列表。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # 按采样帧率计算应该跳过多少原始帧
    frame_skip = max(1, int(round(fps / sample_fps)))

    frames_info: List[Tuple[float, str]] = []

    # 先跳到 start_sec
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    frame_index = 0
    while True:
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_sec = current_msec / 1000.0

        if current_sec > end_sec:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # 只每 frame_skip 帧保存一次
        if frame_index % frame_skip == 0:
            # 保存到临时目录
            filename = os.path.join(tmp_dir, f"frame_{current_sec:.2f}.jpg")
            cv2.imwrite(filename, frame)
            frames_info.append((current_sec, filename))

        frame_index += 1

    cap.release()
    return frames_info


# ==========================
# 调用 vLLM 模型判断每一帧
# ==========================
def build_llm(model_name: str) -> LLM:
    """
    初始化 vLLM，多模态模型（支持图像）。
    需要你本地已经下载好对应权重或能从网络拉取。
    """
    llm = LLM(
        model=model_name,
        trust_remote_code=True,  # 部分多模态模型需要
    )
    return llm


def is_goal_frame(llm: LLM, image_path: str, ts: float) -> bool:
    prompt = (
        "你是一个篮球视频分析助手。下面是一帧球场比赛画面，"
        f"这帧在当前机位片段中的时间约为 {ts:.2f} 秒。\n"
        "请你判断：这一帧是否显示了篮球已经在篮筐内/篮网内，或者刚刚穿过篮筐的瞬间。\n"
        "如果是，请只回答大写单词 IN；如果不是，请只回答大写单词 OUT；不要输出其他内容。"
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8,
    )

    # vLLM 多模态：每个请求是一个 dict，包含 prompt 和 multi_modal_data
    requests = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "images": [image_path],  # 或 "images"，取决于你安装的 vLLM 版本，错了会直接报 key 错
            },
        }
    ]

    outputs = llm.generate(requests, sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip().upper()

    if "IN" in text and "OUT" not in text:
        return True
    return False


# ==========================
# 主逻辑：找出进球时间点
# ==========================
def find_goal_times():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"找不到视频文件: {VIDEO_PATH}")

    print(f"🎬 视频: {VIDEO_PATH}")
    print(f"⏱ 分析时间区间: {START_SEC:.1f}s ~ {END_SEC:.1f}s")
    print(f"🧠 使用 vLLM 模型: {MODEL_NAME}")

    # 1. 抽帧（用临时目录保存图片，避免占用太多内存）
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("⏳ 正在抽取帧并保存临时图片...")
        frames_info = extract_frames(
            VIDEO_PATH,
            START_SEC,
            END_SEC,
            SAMPLE_FPS,
            tmp_dir,
        )
        print(f"共抽取 {len(frames_info)} 帧用于分析。")

        if not frames_info:
            print("⚠️ 在指定时间段内没有抽到任何帧。")
            return

        # 2. 初始化 vLLM
        print("⏳ 正在加载 vLLM 模型（可能需要一些时间）...")
        llm = build_llm(MODEL_NAME)
        print("✅ 模型加载完成，开始逐帧分析。")

        goal_candidates: List[float] = []

        for idx, (ts, img_path) in enumerate(frames_info, start=1):
            print(f"[{idx}/{len(frames_info)}] 分析帧 @ {ts:.2f}s : {os.path.basename(img_path)}")
            try:
                if is_goal_frame(llm, img_path, ts):
                    print(f" 👉 模型判断为进球相关帧 (IN) @ {ts:.2f}s")
                    goal_candidates.append(ts)
                else:
                    print("    模型判断为非进球帧 (OUT)")
            except Exception as e:
                print(f"    调用模型出错: {e}")

        if not goal_candidates:
            print("\n❌ 在 85s~95s 区间内，模型没有发现明显的进球帧。")
            return

        # 3. 对所有候选时间做一个简单汇总：取最早和中位数作为参考
        goal_candidates_sorted = sorted(goal_candidates)
        earliest = goal_candidates_sorted[0]
        median = goal_candidates_sorted[len(goal_candidates_sorted) // 2]

        print("\n✅ 进球时间候选（相对当前 cam3 片段的秒数）：")
        print("所有候选帧时间点：", ", ".join(f"{t:.2f}s" for t in goal_candidates_sorted))
        print(f"建议进球时间（最早候选）：{earliest:.2f}s")
        print(f"建议进球时间（中位候选）：{median:.2f}s")


if __name__ == "__main__":
    find_goal_times()