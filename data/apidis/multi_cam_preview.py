import cv2
import os
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 7 路相机文件名（如有不同，请按实际改）
    video_files = [
        "camera1_from_1h50m_to_end.mp4",
        "camera2_from_1h50m_to_end.mp4",
        "camera3_from_1h50m_to_end.mp4",
        "camera4_from_1h50m_to_end.mp4",
        "camera5_from_1h50m_to_end.mp4",
        "camera6_from_1h50m_to_end.mp4",
        "camera7_from_1h50m_to_end.mp4",
    ]

    video_paths = [os.path.join(base_dir, vf) for vf in video_files]

    # 打开所有视频
    caps = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"⚠️ 无法打开视频: {p}")
            return
        caps.append(cap)

    # 用第一个视频的尺寸 / FPS 作为基准
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 每个小画面的目标尺寸（可以按需改）
    cell_w = width // 2
    cell_h = height // 2

    # 3x3 网格的总尺寸
    grid_cols, grid_rows = 3, 3
    out_w = cell_w * grid_cols
    out_h = cell_h * grid_rows

    # 输出视频（可选）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(base_dir, "multi_cam_10min.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    print(f"💾 输出视频: {out_path}")

    max_seconds = 10 * 60  # 10 分钟
    frame_idx = 0

    # 黑色占位图
    black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    # 把 7 个视频映射到 3x3 网格中的 7 个位置
    # 顺序：第 0 行 3 个格子 + 第 1 行 3 个格子 + 第 2 行前 1 个格子
    # 索引: 0..8  => (row = i//3, col = i%3)
    pos_to_cam = {0: 0, 1: 1, 2: 2,
                  3: 3, 4: 4, 5: 5,
                  6: 6}  # 7 和 8 留空黑屏

    while True:
        # 基于第一个视频的时间控制 10 分钟
        current_sec = frame_idx / fps
        if current_sec > max_seconds:
            break

        # 读取每一路当前帧
        frames = []
        any_fail = False
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                any_fail = True
                break
            # 缩放到 cell 大小
            frame_resized = cv2.resize(frame, (cell_w, cell_h))
            frames.append(frame_resized)

        if any_fail:
            print("⚠️ 有视频提前结束，停止拼接。")
            break

        # 组装 3x3 网格
        cells = []
        for idx in range(grid_rows * grid_cols):
            if idx in pos_to_cam:
                cam_idx = pos_to_cam[idx]
                cell = frames[cam_idx].copy()
                # 标注 CAM 号
                label = f"CAM{cam_idx + 1}"
                cv2.putText(
                    cell,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cell = black_cell.copy()
            cells.append(cell)

        # 按行拼接
        row0 = np.hstack(cells[0:3])
        row1 = np.hstack(cells[3:6])
        row2 = np.hstack(cells[6:9])
        grid_frame = np.vstack([row0, row1, row2])

        # 写入输出视频
        out.write(grid_frame)

        # 如需实时预览，可打开下面两行
        # cv2.imshow("Multi-CAM", grid_frame)
        # if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        #     break

        frame_idx += 1

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ 拼接完成（前 10 分钟）。")

if __name__ == "__main__":
    main()