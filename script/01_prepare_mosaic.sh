#!/bin/bash

# 配置路径
DATA_DIR="/home/SONY/s7000043396/Downloads/demo/data/demo-data"
OUTPUT_DIR="/home/SONY/s7000043396/Downloads/demo/script"
VIDEO_FILES=(
    "atc1_cam_43.mp4" "atc2_render_cam_45.mp4" "atc3_render_cam_0.mp4"
    "atc4_render_cam_36.mp4" "def2_render_cam_39.mp4" "int1_render_cam_9.mp4"
)

echo "--- 开始生成 3x2 拼接预览流 ---"

# 构建 FFmpeg 输入参数
INPUT_ARGS=""
for f in "${VIDEO_FILES[@]}"; do
    INPUT_ARGS="$INPUT_ARGS -i $DATA_DIR/$f"
done

# 使用 xstack 滤镜拼接 (3x2 布局)
# hstack/vstack 兼容性更好，这里采用你要求的 2026 高效方式
ffmpeg $INPUT_ARGS -filter_complex \
"[0:v]scale=640:360[v0]; [1:v]scale=640:360[v1]; [2:v]scale=640:360[v2]; \
 [3:v]scale=640:360[v3]; [4:v]scale=640:360[v4]; [5:v]scale=640:360[v5]; \
 [v0][v1][v2]hstack=inputs=3[top]; \
 [v3][v4][v5]hstack=inputs=3[bottom]; \
 [top][bottom]vstack=inputs=2[v]" \
-map "[v]" -c:v libx264 -preset ultrafast -shortest $OUTPUT_DIR/mosaic_preview_720p.mp4 -y

echo "--- 预览流生成完毕: mosaic_preview_720p.mp4 ---"

# 生成 Camera_Layout.json
cat <<EOF > $OUTPUT_DIR/Camera_Layout.json
{
  "layout": "3x2",
  "mapping": [
    {"grid_pos": [0,0], "file": "${VIDEO_FILES[0]}"},
    {"grid_pos": [0,1], "file": "${VIDEO_FILES[1]}"},
    {"grid_pos": [0,2], "file": "${VIDEO_FILES[2]}"},
    {"grid_pos": [1,0], "file": "${VIDEO_FILES[3]}"},
    {"grid_pos": [1,1], "file": "${VIDEO_FILES[4]}"},
    {"grid_pos": [1,2], "file": "${VIDEO_FILES[5]}"}
  ]
}
EOF
echo "--- 机位布局文件已生成: Camera_Layout.json ---"