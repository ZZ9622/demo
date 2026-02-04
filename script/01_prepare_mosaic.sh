#!/bin/bash

# 配置路径
DATA_DIR="/home/SONY/s7000043396/Downloads/demo/data/demo-data"
OUTPUT_DIR="/home/SONY/s7000043396/Downloads/demo/script"
VIDEO_FILES=(
    "def2_cam_00.mp4" "def2_cam_08.mp4" "def2_cam_15.mp4"
    "def2_cam_23.mp4" "def2_cam_46.mp4" "def2_cam_51.mp4"
    "def2_cam_62.mp4" "def2_cam_73.mp4" 

)

echo "start to generate mosaic preview video 4 x 2 "

# 构建 FFmpeg 输入参数
INPUT_ARGS=""
for f in "${VIDEO_FILES[@]}"; do
    INPUT_ARGS="$INPUT_ARGS -i $DATA_DIR/$f"
done

# 使用 xstack 滤镜拼接 (3x2 布局)
# hstack/vstack 兼容性更好，这里采用你要求的 2026 高效方式
ffmpeg $INPUT_ARGS -filter_complex \
"[0:v]scale=480:270[v0]; [1:v]scale=480:270[v1]; [2:v]scale=480:270[v2]; [3:v]scale=480:270[v3]; \
 [4:v]scale=480:270[v4]; [5:v]scale=480:270[v5]; [6:v]scale=480:270[v6]; [7:v]scale=480:270[v7]; \
 [v0][v1][v2][v3]hstack=inputs=4[top]; \
 [v4][v5][v6][v7]hstack=inputs=4[bottom]; \
 [top][bottom]vstack=inputs=2[v]" \
-map "[v]" -c:v libx264 -preset ultrafast -shortest $OUTPUT_DIR/mosaic_preview_720p.mp4 -y

echo "mosaic preview video generated: mosaic_preview_720p.mp4"

# 生成 Camera_Layout.json
cat <<EOF > $OUTPUT_DIR/Camera_Layout.json
{
  "layout": "4x2",
  "mapping": [
    {"grid_pos": [0,0], "file": "${VIDEO_FILES[0]}"},
    {"grid_pos": [0,1], "file": "${VIDEO_FILES[1]}"},
    {"grid_pos": [0,2], "file": "${VIDEO_FILES[2]}"},
    {"grid_pos": [0,3], "file": "${VIDEO_FILES[3]}"},
    {"grid_pos": [1,0], "file": "${VIDEO_FILES[4]}"},
    {"grid_pos": [1,1], "file": "${VIDEO_FILES[5]}"},
    {"grid_pos": [1,2], "file": "${VIDEO_FILES[6]}"},
    {"grid_pos": [1,3], "file": "${VIDEO_FILES[7]}"}
  ]
}
EOF
echo "Camera_Layout.json generated"