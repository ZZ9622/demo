import subprocess

def download_video_remainder_with_token(camera_id, hf_token, start_time="01:50:00"):
    """
    携带 Hugging Face Token，从指定时间点下载直到视频结束的所有内容
    """
    video_url = (
        "https://huggingface.co/datasets/SportsHL-Team/Sports_highlight_generation/"
        f"resolve/main/apidis/camera{camera_id}/camera{camera_id}_full_stitched_interpolated.mp4"
    )
    
    output_filename = f"camera{camera_id}_from_1h50m_to_end.mp4"
    
    # 构造带有 Token 的 HTTP Header
    # 注意：FFmpeg 要求的 Header 格式末尾必须带有回车换行符 \r\n
    auth_header = f"Authorization: Bearer {hf_token}\r\n"
    
    # FFmpeg 命令
    command = [
        "ffmpeg",
        "-headers", auth_header,  # 将 Token 放进请求头中（必须放在 -i 前面）
        "-ss", start_time,        # 起点：1小时50分
        "-i", video_url,          # 直链输入
        "-c", "copy",             # 复制音视频流，不重新编码
        "-y",                     # 覆盖同名文件
        output_filename
    ]
    
    print(f"🔑 正在使用 Token 验证并下载 Camera {camera_id} 从 {start_time} 到结尾的全部视频...")
    
    try:
        # 运行 FFmpeg 命令
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"✅ 下载完成！已保存为母盘文件：{output_filename}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败，请检查 Token 是否有效，或网络是否正常。\n")

# --- 使用示例 ---

# 替换为你自己的 Hugging Face Token (获取地址: https://huggingface.co/settings/tokens)
# MY_HF_TOKEN = "hf_xxx"  # 请替换为你的实际 Token

# 提取 camera2 从 1h50m 到结尾的内容
# download_video_remainder_with_token(camera_id=1, hf_token=MY_HF_TOKEN)

# 如果你需要同时下载多个摄像头，可以取消下面的注释：
for cam in range(2, 8):
    download_video_remainder_with_token(camera_id=cam, hf_token=MY_HF_TOKEN)