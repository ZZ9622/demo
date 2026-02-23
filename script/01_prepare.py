# !/usr/bin/env python3

"""
This script is used to generate the camera URLs.
"""

import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_URL = "https://huggingface.co/datasets/SportsHL-Team/Sports_highlight_generation/resolve/main/apidis"

def main():
    cameras = []
    for i in range(1, 8):
        url = f"{DATASET_URL}/camera{i}/camera{i}_full_stitched_interpolated.mp4"
        cameras.append({"id": i, "url": url})
    
    output_file = os.path.join(OUTPUT_DIR, "camerasurls.json")
    with open(output_file, "w") as f:
        json.dump(cameras, f, indent=2)
    
    print(f"Generated {len(cameras)} camera URLs -> {output_file}")

if __name__ == "__main__":
    main()