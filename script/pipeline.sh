#!/bin/bash
# 显式使用 bash 运行，并开启遇错即停
set -e

echo "🚀 [RTX 5090] Pipeline started..."

# 1. 拼接
python 01_prepare_mosaic.py

# 2. 特征索引
python 02_feature_encoding.py

# 3. AI 决策
python 03_ai_director.py

# 4. 渲染
python 04_render_final.py

echo "🎉 all processes completed!"