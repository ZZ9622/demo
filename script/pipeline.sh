#!/bin/bash
# pipeline.sh 
# This script is used to run the pipeline for the demo

cd /home/SONY/s7000043396/Downloads/demo/script

echo "start to run the pipeline..."

echo "activate the virtual environment..."
source /home/SONY/s7000043396/hlvenv/bin/activate
echo "virtual environment activated"

echo "run the 01_prepare_mosaic.sh..."
bash 01_prepare_mosaic.sh
echo "01_prepare_mosaic.sh finished"

echo "run the 02_feature_encoding.py..."
python  02_feature_encoding.py
echo "02_feature_encoding.py finished"


echo "run the 03_otio.py..."
python 03_otio.py
echo "03_otio.py finished"

echo "run the 04_render_final_video.py..."
python 04_render_final_video.py
echo "04_render_final_video.py finished"

echo "pipeline finished"