#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python yolo.py --trigger_time 57 --simple
"""
Basketball View Selector using YOLO People Detection

Analyzes left and right court views to determine which video to use for TSM model input.
Selects the view with higher average people count during specified time period.
使用示例：
import sys
sys.path.append('/home/SONY/s7000043396/Downloads/demo/script/yolo')
from LRviewselection import get_view_selection

# 一行调用，获得视角选择
selected_view = get_view_selection(57)

if selected_view == 1:
    print("使用左半场视角")
    camera = "camera4" 
elif selected_view == 2:
    print("使用右半场视角")
    camera = "camera6"
else:
    print("分析失败，使用默认")
    camera = "camera4"
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from typing import Tuple, Dict
import json
from pathlib import Path

class BasketballViewSelector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initialize basketball view selector
        
        Args:
            model_path: YOLO model path (default: YOLOv8n)
        """
        self.model = YOLO(model_path)
        self.left_video_path = "/home/SONY/s7000043396/Downloads/demo/data/apidis/camera4_from_1h50m_to_end.mp4"
        self.right_video_path = "/home/SONY/s7000043396/Downloads/demo/data/apidis/camera6_from_1h50m_to_end.mp4"
        print(f"Loaded: {model_path}")
        
    def extract_video_segment(self, video_path: str, start_time: float, end_time: float, 
                            output_path: str = None) -> str:
        """
        Extract video segment
        
        Args:
            video_path: Input video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output path, generates temp file if None
            
        Returns:
            Path to extracted video segment
        """
        if output_path is None:
            output_path = f"temp_segment_{start_time}_{end_time}.mp4"
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算帧范围
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
            
        cap.release()
        out.release()
        
        return output_path
        
    def count_people_in_video(self, video_path: str, confidence_threshold: float = 0.3) -> Dict:
        """
        Count people in video using YOLO detection
        
        Args:
            video_path: Path to video file
            confidence_threshold: Detection confidence threshold (default 0.3)
            
        Returns:
            Dictionary containing people count statistics
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        people_counts = []
        frame_count = 0
        max_detections_frame = 0
        
        print(f"Analyzing: {os.path.basename(video_path)}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每隔几帧检测一次以提高效率 - 改为每3帧检测一次
            if frame_count % 3 == 0:
                results = self.model(frame, verbose=False)
                
                # 计算当前帧的人数（class 0 是人）- 修复累加逻辑
                people_in_frame = 0
                all_detections = []
                
                # 修复：正确累加所有检测结果
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        # Filter for person class (0) with confidence threshold
                        person_mask = (boxes.cls == 0) & (boxes.conf >= confidence_threshold)
                        person_detections = boxes[person_mask]
                        
                        if len(person_detections) > 0:
                            people_in_frame += len(person_detections)
                
                people_counts.append(people_in_frame)
            
            frame_count += 1
            
        cap.release()
        
        # Calculate statistics
        avg_people = float(np.mean(people_counts)) if people_counts else 0.0
        max_people = int(np.max(people_counts)) if people_counts else 0
        total_detections = int(np.sum(people_counts)) if people_counts else 0
        
        print(f"Complete: avg={avg_people:.2f}, max={max_people}")
        
        return {
            'average_people': avg_people,
            'max_people': max_people,
            'total_detections': total_detections,
            'frames_analyzed': len(people_counts)
        }
        
    def analyze_and_select_view(self, trigger_time: int, analysis_duration: int = 10) -> Dict:
        """
        Analyze and select best camera view
        
        Args:
            trigger_time: Trigger time in seconds
            analysis_duration: Analysis duration in seconds (default 10)
            
        Returns:
            Analysis results and selected video information
        """
        start_time = trigger_time - analysis_duration
        end_time = trigger_time
        
        print(f"Analysis: t={trigger_time}s, period={start_time}s-{end_time}s")
        
        # Check video files exist
        if not os.path.exists(self.left_video_path):
            raise FileNotFoundError(f"Left video not found: {self.left_video_path}")
        if not os.path.exists(self.right_video_path):
            raise FileNotFoundError(f"Right video not found: {self.right_video_path}")
        
        # Extract video segments
        left_segment = self.extract_video_segment(
            self.left_video_path, start_time, end_time, "temp_left_segment.mp4"
        )
        
        right_segment = self.extract_video_segment(
            self.right_video_path, start_time, end_time, "temp_right_segment.mp4"
        )
        
        try:
            # Analyze people count in both segments
            left_stats = self.count_people_in_video(left_segment)
            right_stats = self.count_people_in_video(right_segment)
            
            # Select view based on people count
            selected_view = "left" if left_stats['average_people'] > right_stats['average_people'] else "right"
            selected_video_path = self.left_video_path if selected_view == "left" else self.right_video_path
            
            result = {
                'trigger_time': trigger_time,
                'analysis_period': f"{start_time}s - {end_time}s",
                'left_stats': left_stats,
                'right_stats': right_stats,
                'selected_view': selected_view,
                'selected_video_path': selected_video_path,
                'selection_reason': f"{selected_view} view selected (avg: {left_stats['average_people']:.2f} vs {right_stats['average_people']:.2f})"
            }
            
            print(f"Selected: {selected_view} view")
            
            return result
            
        finally:
            # Clean up temporary files
            for temp_file in [left_segment, right_segment]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def create_simplified_result(self, results: Dict) -> Dict:
        """
        Create simplified result format for program integration
        
        Args:
            results: Complete analysis results
            
        Returns:
            Simplified result dictionary
        """
        # Convert view to numeric: left=1, right=2
        selected_view_code = 1 if results['selected_view'] == 'left' else 2
        
        simplified = {
            'trigger_time': results['trigger_time'],
            'left_avg_people': round(results['left_stats']['average_people'], 2),
            'right_avg_people': round(results['right_stats']['average_people'], 2),
            'selected_view': selected_view_code,  # 1=left, 2=right
            'selected_video_path': results['selected_video_path']
        }
        
        return simplified
                    
    def save_results(self, results: Dict, output_path: str = "/home/SONY/s7000043396/Downloads/demo/output/view_selection_results.json", simplified: bool = False):
        """
        Save analysis results to file with JSON serialization fix
        
        Args:
            results: Analysis results
            output_path: Output file path
            simplified: Whether to use simplified format
        """
        # 如果需要简化格式，转换数据
        if simplified:
            data_to_save = self.create_simplified_result(results)
        else:
            data_to_save = results
        # 自定义JSON编码器，处理numpy数据类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换所有numpy类型
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            else:
                return convert_numpy_types(data)
        
        cleaned_results = clean_for_json(data_to_save)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
        
        print(f"Saved: {output_path}")


# Global selector instance to avoid reinitializing YOLO model
_global_selector = None

def get_selector(model_path='yolov8n.pt'):
    """
    Get or create global YOLO selector instance
    
    Args:
        model_path: YOLO model path
        
    Returns:
        BasketballViewSelector instance
    """
    global _global_selector
    if _global_selector is None:
        _global_selector = BasketballViewSelector(model_path)
    return _global_selector


def get_view_selection(trigger_time, analysis_duration=10, model_path='yolov8n.pt'):
    """
    Get view selection for given trigger time (simple interface)
    
    Args:
        trigger_time: Trigger time in seconds
        analysis_duration: Analysis duration in seconds (default: 10)
        model_path: YOLO model path (default: yolov8n.pt)
        
    Returns:
        int: Selected view (1=left, 2=right) or None if failed
    """
    try:
        selector = get_selector(model_path)
        results = selector.analyze_and_select_view(trigger_time, analysis_duration)
        
        # Convert to numeric code
        return 1 if results['selected_view'] == 'left' else 2
        
    except Exception as e:
        print(f"Error in view selection: {e}")
        return None


def get_detailed_analysis(trigger_time, analysis_duration=10, model_path='yolov8n.pt'):
    """
    Get detailed analysis results for given trigger time
    
    Args:
        trigger_time: Trigger time in seconds
        analysis_duration: Analysis duration in seconds (default: 10)
        model_path: YOLO model path (default: yolov8n.pt)
        
    Returns:
        dict: Detailed analysis results or None if failed
    """
    try:
        selector = get_selector(model_path)
        results = selector.analyze_and_select_view(trigger_time, analysis_duration)
        
        # Create simplified result
        simplified = selector.create_simplified_result(results)
        return simplified
        
    except Exception as e:
        print(f"Error in detailed analysis: {e}")
        return None


def get_selected_video_path(trigger_time, analysis_duration=10, model_path='yolov8n.pt'):
    """
    Get selected video path for given trigger time
    
    Args:
        trigger_time: Trigger time in seconds
        analysis_duration: Analysis duration in seconds (default: 10)
        model_path: YOLO model path (default: yolov8n.pt)
        
    Returns:
        str: Selected video path or None if failed
    """
    try:
        selector = get_selector(model_path)
        results = selector.analyze_and_select_view(trigger_time, analysis_duration)
        return results['selected_video_path']
        
    except Exception as e:
        print(f"Error getting video path: {e}")
        return None


def batch_view_selection(trigger_times, analysis_duration=10, model_path='yolov8n.pt'):
    """
    Get view selections for multiple trigger times
    
    Args:
        trigger_times: List of trigger times in seconds
        analysis_duration: Analysis duration in seconds (default: 10)
        model_path: YOLO model path (default: yolov8n.pt)
        
    Returns:
        dict: Mapping of trigger_time -> selected_view (1 or 2)
    """
    results = {}
    selector = get_selector(model_path)
    
    for trigger_time in trigger_times:
        try:
            analysis = selector.analyze_and_select_view(trigger_time, analysis_duration)
            results[trigger_time] = 1 if analysis['selected_view'] == 'left' else 2
        except Exception as e:
            print(f"Error analyzing trigger_time {trigger_time}: {e}")
            results[trigger_time] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Basketball view selector using YOLO people detection')
    parser.add_argument('--trigger_time', type=int, default=57,
                       help='Trigger time in seconds (default: 57)')
    parser.add_argument('--analysis_duration', type=int, default=10,
                       help='Analysis duration in seconds (default: 10)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Detection confidence threshold (default: 0.3)')
    parser.add_argument('--output', type=str, default='/home/SONY/s7000043396/Downloads/demo/output/view_selection_results.json',
                       help='Output file path')
    parser.add_argument('--simple', action='store_true',
                       help='Use simplified output format')
    
    args = parser.parse_args()
    
    print(f"YOLO Basketball View Selector")
    print(f"Trigger: {args.trigger_time}s, Duration: {args.analysis_duration}s, Model: {args.model}")
    
    # Initialize selector
    try:
        selector = BasketballViewSelector(args.model)
    except Exception as e:
        print(f"Error: Failed to initialize YOLO model - {e}")
        return None
    
    try:
        # Execute analysis
        results = selector.analyze_and_select_view(args.trigger_time, args.analysis_duration)
        
        # Save results
        selector.save_results(results, args.output, simplified=args.simple)
        
        return results['selected_video_path']
        
    except Exception as e:
        print(f"Error: Analysis failed - {e}")
        return None


if __name__ == "__main__":
    selected_video = main()
    if selected_video:
        print(f"Analysis complete. Selected video: {os.path.basename(selected_video)}")
    else:
        print("Analysis failed.")