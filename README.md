```mermaid
graph TD
    subgraph Input Processing
    A[Wide Camera Video] -->|Split| B(Video Stream)
    A -->|Split| C(Audio Stream)
    end

    subgraph Stream 1: Audio 2D CNN
    C -->|STFT| D[Mel-Spectrogram Image]
    D --> E[2D CNN (Simple Custom)]
    E --> F[FC Layer + Sigmoid]
    F --> G[Audio Score (0-1)]
    end

    subgraph Stream 2: Video 2D CNN
    B -->|Grayscale + Stack Frames| H[Stacked Tensor (H x W x Depth)]
    H --> I[Modified Backbone (e.g., ResNet)]
    I -->|Feature Extraction| J[FC Layer + Sigmoid]
    J --> K[Video Score (0-1)]
    end

    subgraph Ensemble & Output
    G & K --> L[Average Fusion]
    L --> M[Sliding Window Smoothing]
    M --> N[Thresholding > 0.5]
    N --> O[JSON Event Generator]
    end


json example 

{
  "match_id": "game_2024_05_21_gsw_vs_lal",
  "fps": 25,
  "events": [
    {
      "id": 101,
      "timestamp_start": "00:04:12.500",
      "timestamp_end": "00:04:17.000",
      "duration": 4.5,
      "primary_label": "Dunk",
      "confidence": 0.92,
      "source_analysis": {
        "visual_score": 0.88, 
        "audio_score": 0.96,
        "note": "Audio spike detected (Rim sound)"
      },
      "clip_filename": "clip_101_dunk.mp4" 
    },
    {
      "id": 102,
      "timestamp_start": "00:06:45.000",
      "timestamp_end": "00:06:50.000",
      "duration": 5.0,
      "primary_label": "Goal", 
      "confidence": 0.85,
      "source_analysis": {
        "visual_score": 0.89,
        "audio_score": 0.60,
        "note": "Visual strong, Audio weak (Likely regular shot)"
      }
    }
  ]
}