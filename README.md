## Simulated Data Source

Backup data source: https://humansensinglab.github.io/basket-multiview/data.html

After downloading, you can do like this:

```bash
ffmpeg -framerate 25 -i %04d.png -c:v libx264 -pix_fmt yuv420p xxx.mp4
```

## Running 

```bash
cd script
bash pipeline.sh
```

```mermaid
graph TD
    A[Multi-Cam Live Stream] --> B{Step 1: Scoreboard Detection}
    B -->|Score Change| C[Log TriggerTime - JSON List]
    
    C --> D{Step 2: Action Analysis - TSM}
    D -->|T-10s Retro-analysis| E[Probability Curve & Segments]
    E -->|Ranking| F[Top N% High-Prob Segments]
    
    F --> G{Step 3: Best View Selection - Qwen2.5-VL}
    G -->|Multi-View Input| H[Compare Cam 1 / 2 / 3...]
    H -->|Reasoning| I[Select Best Camera Angle]
    
    I --> J{Step 4: Final Production}
    J --> K[Automated Clipping & Stitching]
    K --> L[Final Highlight Video]
```
