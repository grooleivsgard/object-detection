# object-detection

Simple object detection using [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics).

## Model

Uses `yolov8n.pt` (nano variant) — pre-trained by Ultralytics on the [COCO dataset](https://cocodataset.org/) (118k images, 80 object categories including person, car, bicycle, dog, etc.). Downloaded automatically on first run (~6MB).

Swap `MODEL` in `main.py` for a larger variant if you need more accuracy:

| Model | Size | Speed |
|-------|------|-------|
| yolov8n | ~6MB | fastest |
| yolov8s | ~22MB | fast |
| yolov8m | ~52MB | balanced |
| yolov8l | ~87MB | accurate |
| yolov8x | ~131MB | most accurate |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Webcam
python main.py

# Image or video file
python main.py path/to/image.jpg
python main.py path/to/video.mp4
```

Results are saved to `runs/detect/`.

## Fine-tuning

To detect classes not in COCO, you can fine-tune the model on your own labeled dataset. See the [Ultralytics docs](https://docs.ultralytics.com/modes/train/).
