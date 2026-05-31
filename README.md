# object-detection

Real-time object detection (persons and cars) from drone footage using [YOLOv8](https://github.com/ultralytics/ultralytics), fine-tuned on the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset.

## Structure

```
object-detection/
├── detect.py               ← run inference
├── scripts/
│   ├── train.py               ← fine-tune model
│   ├── download_visdrone.py   download visdrone dataset (~2.5GB)
│   ├── prepare_visdrone.py    convert to YOLO format
│   ├── upload_model.py        push weights to Hugging Face
│   └── download_model.py      pull weights from Hugging Face
├── training/
│   └── visdrone.yaml       dataset config
├── data/                   dataset (gitignored)
├── videos/                 test footage
├── requirements.txt
└── .gitignore
```

## Model

Base model is `yolov8n.pt` (nano variant) — pre-trained by Ultralytics on [COCO](https://cocodataset.org/), then fine-tuned on VisDrone for aerial perspective detection of persons and cars.

Swap `MODEL` in `detect.py` for a larger variant if you need more accuracy:

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

### Inference

```bash
# Webcam
python detect.py

# Image or video file
python detect.py path/to/video.mp4
```

Results are saved to `runs/detect/`.


# Everyone else
python scripts/download_model.py  # saves best.pt locally
```

