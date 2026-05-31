### Fine-tuning on VisDrone
This folder contains scripts to fine-tune the Yolov8 model on drone footage from the VisDrone dataset. 

```bash
# 1. Download dataset (~2.5GB)
python scripts/download_visdrone.py

# 2. Convert annotations to YOLO format
python scripts/prepare_visdrone.py

# 3. Train (~few hours on Apple M-series)
python scripts/train.py
```

Best weights are saved to `runs/train/visdrone/weights/best.pt`.

## Sharing the model

Weights are not committed to git. Use Hugging Face Hub to share:

```bash
# After training — run once
huggingface-cli login
python scripts/upload_model.py