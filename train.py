import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data="training/visdrone.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=device,
    project="runs/train",
    name="visdrone",
    exist_ok=True,
)
