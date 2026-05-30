from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="training/visdrone.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="mps",
    project="runs/train",
    name="visdrone",
    exist_ok=True,
)
