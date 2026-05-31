from ultralytics import YOLO
import sys

MODEL = "runs/detect/runs/train/visdrone/weights/best.pt"  # nano — fast, small; swap for yolov8s/m/l/x for more accuracy


def detect(source: str = "0") -> None:
    """Run detection on a source: webcam index, image path, video path, or URL."""
    model = YOLO(MODEL)
    model.predict(source=source, show=True, save=True, conf=0.25)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    detect(source)
