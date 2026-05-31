from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import sys

HF_REPO = "groelisabeth/object-detection-visdrone"
WEIGHTS = Path("visdrone.pt")


def load_model() -> YOLO:
    if not WEIGHTS.exists():
        print(f"Downloading model from {HF_REPO}...")
        path = hf_hub_download(repo_id=HF_REPO, filename="best.pt")
        import shutil
        shutil.copy(path, WEIGHTS)
    return YOLO(str(WEIGHTS))


def detect(source: str = "0") -> None:
    model = load_model()
    model.predict(source=source, show=True, save=True, conf=0.25)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    detect(source)
