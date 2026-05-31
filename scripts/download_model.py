"""Download trained model weights from Hugging Face Hub."""

from huggingface_hub import hf_hub_download
from pathlib import Path

HF_REPO = "groelisabeth/object-detection-visdrone"
OUT_PATH = Path("visdrone.pt")


def download() -> Path:
    path = hf_hub_download(repo_id=HF_REPO, filename="best.pt")
    import shutil
    shutil.copy(path, OUT_PATH)
    print(f"Model saved to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    download()
