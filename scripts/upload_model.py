"""Upload trained model weights to Hugging Face Hub."""

from huggingface_hub import HfApi
from pathlib import Path

HF_REPO = "groelisabeth/object-detection-visdrone"
WEIGHTS = Path("runs/detect/runs/train/visdrone/weights/best.pt")


def upload() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"No weights found at {WEIGHTS} — train first.")

    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(WEIGHTS),
        path_in_repo="best.pt",
        repo_id=HF_REPO,
        repo_type="model",
    )
    print(f"Uploaded to https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    upload()
