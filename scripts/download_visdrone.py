"""Download and extract the VisDrone2019-DET dataset."""

import urllib.request
import zipfile
from pathlib import Path

SPLITS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val":   "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test":  "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}

DATA_DIR = Path("data/visdrone")


def download_split(name: str, url: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{name}.zip"

    if not zip_path.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()

    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    print(f"{name} done.")


def _progress(count, block_size, total):
    pct = min(count * block_size / total * 100, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


if __name__ == "__main__":
    for name, url in SPLITS.items():
        download_split(name, url)
    print("All splits downloaded.")
