"""Convert VisDrone annotations to YOLO format, keeping only persons and cars."""

from pathlib import Path
from PIL import Image

# VisDrone category IDs we care about → remapped YOLO class index
# 1=pedestrian, 2=people, 4=car
KEEP = {1: 0, 2: 0, 4: 1}  # person=0, car=1

SPLITS = {
    "train": "data/visdrone/VisDrone2019-DET-train",
    "val":   "data/visdrone/VisDrone2019-DET-val",
    "test":  "data/visdrone/VisDrone2019-DET-test-dev",
}


def convert_split(name: str, src: str) -> None:
    src = Path(src)
    img_dir = src / "images"
    ann_dir = src / "annotations"

    out_img = Path(f"data/images/{name}")
    out_lbl = Path(f"data/labels/{name}")
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for ann_file in sorted(ann_dir.glob("*.txt")):
        img_file = img_dir / ann_file.with_suffix(".jpg").name
        if not img_file.exists():
            continue

        w, h = Image.open(img_file).size
        lines = []

        for row in ann_file.read_text().splitlines():
            parts = row.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, bw, bh, _, cat = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), parts[4], int(parts[5])
            if cat not in KEEP or bw <= 0 or bh <= 0:
                continue

            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(f"{KEEP[cat]} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}")

        if lines:
            (out_lbl / ann_file.name).write_text("\n".join(lines))
            import shutil
            shutil.copy(img_file, out_img / img_file.name)

    print(f"{name}: {len(list(out_lbl.glob('*.txt')))} images with annotations")


if __name__ == "__main__":
    for name, src in SPLITS.items():
        if Path(src).exists():
            convert_split(name, src)
        else:
            print(f"Skipping {name} — not found at {src}")
