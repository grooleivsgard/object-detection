# object-detection

Detect persons and cars in drone footage in real time using [YOLOv8](https://github.com/ultralytics/ultralytics). The model is fine-tuned on aerial imagery so it handles the top-down drone perspective well.

---

## What you need

- **Python 3.10 or newer** — [download here](https://www.python.org/downloads/)
- **Git** — [download here](https://git-scm.com/downloads)
---

## Initial setup
```bash
# 1. Download this project from terminal
git clone https://github.com/grooleivsgard/object-detection.git
cd object-detection

# 2. Create and activate a virtual environment to keep the project's dependencies isolated from the rest of your system.
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the fine-tuned model
python scripts/download_model.py
```

---

## Run model

On a video file:

```bash
python detect.py path/to/your-video.mp4 # Example: python detect.py videos/test.mp4
```

On your camera source:

```bash
python detect.py
```

Results are also saved to `runs/detect/`.

### For inference on-device:

There seems to be two options for your device:

#### Option 1:
----------
SNPE (Qualcomm's Snapdragon Neural Processing Engine) uses the Hexagon DSP on the APQ8096 and is the right call for real-time on-device. The path is: PyTorch → ONNX → DLC (Qualcomm's format) → run via SNPE runtime. More involved setup.

#### Option 2:
----------
Stream to ground skips the on-device compute problem entirely — send video over Microhard, run YOLO on your laptop. Simpler, but depends on link latency and whether Microhard bandwidth can carry the video.

Given your Microhard is already configured and your ground station has a capable CPU, streaming to ground is the fastest path to something working. SNPE is the right long-term answer if you need autonomous on-device inference.

---

## Project structure

```
object-detection/
├── detect.py               ← run this to detect objects
├── train.py                ← fine-tune the model (advanced)
├── scripts/
│   ├── download_model.py      download model weights from Hugging Face
│   ├── upload_model.py        upload trained weights to Hugging Face
│   ├── download_visdrone.py   download the VisDrone training dataset
│   └── prepare_visdrone.py    convert dataset to YOLO format
├── training/
│   └── visdrone.yaml       dataset config used during training
├── slurm/
│   └── train.sh            job script for running training on a compute cluster
├── data/                   dataset lives here (not tracked by git)
├── videos/                 put your drone footage here
└── requirements.txt        Python dependencies
```

---

## How it works

The model is based on **YOLOv8** (You Only Look Once), a fast object detection architecture. It was originally trained on [COCO](https://cocodataset.org/) — a large dataset of everyday images — and then fine-tuned on [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), a dataset of images and videos captured from drones. This makes it much better at detecting small objects from above compared to the base COCO model.

---

## Fine-tuning on your own data

If you want to re-train the model, see the job script in `scripts/train.py` for the training configuration.
