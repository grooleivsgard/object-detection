#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=share-ie-idi
#SBATCH --constraint="v100|a100"
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20GB
#SBATCH --job-name="visdrone-finetune"
#SBATCH --output=slurm/logs/visdrone_%j.log
#SBATCH --error=slurm/logs/visdrone_%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p slurm/logs

module purge
module load Python/3.11.5-GCCcore-13.2.0

source /cluster/work/geoleivs/object-detection/.venv/bin/activate
cd /cluster/work/geoleivs/object-detection

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

echo "[1/3] Downloading VisDrone dataset..."
python3 -u scripts/download_visdrone.py

echo "[2/3] Preparing annotations..."
python3 -u scripts/prepare_visdrone.py

echo "[3/3] Starting training..."
python3 -u train.py

echo "Best weights: runs/train/visdrone/weights/best.pt"
echo "Finished: $(date)"
