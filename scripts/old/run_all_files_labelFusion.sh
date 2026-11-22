#!/bin/bash
#SBATCH -J predict_all_labelFusion
#SBATCH -p scc-gpu
#SBATCH -G A100:1                 # ganze A100 statt "irgendeine GPU" (Name ggf. anpassen)
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 48:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err
#SBATCH --constraint=inet

set -euo pipefail

BASE=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion
mkdir -p $BASE/logs

export TEXTCLASSIFY_ROBERTA_MODEL=roberta-large
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# falls nicht schon im Skript/.env gesetzt:
# export HF_HOME=...; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# zeigt im Log, welche GPU du wirklich bekommen hast
srun nvidia-smi -L
srun nvidia-smi --query-gpu=name,memory.total --format=csv

srun $BASE/env_labelFusion/bin/python \
     $BASE/testing/predict_all_files_labelFusion.py \
     --force --epochs 8 --tune-scope group