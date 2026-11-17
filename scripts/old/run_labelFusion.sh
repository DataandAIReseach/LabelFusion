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

# Two-stage LabelFusion (testing/old/predict_all_labelFusion.py): stage 1 roberta-large alone (4x4 grid +
# Optuna, class weights, warmup, best epoch, retries), stage 2 frozen embeddings + LLM labels -> MLP.
# Output: outputs/labelfusion_roberta_large/ (summary.csv = mean +- std per dataset).
#
#   sbatch scripts/run_labelFusion.sh                                   # all 24 pairs (does not fit into 48 h)
#   sbatch scripts/run_labelFusion.sh --datasets mm-test mm-split-test pc-test pc-split-test --trials 16
#   sbatch scripts/run_labelFusion.sh --datasets sp-test sp-split-test combine-test split-combine-test
#
# summary.csv / summary_seeds.csv are rewritten by every run and only hold that run's pairs: copy
# them away between jobs (the per-pair csv/json and <dataset>_summary.json files stay).

set -euo pipefail

BASE=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion
mkdir -p $BASE/logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# falls nicht schon im Skript/.env gesetzt:
# export HF_HOME=...; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# zeigt im Log, welche GPU du wirklich bekommen hast
srun nvidia-smi -L
srun nvidia-smi --query-gpu=name,memory.total --format=csv

srun $BASE/env_labelFusion/bin/python \
     $BASE/testing/old/predict_all_labelFusion.py \
     --trials 24 --mlp-trials 30 --epochs 10 --retries 2 "$@"
