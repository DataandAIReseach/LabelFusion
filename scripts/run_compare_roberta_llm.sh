#!/bin/bash
#SBATCH -J compare_roberta_llm
#SBATCH -p scc-gpu
#SBATCH -G V100:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 2-00:00:00
#SBATCH --constraint=inet
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/scripts/.logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/scripts/.logs/%x-%j.err

# roberta-large alone (tuned like predict_all_files_roberta_large.py) vs. the LLM predictions on all
# 24 test sets. Writes ./outputs/roberta_vs_llm/predictions/<test file>.csv (text, ground_truth,
# llm_prediction, roberta_prediction) plus comparison.csv / comparison.xlsx / summary.csv.
#
#   sbatch scripts/run_compare_roberta_llm.sh                         # all datasets
#   sbatch scripts/run_compare_roberta_llm.sh --datasets pc-test      # extra arguments are passed on
#   sbatch scripts/run_compare_roberta_llm.sh --retrain               # train roberta-large again
#
# The #SBATCH log directory must exist before submitting: mkdir -p scripts/.logs

set -euo pipefail

ROOT=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion
REPO="${REPO:-$ROOT/LabelFusion}"
PYTHON="${PYTHON:-$ROOT/env_labelFusion/bin/python}"

cd "$REPO"
mkdir -p "$ROOT/scripts/.logs"
export PYTHONUNBUFFERED=1

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "python: $PYTHON  repo: $REPO  args: $*"

srun "$PYTHON" testing/compare_roberta_llm.py --trials 24 --epochs 10 --retries 2 "$@"

echo "end: $(date)"
