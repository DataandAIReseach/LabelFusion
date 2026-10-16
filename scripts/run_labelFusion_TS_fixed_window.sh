#!/bin/bash
#SBATCH -J predict_all_labelFusion_TS
#SBATCH -p scc-gpu
#SBATCH -G A100:1                 # ganze A100 statt "irgendeine GPU" (Name ggf. anpassen)
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 48:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err
#SBATCH --constraint=inet

# LabelFusion + time series (testing/predict_all_labelFusion_TS_fixed_window.py): the two-stage LabelFusion
# plus an embedding of DGS10 / T10Y2Y / VIXCLS (fixed window of the last 21 daily values) from three
# TimesFM encoders. This script always runs with --ts-pool-only: no CrossTSTransformer attention, the three
# per-series embeddings are only pooled (mean, or max with --ts-pooling max) and passed straight to the fusion MLP.
# Output: outputs/labelfusion_roberta_large_ts_fixed_window/
#
# Before the run the job downloads roberta-large and the TimesFM model (google/timesfm-2.5-200m-transformers)
# from Hugging Face (needs --constraint=inet) and checks that the installed transformers knows TimesFM 2.5.
# If outputs/labelfusion_roberta_large/summary_seeds.csv exists (run of run_labelFusion.sh), the
# roberta-large parameters are taken from there instead of tuning stage 1 again.
#
#   sbatch scripts/run_labelFusion_TS_fixed_window.sh                                       # all 24 pairs (does not fit into 48 h)
#   sbatch scripts/run_labelFusion_TS_fixed_window.sh --datasets mm-test mm-split-test pc-test pc-split-test
#   sbatch scripts/run_labelFusion_TS_fixed_window.sh --datasets pc-test --limit-train 40 --trials 0 --mlp-trials 0 --epochs 1   # smoke test
#   sbatch scripts/run_labelFusion_TS_fixed_window.sh --ts-pooling max                      # max instead of mean over the series
#   sbatch scripts/run_labelFusion_TS_fixed_window.sh --ts-window 0                         # whole series (start up to the date) instead of 21 days
#
# Jobs can run in parallel (each on its own datasets): every result file carries the model and the start time
# of the run in its name (<name>_ts_fixed_window_<YYYYMMDD_HHMMSS>.csv/.json), so runs never overwrite each other.
# Every run's summary_*.csv only holds that run's pairs.

set -euo pipefail

BASE=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion
PYTHON=$BASE/env_labelFusion/bin/python
mkdir -p $BASE/logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# falls nicht schon im Skript/.env gesetzt: export HF_HOME=...

# zeigt im Log, welche GPU du wirklich bekommen hast
srun nvidia-smi -L
srun nvidia-smi --query-gpu=name,memory.total --format=csv

# Modelle vorab laden (die Rechenknoten haben nur mit --constraint=inet Internet)
$PYTHON - <<'PY'
import transformers
from huggingface_hub import snapshot_download

print("transformers", transformers.__version__)
for repo in ("roberta-large", "google/timesfm-2.5-200m-transformers"):
    print("downloading", repo, "->", snapshot_download(repo))
try:
    from transformers import TimesFm2_5Model  # noqa: F401
except ImportError:
    raise SystemExit("transformers has no TimesFm2_5Model: install a newer transformers in env_labelFusion")
print("TimesFM 2.5 available")
PY

REUSE=$BASE/outputs/labelfusion_roberta_large/summary_seeds.csv
REUSE_ARGS=()
[ -f "$REUSE" ] && REUSE_ARGS=(--roberta-params-from "$REUSE")

srun $PYTHON \
     $BASE/testing/predict_all_labelFusion_TS_fixed_window.py \
     --trials 24 --mlp-trials 30 --epochs 10 --retries 2 --ts-pool-only ${REUSE_ARGS[@]+"${REUSE_ARGS[@]}"} "$@"
