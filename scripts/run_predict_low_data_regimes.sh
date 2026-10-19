#!/bin/bash
#SBATCH -J predict_low_data_regimes
#SBATCH -p scc-gpu
#SBATCH -G A100:1                 # ganze A100 statt "irgendeine GPU" (Name ggf. anpassen)
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 48:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err
#SBATCH --constraint=inet

# Low-data regimes (testing/predict_low_data_regimes.py): for every test file and every regime p = 10, 20, ..., 100 %
# of its training file roberta-large alone, LabelFusion (roberta-large + LLM label) and LabelFusion with the time series
# (DGS10 / T10Y2Y / VIXCLS, last 21 days, TimesFM encoders pooled over the series) are trained on the regime and
# scored on the full test set. The LLM labels of the test sentences are the regime's (outputs/predictions_low_data_regimes/...).
# Output: outputs/predict_low_data_regimes/
#
# Before the run the job downloads roberta-large and the TimesFM model (google/timesfm-2.5-200m-transformers)
# from Hugging Face (needs --constraint=inet) and checks that the installed transformers knows TimesFM 2.5.
# No Optuna by default (240 runs). If outputs/labelfusion_roberta_large/summary_seeds.csv exists (run of run_labelFusion.sh),
# the roberta-large parameters (lr, batch size, epochs) tuned there for the same test file are used for all its regimes.
#
#   sbatch scripts/run_predict_low_data_regimes.sh --dry-run                                  # which runs, nothing is trained
#   sbatch scripts/run_predict_low_data_regimes.sh --datasets pc-split-test --percents 10 100 --epochs 2   # smoke test
#   sbatch scripts/run_predict_low_data_regimes.sh --datasets mm-test mm-split-test pc-test pc-split-test
#   sbatch scripts/run_predict_low_data_regimes.sh --resume                                   # skip runs that already have a result
#   sbatch scripts/run_predict_low_data_regimes.sh --no-ts                                    # without the time series model
#   sbatch scripts/run_predict_low_data_regimes.sh --mlp-trials 10                            # tune the fusion MLPs per run
#
# 240 runs probably do not fit into one 48 h job: split them by --datasets and/or --percents and use --resume to go on.
# Jobs can run in parallel (each on its own datasets): every result file carries the start time of the run in its name
# (<test file>_<p>_low_data_regimes_<YYYYMMDD_HHMMSS>.csv/.json), so runs never overwrite each other.
# Every run's summary_*.csv only holds that run's runs.

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
     $BASE/testing/predict_low_data_regimes.py \
     --epochs 10 --retries 2 ${REUSE_ARGS[@]+"${REUSE_ARGS[@]}"} "$@"
