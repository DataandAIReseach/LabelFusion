#!/bin/bash
#SBATCH -J predict_low_data_regimes
#SBATCH -p scc-gpu
#SBATCH -G A100:1                 # ganze A100 statt "irgendeine GPU" (Name ggf. anpassen)
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 72:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err
#SBATCH --constraint=inet

# Low-data regimes (testing/predict_low_data_regimes.py): for every test file and every regime
# p = 10, 20, ..., 100 % of its training file models are trained on the regime and scored on the full test set.
#   regimes 80 and 100 %  the FULL program of predict_all_labelFusion.py: roberta-large is FINE-TUNED (Optuna over lr x batch
#                         size x epochs, class weights, warmup, best epoch, retries), then the fusion MLPs are tuned with Optuna and
#                         trained on the validation split: fusion = fine-tuned roberta-large + LLM label, fusion_ts = additionally the
#                         FINE-TUNED time series model (DGS10 / T10Y2Y / VIXCLS: TimesFM encoders + CrossTSTransformer trained with the
#                         MLP; --ts-mlp-trials, default 10, because every trial fine-tunes three TimesFM encoders)
#   all other regimes     FROZEN encoders: roberta-large (pretrained, never fine-tuned), the TimesFM encoders and the time series
#                         transformer are frozen, only MLPs are trained (and tuned with Optuna, --mlp-trials, default 30) on the
#                         whole regime: roberta (embedding alone), fusion (+ LLM label), fusion_ts (+ pooled time series)
# No LLM is called: the LLM labels are read from the LLM caches in outputs/predict_low_data_regimes/llm_cache/ (zero-shot labels of
# the training sentences, and the labels of the test sentences for every regime).
# Output: outputs/predict_low_data_regimes/
#
# Before the run the job downloads roberta-large and the TimesFM model (google/timesfm-2.5-200m-transformers)
# from Hugging Face (needs --constraint=inet) and checks that the installed transformers knows TimesFM 2.5.
# If outputs/labelfusion_roberta_large/summary_seeds.csv exists (run of run_labelFusion.sh), the roberta-large parameters
# (lr, batch size, epochs) tuned there for the same test file are used at 80 / 100 % instead of tuning stage 1 again (the 24
# Optuna trials of roberta-large per run are by far the most expensive part).
#
#   sbatch scripts/run_predict_low_data_regimes.sh --dry-run                                  # which runs, nothing is trained
#   sbatch scripts/run_predict_low_data_regimes.sh --datasets pc-split-test --percents 10 80 --mlp-trials 3 --trials 0 --epochs 2   # smoke test
#   sbatch scripts/run_predict_low_data_regimes.sh --datasets mm-test mm-split-test pc-test pc-split-test
#   sbatch scripts/run_predict_low_data_regimes.sh --percents 10 20 30 40 50 60 70 90         # only the frozen regimes
#   sbatch scripts/run_predict_low_data_regimes.sh --percents 80 100                          # only the full program
#   sbatch scripts/run_predict_low_data_regimes.sh --full-percents                            # frozen everywhere (no value = none)
#   sbatch scripts/run_predict_low_data_regimes.sh --resume                                   # skip runs that already have a result
#   sbatch scripts/run_predict_low_data_regimes.sh --no-ts                                    # without the time series model
#   sbatch scripts/run_predict_low_data_regimes.sh --mlp-trials 10 --trials 16 --ts-mlp-trials 5   # fewer Optuna trials per run
#
# 240 runs (48 of them with the full program) do not fit into one job: split them by --datasets and/or --percents and use
# --resume to go on. Jobs can run in parallel (each on its own datasets): every result file carries the start time of the
# run in its name (<test file>_<p>_low_data_regimes_<YYYYMMDD_HHMMSS>.csv/.json), so runs never overwrite each other.
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
     ${REUSE_ARGS[@]+"${REUSE_ARGS[@]}"} "$@"
