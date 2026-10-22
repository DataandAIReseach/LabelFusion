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
# p = 10, 20, ..., 70 % of its training file models are trained on the regime and scored on the full test set. Only these
# regimes (PERCENTS below) are run by default; --percents ... on the command line replaces them, e.g. --percents 80 90 100.
#   every regime (default) the FULL program of predict_all_labelFusion.py: roberta-large is FINE-TUNED (Optuna over lr x batch
#                         size x epochs, class weights, warmup, best epoch, retries), then the fusion MLPs are tuned with Optuna and
#                         trained on the validation split: fusion = fine-tuned roberta-large + LLM label, fusion_ts = additionally the
#                         FINE-TUNED time series model (DGS10 / T10Y2Y / VIXCLS: TimesFM encoders + CrossTSTransformer trained with the
#                         MLP; --ts-mlp-trials, default 10, because every trial fine-tunes three TimesFM encoders)
#   --full-percents P...  only these get the full program, the others FROZEN encoders: roberta-large (pretrained, never fine-tuned), the TimesFM encoders and the time series
#                         transformer are frozen, only MLPs are trained (and tuned with Optuna, --mlp-trials, default 30) on the
#                         whole regime: roberta (embedding alone), fusion (+ LLM label), fusion_ts (+ pooled time series)
# No LLM is called: the LLM labels are read from the LLM caches in outputs/llm_cache/ (zero-shot labels of
# the training sentences, and the labels of the test sentences for every regime).
# Output: outputs/predict_low_data_regimes/
#
# Before the run the job downloads roberta-large and the TimesFM model (google/timesfm-2.5-200m-transformers)
# from Hugging Face (needs --constraint=inet) and checks that the installed transformers knows TimesFM 2.5.
# roberta-large is tuned in every regime (24 Optuna trials per run, by far the most expensive part). To reuse the parameters
# of a run_labelFusion.sh run instead (tuned on 100 % of the training file), pass
# --roberta-params-from $BASE/outputs/labelfusion_roberta_large/summary_seeds.csv explicitly.
# All 168 runs (24 test files x the regimes 10 to 70 %) take several GPU days: scripts/submit_predict_low_data_regimes.sh submits them as 7 jobs.
#
#   sbatch scripts/run_predict_low_data_regimes.sh --dry-run                                  # which runs, nothing is trained
#   sbatch scripts/run_predict_low_data_regimes.sh --datasets pc-split-test --percents 10 80 --mlp-trials 3 --trials 0 --epochs 2   # smoke test
#   sbatch scripts/run_predict_low_data_regimes.sh --datasets mm-test mm-split-test pc-test pc-split-test
#   sbatch scripts/run_predict_low_data_regimes.sh --percents 10 40                           # only these regimes (instead of 10 to 70 %)
#   sbatch scripts/run_predict_low_data_regimes.sh --full-percents 80 100                     # full program only at 80 / 100 %, frozen elsewhere
#   sbatch scripts/run_predict_low_data_regimes.sh --full-percents                            # frozen everywhere (no value = none)
#   sbatch scripts/run_predict_low_data_regimes.sh --no-resume                                # repeat runs that already have a result (default: skipped)
#   sbatch scripts/run_predict_low_data_regimes.sh --no-ts                                    # without the time series model
#   sbatch scripts/run_predict_low_data_regimes.sh --mlp-trials 10 --trials 16 --ts-mlp-trials 5   # fewer Optuna trials per run
#
# 168 runs (all with the full program) do not fit into one job: split them by --datasets and/or --percents. If a job is cut by the
# time limit, just submit the SAME command again: finished runs are skipped, and the run that was in progress continues from its
# checkpoint (outputs/predict_low_data_regimes/checkpoints/): every Optuna trial and MLP fit that was done is read from disk, only the
# final fine-tuning of roberta-large of that run is repeated. Jobs can run in parallel (each on its own datasets): every result file carries the start time of the
# run in its name (<test file>_<p>_low_data_regimes_<YYYYMMDD_HHMMSS>.csv/.json), so runs never overwrite each other.
# Every run's summary_*.csv only holds that run's runs.

set -euo pipefail

BASE=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion
PYTHON=$BASE/env_labelFusion/bin/python
mkdir -p $BASE/logs

# the regimes that are run unless the command line brings its own --percents
PERCENTS=(10 20 30 40 50 60 70)
PERCENT_ARGS=(--percents "${PERCENTS[@]}")
for arg in "$@"; do
    [ "$arg" = "--percents" ] && PERCENT_ARGS=()
done

# the LLM caches are not in git (.gitignore: /outputs/): copy outputs/llm_cache/ to the cluster first
LLM_CACHE=$BASE/outputs/llm_cache
if ! ls "$LLM_CACHE"/predict_all_train_gpt-5-nano_zero_shot/cache_*.pkl > /dev/null 2>&1; then
    echo "ERROR: no LLM cache in $LLM_CACHE (expected predict_all_train_gpt-5-nano_zero_shot/ and predict_all_test_gpt-5-nano_*)." >&2
    echo "Copy it from your computer, e.g. rsync -av outputs/llm_cache/ <user>@<login-node>:$LLM_CACHE/" >&2
    exit 1
fi

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

srun $PYTHON \
     $BASE/testing/predict_low_data_regimes.py \
     ${PERCENT_ARGS[@]+"${PERCENT_ARGS[@]}"} "$@"
