#!/bin/bash
#SBATCH -J predict_low_data_regimes_logit
#SBATCH -p scc-gpu
#SBATCH -G A100:1                 # ganze A100 statt "irgendeine GPU" (Name ggf. anpassen)
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 48:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err
#SBATCH --constraint=inet

# Fusion on the LOGITS of roberta-large (testing/predict_low_data_regimes_logit.py) instead of its embedding. For every test file and
# every regime p = 10, 20, ..., 100 % roberta-large is fine-tuned ALONE on the train split (80 % of the regime; Optuna over lr x batch size x
# epochs, class weights, warmup, best epoch, retries; the column "roberta" of predict_low_data_regimes.py). A logistic regression
# fitted on the validation split (20 %) then fuses roberta-large's 3 log-probabilities with the LLM label (one-hot):
#   llm, roberta, fusion_logit (logits + LLM label), logit_only (logits alone: recalibration without the LLM label).
# No LLM is called: the LLM labels are read from outputs/llm_cache/. No MLP, no time series, no end-to-end training.
# Output: outputs/predict_low_data_regimes_logit/
#
# Before the run the job downloads roberta-large from Hugging Face (needs --constraint=inet).
# All 240 runs (24 test files x 10 regimes) take several GPU days (24 Optuna trials per run): split them by --datasets and/or --percents
# into several jobs. If a job is cut by the time limit, submit the SAME command again: finished runs are skipped, and the run in
# progress continues from its checkpoint (outputs/predict_low_data_regimes_logit/checkpoints/): only its final fine-tuning is repeated.
# Jobs can run in parallel (each on its own datasets): every result file carries the start time and the job id in its name.
#
#   sbatch scripts/run_predict_low_data_regimes_logit.sh --dry-run                                   # which runs, nothing is trained
#   sbatch scripts/run_predict_low_data_regimes_logit.sh --datasets pc-split-test --percents 10 70 --trials 0 --epochs 2   # smoke test
#   sbatch scripts/run_predict_low_data_regimes_logit.sh --datasets mm-test mm-split-test pc-test pc-split-test
#   sbatch scripts/run_predict_low_data_regimes_logit.sh --percents 10 40 70                         # only these regimes (default: 10 to 100 %)
#   sbatch scripts/run_predict_low_data_regimes_logit.sh --trials 16                                 # fewer Optuna trials per run (default 24)
#   sbatch scripts/run_predict_low_data_regimes_logit.sh --no-resume                                 # repeat runs that already have a result

set -euo pipefail

BASE=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion
PYTHON=$BASE/env_labelFusion/bin/python
mkdir -p $BASE/logs

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

# Modell vorab laden (die Rechenknoten haben nur mit --constraint=inet Internet)
$PYTHON - <<'PY'
import transformers
from huggingface_hub import snapshot_download

print("transformers", transformers.__version__)
print("downloading roberta-large ->", snapshot_download("roberta-large"))
PY

srun $PYTHON \
     $BASE/testing/predict_low_data_regimes_logit.py "$@"
