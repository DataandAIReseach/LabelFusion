#!/bin/bash
#SBATCH -J predict_low_data_regimes_llm_mlp
#SBATCH -p scc-gpu
#SBATCH -G A100:1                 # ganze A100 statt "irgendeine GPU" (Name ggf. anpassen); das kleine MLP braucht kaum etwas davon
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 24:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err

# Check of the MLP training (testing/predict_low_data_regimes_llm_mlp.py): the LLM's label + a downstream fusion MLP, and NOTHING else is
# trained (no roberta-large, no TimesFM, no fine-tuning). For every test file and regime p = 10, 20, ..., 70 % of its training file:
#   llm           the LLM labels of the regime on the full test set (reference, not trained)
#   llm_lookup    LLM label -> most frequent true class of the regime (the best a function of the LLM label alone can do, no training procedure)
#   llm_mlp_all   the project's fusion MLP (Optuna, --mlp-trials, default 30) with the LLM one-hot as its only input, trained on the whole regime
#   llm_mlp_80    the same, trained on the train split (80 %), the data of the end-to-end models
#   llm_mlp_20    the same, trained on the validation split (20 %), the data of the two-stage fusion MLPs
# If llm_mlp_* is clearly worse than llm / llm_lookup, the MLP training is the problem; if it matches them, the weak fusion results come from the
# roberta-large / time series side. No LLM is called: the labels are read from the LLM caches in outputs/llm_cache/.
# Output: outputs/predict_low_data_regimes_llm_mlp/
#
# No model is downloaded (no roberta-large, no TimesFM), so no internet is needed. The job is small: a few GPU-hours at most.
#
#   sbatch scripts/run_predict_low_data_regimes_llm_mlp.sh --dry-run                      # which runs, nothing is trained
#   sbatch scripts/run_predict_low_data_regimes_llm_mlp.sh --datasets pc-split-test --percents 10 70 --mlp-trials 5   # smoke test
#   sbatch scripts/run_predict_low_data_regimes_llm_mlp.sh                                # all 168 runs (24 test files x regimes 10 to 70 %)
#   sbatch scripts/run_predict_low_data_regimes_llm_mlp.sh --percents 80 90 100           # other regimes
#   sbatch scripts/run_predict_low_data_regimes_llm_mlp.sh --no-resume                    # repeat runs that already have a result (default: skipped)
#
# If the job is cut by the time limit, submit the SAME command again: finished runs are skipped, the run in progress continues from its
# checkpoint (outputs/predict_low_data_regimes_llm_mlp/checkpoints/). Every result file carries the start time of the run in its name.

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

export PYTHONUNBUFFERED=1

srun nvidia-smi -L

srun $PYTHON \
     $BASE/testing/predict_low_data_regimes_llm_mlp.py \
     ${PERCENT_ARGS[@]+"${PERCENT_ARGS[@]}"} "$@"
