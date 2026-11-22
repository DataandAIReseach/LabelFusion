#!/bin/bash
#SBATCH -J predict_all_full_model
#SBATCH -p scc-gpu
#SBATCH -G A100:1                 # the GPU is needed to fine-tune roberta-large
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 04:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err
#SBATCH --constraint=inet

# Full synthetic dataset (testing/predict_all_full_model.py): fixed stratified test set and
# 10, 20, ..., 100 percent training regimes over the articles JSON in data/ with
#   llm          the zero-shot GPT-5 nano label (read from the LLM cache)
#   logreg       TF-IDF + logistic regression
#   xgboost      TF-IDF + XGBoost with the fixed baseline configuration
#   roberta_mlp  MLP trained jointly with the roberta-large encoder
#   fusion_mlp   MLP trained jointly with roberta-large and fused with the LLM label (LabelFusion)
# Output: outputs/predict_all_full_model/ (<name>_<run time YYYYMMDD_HHMMSS>_finetuned_regime_predictions.csv, ..._regime_scores.csv)
# Pass --no-finetune-roberta to use frozen, cached embeddings instead.
#
# No LLM is called by default (--no-make-llm-labels): the labels are read from
#   outputs/llm_cache/predict_all_train_gpt-5-nano_zero_shot_<JSON name>/
# The LLM caches are not in git (.gitignore: /outputs/): copy that folder to the cluster first. --make-llm-labels creates
# missing labels via OpenRouter instead (paid; needs OPENROUTER_API_KEY in $BASE/.env).
# The embeddings of the 5000 texts take a few minutes on an A100, the ten regimes of MLPs and text models a few more;
# a second run with the same texts reuses the cached embeddings.
#
#   sbatch scripts/run_predict_all_full_model.sh                                             # the newest articles JSON in data/
#   sbatch scripts/run_predict_all_full_model.sh --json data/<articles>.json                 # a specific JSON
#   sbatch scripts/run_predict_all_full_model.sh --limit 300 --roberta-model roberta-base    # smoke test
#   sbatch scripts/run_predict_all_full_model.sh --drop-meta-text                            # without texts with generator notes
#   sbatch scripts/run_predict_all_full_model.sh --proj-dim 64 --hidden 128 --patience 25    # other MLP settings
#   sbatch scripts/run_predict_all_full_model.sh --make-llm-labels                           # create missing LLM labels (paid API)

set -euo pipefail

BASE=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion
PYTHON=$BASE/env_labelFusion/bin/python
mkdir -p $BASE/logs

# the JSON and dataset name the Python script will use (same defaults: newest JSON in data/, name = its file name)
JSON="" NAME="" LLM_MODEL=gpt-5-nano MAKE_ARGS=(--no-make-llm-labels) FINETUNE_ARGS=(--finetune-roberta)
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
        --json) JSON=${ARGS[$((i + 1))]} ;;
        --name) NAME=${ARGS[$((i + 1))]} ;;
        --llm-model) LLM_MODEL=${ARGS[$((i + 1))]} ;;
        --make-llm-labels|--no-make-llm-labels) MAKE_ARGS=() ;;
        --finetune-roberta|--no-finetune-roberta) FINETUNE_ARGS=() ;;
    esac
done
if [ -z "$JSON" ]; then
    JSON=$(ls -t $BASE/data/*.json 2> /dev/null | grep -v '_failures\.json$' | head -1 || true)
    [ -n "$JSON" ] || { echo "ERROR: no articles JSON in $BASE/data/" >&2; exit 1; }
    ARGS=(--json "$JSON" "${ARGS[@]+"${ARGS[@]}"}")
fi
NAME=${NAME:-$(basename "$JSON" .json)}
echo "articles JSON: $JSON (name $NAME)"

# the LLM cache is not in git: without it (and without --make-llm-labels) the run would stop right away
LLM_CACHE=$BASE/outputs/llm_cache/predict_all_train_${LLM_MODEL}_zero_shot_$NAME
if [ ${#MAKE_ARGS[@]} -gt 0 ] && ! ls "$LLM_CACHE"/cache_*.pkl > /dev/null 2>&1; then
    echo "ERROR: no LLM cache in $LLM_CACHE." >&2
    echo "Copy it from your computer, e.g. rsync -av outputs/llm_cache/$(basename "$LLM_CACHE")/ <user>@<login-node>:$LLM_CACHE/" >&2
    echo "or pass --make-llm-labels to create the labels via OpenRouter (paid)." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# zeigt im Log, welche GPU du wirklich bekommen hast
srun nvidia-smi -L

# Modell vorab laden (die Rechenknoten haben nur mit --constraint=inet Internet)
ROBERTA=roberta-large
for i in "${!ARGS[@]}"; do
    [ "${ARGS[$i]}" = "--roberta-model" ] && ROBERTA=${ARGS[$((i + 1))]}
done
$PYTHON -c "from huggingface_hub import snapshot_download; print('downloading', '$ROBERTA', '->', snapshot_download('$ROBERTA'))"

srun $PYTHON \
     $BASE/testing/predict_all_full_model.py \
     ${MAKE_ARGS[@]+"${MAKE_ARGS[@]}"} ${FINETUNE_ARGS[@]+"${FINETUNE_ARGS[@]}"} "${ARGS[@]}"
