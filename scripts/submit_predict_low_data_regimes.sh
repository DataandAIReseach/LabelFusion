#!/bin/bash
# Submits all 240 runs of testing/predict_low_data_regimes.py (24 test files x 10 regimes, roberta-large tuned in every
# regime) as 9 jobs of scripts/run_predict_low_data_regimes.sh with a 48 h limit each. The split follows the size of the
# training files (combine / split-combine are about half of all data, so they are split by regime as well); estimated
# 12-22 h per job on an A100. The jobs run in parallel without influencing each other (own work folder per job).
# A job that hits the time limit: submit the same line again with --resume, finished runs are skipped.
#
#   bash scripts/submit_predict_low_data_regimes.sh                 # all 9 jobs
#   bash scripts/submit_predict_low_data_regimes.sh --resume        # extra arguments go to every job
#
# --datasets matches substrings of the test file name without the seed: "manual-combine-test" is not
# "split-combine-test", "mm-test" is not "mm-split-test".

set -euo pipefail
cd "$(dirname "$0")/.."

LOW="10 20 30 40 50 60 70"
HIGH="80 90 100"
JOBS=(
    "--datasets manual-combine-test --percents $LOW"
    "--datasets manual-combine-test --percents $HIGH"
    "--datasets split-combine-test --percents $LOW"
    "--datasets split-combine-test --percents $HIGH"
    "--datasets mm-test"
    "--datasets mm-split-test"
    "--datasets sp-test"
    "--datasets sp-split-test"
    "--datasets pc-test pc-split-test"
)

for job in "${JOBS[@]}"; do
    # shellcheck disable=SC2086  # $job is split into its arguments on purpose
    sbatch -t 48:00:00 scripts/run_predict_low_data_regimes.sh $job "$@"
done
