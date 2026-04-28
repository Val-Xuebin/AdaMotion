#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adamotion

export PYTHONPATH="/workspace/AdaMotion:/workspace/AdaMotion/lam:/workspace/AdaMotion/worldmodel:${PYTHONPATH:-}"
export WANDB_DIR="${WANDB_DIR:-/workspace/AdaMotion/wandb}"
export ADAMOTION_TEXT_ENCODER_BACKEND="${ADAMOTION_TEXT_ENCODER_BACKEND:-hf}"

RAW_DIR="/workspace/assets/benchmark/raw"
EVAL_DIR="/workspace/AdaMotion/experiments/evals"
mkdir -p "${RAW_DIR}" "${EVAL_DIR}"

MODE_PRESET="${1:-debug}"
if [[ "${MODE_PRESET}" == "debug" ]]; then
  REPLICATIONS=1
  NUM_BATCHES=2
  MM_BATCHES=0
elif [[ "${MODE_PRESET}" == "benchmark" ]]; then
  REPLICATIONS=20
  NUM_BATCHES_ARG=()
  MM_BATCHES=3
else
  echo "usage: $0 [debug|benchmark]" >&2
  exit 2
fi

python /workspace/assets/benchmark/scripts/validate_standard_layout.py \
  2>&1 | tee "${RAW_DIR}/adamotion.eval_validate_standard_layout.log"

run_eval() {
  local mode="$1"
  local output="${EVAL_DIR}/official_salad_${mode}_${MODE_PRESET}.json"
  local log="${RAW_DIR}/adamotion.official_salad_${mode}_${MODE_PRESET}.log"
  local args=(
    --mode "${mode}"
    --replication-times "${REPLICATIONS}"
    --mm-batches "${MM_BATCHES}"
    --output "${output}"
  )
  if [[ "${MODE_PRESET}" == "debug" ]]; then
    args+=(--num-batches "${NUM_BATCHES}")
  fi
  python /workspace/AdaMotion/scripts/eval_official_salad_action_benchmark.py "${args[@]}" \
    2>&1 | tee "${log}"
}

run_eval salad_no_action
run_eval oracle_action
run_eval prior_action

python /workspace/assets/benchmark/scripts/parse_results.py \
  --rebuild-summary \
  --results-dir /workspace/assets/benchmark/results \
  2>&1 | tee "${RAW_DIR}/adamotion.rebuild_summary_${MODE_PRESET}.log"
