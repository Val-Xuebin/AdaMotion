#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-debug}"
DEVICE="${2:-cuda}"
RUN_NAME="${3:-}"

PYTHON="/root/.conda/envs/adamotion/bin/python"
ROOT="/workspace"
ADAMOTION_ROOT="$ROOT/AdaMotion"
BENCH_ROOT="$ROOT/assets/benchmark"
RAW_DIR="$BENCH_ROOT/raw"
RUNS_DIR="$ADAMOTION_ROOT/experiments"
mkdir -p "$RAW_DIR" "$RUNS_DIR"

case "$MODE" in
  debug)
    BASE_CONFIG="$ADAMOTION_ROOT/configs/humanml_gt_feature_lam_debug.yaml"
    ;;
  full)
    BASE_CONFIG="$ADAMOTION_ROOT/configs/humanml_gt_feature_lam_full.yaml"
    ;;
  *)
    echo "usage: run_adamotion_gt_feature_lam.sh [debug|full] [cpu|cuda] [run_name]" >&2
    exit 2
    ;;
esac

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [ -z "$RUN_NAME" ]; then
  RUN_NAME="humanml_gt_feature_lam_${MODE}"
fi

TMP_CONFIG="/tmp/${RUN_NAME}.${RUN_ID}.yaml"
LOG_PATH="$RAW_DIR/${RUN_NAME}.${RUN_ID}.log"
RESULTS_DIR="$BENCH_ROOT/results"

"$PYTHON" - <<PY
import yaml
from pathlib import Path

base = Path("$BASE_CONFIG")
cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
cfg["config_path"] = str(base)
cfg["train"]["device"] = "$DEVICE"
cfg["train"]["output_dir"] = "/workspace/AdaMotion/experiments/$RUN_NAME"
Path("$TMP_CONFIG").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("$TMP_CONFIG")
PY

echo "[run] config=$TMP_CONFIG"
echo "[run] log=$LOG_PATH"
echo "[run] output_dir=/workspace/AdaMotion/experiments/$RUN_NAME"

"$PYTHON" -u "$ADAMOTION_ROOT/scripts/train_hm.py" --config "$TMP_CONFIG" 2>&1 | tee "$LOG_PATH"

"$PYTHON" "$ADAMOTION_ROOT/scripts/export_gt_feature_lam_result.py" \
  --summary "/workspace/AdaMotion/experiments/$RUN_NAME/summary.json" \
  --source-log "$LOG_PATH" \
  --results-dir "$RESULTS_DIR"

echo "[done] raw_log=$LOG_PATH"
echo "[done] summary=/workspace/AdaMotion/experiments/$RUN_NAME/summary.json"
