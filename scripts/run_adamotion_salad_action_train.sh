#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adamotion

export PYTHONPATH="/workspace/AdaMotion:/workspace/AdaMotion/lam:/workspace/AdaMotion/worldmodel:${PYTHONPATH:-}"
export WANDB_DIR="${WANDB_DIR:-/workspace/AdaMotion/wandb}"
export ADAMOTION_TEXT_ENCODER_BACKEND="${ADAMOTION_TEXT_ENCODER_BACKEND:-hf}"

RAW_DIR="/workspace/experiments/benchmark/raw"
mkdir -p "${RAW_DIR}"

python /workspace/experiments/benchmark/scripts/validate_standard_layout.py \
  2>&1 | tee "${RAW_DIR}/adamotion.validate_standard_layout.log"

if [[ ! -f /workspace/AdaMotion/experiments/humanml_sal_rep_lam_momentum_full/lam_best.pt ]]; then
  python /workspace/AdaMotion/scripts/train_hm.py \
    --config /workspace/AdaMotion/configs/humanml_sal_rep_lam_momentum_full.yaml \
    2>&1 | tee "${RAW_DIR}/adamotion.sal_rep_lam_momentum_full.log"
fi

python /workspace/AdaMotion/scripts/train_hm.py \
  --config /workspace/AdaMotion/configs/humanml_salad_official_action_adapter_momentum_full.yaml \
  2>&1 | tee "${RAW_DIR}/adamotion.official_salad_action_adapter_momentum_full.log"

python /workspace/AdaMotion/scripts/train_hm.py \
  --config /workspace/AdaMotion/configs/humanml_salad_official_action_prior_momentum_full.yaml \
  2>&1 | tee "${RAW_DIR}/adamotion.official_salad_action_prior_momentum_full.log"
