#!/usr/bin/env bash
set -euo pipefail

python3 /work/adamotion/scripts/inspect_dataset.py
python3 /work/adamotion/scripts/train_hm.py --config /work/adamotion/configs/humanml_lam_debug.yaml
python3 /work/adamotion/scripts/train_hm.py --config /work/adamotion/configs/humanml_world_debug.yaml
