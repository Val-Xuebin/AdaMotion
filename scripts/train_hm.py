#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from trainer import train_lam, train_world_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage = cfg["stage"]
    if stage == "lam":
        result = train_lam(cfg)
    elif stage == "world_model":
        result = train_world_model(cfg)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
