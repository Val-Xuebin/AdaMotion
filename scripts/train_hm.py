#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lam"))
sys.path.insert(0, str(ROOT / "worldmodel"))

from lam.model import train_lam_from_config
from train import train_world_model_from_config


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage = cfg["stage"]
    if stage == "lam":
        result = train_lam_from_config(cfg)
    elif stage == "world_model":
        result = train_world_model_from_config(cfg)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
