#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
from lam.model import train_lam_from_config


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AdaMotion LAM with an AdaWorld-style entrypoint.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = train_lam_from_config(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
