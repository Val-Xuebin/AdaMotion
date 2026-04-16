#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.experiment import finalize_run


def main() -> None:
    legacy_runs = [
        {
            "stage": "lam",
            "output_dir": ROOT / "experiments" / "humanml_lam_debug",
            "history_file": ROOT / "experiments" / "humanml_lam_debug" / "lam_history.json",
            "checkpoint_path": ROOT / "experiments" / "humanml_lam_debug" / "lam_best.pt",
            "cfg": {
                "config_path": str(ROOT / "configs" / "humanml_lam_debug.yaml"),
                "train": {"output_dir": str(ROOT / "experiments" / "humanml_lam_debug")},
                "data": {"dataset": {"data_root": str(ROOT / "data" / "HumanML3D"), "representation": "humanml_feature_vector"}},
                "model": {"architecture": "mlp_transition_vae", "family": "feature_mlp"},
            },
            "extra_summary": {
                "legacy": True,
                "notes": "Previous baseline before joint-position spatiotemporal transformer refactor.",
                "representation": "humanml_feature_vector",
            },
        },
        {
            "stage": "world_model",
            "output_dir": ROOT / "experiments" / "humanml_world_debug",
            "history_file": ROOT / "experiments" / "humanml_world_debug" / "world_history.json",
            "checkpoint_path": ROOT / "experiments" / "humanml_world_debug" / "world_best.pt",
            "cfg": {
                "config_path": str(ROOT / "configs" / "humanml_world_debug.yaml"),
                "train": {"output_dir": str(ROOT / "experiments" / "humanml_world_debug")},
                "data": {"dataset": {"data_root": str(ROOT / "data" / "HumanML3D"), "representation": "humanml_feature_vector"}},
                "model": {"architecture": "mlp_context_predictor", "family": "feature_mlp"},
            },
            "extra_summary": {
                "legacy": True,
                "notes": "Previous baseline before joint-position spatiotemporal transformer refactor.",
                "representation": "humanml_feature_vector",
            },
        },
    ]

    for run in legacy_runs:
        with open(run["history_file"], "r", encoding="utf-8") as f:
            history = json.load(f)
        best_metrics = min(
            history,
            key=lambda row: row["val_loss"] if run["stage"] == "lam" else row["val_action_loss"],
        )
        finalize_run(
            repo_root=ROOT,
            stage=run["stage"],
            cfg=run["cfg"],
            history=history,
            checkpoint_path=str(run["checkpoint_path"]),
            best_metrics=best_metrics,
            extra_summary=run["extra_summary"],
        )


if __name__ == "__main__":
    main()
