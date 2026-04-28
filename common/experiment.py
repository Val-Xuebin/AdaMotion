from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_history_csv(path: Path, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    fieldnames: List[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _safe_git_output(repo_root: Path, args: List[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _plot_history(path: Path, history: List[Dict[str, Any]], stage: str) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [row["epoch"] for row in history]
    metric_groups = [
        [
            key
            for key in [
                "train_loss",
                "val_loss",
                "train_mse",
                "val_mse",
                "train_kl",
                "val_kl",
                "train_recon",
                "val_recon",
                "train_latent_mse",
                "val_latent_mse",
            ]
            if key in history[0]
        ],
        [key for key in ["train_action_loss", "val_action_loss", "train_no_action_loss", "val_no_action_loss", "val_gain"] if key in history[0]],
    ]
    series = [group for group in metric_groups if group]
    if not series:
        return

    fig, axes = plt.subplots(len(series), 1, figsize=(8, 3.5 * len(series)), squeeze=False)
    axes = axes[:, 0]
    for ax, group in zip(axes, series):
        for key in group:
            ax.plot(epochs, [row[key] for row in history], marker="o", label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"AdaMotion {stage} metrics")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_history_svg(path: Path, history: List[Dict[str, Any]], stage: str) -> None:
    if not history:
        return
    candidate_keys = [
        key
        for key in [
            "train_loss",
            "val_loss",
            "train_mse",
            "val_mse",
            "train_kl",
            "val_kl",
            "train_action_loss",
            "val_action_loss",
            "train_no_action_loss",
            "val_no_action_loss",
            "val_gain",
        ]
        if key in history[0]
    ]
    if not candidate_keys:
        return

    epochs = [float(row["epoch"]) for row in history]
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    x_span = max(1.0, max_epoch - min_epoch)

    values = [float(row[key]) for row in history for key in candidate_keys]
    min_val = min(values)
    max_val = max(values)
    y_span = max(1e-8, max_val - min_val)

    width = 900
    height = 520
    left = 70
    right = 30
    top = 50
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2"]

    def x_of(epoch: float) -> float:
        return left + ((epoch - min_epoch) / x_span) * plot_w

    def y_of(value: float) -> float:
        return top + (1.0 - ((value - min_val) / y_span)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-size="20" font-family="monospace">AdaMotion {stage} metrics</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
    ]

    for tick in range(5):
        frac = tick / 4
        y = top + (1 - frac) * plot_h
        val = min_val + frac * y_span
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" font-size="12" font-family="monospace">{val:.4f}</text>')

    for idx, epoch in enumerate(epochs):
        x = x_of(epoch)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f1f1f1"/>')
        parts.append(f'<text x="{x - 6:.1f}" y="{top + plot_h + 20}" font-size="12" font-family="monospace">{int(epoch)}</text>')

    for idx, key in enumerate(candidate_keys):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{x_of(float(row['epoch'])):.1f},{y_of(float(row[key])):.1f}" for row in history)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        legend_x = left + (idx % 2) * 300
        legend_y = height - 20 - (idx // 2) * 18
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{legend_y + 4}" font-size="12" font-family="monospace">{key}</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _infer_representation(cfg: Dict[str, Any]) -> str:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if dataset_cfg.get("representation"):
        return str(dataset_cfg["representation"])
    data_root = str(dataset_cfg.get("data_root", ""))
    if "new_joints" in data_root:
        return "joint_positions"
    if "HumanML3D" in data_root:
        joint_dim = cfg.get("model", {}).get("joint_dim")
        if joint_dim == 3:
            return "joint_positions"
        if joint_dim == 13:
            return "sal_rep"
        return "humanml_feature_vector"
    return "unknown"


def _summarize_history(stage: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {"stage": stage}
    if stage == "lam":
        best = min(history, key=lambda row: row["val_loss"])
        return {
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_mse": best.get("val_mse"),
            "best_val_kl": best.get("val_kl"),
            "best_train_loss": best.get("train_loss"),
        }
    if stage == "world_model" and "val_action_loss" in history[0]:
        best = min(history, key=lambda row: row["val_action_loss"])
        return {
            "best_epoch": best["epoch"],
            "best_val_action_loss": best["val_action_loss"],
            "best_val_no_action_loss": best.get("val_no_action_loss"),
            "best_val_gain": best.get("val_gain"),
            "best_train_action_loss": best.get("train_action_loss"),
            "best_train_no_action_loss": best.get("train_no_action_loss"),
        }
    if stage == "world_model":
        best = min(history, key=lambda row: row["val_loss"])
        return {
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_recon": best.get("val_recon"),
            "best_val_latent_mse": best.get("val_latent_mse"),
            "best_train_loss": best.get("train_loss"),
        }
    return {"best_epoch": history[-1]["epoch"]}


def finalize_run(
    repo_root: str | Path,
    stage: str,
    cfg: Dict[str, Any],
    history: List[Dict[str, Any]],
    checkpoint_path: str,
    best_metrics: Dict[str, Any],
    extra_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    output_dir = Path(cfg["train"]["output_dir"])
    summary = {
        "schema_version": 1,
        "run_name": output_dir.name,
        "stage": stage,
        "family": cfg.get("model", {}).get("family"),
        "representation": _infer_representation(cfg),
        "output_dir": str(output_dir),
        "config_path": cfg.get("config_path"),
        "checkpoint_path": checkpoint_path,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _safe_git_output(repo_root, ["rev-parse", "HEAD"]),
        "git_branch": _safe_git_output(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "train": cfg.get("train", {}),
        "data": cfg.get("data", {}),
        "model": cfg.get("model", {}),
        "history_len": len(history),
        "best": best_metrics,
        "summary": _summarize_history(stage, history),
    }
    if extra_summary:
        summary.update(extra_summary)

    _write_json(output_dir / "summary.json", summary)
    _write_history_csv(output_dir / "history.csv", history)
    _plot_history_svg(output_dir / "curves.svg", history, stage)
    _plot_history(output_dir / "curves.png", history, stage)
    refresh_experiment_tables(repo_root)
    return summary


def refresh_experiment_tables(repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    summary_paths = sorted((repo_root / "experiments").glob("**/summary.json"))
    rows: List[Dict[str, Any]] = []
    for path in summary_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row = {
            "run_name": payload.get("run_name"),
            "stage": payload.get("stage"),
            "family": payload.get("family"),
            "representation": payload.get("representation"),
            "config_path": payload.get("config_path"),
            "output_dir": payload.get("output_dir"),
            "checkpoint_path": payload.get("checkpoint_path"),
            "git_commit": payload.get("git_commit"),
            "legacy": payload.get("legacy", False),
            "notes": payload.get("notes", ""),
            "render_root": payload.get("render_root"),
            "best_epoch": payload.get("summary", {}).get("best_epoch"),
            "best_val_loss": payload.get("summary", {}).get("best_val_loss"),
            "best_val_mse": payload.get("summary", {}).get("best_val_mse"),
            "best_val_kl": payload.get("summary", {}).get("best_val_kl"),
            "best_val_action_loss": payload.get("summary", {}).get("best_val_action_loss"),
            "best_val_no_action_loss": payload.get("summary", {}).get("best_val_no_action_loss"),
            "best_val_gain": payload.get("summary", {}).get("best_val_gain"),
        }
        rows.append(row)

    tables_dir = repo_root / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "adamotion_runs.csv"
    md_path = tables_dir / "adamotion_runs.md"

    fieldnames = [
        "run_name",
        "stage",
        "family",
        "representation",
        "best_epoch",
        "best_val_loss",
        "best_val_mse",
        "best_val_kl",
        "best_val_action_loss",
        "best_val_no_action_loss",
        "best_val_gain",
        "legacy",
        "notes",
        "render_root",
        "config_path",
        "output_dir",
        "checkpoint_path",
        "git_commit",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# AdaMotion Runs",
        "",
        "| run_name | stage | family | representation | best_epoch | best_val_loss | best_val_action_loss | best_val_no_action_loss | best_val_gain | legacy | notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {stage} | {family} | {representation} | {best_epoch} | {best_val_loss} | {best_val_action_loss} | {best_val_no_action_loss} | {best_val_gain} | {legacy} | {notes} |".format(
                **{key: ("" if row.get(key) is None else row.get(key)) for key in row}
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
