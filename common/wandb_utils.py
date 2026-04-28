from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _to_jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _wandb_root(repo_root: str | Path) -> Path:
    repo_root = Path(repo_root)
    return repo_root / "wandb"


def init_wandb_run(repo_root: str | Path, stage: str, cfg: Dict[str, Any]):
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    try:
        import wandb
    except Exception:
        return None

    repo_root = Path(repo_root)
    wandb_root = _wandb_root(repo_root)
    wandb_root.mkdir(parents=True, exist_ok=True)
    (wandb_root / "runs").mkdir(parents=True, exist_ok=True)
    (wandb_root / "cache").mkdir(parents=True, exist_ok=True)
    (wandb_root / "config").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")

    wandb_cfg = cfg.get("wandb", {})
    output_dir = Path(cfg["train"]["output_dir"])
    run_name = wandb_cfg.get("name", output_dir.name)
    group = wandb_cfg.get("group", stage)
    project = wandb_cfg.get("project", "adamotion")
    entity = wandb_cfg.get("entity")
    mode = wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "online"))
    tags = list(wandb_cfg.get("tags", []))
    if stage not in tags:
        tags.append(stage)
    representation = cfg.get("data", {}).get("dataset", {}).get("representation")
    if representation and representation not in tags:
        tags.append(str(representation))

    settings = wandb.Settings(_disable_stats=False)
    init_kwargs = dict(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        tags=tags,
        mode=mode,
        dir=str(wandb_root / "runs"),
        config=_to_jsonable(cfg),
        settings=settings,
        reinit="finish_previous",
    )
    try:
        return wandb.init(**init_kwargs)
    except Exception:
        if mode == "offline":
            return None
        init_kwargs["mode"] = "offline"
        try:
            return wandb.init(**init_kwargs)
        except Exception:
            return None


def log_wandb_epoch(run, row: Dict[str, Any], extra: Dict[str, Any] | None = None) -> None:
    if run is None:
        return
    payload = dict(row)
    if extra:
        payload.update(extra)
    run.log(_to_jsonable(payload), step=int(row["epoch"]))


def finish_wandb_run(run, summary: Dict[str, Any] | None = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in _to_jsonable(summary).items():
            run.summary[key] = value
    run.finish()
