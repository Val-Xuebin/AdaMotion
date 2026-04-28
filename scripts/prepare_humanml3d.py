#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
MDM_ROOT = WORKSPACE_ROOT / "humanmodels" / "motion-diffusion-model"
sys.path.insert(0, str(MDM_ROOT))

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.scripts import motion_process as mp
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets


def _iter_motion_ids(data_root: Path, split: str | None) -> list[str]:
    if split is None:
        return sorted(
            path.stem
            for path in (data_root / "new_joints").rglob("*.npy")
            if path.is_file() and not path.name.startswith("._")
        )
    split_file = data_root / f"{split}.txt"
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("._")]


def _resolve_path(data_root: Path, subdir: str, motion_id: str, suffix: str) -> Path:
    base = data_root / subdir
    candidates = [base / f"{motion_id}{suffix}", *(base / shard / f"{motion_id}{suffix}" for shard in ("00", "01", "02"))]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _configure_motion_process(data_root: Path, example_id: str | None) -> str:
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.joints_num = 22
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain

    motion_ids = _iter_motion_ids(data_root, None)
    if not motion_ids:
        raise RuntimeError(f"No source joints found under {data_root / 'new_joints'}")
    example_id = example_id or motion_ids[0]
    example_path = _resolve_path(data_root, "new_joints", example_id, ".npy")
    if not example_path.exists():
        raise FileNotFoundError(f"Example motion not found: {example_path}")

    example_data = np.load(example_path)
    if example_data.ndim != 3 or example_data.shape[1:] != (22, 3):
        raise ValueError(
            f"Expected {example_path} to have shape [T, 22, 3], got {tuple(example_data.shape)}"
        )
    tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
    mp.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example_data[0]).float())
    return example_id


def _process_one(data_root: Path, motion_id: str, overwrite: bool) -> dict:
    src_path = _resolve_path(data_root, "new_joints", motion_id, ".npy")
    dst_path = _resolve_path(data_root, "new_joint_vecs", motion_id, ".npy")
    if not src_path.exists():
        return {"motion_id": motion_id, "status": "missing_source"}
    if dst_path.exists() and not overwrite:
        motion = np.load(dst_path, mmap_mode="r")
        return {
            "motion_id": motion_id,
            "status": "exists",
            "frames": int(motion.shape[0]),
            "feature_dim": int(motion.shape[1]),
        }

    src = np.load(src_path).astype(np.float32)
    if src.ndim != 3 or src.shape[1:] != (22, 3):
        return {"motion_id": motion_id, "status": "bad_shape", "shape": list(src.shape)}

    data, _, _, _ = mp.process_file(src, 0.002)
    recon = mp.recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), mp.joints_num).squeeze(0).numpy()
    recon_mse = float(np.mean((recon - src[:-1]) ** 2))

    _ensure_parent(dst_path)
    np.save(dst_path, data.astype(np.float32))
    return {
        "motion_id": motion_id,
        "status": "written",
        "frames": int(data.shape[0]),
        "feature_dim": int(data.shape[1]),
        "recon_mse": recon_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/dataset/HumanML3D")
    parser.add_argument("--split", choices=["train", "val", "test", "train_val", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--example-id", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not (data_root / "new_joints").exists():
        raise FileNotFoundError(f"Missing source directory: {data_root / 'new_joints'}")

    split = None if args.split in (None, "all") else args.split
    example_id = _configure_motion_process(data_root, args.example_id)

    motion_ids = _iter_motion_ids(data_root, split)
    if args.limit is not None:
        motion_ids = motion_ids[: args.limit]

    summary = {
        "data_root": str(data_root),
        "source": "new_joints",
        "target": "new_joint_vecs",
        "example_id": example_id,
        "requested": len(motion_ids),
        "written": 0,
        "exists": 0,
        "missing_source": 0,
        "missing_target": 0,
        "bad_shape": 0,
        "verified": 0,
        "feature_dim_set": set(),
        "recon_mse_mean": 0.0,
    }
    recon_mse_values: list[float] = []
    details: list[dict] = []

    for motion_id in motion_ids:
        dst_path = _resolve_path(data_root, "new_joint_vecs", motion_id, ".npy")
        if args.verify_only:
            if dst_path.exists():
                motion = np.load(dst_path, mmap_mode="r")
                status = {
                    "motion_id": motion_id,
                    "status": "verified",
                    "frames": int(motion.shape[0]),
                    "feature_dim": int(motion.shape[1]),
                }
            else:
                status = {"motion_id": motion_id, "status": "missing_target"}
        else:
            status = _process_one(data_root, motion_id, overwrite=args.overwrite)
        details.append(status)
        tag = status["status"]
        if tag == "written":
            summary["written"] += 1
            summary["feature_dim_set"].add(status["feature_dim"])
            recon_mse_values.append(status["recon_mse"])
        elif tag == "exists":
            summary["exists"] += 1
            summary["feature_dim_set"].add(status["feature_dim"])
        elif tag == "verified":
            summary["verified"] += 1
            summary["feature_dim_set"].add(status["feature_dim"])
        elif tag == "missing_source":
            summary["missing_source"] += 1
        elif tag == "missing_target":
            summary["missing_target"] += 1
        elif tag == "bad_shape":
            summary["bad_shape"] += 1

    summary["feature_dim_set"] = sorted(summary["feature_dim_set"])
    if recon_mse_values:
        summary["recon_mse_mean"] = float(np.mean(recon_mse_values))

    print(json.dumps({"summary": summary, "details": details[: min(32, len(details))]}, indent=2))


if __name__ == "__main__":
    main()
