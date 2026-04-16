#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lam"))
sys.path.insert(0, str(ROOT / "worldmodel"))

from lam.model import load_lam_from_checkpoint
from train import (
    FeatureActionAgnosticPredictor,
    FeatureActionConditionedPredictor,
    JointActionAgnosticPredictor,
    JointActionConditionedPredictor,
)
from vwm.data.dataset import HumanMLContextDataset


DEFAULT_RENDER_SAMPLE_INDICES = [0, 8, 16, 24]


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    return cfg


def family_from_cfg(cfg: dict) -> str:
    return cfg.get("model", {}).get("family", "joint_st_transformer")


def build_world_models(world_ckpt_path: str):
    world_ckpt = torch.load(world_ckpt_path, map_location="cpu")
    cfg = world_ckpt["config"]
    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    lam.eval()
    data_cfg = cfg["data"]["dataset"]
    family = family_from_cfg(cfg)
    if family == "feature_mlp":
        state_dim = data_cfg.get("state_dim", 263)
        predictor = FeatureActionConditionedPredictor(
            state_dim=state_dim,
            latent_dim=lam.latent_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=data_cfg["context_len"],
            future_len=data_cfg["future_len"],
            dropout=cfg["model"].get("dropout", 0.0),
        )
        no_action = FeatureActionAgnosticPredictor(
            state_dim=state_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=data_cfg["context_len"],
            future_len=data_cfg["future_len"],
            dropout=cfg["model"].get("dropout", 0.0),
        )
    else:
        predictor = JointActionConditionedPredictor(
            state_dim=66,
            latent_dim=lam.latent_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=data_cfg["context_len"],
            future_len=data_cfg["future_len"],
            num_joints=data_cfg.get("num_joints", 22),
            joint_dim=data_cfg.get("joint_dim", 3),
            dropout=cfg["model"].get("dropout", 0.0),
        )
        no_action = JointActionAgnosticPredictor(
            state_dim=66,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=data_cfg["context_len"],
            future_len=data_cfg["future_len"],
            num_joints=data_cfg.get("num_joints", 22),
            joint_dim=data_cfg.get("joint_dim", 3),
            dropout=cfg["model"].get("dropout", 0.0),
        )
    predictor.load_state_dict(world_ckpt["predictor"])
    predictor.eval()
    no_action.load_state_dict(world_ckpt["no_action"])
    no_action.eval()
    return lam, predictor, no_action, cfg


def load_t2m_kinematic_chain():
    mdm_root = Path("/work/motion_research/projects/motion_generation/mdm")
    sys.path.insert(0, str(mdm_root))
    try:
        from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

        return t2m_kinematic_chain
    except Exception:
        return [[idx, idx + 1] for idx in range(21)]


def save_preview_svg(output_path: Path, motion_xyz: np.ndarray, title: str) -> None:
    chain = load_t2m_kinematic_chain()
    frame_ids = np.linspace(0, max(0, motion_xyz.shape[0] - 1), num=min(4, motion_xyz.shape[0]), dtype=int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    height = 340
    panel_w = 260
    panel_h = 220
    top = 70
    left = 40
    gap = 30
    colors = ["#0f172a", "#2563eb", "#dc2626", "#059669", "#d97706"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-size="22" font-family="monospace">{title}</text>',
    ]

    for panel_idx, frame_idx in enumerate(frame_ids):
        frame = motion_xyz[frame_idx] - motion_xyz[frame_idx, 0:1]
        pts2d = np.stack([frame[:, 0], -frame[:, 2]], axis=-1)
        mins = pts2d.min(axis=0)
        maxs = pts2d.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        scale = min(panel_w / span[0], panel_h / span[1]) * 0.75
        panel_x = left + panel_idx * (panel_w + gap)
        panel_y = top
        parts.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#d4d4d8"/>'
        )
        parts.append(
            f'<text x="{panel_x + 10}" y="{panel_y + 20}" font-size="14" font-family="monospace">t={int(frame_idx)}</text>'
        )

        center = 0.5 * (mins + maxs)
        for chain_idx, segment in enumerate(chain):
            seg = np.asarray(segment, dtype=int)
            coords = []
            for joint_idx in seg:
                px = panel_x + panel_w / 2 + (pts2d[joint_idx, 0] - center[0]) * scale
                py = panel_y + panel_h / 2 + (pts2d[joint_idx, 1] - center[1]) * scale
                coords.append(f"{px:.1f},{py:.1f}")
            color = colors[chain_idx % len(colors)]
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(coords)}"/>')
        for joint_idx in range(pts2d.shape[0]):
            px = panel_x + panel_w / 2 + (pts2d[joint_idx, 0] - center[0]) * scale
            py = panel_y + panel_h / 2 + (pts2d[joint_idx, 1] - center[1]) * scale
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="#111827"/>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def export_variant(result_dir: Path, name: str, motion_xyz: np.ndarray, text: str, device: int) -> dict:
    variant_dir = result_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = variant_dir / "motion_xyz.npy"
    preview_path = variant_dir / "preview.svg"
    np.save(xyz_path, motion_xyz)
    save_preview_svg(preview_path, motion_xyz, text)

    payload = {
        "motion": motion_xyz[None].transpose(0, 2, 3, 1),
        "text": [text],
        "lengths": np.array([motion_xyz.shape[0]], dtype=np.int64),
        "num_samples": 1,
        "num_repetitions": 1,
    }
    results_path = variant_dir / "results.npy"
    np.save(results_path, payload)

    artifact = {
        "variant": name,
        "render_mode": "preview_only",
        "xyz_path": str(xyz_path),
        "preview_path": str(preview_path),
        "results_path": str(results_path),
        "frames": int(motion_xyz.shape[0]),
    }
    mdm_root = Path("/work/motion_research/projects/motion_generation/mdm")
    old_cwd = Path.cwd()
    try:
        os.chdir(mdm_root)
        sys.path.insert(0, str(mdm_root))
        from visualize import vis_utils

        npy2obj = vis_utils.npy2obj(str(results_path), sample_idx=0, rep_idx=0, device=device, cuda=True)
        obj_dir = variant_dir / "sample0_rep0_obj"
        if obj_dir.exists():
            shutil.rmtree(obj_dir)
        obj_dir.mkdir(parents=True, exist_ok=True)
        for frame_i in range(npy2obj.real_num_frames):
            npy2obj.save_obj(str(obj_dir / f"frame{frame_i:03d}.obj"), frame_i)
        smpl_path = variant_dir / "sample0_rep0_smpl_params.npy"
        npy2obj.save_npy(str(smpl_path))
        artifact.update(
            {
                "render_mode": "smpl_obj",
                "obj_dir": str(obj_dir),
                "smpl_path": str(smpl_path),
                "frames": int(npy2obj.real_num_frames),
            }
        )
    except Exception as exc:
        artifact["render_warning"] = str(exc)
    finally:
        os.chdir(old_cwd)
    return artifact


def recover_xyz(sequence: torch.Tensor, representation: str) -> np.ndarray:
    if representation == "joint_positions":
        return sequence.cpu().numpy()
    mdm_root = Path("/work/motion_research/projects/motion_generation/mdm")
    sys.path.insert(0, str(mdm_root))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    return recover_from_ric(sequence.unsqueeze(0), 22).squeeze(0).cpu().numpy()


def predict_sequence(lam, predictor, no_action, full_motion: np.ndarray, context_len: int, representation: str):
    gt = torch.from_numpy(full_motion)
    action_pred = gt.clone()
    no_action_pred = gt.clone()
    with torch.no_grad():
        for t in range(context_len, gt.shape[0]):
            context = gt[t - context_len : t].unsqueeze(0)
            x_prev = gt[t - 1].unsqueeze(0)
            x_next = gt[t].unsqueeze(0)
            z = lam.encode(x_prev, x_next)["mu"]
            action_pred[t : t + 1] = predictor(context, z).squeeze(0)
            no_action_pred[t : t + 1] = no_action(context).squeeze(0)

    gt_xyz = recover_xyz(gt, representation)
    action_xyz = recover_xyz(action_pred, representation)
    no_action_xyz = recover_xyz(no_action_pred, representation)
    return gt, action_pred, no_action_pred, gt_xyz, action_xyz, no_action_xyz


def render_sample(dataset, sample_idx: int, seq_len: int, lam, predictor, no_action, cfg: dict, render_root: Path, device: int):
    sample = dataset[sample_idx]
    seq_idx, start_idx = dataset.index[sample_idx]
    motion_id = sample["motion_id"]
    full_motion = dataset.base.load_motion(seq_idx).astype(np.float32)
    seq_len = min(seq_len, full_motion.shape[0] - start_idx)
    full_motion = full_motion[start_idx : start_idx + seq_len]
    context_len = cfg["data"]["dataset"]["context_len"]
    representation = dataset.base.representation
    gt, action_pred, no_action_pred, gt_xyz, action_xyz, no_action_xyz = predict_sequence(
        lam,
        predictor,
        no_action,
        full_motion,
        context_len,
        representation,
    )

    sample_root = render_root / f"{motion_id}_start{start_idx:04d}_len{seq_len}"
    sample_root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        export_variant(sample_root, "ground_truth", gt_xyz, f"gt:{motion_id}", device),
        export_variant(sample_root, "action_conditioned", action_xyz, f"action:{motion_id}", device),
        export_variant(sample_root, "no_action", no_action_xyz, f"no_action:{motion_id}", device),
    ]
    report = {
        "motion_id": motion_id,
        "sample_idx": sample_idx,
        "start_idx": start_idx,
        "seq_len": seq_len,
        "context_len": context_len,
        "representation": representation,
        "feature_mse_action": float(torch.mean((action_pred - gt) ** 2).item()),
        "feature_mse_no_action": float(torch.mean((no_action_pred - gt) ** 2).item()),
        "xyz_mse_action": float(np.mean((action_xyz - gt_xyz) ** 2)),
        "xyz_mse_no_action": float(np.mean((no_action_xyz - gt_xyz) ** 2)),
        "artifacts": artifacts,
    }
    with open(sample_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-config", required=True)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=DEFAULT_RENDER_SAMPLE_INDICES)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.world_config)
    world_ckpt = cfg["train"]["output_dir"] + "/world_best.pt"
    lam, predictor, no_action, train_cfg = build_world_models(world_ckpt)
    dataset = HumanMLContextDataset(
        data_root=train_cfg["data"]["dataset"]["data_root"],
        representation=train_cfg["data"]["dataset"].get("representation", "joint_positions"),
        split=train_cfg["data"].get("val_split", "val"),
        context_len=train_cfg["data"]["dataset"]["context_len"],
        future_len=train_cfg["data"]["dataset"]["future_len"],
        stride=1,
        max_sequences=64,
    )

    render_root = Path(train_cfg["train"]["output_dir"]) / "renders"
    render_root.mkdir(parents=True, exist_ok=True)
    reports = [
        render_sample(dataset, idx, args.seq_len, lam, predictor, no_action, train_cfg, render_root, args.device)
        for idx in args.sample_indices
    ]
    summary = {
        "run_name": Path(train_cfg["train"]["output_dir"]).name,
        "world_checkpoint": world_ckpt,
        "render_root": str(render_root),
        "sample_indices": args.sample_indices,
        "num_samples": len(reports),
        "avg_feature_mse_action": float(np.mean([row["feature_mse_action"] for row in reports])),
        "avg_feature_mse_no_action": float(np.mean([row["feature_mse_no_action"] for row in reports])),
        "avg_xyz_mse_action": float(np.mean([row["xyz_mse_action"] for row in reports])),
        "avg_xyz_mse_no_action": float(np.mean([row["xyz_mse_no_action"] for row in reports])),
        "samples": reports,
    }
    with open(render_root / "render_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_path = Path(train_cfg["train"]["output_dir"]) / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            run_summary = json.load(f)
        run_summary["render_root"] = str(render_root)
        run_summary["render_summary_path"] = str(render_root / "render_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
