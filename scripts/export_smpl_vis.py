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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config
from datasets import MotionContextDataset
from models import ActionAgnosticPredictor, ActionConditionedPredictor, LatentActionAutoencoder


def build_models(world_ckpt_path: str):
    world_ckpt = torch.load(world_ckpt_path, map_location="cpu")
    cfg = world_ckpt["config"]
    lam_path = cfg["model"]["lam_checkpoint"]
    if not os.path.exists(lam_path):
        lam_path = lam_path.replace("/work/adaworld_motion", "/work/adamotion")
    lam_ckpt = torch.load(lam_path, map_location="cpu")
    lam_cfg = lam_ckpt["config"]

    data_cfg = cfg["data"]["dataset"]
    state_dim = 263
    lam = LatentActionAutoencoder(
        state_dim=state_dim,
        hidden_dim=lam_cfg["model"]["hidden_dim"],
        latent_dim=lam_cfg["model"]["latent_dim"],
        beta=lam_cfg["model"]["beta"],
        dropout=lam_cfg["model"].get("dropout", 0.0),
    )
    lam.load_state_dict(lam_ckpt["model"])
    lam.eval()

    predictor = ActionConditionedPredictor(
        state_dim=state_dim,
        latent_dim=lam_cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        context_len=data_cfg["context_len"],
        future_len=data_cfg["future_len"],
        dropout=cfg["model"].get("dropout", 0.0),
    )
    predictor.load_state_dict(world_ckpt["predictor"])
    predictor.eval()

    no_action = ActionAgnosticPredictor(
        state_dim=state_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        context_len=data_cfg["context_len"],
        future_len=data_cfg["future_len"],
        dropout=cfg["model"].get("dropout", 0.0),
    )
    no_action.load_state_dict(world_ckpt["no_action"])
    no_action.eval()
    return lam, predictor, no_action, cfg


def remap_legacy_path(path: str) -> str:
    if path and "/work/adaworld_motion" in path:
        return path.replace("/work/adaworld_motion", "/work/adamotion")
    return path


def export_variant(result_dir: Path, name: str, motion_xyz: np.ndarray, text: str, device: int) -> dict:
    mdm_root = Path("/work/motion_research/projects/motion_generation/mdm")
    old_cwd = Path.cwd()
    os.chdir(mdm_root)
    sys.path.insert(0, str(mdm_root))
    from visualize import vis_utils

    variant_dir = result_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "motion": motion_xyz[None].transpose(0, 2, 3, 1),
        "text": [text],
        "lengths": np.array([motion_xyz.shape[0]], dtype=np.int64),
        "num_samples": 1,
        "num_repetitions": 1,
    }
    results_path = variant_dir / "results.npy"
    np.save(results_path, payload)

    npy2obj = vis_utils.npy2obj(str(results_path), sample_idx=0, rep_idx=0, device=device, cuda=True)
    obj_dir = variant_dir / "sample0_rep0_obj"
    if obj_dir.exists():
        shutil.rmtree(obj_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)
    for frame_i in range(npy2obj.real_num_frames):
        npy2obj.save_obj(str(obj_dir / f"frame{frame_i:03d}.obj"), frame_i)
    smpl_path = variant_dir / "sample0_rep0_smpl_params.npy"
    npy2obj.save_npy(str(smpl_path))
    os.chdir(old_cwd)
    return {
        "variant": name,
        "results_path": str(results_path),
        "obj_dir": str(obj_dir),
        "smpl_path": str(smpl_path),
        "frames": int(npy2obj.real_num_frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-config", default="/work/adamotion/configs/humanml_world_debug.yaml")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.world_config)
    world_ckpt = cfg["train"]["output_dir"] + "/world_best.pt"
    lam, predictor, no_action, train_cfg = build_models(world_ckpt)

    dataset = MotionContextDataset(
        data_root=remap_legacy_path(train_cfg["data"]["dataset"]["data_root"]),
        split=train_cfg["data"].get("val_split", "val"),
        context_len=train_cfg["data"]["dataset"]["context_len"],
        future_len=train_cfg["data"]["dataset"]["future_len"],
        stride=1,
        max_sequences=64,
    )
    sample = dataset[args.sample_idx]
    motion_id = sample["motion_id"]
    full_motion = dataset.base.load_motion(dataset.index[args.sample_idx][0]).astype(np.float32)
    seq_len = min(args.seq_len, full_motion.shape[0])
    context_len = train_cfg["data"]["dataset"]["context_len"]

    gt_feat = torch.from_numpy(full_motion[:seq_len])
    action_pred = gt_feat.clone()
    no_action_pred = gt_feat.clone()

    with torch.no_grad():
        for t in range(context_len, seq_len):
            context = gt_feat[t - context_len:t].unsqueeze(0)
            x_prev = gt_feat[t - 1].unsqueeze(0)
            x_next = gt_feat[t].unsqueeze(0)
            z = lam.encode(x_prev, x_next)["mu"]
            action_pred[t : t + 1] = predictor(context, z).squeeze(0)
            no_action_pred[t : t + 1] = no_action(context).squeeze(0)

    mdm_root = Path("/work/motion_research/projects/motion_generation/mdm")
    sys.path.insert(0, str(mdm_root))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    gt_xyz = recover_from_ric(gt_feat.unsqueeze(0), 22).squeeze(0).cpu().numpy()
    action_xyz = recover_from_ric(action_pred.unsqueeze(0), 22).squeeze(0).cpu().numpy()
    no_action_xyz = recover_from_ric(no_action_pred.unsqueeze(0), 22).squeeze(0).cpu().numpy()

    out_root = ROOT / "experiments" / "smpl_vis" / f"{motion_id}_len{seq_len}"
    out_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    manifests.append(export_variant(out_root, "ground_truth", gt_xyz, f"gt:{motion_id}", args.device))
    manifests.append(export_variant(out_root, "action_conditioned", action_xyz, f"action:{motion_id}", args.device))
    manifests.append(export_variant(out_root, "no_action", no_action_xyz, f"no_action:{motion_id}", args.device))

    report = {
        "motion_id": motion_id,
        "seq_len": seq_len,
        "context_len": context_len,
        "world_checkpoint": world_ckpt,
        "artifacts": manifests,
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
