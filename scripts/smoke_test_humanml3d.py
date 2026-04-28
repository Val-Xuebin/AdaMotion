#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lam"))
sys.path.insert(0, str(REPO_ROOT / "worldmodel"))

from common.humanml_representation import resolve_motion_path, resolve_text_path
from lam.dataset import HumanMLTransitionDataset
from lam.model import build_lam_from_config, load_lam_from_checkpoint
from vwm.data.full_motion_action_dataset import HumanMLFullMotionActionDataset
from vwm.models.action_prior import TextLengthActionPrior
from vwm.models.salad_official import load_official_salad_action_denoiser, load_official_salad_vae
from worldmodel.train import _pool_action_sequence


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _dataset_summary(data_root: Path, split: str) -> dict:
    with open(data_root / f"{split}.txt", "r", encoding="utf-8") as f:
        split_ids = [line.strip() for line in f if line.strip() and not line.startswith("._")]
    feature_missing = 0
    text_missing = 0
    for motion_id in split_ids:
        if not resolve_motion_path(data_root, "new_joint_vecs", motion_id).exists():
            feature_missing += 1
        if not resolve_text_path(data_root, motion_id).exists():
            text_missing += 1
    return {
        "split": split,
        "num_ids": len(split_ids),
        "missing_new_joint_vecs": feature_missing,
        "missing_texts": text_missing,
    }


def _lam_step(cfg: dict, device: torch.device) -> dict:
    dataset = HumanMLTransitionDataset(**cfg["data"]["dataset"])
    sample = dataset[0]
    model = build_lam_from_config(cfg, dataset).to(device)
    joints = sample["joints"].unsqueeze(0).to(device)
    outputs = model.forward_sequence(joints, texts=[sample.get("text", "")])
    outputs["loss"].backward()
    return {
        "sample_shape": list(joints.shape),
        "loss": float(outputs["loss"].detach().cpu()),
        "params": _count_params(model),
    }


def _adapter_step(adapter_cfg: dict, device: torch.device) -> dict:
    dataset = HumanMLFullMotionActionDataset(**adapter_cfg["data"]["dataset"])
    sample = dataset[0]
    lam = load_lam_from_checkpoint(adapter_cfg["model"]["lam_checkpoint"]).to(device).eval()
    vae = load_official_salad_vae(
        adapter_cfg["model"]["official_salad_vae_opt"],
        adapter_cfg["model"]["official_salad_vae_checkpoint"],
        device,
    )
    denoiser = load_official_salad_action_denoiser(
        adapter_cfg["model"]["official_salad_denoiser_opt"],
        adapter_cfg["model"]["official_salad_denoiser_checkpoint"],
        vae_dim=vae.latent_dim,
        action_dim=lam.latent_dim,
        device=device,
        train_base=adapter_cfg["model"].get("train_base_denoiser", False),
    )

    motion = sample["motion"].unsqueeze(0).to(device)
    action_source = sample["action_source"].unsqueeze(0).to(device)
    with torch.no_grad():
        z_motion = vae.encode_deterministic(motion)[0]
        action_seq = lam.encode_action_sequence(
            action_source,
            texts=[sample["text"]],
            start_timestep=torch.tensor([sample["start_idx"]], device=device),
        )
        action_seq = _pool_action_sequence(action_seq, z_motion.shape[1])
    timesteps = torch.zeros(1, dtype=torch.long, device=device)
    pred = denoiser(
        noisy_motion_latent=z_motion,
        timesteps=timesteps,
        texts=[sample["text"]],
        action_latent_seq=action_seq,
        len_mask=torch.ones(1, z_motion.shape[1], dtype=torch.bool, device=device),
    )
    return {
        "motion_shape": list(motion.shape),
        "latent_shape": list(z_motion.shape),
        "action_shape": list(action_seq.shape),
        "pred_shape": list(pred.shape),
        "params": _count_params(denoiser),
    }


def _prior_step(prior_cfg: dict, device: torch.device) -> dict:
    dataset = HumanMLFullMotionActionDataset(**prior_cfg["data"]["dataset"])
    sample = dataset[0]
    lam = load_lam_from_checkpoint(prior_cfg["model"]["lam_checkpoint"]).to(device).eval()
    prior = TextLengthActionPrior(
        action_dim=lam.latent_dim,
        hidden_dim=prior_cfg["model"].get("hidden_dim", 256),
        latent_steps=prior_cfg["model"].get("latent_steps", 49),
        dropout=prior_cfg["model"].get("dropout", 0.1),
        clip_version=prior_cfg["model"].get("clip_version", "ViT-B/32"),
        max_text_tokens=prior_cfg["model"].get("max_text_tokens", 32),
        max_motion_length=prior_cfg["data"]["dataset"].get("max_motion_length", 196),
    ).to(device)
    pred = prior([sample["text"]], torch.tensor([sample["length"]], device=device))
    return {
        "pred_shape": list(pred.shape),
        "params": _count_params(prior),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/dataset/HumanML3D")
    parser.add_argument("--lam-config", default="/workspace/AdaMotion/configs/lam_mom_full.yaml")
    parser.add_argument("--adapter-config", default="/workspace/AdaMotion/configs/salad_adapter_mom_full.yaml")
    parser.add_argument("--prior-config", default="/workspace/AdaMotion/configs/salad_prior_mom_full.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    lam_cfg = _load_yaml(Path(args.lam_config))
    adapter_cfg = _load_yaml(Path(args.adapter_config))
    prior_cfg = _load_yaml(Path(args.prior_config))
    for cfg in (lam_cfg, adapter_cfg, prior_cfg):
        cfg["data"]["dataset"]["data_root"] = str(data_root)
        cfg["train"]["device"] = args.device

    device = _pick_device(args.device)
    report = {
        "hardware": {
            "requested_device": args.device,
            "resolved_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "train": _dataset_summary(data_root, "train"),
            "val": _dataset_summary(data_root, "val"),
            "test": _dataset_summary(data_root, "test"),
        },
        "lam": _lam_step(lam_cfg, device),
        "salad_adapter": _adapter_step(adapter_cfg, device),
        "salad_prior": _prior_step(prior_cfg, device),
    }
    if device.type == "cuda":
        report["hardware"]["max_memory_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
