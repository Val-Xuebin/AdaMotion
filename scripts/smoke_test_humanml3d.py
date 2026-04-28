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
sys.path.insert(0, str(REPO_ROOT / "motionvae"))
sys.path.insert(0, str(REPO_ROOT / "worldmodel"))

from common.humanml_representation import resolve_motion_path, resolve_text_path
from lam.dataset import HumanMLTransitionDataset
from lam.model import build_lam_from_config
from motionvae.dataset import HumanMLMotionAutoencoderDataset
from motionvae.model import MotionVAE
from worldmodel.train import _build_noise_scheduler, _pool_action_sequence
from vwm.data.text_future_dataset import HumanMLTextFutureDataset
from vwm.models.motion_diffusion import ActionConditionedMotionDenoiser


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
    split_ids = []
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
    outputs = model.forward_sequence(joints)
    outputs["loss"].backward()
    return {
        "representation": cfg["data"]["dataset"]["representation"],
        "sample_shape": list(joints.shape),
        "loss": float(outputs["loss"].detach().cpu()),
        "params": _count_params(model),
    }


def _motion_vae_step(cfg: dict, device: torch.device) -> dict:
    dataset = HumanMLMotionAutoencoderDataset(**cfg["data"]["dataset"])
    sample = dataset[0]
    model = MotionVAE(cfg).to(device)
    motion = sample["motion"].unsqueeze(0).to(device)
    outputs = model(motion)
    outputs["loss"].backward()
    return {
        "sample_shape": list(motion.shape),
        "loss": float(outputs["loss"].detach().cpu()),
        "params": _count_params(model),
    }


def _world_model_step(lam_cfg: dict, vae_cfg: dict, world_cfg: dict, device: torch.device) -> dict:
    lam_set = HumanMLTransitionDataset(**lam_cfg["data"]["dataset"])
    lam = build_lam_from_config(lam_cfg, lam_set).to(device)
    vae = MotionVAE(vae_cfg).to(device)
    vae.train()
    dataset = HumanMLTextFutureDataset(**world_cfg["data"]["dataset"])
    sample = dataset[0]

    context_motion = sample["context_motion"].unsqueeze(0).to(device)
    future_motion = sample["future_motion"].unsqueeze(0).to(device)
    action_source = sample["action_source"].unsqueeze(0).to(device)

    with torch.no_grad():
        z_past, _ = vae.encode(context_motion)
        z_future, _ = vae.encode(future_motion)
    action_seq = lam.encode_action_sequence(action_source, texts=texts)
    action_seq = _pool_action_sequence(action_seq, z_future.shape[1])

    denoiser = ActionConditionedMotionDenoiser(
        latent_dim=world_cfg["model"].get("hidden_dim", 256),
        vae_latent_dim=vae.latent_dim,
        action_dim=lam.latent_dim,
        motion_joints=world_cfg["model"].get("motion_latent_joints", 7),
        n_heads=world_cfg["model"].get("n_heads", 4),
        n_layers=world_cfg["model"].get("n_layers", 5),
        ff_dim=world_cfg["model"].get("ff_dim", 512),
        dropout=world_cfg["model"].get("dropout", 0.1),
        activation=world_cfg["model"].get("activation", "gelu"),
        max_text_tokens=world_cfg["model"].get("max_text_tokens", 32),
    ).to(device)

    scheduler = _build_noise_scheduler(world_cfg)
    timesteps = torch.randint(0, world_cfg["model"].get("num_train_timesteps", 1000), (1,), device=device).long()
    noise = torch.randn_like(z_future)
    noisy_future = scheduler.add_noise(z_future, noise, timesteps)
    pred = denoiser(
        noisy_future=noisy_future,
        timesteps=timesteps,
        texts=[sample["text"]],
        past_latent=z_past,
        action_latent_seq=action_seq,
        len_mask=torch.ones(1, z_future.shape[1], dtype=torch.bool, device=device),
    )
    loss = (pred - noise).pow(2).mean()
    loss.backward()
    return {
        "context_shape": list(context_motion.shape),
        "future_shape": list(future_motion.shape),
        "action_shape": list(action_source.shape),
        "latent_shape": list(z_future.shape),
        "loss": float(loss.detach().cpu()),
        "params": _count_params(denoiser),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/dataset/HumanML3D")
    parser.add_argument("--lam-config", default="/workspace/AdaMotion/configs/humanml_sal_rep_lam_debug.yaml")
    parser.add_argument("--vae-config", default="/workspace/AdaMotion/configs/humanml_motion_vae_debug.yaml")
    parser.add_argument("--world-config", default="/workspace/AdaMotion/configs/humanml_salad_diffusion_debug.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    lam_cfg = _load_yaml(Path(args.lam_config))
    vae_cfg = _load_yaml(Path(args.vae_config))
    world_cfg = _load_yaml(Path(args.world_config))
    for cfg in (lam_cfg, vae_cfg, world_cfg):
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
        "motion_vae": _motion_vae_step(vae_cfg, device),
        "world_model": _world_model_step(lam_cfg, vae_cfg, world_cfg, device),
    }
    if device.type == "cuda":
        report["hardware"]["max_memory_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
