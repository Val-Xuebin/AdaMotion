from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.experiment import finalize_run
from lam.dataset import HumanMLTransitionDataset
from lam.modules import LatentActionModel


class MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureMLPLAM(nn.Module):
    family = "feature_mlp"

    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 1024,
        latent_dim: int = 32,
        beta: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout = dropout
        self.encoder = MLP([state_dim * 2, hidden_dim, hidden_dim], dropout=dropout)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLP([state_dim + latent_dim, hidden_dim, hidden_dim, state_dim], dropout=dropout)

    def encode(self, x_t: torch.Tensor, x_tp1: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(torch.cat([x_t, x_tp1], dim=-1))
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        return {"z": z, "mu": mu, "logvar": logvar}

    def forward_transition(self, x_t: torch.Tensor, x_tp1: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(x_t, x_tp1)
        recon = self.decoder(torch.cat([x_t, encoded["z"]], dim=-1))
        mse_loss = F.mse_loss(recon, x_tp1)
        kl_loss = -0.5 * torch.mean(1 + encoded["logvar"] - encoded["mu"].pow(2) - encoded["logvar"].exp())
        loss = mse_loss + self.beta * kl_loss
        return {
            **encoded,
            "recon": recon,
            "mse_loss": mse_loss,
            "kl_loss": kl_loss,
            "loss": loss,
        }


class JointSpatioTemporalLAM(nn.Module):
    family = "joint_st_transformer"

    def __init__(
        self,
        num_joints: int = 22,
        joint_dim: int = 3,
        model_dim: int = 256,
        latent_dim: int = 32,
        enc_blocks: int = 4,
        dec_blocks: int = 4,
        num_heads: int = 4,
        beta: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.enc_blocks = enc_blocks
        self.dec_blocks = dec_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.lam = LatentActionModel(
            in_dim=joint_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def encode(self, x_t: torch.Tensor, x_tp1: torch.Tensor) -> Dict[str, torch.Tensor]:
        joints = torch.stack([x_t, x_tp1], dim=1)
        encoded = self.lam.encode_sequence(joints)
        mu = encoded["z_mu"][:, 0]
        logvar = encoded["z_logvar"][:, 0]
        z = encoded["z_rep"][:, 0, 0]
        return {"z": z, "mu": mu, "logvar": logvar}

    def forward_sequence(self, joints: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.lam({"joints": joints})
        mse_loss = F.mse_loss(outputs["recon"], joints[:, 1:])
        kl_loss = -0.5 * torch.mean(1 + outputs["z_logvar"] - outputs["z_mu"].pow(2) - outputs["z_logvar"].exp())
        loss = mse_loss + self.beta * kl_loss
        return {
            "recon": outputs["recon"],
            "z": outputs["z_rep"],
            "mu": outputs["z_mu"],
            "logvar": outputs["z_logvar"],
            "mse_loss": mse_loss,
            "kl_loss": kl_loss,
            "loss": loss,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle)


def _family_from_cfg(cfg: Dict) -> str:
    model_cfg = cfg.get("model", {})
    if "family" in model_cfg:
        return model_cfg["family"]
    if "hidden_dim" in model_cfg and "model_dim" not in model_cfg:
        return "feature_mlp"
    return "joint_st_transformer"


def build_lam_from_config(cfg: Dict, dataset: HumanMLTransitionDataset | None = None) -> nn.Module:
    family = _family_from_cfg(cfg)
    model_cfg = cfg["model"]
    if family == "feature_mlp":
        state_dim = dataset.feature_dim if dataset is not None else model_cfg.get("state_dim", 263)
        return FeatureMLPLAM(
            state_dim=state_dim,
            hidden_dim=model_cfg["hidden_dim"],
            latent_dim=model_cfg["latent_dim"],
            beta=model_cfg["beta"],
            dropout=model_cfg.get("dropout", 0.0),
        )
    if family == "joint_st_transformer":
        if dataset is None:
            num_joints = model_cfg.get("num_joints", 22)
            joint_dim = model_cfg.get("joint_dim", 3)
        else:
            num_joints = dataset.num_joints
            joint_dim = dataset.joint_dim
        return JointSpatioTemporalLAM(
            num_joints=num_joints,
            joint_dim=joint_dim,
            model_dim=model_cfg["model_dim"],
            latent_dim=model_cfg["latent_dim"],
            enc_blocks=model_cfg["enc_blocks"],
            dec_blocks=model_cfg["dec_blocks"],
            num_heads=model_cfg["num_heads"],
            beta=model_cfg["beta"],
            dropout=model_cfg.get("dropout", 0.0),
        )
    raise ValueError(f"Unsupported LAM family: {family}")


def load_lam_from_checkpoint(ckpt_path: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    family = _family_from_cfg(cfg)
    if family == "feature_mlp":
        state_dim = cfg["model"].get("state_dim")
        if state_dim is None:
            decoder_weight_keys = [
                key for key in ckpt["model"].keys() if key.startswith("decoder.net.") and key.endswith(".weight")
            ]
            if not decoder_weight_keys:
                raise KeyError("Could not infer state_dim from feature-MLP checkpoint: no decoder weights found.")
            last_decoder_weight = max(decoder_weight_keys, key=lambda key: int(key.split(".")[2]))
            state_dim = ckpt["model"][last_decoder_weight].shape[0]
        model = FeatureMLPLAM(
            state_dim=state_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            latent_dim=cfg["model"]["latent_dim"],
            beta=cfg["model"]["beta"],
            dropout=cfg["model"].get("dropout", 0.0),
        )
    elif family == "joint_st_transformer":
        model = JointSpatioTemporalLAM(
            num_joints=cfg["model"].get("num_joints", 22),
            joint_dim=cfg["model"].get("joint_dim", 3),
            model_dim=cfg["model"]["model_dim"],
            latent_dim=cfg["model"]["latent_dim"],
            enc_blocks=cfg["model"]["enc_blocks"],
            dec_blocks=cfg["model"]["dec_blocks"],
            num_heads=cfg["model"]["num_heads"],
            beta=cfg["model"]["beta"],
            dropout=cfg["model"].get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unsupported LAM family in checkpoint: {family}")
    model.load_state_dict(ckpt["model"])
    return model


def train_lam_from_config(cfg: Dict) -> Dict:
    if cfg.get("stage") not in (None, "lam"):
        raise ValueError(f"LAM config expected stage=lam, got {cfg.get('stage')!r}")

    cfg = dict(cfg)
    cfg["stage"] = "lam"
    set_seed(cfg["seed"])

    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    train_set = HumanMLTransitionDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_set = HumanMLTransitionDataset(**val_args)

    family = _family_from_cfg(cfg)
    model = build_lam_from_config(cfg, train_set).to(device)
    cfg.setdefault("model", {})
    if family == "feature_mlp":
        cfg["model"]["state_dim"] = model.state_dim
    else:
        cfg["model"]["num_joints"] = model.num_joints
        cfg["model"]["joint_dim"] = model.joint_dim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    train_loader = _make_loader(train_set, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    val_loader = _make_loader(val_set, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))

    history = []
    best = {"val_loss": float("inf")}
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lam_last.pt"

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_metrics = {"loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0, "count": 0}
        for batch in train_loader:
            batch = _to_device(batch, device)
            if family == "feature_mlp":
                outputs = model.forward_transition(batch["x_t"], batch["x_tp1"])
                batch_size = batch["x_t"].shape[0]
            else:
                outputs = model.forward_sequence(batch["joints"])
                batch_size = batch["joints"].shape[0]
            optimizer.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            optimizer.step()
            train_metrics["count"] += batch_size
            for key in ["loss", "mse_loss", "kl_loss"]:
                train_metrics[key] += float(outputs[key].detach()) * batch_size

        model.eval()
        val_metrics = {"loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0, "count": 0}
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                if family == "feature_mlp":
                    outputs = model.forward_transition(batch["x_t"], batch["x_tp1"])
                    batch_size = batch["x_t"].shape[0]
                else:
                    outputs = model.forward_sequence(batch["joints"])
                    batch_size = batch["joints"].shape[0]
                val_metrics["count"] += batch_size
                for key in ["loss", "mse_loss", "kl_loss"]:
                    val_metrics[key] += float(outputs[key].detach()) * batch_size

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"] / max(1, train_metrics["count"]),
            "train_mse": train_metrics["mse_loss"] / max(1, train_metrics["count"]),
            "train_kl": train_metrics["kl_loss"] / max(1, train_metrics["count"]),
            "val_loss": val_metrics["loss"] / max(1, val_metrics["count"]),
            "val_mse": val_metrics["mse_loss"] / max(1, val_metrics["count"]),
            "val_kl": val_metrics["kl_loss"] / max(1, val_metrics["count"]),
        }
        history.append(row)
        save_payload = {"model": model.state_dict(), "config": cfg}
        if row["val_loss"] < best["val_loss"]:
            best = row
            torch.save({**save_payload, "best": best}, out_dir / "lam_best.pt")
        torch.save({**save_payload, "history": history}, ckpt_path)

    with open(out_dir / "lam_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    finalize_run(
        repo_root=Path(__file__).resolve().parents[2],
        stage="lam",
        cfg=cfg,
        history=history,
        checkpoint_path=str(ckpt_path),
        best_metrics=best,
        extra_summary={"family": family},
    )
    return {"best": best, "checkpoint": str(ckpt_path)}


LAM = JointSpatioTemporalLAM

__all__ = [
    "FeatureMLPLAM",
    "JointSpatioTemporalLAM",
    "LAM",
    "MLP",
    "build_lam_from_config",
    "load_lam_from_checkpoint",
    "set_seed",
    "train_lam_from_config",
]
