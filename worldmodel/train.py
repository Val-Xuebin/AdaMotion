#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lam"))

from common.experiment import finalize_run
from lam.model import load_lam_from_checkpoint
from vwm.data.dataset import HumanMLContextDataset


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


class FeatureActionConditionedPredictor(nn.Module):
    family = "feature_mlp"

    def __init__(
        self,
        state_dim: int = 263,
        latent_dim: int = 32,
        hidden_dim: int = 1024,
        context_len: int = 6,
        future_len: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_len = future_len
        self.context_encoder = MLP([context_len * state_dim, hidden_dim, hidden_dim], dropout=dropout)
        self.head = MLP([hidden_dim + latent_dim, hidden_dim, future_len * state_dim], dropout=dropout)

    def forward(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = context.shape[0]
        context_hidden = self.context_encoder(context.reshape(batch_size, -1))
        outputs = self.head(torch.cat([context_hidden, z], dim=-1))
        return outputs.reshape(batch_size, self.future_len, self.state_dim)


class FeatureActionAgnosticPredictor(nn.Module):
    family = "feature_mlp"

    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 1024,
        context_len: int = 6,
        future_len: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_len = future_len
        self.net = MLP([context_len * state_dim, hidden_dim, hidden_dim, future_len * state_dim], dropout=dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch_size = context.shape[0]
        outputs = self.net(context.reshape(batch_size, -1))
        return outputs.reshape(batch_size, self.future_len, self.state_dim)


class JointActionConditionedPredictor(nn.Module):
    family = "joint_st_transformer"

    def __init__(
        self,
        state_dim: int = 66,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        context_len: int = 6,
        future_len: int = 1,
        num_joints: int = 22,
        joint_dim: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_len = future_len
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.context_encoder = MLP([context_len * state_dim, hidden_dim, hidden_dim], dropout=dropout)
        self.head = MLP([hidden_dim + latent_dim, hidden_dim, future_len * state_dim], dropout=dropout)

    def forward(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = context.shape[0]
        context_hidden = self.context_encoder(context.reshape(batch_size, -1))
        outputs = self.head(torch.cat([context_hidden, z], dim=-1))
        return outputs.reshape(batch_size, self.future_len, self.num_joints, self.joint_dim)


class JointActionAgnosticPredictor(nn.Module):
    family = "joint_st_transformer"

    def __init__(
        self,
        state_dim: int = 66,
        hidden_dim: int = 512,
        context_len: int = 6,
        future_len: int = 1,
        num_joints: int = 22,
        joint_dim: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_len = future_len
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.net = MLP([context_len * state_dim, hidden_dim, hidden_dim, future_len * state_dim], dropout=dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch_size = context.shape[0]
        outputs = self.net(context.reshape(batch_size, -1))
        return outputs.reshape(batch_size, self.future_len, self.num_joints, self.joint_dim)


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    return cfg


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
    representation = cfg.get("data", {}).get("dataset", {}).get("representation")
    if representation == "humanml_feature_vector":
        return "feature_mlp"
    return "joint_st_transformer"


def build_world_models(cfg: Dict, train_set: HumanMLContextDataset, lam: nn.Module):
    family = _family_from_cfg(cfg)
    if family == "feature_mlp":
        predictor = FeatureActionConditionedPredictor(
            state_dim=train_set.feature_dim,
            latent_dim=lam.latent_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=cfg["data"]["dataset"]["context_len"],
            future_len=cfg["data"]["dataset"]["future_len"],
            dropout=cfg["model"].get("dropout", 0.0),
        )
        no_action = FeatureActionAgnosticPredictor(
            state_dim=train_set.feature_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=cfg["data"]["dataset"]["context_len"],
            future_len=cfg["data"]["dataset"]["future_len"],
            dropout=cfg["model"].get("dropout", 0.0),
        )
    elif family == "joint_st_transformer":
        predictor = JointActionConditionedPredictor(
            state_dim=train_set.feature_dim,
            latent_dim=lam.latent_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=cfg["data"]["dataset"]["context_len"],
            future_len=cfg["data"]["dataset"]["future_len"],
            num_joints=train_set.base.num_joints,
            joint_dim=train_set.base.joint_dim,
            dropout=cfg["model"].get("dropout", 0.0),
        )
        no_action = JointActionAgnosticPredictor(
            state_dim=train_set.feature_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            context_len=cfg["data"]["dataset"]["context_len"],
            future_len=cfg["data"]["dataset"]["future_len"],
            num_joints=train_set.base.num_joints,
            joint_dim=train_set.base.joint_dim,
            dropout=cfg["model"].get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unsupported world-model family: {family}")
    return predictor, no_action


def train_world_model_from_config(cfg: Dict) -> Dict:
    if cfg.get("stage") not in (None, "world_model"):
        raise ValueError(f"World-model config expected stage=world_model, got {cfg.get('stage')!r}")

    cfg = dict(cfg)
    cfg["stage"] = "world_model"
    set_seed(cfg["seed"])

    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    train_set = HumanMLContextDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_set = HumanMLContextDataset(**val_args)

    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    lam.to(device).eval()
    for param in lam.parameters():
        param.requires_grad = False

    family = _family_from_cfg(cfg)
    predictor, no_action = build_world_models(cfg, train_set, lam)
    predictor = predictor.to(device)
    no_action = no_action.to(device)

    opt_pred = torch.optim.AdamW(
        predictor.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    opt_noact = torch.optim.AdamW(
        no_action.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    train_loader = _make_loader(train_set, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    val_loader = _make_loader(val_set, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))

    history = []
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"val_action_loss": float("inf")}

    for epoch in range(cfg["train"]["epochs"]):
        predictor.train()
        no_action.train()
        train_action_loss = 0.0
        train_no_action_loss = 0.0
        train_count = 0

        for batch in train_loader:
            batch = _to_device(batch, device)
            with torch.no_grad():
                z = lam.encode(batch["x_prev"], batch["x_next"])["mu"]

            pred = predictor(batch["context"], z)
            pred_no = no_action(batch["context"])
            loss_pred = F.mse_loss(pred, batch["target"])
            loss_no = F.mse_loss(pred_no, batch["target"])

            opt_pred.zero_grad(set_to_none=True)
            loss_pred.backward()
            opt_pred.step()

            opt_noact.zero_grad(set_to_none=True)
            loss_no.backward()
            opt_noact.step()

            batch_size = batch["context"].shape[0]
            train_count += batch_size
            train_action_loss += float(loss_pred.detach()) * batch_size
            train_no_action_loss += float(loss_no.detach()) * batch_size

        predictor.eval()
        no_action.eval()
        val_action_loss = 0.0
        val_no_action_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                z = lam.encode(batch["x_prev"], batch["x_next"])["mu"]
                pred = predictor(batch["context"], z)
                pred_no = no_action(batch["context"])
                loss_pred = F.mse_loss(pred, batch["target"])
                loss_no = F.mse_loss(pred_no, batch["target"])
                batch_size = batch["context"].shape[0]
                val_count += batch_size
                val_action_loss += float(loss_pred.detach()) * batch_size
                val_no_action_loss += float(loss_no.detach()) * batch_size

        row = {
            "epoch": epoch,
            "train_action_loss": train_action_loss / max(1, train_count),
            "train_no_action_loss": train_no_action_loss / max(1, train_count),
            "val_action_loss": val_action_loss / max(1, val_count),
            "val_no_action_loss": val_no_action_loss / max(1, val_count),
            "val_gain": (val_no_action_loss - val_action_loss) / max(1, val_count),
        }
        history.append(row)
        save_payload = {"predictor": predictor.state_dict(), "no_action": no_action.state_dict(), "config": cfg}
        if row["val_action_loss"] < best["val_action_loss"]:
            best = row
            torch.save({**save_payload, "best": best}, out_dir / "world_best.pt")
        torch.save({**save_payload, "history": history}, out_dir / "world_last.pt")

    with open(out_dir / "world_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    finalize_run(
        repo_root=REPO_ROOT,
        stage="world_model",
        cfg=cfg,
        history=history,
        checkpoint_path=str(out_dir / "world_last.pt"),
        best_metrics=best,
        extra_summary={"family": family},
    )
    return {"best": best, "checkpoint": str(out_dir / "world_last.pt")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AdaMotion world model with an AdaWorld-style entrypoint.")
    parser.add_argument("--base", nargs="+", required=True, help="Compatible with AdaWorld's --base config argument.")
    args = parser.parse_args()

    if len(args.base) != 1:
        raise ValueError("AdaMotion currently expects a single config file passed via --base.")

    cfg = load_config(args.base[0])
    result = train_world_model_from_config(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
