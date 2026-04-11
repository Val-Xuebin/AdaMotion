from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import MotionContextDataset, MotionTransitionDataset
from models import ActionAgnosticPredictor, ActionConditionedPredictor, LatentActionAutoencoder


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


def train_lam(cfg: Dict) -> Dict:
    set_seed(cfg["seed"])
    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    train_set = MotionTransitionDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_set = MotionTransitionDataset(**val_args)

    model = LatentActionAutoencoder(
        state_dim=train_set.feature_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        beta=cfg["model"]["beta"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))

    train_loader = _make_loader(train_set, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    val_loader = _make_loader(val_set, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))

    history = []
    best = {"val_loss": float("inf")}
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lam_last.pt"

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        agg = {"loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0, "count": 0}
        for batch in train_loader:
            batch = _to_device(batch, device)
            out = model(batch["x_t"], batch["x_tp1"])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            bs = batch["x_t"].shape[0]
            agg["count"] += bs
            for key in ["loss", "mse_loss", "kl_loss"]:
                agg[key] += float(out[key].detach()) * bs

        model.eval()
        val = {"loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0, "count": 0}
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                out = model(batch["x_t"], batch["x_tp1"])
                bs = batch["x_t"].shape[0]
                val["count"] += bs
                for key in ["loss", "mse_loss", "kl_loss"]:
                    val[key] += float(out[key].detach()) * bs

        row = {
            "epoch": epoch,
            "train_loss": agg["loss"] / max(1, agg["count"]),
            "train_mse": agg["mse_loss"] / max(1, agg["count"]),
            "train_kl": agg["kl_loss"] / max(1, agg["count"]),
            "val_loss": val["loss"] / max(1, val["count"]),
            "val_mse": val["mse_loss"] / max(1, val["count"]),
            "val_kl": val["kl_loss"] / max(1, val["count"]),
        }
        history.append(row)
        if row["val_loss"] < best["val_loss"]:
            best = row
            torch.save({"model": model.state_dict(), "config": cfg, "best": best}, out_dir / "lam_best.pt")
        torch.save({"model": model.state_dict(), "config": cfg, "history": history}, ckpt_path)

    with open(out_dir / "lam_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"best": best, "checkpoint": str(ckpt_path)}


def train_world_model(cfg: Dict) -> Dict:
    set_seed(cfg["seed"])
    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    train_set = MotionContextDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_set = MotionContextDataset(**val_args)

    lam_ckpt = torch.load(cfg["model"]["lam_checkpoint"], map_location="cpu")
    lam_cfg = lam_ckpt["config"]["model"]
    lam = LatentActionAutoencoder(
        state_dim=train_set.feature_dim,
        hidden_dim=lam_cfg["hidden_dim"],
        latent_dim=lam_cfg["latent_dim"],
        beta=lam_cfg["beta"],
        dropout=lam_cfg.get("dropout", 0.0),
    )
    lam.load_state_dict(lam_ckpt["model"])
    lam.to(device).eval()
    for p in lam.parameters():
        p.requires_grad = False

    predictor = ActionConditionedPredictor(
        state_dim=train_set.feature_dim,
        latent_dim=lam_cfg["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        context_len=cfg["data"]["dataset"]["context_len"],
        future_len=cfg["data"]["dataset"]["future_len"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    no_action = ActionAgnosticPredictor(
        state_dim=train_set.feature_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        context_len=cfg["data"]["dataset"]["context_len"],
        future_len=cfg["data"]["dataset"]["future_len"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    opt_pred = torch.optim.AdamW(predictor.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))
    opt_noact = torch.optim.AdamW(no_action.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))

    train_loader = _make_loader(train_set, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    val_loader = _make_loader(val_set, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))

    history = []
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"val_action_loss": float("inf")}
    for epoch in range(cfg["train"]["epochs"]):
        predictor.train()
        no_action.train()
        tr_act = tr_no = count = 0
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

            bs = batch["context"].shape[0]
            count += bs
            tr_act += float(loss_pred.detach()) * bs
            tr_no += float(loss_no.detach()) * bs

        predictor.eval()
        no_action.eval()
        va_act = va_no = count_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                z = lam.encode(batch["x_prev"], batch["x_next"])["mu"]
                pred = predictor(batch["context"], z)
                pred_no = no_action(batch["context"])
                loss_pred = F.mse_loss(pred, batch["target"])
                loss_no = F.mse_loss(pred_no, batch["target"])
                bs = batch["context"].shape[0]
                count_val += bs
                va_act += float(loss_pred.detach()) * bs
                va_no += float(loss_no.detach()) * bs

        row = {
            "epoch": epoch,
            "train_action_loss": tr_act / max(1, count),
            "train_no_action_loss": tr_no / max(1, count),
            "val_action_loss": va_act / max(1, count_val),
            "val_no_action_loss": va_no / max(1, count_val),
            "val_gain": (va_no - va_act) / max(1, count_val),
        }
        history.append(row)
        if row["val_action_loss"] < best["val_action_loss"]:
            best = row
            torch.save({"predictor": predictor.state_dict(), "no_action": no_action.state_dict(), "config": cfg, "best": best}, out_dir / "world_best.pt")
        torch.save({"predictor": predictor.state_dict(), "no_action": no_action.state_dict(), "config": cfg, "history": history}, out_dir / "world_last.pt")

    with open(out_dir / "world_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"best": best, "checkpoint": str(out_dir / "world_last.pt")}
