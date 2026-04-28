from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SALAD_ROOT = REPO_ROOT.parent / "humanmodels" / "salad"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SALAD_ROOT))

from common.experiment import finalize_run
from common.progress import ExperimentProgress
from common.wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_epoch
from motionvae.dataset import HumanMLMotionAutoencoderDataset
from models.vae.model import VAE as SaladVAE


def _make_salad_opt(cfg: Dict) -> SimpleNamespace:
    model_cfg = cfg["model"]
    return SimpleNamespace(
        dataset_name="t2m",
        pose_dim=263,
        joints_num=22,
        contact_joints=[7, 10, 8, 11],
        latent_dim=model_cfg["latent_dim"],
        kernel_size=model_cfg.get("kernel_size", 3),
        n_layers=model_cfg.get("n_layers", 2),
        n_extra_layers=model_cfg.get("n_extra_layers", 1),
        norm=model_cfg.get("norm", "none"),
        activation=model_cfg.get("activation", "gelu"),
        dropout=model_cfg.get("dropout", 0.1),
    )


class MotionVAE(nn.Module):
    family = "salad_motion_vae"

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.opt = _make_salad_opt(cfg)
        self.model = SaladVAE(self.opt)
        self.latent_dim = self.opt.latent_dim
        self.pose_dim = self.opt.pose_dim
        self.lambda_kl = cfg["train"].get("lambda_kl", 0.02)
        self.lambda_vel = cfg["train"].get("lambda_vel", 0.5)
        self.lambda_pos = cfg["train"].get("lambda_pos", 0.5)

    def freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, motion: torch.Tensor):
        return self.model.encode(motion)

    def encode_stats(self, motion: torch.Tensor):
        x = self.model.motion_enc(motion)
        x = self.model.conv_enc(x)
        moments = self.model.dist(x)
        mu, logvar = moments.chunk(2, dim=-1)
        return mu, logvar

    def encode_deterministic(self, motion: torch.Tensor):
        mu, logvar = self.encode_stats(motion)
        return mu, {"loss_kl": 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)}

    def decode(self, latent: torch.Tensor):
        return self.model.decode(latent)

    def forward(self, motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon, latent_logs = self.model.forward(motion)
        _, ric, _, vel, _ = torch.split(motion, [4, 63, 126, 66, 4], dim=-1)
        _, pred_ric, _, pred_vel, _ = torch.split(recon, [4, 63, 126, 66, 4], dim=-1)
        loss_recon = F.smooth_l1_loss(recon, motion)
        loss_vel = F.smooth_l1_loss(pred_vel, vel)
        loss_pos = F.smooth_l1_loss(pred_ric, ric)
        loss = (
            loss_recon
            + self.lambda_vel * loss_vel
            + self.lambda_pos * loss_pos
            + self.lambda_kl * latent_logs["loss_kl"]
        )
        return {
            "recon": recon,
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_vel": loss_vel,
            "loss_pos": loss_pos,
            "loss_kl": latent_logs["loss_kl"],
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


def load_motion_vae_from_checkpoint(ckpt_path: str) -> MotionVAE:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MotionVAE(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model


def train_motion_vae_from_config(cfg: Dict) -> Dict:
    if cfg.get("stage") not in (None, "motion_vae"):
        raise ValueError(f"Motion-VAE config expected stage=motion_vae, got {cfg.get('stage')!r}")

    cfg = dict(cfg)
    cfg["stage"] = "motion_vae"
    set_seed(cfg["seed"])

    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    train_set = HumanMLMotionAutoencoderDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_set = HumanMLMotionAutoencoderDataset(**val_args)

    model = MotionVAE(cfg).to(device)
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
    ckpt_path = out_dir / "motion_vae_last.pt"
    wandb_run = init_wandb_run(REPO_ROOT, "motion_vae", cfg)
    progress = ExperimentProgress(
        stage="motion_vae",
        epochs=cfg["train"]["epochs"],
        train_steps=len(train_loader),
        val_steps=len(val_loader),
        run=wandb_run,
        log_interval=cfg["train"].get("progress_log_interval", 20),
    )

    try:
        for epoch in range(cfg["train"]["epochs"]):
            progress.start_epoch(epoch)
            model.train()
            train_metrics = {"loss": 0.0, "loss_recon": 0.0, "loss_vel": 0.0, "loss_pos": 0.0, "loss_kl": 0.0, "count": 0}
            progress.start_phase("train", len(train_loader))
            for batch in train_loader:
                batch = _to_device(batch, device)
                outputs = model(batch["motion"])
                optimizer.zero_grad(set_to_none=True)
                outputs["loss"].backward()
                optimizer.step()
                batch_size = batch["motion"].shape[0]
                train_metrics["count"] += batch_size
                for key in ["loss", "loss_recon", "loss_vel", "loss_pos", "loss_kl"]:
                    train_metrics[key] += float(outputs[key].detach()) * batch_size
                denom = max(1, train_metrics["count"])
                progress.update(
                    "train",
                    {
                        "loss": train_metrics["loss"] / denom,
                        "recon": train_metrics["loss_recon"] / denom,
                        "kl": train_metrics["loss_kl"] / denom,
                    },
                )
            progress.end_phase()

            model.eval()
            val_metrics = {"loss": 0.0, "loss_recon": 0.0, "loss_vel": 0.0, "loss_pos": 0.0, "loss_kl": 0.0, "count": 0}
            with torch.no_grad():
                progress.start_phase("val", len(val_loader))
                for batch in val_loader:
                    batch = _to_device(batch, device)
                    outputs = model(batch["motion"])
                    batch_size = batch["motion"].shape[0]
                    val_metrics["count"] += batch_size
                    for key in ["loss", "loss_recon", "loss_vel", "loss_pos", "loss_kl"]:
                        val_metrics[key] += float(outputs[key].detach()) * batch_size
                    denom = max(1, val_metrics["count"])
                    progress.update(
                        "val",
                        {
                            "loss": val_metrics["loss"] / denom,
                            "recon": val_metrics["loss_recon"] / denom,
                            "kl": val_metrics["loss_kl"] / denom,
                        },
                    )
                progress.end_phase()

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"] / max(1, train_metrics["count"]),
                "train_recon": train_metrics["loss_recon"] / max(1, train_metrics["count"]),
                "train_vel": train_metrics["loss_vel"] / max(1, train_metrics["count"]),
                "train_pos": train_metrics["loss_pos"] / max(1, train_metrics["count"]),
                "train_kl": train_metrics["loss_kl"] / max(1, train_metrics["count"]),
                "val_loss": val_metrics["loss"] / max(1, val_metrics["count"]),
                "val_recon": val_metrics["loss_recon"] / max(1, val_metrics["count"]),
                "val_vel": val_metrics["loss_vel"] / max(1, val_metrics["count"]),
                "val_pos": val_metrics["loss_pos"] / max(1, val_metrics["count"]),
                "val_kl": val_metrics["loss_kl"] / max(1, val_metrics["count"]),
            }
            print(
                f"[motion_vae][epoch {epoch + 1}/{cfg['train']['epochs']}] "
                f"train_loss={row['train_loss']:.6f} train_recon={row['train_recon']:.6f} train_kl={row['train_kl']:.6f} "
                f"val_loss={row['val_loss']:.6f} val_recon={row['val_recon']:.6f} val_kl={row['val_kl']:.6f}",
                flush=True,
            )
            history.append(row)
            log_wandb_epoch(
                wandb_run,
                row,
                extra={
                    "stage": "motion_vae",
                    "output_dir": str(out_dir),
                    "best_val_loss_so_far": min(item["val_loss"] for item in history),
                },
            )
            save_payload = {"model": model.state_dict(), "config": cfg, "history": history}
            if row["val_loss"] < best["val_loss"]:
                best = row
                torch.save({**save_payload, "best": best}, out_dir / "motion_vae_best.pt")
            torch.save(save_payload, ckpt_path)
    finally:
        progress.close()

    with open(out_dir / "motion_vae_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    summary = finalize_run(
        repo_root=REPO_ROOT,
        stage="motion_vae",
        cfg=cfg,
        history=history,
        checkpoint_path=str(ckpt_path),
        best_metrics=best,
        extra_summary={"family": model.family},
    )
    finish_wandb_run(wandb_run, summary)
    return {"best": best, "checkpoint": str(ckpt_path)}
