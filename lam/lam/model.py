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
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.experiment import finalize_run
from common.progress import ExperimentProgress
from common.wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_epoch
from lam.dataset import HumanMLTransitionDataset
from lam.modules import LatentActionModel

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
        predict_next_from_momentum: bool = False,
        window_len: int = 2,
        target_offset: int = 1,
        use_text_condition: bool = False,
        use_timestep_condition: bool = False,
        clip_version: str = "ViT-B/32",
        max_text_tokens: int = 32,
        max_timestep: int = 512,
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
        self.predict_next_from_momentum = predict_next_from_momentum
        self.window_len = window_len
        self.target_offset = target_offset
        self.use_text_condition = use_text_condition
        self.use_timestep_condition = use_timestep_condition
        self.lam = LatentActionModel(
            in_dim=joint_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
            use_text_condition=use_text_condition,
            use_timestep_condition=use_timestep_condition,
            clip_version=clip_version,
            max_text_tokens=max_text_tokens,
            max_timestep=max_timestep,
        )

    def state_dict_without_clip(self) -> Dict[str, torch.Tensor]:
        state_dict = self.state_dict()
        for key in [key for key in state_dict if key.startswith("lam.text_encoder.model.")]:
            del state_dict[key]
        return state_dict

    def load_state_dict_without_clip(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            raise ValueError(f"Unexpected keys when loading LAM: {unexpected_keys}")
        non_clip_missing = [key for key in missing_keys if not key.startswith("lam.text_encoder.model.")]
        if non_clip_missing:
            raise ValueError(f"Missing non-CLIP keys when loading LAM: {non_clip_missing}")

    def encode(
        self,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        texts: list[str] | None = None,
        timestep_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        joints = torch.stack([x_t, x_tp1], dim=1)
        encoded = self.lam.encode_sequence(joints, texts=texts, timestep_ids=timestep_ids)
        mu = encoded["z_mu"][:, 0]
        logvar = encoded["z_logvar"][:, 0]
        z = encoded["z_rep"][:, 0, 0]
        return {"z": z, "mu": mu, "logvar": logvar}

    def _legacy_timestep_ids(self, frame_idx: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, joints.shape[1], device=joints.device, dtype=torch.long)
        return frame_idx[:, None].to(device=joints.device, dtype=torch.long) + steps[None, :]

    def _momentum_timestep_ids(self, frame_idx: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return frame_idx.to(device=joints.device, dtype=torch.long) + self.window_len

    def forward_sequence(self, batch_or_joints: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(batch_or_joints):
            batch = {"joints": batch_or_joints}
        else:
            batch = batch_or_joints
        joints = batch["joints"]
        raw_text = batch.get("text")
        if isinstance(raw_text, str):
            texts = [raw_text]
        elif isinstance(raw_text, (list, tuple)):
            texts = list(raw_text)
        else:
            texts = raw_text
        if self.predict_next_from_momentum:
            if joints.shape[1] < self.window_len + self.target_offset:
                raise ValueError(
                    f"Momentum LAM requires at least {self.window_len + self.target_offset} frames, got {joints.shape[1]}"
                )
            encoder_joints = joints[:, : self.window_len]
            decoder_joints = joints[:, self.window_len - 1 : self.window_len]
            target = joints[:, self.window_len : self.window_len + self.target_offset]
            timestep_ids = None
            if self.use_timestep_condition and "frame_idx" in batch:
                timestep_ids = self._momentum_timestep_ids(batch["frame_idx"], joints)
            outputs = self.lam.forward_momentum(
                encoder_joints=encoder_joints,
                decoder_joints=decoder_joints,
                texts=texts,
                timestep_ids=timestep_ids,
            )
        else:
            timestep_ids = None
            if self.use_timestep_condition and "frame_idx" in batch:
                timestep_ids = self._legacy_timestep_ids(batch["frame_idx"], joints)
            outputs = self.lam({"joints": joints, "text": texts, "timestep_ids": timestep_ids})
            target = joints[:, 1:]
        mse_loss = F.mse_loss(outputs["recon"], target)
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

    def encode_action_sequence(
        self,
        joints: torch.Tensor,
        deterministic: bool = True,
        texts: list[str] | None = None,
        start_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.predict_next_from_momentum:
            timestep_ids = None
            if self.use_timestep_condition and start_timestep is not None:
                timestep_ids = start_timestep[:, None].to(device=joints.device, dtype=torch.long) + torch.arange(
                    1, joints.shape[1], device=joints.device, dtype=torch.long
                )[None, :]
            encoded = self.lam.encode_sequence(joints, texts=texts, timestep_ids=timestep_ids)
            return encoded["z_mu"] if deterministic else encoded["z_rep"].squeeze(2)

        if joints.shape[1] < self.window_len + self.target_offset:
            return torch.zeros(
                joints.shape[0],
                0,
                self.latent_dim,
                device=joints.device,
                dtype=joints.dtype,
            )
        prev = joints[:, :-2]
        curr = joints[:, 1:-1]
        batch_size, num_steps = prev.shape[:2]
        pair_inputs = torch.stack([prev, curr], dim=2).reshape(batch_size * num_steps, self.window_len, *joints.shape[2:])
        repeated_texts = None
        if texts is not None:
            repeated_texts = [text for text in texts for _ in range(num_steps)]
        timestep_ids = None
        if self.use_timestep_condition:
            if start_timestep is None:
                start_timestep = torch.zeros(batch_size, dtype=torch.long, device=joints.device)
            local_steps = torch.arange(self.window_len, self.window_len + num_steps, device=joints.device, dtype=torch.long)
            timestep_ids = start_timestep[:, None].to(device=joints.device, dtype=torch.long) + local_steps[None, :]
            timestep_ids = timestep_ids.reshape(batch_size * num_steps)
        encoded = self.lam.encode_sequence(pair_inputs, texts=repeated_texts, timestep_ids=timestep_ids)
        if deterministic:
            actions = encoded["z_mu"][:, 0]
        else:
            actions = encoded["z_rep"][:, 0, 0]
        return actions.reshape(batch_size, num_steps, self.latent_dim)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": shuffle,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _family_from_cfg(cfg: Dict) -> str:
    model_cfg = cfg.get("model", {})
    family = model_cfg.get("family", "joint_st_transformer")
    if family != "joint_st_transformer":
        raise ValueError(f"Unsupported LAM family: {family}. AdaMotion now supports only joint_st_transformer.")
    return family


def build_lam_from_config(cfg: Dict, dataset: HumanMLTransitionDataset | None = None) -> nn.Module:
    family = _family_from_cfg(cfg)
    model_cfg = cfg["model"]
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
        predict_next_from_momentum=model_cfg.get("predict_next_from_momentum", False),
        window_len=model_cfg.get("window_len", 2),
        target_offset=model_cfg.get("target_offset", 1),
        use_text_condition=model_cfg.get("use_text_condition", False),
        use_timestep_condition=model_cfg.get("use_timestep_condition", False),
        clip_version=model_cfg.get("clip_version", "ViT-B/32"),
        max_text_tokens=model_cfg.get("max_text_tokens", 32),
        max_timestep=model_cfg.get("max_timestep", 512),
    )


def load_lam_from_checkpoint(ckpt_path: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    family = _family_from_cfg(cfg)
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
        predict_next_from_momentum=cfg["model"].get("predict_next_from_momentum", False),
        window_len=cfg["model"].get("window_len", 2),
        target_offset=cfg["model"].get("target_offset", 1),
        use_text_condition=cfg["model"].get("use_text_condition", False),
        use_timestep_condition=cfg["model"].get("use_timestep_condition", False),
        clip_version=cfg["model"].get("clip_version", "ViT-B/32"),
        max_text_tokens=cfg["model"].get("max_text_tokens", 32),
        max_timestep=cfg["model"].get("max_timestep", 512),
    )
    model.load_state_dict_without_clip(ckpt["model"])
    return model


def train_lam_from_config(cfg: Dict) -> Dict:
    if cfg.get("stage") not in (None, "lam"):
        raise ValueError(f"LAM config expected stage=lam, got {cfg.get('stage')!r}")

    cfg = dict(cfg)
    cfg["stage"] = "lam"
    set_seed(cfg["seed"])

    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    use_amp = bool(cfg["train"].get("use_amp", device.type == "cuda"))
    amp_dtype_name = str(cfg["train"].get("amp_dtype", "float16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda" and amp_dtype == torch.float16)
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_memory = bool(cfg["train"].get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(cfg["train"].get("persistent_workers", num_workers > 0))
    prefetch_factor = cfg["train"].get("prefetch_factor")

    train_set = HumanMLTransitionDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_args["random_caption"] = False
    val_set = HumanMLTransitionDataset(**val_args)

    model = build_lam_from_config(cfg, train_set).to(device)
    cfg.setdefault("model", {})
    cfg["model"]["num_joints"] = model.num_joints
    cfg["model"]["joint_dim"] = model.joint_dim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    train_loader = _make_loader(
        train_set,
        cfg["train"]["batch_size"],
        True,
        num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _make_loader(
        val_set,
        cfg["train"]["batch_size"],
        False,
        num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    history = []
    best = {"val_loss": float("inf")}
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lam_last.pt"
    wandb_run = init_wandb_run(REPO_ROOT, "lam", cfg)
    progress = ExperimentProgress(
        stage="lam",
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
            train_metrics = {"loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0, "count": 0}
            progress.start_phase("train", len(train_loader))
            for batch in train_loader:
                batch = _to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                    outputs = model.forward_sequence(batch)
                batch_size = batch["joints"].shape[0]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(outputs["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
                train_metrics["count"] += batch_size
                for key in ["loss", "mse_loss", "kl_loss"]:
                    train_metrics[key] += float(outputs[key].detach()) * batch_size
                denom = max(1, train_metrics["count"])
                progress.update(
                    "train",
                    {
                        "loss": train_metrics["loss"] / denom,
                        "mse": train_metrics["mse_loss"] / denom,
                        "kl": train_metrics["kl_loss"] / denom,
                    },
                )
            progress.end_phase()

            model.eval()
            val_metrics = {"loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0, "count": 0}
            with torch.no_grad():
                progress.start_phase("val", len(val_loader))
                for batch in val_loader:
                    batch = _to_device(batch, device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                        outputs = model.forward_sequence(batch)
                    batch_size = batch["joints"].shape[0]
                    val_metrics["count"] += batch_size
                    for key in ["loss", "mse_loss", "kl_loss"]:
                        val_metrics[key] += float(outputs[key].detach()) * batch_size
                    denom = max(1, val_metrics["count"])
                    progress.update(
                        "val",
                        {
                            "loss": val_metrics["loss"] / denom,
                            "mse": val_metrics["mse_loss"] / denom,
                            "kl": val_metrics["kl_loss"] / denom,
                        },
                    )
                progress.end_phase()

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"] / max(1, train_metrics["count"]),
                "train_mse": train_metrics["mse_loss"] / max(1, train_metrics["count"]),
                "train_kl": train_metrics["kl_loss"] / max(1, train_metrics["count"]),
                "val_loss": val_metrics["loss"] / max(1, val_metrics["count"]),
                "val_mse": val_metrics["mse_loss"] / max(1, val_metrics["count"]),
                "val_kl": val_metrics["kl_loss"] / max(1, val_metrics["count"]),
            }
            print(
                f"[lam][epoch {epoch + 1}/{cfg['train']['epochs']}] "
                f"train_loss={row['train_loss']:.6f} train_mse={row['train_mse']:.6f} train_kl={row['train_kl']:.6f} "
                f"val_loss={row['val_loss']:.6f} val_mse={row['val_mse']:.6f} val_kl={row['val_kl']:.6f}",
                flush=True,
            )
            history.append(row)
            log_wandb_epoch(
                wandb_run,
                row,
                extra={
                    "stage": "lam",
                    "output_dir": str(out_dir),
                    "best_val_loss_so_far": min(item["val_loss"] for item in history),
                },
            )
            save_payload = {"model": model.state_dict_without_clip(), "config": cfg}
            if row["val_loss"] < best["val_loss"]:
                best = row
                torch.save({**save_payload, "best": best}, out_dir / "lam_best.pt")
            torch.save({**save_payload, "history": history}, ckpt_path)
    finally:
        progress.close()

    with open(out_dir / "lam_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = finalize_run(
        repo_root=Path(__file__).resolve().parents[2],
        stage="lam",
        cfg=cfg,
        history=history,
        checkpoint_path=str(ckpt_path),
        best_metrics=best,
        extra_summary={"family": model.family},
    )
    
    finish_wandb_run(wandb_run, summary)
    return {"best": best, "checkpoint": str(ckpt_path)}


LAM = JointSpatioTemporalLAM

__all__ = [
    "JointSpatioTemporalLAM",
    "LAM",
    "build_lam_from_config",
    "load_lam_from_checkpoint",
    "set_seed",
    "train_lam_from_config",
]
