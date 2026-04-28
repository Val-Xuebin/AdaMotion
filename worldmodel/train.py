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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from diffusers import DDIMScheduler
except Exception:
    class DDIMScheduler:  # pragma: no cover
        def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.00085,
            beta_end: float = 0.012,
            beta_schedule: str = "scaled_linear",
            prediction_type: str = "v_prediction",
            clip_sample: bool = False,
        ) -> None:
            self.num_train_timesteps = num_train_timesteps
            self.prediction_type = prediction_type
            if beta_schedule == "scaled_linear":
                betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            else:
                betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.init_noise_sigma = 1.0
            self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)

        def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            alphas = self.alphas_cumprod.to(original_samples.device)[timesteps].view(-1, 1, 1, 1)
            return alphas.sqrt() * original_samples + (1 - alphas).sqrt() * noise

        def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            alphas = self.alphas_cumprod.to(sample.device)[timesteps].view(-1, 1, 1, 1)
            return alphas.sqrt() * noise - (1 - alphas).sqrt() * sample


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lam"))

from common.experiment import finalize_run
from common.progress import ExperimentProgress
from common.wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_epoch
from lam.model import load_lam_from_checkpoint
from vwm.data.full_motion_action_dataset import HumanMLFullMotionActionDataset
from vwm.models.action_prior import TextLengthActionPrior
from vwm.models.salad_official import (
    load_official_salad_action_denoiser,
    load_official_salad_vae,
)


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


def _generator_type_from_cfg(cfg: Dict) -> str:
    generator_type = cfg.get("model", {}).get("generator_type", "official_salad_action_adapter")
    if generator_type != "official_salad_action_adapter":
        raise ValueError(f"Unsupported world-model generator_type: {generator_type}")
    return generator_type


def _pool_action_sequence(action_seq: torch.Tensor, target_steps: int) -> torch.Tensor:
    if action_seq.shape[1] == target_steps:
        return action_seq
    if action_seq.shape[1] == 0:
        return torch.zeros(
            action_seq.shape[0],
            target_steps,
            action_seq.shape[-1],
            device=action_seq.device,
            dtype=action_seq.dtype,
        )
    if action_seq.shape[1] % target_steps == 0:
        factor = action_seq.shape[1] // target_steps
        return action_seq.reshape(action_seq.shape[0], target_steps, factor, action_seq.shape[-1]).mean(dim=2)
    return F.adaptive_avg_pool1d(action_seq.transpose(1, 2), target_steps).transpose(1, 2)


def _lengths_to_latent_mask(lengths: torch.Tensor, latent_steps: int, unit_length: int = 4) -> torch.Tensor:
    latent_lengths = torch.div(lengths, unit_length, rounding_mode="floor").clamp(min=1, max=latent_steps)
    return torch.arange(latent_steps, device=lengths.device).unsqueeze(0) < latent_lengths.unsqueeze(1)


def _build_noise_scheduler(cfg: Dict) -> DDIMScheduler:
    model_cfg = cfg["model"]
    return DDIMScheduler(
        num_train_timesteps=model_cfg.get("num_train_timesteps", 1000),
        beta_start=model_cfg.get("beta_start", 0.00085),
        beta_end=model_cfg.get("beta_end", 0.012),
        beta_schedule=model_cfg.get("beta_schedule", "scaled_linear"),
        prediction_type=model_cfg.get("prediction_type", "v_prediction"),
        clip_sample=False,
    )


def _maybe_drop_text_condition(texts: list[str], drop_prob: float) -> list[str]:
    if drop_prob <= 0:
        return texts
    return ["" if random.random() < drop_prob else text for text in texts]


def build_action_prior_from_config(cfg: Dict, lam, device: torch.device):
    model_cfg = cfg["model"]
    dataset_cfg = cfg["data"]["dataset"]
    if model_cfg.get("prior_type") != "text_length_action":
        raise ValueError(f"Unsupported prior_type: {model_cfg.get('prior_type')}")
    return TextLengthActionPrior(
        action_dim=lam.latent_dim,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        latent_steps=model_cfg.get(
            "latent_steps",
            dataset_cfg.get("max_motion_length", 196) // dataset_cfg.get("unit_length", 4),
        ),
        dropout=model_cfg.get("dropout", 0.1),
        clip_version=model_cfg.get("clip_version", "ViT-B/32"),
        max_text_tokens=model_cfg.get("max_text_tokens", 32),
        max_motion_length=dataset_cfg.get("max_motion_length", 196),
    ).to(device)


def _normalize_legacy_paths(value):
    if isinstance(value, dict):
        return {key: _normalize_legacy_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_legacy_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace("/work/adamotion/data/HumanML3D", "/workspace/assets/dataset/HumanML3D").replace(
            "/work/adamotion", "/workspace/AdaMotion"
        )
    return value


def load_action_prior_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = _normalize_legacy_paths(ckpt["config"])
    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    prior = build_action_prior_from_config(cfg, lam, torch.device("cpu"))
    prior.load_state_dict_without_clip(ckpt["action_prior"])
    return prior, cfg


def load_official_action_adapter_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = _normalize_legacy_paths(ckpt["config"])
    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    vae = load_official_salad_vae(
        cfg["model"]["official_salad_vae_opt"],
        cfg["model"]["official_salad_vae_checkpoint"],
        torch.device("cpu"),
    )
    denoiser = load_official_salad_action_denoiser(
        cfg["model"]["official_salad_denoiser_opt"],
        cfg["model"]["official_salad_denoiser_checkpoint"],
        vae_dim=vae.latent_dim,
        action_dim=lam.latent_dim,
        device=torch.device("cpu"),
        train_base=cfg["model"].get("train_base_denoiser", False),
    )
    denoiser.load_state_dict_without_clip(ckpt["denoiser"])
    return denoiser, vae, lam, cfg


def train_official_salad_action_adapter_from_config(cfg: Dict) -> Dict:
    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    train_set = HumanMLFullMotionActionDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val_usable")
    val_args["random_caption"] = False
    val_set = HumanMLFullMotionActionDataset(**val_args)

    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"]).to(device).eval()
    for param in lam.parameters():
        param.requires_grad = False
    official_vae = load_official_salad_vae(
        cfg["model"]["official_salad_vae_opt"],
        cfg["model"]["official_salad_vae_checkpoint"],
        device,
    )
    denoiser = load_official_salad_action_denoiser(
        cfg["model"]["official_salad_denoiser_opt"],
        cfg["model"]["official_salad_denoiser_checkpoint"],
        vae_dim=official_vae.latent_dim,
        action_dim=lam.latent_dim,
        device=device,
        train_base=cfg["model"].get("train_base_denoiser", False),
    )
    optimizer = torch.optim.AdamW(
        denoiser.parameters_without_clip(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = _build_noise_scheduler(cfg)
    prediction_type = cfg["model"].get("prediction_type", "v_prediction")
    unit_length = cfg["data"]["dataset"].get("unit_length", 4)

    train_loader = _make_loader(train_set, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    val_loader = _make_loader(val_set, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "world_last.pt"
    history = []
    best = {"val_loss": float("inf")}
    wandb_run = init_wandb_run(REPO_ROOT, "world_model", cfg)
    progress = ExperimentProgress(
        stage="world_model_official_adapter",
        epochs=cfg["train"]["epochs"],
        train_steps=len(train_loader),
        val_steps=len(val_loader),
        run=wandb_run,
        log_interval=cfg["train"].get("progress_log_interval", 20),
    )

    def compute_step(batch: Dict) -> Dict[str, torch.Tensor]:
        batch = _to_device(batch, device)
        with torch.no_grad():
            z_motion = official_vae.encode_deterministic(batch["motion"])[0]
            len_mask = _lengths_to_latent_mask(batch["length"], z_motion.shape[1], unit_length=unit_length)
            z_motion = z_motion * len_mask[..., None, None].float()
            action_seq = lam.encode_action_sequence(
                batch["action_source"],
                texts=list(batch["text"]),
                start_timestep=batch.get("start_idx"),
            )
            action_seq = _pool_action_sequence(action_seq, z_motion.shape[1])
            action_seq = action_seq * len_mask[..., None].float()
        timesteps = torch.randint(
            0,
            cfg["model"].get("num_train_timesteps", 1000),
            (z_motion.shape[0],),
            device=device,
        ).long()
        noise = torch.randn_like(z_motion) * len_mask[..., None, None].float()
        noisy_motion = scheduler.add_noise(z_motion, noise, timesteps)
        pred = denoiser(
            noisy_motion_latent=noisy_motion,
            timesteps=timesteps,
            texts=_maybe_drop_text_condition(list(batch["text"]), cfg["model"].get("cond_drop_prob", 0.0)),
            action_latent_seq=action_seq,
            len_mask=len_mask,
        )
        if prediction_type == "sample":
            target = z_motion
        elif prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = scheduler.get_velocity(z_motion, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        pred = pred * len_mask[..., None, None].float()
        target = target * len_mask[..., None, None].float()
        loss = F.mse_loss(pred, target)
        return {"loss": loss, "latent_mse": F.mse_loss(noisy_motion, z_motion)}

    try:
        for epoch in range(cfg["train"]["epochs"]):
            denoiser.train()
            train_metrics = {"loss": 0.0, "latent_mse": 0.0, "count": 0}
            progress.start_epoch(epoch)
            progress.start_phase("train", len(train_loader))
            for batch in train_loader:
                outputs = compute_step(batch)
                optimizer.zero_grad(set_to_none=True)
                outputs["loss"].backward()
                optimizer.step()
                batch_size = len(batch["text"])
                train_metrics["count"] += batch_size
                for key in ["loss", "latent_mse"]:
                    train_metrics[key] += float(outputs[key].detach()) * batch_size
                denom = max(1, train_metrics["count"])
                progress.update("train", {"loss": train_metrics["loss"] / denom, "latent": train_metrics["latent_mse"] / denom})
            progress.end_phase()

            denoiser.eval()
            val_metrics = {"loss": 0.0, "latent_mse": 0.0, "count": 0}
            with torch.no_grad():
                progress.start_phase("val", len(val_loader))
                for batch in val_loader:
                    outputs = compute_step(batch)
                    batch_size = len(batch["text"])
                    val_metrics["count"] += batch_size
                    for key in ["loss", "latent_mse"]:
                        val_metrics[key] += float(outputs[key].detach()) * batch_size
                    denom = max(1, val_metrics["count"])
                    progress.update("val", {"loss": val_metrics["loss"] / denom, "latent": val_metrics["latent_mse"] / denom})
                progress.end_phase()
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"] / max(1, train_metrics["count"]),
                "train_latent_mse": train_metrics["latent_mse"] / max(1, train_metrics["count"]),
                "val_loss": val_metrics["loss"] / max(1, val_metrics["count"]),
                "val_latent_mse": val_metrics["latent_mse"] / max(1, val_metrics["count"]),
            }
            print(
                f"[world official adapter][epoch {epoch + 1}/{cfg['train']['epochs']}] "
                f"train_loss={row['train_loss']:.6f} train_latent={row['train_latent_mse']:.6f} "
                f"val_loss={row['val_loss']:.6f} val_latent={row['val_latent_mse']:.6f}",
                flush=True,
            )
            history.append(row)
            log_wandb_epoch(
                wandb_run,
                row,
                extra={
                    "stage": "world_model",
                    "generator_type": "official_salad_action_adapter",
                    "output_dir": str(out_dir),
                    "best_val_loss_so_far": min(item["val_loss"] for item in history),
                },
            )
            save_payload = {"denoiser": denoiser.state_dict_without_clip(), "config": cfg, "history": history}
            if row["val_loss"] < best["val_loss"]:
                best = row
                torch.save({**save_payload, "best": best}, out_dir / "world_best.pt")
            torch.save(save_payload, ckpt_path)
    finally:
        progress.close()

    with open(out_dir / "world_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    summary = finalize_run(
        repo_root=REPO_ROOT,
        stage="world_model",
        cfg=cfg,
        history=history,
        checkpoint_path=str(ckpt_path),
        best_metrics=best,
        extra_summary={
            "family": "official_salad_action_adapter",
            "generator_type": "official_salad_action_adapter",
            "official_salad_vae_checkpoint": cfg["model"]["official_salad_vae_checkpoint"],
            "official_salad_denoiser_checkpoint": cfg["model"]["official_salad_denoiser_checkpoint"],
        },
    )
    finish_wandb_run(wandb_run, summary)
    return {"best": best, "checkpoint": str(ckpt_path)}


def train_action_prior_from_config(cfg: Dict) -> Dict:
    if cfg.get("stage") not in (None, "action_prior"):
        raise ValueError(f"Action-prior config expected stage=action_prior, got {cfg.get('stage')!r}")
    if cfg["model"].get("prior_type") != "text_length_action":
        raise ValueError(f"Unsupported prior_type: {cfg['model'].get('prior_type')}")
    cfg = dict(cfg)
    cfg["stage"] = "action_prior"
    set_seed(cfg["seed"])

    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    train_set = HumanMLFullMotionActionDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val_usable")
    val_args["random_caption"] = False
    val_set = HumanMLFullMotionActionDataset(**val_args)

    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"]).to(device).eval()
    for param in lam.parameters():
        param.requires_grad = False
    prior = build_action_prior_from_config(cfg, lam, device)
    optimizer = torch.optim.AdamW(
        prior.parameters_without_clip(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    train_loader = _make_loader(train_set, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    val_loader = _make_loader(val_set, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "action_prior_last.pt"
    history = []
    best = {"val_loss": float("inf")}
    wandb_run = init_wandb_run(REPO_ROOT, "action_prior", cfg)
    progress = ExperimentProgress(
        stage="action_prior",
        epochs=cfg["train"]["epochs"],
        train_steps=len(train_loader),
        val_steps=len(val_loader),
        run=wandb_run,
        log_interval=cfg["train"].get("progress_log_interval", 20),
    )

    def compute_step(batch: Dict) -> Dict[str, torch.Tensor]:
        batch = _to_device(batch, device)
        with torch.no_grad():
            target_action = lam.encode_action_sequence(
                batch["action_source"],
                texts=list(batch["text"]),
                start_timestep=batch.get("start_idx"),
            )
            target_action = _pool_action_sequence(target_action, prior.latent_steps)
            len_mask = _lengths_to_latent_mask(
                batch["length"],
                prior.latent_steps,
                unit_length=cfg["data"]["dataset"].get("unit_length", 4),
            )
            target_action = target_action * len_mask[..., None].float()
        pred_action = prior(list(batch["text"]), batch["length"])
        pred_action = pred_action * len_mask[..., None].float()
        loss = F.mse_loss(pred_action, target_action)
        return {"loss": loss, "target_std": target_action.std(), "pred_std": pred_action.std()}

    try:
        for epoch in range(cfg["train"]["epochs"]):
            prior.train()
            train_metrics = {"loss": 0.0, "target_std": 0.0, "pred_std": 0.0, "count": 0}
            progress.start_epoch(epoch)
            progress.start_phase("train", len(train_loader))
            for batch in train_loader:
                outputs = compute_step(batch)
                optimizer.zero_grad(set_to_none=True)
                outputs["loss"].backward()
                optimizer.step()
                batch_size = len(batch["text"])
                train_metrics["count"] += batch_size
                for key in ["loss", "target_std", "pred_std"]:
                    train_metrics[key] += float(outputs[key].detach()) * batch_size
                denom = max(1, train_metrics["count"])
                progress.update(
                    "train",
                    {
                        "loss": train_metrics["loss"] / denom,
                        "target_std": train_metrics["target_std"] / denom,
                        "pred_std": train_metrics["pred_std"] / denom,
                    },
                )
            progress.end_phase()

            prior.eval()
            val_metrics = {"loss": 0.0, "target_std": 0.0, "pred_std": 0.0, "count": 0}
            with torch.no_grad():
                progress.start_phase("val", len(val_loader))
                for batch in val_loader:
                    outputs = compute_step(batch)
                    batch_size = len(batch["text"])
                    val_metrics["count"] += batch_size
                    for key in ["loss", "target_std", "pred_std"]:
                        val_metrics[key] += float(outputs[key].detach()) * batch_size
                    denom = max(1, val_metrics["count"])
                    progress.update(
                        "val",
                        {
                            "loss": val_metrics["loss"] / denom,
                            "target_std": val_metrics["target_std"] / denom,
                            "pred_std": val_metrics["pred_std"] / denom,
                        },
                    )
                progress.end_phase()

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"] / max(1, train_metrics["count"]),
                "train_target_std": train_metrics["target_std"] / max(1, train_metrics["count"]),
                "train_pred_std": train_metrics["pred_std"] / max(1, train_metrics["count"]),
                "val_loss": val_metrics["loss"] / max(1, val_metrics["count"]),
                "val_target_std": val_metrics["target_std"] / max(1, val_metrics["count"]),
                "val_pred_std": val_metrics["pred_std"] / max(1, val_metrics["count"]),
            }
            print(
                f"[action_prior][epoch {epoch + 1}/{cfg['train']['epochs']}] "
                f"train_loss={row['train_loss']:.6f} train_target_std={row['train_target_std']:.6f} "
                f"train_pred_std={row['train_pred_std']:.6f} val_loss={row['val_loss']:.6f} "
                f"val_target_std={row['val_target_std']:.6f} val_pred_std={row['val_pred_std']:.6f}",
                flush=True,
            )
            history.append(row)
            log_wandb_epoch(
                wandb_run,
                row,
                extra={
                    "stage": "action_prior",
                    "output_dir": str(out_dir),
                    "best_val_loss_so_far": min(item["val_loss"] for item in history),
                },
            )
            save_payload = {"action_prior": prior.state_dict_without_clip(), "config": cfg, "history": history}
            if row["val_loss"] < best["val_loss"]:
                best = row
                torch.save({**save_payload, "best": best}, out_dir / "action_prior_best.pt")
            torch.save(save_payload, ckpt_path)
    finally:
        progress.close()

    with open(out_dir / "action_prior_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    summary = finalize_run(
        repo_root=REPO_ROOT,
        stage="action_prior",
        cfg=cfg,
        history=history,
        checkpoint_path=str(ckpt_path),
        best_metrics=best,
        extra_summary={"family": prior.family},
    )
    finish_wandb_run(wandb_run, summary)
    return {"best": best, "checkpoint": str(ckpt_path)}


def train_world_model_from_config(cfg: Dict) -> Dict:
    if cfg.get("stage") not in (None, "world_model"):
        raise ValueError(f"World-model config expected stage=world_model, got {cfg.get('stage')!r}")
    cfg = dict(cfg)
    cfg["stage"] = "world_model"
    set_seed(cfg["seed"])
    _generator_type_from_cfg(cfg)
    return train_official_salad_action_adapter_from_config(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AdaMotion world model with an AdaWorld-style entrypoint.")
    parser.add_argument("--base", nargs="+", required=True, help="Compatible with AdaWorld's --base config argument.")
    args = parser.parse_args()
    if len(args.base) != 1:
        raise ValueError("AdaMotion expects exactly one config path in --base.")
    cfg = load_config(args.base[0])
    result = train_world_model_from_config(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
