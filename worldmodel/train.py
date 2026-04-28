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
from tqdm.auto import tqdm

try:
    from diffusers import DDIMScheduler
except Exception:
    class DDIMScheduler:  # pragma: no cover - lightweight fallback for environments without diffusers
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

        def set_timesteps(self, num_inference_timesteps: int) -> None:
            self.timesteps = torch.linspace(
                self.num_train_timesteps - 1,
                0,
                num_inference_timesteps,
                dtype=torch.long,
            )

        def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            alphas = self.alphas_cumprod.to(original_samples.device)[timesteps].view(-1, 1, 1, 1)
            return alphas.sqrt() * original_samples + (1 - alphas).sqrt() * noise

        def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            alphas = self.alphas_cumprod.to(sample.device)[timesteps].view(-1, 1, 1, 1)
            return alphas.sqrt() * noise - (1 - alphas).sqrt() * sample

        class _StepOutput:
            def __init__(self, prev_sample: torch.Tensor) -> None:
                self.prev_sample = prev_sample

        def step(self, model_output: torch.Tensor, timestep: torch.Tensor | int, sample: torch.Tensor) -> "_StepOutput":
            return self._StepOutput(sample - model_output)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lam"))
sys.path.insert(0, str(REPO_ROOT / "motionvae"))

from common.experiment import finalize_run
from common.progress import ExperimentProgress
from common.humanml_representation import humanml_vector_to_sal_rep
from common.wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_epoch
from lam.model import load_lam_from_checkpoint
from motionvae.model import load_motion_vae_from_checkpoint
from vwm.data.dataset import HumanMLContextDataset
from vwm.data.full_motion_action_dataset import HumanMLFullMotionActionDataset
from vwm.data.text_future_dataset import HumanMLTextFutureDataset
from vwm.models.action_prior import ActionLatentPrior, TextLengthActionPrior
from vwm.models.motion_diffusion import ActionConditionedMotionDenoiser
from vwm.models.salad_official import (
    OfficialSaladActionDenoiser,
    load_official_salad_action_denoiser,
    load_official_salad_vae,
)


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


def _generator_type_from_cfg(cfg: Dict) -> str:
    return cfg.get("model", {}).get("generator_type", "legacy_predictor")


def _family_from_cfg(cfg: Dict) -> str:
    generator_type = _generator_type_from_cfg(cfg)
    if generator_type == "legacy_predictor":
        family = cfg.get("model", {}).get("family", "joint_st_transformer")
        if family != "joint_st_transformer":
            raise ValueError(
                f"Unsupported world-model family: {family}. AdaMotion supports joint_st_transformer predictors only."
            )
        return family
    if generator_type == "motion_diffusion":
        return "salad_motion_diffusion"
    if generator_type == "official_salad_action_adapter":
        return "official_salad_action_adapter"
    raise ValueError(f"Unsupported world-model generator_type: {generator_type}")


def build_world_models(cfg: Dict, train_set: HumanMLContextDataset, lam: nn.Module):
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
    return predictor, no_action


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
    pooled = F.adaptive_avg_pool1d(action_seq.transpose(1, 2), target_steps).transpose(1, 2)
    return pooled


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


def build_motion_diffusion_from_config(cfg: Dict, lam, motion_vae, device: torch.device):
    model = ActionConditionedMotionDenoiser(
        latent_dim=cfg["model"].get("latent_dim", cfg["model"].get("hidden_dim", 256)),
        vae_latent_dim=motion_vae.latent_dim,
        action_dim=lam.latent_dim,
        motion_joints=cfg["model"].get("motion_latent_joints", 7),
        n_heads=cfg["model"].get("n_heads", 8),
        n_layers=cfg["model"].get("n_layers", 5),
        ff_dim=cfg["model"].get("ff_dim", 1024),
        dropout=cfg["model"].get("dropout", 0.1),
        activation=cfg["model"].get("activation", "gelu"),
        clip_version=cfg["model"].get("clip_version", "ViT-B/32"),
        max_text_tokens=cfg["model"].get("max_text_tokens", 32),
    ).to(device)
    return model


def build_action_prior_from_config(cfg: Dict, lam, motion_vae, device: torch.device):
    model_cfg = cfg["model"]
    dataset_cfg = cfg["data"]["dataset"]
    if model_cfg.get("prior_type") == "text_length_action":
        return TextLengthActionPrior(
            action_dim=lam.latent_dim,
            hidden_dim=model_cfg.get("hidden_dim", 256),
            latent_steps=model_cfg.get("latent_steps", dataset_cfg.get("max_motion_length", 196) // dataset_cfg.get("unit_length", 4)),
            dropout=model_cfg.get("dropout", 0.1),
            clip_version=model_cfg.get("clip_version", "ViT-B/32"),
            max_text_tokens=model_cfg.get("max_text_tokens", 32),
            max_motion_length=dataset_cfg.get("max_motion_length", 196),
        ).to(device)
    return ActionLatentPrior(
        action_dim=lam.latent_dim,
        vae_latent_dim=motion_vae.latent_dim,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        future_len=dataset_cfg.get("future_len", model_cfg.get("future_len", 8)),
        dropout=model_cfg.get("dropout", 0.1),
        clip_version=model_cfg.get("clip_version", "ViT-B/32"),
        max_text_tokens=model_cfg.get("max_text_tokens", 32),
    ).to(device)


def load_motion_diffusion_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = _normalize_legacy_paths(ckpt["config"])
    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    motion_vae = load_motion_vae_from_checkpoint(cfg["model"]["motion_vae_checkpoint"])
    denoiser = build_motion_diffusion_from_config(cfg, lam, motion_vae, torch.device("cpu"))
    denoiser.load_state_dict_without_clip(ckpt["denoiser"])
    return denoiser, cfg


def load_action_prior_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = _normalize_legacy_paths(ckpt["config"])
    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    if cfg["model"].get("official_salad_vae_checkpoint"):
        motion_vae = load_official_salad_vae(
            cfg["model"]["official_salad_vae_opt"],
            cfg["model"]["official_salad_vae_checkpoint"],
            torch.device("cpu"),
        )
    else:
        motion_vae = load_motion_vae_from_checkpoint(cfg["model"]["motion_vae_checkpoint"])
    prior = build_action_prior_from_config(cfg, lam, motion_vae, torch.device("cpu"))
    prior.load_state_dict_without_clip(ckpt["action_prior"])
    return prior, cfg


def _normalize_legacy_paths(value):
    if isinstance(value, dict):
        return {key: _normalize_legacy_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_legacy_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace("/work/adamotion/data/HumanML3D", "/workspace/dataset/HumanML3D").replace(
            "/work/adamotion", "/workspace/AdaMotion"
        )
    return value


@torch.no_grad()
def sample_motion_diffusion(
    denoiser: nn.Module,
    motion_vae,
    scheduler: DDIMScheduler,
    texts: list[str],
    context_motion: torch.Tensor,
    action_source: torch.Tensor | None,
    lam,
    action_latent_seq: torch.Tensor | None = None,
    future_len: int | None = None,
    num_inference_timesteps: int = 50,
    cond_scale: float = 1.0,
    use_action_condition: bool = True,
) -> torch.Tensor:
    device = context_motion.device
    z_past = motion_vae.encode_deterministic(context_motion)[0]
    if action_latent_seq is None:
        if action_source is None:
            if future_len is None:
                raise ValueError("future_len is required when action_source and action_latent_seq are both absent.")
            action_seq = torch.zeros(
                context_motion.shape[0],
                future_len,
                lam.latent_dim,
                device=device,
                dtype=context_motion.dtype,
            )
            latent_future_steps = max(1, future_len // 4)
        else:
            action_seq = lam.encode_action_sequence(action_source, texts=texts)
            latent_future_steps = max(1, (action_source.shape[1] - 1) // 4)
    else:
        action_seq = action_latent_seq
        latent_future_steps = max(1, action_seq.shape[1] // 4)
    noise_shape = (context_motion.shape[0], latent_future_steps, 7, motion_vae.latent_dim)
    latents = torch.randn(noise_shape, device=device, dtype=context_motion.dtype) * scheduler.init_noise_sigma
    action_seq = _pool_action_sequence(action_seq, latents.shape[1])
    if not use_action_condition:
        action_seq = torch.zeros_like(action_seq)

    scheduler.set_timesteps(num_inference_timesteps)
    timesteps = scheduler.timesteps.to(device)
    len_mask = torch.ones(latents.shape[0], latents.shape[1], dtype=torch.bool, device=device)

    for timestep in timesteps:
        if cond_scale > 1.0:
            latent_input = torch.cat([latents, latents], dim=0)
            past_input = torch.cat([z_past, z_past], dim=0)
            action_input = torch.cat([torch.zeros_like(action_seq), action_seq], dim=0)
            text_input = [""] * len(texts) + texts
            mask_input = torch.cat([len_mask, len_mask], dim=0)
            pred = denoiser(
                noisy_future=latent_input,
                timesteps=timestep.expand(latent_input.shape[0]),
                texts=text_input,
                past_latent=past_input,
                action_latent_seq=action_input,
                len_mask=mask_input,
                use_cached_clip=True,
            )
            pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + cond_scale * (pred_cond - pred_uncond)
        else:
            pred = denoiser(
                noisy_future=latents,
                timesteps=timestep.expand(latents.shape[0]),
                texts=texts,
                past_latent=z_past,
                action_latent_seq=action_seq,
                len_mask=len_mask,
                use_cached_clip=True,
            )
        latents = scheduler.step(pred, timestep, latents).prev_sample
        latents = latents * len_mask[..., None, None].float()

    denoiser.remove_clip_cache()
    return motion_vae.decode(latents)


def _motion_vector_to_lam_source(motion: torch.Tensor) -> torch.Tensor:
    sal_rep = humanml_vector_to_sal_rep(motion.detach().cpu().numpy())
    return torch.from_numpy(sal_rep).to(device=motion.device, dtype=motion.dtype)


def _lam_required_motion_frames(lam) -> int:
    if getattr(lam, "predict_next_from_momentum", False):
        return int(getattr(lam, "window_len", 2) + getattr(lam, "target_offset", 1))
    return 2


def _prior_action_latent(
    action_prior,
    texts: list[str],
    context_motion: torch.Tensor,
    motion_vae,
    future_len: int,
) -> torch.Tensor:
    family = getattr(action_prior, "family", "")
    if family == "text_length_action_prior":
        lengths = torch.full(
            (context_motion.shape[0],),
            future_len,
            device=context_motion.device,
            dtype=torch.long,
        )
        return action_prior(texts, lengths)
    past_latent = motion_vae.encode_deterministic(context_motion)[0]
    return action_prior(texts, past_latent)


def _encode_pair_action(lam, x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
    if x_prev.dim() == 3 and x_prev.shape[-1] == getattr(lam, "joint_dim", x_prev.shape[-1]):
        return lam.encode(x_prev, x_next)["mu"]
    if x_prev.dim() == 2:
        action_source = _motion_vector_to_lam_source(torch.stack([x_prev, x_next], dim=1))
        return lam.encode_action_sequence(action_source)[:, 0]
    raise ValueError(
        f"Unsupported pair shapes for LAM encoding: x_prev={tuple(x_prev.shape)}, x_next={tuple(x_next.shape)}, "
        f"lam_joint_dim={getattr(lam, 'joint_dim', 'unknown')}"
    )


@torch.no_grad()
def rollout_motion_diffusion_autoregressive(
    denoiser: nn.Module,
    motion_vae,
    scheduler: DDIMScheduler,
    texts: list[str],
    context_motion: torch.Tensor,
    lam,
    num_segments: int,
    action_prior=None,
    future_len: int | None = None,
    latent_action_policy: str = "autoregressive_latent_action",
    num_inference_timesteps: int = 50,
    cond_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    valid_policies = {"no_latent_action", "prior_latent_action", "autoregressive_latent_action"}
    if latent_action_policy not in valid_policies:
        raise ValueError(f"Unsupported latent_action_policy: {latent_action_policy}")
    if future_len is None:
        future_len = context_motion.shape[1]
    if latent_action_policy == "prior_latent_action" and action_prior is None:
        raise ValueError("action_prior is required when latent_action_policy='prior_latent_action'.")

    current_context = context_motion
    current_action_source = None
    generated_segments = []
    action_latents = []
    action_sources = []
    lam_autoregressive_start_segment = None
    required_motion_frames = _lam_required_motion_frames(lam)

    for segment_idx in range(num_segments):
        if latent_action_policy == "no_latent_action":
            segment_uses_action = False
            action_source_label = "no_action"
            action_latent = torch.zeros(
                current_context.shape[0],
                future_len,
                lam.latent_dim,
                device=current_context.device,
                dtype=current_context.dtype,
            )
        else:
            segment_uses_action = True
            if current_action_source is not None and current_action_source.shape[1] >= required_motion_frames:
                action_source_label = "lam_ar"
                if lam_autoregressive_start_segment is None:
                    lam_autoregressive_start_segment = segment_idx
                action_latent = lam.encode_action_sequence(current_action_source, texts=texts)
            elif latent_action_policy == "prior_latent_action":
                action_source_label = "prior"
                action_latent = _prior_action_latent(
                    action_prior=action_prior,
                    texts=texts,
                    context_motion=current_context,
                    motion_vae=motion_vae,
                    future_len=future_len,
                )
            else:
                action_source_label = "lam_unavailable"
                action_latent = torch.zeros(
                    current_context.shape[0],
                    future_len,
                    lam.latent_dim,
                    device=current_context.device,
                    dtype=current_context.dtype,
                )
                segment_uses_action = False
        latent_future_steps = max(1, future_len // 4)
        action_latent = _pool_action_sequence(action_latent, latent_future_steps)
        pred_future = sample_motion_diffusion(
            denoiser=denoiser,
            motion_vae=motion_vae,
            scheduler=scheduler,
            texts=texts,
            context_motion=current_context,
            action_source=None,
            lam=lam,
            action_latent_seq=action_latent,
            future_len=future_len,
            num_inference_timesteps=num_inference_timesteps,
            cond_scale=cond_scale,
            use_action_condition=segment_uses_action,
        )
        generated_segments.append(pred_future)
        action_latents.append(action_latent)
        action_sources.append(action_source_label)
        current_action_source = _motion_vector_to_lam_source(torch.cat([current_context[:, -1:], pred_future], dim=1))
        full_motion = torch.cat([current_context, pred_future], dim=1)
        current_context = full_motion[:, -context_motion.shape[1] :]

    return {
        "motion": torch.cat([context_motion, *generated_segments], dim=1),
        "future": torch.cat(generated_segments, dim=1),
        "action_latents": torch.cat(action_latents, dim=1),
        "action_source_modes": action_sources,
        "lam_autoregressive_start_segment": lam_autoregressive_start_segment,
    }


def train_legacy_world_model_from_config(cfg: Dict) -> Dict:
    train_set = HumanMLContextDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_set = HumanMLContextDataset(**val_args)

    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    lam.to(device).eval()
    for param in lam.parameters():
        param.requires_grad = False

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
    wandb_run = init_wandb_run(REPO_ROOT, "world_model", cfg)
    progress = ExperimentProgress(
        stage="world_model_legacy",
        epochs=cfg["train"]["epochs"],
        train_steps=len(train_loader),
        val_steps=len(val_loader),
        run=wandb_run,
        log_interval=cfg["train"].get("progress_log_interval", 20),
    )

    try:
      for epoch in range(cfg["train"]["epochs"]):
        predictor.train()
        no_action.train()
        train_action_loss = 0.0
        train_no_action_loss = 0.0
        train_count = 0
        progress.start_epoch(epoch)
        progress.start_phase("train", len(train_loader))
        for batch in train_loader:
            batch = _to_device(batch, device)
            with torch.no_grad():
                z = _encode_pair_action(lam, batch["x_prev"], batch["x_next"])
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
            denom = max(1, train_count)
            progress.update(
                "train",
                {
                    "action": train_action_loss / denom,
                    "no_action": train_no_action_loss / denom,
                },
            )
        progress.end_phase()

        predictor.eval()
        no_action.eval()
        val_action_loss = 0.0
        val_no_action_loss = 0.0
        val_count = 0
        with torch.no_grad():
            progress.start_phase("val", len(val_loader))
            for batch in val_loader:
                batch = _to_device(batch, device)
                z = _encode_pair_action(lam, batch["x_prev"], batch["x_next"])
                pred = predictor(batch["context"], z)
                pred_no = no_action(batch["context"])
                loss_pred = F.mse_loss(pred, batch["target"])
                loss_no = F.mse_loss(pred_no, batch["target"])
                batch_size = batch["context"].shape[0]
                val_count += batch_size
                val_action_loss += float(loss_pred.detach()) * batch_size
                val_no_action_loss += float(loss_no.detach()) * batch_size
                denom = max(1, val_count)
                progress.update(
                    "val",
                    {
                        "action": val_action_loss / denom,
                        "no_action": val_no_action_loss / denom,
                    },
                )
            progress.end_phase()

        row = {
            "epoch": epoch,
            "train_action_loss": train_action_loss / max(1, train_count),
            "train_no_action_loss": train_no_action_loss / max(1, train_count),
            "val_action_loss": val_action_loss / max(1, val_count),
            "val_no_action_loss": val_no_action_loss / max(1, val_count),
            "val_gain": (val_no_action_loss - val_action_loss) / max(1, val_count),
        }
        print(
            f"[world legacy][epoch {epoch + 1}/{cfg['train']['epochs']}] "
            f"train_action={row['train_action_loss']:.6f} train_no_action={row['train_no_action_loss']:.6f} "
            f"val_action={row['val_action_loss']:.6f} val_no_action={row['val_no_action_loss']:.6f} val_gain={row['val_gain']:.6f}",
            flush=True,
        )
        history.append(row)
        log_wandb_epoch(
            wandb_run,
            row,
            extra={
                "stage": "world_model",
                "generator_type": "legacy_predictor",
                "output_dir": str(out_dir),
                "best_val_action_loss_so_far": min(item["val_action_loss"] for item in history),
            },
        )
        save_payload = {"predictor": predictor.state_dict(), "no_action": no_action.state_dict(), "config": cfg}
        if row["val_action_loss"] < best["val_action_loss"]:
            best = row
            torch.save({**save_payload, "best": best}, out_dir / "world_best.pt")
        torch.save({**save_payload, "history": history}, out_dir / "world_last.pt")
    finally:
        progress.close()

    with open(out_dir / "world_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    summary = finalize_run(
        repo_root=REPO_ROOT,
        stage="world_model",
        cfg=cfg,
        history=history,
        checkpoint_path=str(out_dir / "world_last.pt"),
        best_metrics=best,
        extra_summary={"family": "joint_st_transformer", "generator_type": "legacy_predictor"},
    )
    finish_wandb_run(wandb_run, summary)
    return {"best": best, "checkpoint": str(out_dir / "world_last.pt")}


def train_motion_diffusion_from_config(cfg: Dict) -> Dict:
    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    train_set = HumanMLTextFutureDataset(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val")
    val_args["random_caption"] = False
    val_set = HumanMLTextFutureDataset(**val_args)

    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"])
    lam.to(device).eval()
    for param in lam.parameters():
        param.requires_grad = False

    motion_vae = load_motion_vae_from_checkpoint(cfg["model"]["motion_vae_checkpoint"])
    motion_vae.to(device).eval()
    motion_vae.freeze()

    denoiser = build_motion_diffusion_from_config(cfg, lam, motion_vae, device)
    optimizer = torch.optim.AdamW(
        denoiser.parameters_without_clip(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = _build_noise_scheduler(cfg)
    prediction_type = cfg["model"].get("prediction_type", "v_prediction")
    use_action_condition = cfg["model"].get("use_action_condition", True)
    cond_drop_prob = cfg["model"].get("cond_drop_prob", 0.0)

    train_loader = _make_loader(train_set, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    val_loader = _make_loader(val_set, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))

    history = []
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"val_loss": float("inf")}
    ckpt_path = out_dir / "world_last.pt"
    wandb_run = init_wandb_run(REPO_ROOT, "world_model", cfg)
    progress = ExperimentProgress(
        stage="world_model_diffusion",
        epochs=cfg["train"]["epochs"],
        train_steps=len(train_loader),
        val_steps=len(val_loader),
        run=wandb_run,
        log_interval=cfg["train"].get("progress_log_interval", 20),
    )

    def compute_step(batch: Dict) -> Dict[str, torch.Tensor]:
        batch = _to_device(batch, device)
        with torch.no_grad():
            z_past = motion_vae.encode_deterministic(batch["context_motion"])[0]
            z_future = motion_vae.encode_deterministic(batch["future_motion"])[0]
            action_seq = lam.encode_action_sequence(
                batch["action_source"],
                texts=list(batch["text"]),
                start_timestep=batch.get("start_idx"),
            )
            action_seq = _pool_action_sequence(action_seq, z_future.shape[1])
            if not use_action_condition:
                action_seq = torch.zeros_like(action_seq)

        len_mask = torch.ones(z_future.shape[0], z_future.shape[1], dtype=torch.bool, device=device)
        timesteps = torch.randint(
            0,
            cfg["model"].get("num_train_timesteps", 1000),
            (z_future.shape[0],),
            device=device,
        ).long()
        noise = torch.randn_like(z_future)
        noisy_future = scheduler.add_noise(z_future, noise, timesteps)
        pred = denoiser(
            noisy_future=noisy_future,
            timesteps=timesteps,
            texts=_maybe_drop_text_condition(list(batch["text"]), cond_drop_prob),
            past_latent=z_past,
            action_latent_seq=action_seq,
            len_mask=len_mask,
        )
        if prediction_type == "sample":
            target = z_future
        elif prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = scheduler.get_velocity(z_future, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        loss = F.mse_loss(pred, target)
        return {
            "loss": loss,
            "loss_recon": F.mse_loss(pred, target),
            "latent_mse": F.mse_loss(noisy_future, z_future),
        }

    try:
      for epoch in range(cfg["train"]["epochs"]):
        denoiser.train()
        train_metrics = {"loss": 0.0, "loss_recon": 0.0, "latent_mse": 0.0, "count": 0}
        progress.start_epoch(epoch)
        progress.start_phase("train", len(train_loader))
        for batch in train_loader:
            outputs = compute_step(batch)
            optimizer.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            optimizer.step()
            batch_size = len(batch["text"])
            train_metrics["count"] += batch_size
            for key in ["loss", "loss_recon", "latent_mse"]:
                train_metrics[key] += float(outputs[key].detach()) * batch_size
            denom = max(1, train_metrics["count"])
            progress.update(
                "train",
                {
                    "loss": train_metrics["loss"] / denom,
                    "recon": train_metrics["loss_recon"] / denom,
                    "latent": train_metrics["latent_mse"] / denom,
                },
            )
        progress.end_phase()

        denoiser.eval()
        val_metrics = {"loss": 0.0, "loss_recon": 0.0, "latent_mse": 0.0, "count": 0}
        with torch.no_grad():
            progress.start_phase("val", len(val_loader))
            for batch in val_loader:
                outputs = compute_step(batch)
                batch_size = len(batch["text"])
                val_metrics["count"] += batch_size
                for key in ["loss", "loss_recon", "latent_mse"]:
                    val_metrics[key] += float(outputs[key].detach()) * batch_size
                denom = max(1, val_metrics["count"])
                progress.update(
                    "val",
                    {
                        "loss": val_metrics["loss"] / denom,
                        "recon": val_metrics["loss_recon"] / denom,
                        "latent": val_metrics["latent_mse"] / denom,
                    },
                )
            progress.end_phase()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"] / max(1, train_metrics["count"]),
            "train_recon": train_metrics["loss_recon"] / max(1, train_metrics["count"]),
            "train_latent_mse": train_metrics["latent_mse"] / max(1, train_metrics["count"]),
            "val_loss": val_metrics["loss"] / max(1, val_metrics["count"]),
            "val_recon": val_metrics["loss_recon"] / max(1, val_metrics["count"]),
            "val_latent_mse": val_metrics["latent_mse"] / max(1, val_metrics["count"]),
        }
        print(
            f"[world diffusion][epoch {epoch + 1}/{cfg['train']['epochs']}] "
            f"train_loss={row['train_loss']:.6f} train_recon={row['train_recon']:.6f} train_latent={row['train_latent_mse']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_recon={row['val_recon']:.6f} val_latent={row['val_latent_mse']:.6f}",
            flush=True,
        )
        history.append(row)
        log_wandb_epoch(
            wandb_run,
            row,
            extra={
                "stage": "world_model",
                "generator_type": "motion_diffusion",
                "use_action_condition": use_action_condition,
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
            "family": "salad_motion_diffusion",
            "generator_type": "motion_diffusion",
            "use_action_condition": use_action_condition,
        },
    )
    finish_wandb_run(wandb_run, summary)
    return {"best": best, "checkpoint": str(ckpt_path)}


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
    motion_vae = load_official_salad_vae(
        cfg["model"]["official_salad_vae_opt"],
        cfg["model"]["official_salad_vae_checkpoint"],
        device,
    )
    denoiser = load_official_salad_action_denoiser(
        cfg["model"]["official_salad_denoiser_opt"],
        cfg["model"]["official_salad_denoiser_checkpoint"],
        vae_dim=motion_vae.latent_dim,
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
            z_motion = motion_vae.encode_deterministic(batch["motion"])[0]
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
            progress.update(
                "train",
                {
                    "loss": train_metrics["loss"] / denom,
                    "latent": train_metrics["latent_mse"] / denom,
                },
            )
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
                progress.update(
                    "val",
                    {
                        "loss": val_metrics["loss"] / denom,
                        "latent": val_metrics["latent_mse"] / denom,
                    },
                )
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
    cfg = dict(cfg)
    cfg["stage"] = "action_prior"
    set_seed(cfg["seed"])

    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    prior_type = cfg["model"].get("prior_type", "context_future")
    dataset_cls = HumanMLFullMotionActionDataset if prior_type == "text_length_action" else HumanMLTextFutureDataset
    train_set = dataset_cls(**cfg["data"]["dataset"])
    val_args = dict(cfg["data"]["dataset"])
    val_args["split"] = cfg["data"].get("val_split", "val_usable")
    val_args["random_caption"] = False
    val_set = dataset_cls(**val_args)

    lam = load_lam_from_checkpoint(cfg["model"]["lam_checkpoint"]).to(device).eval()
    for param in lam.parameters():
        param.requires_grad = False
    if cfg["model"].get("official_salad_vae_checkpoint"):
        motion_vae = load_official_salad_vae(
            cfg["model"]["official_salad_vae_opt"],
            cfg["model"]["official_salad_vae_checkpoint"],
            device,
        )
    else:
        motion_vae = load_motion_vae_from_checkpoint(cfg["model"]["motion_vae_checkpoint"]).to(device).eval()
        motion_vae.freeze()

    prior = build_action_prior_from_config(cfg, lam, motion_vae, device)
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
        if prior_type == "text_length_action":
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
        else:
            with torch.no_grad():
                past_latent = motion_vae.encode_deterministic(batch["context_motion"])[0]
                target_action = lam.encode_action_sequence(
                    batch["action_source"],
                    texts=list(batch["text"]),
                    start_timestep=batch.get("start_idx"),
                )
            pred_action = prior(list(batch["text"]), past_latent)
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
            f"train_loss={row['train_loss']:.6f} train_target_std={row['train_target_std']:.6f} train_pred_std={row['train_pred_std']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_target_std={row['val_target_std']:.6f} val_pred_std={row['val_pred_std']:.6f}",
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
    if _generator_type_from_cfg(cfg) == "official_salad_action_adapter":
        return train_official_salad_action_adapter_from_config(cfg)
    if _generator_type_from_cfg(cfg) == "motion_diffusion":
        return train_motion_diffusion_from_config(cfg)
    return train_legacy_world_model_from_config(cfg)


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
