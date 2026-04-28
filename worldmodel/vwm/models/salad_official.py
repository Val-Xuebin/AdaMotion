from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
from torch import nn

SALAD_ROOT = Path(__file__).resolve().parents[4] / "humanmodels" / "salad"
if str(SALAD_ROOT) not in sys.path:
    sys.path.append(str(SALAD_ROOT))

from models.denoiser.embedding import PositionalEmbedding, TimestepEmbedding
from models.denoiser.transformer import SkipTransformer
from models.vae.model import VAE as SaladVAE

from vwm.models.motion_diffusion import FrozenCLIPTextEncoder


def salad_opt_from_checkpoint_opt(
    opt_path: str | Path,
    device: torch.device | str = "cpu",
    checkpoints_dir: str | Path | None = None,
) -> SimpleNamespace:
    values: dict[str, object] = {}
    with open(opt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-") or ": " not in line:
                continue
            key, raw_value = line.split(": ", 1)
            values[key] = _parse_opt_value(raw_value)
    if checkpoints_dir is not None:
        values["checkpoints_dir"] = str(checkpoints_dir)
    values["device"] = torch.device(device)
    values.setdefault("dataset_name", "t2m")
    values.setdefault("pose_dim", 263)
    values.setdefault("joints_num", 22)
    values.setdefault("contact_joints", [7, 10, 8, 11])
    values.setdefault("clip_version", "ViT-B/32")
    values.setdefault("dropout", 0.1)
    values.setdefault("latent_dim", 32)
    return SimpleNamespace(**values)


def _parse_opt_value(value: str):
    if value in {"True", "False"}:
        return value == "True"
    if value == "None":
        return None
    if value.startswith("[") and value.endswith("]"):
        import ast

        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


class OfficialSaladVAE(nn.Module):
    family = "official_salad_vae"

    def __init__(self, opt: SimpleNamespace) -> None:
        super().__init__()
        self.opt = opt
        self.model = SaladVAE(opt)
        self.latent_dim = opt.latent_dim
        self.pose_dim = opt.pose_dim

    def freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, motion: torch.Tensor):
        return self.model.encode(motion)

    def encode_deterministic(self, motion: torch.Tensor):
        x = self.model.motion_enc(motion)
        x = self.model.conv_enc(x)
        moments = self.model.dist(x)
        mu, logvar = moments.chunk(2, dim=-1)
        return mu, {"loss_kl": 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)}

    def decode(self, latent: torch.Tensor):
        return self.model.decode(latent)


def load_official_salad_vae(opt_path: str | Path, checkpoint_path: str | Path, device: torch.device) -> OfficialSaladVAE:
    opt = salad_opt_from_checkpoint_opt(opt_path, device=device)
    model = OfficialSaladVAE(opt)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.model.load_state_dict(ckpt["vae"])
    model.to(device)
    model.freeze()
    return model


class InputProcess(nn.Module):
    def __init__(self, hidden_dim: int, in_features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OutputProcess(nn.Module):
    def __init__(self, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OfficialSaladActionDenoiser(nn.Module):
    family = "official_salad_action_adapter"

    def __init__(
        self,
        opt: SimpleNamespace,
        vae_dim: int = 32,
        action_dim: int = 32,
        max_text_tokens: int = 77,
        train_base: bool = False,
    ) -> None:
        super().__init__()
        self.opt = opt
        self.latent_dim = opt.latent_dim
        self.action_dim = action_dim
        self.max_text_tokens = max_text_tokens
        self.input_process = InputProcess(opt.latent_dim, vae_dim)
        self.output_process = OutputProcess(opt.latent_dim, vae_dim)
        self.timestep_emb = TimestepEmbedding(opt.latent_dim)
        self.text_encoder = FrozenCLIPTextEncoder(opt.clip_version)
        self.clip_dim = 512 if opt.clip_version == "ViT-B/32" else 768
        self.word_emb = nn.Linear(self.clip_dim, opt.latent_dim)
        self.pos_emb = PositionalEmbedding(opt.latent_dim, opt.dropout)
        self.action_proj = nn.Linear(action_dim, opt.latent_dim)
        self.action_pos_emb = PositionalEmbedding(opt.latent_dim, opt.dropout)
        self.transformer = SkipTransformer(opt)
        self._cache_word_emb: torch.Tensor | None = None
        self._cache_ca_mask: torch.Tensor | None = None
        self.freeze_base(train_base=train_base)

    def freeze_base(self, train_base: bool = False) -> None:
        base_prefixes = ("input_process", "output_process", "timestep_emb", "word_emb", "pos_emb", "transformer")
        for name, param in self.named_parameters():
            if name.startswith("text_encoder.model."):
                param.requires_grad = False
            elif name.startswith(base_prefixes):
                param.requires_grad = train_base
            else:
                param.requires_grad = True

    def load_official_denoiser_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("clip_model."):
                continue
            mapped[key] = value
        missing_keys, unexpected_keys = self.load_state_dict(mapped, strict=False)
        if unexpected_keys:
            raise ValueError(f"Unexpected official SALAD denoiser keys: {unexpected_keys}")
        allowed_missing = tuple(["text_encoder.model.", "action_proj.", "action_pos_emb."])
        non_adapter_missing = [key for key in missing_keys if not key.startswith(allowed_missing)]
        if non_adapter_missing:
            raise ValueError(f"Missing non-adapter SALAD denoiser keys: {non_adapter_missing}")

    def parameters_without_clip(self):
        return [param for name, param in self.named_parameters() if param.requires_grad and "text_encoder.model" not in name]

    def state_dict_without_clip(self):
        state_dict = self.state_dict()
        for key in [key for key in state_dict if key.startswith("text_encoder.model.") or key.startswith("_cache_")]:
            del state_dict[key]
        return state_dict

    def remove_clip_cache(self) -> None:
        self._cache_word_emb = None
        self._cache_ca_mask = None

    def load_state_dict_without_clip(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            raise ValueError(f"Unexpected keys when loading official action denoiser: {unexpected_keys}")
        non_clip_missing = [key for key in missing_keys if not key.startswith("text_encoder.model.")]
        if non_clip_missing:
            raise ValueError(f"Missing non-CLIP keys when loading official action denoiser: {non_clip_missing}")

    def _encode_text(self, texts: list[str], device: torch.device, use_cached_clip: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if use_cached_clip and self._cache_word_emb is not None and self._cache_ca_mask is not None:
            return self._cache_word_emb, self._cache_ca_mask
        if self.text_encoder.model is not None:
            self.text_encoder.model.to(device)
        text_tokens, text_mask, _ = self.text_encoder.encode_text(texts)
        text_tokens = text_tokens.to(device)[:, : self.max_text_tokens]
        text_mask = text_mask.to(device)[:, : self.max_text_tokens]
        text_tokens = self.word_emb(text_tokens)
        if use_cached_clip:
            self._cache_word_emb = text_tokens
            self._cache_ca_mask = text_mask
        return text_tokens, text_mask

    def forward(
        self,
        noisy_motion_latent: torch.Tensor,
        timesteps: torch.Tensor,
        texts: list[str],
        action_latent_seq: torch.Tensor,
        len_mask: torch.Tensor | None = None,
        use_cached_clip: bool = False,
        use_action_condition: bool = True,
    ) -> torch.Tensor:
        x = self.input_process(noisy_motion_latent)
        batch_size, time_steps, joints, hidden_dim = x.shape
        timestep_emb = self.timestep_emb(timesteps).expand(batch_size, hidden_dim)

        text_tokens, text_mask = self._encode_text(texts, x.device, use_cached_clip)
        if use_action_condition and action_latent_seq.shape[1] > 0:
            action_tokens = self.action_pos_emb(self.action_proj(action_latent_seq))
            action_mask = torch.ones(batch_size, action_tokens.shape[1], dtype=torch.bool, device=x.device)
            memory = torch.cat([text_tokens, action_tokens], dim=1)
            memory_mask = torch.cat([text_mask, action_mask], dim=1)
        else:
            memory = text_tokens
            memory_mask = text_mask

        x = x.reshape(batch_size, time_steps * joints, hidden_dim)
        x = self.pos_emb(x)
        x = x.reshape(batch_size, time_steps, joints, hidden_dim)
        sa_mask = len_mask.repeat_interleave(joints, dim=0) if len_mask is not None else None
        x, _ = self.transformer(
            x,
            timestep_emb,
            memory,
            sa_mask=None if sa_mask is None else ~sa_mask,
            ca_mask=~memory_mask,
            need_attn=False,
        )
        return self.output_process(x)


def load_official_salad_action_denoiser(
    opt_path: str | Path,
    checkpoint_path: str | Path,
    vae_dim: int,
    action_dim: int,
    device: torch.device,
    train_base: bool = False,
) -> OfficialSaladActionDenoiser:
    opt = salad_opt_from_checkpoint_opt(opt_path, device=device)
    model = OfficialSaladActionDenoiser(opt, vae_dim=vae_dim, action_dim=action_dim, train_base=train_base)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_official_denoiser_state(ckpt["denoiser"])
    model.to(device)
    model.freeze_base(train_base=train_base)
    return model
