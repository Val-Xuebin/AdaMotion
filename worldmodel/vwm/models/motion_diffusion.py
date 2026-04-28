from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

SALAD_ROOT = Path(__file__).resolve().parents[4] / "humanmodels" / "salad"
if str(SALAD_ROOT) not in sys.path:
    sys.path.append(str(SALAD_ROOT))

from models.denoiser.embedding import PositionalEmbedding, TimestepEmbedding
from models.denoiser.transformer import SkipTransformer


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, clip_version: str = "ViT-B/32") -> None:
        super().__init__()
        if clip_version == "ViT-B/32":
            model_name = "openai/clip-vit-base-patch32"
            clip_dim = 512
        elif clip_version == "ViT-L/14":
            model_name = "openai/clip-vit-large-patch14"
            clip_dim = 768
        else:
            raise ValueError(f"Unsupported CLIP version: {clip_version}")
        self.clip_dim = clip_dim
        self.backend = "hf"
        if os.environ.get("ADAMOTION_TEXT_ENCODER_BACKEND", "").lower() == "hash":
            self.backend = "hash"
            self.tokenizer = None
            self.model = None
            self.max_length = 77
            print(
                "[motion_diffusion] Using hash text encoder because ADAMOTION_TEXT_ENCODER_BACKEND=hash",
                file=sys.stderr,
            )
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.max_length = self.tokenizer.model_max_length
            self.freeze()
        except Exception as exc:
            self.backend = "hash"
            self.tokenizer = None
            self.model = None
            self.max_length = 77
            print(
                f"[motion_diffusion] Falling back to hash text encoder because CLIP load failed: {exc}",
                file=sys.stderr,
            )

    def freeze(self) -> None:
        if self.model is None:
            return
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _hash_encode_text(self, text: list[str]):
        batch_size = len(text)
        max_tokens = self.max_length
        device = torch.device("cpu")
        word_emb = torch.zeros(batch_size, max_tokens, self.clip_dim, dtype=torch.float32, device=device)
        attn_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
        eos_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        for batch_idx, sentence in enumerate(text):
            tokens = sentence.strip().split()
            if not tokens:
                continue
            tokens = tokens[:max_tokens]
            eos_positions[batch_idx] = len(tokens) - 1
            for token_idx, token in enumerate(tokens):
                attn_mask[batch_idx, token_idx] = True
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                values = list(digest)
                repeats = (self.clip_dim + len(values) - 1) // len(values)
                vec = (values * repeats)[: self.clip_dim]
                word_emb[batch_idx, token_idx] = torch.tensor(vec, dtype=torch.float32, device=device) / 255.0
        return word_emb, attn_mask, eos_positions

    @torch.no_grad()
    def encode_text(self, text: list[str]):
        if self.backend == "hash":
            return self._hash_encode_text(text)
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        text_input_ids = tokens.input_ids.to(self.model.device)
        text_attn_mask = tokens.attention_mask.to(self.model.device).bool()
        word_emb = self.model.text_model(text_input_ids).last_hidden_state
        return word_emb, text_attn_mask, text_input_ids.argmax(dim=-1)


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


class ActionConditionedMotionDenoiser(nn.Module):
    family = "salad_motion_diffusion"

    def __init__(
        self,
        latent_dim: int = 256,
        vae_latent_dim: int = 32,
        action_dim: int = 32,
        motion_joints: int = 7,
        n_heads: int = 8,
        n_layers: int = 5,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        clip_version: str = "ViT-B/32",
        max_text_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.motion_joints = motion_joints
        self.max_text_tokens = max_text_tokens
        opt = SimpleNamespace(
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            clip_version=clip_version,
            device="cpu",
        )
        self.input_process = InputProcess(latent_dim, vae_latent_dim)
        self.output_process = OutputProcess(latent_dim, vae_latent_dim)
        self.timestep_emb = TimestepEmbedding(latent_dim)
        self.text_encoder = FrozenCLIPTextEncoder(clip_version)
        self.clip_dim = 512 if clip_version == "ViT-B/32" else 768
        self.word_emb = nn.Linear(self.clip_dim, latent_dim)
        self.past_proj = nn.Linear(vae_latent_dim, latent_dim)
        self.action_proj = nn.Linear(action_dim, latent_dim)
        self.past_pos_emb = PositionalEmbedding(latent_dim, dropout)
        self.action_pos_emb = PositionalEmbedding(latent_dim, dropout)
        self.latent_pos_emb = PositionalEmbedding(latent_dim, dropout)
        self.transformer = SkipTransformer(opt)
        self._cache_word_emb: torch.Tensor | None = None
        self._cache_ca_mask: torch.Tensor | None = None

    def parameters_without_clip(self):
        return [param for name, param in self.named_parameters() if "text_encoder.model" not in name]

    def state_dict_without_clip(self):
        state_dict = self.state_dict()
        remove_weights = [key for key in state_dict if key.startswith("text_encoder.model.") or key.startswith("_cache_")]
        for key in remove_weights:
            del state_dict[key]
        return state_dict

    def remove_clip_cache(self) -> None:
        self._cache_word_emb = None
        self._cache_ca_mask = None

    def load_state_dict_without_clip(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            raise ValueError(f"Unexpected keys when loading denoiser: {unexpected_keys}")
        non_clip_missing = [key for key in missing_keys if not key.startswith("text_encoder.model.")]
        if non_clip_missing:
            raise ValueError(f"Missing non-CLIP keys when loading denoiser: {non_clip_missing}")

    def forward(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        texts: list[str],
        past_latent: torch.Tensor,
        action_latent_seq: torch.Tensor,
        len_mask: torch.Tensor | None = None,
        use_cached_clip: bool = False,
    ) -> torch.Tensor:
        x = self.input_process(noisy_future)
        batch_size, time_steps, joints, hidden_dim = x.shape

        timestep_emb = self.timestep_emb(timesteps).expand(batch_size, hidden_dim)

        if use_cached_clip and self._cache_word_emb is not None and self._cache_ca_mask is not None:
            text_tokens = self._cache_word_emb
            text_mask = self._cache_ca_mask
        else:
            if self.text_encoder.model is not None:
                self.text_encoder.model.to(x.device)
            text_tokens, text_mask, _ = self.text_encoder.encode_text(texts)
            text_tokens = text_tokens.to(x.device)
            text_mask = text_mask.to(x.device)
            text_tokens = self.word_emb(text_tokens[:, : self.max_text_tokens])
            text_mask = text_mask[:, : self.max_text_tokens]
            if use_cached_clip:
                self._cache_word_emb = text_tokens
                self._cache_ca_mask = text_mask

        past_tokens = self.past_proj(past_latent.reshape(batch_size, -1, past_latent.shape[-1]))
        past_tokens = self.past_pos_emb(past_tokens)
        past_mask = torch.ones(batch_size, past_tokens.shape[1], dtype=torch.bool, device=x.device)

        action_tokens = self.action_proj(action_latent_seq)
        action_tokens = self.action_pos_emb(action_tokens)
        action_mask = torch.ones(batch_size, action_tokens.shape[1], dtype=torch.bool, device=x.device)

        memory = torch.cat([text_tokens, past_tokens, action_tokens], dim=1)
        memory_mask = torch.cat([text_mask, past_mask, action_mask], dim=1)

        x = x.reshape(batch_size, time_steps * joints, hidden_dim)
        x = self.latent_pos_emb(x)
        x = x.reshape(batch_size, time_steps, joints, hidden_dim)

        if len_mask is not None:
            sa_mask = len_mask.repeat_interleave(joints, dim=0)
        else:
            sa_mask = None

        x, _ = self.transformer(
            x,
            timestep_emb,
            memory,
            sa_mask=None if sa_mask is None else ~sa_mask,
            ca_mask=~memory_mask,
            need_attn=False,
        )
        return self.output_process(x)
