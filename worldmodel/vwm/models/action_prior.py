from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from vwm.models.motion_diffusion import FrozenCLIPTextEncoder


class TextLengthActionPrior(nn.Module):
    family = "text_length_action_prior"

    def __init__(
        self,
        action_dim: int = 32,
        hidden_dim: int = 256,
        latent_steps: int = 49,
        dropout: float = 0.1,
        clip_version: str = "ViT-B/32",
        max_text_tokens: int = 32,
        max_motion_length: int = 196,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_steps = latent_steps
        self.max_text_tokens = max_text_tokens
        self.max_motion_length = max_motion_length
        self.text_encoder = FrozenCLIPTextEncoder(clip_version)
        self.clip_dim = 512 if clip_version == "ViT-B/32" else 768
        self.text_proj = nn.Linear(self.clip_dim, hidden_dim)
        self.length_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_steps * action_dim),
        )

    def parameters_without_clip(self):
        return [param for name, param in self.named_parameters() if "text_encoder.model" not in name]

    def state_dict_without_clip(self):
        state_dict = self.state_dict()
        for key in [key for key in state_dict if key.startswith("text_encoder.model.")]:
            del state_dict[key]
        return state_dict

    def load_state_dict_without_clip(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            raise ValueError(f"Unexpected keys when loading text-length action prior: {unexpected_keys}")
        non_clip_missing = [key for key in missing_keys if not key.startswith("text_encoder.model.")]
        if non_clip_missing:
            raise ValueError(f"Missing non-CLIP keys when loading text-length action prior: {non_clip_missing}")

    @torch.no_grad()
    def _encode_text(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder.model is not None:
            self.text_encoder.model.to(device)
        text_tokens, text_mask, _ = self.text_encoder.encode_text(texts)
        return text_tokens.to(device)[:, : self.max_text_tokens], text_mask.to(device)[:, : self.max_text_tokens]

    def forward(self, texts: list[str], lengths: torch.Tensor) -> torch.Tensor:
        device = lengths.device
        text_tokens, text_mask = self._encode_text(texts, device)
        text_mask_f = text_mask.float().unsqueeze(-1)
        text_pooled = (text_tokens * text_mask_f).sum(dim=1) / text_mask_f.sum(dim=1).clamp_min(1.0)
        text_hidden = self.text_proj(text_pooled)
        length_value = lengths.float().view(-1, 1) / float(self.max_motion_length)
        length_hidden = self.length_proj(length_value.clamp(0.0, 1.0))
        action = self.net(torch.cat([text_hidden, length_hidden], dim=-1))
        return action.reshape(lengths.shape[0], self.latent_steps, self.action_dim)
