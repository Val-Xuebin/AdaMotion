from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from lam.modules.blocks import SpatioTemporalTransformer, SpatioTransformer
from worldmodel.vwm.models.motion_diffusion import FrozenCLIPTextEncoder


class LatentActionModel(nn.Module):
    """
    AdaWorld-style latent action VAE over joint tokens instead of image patches.
    Input shape: (B, T, J, 3)
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        use_text_condition: bool = False,
        use_timestep_condition: bool = False,
        clip_version: str = "ViT-B/32",
        max_text_tokens: int = 32,
        max_timestep: int = 512,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.use_text_condition = use_text_condition
        self.use_timestep_condition = use_timestep_condition
        self.max_text_tokens = max_text_tokens
        self.max_timestep = max_timestep
        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, in_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=in_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        self.joint_up = nn.Linear(in_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=in_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        if self.use_text_condition:
            self.text_encoder = FrozenCLIPTextEncoder(clip_version)
            self.clip_dim = 512 if clip_version == "ViT-B/32" else 768
            self.text_proj = nn.Linear(self.clip_dim, model_dim)
        else:
            self.text_encoder = None
            self.clip_dim = 0
            self.text_proj = None
        if self.use_timestep_condition:
            self.timestep_proj = nn.Embedding(max_timestep, model_dim)
        else:
            self.timestep_proj = None

    def _encode_text_condition(self, texts: list[str], device: torch.device, dtype: torch.dtype) -> Tensor:
        assert self.text_encoder is not None and self.text_proj is not None
        if self.text_encoder.model is not None:
            self.text_encoder.model.to(device)
        text_tokens, text_mask, _ = self.text_encoder.encode_text(texts)
        text_tokens = text_tokens.to(device=device, dtype=dtype)[:, : self.max_text_tokens]
        text_mask = text_mask.to(device=device)[:, : self.max_text_tokens]
        text_mask_f = text_mask.float().unsqueeze(-1)
        text_pooled = (text_tokens * text_mask_f).sum(dim=1) / text_mask_f.sum(dim=1).clamp_min(1.0)
        return self.text_proj(text_pooled)

    def _build_condition_sequence(
        self,
        batch_size: int,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        texts: list[str] | None = None,
        timestep_ids: Tensor | None = None,
        text_cond_base: Tensor | None = None,
    ) -> Tensor | None:
        cond = None
        if self.use_text_condition:
            if text_cond_base is None:
                if texts is None:
                    texts = [""] * batch_size
                text_cond_base = self._encode_text_condition(list(texts), device, dtype)
            text_cond = text_cond_base
            text_cond = text_cond[:, None, :].expand(batch_size, num_steps, self.model_dim)
            cond = text_cond if cond is None else cond + text_cond
        if self.use_timestep_condition:
            if timestep_ids is None:
                timestep_ids = torch.zeros(batch_size, num_steps, dtype=torch.long, device=device)
            elif timestep_ids.ndim == 1:
                timestep_ids = timestep_ids[:, None]
            timestep_ids = timestep_ids.to(device=device, dtype=torch.long)
            if timestep_ids.shape[1] != num_steps:
                if timestep_ids.shape[1] == 1:
                    timestep_ids = timestep_ids.expand(batch_size, num_steps)
                else:
                    raise ValueError(
                        f"Expected timestep_ids second dim {num_steps}, got {tuple(timestep_ids.shape)}"
                    )
            timestep_ids = timestep_ids.clamp(0, self.max_timestep - 1)
            time_cond = self.timestep_proj(timestep_ids)
            cond = time_cond if cond is None else cond + time_cond
        return cond

    def encode_sequence(
        self,
        joints: Tensor,
        texts: list[str] | None = None,
        timestep_ids: Tensor | None = None,
        text_cond_base: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        batch_size, time_steps = joints.shape[:2]
        action_pad = self.action_prompt.expand(batch_size, time_steps, -1, -1)
        padded_joints = torch.cat([action_pad, joints], dim=2)
        encoded = self.encoder(padded_joints)
        z_tokens = encoded[:, 1:, 0]
        cond = self._build_condition_sequence(
            batch_size=batch_size,
            num_steps=time_steps - 1,
            device=joints.device,
            dtype=z_tokens.dtype,
            texts=texts,
            timestep_ids=timestep_ids,
            text_cond_base=text_cond_base,
        )
        if cond is not None:
            z_tokens = z_tokens + cond
        z_tokens = z_tokens.reshape(batch_size * (time_steps - 1), self.model_dim)
        moments = self.fc(z_tokens)
        z_mu, z_logvar = torch.chunk(moments, 2, dim=1)
        if self.training:
            z_rep = z_mu + torch.randn_like(z_logvar) * torch.exp(0.5 * z_logvar)
        else:
            z_rep = z_mu
        z_rep = z_rep.reshape(batch_size, time_steps - 1, 1, self.latent_dim)
        return {
            "joints": joints,
            "z_rep": z_rep,
            "z_mu": z_mu.reshape(batch_size, time_steps - 1, self.latent_dim),
            "z_logvar": z_logvar.reshape(batch_size, time_steps - 1, self.latent_dim),
        }

    def forward(self, batch: Dict[str, Tensor | list[str]]) -> Dict[str, Tensor]:
        texts = batch.get("text")
        timestep_ids = batch.get("timestep_ids")
        text_cond_base = None
        if self.use_text_condition:
            if texts is None:
                texts = [""] * batch["joints"].shape[0]
            text_cond_base = self._encode_text_condition(list(texts), batch["joints"].device, batch["joints"].dtype)
        outputs = self.encode_sequence(
            batch["joints"],
            texts=texts,
            timestep_ids=timestep_ids,
            text_cond_base=text_cond_base,
        )
        joint_tokens = self.joint_up(outputs["joints"][:, :-1])
        cond = self._build_condition_sequence(
            batch_size=joint_tokens.shape[0],
            num_steps=joint_tokens.shape[1],
            device=joint_tokens.device,
            dtype=joint_tokens.dtype,
            texts=texts,
            timestep_ids=timestep_ids,
            text_cond_base=text_cond_base,
        )
        if cond is not None:
            joint_tokens = joint_tokens + cond[:, :, None, :]
        action_tokens = self.action_up(outputs["z_rep"])
        recon = self.decoder(joint_tokens + action_tokens)
        outputs["recon"] = recon
        return outputs

    def forward_momentum(
        self,
        encoder_joints: Tensor,
        decoder_joints: Tensor,
        texts: list[str] | None = None,
        timestep_ids: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        text_cond_base = None
        if self.use_text_condition:
            if texts is None:
                texts = [""] * encoder_joints.shape[0]
            text_cond_base = self._encode_text_condition(list(texts), encoder_joints.device, encoder_joints.dtype)
        outputs = self.encode_sequence(
            encoder_joints,
            texts=texts,
            timestep_ids=timestep_ids,
            text_cond_base=text_cond_base,
        )
        if decoder_joints.ndim == 3:
            decoder_joints = decoder_joints[:, None]
        joint_tokens = self.joint_up(decoder_joints)
        cond = self._build_condition_sequence(
            batch_size=joint_tokens.shape[0],
            num_steps=joint_tokens.shape[1],
            device=joint_tokens.device,
            dtype=joint_tokens.dtype,
            texts=texts,
            timestep_ids=timestep_ids,
            text_cond_base=text_cond_base,
        )
        if cond is not None:
            joint_tokens = joint_tokens + cond[:, :, None, :]
        action_tokens = self.action_up(outputs["z_rep"])
        outputs["recon"] = self.decoder(joint_tokens + action_tokens)
        return outputs
