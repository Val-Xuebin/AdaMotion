from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from lam.modules.blocks import SpatioTemporalTransformer, SpatioTransformer


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
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
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

    def encode_sequence(self, joints: Tensor) -> Dict[str, Tensor]:
        batch_size, time_steps = joints.shape[:2]
        action_pad = self.action_prompt.expand(batch_size, time_steps, -1, -1)
        padded_joints = torch.cat([action_pad, joints], dim=2)
        encoded = self.encoder(padded_joints)
        z_tokens = encoded[:, 1:, 0]
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

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = self.encode_sequence(batch["joints"])
        joint_tokens = self.joint_up(outputs["joints"][:, :-1])
        action_tokens = self.action_up(outputs["z_rep"])
        recon = self.decoder(joint_tokens + action_tokens)
        outputs["recon"] = recon
        return outputs
