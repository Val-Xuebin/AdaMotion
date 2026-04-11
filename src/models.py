from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LatentActionAutoencoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        beta: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.encoder = MLP([state_dim * 2, hidden_dim, hidden_dim], dropout=dropout)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLP([state_dim + latent_dim, hidden_dim, hidden_dim, state_dim], dropout=dropout)

    def encode(self, x_t: torch.Tensor, x_tp1: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(torch.cat([x_t, x_tp1], dim=-1))
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return {"z": z, "mu": mu, "logvar": logvar}

    def forward(self, x_t: torch.Tensor, x_tp1: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(x_t, x_tp1)
        recon = self.decoder(torch.cat([x_t, enc["z"]], dim=-1))
        mse = F.mse_loss(recon, x_tp1)
        kl = -0.5 * torch.mean(1 + enc["logvar"] - enc["mu"].pow(2) - enc["logvar"].exp())
        loss = mse + self.beta * kl
        return {
            **enc,
            "recon": recon,
            "mse_loss": mse,
            "kl_loss": kl,
            "loss": loss,
        }


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        state_dim: int = 263,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        context_len: int = 6,
        future_len: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_len = context_len
        self.future_len = future_len
        self.context_encoder = MLP([context_len * state_dim, hidden_dim, hidden_dim], dropout=dropout)
        self.head = MLP([hidden_dim + latent_dim, hidden_dim, future_len * state_dim], dropout=dropout)

    def forward(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        bsz = context.shape[0]
        ctx_h = self.context_encoder(context.reshape(bsz, -1))
        out = self.head(torch.cat([ctx_h, z], dim=-1))
        return out.reshape(bsz, self.future_len, self.state_dim)


class ActionAgnosticPredictor(nn.Module):
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        context_len: int = 6,
        future_len: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.future_len = future_len
        self.net = MLP([context_len * state_dim, hidden_dim, hidden_dim, future_len * state_dim], dropout=dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        bsz = context.shape[0]
        out = self.net(context.reshape(bsz, -1))
        return out.reshape(bsz, self.future_len, self.state_dim)
