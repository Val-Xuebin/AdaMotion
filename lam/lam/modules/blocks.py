from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import Tensor, nn

from lam.modules.embeddings import RotaryEmbedding


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[: x.shape[2]].to(device=x.device, dtype=x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super().__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(dropout))
        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim)

    def scaled_dot_product_attention(self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False) -> Tensor:
        q_len, k_len = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(q_len, k_len, dtype=query.dtype, device=query.device)
        if is_causal:
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(~mask, float("-inf"))
        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight = torch.softmax(attn_weight + attn_bias, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = q.contiguous(), k.contiguous()
        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        t_len = x.shape[1]
        x = rearrange(x, "b t s e -> (b t) s e")
        x = x + self.spatial_attn(self.norm1(x))
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)
        x = x + self.ffn(self.norm2(x))
        return x


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout, rot_emb=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False) -> Tensor:
        t_len, s_len = x.shape[1:3]
        x = rearrange(x, "b t s e -> (b t) s e")
        x = x + self.spatial_attn(self.norm1(x))
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)
        x = rearrange(x, "b t s e -> (b s) t e")
        x = x + self.temporal_attn(self.norm2(x), is_causal=causal_temporal)
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)
        x = x + self.ffn(self.norm3(x))
        return x


class SpatioTransformer(nn.Module):
    def __init__(self, in_dim: int, model_dim: int, out_dim: int, num_blocks: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim))
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList([SpatioBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_enc(self.ffn(x))
        for block in self.transformer_blocks:
            x = block(x)
        return self.out(x)


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = False,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim))
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList([SpatioTemporalBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.out = nn.Linear(model_dim, out_dim)
        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_enc(self.ffn(x))
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal)
        return self.out(x)
