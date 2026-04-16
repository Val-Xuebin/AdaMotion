from __future__ import annotations

from math import pi
from typing import Literal, Union

import torch
from einops import rearrange, repeat
from torch import Tensor, einsum, nn
from torch.amp import autocast


def exists(val) -> bool:
    return val is not None


def default(val, d):
    return val if exists(val) else d


def rotate_half(x) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if rot_dim > t.shape[-1]:
        raise ValueError(f"Rotary dim {rot_dim} exceeds tensor dim {t.shape[-1]}")

    left = t[..., :start_index]
    middle = t[..., start_index:end_index]
    right = t[..., end_index:]
    transformed = (middle * freqs.cos() * scale) + (rotate_half(middle) * freqs.sin() * scale)
    return torch.cat([left, transformed, right], dim=-1).type(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        custom_freqs: Union[Tensor, None] = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta: int = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        learned_freq: bool = False,
        cache_if_possible: bool = True,
        cache_max_seq_len: int = 8192,
    ) -> None:
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float()[: (dim // 2)] / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"Unsupported freqs_for={freqs_for}")

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len
        self.learned_freq = learned_freq
        self.register_buffer("cached_freqs", torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.register_buffer("cached_freqs_seq_len", torch.tensor(0), persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return torch.arange(seq_len, device=device, dtype=dtype) + offset

    def rotate_queries_or_keys(self, t, freqs, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, -2)
        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]
        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)
        seq_freqs = self.forward(seq, freqs, seq_len=seq_len, offset=offset)
        return apply_rotary_emb(seq_freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, freqs: Tensor, seq_len=None, offset=0):
        should_cache = self.cache_if_possible and not self.learned_freq and exists(seq_len) and (offset + seq_len) <= self.cache_max_seq_len
        if should_cache and (offset + seq_len) <= self.cached_freqs_seq_len.item():
            return self.cached_freqs[offset : offset + seq_len].detach()

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.copy_(torch.tensor(seq_len, device=self.cached_freqs_seq_len.device))
        return freqs
