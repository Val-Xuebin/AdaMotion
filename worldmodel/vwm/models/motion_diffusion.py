from __future__ import annotations

import hashlib
import os
import sys

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


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

    def _hash_encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(texts)
        max_tokens = self.max_length
        device = torch.device("cpu")
        word_emb = torch.zeros(batch_size, max_tokens, self.clip_dim, dtype=torch.float32, device=device)
        attn_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
        eos_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        for batch_idx, sentence in enumerate(texts):
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
    def encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backend == "hash":
            return self._hash_encode_text(texts)
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        text_input_ids = tokens.input_ids.to(self.model.device)
        text_attn_mask = tokens.attention_mask.to(self.model.device).bool()
        word_emb = self.model.text_model(text_input_ids).last_hidden_state
        return word_emb, text_attn_mask, text_input_ids.argmax(dim=-1)
