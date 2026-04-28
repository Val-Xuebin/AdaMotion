from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from common.humanml_representation import humanml_vector_to_sal_rep, resolve_motion_path, resolve_text_path


def _read_split_ids(split_file: str | Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("._")]


def _motion_path(data_root: str | Path, motion_id: str) -> Path:
    return resolve_motion_path(data_root, "new_joint_vecs", motion_id)


def _text_path(data_root: str | Path, motion_id: str) -> Path:
    return resolve_text_path(data_root, motion_id)


def _read_captions(path: str | Path) -> List[str]:
    captions: List[str] = []
    text_path = Path(path)
    if not text_path.exists():
        return captions
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "\x00" in line:
                continue
            if "#" in line:
                caption = line.split("#", 1)[0].strip()
            else:
                caption = line.strip()
            if caption:
                captions.append(caption)
    return captions


class HumanMLTextFutureDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        context_len: int = 8,
        future_len: int = 8,
        stride: int = 4,
        max_sequences: int | None = None,
        random_caption: bool = True,
    ) -> None:
        if context_len % 4 != 0 or future_len % 4 != 0:
            raise ValueError("context_len and future_len must be divisible by 4 for motion-latent diffusion.")
        self.data_root = Path(data_root)
        self.context_len = context_len
        self.future_len = future_len
        self.stride = stride
        self.random_caption = random_caption
        split_ids = _read_split_ids(self.data_root / f"{split}.txt")
        if max_sequences:
            split_ids = split_ids[:max_sequences]

        self.motion_ids: List[str] = []
        self.lengths: List[int] = []
        self.captions: List[List[str]] = []
        self.index: List[Tuple[int, int]] = []
        required = context_len + future_len

        for motion_id in split_ids:
            motion_path = _motion_path(self.data_root, motion_id)
            if not motion_path.exists():
                continue
            motion = np.load(motion_path, mmap_mode="r")
            if motion.shape[0] < required:
                continue
            seq_idx = len(self.motion_ids)
            self.motion_ids.append(motion_id)
            self.lengths.append(int(motion.shape[0]))
            self.captions.append(_read_captions(_text_path(self.data_root, motion_id)))
            last_start = motion.shape[0] - required
            for start in range(0, max(0, last_start) + 1, stride):
                self.index.append((seq_idx, start))

        if not self.motion_ids:
            raise RuntimeError(f"No valid text-motion sequences found in {self.data_root} split={split}")

        first_motion = np.load(_motion_path(self.data_root, self.motion_ids[0]), mmap_mode="r")
        self.feature_dim = int(first_motion.shape[1])
        self.num_joints = 22
        self.sal_joint_dim = 13

    def __len__(self) -> int:
        return len(self.index)

    def load_motion(self, seq_idx: int) -> np.ndarray:
        return np.load(_motion_path(self.data_root, self.motion_ids[seq_idx])).astype(np.float32)

    def _select_caption(self, seq_idx: int) -> str:
        candidates = self.captions[seq_idx]
        if not candidates:
            return ""
        if self.random_caption:
            return random.choice(candidates)
        return candidates[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        seq_idx, start = self.index[index]
        motion = self.load_motion(seq_idx)
        context = motion[start : start + self.context_len]
        future = motion[start + self.context_len : start + self.context_len + self.future_len]
        action_source = motion[start + self.context_len - 1 : start + self.context_len + self.future_len]
        action_source = humanml_vector_to_sal_rep(action_source)
        return {
            "text": self._select_caption(seq_idx),
            "context_motion": torch.from_numpy(context),
            "future_motion": torch.from_numpy(future),
            "action_source": torch.from_numpy(action_source),
            "motion_id": self.motion_ids[seq_idx],
            "start_idx": torch.tensor(start, dtype=torch.long),
        }
