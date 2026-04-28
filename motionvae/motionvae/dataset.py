from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from common.humanml_representation import resolve_motion_path


def _read_split_ids(split_file: str | Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("._")]


def _motion_path(data_root: str | Path, motion_id: str) -> Path:
    return resolve_motion_path(data_root, "new_joint_vecs", motion_id)


@dataclass
class AutoencoderDatasetStats:
    num_sequences: int
    num_windows: int
    window_len: int
    feature_dim: int


class HumanMLMotionAutoencoderDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        window_size: int = 16,
        stride: int = 4,
        min_frames: int | None = None,
        max_sequences: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.stride = stride
        self.min_frames = max(window_size, min_frames or window_size)
        split_ids = _read_split_ids(self.data_root / f"{split}.txt")
        if max_sequences:
            split_ids = split_ids[:max_sequences]

        self.motion_ids: List[str] = []
        self.lengths: List[int] = []
        self.index: List[Tuple[int, int]] = []

        for motion_id in split_ids:
            path = _motion_path(self.data_root, motion_id)
            if not path.exists():
                continue
            motion = np.load(path, mmap_mode="r")
            if motion.shape[0] < self.min_frames:
                continue
            seq_idx = len(self.motion_ids)
            self.motion_ids.append(motion_id)
            self.lengths.append(int(motion.shape[0]))
            last_start = motion.shape[0] - self.window_size
            for start in range(0, max(0, last_start) + 1, self.stride):
                self.index.append((seq_idx, start))

        if not self.motion_ids:
            raise RuntimeError(f"No valid HumanML feature sequences found in {self.data_root} split={split}")

        first_motion = np.load(_motion_path(self.data_root, self.motion_ids[0]), mmap_mode="r")
        self.feature_dim = int(first_motion.shape[1])

    def __len__(self) -> int:
        return len(self.index)

    def load_motion(self, seq_idx: int) -> np.ndarray:
        return np.load(_motion_path(self.data_root, self.motion_ids[seq_idx])).astype(np.float32)

    def stats(self) -> AutoencoderDatasetStats:
        return AutoencoderDatasetStats(
            num_sequences=len(self.motion_ids),
            num_windows=len(self.index),
            window_len=self.window_size,
            feature_dim=self.feature_dim,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_idx, start = self.index[index]
        motion = self.load_motion(seq_idx)
        clip = motion[start : start + self.window_size]
        return {
            "motion": torch.from_numpy(clip),
            "length": torch.tensor(clip.shape[0], dtype=torch.long),
            "motion_id": self.motion_ids[seq_idx],
            "start_idx": torch.tensor(start, dtype=torch.long),
        }
