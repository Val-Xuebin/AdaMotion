from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_split_ids(split_file: str | Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _motion_path(data_root: str | Path, motion_id: str) -> Path:
    return Path(data_root) / "new_joint_vecs" / f"{motion_id}.npy"


@dataclass
class DatasetStats:
    num_sequences: int
    num_frames: int
    mean_length: float
    feature_dim: int


class HumanMLMotionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        min_frames: int = 2,
        max_sequences: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        split_ids = _read_split_ids(self.data_root / f"{split}.txt")
        if max_sequences:
            split_ids = split_ids[:max_sequences]
        self.motion_ids = []
        self.lengths = []
        for motion_id in split_ids:
            path = _motion_path(self.data_root, motion_id)
            if not path.exists():
                continue
            arr = np.load(path, mmap_mode="r")
            if arr.shape[0] < min_frames:
                continue
            self.motion_ids.append(motion_id)
            self.lengths.append(int(arr.shape[0]))

        if not self.motion_ids:
            raise RuntimeError(f"No valid sequences found in {self.data_root} split={split}")
        self.feature_dim = int(np.load(_motion_path(self.data_root, self.motion_ids[0]), mmap_mode="r").shape[1])

    def __len__(self) -> int:
        return len(self.motion_ids)

    def load_motion(self, index: int) -> np.ndarray:
        return np.load(_motion_path(self.data_root, self.motion_ids[index])).astype(np.float32)

    def stats(self) -> DatasetStats:
        return DatasetStats(
            num_sequences=len(self.motion_ids),
            num_frames=sum(self.lengths),
            mean_length=float(np.mean(self.lengths)),
            feature_dim=self.feature_dim,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        motion = self.load_motion(index)
        return {
            "motion": torch.from_numpy(motion),
            "length": torch.tensor(motion.shape[0], dtype=torch.long),
            "motion_id": self.motion_ids[index],
        }


class MotionTransitionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        max_transitions_per_sequence: int | None = None,
        max_sequences: int | None = None,
    ) -> None:
        base = HumanMLMotionDataset(data_root=data_root, split=split, max_sequences=max_sequences)
        self.data_root = base.data_root
        self.motion_ids = base.motion_ids
        self.feature_dim = base.feature_dim
        self.index: List[Tuple[int, int]] = []

        for seq_idx, seq_len in enumerate(base.lengths):
            starts = list(range(seq_len - 1))
            if max_transitions_per_sequence is not None:
                starts = starts[:max_transitions_per_sequence]
            self.index.extend((seq_idx, t) for t in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_idx, t = self.index[index]
        motion = np.load(_motion_path(self.data_root, self.motion_ids[seq_idx])).astype(np.float32)
        x_t = motion[t]
        x_tp1 = motion[t + 1]
        return {
            "x_t": torch.from_numpy(x_t),
            "x_tp1": torch.from_numpy(x_tp1),
            "delta": torch.from_numpy(x_tp1 - x_t),
            "motion_id": self.motion_ids[seq_idx],
            "frame_idx": torch.tensor(t, dtype=torch.long),
        }


class MotionContextDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        context_len: int = 6,
        future_len: int = 1,
        stride: int = 1,
        max_sequences: int | None = None,
    ) -> None:
        self.base = HumanMLMotionDataset(data_root=data_root, split=split, max_sequences=max_sequences)
        self.context_len = context_len
        self.future_len = future_len
        self.stride = stride
        self.index: List[Tuple[int, int]] = []
        required = context_len + future_len
        for seq_idx, seq_len in enumerate(self.base.lengths):
            last_start = seq_len - required
            for start in range(0, max(0, last_start) + 1, stride):
                self.index.append((seq_idx, start))

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_idx, start = self.index[index]
        motion = self.base.load_motion(seq_idx)
        context = motion[start : start + self.context_len]
        target = motion[start + self.context_len : start + self.context_len + self.future_len]
        x_prev = motion[start + self.context_len - 1]
        x_next = target[0]
        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
            "x_prev": torch.from_numpy(x_prev),
            "x_next": torch.from_numpy(x_next),
            "motion_id": self.base.motion_ids[seq_idx],
            "start_idx": torch.tensor(start, dtype=torch.long),
        }
