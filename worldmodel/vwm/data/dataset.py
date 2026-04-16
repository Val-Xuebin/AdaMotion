from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_split_ids(split_file: str | Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


REPRESENTATION_DIRS = {
    "joint_positions": "new_joints",
    "humanml_feature_vector": "new_joint_vecs",
}


def _representation_dir(representation: str) -> str:
    try:
        return REPRESENTATION_DIRS[representation]
    except KeyError as exc:
        raise ValueError(f"Unsupported representation: {representation}") from exc


def _motion_path(data_root: str | Path, motion_id: str, representation: str) -> Path:
    return Path(data_root) / _representation_dir(representation) / f"{motion_id}.npy"


class HumanMLMotionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        min_frames: int = 2,
        max_sequences: int | None = None,
        representation: str = "joint_positions",
    ) -> None:
        self.data_root = Path(data_root)
        self.representation = representation
        split_ids = _read_split_ids(self.data_root / f"{split}.txt")
        if max_sequences:
            split_ids = split_ids[:max_sequences]

        self.motion_ids: List[str] = []
        self.lengths: List[int] = []
        for motion_id in split_ids:
            path = _motion_path(self.data_root, motion_id, self.representation)
            if not path.exists():
                continue
            motion = np.load(path, mmap_mode="r")
            if motion.shape[0] < min_frames:
                continue
            self.motion_ids.append(motion_id)
            self.lengths.append(int(motion.shape[0]))

        if not self.motion_ids:
            raise RuntimeError(f"No valid sequences found in {self.data_root} split={split}")

        first_motion = np.load(_motion_path(self.data_root, self.motion_ids[0], self.representation), mmap_mode="r")
        if self.representation == "joint_positions":
            self.num_joints = int(first_motion.shape[1])
            self.joint_dim = int(first_motion.shape[2])
            self.feature_dim = self.num_joints * self.joint_dim
        else:
            self.num_joints = None
            self.joint_dim = None
            self.feature_dim = int(first_motion.shape[1])

    def __len__(self) -> int:
        return len(self.motion_ids)

    def load_motion(self, index: int) -> np.ndarray:
        return np.load(_motion_path(self.data_root, self.motion_ids[index], self.representation)).astype(np.float32)


class HumanMLContextDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        context_len: int = 6,
        future_len: int = 1,
        stride: int = 1,
        max_sequences: int | None = None,
        representation: str = "joint_positions",
    ) -> None:
        self.base = HumanMLMotionDataset(
            data_root=data_root,
            split=split,
            max_sequences=max_sequences,
            representation=representation,
        )
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


__all__ = ["HumanMLMotionDataset", "HumanMLContextDataset"]
