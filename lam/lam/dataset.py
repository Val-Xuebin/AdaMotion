from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from common.humanml_representation import (
    SAL_REP_JOINT_DIM,
    HUMANML_NUM_JOINTS,
    humanml_vector_to_sal_rep,
    resolve_motion_path,
    resolve_text_path,
)


def _read_split_ids(split_file: str | Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


REPRESENTATION_DIRS = {
    "joint_positions": "new_joints",
    "humanml_feature_vector": "new_joint_vecs",
    "sal_rep": "new_joint_vecs",
}


def _representation_dir(representation: str) -> str:
    try:
        return REPRESENTATION_DIRS[representation]
    except KeyError as exc:
        raise ValueError(f"Unsupported representation: {representation}") from exc


def _motion_path(data_root: str | Path, motion_id: str, representation: str) -> Path:
    return resolve_motion_path(data_root, _representation_dir(representation), motion_id)


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
        elif self.representation == "sal_rep":
            self.num_joints = HUMANML_NUM_JOINTS
            self.joint_dim = SAL_REP_JOINT_DIM
            self.feature_dim = self.num_joints * self.joint_dim
        else:
            self.num_joints = None
            self.joint_dim = None
            self.feature_dim = int(first_motion.shape[1])

    def __len__(self) -> int:
        return len(self.motion_ids)

    def load_motion(self, index: int) -> np.ndarray:
        motion = np.load(_motion_path(self.data_root, self.motion_ids[index], self.representation)).astype(np.float32)
        if self.representation == "sal_rep":
            motion = humanml_vector_to_sal_rep(motion)
        return motion

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


class HumanMLTransitionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        num_frames: int = 3,
        stride: int = 1,
        max_transitions_per_sequence: int | None = None,
        max_sequences: int | None = None,
        representation: str = "joint_positions",
        random_caption: bool = True,
        cache_motion: bool = False,
        max_cached_sequences: int | None = None,
    ) -> None:
        base = HumanMLMotionDataset(
            data_root=data_root,
            split=split,
            min_frames=num_frames,
            max_sequences=max_sequences,
            representation=representation,
        )
        self.data_root = base.data_root
        self.representation = base.representation
        self.motion_ids = base.motion_ids
        self.feature_dim = base.feature_dim
        self.num_joints = base.num_joints
        self.joint_dim = base.joint_dim
        self.num_frames = num_frames
        self.random_caption = random_caption
        self.cache_motion = cache_motion
        self.max_cached_sequences = max_cached_sequences
        self.captions = [_read_captions(resolve_text_path(self.data_root, motion_id)) for motion_id in self.motion_ids]
        self.index: List[Tuple[int, int]] = []
        self._motion_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        for seq_idx, seq_len in enumerate(base.lengths):
            last_start = seq_len - num_frames
            starts = list(range(0, max(0, last_start) + 1, stride))
            if max_transitions_per_sequence is not None:
                starts = starts[:max_transitions_per_sequence]
            self.index.extend((seq_idx, frame_idx) for frame_idx in starts)

    def __len__(self) -> int:
        return len(self.index)

    def _select_caption(self, seq_idx: int) -> str:
        candidates = self.captions[seq_idx]
        if not candidates:
            return ""
        if self.random_caption:
            return random.choice(candidates)
        return candidates[0]

    def _load_sequence_motion(self, seq_idx: int) -> np.ndarray:
        if self.cache_motion:
            cached = self._motion_cache.get(seq_idx)
            if cached is not None:
                self._motion_cache.move_to_end(seq_idx)
                return cached

        motion = np.load(_motion_path(self.data_root, self.motion_ids[seq_idx], self.representation)).astype(np.float32)
        if self.representation == "sal_rep":
            motion = humanml_vector_to_sal_rep(motion)

        if self.cache_motion:
            self._motion_cache[seq_idx] = motion
            if self.max_cached_sequences is not None:
                while len(self._motion_cache) > self.max_cached_sequences:
                    self._motion_cache.popitem(last=False)
        return motion

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_idx, frame_idx = self.index[index]
        motion = self._load_sequence_motion(seq_idx)
        clip = motion[frame_idx : frame_idx + self.num_frames]
        x_t = clip[0]
        x_tp1 = clip[1]
        batch = {
            "x_t": torch.from_numpy(x_t),
            "x_tp1": torch.from_numpy(x_tp1),
            "delta": torch.from_numpy(x_tp1 - x_t),
            "text": self._select_caption(seq_idx),
            "motion_id": self.motion_ids[seq_idx],
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }
        if clip.shape[0] > 2:
            batch["x_tp2"] = torch.from_numpy(clip[2])
        if self.representation in {"joint_positions", "sal_rep"}:
            batch["joints"] = torch.from_numpy(clip)
        else:
            batch["motion_clip"] = torch.from_numpy(clip)
        return batch


__all__ = ["DatasetStats", "HumanMLMotionDataset", "HumanMLTransitionDataset"]
