from __future__ import annotations

import codecs as cs
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from common.humanml_representation import humanml_vector_to_sal_rep, resolve_motion_path, resolve_text_path


def _read_split_ids(split_file: str | Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("._")]


def _read_caption_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    text_path = Path(path)
    if not text_path.exists():
        return records
    with cs.open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            try:
                f_tag = 0.0 if np.isnan(float(parts[2])) else float(parts[2])
                to_tag = 0.0 if np.isnan(float(parts[3])) else float(parts[3])
            except ValueError:
                continue
            records.append({"caption": parts[0], "tokens": parts[1].split(" "), "f_tag": f_tag, "to_tag": to_tag})
    return records


class HumanMLFullMotionActionDataset(Dataset):
    """SALAD-style full-motion text dataset with LAM action sources."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train_usable",
        max_motion_length: int = 196,
        unit_length: int = 4,
        min_motion_len: int = 40,
        max_sequences: int | None = None,
        random_caption: bool = True,
        mean_path: str | Path | None = None,
        std_path: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.random_caption = random_caption
        self.mean = np.load(mean_path or self.data_root / "Mean.npy").astype(np.float32)
        self.std = np.load(std_path or self.data_root / "Std.npy").astype(np.float32)

        ids = _read_split_ids(self.data_root / f"{split}.txt")
        if max_sequences:
            ids = ids[:max_sequences]
        self.samples: list[dict] = []
        for motion_id in ids:
            motion_path = resolve_motion_path(self.data_root, "new_joint_vecs", motion_id)
            text_path = resolve_text_path(self.data_root, motion_id)
            if not motion_path.exists() or not text_path.exists():
                continue
            try:
                motion = np.load(motion_path, mmap_mode="r")
            except Exception:
                continue
            if len(motion) < min_motion_len or len(motion) >= 200:
                continue
            records = _read_caption_records(text_path)
            whole_records = [record for record in records if record["f_tag"] == 0.0 and record["to_tag"] == 0.0]
            if not whole_records:
                continue
            self.samples.append(
                {
                    "motion_id": motion_id,
                    "motion_path": str(motion_path),
                    "length": int(len(motion)),
                    "texts": whole_records,
                }
            )
        if not self.samples:
            raise RuntimeError(f"No valid full-motion text samples found in {self.data_root} split={split}")

    def __len__(self) -> int:
        return len(self.samples)

    def _select_text(self, sample: dict) -> dict:
        if self.random_caption:
            return random.choice(sample["texts"])
        return sample["texts"][0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        motion = np.load(sample["motion_path"]).astype(np.float32)
        text_record = self._select_text(sample)
        m_length = int(sample["length"])
        if self.unit_length < 10:
            coin = np.random.choice(["single", "single", "double"])
        else:
            coin = "single"
        if coin == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        m_length = max(self.unit_length, min(m_length, self.max_motion_length))
        start = random.randint(0, max(0, len(motion) - m_length))
        motion = motion[start : start + m_length]
        norm_motion = (motion - self.mean) / self.std
        if m_length < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - m_length, motion.shape[1]), dtype=np.float32)
            norm_motion = np.concatenate([norm_motion, pad], axis=0)
            raw_motion = np.concatenate([motion, pad], axis=0)
        else:
            raw_motion = motion
        action_source = humanml_vector_to_sal_rep(raw_motion[:m_length])
        if m_length < self.max_motion_length:
            action_pad = np.zeros((self.max_motion_length - m_length, *action_source.shape[1:]), dtype=np.float32)
            action_source = np.concatenate([action_source, action_pad], axis=0)
        return {
            "text": text_record["caption"],
            "motion": torch.from_numpy(norm_motion.astype(np.float32)),
            "action_source": torch.from_numpy(action_source.astype(np.float32)),
            "length": torch.tensor(m_length, dtype=torch.long),
            "motion_id": sample["motion_id"],
            "start_idx": torch.tensor(start, dtype=torch.long),
        }
