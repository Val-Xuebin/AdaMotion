from __future__ import annotations

from pathlib import Path

import numpy as np

HUMANML_NUM_JOINTS = 22
HUMANML_FEATURE_DIM = 263
SAL_REP_JOINT_DIM = 13
SAL_REP_CONTACT_JOINTS = [7, 10, 8, 11]


def resolve_motion_path(data_root: str | Path, subdir: str, motion_id: str) -> Path:
    base = Path(data_root) / subdir
    candidates = [base / f"{motion_id}.npy", *(base / shard / f"{motion_id}.npy" for shard in ("00", "01", "02"))]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_text_path(data_root: str | Path, motion_id: str) -> Path:
    base = Path(data_root) / "texts"
    candidates = [base / f"{motion_id}.txt", *(base / shard / f"{motion_id}.txt" for shard in ("00", "01", "02"))]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def humanml_vector_to_sal_rep(motion: np.ndarray) -> np.ndarray:
    if motion.shape[-1] != HUMANML_FEATURE_DIM:
        raise ValueError(f"Expected HumanML feature dim {HUMANML_FEATURE_DIM}, got {motion.shape[-1]}")

    root, ric, rot, vel, contact = np.split(
        motion,
        [4, 4 + 3 * (HUMANML_NUM_JOINTS - 1), 4 + 9 * (HUMANML_NUM_JOINTS - 1), 4 + 9 * (HUMANML_NUM_JOINTS - 1) + 3 * HUMANML_NUM_JOINTS],
        axis=-1,
    )
    ric = ric.reshape(*motion.shape[:-1], HUMANML_NUM_JOINTS - 1, 3)
    rot = rot.reshape(*motion.shape[:-1], HUMANML_NUM_JOINTS - 1, 6)
    vel = vel.reshape(*motion.shape[:-1], HUMANML_NUM_JOINTS, 3)

    sal_rep = np.zeros((*motion.shape[:-1], HUMANML_NUM_JOINTS, SAL_REP_JOINT_DIM), dtype=motion.dtype)
    sal_rep[..., 0, :4] = root
    sal_rep[..., 0, 4:7] = vel[..., 0, :]
    for joint_idx in range(1, HUMANML_NUM_JOINTS):
        sal_rep[..., joint_idx, :3] = ric[..., joint_idx - 1, :]
        sal_rep[..., joint_idx, 3:9] = rot[..., joint_idx - 1, :]
        sal_rep[..., joint_idx, 9:12] = vel[..., joint_idx, :]
    for contact_idx, joint_idx in enumerate(SAL_REP_CONTACT_JOINTS):
        sal_rep[..., joint_idx, 12] = contact[..., contact_idx]
    return sal_rep


def sal_rep_to_humanml_vector(sal_rep: np.ndarray) -> np.ndarray:
    if sal_rep.shape[-2:] != (HUMANML_NUM_JOINTS, SAL_REP_JOINT_DIM):
        raise ValueError(
            f"Expected sal_rep shape (..., {HUMANML_NUM_JOINTS}, {SAL_REP_JOINT_DIM}), got {sal_rep.shape}"
        )

    root = sal_rep[..., 0, :4]
    root_vel = sal_rep[..., 0, 4:7]
    ric = sal_rep[..., 1:, :3].reshape(*sal_rep.shape[:-2], (HUMANML_NUM_JOINTS - 1) * 3)
    rot = sal_rep[..., 1:, 3:9].reshape(*sal_rep.shape[:-2], (HUMANML_NUM_JOINTS - 1) * 6)
    vel_rest = sal_rep[..., 1:, 9:12].reshape(*sal_rep.shape[:-2], (HUMANML_NUM_JOINTS - 1) * 3)
    vel = np.concatenate([root_vel, vel_rest], axis=-1)
    contact = np.stack([sal_rep[..., joint_idx, 12] for joint_idx in SAL_REP_CONTACT_JOINTS], axis=-1)
    return np.concatenate([root, ric, rot, vel, contact], axis=-1)
