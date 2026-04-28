#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.humanml_representation import resolve_motion_path, resolve_text_path


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("._")]


def _write_ids(path: Path, motion_ids: list[str]) -> None:
    path.write_text("\n".join(motion_ids) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/assets/dataset/HumanML3D")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    summary = {}
    usable_splits: dict[str, list[str]] = {}

    for split in ["train", "val", "test"]:
        split_ids = _read_ids(data_root / f"{split}.txt")
        usable = [
            motion_id
            for motion_id in split_ids
            if resolve_text_path(data_root, motion_id).exists()
            and resolve_motion_path(data_root, "new_joint_vecs", motion_id).exists()
        ]
        usable_splits[split] = usable
        _write_ids(data_root / f"{split}_usable.txt", usable)
        summary[split] = {"total": len(split_ids), "usable": len(usable)}

    train_val = usable_splits["train"] + usable_splits["val"]
    all_usable = train_val + usable_splits["test"]
    _write_ids(data_root / "train_val_usable.txt", train_val)
    _write_ids(data_root / "all_usable.txt", all_usable)

    payload = {
        "data_root": str(data_root),
        "splits": summary,
        "train_val_usable": len(train_val),
        "all_usable": len(all_usable),
    }
    (data_root / "usable_split_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
