#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def iter_files(src_dir: Path, suffix: str):
    for path in sorted(src_dir.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        yield path


def symlink_dir(src_dir: Path, dst_dir: Path, suffix: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    skipped = 0
    for src in iter_files(src_dir, suffix):
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == src.resolve():
                skipped += 1
                continue
            dst.unlink()
        os.symlink(src, dst)
        linked += 1
    return linked, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/workspace/dataset/HumanML3D")
    parser.add_argument("--view-root", default="/workspace/benchmarks/humanmodels/data_views/HumanML3D_flat")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    view_root = Path(args.view_root).resolve()
    view_root.mkdir(parents=True, exist_ok=True)

    for name in [
        "train.txt",
        "val.txt",
        "test.txt",
        "train_val.txt",
        "all.txt",
        "train_usable.txt",
        "val_usable.txt",
        "test_usable.txt",
        "usable_split_summary.json",
        "Mean.npy",
        "Std.npy",
    ]:
        src = source_root / name
        if src.exists():
            dst = view_root / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)

    stats = {}
    for subdir, suffix in [
        ("new_joint_vecs", ".npy"),
        ("texts", ".txt"),
        ("new_joints", ".npy"),
    ]:
        linked, skipped = symlink_dir(source_root / subdir, view_root / subdir, suffix)
        stats[subdir] = {"linked": linked, "skipped": skipped}

    for key, value in stats.items():
        print(f"{key}: linked={value['linked']} skipped={value['skipped']}")
    print(f"flat view ready at {view_root}")


if __name__ == "__main__":
    main()
