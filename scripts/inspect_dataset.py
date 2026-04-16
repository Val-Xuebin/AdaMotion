#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lam"))
sys.path.insert(0, str(ROOT / "worldmodel"))

from lam.dataset import HumanMLMotionDataset, HumanMLTransitionDataset
from vwm.data.dataset import HumanMLContextDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/work/adamotion/data/HumanML3D")
    parser.add_argument("--split", default="train")
    parser.add_argument("--representation", default="joint_positions", choices=["joint_positions", "humanml_feature_vector"])
    args = parser.parse_args()

    seq = HumanMLMotionDataset(args.data_root, split=args.split, representation=args.representation)
    trans = HumanMLTransitionDataset(args.data_root, split=args.split, max_transitions_per_sequence=8, representation=args.representation)
    ctx = HumanMLContextDataset(args.data_root, split=args.split, context_len=6, future_len=1, stride=4, representation=args.representation)
    out = {
        "representation": args.representation,
        "sequence_stats": seq.stats().__dict__,
        "transition_examples": len(trans),
        "context_examples": len(ctx),
        "sample_transition_shape": list(trans[0]["x_t"].shape),
        "sample_context_shape": list(ctx[0]["context"].shape),
        "sample_target_shape": list(ctx[0]["target"].shape),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
