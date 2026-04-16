# AdaMotion

AdaMotion is a focused repo for adapting AdaWorld to human motion pretraining on HumanML3D.

For side-by-side comparison, the upstream repository is available at `/work/adaworld`, and this repo is kept at
`/work/adamotion`.

The repository now mirrors AdaWorld's top-level structure as closely as possible while keeping the lightweight
HumanML3D training code:

- `lam/`: AdaWorld-style latent action model entrypoint and package for HumanML3D transitions
- `worldmodel/`: AdaWorld-style world model entrypoint and package for HumanML3D context prediction
- `configs/`: legacy flat configs kept for backward compatibility
- `scripts/`: legacy entry scripts kept for backward compatibility
- `reports/`: notes and result summaries
- `experiments/`: checkpoints and exported artifacts
- `data/HumanML3D/`: expected HumanML3D feature root, either real data or a local symlink

Current baseline:

- `Feature-MLP`: formal baseline on HumanML3D `new_joint_vecs`
- `Joint-ST`: alternative joints-based spatiotemporal-transformer line
- `WorldModel-HM`: short-horizon predictor conditioned on latent action

State representation:

- Per-frame HumanML3D joint positions of shape `[T, 22, 3]`

Primary goal of this first baseline:

- learn transition-level latent actions without text supervision
- test whether action-aware short-horizon prediction is better than action-agnostic prediction

The repository is self-contained. It does not depend on the checked-in upstream `repos/AdaWorld` copy to run these
HumanML3D experiments.

Recommended AdaWorld-style entrypoints:

```bash
python3 /work/adamotion/lam/main.py \
  --config /work/adamotion/lam/config/humanml_lam_debug.yaml
```

```bash
python3 /work/adamotion/worldmodel/train.py \
  --base /work/adamotion/worldmodel/configs/training/humanml_world_debug.yaml
```

Paper-facing MLP baseline setup:

```bash
python3 /work/adamotion/scripts/backfill_legacy_baselines.py
python3 /work/adamotion/lam/main.py \
  --config /work/adamotion/configs/humanml_feature_mlp_lam_debug.yaml
python3 /work/adamotion/worldmodel/train.py \
  --base /work/adamotion/configs/humanml_feature_mlp_world_debug.yaml
python3 /work/adamotion/scripts/aggregate_experiments.py
```

Unified run tables are written to:

- `/work/adamotion/reports/tables/adamotion_runs.csv`
- `/work/adamotion/reports/tables/adamotion_runs.md`

Backward-compatible legacy entrypoints:

```bash
python3 /work/adamotion/scripts/train_hm.py \
  --config /work/adamotion/configs/humanml_lam_debug.yaml
```

```bash
python3 /work/adamotion/scripts/train_hm.py \
  --config /work/adamotion/configs/humanml_world_debug.yaml
```
