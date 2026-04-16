# Paper Experiment Setup

## Scope

- Keep the current `adamotion` structure centered on `lam/`, `worldmodel/`, and `scripts/`
- Treat the previous `new_joint_vecs + MLP` results as a legacy baseline
- Use `humanml_feature_mlp_*` as the paper-facing formal MLP baseline naming

## Run Organization

- Legacy baseline outputs remain under `/work/adamotion/experiments/humanml_*`
- Paper-facing runs should use `/work/adamotion/experiments/paper/*`
- Each completed run now writes:
  - `summary.json`
  - `history.json` or the stage-specific history file already produced by training
  - `history.csv`
  - `curves.svg`
  - `curves.png`

## Aggregated Tracking

- Central run table:
  - `/work/adamotion/reports/tables/adamotion_runs.csv`
  - `/work/adamotion/reports/tables/adamotion_runs.md`
- Refresh command:

```bash
python3 /work/adamotion/scripts/aggregate_experiments.py
```

- Backfill legacy baseline summaries:

```bash
python3 /work/adamotion/scripts/backfill_legacy_baselines.py
```

## Recommended Commands

LAM debug:

```bash
python3 /work/adamotion/lam/main.py \
  --config /work/adamotion/configs/humanml_feature_mlp_lam_debug.yaml
```

World-model debug:

```bash
python3 /work/adamotion/worldmodel/train.py \
  --base /work/adamotion/configs/humanml_feature_mlp_world_debug.yaml
```

LAM full:

```bash
python3 /work/adamotion/lam/main.py \
  --config /work/adamotion/configs/humanml_feature_mlp_lam_full.yaml
```

World-model full:

```bash
python3 /work/adamotion/worldmodel/train.py \
  --base /work/adamotion/configs/humanml_feature_mlp_world_full.yaml
```
