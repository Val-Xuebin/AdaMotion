# Initial Results

This document records the previous legacy baseline, not the current paper-facing `new_joints + spatiotemporal transformer`
setup. The runs below use the earlier `new_joint_vecs + MLP` pipeline and should be treated as the baseline-v1 reference.

## Workspace

- Upstream repo available at `/work/adaworld`
- HumanML3D linked to `/work/adamotion/data/HumanML3D`
- Human-motion baseline code organized under `/work/adamotion/lam` and `/work/adamotion/worldmodel`

## Dataset Sanity Check

Command:

```bash
python3 /work/adamotion/scripts/inspect_dataset.py
```

Observed:

- train sequences: `23374`
- total train frames: `3293362`
- mean length: `140.90`
- feature dim: `263`
- transition sample shape: `[263]`
- context sample shape: `[6, 263]`
- target sample shape: `[1, 263]`

## LAM-HM Debug Run

Command:

```bash
python3 /work/adamotion/scripts/train_hm.py \
  --config /work/adamotion/configs/humanml_lam_debug.yaml
```

Best validation summary:

- epoch: `1`
- train loss: `0.02871`
- val loss: `0.02824`
- val mse: `0.02784`
- val kl: `0.39899`

Artifact:

- `/work/adamotion/experiments/humanml_lam_debug/lam_best.pt`

Interpretation:

- the latent-action autoencoder trains stably on HumanML3D feature transitions
- the latent posterior does not collapse immediately
- this is sufficient for the first action-aware short-horizon prediction test

## WorldModel-HM Debug Run

Command:

```bash
python3 /work/adamotion/scripts/train_hm.py \
  --config /work/adamotion/configs/humanml_world_debug.yaml
```

Best validation summary:

- epoch: `1`
- action-conditioned loss: `0.01414`
- no-action loss: `0.01243`
- val gain: `-0.00170`

Artifact:

- `/work/adamotion/experiments/humanml_world_debug/world_best.pt`

Interpretation:

- the full training chain runs end-to-end
- the current debug setting is not enough to show an advantage for latent actions
- this is expected under a tiny budget and a very shallow baseline
- the next work should focus on longer training, stronger temporal encoders, and better action extraction than a pure two-frame transition bottleneck

## Immediate Next Steps

1. Run `humanml_lam_full.yaml` and `humanml_world_full.yaml`.
2. Add latent-space diagnostics: variance, UMAP/PCA, nearest-neighbor transfer.
3. Replace the flat MLP context encoder with a temporal transformer encoder.
4. Add a stronger comparison against direct-difference conditioning.
