# AdaMotion

AdaMotion is a focused repo for human motion pretraining on HumanML3D.

Layout:

- `src/`: core training code
- `configs/`: experiment configs
- `scripts/`: entry scripts
- `reports/`: notes and result summaries
- `experiments/`: checkpoints and exported artifacts
- `data/HumanML3D/`: expected HumanML3D feature root, either real data or a local symlink

Current baseline:

- `LAM-HM`: latent action autoencoder on HumanML3D `new_joint_vecs`
- `WorldModel-HM`: short-horizon predictor conditioned on latent action

State representation:

- Per-frame HumanML3D features of shape `[T, 263]`

Primary goal of this first baseline:

- learn transition-level latent actions without text supervision
- test whether action-aware short-horizon prediction is better than action-agnostic prediction

The repository is self-contained. It does not depend on a checked-in `AdaWorld` repo to run these scripts.

Quick start:

```bash
python3 /work/adamotion/scripts/train_hm.py \
  --config /work/adamotion/configs/humanml_lam_debug.yaml
```

```bash
python3 /work/adamotion/scripts/train_hm.py \
  --config /work/adamotion/configs/humanml_world_debug.yaml
```
