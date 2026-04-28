# AdaMotion

AdaMotion in this workspace is trimmed to the current HumanML3D main experiment chain:

1. momentum LAM pretraining on `sal_rep`
2. official SALAD action adapter training
3. official SALAD action prior training
4. official SALAD benchmark evaluation

Environment:

```bash
conda env create -f /workspace/AdaMotion/environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adamotion
```

Dataset preparation:

```bash
python3 /workspace/AdaMotion/scripts/prepare_humanml3d.py \
  --data-root /workspace/dataset/HumanML3D
```

Optional smoke check:

```bash
python3 /workspace/AdaMotion/scripts/smoke_test_humanml3d.py \
  --data-root /workspace/dataset/HumanML3D \
  --device cuda
```

Main training entrypoint:

```bash
python3 /workspace/AdaMotion/scripts/train_hm.py \
  --config /workspace/AdaMotion/configs/humanml_sal_rep_lam_momentum_full.yaml
```

Full main-experiment train chain:

```bash
bash /workspace/AdaMotion/scripts/run_adamotion_salad_action_train.sh
```

Main benchmark evaluation:

```bash
bash /workspace/AdaMotion/scripts/run_adamotion_salad_action_eval.sh benchmark
```

Kept configs:

- `configs/humanml_sal_rep_lam_momentum_full.yaml`
- `configs/humanml_salad_official_action_adapter_momentum_full.yaml`
- `configs/humanml_salad_official_action_prior_momentum_full.yaml`

Kept scripts:

- `scripts/train_hm.py`
- `scripts/prepare_humanml3d.py`
- `scripts/build_humanml3d_usable_splits.py`
- `scripts/build_humanml3d_flat_view.py`
- `scripts/smoke_test_humanml3d.py`
- `scripts/eval_official_salad_action_benchmark.py`
- `scripts/run_adamotion_salad_action_train.sh`
- `scripts/run_adamotion_salad_action_eval.sh`

Local outputs remain under ignored paths such as `experiments/`, `wandb/`, and benchmark raw logs outside the repo tree.
