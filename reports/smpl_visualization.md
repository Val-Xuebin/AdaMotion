# SMPL Visualization

## Setup

- Workspace: `/work/adamotion`
- Source checkpoint: `/work/adamotion/experiments/humanml_world_debug/world_best.pt`
- Visualization script: `/work/adamotion/scripts/export_smpl_vis.py`
- Export backend: MDM `recover_from_ric` + `joints2smpl` + `SMPLify`

## Sample

- `motion_id`: `012698`
- sequence length: `10`
- context length: `6`
- evaluation mode: teacher-forced one-step prediction over the future segment

This export produces three variants:

- `ground_truth`
- `action_conditioned`
- `no_action`

## Artifacts

Root:

- `/work/adamotion/experiments/smpl_vis/012698_len10`

Per variant:

- `results.npy`: recovered 3D joint trajectory
- `sample0_rep0_obj/`: one OBJ mesh per frame
- `sample0_rep0_smpl_params.npy`: fitted SMPL parameters and vertices

Concrete outputs:

- `/work/adamotion/experiments/smpl_vis/012698_len10/ground_truth/sample0_rep0_smpl_params.npy`
- `/work/adamotion/experiments/smpl_vis/012698_len10/action_conditioned/sample0_rep0_smpl_params.npy`
- `/work/adamotion/experiments/smpl_vis/012698_len10/no_action/sample0_rep0_smpl_params.npy`

All three SMPL exports contain:

- `motion`: `(25, 6, 10)`
- `vertices`: `(6890, 3, 10)`
- `length`: `10`

## Quantitative Snapshot

On this visualization sample:

- feature MSE, action-conditioned: `0.001429`
- feature MSE, no-action: `0.001618`
- xyz joint MSE, action-conditioned: `0.002108`
- xyz joint MSE, no-action: `0.001023`

## Interpretation

- The current latent-action baseline already changes the predicted trajectory in feature space in a measurable way.
- On this sample, latent-action conditioning improves prediction in the native HumanML feature space but does not yet improve recovered joint-space accuracy.
- This is consistent with the earlier debug benchmark: the pipeline is functional, but the current action interface is not yet strong enough to dominate a no-action predictor after geometric recovery.
- The most likely next improvement is replacing the flat context MLP with a temporal transformer and revisiting the latent-action encoder so it models longer transitions instead of a strict two-frame bottleneck.
