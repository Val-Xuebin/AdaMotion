# AdaMotion Runs

| run_name | stage | family | representation | best_epoch | best_val_loss | best_val_action_loss | best_val_no_action_loss | best_val_gain | legacy | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| humanml_lam_debug | lam | feature_mlp | humanml_feature_vector | 1 | 0.028240468640855006 |  |  |  | True | Previous baseline before joint-position spatiotemporal transformer refactor. |
| humanml_world_debug | world_model | feature_mlp | humanml_feature_vector | 1 |  | 0.014138327111059557 | 0.012434584057014643 | -0.0017037430540449148 | True | Previous baseline before joint-position spatiotemporal transformer refactor. |
| humanml_feature_mlp_lam_debug | lam | feature_mlp | humanml_feature_vector | 4 | 0.006464823236245485 |  |  |  | False |  |
| humanml_feature_mlp_world_debug | world_model | feature_mlp | humanml_feature_vector | 4 |  | 0.006048036974216204 | 0.005989757514711155 | -5.8279459505048515e-05 | False |  |
