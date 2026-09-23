# WDL Prior Checkpoint and Archive Contract

The checkpoint contains `model_variant`, `input_contract`, `target_contract`, `height_global_min`, `height_global_max`, `model`, and training metrics. `input_contract` is `minimap_rgb_only_imagenet_normalized`.

Inference writes an NPZ archive with `rows`, `outer_17`, `inner_16`, and `metadata_json`. V8 accepts it only when every requested source row is present once and the outer grid has shape 17×17.

The archive is a generated prior. It is distinct from ground truth and must be named in V8 inference evidence.
