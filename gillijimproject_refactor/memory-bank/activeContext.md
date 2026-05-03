# ACTIVE CONTEXT — V11.1

## BRANCH
`v0.4.9`. Clean restart from ced5899. V11.1 is active training.

## WHAT CHANGED FROM V11
- **Overlapping conv stem** replaces ConvNeXt patch stem (7x7 stride2 x2). Fixes grid artifacts.
- **Frequency ramp loss** — detail-first training: high-freq loss weight starts at 1.0, decays to 0.1 over 60 epochs. Low-freq ramps opposite. Forces model to learn texture before shape.
- **Separate validation** — training no longer runs val loop. Use `validate_v11.py` to check progress against latest checkpoint.
- **Dataset:** 7000+ curated tiles across 6 clients (0.5.3 through 4.0.0).

## EXTRACTED DATASET
Cache at `output/tmp/v11_cache/`. ~7000 tiles, ~100% MCAL/MCLY coverage.

## NEXT
1. Run `dataset-build-cache` with minimaps (user command provided)
2. `train_v11.py` with `--freq-ramp-epochs 60 --use-compile`
3. Validate periodically with `validate_v11.py`
