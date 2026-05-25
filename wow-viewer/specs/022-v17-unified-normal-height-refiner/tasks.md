# Tasks: V17 Unified Normal Trainer

## Phase 1: Explicit Variant Contract

- [ ] T001 Add `--normal-variant` CLI flag in `train_v16_1_common.py`
  - Allowed values: `v16_1_1_base`, `v16_1_2_refiner`, `v16_1_3_height`, `v17_hybrid`
  - Default: `v17_hybrid`

- [ ] T002 Implement variant resolver
  - Resolve booleans: `resolved_height_channel`, `resolved_refiner_enabled`
  - Reject conflicting manual flags with clear error messages
  - Remove silent fallback behavior

- [ ] T003 Persist and print resolved contract
  - Startup log prints: variant + resolved toggles + distill weight
  - `config.json` writes: `normal_variant`, `resolved_height_channel`, `resolved_refiner_enabled`

## Phase 2: V17 Hybrid Wiring

- [ ] T004 Wire model and dataset selection to resolved variant
  - Main model uses 4ch input when resolved height-channel is true
  - Dataset receives height-channel mode from resolver, not raw flag checks

- [ ] T005 Wire refiner/distillation logic to resolved variant
  - Permit refiner path with height-channel only for `v17_hybrid`
  - Keep existing V16.1.2 distillation behavior unchanged otherwise

## Phase 3: Curated Defaults

- [ ] T006 Apply v17 defaults when unset
  - `epochs=50`, `train_max_tiles=80`, `val_max_tiles=10`

- [ ] T007 Enforce curated workflow for v17
  - Validate curation manifest path presence for `v17_hybrid`
  - Emit fail-fast message if missing

## Phase 4: Validation

- [ ] T008 Run 1-epoch sanity proof
  - Command uses `--normal-variant v17_hybrid`
  - Confirm resolved contract in logs and `config.json`

- [ ] T009 Run curated bounded launch command
  - 50 epochs, 80 train tiles, 10 val tiles, curated manifest

- [ ] T010 Verify preview quality
  - Confirm `normal_gt` in validation previews no longer shows checkerboard halftone artifact
