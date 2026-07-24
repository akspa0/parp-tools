# Data Model: V7-Style WDL-Prior Height Reconstruction (Small Model Lane)

**Created**: 2026-07-24
**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)

No new on-disk schemas are invented. All entities reuse existing contracts; this document binds
them to the lane.

## Entities

### WDL Lattice Target (reused — Spec 117)

- Arrays (v50 store, per tile row): `wdl_outer_17` (17×17 f32), `wdl_inner_16` (16×16 f32),
  `wdl_outer_present` (17×17 bool), `wdl_inner_present` (16×16 bool).
- Encoded via `harvester/spec117/lattice_model.encode_lattice_target` → `(target[545] ∈ [0,1],
  mask[545] ∈ {0,1}, tile_min, tile_max)`; per-tile min-max with `RANGE_FLOOR = 1.0`; absent
  samples never affect normalization or loss.
- Validation: rows with zero present samples are excluded by `select_lattice_rows`, never
  fabricated.

### Stage A Checkpoint (`mit_b0_lattice` | `lattice_net_v2`)

- Torch checkpoint containing: `model_state_dict`, `architecture` (`id`, `config_sha256`,
  `parameter_count` ∈ [3M, 30M] for `mit_b0_lattice`), explicit config payload (backbone id +
  revision OR from-scratch base), split identity, normalization provenance.
- Reconstruction rule (FR-003): config payload alone must rebuild the exact module before
  `load_state_dict`. (Spec 117's `lattice_config.base` lesson applies to backbone fields too.)

### Prior Coarse Store (reused schema — Spec 114)

- Zarr group + `index.parquet` in the exact shape `geometry_detailer_train.validate_coarse_store`
  accepts, produced by the new bridge instead of `direct_geometry_materialize.py`.
- Dense field: 257×257 f32 prior, bilinear-upsampled from outer/inner grids (Spec 117 bridge
  rule: upsample both, average overlapping region).
- Attrs must name the producing Stage A checkpoint sha256 (FR-014 provenance).

### Object-Mask Loss Weighting (reused + one new variant)

- Stage B (pixel-level, existing): `harvester/spec118/object_loss.py` — per-point
  `1 − w·mask`, `w` = `--object-mask-weight` (default 0.0), touched/untouched region MAE in the
  run record.
- Stage A (tile-level, new): per-tile scale `1 − w·coverage`, `coverage` = mean of
  `object_geometry_visible_mask_257` for that row; identical warn+disable semantics when the
  array is absent; tile-level touched (coverage ≥ 0.05) vs untouched MAE in the record.

### Held-Out Split (reused — Spec 116)

- Frozen split artifact (8-neighbour isolation, `verified_violation_count = 0` mandatory);
  `--held-out-split` required by every trainer in the lane.

### Model Stage Run Record (reused — v50)

- `v50-model-stage-run-v1`; Stage A uses stage `"lattice_prior"`; architecture identity
  distinguishes backbone. `promotion_verdict = "pending"` until the user visual gate.
- Additional recorded fields (within existing schema's metrics/params payload): backbone hub id +
  revision (or `null` for from-scratch), `object_mask_weight`, `object_mask_signal_present`,
  tile-mean baseline MAE (Stage A), prior-only baseline MAE + GT-prior ablation MAE (Stage B).

## Relationships

```text
v50 store ──(Spec 117 contract)──> Stage A trainer ──> Stage A checkpoint
Stage A checkpoint ──(prior_coarse_bridge)──> prior coarse store
prior coarse store ──(--coarse-store)──> Stage B detailer trainer ──> Stage B checkpoint
object_geometry_visible_mask_257 ──(loss-side only)──> Stage A + Stage B trainers
Spec 116 split ──(required)──> both trainers
```

## State Transitions

- Run record: `promotion_verdict: pending → (user visual gate) → promoted | rejected`.
- Lane: Phase gates G1 → G2 → G3; a failed gate records a negative result and stops downstream
  phases (no Stage B training if G1 fails).
