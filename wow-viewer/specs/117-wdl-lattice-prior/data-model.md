# Data Model: WDL-Lattice Coarse Prior for Terrain Geometry

**Feature**: 117-wdl-lattice-prior | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

This document defines the entities and on-disk schemas for Spec 117. It references decisions from
[research.md](research.md) by id (D-0n). New artifacts are written under
`wow-viewer/output/datasets/spec117/` (or a feature-scoped subroot the operator configures); source
v50 stores and the existing coarse/detailer checkpoints are read-only.

---

## Entities

### WDL Lattice
The 545-sample coarse height sampling of one tile: 17×17 outer at `height_257[::16,::16]` and
16×16 inner at `height_257[8::16,8::16]` (spec FR-001, matching `TerrainWdlLattice` and Spec 108
FR-001 exactly). Deterministically derived from real MCVT vertex data already backing `height_257`.

| Field | Type | Notes |
|-------|------|-------|
| `outer17` | float32 (17, 17) | dense half-step samples, world height units |
| `inner16` | float32 (16, 16) | offset half-step samples, world height units |
| `outer_present`, `inner_present` | bool, same shapes | per-sample validity; a gap is excluded, never fabricated (spec Edge Cases) |

**Validation rules**:
- A tile lacking real `height_257` ground truth produces no lattice row at all (excluded, counted —
  never zero-filled).
- A lattice sample whose backing MCVT vertex is itself absent is marked `..._present=false` at that
  coordinate; it is never interpolated to fill the gap (spec Edge Cases).

### Lattice Predictor
An independently trained, independently checkpointed model mapping minimap RGB alone to a generated
WDL Lattice. No shared weights with the coarse or detailer stage (constitution IV). Evaluated only
against the spatially-isolated held-out split (D-03).

### Generated Lattice Feature Store
The derived, checkpoint-bound store of the frozen predictor's generated (never ground-truth)
output, upsampled and shaped to satisfy the existing `--feature-store` contract verbatim (D-01) —
no new consumer-side schema, no trainer changes.

| Field | Type | Notes |
|-------|------|-------|
| `feature_map` | float16 (N, 1, 256, 256) | bilinear-upsampled dense field from the 545-sample lattice (D-01) |
| `index.parquet` | — | `source_row_index`, `map`, `tile_x`, `tile_y` — row-aligned to the source curriculum, same contract `structure_materialize.py` already uses |
| attrs.`schema` | const | `"v115-feature-map-v1"` (D-01: reused verbatim, not a new schema) |
| attrs.`class_count` | const | `1` |
| attrs.`checkpoint_path`, `checkpoint_sha256` | string | binds the store to the exact predictor checkpoint that produced it |
| attrs.`source_signal` | const | `"wdl_lattice"` — the only attr a consumer would need to distinguish this from a Spec 115/116 class-probability feature store, informational only, never read by the trainers |

---

## Store Schema Additions (v50 curriculum store)

One new array added to the existing per-build v50 curriculum store via the harvest signal-config
widening (US1, C# side — no new store, no new tool):

| Array | Shape | Dtype | Notes |
|-------|-------|-------|-------|
| `wdl_lattice_outer17` | (N, 17, 17) | float32 | world height units, NaN or a documented sentinel where `outer_present=false` |
| `wdl_lattice_inner16` | (N, 16, 16) | float32 | world height units, same absence convention |
| `wdl_lattice_outer_present` | (N, 17, 17) | bool | per-sample validity |
| `wdl_lattice_inner_present` | (N, 16, 16) | bool | per-sample validity |

Rows where every sample is absent are excluded from the exported array set entirely at harvest time
(never a phantom all-absent row), matching the existing project-wide convention of counting
exclusions rather than emitting degenerate rows.

## Run-Record Schema

The standalone Lattice Predictor reuses the existing **`v50-model-stage-run-v1`** schema verbatim
— the same schema `direct_geometry_train.py` and `geometry_detailer_train.py` already write. No new
schema is defined for this feature; the predictor is structurally one more stage in the same
residual chain (single dense output, checkpoint + baselines + metrics + `promotion_verdict`), and
inventing a parallel schema would duplicate validated infrastructure for no benefit (same reasoning
as D-01).

- `stage`: `"lattice_prior"` (new stage name; distinguishes this run from `direct_geometry` runs in
  the same schema).
- `output_signal`: `"wdl_lattice_545"`.
- `baselines.tile_mean`: the per-tile mean of the tile's own real lattice samples (D-02) — computed
  identically to how the coarse/detailer stages already compute their own tile-mean baseline, just
  at lattice resolution.
- `metrics.best_val_mae`: held-out lattice-point MAE, computed only against the spatially-isolated
  split (D-03).
- `upstream_models`: `[]` — the predictor has no upstream model; it is the coarsest stage in the
  chain.

The paired chain-integration comparison (US3) reuses the existing `model_stage_run.json` records
already produced by `direct_geometry_train.py`/`geometry_detailer_train.py` for each condition
(with/without the lattice prior, per feed point) — no new comparison schema; the same
before/after reading pattern used for the structure-augmented detailer result this session applies
unchanged.
