# Implementation Plan: Terrain Feature Classification for Geometry Deconfounding

**Branch**: `115-terrain-feature-classifier` | **Date**: 2026-07-20 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/115-terrain-feature-classifier/spec.md`

## Summary

The promoted-pending geometry chain (`mit_b0-authored-v1` + `detailer-mit_b0-authored-v2-bandsplit-continued`)
maps minimap RGB directly to relative height, and real out-of-distribution testing showed it using
color as a depth proxy — decoding flat roads as sloping ridges. This plan adds a separate,
independently promoted **terrain-feature classifier** (RGB → per-pixel feature-family
probabilities), supervised by real per-tile MTEX texture names, then retrains geometry with that
classifier's *generated* output as extra input channels so height prediction has an explicit
"road, not slope" signal at deployment on any image.

Technical approach, in dependency order:

1. **Label pipeline** — a names-only C# dump joins to the curriculum index by `(map, tile_x,
   tile_y)`; a Python module resolves each pixel's dominant texture layer through `alpha_256` and
   maps its name to a canonical family via a versioned substring rule list.
2. **Classifier** — a small from-scratch encoder/decoder in the existing `HeightRelativeNet`
   capacity class, emitting `(K, 256, 256)` class logits.
3. **Classifier trainer + inference** — cloned conventions from `direct_geometry_train.py` /
   `direct_geometry_infer.py`: dry-run default, `--confirm-run` gate, in-run baselines,
   schema-validated `model_stage_run.json`, audit manifest.
4. **Geometry input-contract extension** — `in_channels` parameter (default 3) so RGB-only variants
   and their existing checkpoints stay bit-identical, plus a trainer flag to concatenate a
   materialized feature-map store.

## Technical Context

**Language/Version**: Python 3.11+ (uv-managed), C# / .NET 10 for the names dump

**Primary Dependencies**: PyTorch, zarr v3, pyarrow, numpy, Pillow; `WowViewer.Core.IO` for the C# side

**Storage**: Zarr v3 store `output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr` (2990 rows)
plus a new derived label store and a JSON texture-name dump

**Testing**: pytest under `data-harvester/tests/v50/`; xUnit for the C# command

**Target Platform**: Windows local (CUDA training user-run); CPU-only for label/inference paths

**Project Type**: ML dataset + model pipeline extending an existing library

**Performance Goals**: label derivation over 2990 rows in minutes on CPU; classifier training within
the same envelope as the existing geometry models (16 GB VRAM, batch 16)

**Constraints**: no ground-truth signal in any inference path; no shared weights across stages; all
GPU/heavy runs user-executed

**Scale/Scope**: 2990 curriculum rows; ~194 distinct tilesets; 5-class taxonomy; 700-tile OOD set

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Repo Independence | PASS | All new files under `wow-viewer/`; no outside references. |
| II. Library-First | PASS | Logic lands in `src/harvester/v50/` modules; `scripts/*.py` stay thin wrappers. The C# dump reuses existing `WowViewer.Core.IO` readers rather than adding a parser. |
| III. Real-Data Validation | PASS | Labels derive from the real 0.5.3.3368 corpus; the OOD gate uses a real image. Research findings were established against real data, and one hypothesis was falsified that way. |
| IV. Residual Model Chain | PASS | Classifier and geometry are separate models, separate checkpoints, no shared weights, no multi-task head. Classifier output feeds geometry as an *input*, which is exactly the chaining this principle prescribes. |
| V. Streaming-First Dataset Pipeline | **DEVIATION — see Complexity Tracking** | The texture-name dump is a JSON side-car, not a Zarr array reached through the length-prefixed stream. |
| VI. No Client Path Assumptions | PASS | Client root stays a required CLI argument; nothing hardcoded. |
| Training Script Changes Require Validation | PASS | The geometry input-channel change is explicitly a model-spec change: `in_channels` is recorded in the architecture identity hash, so old and new contracts cannot be confused. |
| Bite-Sized Plans | PASS | Decomposed into single-concern, independently validatable steps below. |

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Principle V: texture names arrive as a JSON side-car rather than through the Zarr streaming pipeline | The global tileset→name list is provably absent from the v50 store (research Decision 1) and its only producer is a transient build-time stream. Labels cannot be derived without recovering names from the client. | Putting names back into the Zarr store means changing the store builder and re-running a multi-hour client-backed rebuild of stores validated earlier this session. The side-car is additive, cheap, and leaves every existing store byte-identical. Revisit if the store is rebuilt for other reasons. |

## Project Structure

### Documentation (this feature)

```text
specs/115-terrain-feature-classifier/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── terrain-feature-label-contract.md
├── checklists/
│   └── requirements.md  # From speckit-specify
└── tasks.md             # speckit-tasks output (not created here)
```

### Source Code (repository root)

```text
wow-viewer/
├── tools/harvest/WowViewer.Tool.Harvest/
│   └── Program.cs                                   # MODIFY: add `dump-texture-names`
└── data-harvester/
    ├── src/harvester/v50/
    │   ├── terrain_feature_labels.py                # NEW: taxonomy + dominant-layer + label derivation
    │   ├── terrain_feature_model.py                 # NEW: classifier architecture + identity
    │   ├── terrain_feature_train.py                 # NEW: trainer (dry-run default, gates, run record)
    │   ├── terrain_feature_infer.py                 # NEW: deployment inference (RGB -> feature map)
    │   ├── direct_geometry_model.py                 # MODIFY: in_channels (default 3, back-compatible)
    │   └── direct_geometry_train.py                 # MODIFY: optional --feature-store extra channels
    ├── scripts/
    │   ├── v50_build_terrain_feature_labels.py      # NEW: thin CLI, --write gate
    │   ├── v50_train_terrain_features.py            # NEW: thin CLI, --confirm-run gate
    │   ├── v50_infer_terrain_features.py            # NEW: thin CLI, --write gate
    │   └── v50_materialize_feature_maps.py          # NEW: generated feature maps for geometry retrain
    └── tests/v50/
        ├── test_terrain_feature_labels.py           # NEW
        └── test_terrain_feature_model.py            # NEW
```

**Structure Decision**: mirrors the existing Spec 114 layout exactly — one library module per
concern under `src/harvester/v50/`, one thin `scripts/v50_*.py` CLI per user-facing action, tests
under `tests/v50/`. Names follow the established `<stage>_<concern>.py` convention already used by
`direct_geometry_*` and `geometry_detailer_*`.

## Implementation Steps

### Phase A — Label pipeline (US1 prerequisite)

- **A1** Add `dump-texture-names` to `Program.cs`: iterate occupied tiles, emit
  `{map, tile_x, tile_y, texture_names[]}` JSON. Validate against the known tile Kalimdor 24,40,
  whose true table is the four `Darkshore*` textures.
- **A2** Implement `terrain_feature_labels.py`: the versioned family taxonomy, the ordered substring
  rule list, the dominant-layer resolution over `alpha_256`, and per-row `(256, 256)` label
  derivation with explicit unknown/excluded handling.
- **A3** Implement `v50_build_terrain_feature_labels.py`: join the dump to `index.parquet`, derive
  all rows, report class-coverage and exclusion counts, write a derived label store only under
  `--write`.

### Phase B — Classifier (US1)

- **B1** Implement `terrain_feature_model.py`: architecture, `in_channels=3` fixed, `K`-class
  output at 256×256, plus `architecture_identity()` returning a config hash.
- **B2** Implement `terrain_feature_train.py`: row selection reusing the frozen split, majority-class
  in-run baseline, per-class metrics, dry-run default, `--confirm-run`, schema-validated
  `model_stage_run.json` with `stage: "terrain_features"`.
- **B3** Implement `terrain_feature_infer.py` + CLI: loose 256×256 tiles in, per-tile feature map
  and review sheet out, `--write` gate, audit manifest binding input/checkpoint/output hashes.

### Phase C — Geometry retrain (US2)

- **C1** Add `in_channels` to `direct_geometry_model.py` (default 3) and fold it into the
  architecture config hash so a 4+-channel model cannot be mistaken for the RGB-only baseline.
- **C2** Implement `v50_materialize_feature_maps.py`: run the promoted classifier over curriculum
  rows, persist generated feature maps (mirrors `v50_materialize_coarse_relief.py`).
- **C3** Extend `direct_geometry_train.py` with an optional `--feature-store`, concatenating the
  generated map to RGB, and add the road-region error metric required by FR-008.

### Phase D — Detailer re-pair (US3)

- **D1** Re-materialize coarse relief from the Phase C checkpoint and retrain the detailer, binding
  `upstream_models` to the new coarse hash. Reuses existing scripts; no new modules.

## Validation Gates

Each phase ends with validation against real data, per the constitution's One Phase at a Time rule:

- **A** — the dump reproduces Kalimdor 24,40's known Darkshore table; label derivation reports real
  class coverage over 2990 rows with reconciled exclusion counts.
- **B** — focused tests pass; the trainer dry-runs against the real curriculum and refuses to train
  without `--confirm-run`; after the user's training run, the classifier beats the majority-class
  baseline and its OOD sheet flags visibly road-like regions.
- **C** — road-region height error improves over the frozen baseline; non-road error within
  tolerance; the RGB-only architecture identity is unchanged (existing checkpoints still load).
- **D** — the detailer's existing ≥5% gate re-passes against the new coarse checkpoint.
