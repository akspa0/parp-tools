# Implementation Plan: V60 Unified Dataset and Shadow-First Terrain Model

**Branch**: `134-v60-unified-dataset-model` | **Date**: 2026-08-05 | **Spec**: [spec.md](./spec.md)

## Summary

Four phases, executed in dependency order:

1. **Unified v60 dataset** (US1, P1) — consolidate all scattered stores into a single v60 Zarr, re-harvest with the new C# code to get `terrain_shadow_256`, update the v50 build pipeline to v60.
2. **Curation fix** (US2, P1) — apply `surviving_height_levels` gating, rebuild the training curriculum.
3. **Shadow→height model** (US3, P1) — train a model that takes `terrain_shadow_256 → height_257`, beats the tile-mean baseline.
4. **Release v0.5.2** (US4, P2) — merge branches, update docs, tag release, publish via CI.

## Technical Context

**Language/Version**: Python 3.14 / uv (dataset + model), C# / .NET 10 (harvest tool)
**Primary Dependencies**: `zarr`, `pyarrow`, `numpy`, `torch` (model, CUDA required for training)
**Storage**: Single v60 Zarr store under `wow-viewer/output/datasets/v60/v60.1/`
**Target Platform**: Windows desktop (existing trainer environment); user's local RTX 4070 Ti SUPER (16 GB VRAM)
**Testing**: `pytest` for Python changes, `dotnet test` for C# changes

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | PASS | All new code under `wow-viewer/`. |
| Library-First | PASS | Reuses existing `harvester.v50.*` modules. |
| Real-Data Validation | PASS | Existing v50.1 stores on disk. |
| Per-Signal Evidence | PASS | shadow→height model reports per-signal metrics. |
| Streaming-First | PASS | Zarr stores are the output format. |
| No Hardcoded Paths | PASS | Client root is CLI argument. |
| User Runs Training | PASS | Training commands are prepared, never launched. |
| One Phase at a Time | PASS | Four phases, each ending in validation. |

## Project Structure

```
wow-viewer/
├── data-harvester/scripts/
│   ├── v60_build_from_npz.py              # NEW — build v60 Zarr from NPZ harvest shards
│   └── v60_train_shadow_height.py          # NEW — shadow→height model training
├── data-harvester/src/harvester/v50/
│   ├── classify.py                         # EXISTING — three-tier classification
│   ├── training_curriculum.py             # MODIFIED — curation gating (height levels)
│   └── v60_store.py                        # NEW — v60 store builder library
├── output/datasets/v60/
│   └── v60.1/
│       └── unified.zarr/                   # NEW — single consolidated store
├── docs/releases/
│   └── v0.5.2.md                           # EXISTING — update for release notes
├── README.md                               # UPDATE
└── docs/WoWViewer/USERGUIDE.md             # UPDATE
```

## Phases

### Phase 1: Unified v60 dataset (US1, P1)

**Goal**: Single v60 Zarr store with all signals, including `terrain_shadow_256`.

**Implementation**:

1. **Create `v60_build_from_npz.py`** — reads NPZ shards from the harvest tool's output (harvest-map-mpq), builds a single v60 Zarr store with all signals including `terrain_shadow_256`, and writes a unified index across all builds and maps. The v60 store is the training dataset — NOT the archaeology pipeline.
2. **Re-harvest with spec 133 C# changes** — user runs the updated harvest tool to get NPZ shards with `terrain_shadow_256` for all desired builds (0.5.3, 1.0.0, 3.3.5, 4.0.0.11927).
3. **Build v60 store from re-harvested NPZ** — the v60 builder reads the new NPZ shards and writes the consolidated store with `terrain_shadow_256` included.
4. **Update the signal catalog** — add `terrain_shadow_256`, `signal_class`, `surviving_height_levels` to the frozen catalog.
5. **Validate** — verify every tile has all expected signals, check determinism.

**Gate**: A single v60 store exists with all signals from both Kalimdor and Azeroth, deterministic.

### Phase 2: Curation fix (US2, P1)

**Goal**: Training curriculum with `surviving_height_levels` gating.

**Implementation**:

1. **Implement curation gating in `training_curriculum.py`** — add `--min-height-levels` and `--max-height-levels` options that filter tiles by `surviving_height_levels`. Default: exclude ≤64, admit compressed-rich.
2. **Rebuild curriculum** — run the curriculum builder against the v60 store with the curation fix applied.
3. **Validate** — confirm the excluded tiles are the ones that teach wrong relationships (2-level tiles, etc.) and the admitted tiles are the compressed-rich ones with correct targets.

**Gate**: A v60 curriculum exists with the curation fix, and the excluded/admitted tile lists are correct by manual inspection.

### Phase 3: Shadow→height model (US3, P1)

**Goal**: A model that takes `terrain_shadow_256 → height_257` and beats the tile-mean baseline.

**Implementation**:

1. **Create `v60_train_shadow_height.py`** — training script that loads `terrain_shadow_256` as input (1 channel) and `height_257` as target. Reuses `direct_cnn_v112` architecture with `in_channels=1`.
2. **User runs training** — exact command:
   ```
   cd wow-viewer/data-harvester
   uv run python scripts/v60_train_shadow_height.py \
       --store ../output/datasets/v60/v60.1/unified.zarr \
       --output ../output/runs/shadow-height-v1 \
       --epochs 200
   ```
3. **Evaluate** — compare val_mae against the frozen baseline (0.1493). Target: beat by 5% relative (val_mae < 0.142).

**Gate**: A trained checkpoint with val_mae < 0.142 on the held-out validation set.

### Phase 4: Release v0.5.2 (US4, P2)

**Goal**: Release v0.5.2, merge branches, update docs, start new dev branch.

**Implementation**:

1. **Update docs**: README.md, USERGUIDE.md, v0.5.2.md release notes with the current state.
2. **Merge branches**: 131-pm4-scene-graph-doodads → main, 132-terrain-brush-signature-classification → main, 133-unbaked-minimap-decomposition → main, 134-v60-unified-dataset-model → main.
3. **Tag v0.5.2**: `git tag v0.5.2 && git push origin v0.5.2` triggers CI to publish release.
4. **Create new dev branch**: branch off main for continued work.

**Gate**: GitHub Release published with v0.5.2 binaries, README current, main contains all work.

## Dependency Graph

```
Phase 1 (v60 store) ──> Phase 2 (curation) ──> Phase 3 (model) 
                                                         
Phase 4 (release) ── independent, can run in parallel with Phases 1-3
```

Phase 1 must complete before Phase 2 (curriculum needs the v60 store). Phase 2 must complete before Phase 3 (model needs the curated curriculum). Phase 4 is independent but should be the last phase to complete (it's the release).

## MVP scope

Phase 1 (v60 unified store) + Phase 2 (curation fix) + Phase 3 (shadow→height model) are the P1 deliverables. Phase 4 (release) is P2 and can be deferred if needed.