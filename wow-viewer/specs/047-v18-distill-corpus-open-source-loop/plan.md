# Implementation Plan: V18 Focused Two-Build Terrain Reconstruction System

**Branch**: `047-v18-distill-corpus-open-source-loop` | **Date**: 2026-06-05 | **Spec**: [`wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`](spec.md)

**Input**: Feature specification from `/specs/047-v18-distill-corpus-open-source-loop/spec.md`

## Summary

The V18 owner path is:

1. treat the focused V18 corpus (`0_5_3_3368`, `3_3_5_12340`) as final enough
   to train on,
2. generate a dedicated focused V18 curation manifest on those stores,
3. train two separate V18 models from minimap input:
   - height
   - normals
4. use those outputs later in a quilt-level terrain reconstruction stage aimed
   at stitched ADT terrain, not isolated tile previews.

The implementation slice in this plan is the design closure plus the curation
and training wrappers that make the focused lane easy to run.

## Technical Context

**Language/Version**: Python 3.11+ via `uv`; existing C# / .NET 10 tooling remains the dataset producer and later terrain consumer

**Primary Dependencies**: PyTorch, Zarr v3, PyArrow/Parquet, NumPy, Pillow

**Storage**: V18 Zarr stores under `wow-viewer/output/datasets/v18/`; training artifacts under `wow-viewer/models/v18/`

**Testing**: `uv run python -m py_compile` plus bounded training/capture/validation command proofs

**Target Platform**: local Windows operator workflow with staged WoWArchive clients

**Project Type**: Python CLI/training pipeline inside the `wow-viewer` repo

**Performance Goals**: bounded two-build training runs must be operationally cheap enough to scout quickly on a single local CUDA device

**Constraints**:
- only the focused corpus builds are in scope
- no renderer-truth dependency in the active training lane
- no monolithic multitask model
- liquid masks remain available to curation/validity logic and terrain-valid
  loss masking

**Scale/Scope**:
- two V18 stores
- two primary model runs
- one focused curation manifest
- future quilt/stitch stage documented but not fully implemented in this slice

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo independence**: pass. All planned code and docs stay inside
  `wow-viewer/`.
- **Library-first**: pass. No new parser or format-surface duplication is
  introduced; this slice stays in Python orchestration/training.
- **Real-data validation**: pass. The focused lane depends on staged 0.5.3 and
  3.3.5 client roots and V18 Zarr stores derived from them.
- **Residual model chain**: pass with clarification. Height and normal remain
  independent models/checkpoints; this plan explicitly rejects a monolithic
  multitask terrain model.
- **No untrusted client paths**: pass. Only staged clients under
  `output/tmp/wowarchive-clients/`.
- **Training-script-change validation**: required. Any script changes land with
  `py_compile` and focused command proof.

## Project Structure

### Documentation (this feature)

```text
specs/047-v18-distill-corpus-open-source-loop/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── v18-focused-curation-manifest.schema.json
│   └── v18-focused-training-run.schema.json
└── tasks.md
```

### Source Code (repository root)

```text
wow-viewer/
├── data-harvester/
│   ├── scripts/
│   │   ├── build_focused_two_build_corpus.py
│   │   ├── build_v16_curation_manifest.py
│   │   ├── build_v18_curation_manifest.py
│   │   ├── build_v18_tiny_manifest.py
│   │   ├── train_v18.py
│   │   ├── train_v18_focus.py
│   │   └── train_v16_1_common.py
│   └── src/harvester/
│       ├── v16_1_dataset.py
│       ├── v16_1_models.py
│       └── v16_curation.py
├── docs/architecture/
│   └── v18-distill-corpus-open-source-loop-2026-06-04.md
└── output/datasets/v18/
```

**Structure Decision**: keep feature docs under `specs/047-.../` and keep the
implementation slice in `data-harvester/scripts/` as thin focused wrappers over
the existing V18/V16.1 training and curation surfaces.

## Phase 0: Research

See [research.md](research.md). The open design questions resolved for this
feature are:

- which builds remain in scope,
- whether liquids remain part of the active signal story,
- whether V18 should be one model or two,
- whether a focused V18 wrapper layer is needed over the older V16-named
  scripts,
- how quilt-level terrain reconstruction fits the final V18 design.

## Phase 1: Design & Contracts

Artifacts produced in this phase:

- [research.md](research.md)
- [data-model.md](data-model.md)
- [quickstart.md](quickstart.md)
- [`contracts/v18-focused-curation-manifest.schema.json`](contracts/v18-focused-curation-manifest.schema.json)
- [`contracts/v18-focused-training-run.schema.json`](contracts/v18-focused-training-run.schema.json)

Agent context update script is not present under this repo’s `.specify`
PowerShell surface, so there is no local script to run for that part of the
template workflow. Continuity updates remain the replacement mechanism here.

## Phase 2: Immediate Implementation

### Phase 2A - Focused Curation Surface

**Goal**: expose a V18-focused curation entrypoint that writes manifests under
the V18 dataset root and defaults to the two focused builds.

1. Add `build_v18_curation_manifest.py` as a focused wrapper over the existing
   curation implementation.
2. Default it to `output/datasets/v18/`.
3. Default it to the two focused builds and a stable V18 run-name/output root.
4. Keep low-trainable liquid/object wipeout rows out of the focused manifest.

**Validation**:

- `--help` and `py_compile` succeed.
- operators can build a focused manifest without remembering V16 dataset paths
  or six-build defaults.

### Phase 2B - Focused Training Surface

**Goal**: expose a V18-focused training entrypoint that defaults to the two
focused builds and prefers the latest focused V18 curation manifest.

1. Add `train_v18_focus.py` as a focused wrapper over `train_v18.py`.
2. Default dataset root to V18.
3. Default builds to the two focused anchors.
4. Auto-resolve the latest focused V18 `kept_tiles.parquet` when the user does
   not pass `--curation-manifest`.
5. Default focused runs toward the real 8 GB lane through startup batch
   autotune, and keep height/normal losses masked to terrain-valid regions.
6. Default focused runs to strict near-equal per-build sampling so a skewed corpus
   cannot silently dominate full-size epochs.
7. Default focused runs to restrained rotating bucket coverage so later epochs
   traverse the full curated pool instead of replaying the same full-pool
   gradients every epoch.
8. Keep terrain-valid masking honest by including both WMO basement/object
   footprints and WMO roof/top-geometry occlusion when the harvested store
   carries both signals.
9. Expose a focused minimap-only inference proof entrypoint so deployment-surface
   validation is distinct from offline supervised training evaluation.
10. Keep super-tiny experiments explicit by deriving a separate tiny manifest
    instead of silently changing the default focused manifest resolver.
11. Keep focused base height/base-normal runs operational on the
    `object_precise_mask` contract by applying safer auto DataLoader defaults
    only when the operator leaves loader tuning on auto.

**Validation**:

- `--help` and `py_compile` succeed.
- height/normal focused commands can be launched with shorter, safer operator
  syntax.
- focused runs can derive their per-epoch subset from a bucket-rotation
  fraction instead of requiring a fixed `--train-epoch-tiles` count.
- focused operators can run a minimap-only inference proof command against V18
  checkpoints without relying on supervision-only tensors during the run.
- focused auto-loader safety must not override explicit `--num-workers`,
  `--prefetch-factor`, or `--persistent-workers` choices.

### Phase 2C - Super-Tiny Focused Corpus Surface

**Goal**: let the operator derive a truly tiny balanced scouting manifest from
the focused kept pool without changing the trainer contract.

1. Add `build_v18_tiny_manifest.py` as a focused manifest-derivation utility.
2. Default it to the canonical focused source manifest
   `v18_focus_terrain_v1`.
3. Support per-build/per-bucket caps plus an optional fractional cap.
4. Prefer map diversity inside each selected build/bucket stratum so a tiny run
   does not collapse to a single map family.
5. Keep the focused trainer default resolver unchanged; tiny runs pass the tiny
   manifest explicitly.
6. Document that tiny-manifest runs should use
   `--train-bucket-rotation-fraction 1.0` because the tiny manifest itself is
   already the throttle.

**Validation**:

- `py_compile` succeeds.
- targeted tests prove the tiny selector keeps build/bucket balance and map
  round-robin behavior.
- a real tiny manifest can be generated from the active focused kept pool.

### Phase 2C - Operator Docs

**Goal**: make the focused V18 lane discoverable without reading old V16-heavy
README sections.

1. Add focused V18 curation/training quickstart commands to
   `data-harvester/README.md`.
2. Make the full focused-manifest training commands the primary examples and
   keep smaller-manifest scouting clearly optional.
3. Align architecture and memory-bank wording with the final owner design.

**Validation**:

- docs name the focused wrappers and the two-build scope consistently.

## Complexity Tracking

No constitution violations are required for this slice.
