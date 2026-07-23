# Implementation Plan: Per-Object Occlusion-Aware Masks for Object-Deconfounded Terrain Height

**Branch**: `118-object-occlusion-masks` | **Date**: 2026-07-22 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/118-object-occlusion-masks/spec.md`

## Summary

Authored minimaps paint world objects onto the ground; a height model regressing terrain from
minimap RGB is fed a confounded input on ~52–54% of tiles, and the Spec 117 RGB→lattice result
(no baseline beat) makes the unhandled object confound a leading suspect. This feature reintroduces
the object signal dropped from v50 — correctly: a **per-object, occlusion-aware
(visible-portion-only) mask with a class label**, harvested from geometry truth, used first as a
**loss-side** deconfound and then as supervision for a small from-scratch segmenter whose
prediction feeds the geometry chain as a generated input (the Spec 115 pattern).

The key enabler (research D-01/D-03): the C# harvester **already computes** the exact
visibility-correct mask via `TerrainVisibleObjectMaskRasterizer` + the strict object-geometry
target in `AdtTensorPackBuilder` — transformed M2/WMO triangles retained only above the raw MCVT
surface (+0.25 clearance, liquid-aware, front-most overlap rule) — and already streams
`object_geometry_visible_mask_257` / `object_geometry_visible_source_257` in the Full/V16 profiles.
US1 is therefore catalog wiring plus **one** bounded C# addition (a dense per-tile instance-id
array, the only missing FR-002 entity). US2 mirrors the proven `--liquid-mask-weight` trainer flag.
US3 mirrors the Spec 116/117 package pattern (contract/model/train/infer/bridge) with zero changes
to the consuming trainers.

Delivery order: US1 (signal in store) → US2 (ground-truth-mask loss proof — the cheap gate for the
whole direction) → US3 (segmenter + generated-feature bridge).

## Technical Context

**Language/Version**: C# / .NET 10 for the one dense instance-array addition (rasterizer paint
param + builder + serializers, in the existing `WowViewer.Core`/`Core.IO` tree); Python 3.11+
managed by `uv` under `wow-viewer/data-harvester/` for everything else.

**Primary Dependencies**: PyTorch (segmenter; CPU dry-run + user-run CUDA), NumPy, Zarr v3 +
PyArrow (store/index I/O). No new dependency beyond what Specs 114–117 added.

**Storage**: The existing v50 store gains three cataloged arrays
(`object_geometry_visible_mask_257`, `object_geometry_visible_source_257`,
`object_geometry_visible_instance_257`) via the signal-catalog path — never an in-place mutation
(FR-013). The segmenter's generated output lands in a derived `v115-feature-map-v1` store bound to
its checkpoint hash. Zarr only; no NPZ (constitution V).

**Testing**: `pytest` under `data-harvester/tests/` (new `tests/spec118/`); C# focused tests in
`WowViewer.Core.Tests` (rasterizer instance paint + serializer round-trip); Ruff + `py_compile`;
dry-run-first CLIs refuse to write without explicit flags.

**Target Platform**: CPU for catalog/config, audit, dry-runs, and unit tests; CUDA for user-run
training. No client-path assumptions beyond the configured harvest client root (Rule 9).

**Project Type**: One scoped C# addition to the existing harvest signal pipeline (library-first) +
a new Python package `harvester/spec118/` mirroring the Spec 116/117 convention + a two-line-flag
change to each existing geometry trainer (research D-05, separately committed per Rule 6).

**Performance Goals**: The C# addition is one int32 write per visible fragment inside an existing
raster loop — negligible harvest cost. Segmenter stays in the small-model class (≤ a few hundred K
params at `--base 24`, SC-005).

**Constraints**: User executes all training/heavy rebuilds (FR-012); dry-run-first CLIs; the
segmenter is independently trained/checkpointed (FR-010, constitution IV); ground-truth masks are
loss-side only — deployed inputs stay minimap + model-predicted maps (FR-014); no
GAN/adversarial/generative technique (spec boundary); judgments use the Spec 116
spatially-isolated split (FR-008).

**Scale/Scope**: Same corpus as Specs 116–117 (corrected dual-view v50 curriculum, Kalimdor +
Azeroth, build 0.5.3.3368). Three new store arrays; one new small model; one new flag per trainer.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Repo Independence | PASS | All new code under `wow-viewer/` (`src/core/WowViewer.Core.IO`, `data-harvester/`). |
| II. Library-First | PASS | C# change inside existing Core.IO rasterizer/builder/serializer; Python logic in `src/harvester/spec118/`; `scripts/` stay thin. |
| III. Real-Data Validation | PASS | US1 audit + US2/US3 scoring all run against the real v50 store and the real Spec 116 split; eyeball proof on real city/underground tiles (user-run). |
| IV. Residual Model Chain | PASS | One single-output specialist (RGB → 3-class pixel map), independently checkpointed, feeding downstream by output only. |
| V. Streaming-First Dataset | PASS | Three arrays via the existing streaming harvest + catalog; no NPZ. |
| VI. No Game Client Path Assumptions | PASS | Client root stays runtime config (`H:\CLIENTS` approved per Rule 9). |
| Read-Only Reference Codebase | PASS | No writes to `gillijimproject_refactor`. |
| Format Reader/Writer Ownership | PASS | Reuses the existing strict rasterizer/placement readers; no new/duplicate ADT/WMO/M2 parser — the one addition paints ids in the existing loop. |
| Terrain Alpha Risk Area | N/A | No MCAL/alpha/shader changes. |
| AlphaWdtWriter Frozen | N/A | Not touched. |
| One Phase at a Time | PASS | US1 → US2 → US3, each independently validated; US2 null result stops the line honestly. |
| Spec Docs Source of Truth | PASS | This plan + spec.md + research.md decisions D-01..D-07. |
| Training Script Changes Require Validation | PASS | The `--object-mask-weight` flag is one isolated change per trainer, parity-defaulted, validated by paired dry-run + user-run comparison before any conclusion. |
| Bite-Sized Plans | PASS | ≤10 tasks per phase in tasks.md. |

**Gate verdict**: PASS, no violations, no Complexity Tracking entries required.

### Post-design re-check (after Phase 1 data-model + contracts)

Re-evaluated after `data-model.md` and `contracts/cli-contract.md` were written. All principles
still PASS. The bridge output satisfies the existing `--feature-store` contract by construction
(research D-06), so US3's integration adds zero trainer surface beyond US2's one flag.
**Post-design verdict: PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/118-object-occlusion-masks/
├── plan.md              # This file
├── research.md          # Phase 0 output (D-01..D-07)
├── data-model.md        # Phase 1 output (arrays, instance table, loss/target entities, schemas)
├── quickstart.md        # Phase 1 output (user-run commands)
├── contracts/
│   └── cli-contract.md  # Phase 1 output (CLI invocations + failure contracts)
└── tasks.md             # Phase 2 output (speckit-tasks — NOT created by this plan)
```

### Source Code (repository root)

```text
wow-viewer/
├── docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md   # +3 catalog rows (US1)
├── src/core/WowViewer.Core.IO/Maps/
│   ├── TerrainVisibleObjectMaskRasterizer.cs                           # + optional instance paint (US1)
│   ├── AdtTensorPackBuilder.cs                                         # + compact-id map + result field (US1)
│   ├── RawArraySerializer.cs                                           # + object_geometry_visible_instance_257 (US1)
│   └── NpzTileSerializer.cs                                            # + same, NPZ parity (US1)
├── tests/WowViewer.Core.Tests/
│   └── TerrainVisibleObjectMaskRasterizerTests.cs                      # + instance-paint tests (US1)
└── data-harvester/
    ├── v50_configs/                                                    # regenerated template + signals (US1)
    ├── src/harvester/spec118/
    │   ├── __init__.py
    │   ├── object_contract.py                                          # stage constants, run-record builder
    │   ├── object_mask_audit.py                                        # US1 audit → v118-object-mask-audit-v1
    │   ├── object_loss.py                                              # shared US2 weight/metric helpers
    │   ├── object_segment_model.py                                     # ObjectSegmentNet (US3)
    │   ├── object_segment_train.py                                     # trainer (US3)
    │   ├── object_segment_infer.py                                     # two-mode inference (US3)
    │   └── object_feature_bridge.py                                    # → v115-feature-map-v1 (US3)
    ├── src/harvester/v50/
    │   ├── model_stage_contract.py                                     # STAGES += "object_segmentation"
    │   ├── direct_geometry_train.py                                    # + --object-mask-weight (US2)
    │   └── geometry_detailer_train.py                                  # + --object-mask-weight (US2)
    ├── scripts/
    │   ├── spec118_audit_object_masks.py
    │   ├── spec118_train_objects.py
    │   ├── spec118_infer_objects.py
    │   └── spec118_objects_to_feature_map.py
    └── tests/spec118/                                                  # focused tests per slice
```

## Phase 0 → Phase 1 Traceability

- research.md D-01..D-07 resolve every unknown (visibility source, taxonomy, instance identity,
  catalog mechanics, loss pattern, US3 shape, thresholds).
- data-model.md pins array names/dtypes/shapes, eligibility semantics, the instance table, the
  loss-weight entity, the 3-class target, and all record schemas.
- contracts/cli-contract.md pins every CLI invocation and its failure behavior.
- tasks.md (next step, `speckit-tasks`) decomposes into Setup → Foundational → US1 → US2 → US3 →
  Polish with ≤10 tasks per phase.
