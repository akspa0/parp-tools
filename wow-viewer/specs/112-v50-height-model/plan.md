# Implementation Plan: V50-Native Height-First Terrain Model with Dataset Corrections

**Branch**: `112-v50-height-model` | **Date**: 2026-07-18 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/112-v50-height-model/spec.md`

## Summary

Two phases. Phase 1 (US1/US2) corrects three real, code-cited defects in the v50.1 dataset pipeline
(a manifest template that contradicts the frozen signal catalog, a C# wiring gap that leaves
`mcnk_flags_16` permanently zero-filled for Alpha-format tiles, and a suspected data race in
concurrent MPQ reads that under-covers `minimap_rgb_1024`), rebuilds Kalimdor and Azeroth, and
produces a full-catalog training curriculum restricted to those two maps. Phase 2 (US3) trains a
small CNN that maps minimap RGB to a per-tile *relative* (altitude-invariant) height field,
replacing the rejected absolute-elevation `WdlPriorNet` lane. All training remains user-executed.

## Technical Context

**Language/Version**: C# / .NET 10 (harvester fixes), Python 3.11+ / uv (dataset + model)

**Primary Dependencies**: existing `WowViewer.Core.IO` readers (no new format parsers per
constitution II); `zarr` v3, `pyarrow` (dataset); `torch` (model, CUDA required for training)

**Storage**: per-build Zarr stores (constitution V) — no NPZ, no new storage technology

**Testing**: `dotnet test` for the C# fix (focused MCNK-flags/synthesis-race regression), `pytest`
for the Python catalog/curriculum/target-contract changes — matches every prior Spec 109/111 phase

**Target Platform**: Windows desktop (existing harvester/trainer environment); training runs on the
user's local RTX 4070 Ti SUPER (16 GB VRAM)

**Project Type**: Data pipeline correction + single-model training addition inside the existing
`wow-viewer` monorepo (`tools/harvest/`, `data-harvester/`) — not a new project

**Performance Goals**: not a latency-sensitive feature; dataset rebuild must stay within the
existing full-corpus wall-time envelope (Kalimdor 8-12 min, Azeroth 5-8 min, per Spec 109)

**Constraints**: constitution IV (one residual signal per model, no multi-task, no shared weights);
no DepthAnything-family architectures (standing memory); training is user-executed only (standing
memory, hardened 2026-07-18 after an execution-boundary violation this session)

**Scale/Scope**: two source maps (~421 + ~328 raw tiles before curation), one new model artifact,
one C# fix, three Python-side dataset/curriculum corrections

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design below.*

| Principle | Check | Result |
|---|---|---|
| I. Repo Independence | All new/changed files stay under `wow-viewer/` | PASS |
| II. Library-First | MCNK-flags fix lives in `WowViewer.Core.IO` (the existing terrain-tensor owner); no new parser, no duplicate reader | PASS |
| III. Real-Data Validation | US1 acceptance scenarios require rebuilding against `H:\CLIENTS` and re-running finalize/verify with recorded commands, root, build id, hashes | PASS |
| IV. Residual Model Chain | US3's model predicts exactly one signal (relative height); FR-007/FR-008 forbid embedding other targets or non-image inputs; growth to other signals is explicitly deferred to a future spec, not designed in now | PASS |
| V. Streaming-First Pipeline | Corrections operate on the existing harvest-stream → Zarr path; curriculum builder (already merged, Spec 109) also streams from Zarr, no NPZ | PASS |
| VI. No Client Path Assumptions | Rebuild commands take `--clients-root` / configured root as parameters, matching existing `v50_build_dataset.py` conventions; no hardcoded path | PASS |
| Development Workflow — One Phase at a Time | Task ordering (see tasks.md once generated) gates Phase 2 model work behind Phase 1's validated, re-audited corpus | PASS (encoded in task dependencies, not code) |
| Development Workflow — Bite-Sized Plans | Each phase's task list must stay ≤10 steps, one concern per step (enforced at /speckit-tasks time) | Deferred to tasks.md |

No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/112-v50-height-model/
├── plan.md              # this file
├── research.md           # Phase 0 output
├── data-model.md         # Phase 1 output
├── contracts/            # Phase 1 output
│   ├── relative-height-target-contract.md
│   └── coverage-audit-report.schema.json
├── quickstart.md         # Phase 1 output
└── tasks.md               # /speckit-tasks output (not created by this command)
```

### Source Code (repository root: `wow-viewer/`)

```text
tools/harvest/WowViewer.Tool.Harvest/
└── Program.cs                                  # synthesis race fix (US1)

src/core/WowViewer.Core.IO/Maps/
├── AlphaTensorPackBuilder.cs                    # set McnkFlags16 on the output pack (US1)
└── ... /Files/NativeMpqService.cs               # synchronize the shared archive-scan cache (US1)

tests/WowViewer.Core.Tests/                      # focused regression tests for both C# fixes

data-harvester/
├── docs/architecture/                            # or wherever the catalog lives: catalog/template
│   └── v50-clean-room-dataset-repo-audit-2026-07-15.md   # catalog stays source of truth (existing doc)
├── v50_configs/
│   └── v50-signals-0_5_3_3368.json               # regenerated from the catalog, not hand-edited (US1)
├── scripts/
│   ├── v50_generate_manifest_template.py         # NEW: derives the manifest template from the frozen catalog (US1)
│   ├── v50_audit_signal_coverage.py               # NEW: per-signal coverage report against SC-001/SC-002 (US1)
│   ├── v50_build_training_curriculum.py           # EXISTING (Spec 109): gains a maps allow-list restricted to Kalimdor/Azeroth (US2)
│   └── v50_train_height_relative.py               # NEW: canonical v50 entry point for the height-first model (US3)
├── src/harvester/v50/
│   ├── contracts.py                               # gains UnavailableSignal reason vocabulary (era_unavailable, no_source_data) (US1)
│   ├── training_curriculum.py                     # EXISTING: map allow-list enforcement (US2)
│   └── height_relative_model.py                   # NEW: target contract (encode/decode) + lean CNN (US3)
└── tests/v50/
    ├── test_manifest_template_matches_catalog.py   # NEW (US1)
    ├── test_training_curriculum.py                 # EXISTING: extend for map allow-list (US2)
    └── test_height_relative_model.py               # NEW: target invariance property test (US3)
```

**Structure Decision**: extends the existing Spec 109 `harvester/v50/` package and `data-harvester/scripts/` CLI convention rather than introducing a new project; the C# fix lands in the two files that already own MCNK-flags assembly and MPQ archive access, per constitution II (one canonical owner per format surface) — no new reader is written.

## Complexity Tracking

*No constitution violations identified; table not needed.*
