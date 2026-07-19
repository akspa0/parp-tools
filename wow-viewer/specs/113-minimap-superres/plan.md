# Implementation Plan: Minimap Super-Resolution (Real-ESRGAN)

**Branch**: `113-minimap-superres` | **Date**: 2026-07-18 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/113-minimap-superres/spec.md`

## Summary

Three phases. Phase 1 (US1, the hard gate) adds a detail-preserving 1024 render mode to the terrain
minimap compositor — sampling real BLP texels instead of per-texture average colors — and proves
the resulting HR is (a) genuinely more detailed than a bicubic upscale of the material-average
render and (b) spatially registered to the authored client minimap for the same tile. If authored↔
detail registration cannot be established, the spec halts here with the finding rather than training
on invalid pairs. Phase 2 (US2) assembles a leak-safe (authored LR, detail HR) pair set from
Kalimdor and Azeroth. Phase 3 (US3) trains a Real-ESRGAN model that upscales real authored minimaps
and evaluates it against a bicubic baseline plus a user visual gate. All rendering-at-scale and
training runs are user-executed.

## Technical Context

**Language/Version**: C# / .NET 10 (detail render mode in `WowViewer.Core.IO`; the harvester already
owns minimap synthesis), Python 3.11+ / uv (alignment analysis, pair-set assembly, training, eval)

**Primary Dependencies**: existing `TerrainMinimapCompositor` / `TerrainTextureSampler` (extended,
not replaced — constitution II); `torch` (already a dep, CUDA 13.0 wheels); a Real-ESRGAN
implementation + an SR perceptual metric (LPIPS) — dependency-vs-vendor decided in research.md

**Storage**: per-build Zarr (constitution V). The detail HR upgrades the existing
`minimap_rgb_1024` signal's render semantics (same shape/coverage contract as Spec 112). The pair
set is a curriculum-style store referencing store rows — no NPZ

**Testing**: `dotnet test` for the detail-render mode (a focused test proving texel sampling +
no-moire behavior on a synthetic texture); `pytest` for alignment analysis, pair assembly, and the
trainer's contract/gate logic (CPU-safe). Real renders and training are user-run

**Target Platform**: Windows desktop; training on the user's local RTX 4070 Ti SUPER (16 GB) —
Real-ESRGAN is heavier than the Spec 112 height model and likely needs patch-based training

**Project Type**: Extension of the existing `wow-viewer` monorepo (compositor + `data-harvester`),
one new model lane; not a new project

**Performance Goals**: detail render must stay within a practical per-map render budget (it does
more work per pixel than material-average; a whole-map 1024 render is the heavy user-run step);
training is not latency-sensitive

**Constraints**: single-purpose SR model, no multi-task, no weight-sharing with terrain models
(constitution IV lens); training user-executed only (standing rule); render/pipeline must not
hardcode 1024 against a future scale bump (FR-011)

**Scale/Scope**: two maps (~700-950 tiles each, minus tiles lacking an authored minimap or a
successful detail render), one new render mode, one new model + trainer + eval, ×4 only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Repo Independence | All new files under `wow-viewer/`; SR deps are packages, never path refs | PASS |
| II. Library-First | Detail render extends `TerrainMinimapCompositor`/`TerrainTextureSampler` (the existing owner); no new/duplicate minimap renderer | PASS |
| III. Real-Data Validation | US1/US2/US3 validate against real `H:\CLIENTS` renders + authored minimaps with recorded commands, root, build id, hashes; SC-005 is a user visual gate | PASS |
| IV. Residual Model Chain | Governs terrain-signal residual models. The minimap SR model is a distinct single-purpose image model (SR only, one output, no multi-task, no shared weights) — it does not enter the terrain residual chain and does not violate it; recorded explicitly rather than force-fit | PASS (with note) |
| V. Streaming-First Pipeline | Detail render flows through the same harvest/synthesis path into Zarr; pair set references store rows; no NPZ | PASS |
| VI. No Client Path Assumptions | Renders/authored reads take a configured `--clients-root`; no hardcoded path | PASS |
| One Phase at a Time | US1 is a hard gate (can fail); US2/US3 blocked behind it in task ordering | PASS |
| Bite-Sized Plans | Each phase ≤10 tasks, enforced at /speckit-tasks | Deferred to tasks.md |

No violations requiring the Complexity Tracking table. The principle-IV note is a scope
clarification, not a waiver: nothing here adds a multi-task or weight-shared model.

## Project Structure

### Documentation (this feature)

```text
specs/113-minimap-superres/
├── plan.md              # this file
├── research.md           # Phase 0 output
├── data-model.md         # Phase 1 output
├── contracts/
│   ├── detail-render-contract.md
│   └── sr-pairset-and-run.schema.json
├── quickstart.md         # Phase 1 output
└── tasks.md               # /speckit-tasks output (not created by this command)
```

### Source Code (repository root: `wow-viewer/`)

```text
src/core/WowViewer.Core.IO/Maps/
└── TerrainMinimapCompositor.cs                 # add a detail-preserving texel-sampling render mode (US1)

tools/harvest/WowViewer.Tool.Harvest/
└── Program.cs                                   # synthetic-minimap gains a --detail flag for the 1024 pass (US1)

tests/WowViewer.Core.Tests/
└── TerrainMinimapDetailRenderTests.cs           # texel sampling + no-moire on a synthetic high-freq texture (US1)

data-harvester/
├── src/harvester/v50/
│   ├── minimap_alignment.py                     # authored<->detail registration analysis + corrective transform (US1)
│   ├── sr_pairset.py                            # leak-safe (authored LR, detail HR) pair-set builder (US2)
│   ├── sr_esrgan_model.py                        # RRDBNet generator (+ discriminator/losses) or dependency wrapper (US3)
│   └── sr_esrgan_train.py                        # user-run trainer + bicubic-baseline eval + summary (US3)
├── scripts/
│   ├── v50_analyze_minimap_alignment.py          # thin CLI (US1)
│   ├── v50_build_sr_pairset.py                    # thin CLI (US2)
│   └── v50_train_minimap_superres.py              # thin CLI (US3, user-run)
└── tests/v50/
    ├── test_minimap_alignment.py                  # registration + corrective-transform detection (US1)
    ├── test_sr_pairset.py                         # coverage honesty + leak-safe split (US2)
    └── test_sr_esrgan_train.py                    # gate/contract/summary logic, CPU-safe (US3)
```

**Structure Decision**: extend the existing compositor (one canonical minimap renderer, constitution
II) with a detail mode rather than a second renderer; keep the Python SR lane inside the established
`harvester/v50/` + `scripts/` convention used by Specs 109-112. The detail HR is not a new store
signal — it upgrades `minimap_rgb_1024`'s render semantics, so Spec 112's coverage/parity contract
carries over unchanged.

## Complexity Tracking

*No constitution violations identified; table not needed.*
