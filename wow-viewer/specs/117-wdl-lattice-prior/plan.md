# Implementation Plan: WDL-Lattice Coarse Prior for Terrain Geometry

**Branch**: `117-wdl-lattice-prior` | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/117-wdl-lattice-prior/spec.md`

## Summary

The v50 coarse+detailer geometry chain is honestly validated for the first time this session: 56.1%
relief-region error reduction on a spatially-isolated held-out split. This feature adds one more
generated input to that chain — a per-tile 545-point WDL-scale height lattice (17×17 outer + 16×16
inner, the exact sampling Spec 108 FR-001 already defines and `TerrainWdlLattice` already computes
from real MCVT vertices), predicted from minimap RGB alone, one level coarser than the chain's
existing "coarse" stage. Delivered in priority order: (US1) export the lattice as a real v50 signal
— data plumbing, no model; (US2) prove a standalone minimap-only predictor can learn it at all,
scored on the honest held-out split, before any integration is attempted; (US3) feed the generated
lattice into the coarse stage, the detailer stage, or both, and let measured relief-region error —
not design preference — decide where it belongs.

The technical approach reuses the existing v50 Zarr store and harvest pipeline (one new exported
array, no new harvest pass), the Spec 116 spatially-isolated held-out split, and — critically — the
generated-feature-store input contract already built and proven this session on both the coarse
trainer (`direct_geometry_train.py`) and the detailer trainer (`geometry_detailer_train.py`). No
GAN, adversarial loss, or generative-image technique anywhere in this feature.

## Technical Context

**Language/Version**: C# / .NET 10 for the one signal-export addition (wiring the existing
`TerrainWdlLattice` into the harvester's selectable v50 signal set); Python 3.11+ managed by `uv`
under `wow-viewer/data-harvester/` for everything else (signal consumption, predictor, integration).

**Primary Dependencies**: PyTorch (predictor training/inference; CPU dry-run + user-run CUDA),
NumPy, Zarr v3 + PyArrow (store/index I/O). No new dependency beyond what Specs 114–116 already
added.

**Storage**: The existing v50 curriculum store gains one new array (`wdl_lattice_outer17`,
`wdl_lattice_inner16`, or equivalent — finalized in data-model.md) via a harvest signal-config
addition, not a new store. The standalone predictor's generated output is materialized into a
derived store bound to its checkpoint hash, mirroring `structure_materialize.py` /
`spec116_structure_to_feature_map.py`. No NPZ; Zarr remains the only on-disk artifact
(constitution V).

**Testing**: `pytest` under `data-harvester/tests/` (new `tests/spec117/` for the predictor/bridge;
a C#-side focused test for the signal export), Ruff clean, `python -m py_compile`. Dry-run-first
CLIs refuse to write without an explicit flag, matching every prior spec's convention.

**Target Platform**: CPU for signal export, dry-run validation, and standalone-predictor scoring;
CUDA for user-run predictor training and chain-integration training. No client-path assumptions —
the signal export reads from the already-configured harvest client root at harvest time only.

**Project Type**: One scoped C# addition to the existing harvest signal pipeline (library-first,
constitution II — no new tool, extends the existing selectable-signal mechanism) + a new Python
library package `harvester/spec117/` for the standalone predictor and generated-lattice bridge,
mirroring the Spec 116 package convention. **No changes required to the existing coarse or detailer
trainers** (`direct_geometry_train.py`, `geometry_detailer_train.py`): both already accept an
arbitrary `--feature-store`, and this feature's bridge output is designed to satisfy that existing
contract directly (see Decision D-01 below) rather than modify two already-validated trainers again.

**Performance Goals**: Signal export adds negligible harvest time (the underlying MCVT read already
happens; this is one more coarse resample of data already in memory). Standalone predictor stays in
the existing small-model class (~1–1.6M params, matching `HeightRelativeNet`/`StructureSlotNet`).
Time-to-signal preferred over architecture search.

**Constraints**: User executes all training and heavy rebuilds (spec FR-011); every training run
validates and prints its plan without training by default (FR-008); the predictor is independently
trained/checkpointed/promoted, no shared weights with the coarse or detailer stage (FR-007,
constitution IV); no GAN/adversarial/generative-image technique anywhere (FR-005); the standalone
predictor is scored only against the spatially-isolated held-out split (FR-004).

**Scale/Scope**: Same corpus as Spec 116/the current chain — the corrected dual-view v50 curriculum
(2,973 rows after the synthetic-lighting refresh), Kalimdor + Azeroth, build 0.5.3.3368. One new
545-sample-per-tile signal; one new small predictor; zero new architecture on the consuming side.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Repo Independence | PASS | All new code under `wow-viewer/`; C# addition in the existing `WowViewer.Core`/harvest tree, Python under `data-harvester/`. |
| II. Library-First | PASS | `TerrainWdlLattice` already lives in `WowViewer.Core`; this feature only widens what the harvester selects, no new tool. Python logic in `src/harvester/spec117/`; `scripts/` stay thin. |
| III. Real-Data Validation | PASS | Signal export validated against the real corrected v50 store; predictor trained/scored against real corpus rows on the real spatially-isolated split. |
| IV. Residual Model Chain | PASS | The standalone lattice predictor is one more single-output model (RGB → 545-sample lattice), independently checkpointed, feeding downstream stages as a generated input — not a shared-weight or multi-task addition. No tension: unlike Spec 116's per-slot structure head, this output is already a single dense field, the same output shape category as the existing coarse stage. |
| V. Streaming-First Dataset | PASS | No new NPZ; one new array added to the existing Zarr signal set via the existing streaming harvest path. |
| VI. No Game Client Path Assumptions | PASS | Signal export reads the already-configured client root at harvest time; no path is hardcoded. |
| Read-Only Reference Codebase | PASS | No writes to `gillijimproject_refactor`. |
| Format Reader/Writer Ownership | PASS | Reuses the existing `TerrainWdlLattice` reader; no new/duplicate MCVT or WDL parser. |
| Terrain Alpha Risk Area | N/A | No MCAL/alpha-packing/shader changes. |
| AlphaWdtWriter Frozen | N/A | Not touched. |
| One Phase at a Time | PASS | US1 (export) → US2 (standalone learnability) → US3 (integration) ordering; each validated before the next, mirroring Spec 116's own phase discipline. |
| Spec Docs Source of Truth | PASS | This plan + spec.md are the source of truth for this feature. |
| Training Script Changes Require Validation | PASS | Zero changes to existing trainers proposed (see D-01); the one new predictor's training script is new, scoped, and dry-run-validated before any CUDA run. |
| Bite-Sized Plans | PASS | Max 10 steps per phase; one concern per step, generated in tasks.md by `speckit-tasks`. |

**Gate verdict**: PASS, no violations, no Complexity Tracking entries required.

### Post-design re-check (after Phase 1 data-model + contracts)

Re-evaluated after `data-model.md` and `contracts/` were written. All principles still PASS.
Decision D-01 (below) confirms the integration point requires zero changes to
`direct_geometry_train.py`/`geometry_detailer_train.py` — the bridge output satisfies their
existing `--feature-store` contract by construction, so "Training Script Changes Require
Validation" has nothing new to validate on the consuming side. **Post-design verdict: PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/117-wdl-lattice-prior/
├── plan.md              # This file
├── research.md          # Phase 0 output (decision D-01 and any others)
├── data-model.md        # Phase 1 output (entities, store schema additions, run-record schema)
├── quickstart.md        # Phase 1 output (user-run commands with time/memory estimates)
├── contracts/           # Phase 1 output (CLI + JSON schema contracts)
└── tasks.md             # Phase 2 output (speckit-tasks — NOT created by this plan)
```

### Source Code (repository root)

```text
wow-viewer/
├── tools/harvest/WowViewer.Tool.Harvest/          # existing signal-selection wiring
│   └── (widen the existing tensor-pack signal set to include TerrainWdlLattice's
│        already-computed output; no new reader, no new tool)
└── data-harvester/
    ├── src/harvester/spec117/
    │   ├── __init__.py
    │   ├── lattice_model.py         # US2: standalone RGB -> 545-sample lattice predictor
    │   ├── lattice_train.py         # US2: dry-run-first trainer, held-out-split-only evaluation
    │   ├── lattice_bridge.py        # US3: generated lattice -> existing --feature-store contract
    │   └── lattice_contract.py      # schema validators + sha256 identity binding
    ├── scripts/
    │   ├── spec117_train_lattice.py
    │   └── spec117_lattice_to_feature_map.py
    └── tests/spec117/
        ├── test_lattice_model.py
        ├── test_lattice_train.py
        ├── test_lattice_bridge.py
        └── test_lattice_contract.py
```

**Structure Decision**: Single-project Python layout under `data-harvester/`, mirroring the
established `harvester.spec116` convention exactly — a spec-scoped package for what's genuinely new
(the lattice signal and predictor), with zero modification to the already-validated `harvester.v50`
coarse/detailer trainers. The one C# change is additive (widen an existing selectable signal set),
not a new reader or tool, consistent with constitution II.

## Complexity Tracking

*No violations. Constitution Check passed cleanly; this table is intentionally empty.*

## Phases

> Phases follow the spec's user-story priority (US1 export before US2 learnability before US3
> integration). Each phase ends with validation against the real, corrected v50 store. The user
> runs every training/heavy step; the assistant prepares scripts and hands off CLI invocations.

### Phase 0 — Research (this plan, Phase 0 output: research.md)

Resolve the one real open design question: how does a generated scalar-field lattice satisfy an
input contract (`--feature-store`) originally shaped for class-probability feature maps, without
either misusing that contract's semantics or requiring changes to two already-validated trainers.
Decision D-01, below.

### Phase 1 — Design & Contracts (this plan, Phase 1 output: data-model.md, contracts/, quickstart.md)

Entity model, store schema addition, generated-lattice-store schema, CLI contracts, user-run
quickstart. No code.

### Phase 2 — Tasks (speckit-tasks output: tasks.md)

Dependency-ordered, bite-sized implementation tasks. Generated by the `speckit-tasks` command, not
by this plan.

### Implementation phases (after tasks.md, executed one at a time)

- **Phase A — US1 signal export** (data plumbing, no model). Widen the harvester's signal
  selection to include `TerrainWdlLattice`'s already-computed output; add it to the v50 store
  writer. Validate: every tile with real height ground truth exports exactly 545 finite samples;
  gaps are excluded and counted, never fabricated.
- **Phase B — US2 standalone learnability** (model, user-run training). Build and train the
  minimap-RGB-only lattice predictor against the spatially-isolated held-out split only. Validate:
  lattice-point MAE reported against the trivial per-tile-mean lattice baseline; a plain
  learnable/not-learnable verdict is read before Phase C starts.
- **Phase C — US3 chain integration** (bridge + paired training, user-run). Materialize the frozen
  predictor's generated output, bridge it into the existing `--feature-store` shape via D-01, and
  run paired coarse/detailer training with and without it (coarse-fed, detailer-fed, both) on the
  identical held-out split against the already-established real baseline. Validate: relief-region
  MAE reported per condition; the report states plainly which feed point helped, if any.
