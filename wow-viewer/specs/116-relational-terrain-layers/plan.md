# Implementation Plan: Relational Terrain Layer Reconstruction

**Branch**: `116-relational-terrain-layers` | **Date**: 2026-07-21 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/116-relational-terrain-layers/spec.md`

## Summary

Terrain reconstruction has been framed as continuous raster regression (minimap in, per-vertex
height out). Spec 116 reframes it: a terrain tile is a **serialized relational schema** — layer
entries are ordered rows, a layer's texture reference is a foreign key into that tile's own local
texture table, and the corpus is assembled from a discrete alphabet of reused pieces. The feature
delivers, in priority order: (US1) a measurement of how consistently each surface family occupies
each layer slot, fixing the output vocabulary before any model is trained; (US2) a measurement of
whether layer masks derive from terrain shape, fixing whether structure is derivable from geometry;
(US4) a spatially-isolated held-out set with relief-stratified evaluation that makes every later
result trustworthy; (US3) a structure predictor that emits legal, non-degenerate layer rows from a
minimap alone; and (US5) feeding predicted structure into height reconstruction.

The technical approach reuses the existing v50 Zarr store (no new harvest), the existing
surface-family taxonomy from Spec 115 (`terrain_feature_labels.py`, revision `v115.1`), and the
existing identity-binding / dry-run-first / model-stage-run contract pattern from Spec 114
(`model_stage_contract.py`). All new code is Python under `wow-viewer/data-harvester/`. The user
runs every training and heavy rebuild; the assistant prepares scripts and hands off CLI invocations.

## Technical Context

**Language/Version**: Python 3.11+ managed by `uv` under `wow-viewer/data-harvester/`. No C# is
required — the v50 store already carries every signal this feature consumes (`mcly_texture_ids`,
`mcly_layer_mask`, `mcly_tileset_ids`, `alpha_256`, `height_257`, `minimap_rgb`, `mcnk_flags_16`).

**Primary Dependencies**: PyTorch (model training/inference, CPU-dry-run + user-run CUDA),
NumPy, Zarr v3 + PyArrow (store/index I/O), scikit-learn (US2 non-linear explained-variance fit),
scipy (US2 bimodality test — Hartigan's dip test or KDE two-component mixture). All already
available or addable to `data-harvester/pyproject.toml` under the existing dependency policy.

**Storage**: Existing per-build v50 Zarr curriculum store (read-only input). New derived artifacts
written under `wow-viewer/output/datasets/` (or a feature-scoped subroot): a spatially-isolated
held-out split manifest (Parquet + JSON identity), analysis report JSONs (US1/US2), and — only when
a model is promoted — a predicted-structure derived store bound to its checkpoint hash. No NPZ; the
Zarr store remains the only on-disk artifact (constitution V).

**Testing**: `pytest` under `wow-viewer/data-harvester/tests/` (new `tests/spec116/`), Ruff clean,
`uv run python -m py_compile`. Focused tests per slice; full v50 suite re-run when a shared module
changes. Dry-run-first CLIs must refuse to write without an explicit flag.

**Target Platform**: CPU for all analysis (US1/US2) and dry-run validation; CUDA for user-run
training (US3/US5). No client-path assumptions; the configured client root is runtime config only
and is not needed at analysis/training time because the v50 store is already built.

**Project Type**: Shared library + thin CLI scripts. New library code in
`data-harvester/src/harvester/spec116/`; thin entrypoint scripts in `data-harvester/scripts/`.

**Performance Goals**: Analysis (US1/US2) completes in minutes on CPU over ~1.4k authored rows.
Model scale stays in the existing small-model class (~1–1.6M params), consistent with
`HeightRelativeNet`/`TerrainFeatureNet`. Time-to-signal preferred over exhaustive architecture
search.

**Constraints**: User executes all training and heavy rebuilds (FR-018); every training run
validates and prints its plan without training by default and requires explicit confirmation
(FR-015); each model is independently trained/checkpointed/promoted with no shared weights and no
multi-task heads (FR-014, constitution IV); ground-truth tables are never a prediction input
(FR-006); predicted references must be legal entries for the tile (FR-007); the always-opaque base
layer is excluded from any alpha stack (FR-008).

**Scale/Scope**: ~1,384 authored curriculum rows (Kalimdor + Azeroth, build 0.5.3.3368), 5 surface
families (unknown/terrain/road/water/structure), 4 layer slots (base + 3 detail), 16×16 chunks per
tile. The rarest structural class covers ~2% of locations.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Repo Independence | PASS | All new code under `wow-viewer/data-harvester/`; no references outside `wow-viewer/`. |
| II. Library-First | PASS | Logic in `src/harvester/spec116/`; `scripts/` are thin argparse wrappers. |
| III. Real-Data Validation | PASS | Every measurement/model validates against the real v50 store (build 0.5.3.3368). |
| IV. Residual Model Chain | **TENSION** | Constitution forbids multi-task heads / shared weights; US3 "layer structure" has multiple aspects (family per slot, coverage per slot). Resolved in research.md D-04: decompose into independent single-output models, one per slot, each its own checkpoint. |
| V. Streaming-First Dataset | PASS | No new harvest; consumes the existing v50 Zarr store. No NPZ. |
| VI. No Game Client Path Assumptions | PASS | Client root is runtime config only; not needed at analysis/training time. |
| Read-Only Reference Codebase | PASS | No writes to `gillijimproject_refactor`. |
| Format Reader/Writer Ownership | PASS | No new format readers; reuses existing v50 signals. |
| Terrain Alpha Risk Area | N/A | No MCAL decode / alpha-packing changes; reads `alpha_256` as-is. |
| AlphaWdtWriter Frozen | N/A | Not touched. |
| One Phase at a Time | PASS | US1→US2→US4→US3→US5 ordering; each phase validated before the next. |
| Spec Docs Source of Truth | PASS | This plan + spec are the source of truth; architecture doc created if needed. |
| Training Script Changes Require Validation | PASS | Each training script change is a separate testable commit with a validation path. |
| Bite-Sized Plans | PASS | Max 10 steps per phase; one concern per step. |

**Gate verdict**: PASS with one justified tension (IV), documented in the Complexity Tracking
table and resolved in research.md D-04 before any model code is written.

### Post-design re-check (after Phase 1 data-model + contracts)

Re-evaluated after `data-model.md` and `contracts/` were written. All principles still PASS. The
Principle IV tension is now **resolved by design**, not just justified: the data model defines one
`StructureSlotNet` per detail slot (1–3), each with its own checkpoint and `v50-structure-run-v1`
record, and the base slot (0) is never predicted (FR-008). The `structure-run.schema.json` enforces
`held_out_split.verified_violation_count == 0` (SC-005) and a `per_class_iou_recall` gate that
never references aggregate accuracy (D-08). No new gates were introduced by the design; no
principle regressed. **Post-design verdict: PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/116-relational-terrain-layers/
├── plan.md              # This file
├── research.md          # Phase 0 output (decisions D-01..D-0n)
├── data-model.md        # Phase 1 output (entities, store schemas, run-record schema)
├── quickstart.md        # Phase 1 output (user-run commands with time/memory estimates)
├── contracts/           # Phase 1 output (CLI + JSON schema contracts)
│   ├── cli-contract.md
│   ├── held-out-split.schema.json
│   ├── analysis-report.schema.json
│   └── structure-run.schema.json
└── tasks.md             # Phase 2 output (speckit-tasks — NOT created by this plan)
```

### Source Code (repository root)

```text
wow-viewer/data-harvester/
├── src/harvester/spec116/
│   ├── __init__.py
│   ├── relational_extract.py      # US1/US3: extract layer-entry rows from v50 store
│   ├── family_slot_consistency.py # US1: family→slot consistency measurement
│   ├── shape_coverage_coupling.py # US2: surface-shape→coverage non-linear fit + bimodality
│   ├── held_out_split.py          # US4: spatially-isolated held-out set builder
│   ├── relief_stratification.py   # US4: relief stratum + stratified metrics + trivial baseline
│   ├── structure_model.py         # US3: per-slot family classifier (one output each)
│   ├── structure_train.py         # US3: dry-run-first trainer, per-class IoU/recall gate
│   ├── structure_infer.py         # US3: predict + legality check/repair + audit record
│   ├── structure_materialize.py   # US5: frozen checkpoint → predicted-structure derived store
│   └── structure_contract.py      # JSON schema validators + sha256 identity binding
├── scripts/
│   ├── spec116_family_slot_consistency.py
│   ├── spec116_shape_coverage_coupling.py
│   ├── spec116_build_held_out_split.py
│   ├── spec116_train_structure.py
│   ├── spec116_infer_structure.py
│   └── spec116_materialize_structure.py
└── tests/spec116/
    ├── test_relational_extract.py
    ├── test_family_slot_consistency.py
    ├── test_shape_coverage_coupling.py
    ├── test_held_out_split.py
    ├── test_relief_stratification.py
    ├── test_structure_model.py
    ├── test_structure_train.py
    ├── test_structure_infer.py
    └── test_structure_contract.py
```

**Structure Decision**: Single-project Python layout under `data-harvester/`, mirroring the
established `src/harvester/v50/` + `scripts/` + `tests/` convention. Library code namespaced
`harvester.spec116` so it is independently importable and testable without coupling to the `v50`
package internals (it reads the v50 *store*, not the v50 *modules*, except for the reused
surface-family taxonomy from `harvester.v50.terrain_feature_labels`).

## Complexity Tracking

> **Constitution Check has one justified tension (Principle IV).**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple independent models for one "structure" prediction (vs one multi-task head) | US3 predicts layer structure as rows: a family per slot and coverage per slot. Constitution IV + FR-014 forbid multi-task heads and shared weights. | A single multi-task head sharing a trunk is the obvious "one model" design, but it violates the residual-chain constitution, prevents independent per-slot checkpoint replacement, and is exactly the monolithic pattern the project retired. Decomposing into one independent classifier per slot preserves independent trainability and matches how the corpus actually assembles tiles (each slot is an ordered row, not a frequency band). |
| Reusing the Spec 115 surface-family taxonomy as the structure vocabulary | US1 must decide family-keyed vs slot-keyed vocabulary; the family taxonomy already exists and the spec assumes it remains valid. | Re-deriving families from scratch would duplicate Spec 115 work and risk a vocabulary drift the spec explicitly forbids. US1 measures whether families map to slots consistently; it does not redefine families. |

## Phases

> Phases follow the spec's user-story priority (US1, US2 before any model; US4 before US3/US5 so
> evaluation is trustworthy). Each phase ends with validation against the real v50 store. The user
> runs every training/heavy step; the assistant prepares scripts and hands off CLI invocations.

### Phase 0 — Research (this plan, Phase 0 output: research.md)

Resolve all NEEDS CLARIFICATION and the Principle IV tension. No code.

### Phase 1 — Design & Contracts (this plan, Phase 1 output: data-model.md, contracts/, quickstart.md)

Entity model, store schemas, run-record schema, CLI contracts, user-run quickstart. No code.

### Phase 2 — Tasks (speckit-tasks output: tasks.md)

Dependency-ordered, bite-sized implementation tasks. Generated by the `speckit-tasks` command,
not by this plan.

### Implementation phases (after tasks.md, executed one at a time under code mode)

- **Phase A — US1 family→slot consistency** (analysis, no model). Deliver the vocabulary decision
  artifact consumed by US3. Validate: report runs over the real store; decision recorded.
- **Phase B — US2 shape→coverage coupling** (analysis, no model). Deliver the derivability
  decision. Validate: per-tile explained variance + bimodality finding reported.
- **Phase C — US4 trustworthy evaluation** (split + stratification). Build the spatially-isolated
  held-out set; verify zero edge/corner adjacency; re-score an existing model stratified by relief.
  Validate: violation count == 0; trivial baseline reported per stratum.
- **Phase D — US3 structure prediction** (model). Decompose into independent per-slot family
  classifiers per research D-04; train (user-run) on the Phase C split; gate on per-class IoU/recall,
  never aggregate accuracy; legality check/repair; OOD hand-painted image audit. Validate: SC-003,
  SC-004, SC-009.
- **Phase E — US5 feed structure into geometry** (model). Materialize predicted structure into a
  derived store; train height reconstruction with/without predicted structure on the same held-out
  set; compare relief-region error. Validate: SC-007 (first model to beat trivial baseline on
  relief-bearing regions) or an honest negative finding.