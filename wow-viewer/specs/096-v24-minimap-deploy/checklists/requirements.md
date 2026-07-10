# Specification Quality Checklist: 096-v24-minimap-deploy

**Purpose**: Validate specification completeness and quality before proceeding to planning and implementation.
**Created**: 2026-07-09
**Feature**: [`../spec.md`](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) leak into the user-facing sections. The "What This Spec Does" section names the files that will be touched because they are deliverables, not implementation choices. Architecture Sketch is in the User Story acceptance criteria, where the WHAT is mixed with concrete file paths.
- [x] Focused on user value and business needs. The spec exists to fix a real, user-reported gap: the minimap-to-prior deployment story is not wired.
- [x] Written for non-technical stakeholders. The Problem Statement, User Scenarios, and Success Criteria are in plain language. The technical slice breakdown is at the bottom under "What This Spec Does."
- [x] All mandatory sections completed. Spec has Problem Statement, User Scenarios, Requirements, Success Criteria, Out of Scope, Risks, Assumptions, Open Questions.

## Requirement Completeness

- [x] No `[NEEDS CLARIFICATION]` markers remain. The 3 open questions are at the end as "Open Questions (For User Review Before Plan)" with recommended defaults documented in the spec body.
- [x] Requirements are testable and unambiguous. Every FR-### and SC-### is a concrete, measurable check.
- [x] Success criteria are measurable. SC-096-001 through SC-096-007 have numeric or binary pass/fail conditions.
- [x] Success criteria are technology-agnostic. Hardware constraints are documented as "6 GB consumer GPU" / "12 GB envelope" which are properties of the success criterion, not tech choices.
- [x] All acceptance scenarios are defined. Each User Story has 4-7 numbered acceptance scenarios.
- [x] Edge cases are identified. A dedicated "Edge Cases" section under each User Story group.
- [x] Scope is clearly bounded. "What This Spec Does NOT Do (Explicit Out of Scope)" is explicit and exhaustive.
- [x] Dependencies and assumptions identified. The "Assumptions" section lists every dependency, and the "Risks" section enumerates the known failure modes.

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria. FR-### items map to specific User Story acceptance scenarios.
- [x] User scenarios cover primary flows. Four user stories cover: training the deployment model, deploying from a PNG, measuring the deployment regime, and validating against arbitrary minimap sources.
- [x] Feature meets measurable outcomes defined in Success Criteria. SC-096-001 through SC-096-007 are concrete.
- [x] No implementation details leak into the spec beyond the named file paths (which are deliverables, not implementation choices).

## Constitution Alignment (wow-viewer AGENTS.md)

- [x] **RULE 1 (read-only `gillijimproject_refactor`)**: Spec 096 does not touch `gillijimproject_refactor`. The C# WDL reader and the C# terrain→WDL path are inherited from Spec 094 and not modified.
- [x] **RULE 2 (all new code in `wow-viewer`)**: Spec 096 lives entirely under `wow-viewer/data-harvester/` (training script reused, new inference script, new tests, new docs). No other paths.
- [x] **RULE 3 (no rewrite of game client reading tooling)**: Spec 096 does not add any new WDL reader / terrain parser. It wraps the existing minimap-only Stage A model and adds a PNG entry point.
- [x] **RULE 4 (`wow-viewer` repo-independent)**: Spec 096 does not introduce any cross-repo imports.
- [x] **RULE 5 (one Python environment)**: All new code is under `wow-viewer/data-harvester/`. New script goes under `wow-viewer/data-harvester/scripts/`. New tests go under `wow-viewer/data-harvester/tests/v24/`.
- [x] **RULE 6 (no mutation of training scripts without a plan)**: Spec 096 reuses `train_v24_stage_a.py` exactly as it is. The only "new" training code is the inference script. The minimap-only training run is a documented bounded change.
- [x] **RULE 7 (small, modular, residual-predicting models)**: Spec 096 does not change the model architecture. `StageAMinimapOnly` is a separate `nn.Module` with 3 input channels and one output (a 17×17 + 16×16 prior). No multi-task heads, no shared weights. ≤ 1M params.
- [x] **RULE 9 (no `H:\CLIENTS`)**: Spec 096 reads from the V18 store at `output/datasets/v18/3_3_5_12340.zarr` (or the standard staged-client path). The PNG inference script does not touch the staged client at all.
- [x] **RULE 10 (AlphaWdtWriter frozen)**: Spec 096 does not touch `AlphaWdtWriter.cs`. The V24 WDL reader and writer are unchanged.
- [x] **RULE 11 (doc hygiene, plans bite-sized)**: Spec 096 is decomposed into 4 slices (train minimap-only, write inference script, validate, sync memory bank). Each slice is small, testable, and ends with a concrete validation step. The memory bank will be updated at slice completion.
- [x] **RULE 8 (one phase at a time)**: Spec 096 is structured as 4 sequential slices. Slice 1 ends with a trained checkpoint. Slice 2 ends with a working inference script. Slice 3 ends with a validation report. Slice 4 ends with a memory bank update. Each slice validates before the next.

## Memory Bank Updates Required

- [ ] `wow-viewer/memory-bank/activeContext.md` "WDL prior + lattice detailer lane (V24)" section will be updated with the minimap-only deployment training result, the inference script entry, and the SC-002-MINIMAP gate pass/fail (Slice 4).
- [ ] `wow-viewer/memory-bank/progress.md` will get a new 2026-07-09 entry summarising the slice, the metric, and the open question (Slice 4).

## Open Questions Tracked

- [x] Q1: Minimap-only training corpus scope — resolved by recommended default. Full 2,011-tile corpus for parity with the cheat regime validation. Will confirm in the plan.
- [x] Q2: Optional minimap cleaning on PNG input — resolved by recommended default. No cleaning by default; optional `--alpha-mask <npz>` flag.
- [x] Q3: Where the inference script lives — resolved by recommended default. `wow-viewer/data-harvester/scripts/infer_v24_stage_a_png.py`.

## Notes

- Spec 096 is intentionally narrow. It does one thing: wire the minimap-only deployment path. It does not touch Stage B deployment (that is a separate question because Stage B needs more than a PNG; see Out of Scope).
- The risk that the minimap-only regime underperforms the cheat regime is recorded as a real possibility (Risk 1) with a documented fallback (Spec 095 / 097). The honest failure mode is part of the spec, not a future embarrassment.
- The trained checkpoint is small (≤ 1M params) and can be committed to git. The V24 store paths and the curated corpus are existing artifacts from Spec 094 and are not rebuilt by this spec.
- Items marked incomplete require user input before proceeding to `plan.md` and implementation.
