# Specification Quality Checklist: 097-v18-to-wdl-adt

**Purpose**: Validate specification completeness and quality before proceeding to planning and implementation.
**Created**: 2026-07-10
**Feature**: [`../spec.md`](../spec.md)

## Content Quality

- [x] No implementation details leak into user-facing sections. Architecture Sketch and key-entity descriptions are bounded to the data flow; the Slice 1 algorithm lives in `plan.md` where it belongs.
- [x] Focused on user value and business needs. The spec exists to close a real user-reported gap: round-trip the V18 dataset through the V24 prior pipeline and the viewer.
- [x] Written for non-technical stakeholders. The Problem Statement and User Scenarios are in plain language.
- [x] All mandatory sections completed. Spec has Problem Statement, User Scenarios, Requirements, Success Criteria, Out of Scope, Risks, Assumptions, Open Questions.

## Requirement Completeness

- [x] No `[NEEDS CLARIFICATION]` markers remain. The 3 open questions are at the end as documented design decisions with recommended defaults.
- [x] Requirements are testable and unambiguous. Every FR-### and SC-### is a concrete, measurable check.
- [x] Success criteria are measurable. SC-097-001 through SC-097-007 are concrete.
- [x] Success criteria are technology-agnostic where possible. The hardware constraint (6 GB GPU, 10 minutes) is a property of the success criterion, not a tech choice.
- [x] All acceptance scenarios are defined. Each user story has 3-5 numbered scenarios.
- [x] Edge cases are identified. A dedicated section covers non-64-aligned footprints, missing tiles, custom curated corpora, multi-build, and edge tiles.
- [x] Scope is clearly bounded. "What This Spec Does NOT Do" is explicit and exhaustive.
- [x] Dependencies and assumptions identified. "Assumptions" section lists every dependency.

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria. FR-### items map to specific User Story acceptance scenarios.
- [x] User scenarios cover primary flows. Four user stories cover: stitched mesh, WDL writer, ADT writer, round-trip smoke.
- [x] Feature meets measurable outcomes defined in Success Criteria. SC-001 through SC-007 are concrete.
- [x] No implementation details leak into the spec beyond the named file paths and the data-flow sketch (which is a data contract, not a tech choice).

## Constitution Alignment (wow-viewer AGENTS.md)

- [x] **RULE 1 (read-only `gillijimproject_refactor`)**: Spec 097 explicitly does not touch `gillijimproject_refactor`. The C# WDL/ADT readers are referenced as "do not modify"; V24 wraps them.
- [x] **RULE 2 (all new code in `wow-viewer`)**: Spec 097 lives entirely under `wow-viewer/data-harvester/`. New Python modules, new scripts, new tests, no other paths.
- [x] **RULE 3 (no rewrite of game client reading tooling)**: Spec 097 adds new WDL/ADT writers but does not modify the existing C# readers. The contract is "write files the readers can already open."
- [x] **RULE 4 (`wow-viewer` repo-independent)**: Spec 097 does not introduce any cross-repo imports.
- [x] **RULE 5 (one Python environment)**: All new code is under `wow-viewer/data-harvester/`. New modules go under `wow-viewer/data-harvester/src/harvester/v24/`. New scripts go under `wow-viewer/data-harvester/scripts/`.
- [x] **RULE 6 (no mutation of training scripts without a plan)**: Spec 097 does not modify the V24 trainer. It calls the trainer via the inference path.
- [x] **RULE 7 (small modular residual-predicting models)**: Spec 097 does not introduce new models. It reuses the existing Stage A and Stage B.
- [x] **RULE 9 (no `H:\CLIENTS`)**: Spec 097 reads from `output/datasets/v18/...` exclusively. No `H:\CLIENTS` references.
- [x] **RULE 10 (`AlphaWdtWriter` frozen)**: Spec 097 does not touch `AlphaWdtWriter.cs`. The WDT format is separate from WDL/ADT.
- [x] **RULE 11 (doc hygiene, plans bite-sized)**: Spec 097 is decomposed into 4 slices, each independently validatable. Memory bank will be updated at slice completion.
- [x] **RULE 8 (one phase at a time)**: Spec 097 is structured as 4 sequential slices. Each ends with a concrete validation gate.

## Memory Bank Updates Required

- [ ] `wow-viewer/memory-bank/activeContext.md` will get a Spec 097 entry at Slice 4 completion.
- [ ] `wow-viewer/memory-bank/progress.md` will get a 2026-07-10 entry at Slice 4 completion.

## Open Questions Tracked

- [x] Q1: Edge alignment algorithm — resolved by recommended default. Average-the-border; the low-pass is a future spec if smoothing is visible.
- [x] Q2: WDL writer location — resolved by recommended default. Python first; C# extension only if the format is too subtle for Python.
- [x] Q3: ADT minimal chunks — resolved by recommended default. MCNK + MCAL + MCNR + MCSH stub; other chunks follow-up if the viewer needs them.

## Notes

- Spec 097 is the second major milestone for V24 after Spec 096. It takes the deployment wiring and points it at a real per-map use case.
- The WDL/ADT writers are bounded, format-specific, and live in Python. If the byte layout diverges from the C# reader's contract, the test catches it and a small C# shim extension is the follow-up.
- The user is the proof owner. The round-trip succeeds when the user can load the output in the WoWViewer app and see their pre-alpha prior as a viewable surface.
- Items marked incomplete require user input before proceeding to implementation.
