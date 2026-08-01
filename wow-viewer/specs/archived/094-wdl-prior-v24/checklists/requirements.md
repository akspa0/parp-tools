# Specification Quality Checklist: 094-wdl-prior-v24

**Purpose**: Validate specification completeness and quality before proceeding to planning and implementation.
**Created**: 2026-07-06
**Feature**: [`../spec.md`](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) leak into the user-facing sections. Implementation details are confined to the Architecture Sketch and Key Entities sections, which are explicitly labeled "for review; not implementation detail."
- [x] Focused on user value and business needs. The spec describes a research slice with measurable success criteria, not a feature roadmap.
- [x] Written for non-technical stakeholders. The Problem Statement, User Scenarios, and Success Criteria are written in plain language. The "what" and "why" are separated from the "how."
- [x] All mandatory sections completed. The spec has Problem Statement, User Scenarios, Requirements, Success Criteria, Out of Scope, Risks, Assumptions, and Open Questions.

## Requirement Completeness

- [x] No `[NEEDS CLARIFICATION]` markers remain. The 3 open questions in the spec are at the end as "Open Questions (For User Review Before Plan)" with recommended defaults documented in the spec body.
- [x] Requirements are testable and unambiguous. Every FR-### and SC-### is a concrete, measurable check. Example: "SC-002: Stage A's L1 on real-WDL cells < L1 on synthetic-only cells < `block_reduce` baseline L1."
- [x] Success criteria are measurable. Every SC-### has a numeric or binary pass/fail condition.
- [x] Success criteria are technology-agnostic. No mention of PyTorch / TensorFlow / specific GPU models in the success criteria. Hardware constraints are documented as "6 GB consumer GPU" which is a property of the success criterion, not a tech choice.
- [x] All acceptance scenarios are defined. Each User Story has 4-6 numbered acceptance scenarios.
- [x] Edge cases are identified. A dedicated "Edge Cases" section under each User Story group.
- [x] Scope is clearly bounded. "Out of Scope" section is explicit and exhaustive.
- [x] Dependencies and assumptions identified. The "Relationship To Existing Specs" and "Assumptions" sections list every dependency.

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria. FR-### items map to specific User Story acceptance scenarios.
- [x] User scenarios cover primary flows. Five user stories cover: prior coverage, minimap cleaning, Stage A training, Stage B training, validation.
- [x] Feature meets measurable outcomes defined in Success Criteria. SC-001 through SC-008 are concrete.
- [x] No implementation details leak into specification. Confirmed: the spec describes inputs, outputs, and behaviors, not specific Python classes or torch.nn.Module configurations.

## Constitution Alignment (wow-viewer AGENTS.md)

- [x] **RULE 1 (read-only `gillijimproject_refactor`)**: V24 explicitly does not edit `gillijimproject_refactor`. The C# WDL reader and the C# terrain→WDL path are referenced as "do not modify" — V24 wraps them.
- [x] **RULE 2 (all new code in `wow-viewer`)**: V24 lives entirely under `wow-viewer/data-harvester/`. New Python modules, new scripts, new Zarr schema, no other paths.
- [x] **RULE 3 (no rewrite of game client reading tooling)**: V24 wraps the existing C# WDL reader and the existing C# terrain→WDL path. No new WDL parser in Python. No re-implementation of format readers.
- [x] **RULE 4 (`wow-viewer` repo-independent)**: V24 does not introduce any cross-repo imports. Python module `v24` is self-contained. C# shim is a tiny CLI that calls the existing in-repo C# WDL reader.
- [x] **RULE 5 (one Python environment)**: All V24 Python code is under `wow-viewer/data-harvester/`. New modules go under `wow-viewer/data-harvester/src/harvester/v24/`. New scripts go under `wow-viewer/data-harvester/scripts/`.
- [x] **RULE 6 (no mutation of training scripts without a plan)**: V24 introduces new training scripts, not modified versions of existing ones. The plan.md will document each slice as a separate, testable change.
- [x] **RULE 7 (small, modular, residual-predicting models)**: V24 has two models, both height-only. Stage A predicts WDL prior directly (not a multi-task head). Stage B predicts a residual (height_257 - upsampled_prior). No shared weights. ≤ 3M total params.
- [x] **RULE 9 (no `H:\CLIENTS`)**: V24 uses `output/tmp/wowarchive-clients/` exclusively. The `build_wdl_prior.py build` script takes `--staged-client <path>` which is expected to be a staged-client path under the wow-viewer output tree.
- [x] **RULE 10 (AlphaWdtWriter frozen)**: V24 does not touch `AlphaWdtWriter.cs`. The C# WDL reader is the relevant surface; AlphaWdtWriter is the WDT writer (a different file).
- [x] **RULE 11 (doc hygiene, plans bite-sized)**: The spec itself is one file. The plan.md will break it into bite-sized slices. The memory bank will be updated at session end per the doc-hygiene skill.
- [x] **RULE 8 (one phase at a time)**: The spec is structured as 7 sequential phases (C# shim → synthetic WDL builder → merged prior → minimap cleaning → Stage A → Stage B → validation). Each phase ends with a validation check (test passes, dataset is built, etc.) before the next phase starts.

## Memory Bank Updates Required

- [x] Updated `wow-viewer/memory-bank/activeContext.md`: added "WDL prior + lattice detailer lane (V24)" section.
- [x] Updated `wow-viewer/memory-bank/progress.md` with a 2026-07-06 entry summarizing the implementation and validation results.

## Open Questions Tracked

- [x] Q1: C# Python shim form — resolved. Small CLI shim (`WowViewer.Tool.WdlRead`, batch-first `read`/`synth` modes), called via subprocess from `harvester/v24/shim.py`.
- [x] Q2: Minimap cleaning quality — resolved. NumPy 8-connected median cleaner shipped (`harvester/v24/clean_minimap.py`), amended to prefer the viewer-rendered `no_object_minimap` where the V18 store carries it (0_5_3_3368 only).
- [x] Q3: WDL grid shape confirmation — resolved by the Phase 0 audit. 17×17 outer + 16×16 inner int16 on both `3_3_5_12340` and `0_5_3_3368`; MAHO is not exposed by the C# reader. See `docs/architecture/wdl-reader-shape-audit-2026-07-06.md` and spec.md's "Implementation Amendments" (A1).

## Notes

- The spec is intentionally research-shaped. It is not a production cutover. SC-006 is the gate that proves the v7 idea works on a clean V18 substrate with a now-complete WDL prior.
- The v7 model is deleted. There is no v7 to compare against. The success criteria are self-consistency against the trivial `block_reduce(height_257)` baseline, not a v7 L1 number.
- V24's relationship to V23 (Spec 089) is "separate lane, not a replacement." If V24 proves it should replace V23, that is a separate spec filed later.
- V24's relationship to V22 (Spec 088) is "V22 is out of scope." V24 consumes V18 directly. If V22's per-object mask data is ever verified to work, a future spec can promote it to per-object minimap cleaning.
- The synthetic-WDL builder is not a research problem in this spec — it is a thin wrapper around the existing C# terrain→WDL path that the user has already implemented for the "click on map to spawn" visualization.
- Items marked incomplete require user input before proceeding to `plan.md` and implementation.
