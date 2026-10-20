# Tasks: PM4 Format Comprehension and Evidence-Based Reference

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), [evidence contract](contracts/evidence-contract.md).

**Rule**: One validated phase at a time; every checked task needs a receipt containing files,
exact commands/status and criterion → real evidence. A research outcome can be inconclusive.

## Phase 0 — US1: Current reference (Gate R0)

- [x] T001 [US1] Audit all 16 parsed PM4 chunk signatures/raw fields against unchanged reader/models;
  inventory stale assertions in the old draft and adopt the Spec 270 evidence links.
- [x] T002 [US1] Write `docs/research/pm4-format-reference.md`: overview, confidence legend,
  structure/relationships, every supported layout, implications and complete unknown register.
- [x] T003 [US1] Route the old PM4/PD4 wiki draft and epic/status/triage to the new reference/spec,
  retaining old research as history rather than competing current authority.
- [x] T004 [US1] Validate chunk/field/evidence/local-link coverage and write the Phase 0 receipt.
- [x] T005 [US1] Synchronize live memory-bank handoff; certify Gate R0 only after the receipt exists.

T001–T005 receipt: [phase-0-reference-2026-10-20.md](evidence/phase-0-reference-2026-10-20.md).
**Gate R0 passed:** source/evidence-backed reference and continuity are validated; no full-decode claim.

## Phase 1 — US2: Raw link +0x04 experiment (Gate R1)

- [x] T010 [US2] Add `src/core/WowViewer.Core.PM4/Research/Pm4FieldDomainProbe.cs` and
  `tests/WowViewer.Core.PM4.Tests/Pm4FieldDomainProbeTests.cs` for explicit index/reverse-link
  alternatives with true/wrong-domain/missing-target controls. Requires Gate R0.
- [x] T011 [US2] Add a thin runner under this spec's `evidence/field-domain-probe/`; capture same-byte
  hashes and compact chunk/header/version inventory, not a large decoded-data export.
- [x] T012 [US2] Run focused controls and bounded full-map probe; report range fits separately from
  back-reference evidence and retain all incomplete/absent target cases.
- [x] T013 [US2] Update reference/research ledger with the supported/refuted/inconclusive result and
  exact receipt; certify Gate R1 without requiring a positive semantic discovery.

T010–T013 receipt: [phase-1-raw-04-2026-10-20.md](evidence/phase-1-raw-04-2026-10-20.md).
**Gate R1 passed:** general reverse-link-pointer hypothesis rejected; several range domains fit,
so meaning remains unknown. Phase 2 is planned, not certified or started by this checkbox.

## Phase 2 — US2: Authored-point/reference layer (Gate R2)

- [x] T020 [US2] Select and specify one next point/reference concern from the unknown register after
  Gate R1; give exact read/write scope, control and validation commands before coding.
  Receipt: [sub-step 2A](evidence/phase-2a-edge-ordinal-2026-10-20.md).
- [ ] T021 [US2] Test MSCN/MPRL point-domain/coordinate relationships against the addressed graph,
  with rival domains/axes; do not assume their legacy aliases identify editor path nodes.
  Active bounded sub-step 2A: `Pm4PointReferenceProbe` point/segment and independent subtype-edge
  controls, defined in plan.md. Completing 2A does not check all point-layer questions or Gate R2.
- [x] T021A [US2] Complete source-edge ordinal/point-candidate sub-step 2A with full-map spatial
  controls and auxiliary-geometry follow-up. [Receipt](evidence/phase-2a-edge-ordinal-2026-10-20.md).
- [ ] T021B [US2] Test +0x04 reciprocal-edge value/reuse alternatives in selected sub-step 2B,
  independently resolving reverse records and retaining ambiguous/missing targets; see plan.md.
- [ ] T022 [US2] Test MPRR record/run relationships in a separately validated sub-step.
- [ ] T023 [US2] Operator supplies/locates the relevant editor screenshots; verify any proposed
  authored-node correspondence, or leave that visual criterion explicitly unresolved.
- [ ] T024 [US2] Write per-question real-data receipts and update the reference before Gate R2.

## Phase 3 — US2: Auxiliary meanings and M2 ownership (Gate R3)

- [ ] T030 [US2] Select remaining header/plane/raw-word/auxiliary-state questions individually and
  define discriminating tests using the reference unknown register.
- [ ] T031 [US2] Test proposed M2 ownership separators against independently authored Museum
  copies/transforms/scales, including merged/equal-value counterexamples and false controls.
- [ ] T032 [US2] Record source-build/subset limits, failures and supported relationships; no field
  is called a global object ID without independent identity evidence.

## Phase 4 — US3: Evidence-gated consumer integration (Gate R4)

- [ ] T040 [US3] Produce exact viewer/matching consumer proposals tied to verified claims and baseline
  receipts; operator approves each particular behavior change before implementation.
- [ ] T041 [US3] Implement one approved owned-service/adapter slice with regression coverage, preserving
  unrelated behavior and the existing WMO baseline.
- [ ] T042 [US3] Operator provides real viewer/matching evidence; keep runtime/accuracy criteria open
  until satisfied. Update reference, receipts and continuity in the same pass.

## Dependencies and safe parallelism

T001 can use disjoint read-only source and stale-document reviews. T002–T005 are ordered behind
that inventory. Phase 1 cannot start before Gate R0. Later phases are gated, not an invitation to
implement all unknowns at once. Independent read-only discovery is allowed; heavy/client runs are
not delegated. PD4 generation and viewer annotation UI stay outside this child spec's immediate phase.
