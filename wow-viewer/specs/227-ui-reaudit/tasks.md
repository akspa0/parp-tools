# Tasks: UI Re-Audit — Sidebar Standardization & Deduplication

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), [quickstart.md](quickstart.md)

**Governing rule**: a checked task must link a receipt that names changed files, commands and exit
status, and criterion-to-evidence mapping. Build/test output does not close visual, input, or
teleport behavior. No source consolidation is authorized until the corresponding inventory v2 row
names its replacement.

## Phase 1: Inventory baseline (US1 — P0)

**Goal**: establish an evidence-gated inventory v2 before any UI consolidation.

**Independent test**: source scan covers each declared visible workbench root and its named duplicate
family; the operator screenshot matrix either records every row or keeps the visual gate open.

- [x] T001 [US1] Create the dated source-audit inventory and screenshot matrix in `specs/227-ui-reaudit/surface-inventory-v2.md`. Receipt: `evidence/t001-source-audit.md`.
- [x] T002 [US1] Compare every Spec 223 retire/merge disposition against its current source route and record unresolved rows in `specs/227-ui-reaudit/surface-inventory-v2.md`. Receipt: `evidence/t002-spec223-reconciliation.md`.
- [ ] T003 [US1] Capture the current-build, profile-by-profile screenshot and interaction matrix in `specs/227-ui-reaudit/evidence/screenshots/` and link the records from `specs/227-ui-reaudit/surface-inventory-v2.md`.
- [ ] T004 [US1] Re-check the inventory gate and record the criterion-to-evidence receipt in `specs/227-ui-reaudit/evidence/t004-inventory-gate.md`.

**Checkpoint**: no source edit is permitted until T004 has a source receipt and every changed
surface has a named authoritative replacement; runtime claims still require T003.

## Phase 2: One sidebar surface at a time (US2 — P0)

**Goal**: standardize the first inventory-approved Archaeology or Editor surface without changing
its existing action or route authority.

**Independent test**: the selected surface uses `SharedUiWidgets` for its section/action
presentation, builds, and remains reachable from its documented profile route.

- [ ] T005 [US2] Select the first inventory-approved surface and add its source path, replacement row, and three-interaction route to `specs/227-ui-reaudit/surface-inventory-v2.md`.
- [ ] T006 [US2] Replace only the selected surface's bespoke sidebar presentation with existing `SharedUiWidgets` primitives in its inventory-named source file.
- [ ] T007 [US2] Run the affected focused test suite and `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; record the receipt in `specs/227-ui-reaudit/evidence/`.
- [ ] T008 [US2] Record the operator screenshot and reachability result for the selected surface in `specs/227-ui-reaudit/evidence/screenshots/`.

**Checkpoint**: do not standardize a second surface before the first has source/build evidence and
the operator validation result is recorded.

## Phase 3: Duplicate-family reductions (US3 — P0)

**Goal**: make one documented surface authoritative for each operator-named duplicate family.

**Independent test**: each formerly duplicate entry routes to exactly one named owner, and the
operator can verify the owner from a current build.

- [ ] T009 [US3] Map weak-signal amplifier entry points to the Spec 194 owner in `specs/227-ui-reaudit/surface-inventory-v2.md` before converting any duplicate entry to a route/link.
- [ ] T010 [US3] Apply the approved weak-signal route/link or retirement in the inventory-named source file and record a focused build receipt in `specs/227-ui-reaudit/evidence/`.
- [ ] T011 [US3] Run the minimap click/teleport matrix and select the canonical host in `specs/227-ui-reaudit/surface-inventory-v2.md`; do not infer the result from shared renderer code.
- [ ] T012 [US3] Apply the inventory-approved minimap route/link or interaction repair in the inventory-named source file and record source/build plus operator input evidence in `specs/227-ui-reaudit/evidence/`.
- [ ] T013 [US3] Map each Inspector object type and duplicate readout to its single authoritative page in `specs/227-ui-reaudit/surface-inventory-v2.md`.
- [ ] T014 [US3] Apply the inventory-approved Inspector route/link or retirement in the inventory-named source file and record focused source/build and operator evidence in `specs/227-ui-reaudit/evidence/`.

## Phase 4: Approachability and cross-cutting validation (US4 — P2)

**Goal**: surface Load, fog, wireframe, capture, and inspect from the accepted inventory and make
the guide reflect the actual viewer.

- [ ] T015 [US4] Record the accepted per-profile action order and corresponding inventory rows in `specs/227-ui-reaudit/surface-inventory-v2.md`.
- [ ] T016 [US4] Update the post-audit UI guidance in `wow-viewer/docs/WoWViewer/USERGUIDE.md` from the accepted inventory.
- [ ] T017 [US4] Record the final profile-by-profile screenshot comparison and acceptance receipt in `specs/227-ui-reaudit/evidence/`.

## Dependencies and execution order

`T001 → T002 → T003 → T004 → T005–T008 → T009–T014 → T015–T017`.

- T009 and T013 can be source-audited in parallel after T004; neither authorizes a source edit
  until its own replacement row is complete.
- T011 is operator-owned and blocks T012 only; it does not allow a weak-signal or Inspector edit.
- Phase 4 starts only after the desired duplicate-family slices are accepted.

## Implementation strategy

The first deliverable is the source-audit inventory, not a new UI implementation. It makes the
first subsequent source mutation small, scoped, and reviewable. Stop after each phase's receipt;
never claim a live UI result from a build alone.
