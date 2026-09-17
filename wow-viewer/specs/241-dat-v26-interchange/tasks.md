# Tasks: DAT v26 as the Project Interchange Format

**Input**: `specs/241-dat-v26-interchange/spec.md` · **Release**: v0.6 · **Branch**: `v0.5.4-dev`
**Status**: spec draft; plan to follow. Paths relative to `wow-viewer/`.

## Already done (Spec 237, commits d78cca68..f1138874)

- [x] T000a Full v26 decode with byte-identical rewrite, 700/700 (`AdtAhdrWriter`, `inspect adt-ahdr roundtrip`)
- [x] T000b ADT → DAT builder through measured frames (`AdtAhdrTileBuilder`), viewer export of nearby tiles
- [x] T000c Weight ↔ sequential alpha conversion (`AdtAhdrAlpha`)

## Phase 1: Decide the lossless carrier (blocks everything)

- [ ] T001 Operator runs the viewer export on a real map and the checks in docs/architecture/adt-v26-format.md ("Writer and round trip"); record results in specs/241-dat-v26-interchange/evidence/export-check.md
- [ ] T002 Plan: sidecar vs extension chunks for liquids, holes, area IDs, MCSE, MFBO, doodad sets, MTXP; record the decision and reason in specs/241-dat-v26-interchange/plan.md

## Phase 2: DAT → ADT (US2)

- [ ] T003 [US2] `AdtAhdrTile` (+ carrier) → the Spec 234 LK/split ADT writer inputs in src/core/WowViewer.Core.IO/Maps/
- [ ] T004 [US2] ADT → DAT → ADT round-trip report (per field, per tile) as an inspect command
- [ ] T005 [US2] SC-002 on 25+ tiles from two eras; evidence/roundtrip-adt.md

## Phase 3: Editor saves (US1)

- [ ] T006 [US1] Save-as-DAT for edited tiles from the editor/cartography save path (Spec 234 save targets)
- [ ] T007 [US1] Open a DAT save as an editable session (not only the read-only DAT viewer adapter)
- [ ] T008 [US1] SC-001 witness

## Phase 4: Harvest (US3)

- [ ] T009 [US3] DAT folder as a harvest terrain source (heights ÷ 36, weights → alpha, ACDO placements) feeding the existing streams
- [ ] T010 [US3] SC-003 tensor comparison ADT vs DAT export
