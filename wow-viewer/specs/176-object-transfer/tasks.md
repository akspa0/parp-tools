# Tasks: PM4-Guided Object Transfer and Museum Placement Repair

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)
**Status**: Phases 1–2 done; Phase 3/4 partial. Do not start P1 transfer until the Reconcile compile is clean.

Each step is one concern and independently validatable. Max 10 open steps.

## Open — next implementation slice

- [ ] **T001** Finish scene-discerned inputs so the viewer compiles. Remove the leftover `PrefillReconciliationPathsFromScene()` call in [`ViewerApp_Sidebars.cs`](../../src/viewer/WoWViewer/ViewerApp_Sidebars.cs). Delete unused `DrawReconciliationPathField` and unused `_reconciliationPm4Path` / `_reconciliationMuseumPath` if nothing reads them. Keep `WorldScene.LoadedPm4Tiles` + `TryResolveReconciliationPairs`. Validate: `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug`.
- [ ] **T002** After T001, drop the leftover **Build fingerprint** text box if the session already has `_dbcBuild`; record the fingerprint used in the provenance sidecar. Validate: focused editor tests still 81/81.
- [ ] **T003** Phase 3 step 3 — draw accepted/reviewable proposals as in-scene overlays (current vs proposed) without writing. Last committed scene stays visible. Validate: source check + user-owned visual on one tile.
- [ ] **T004** Move `RunReconciliationPreview` off the render thread and keep the last proposal snapshot while it runs (plan performance constraint). Validate: Preview on a multi-tile load does not freeze the frame.
- [ ] **T005** User-owned real pair: Preview → accept one align → Apply → reload output ADT in a fresh session; record hashes in research.md. Independent-reader proof is also user-owned.

## Parked until T001–T005

- [ ] **T006** P1 cross-tile / cross-era transfer (FR-001–FR-006). Separate phase; do not mix with Reconcile UX.
- [ ] **T007** Atomic multi-file apply failure leaves every target unchanged (SC-004).
- [ ] **T008** Migrate remaining WinForms `ShowFileDialogSTA` call sites to `ImGuiPathPicker` only when a surface still needs a picker. Reconcile should not.

## Done (do not re-implement)

- [x] Phase 1 proposal engine + adapter
- [x] Phase 2 `AdtPlacementEditor` (ID high-water mark, name tables, MODF bounds translation)
- [x] Delete `AdtPlacementWriter` / `AdtPlacementEditTransaction`
- [x] Reconcile tab; retire freezing Match tab; camera-scope leftover reports
- [x] Guarded apply + provenance sidecar + undo operation
- [x] Review UX: residual confidence, `AlreadyAligned`, bulk accept
- [x] Shared staged-save queue for authored placement edits
- [x] Timestamped project output folder
