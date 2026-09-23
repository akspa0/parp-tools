# Batch F1 — Editor Platform (166-178) + Client Datastore (179-183)

Audited against: `epic-editor-platform/epic.md`, `epic-client-datastore/epic.md`, and each spec's
`spec.md` (only 176 has plan/tasks/etc.; the rest are spec-only drafts). Code verified under
`src/core/WowViewer.Core.Editor/`, `src/viewer/WoWViewer/ViewerApp_Editor.cs`,
`tests/WowViewer.Core.Editor.Tests/`, `src/core/WowViewer.Core.IO/Files/{MpqArchiveCatalog,NativeMpqService}.cs`,
`data-harvester/src/harvester/{zarr_io.py,v25/dataset.py}`.

**Headline finding**: A real Editor plugin host exists in `src/core/WowViewer.Core.Editor/` (host,
registry, session/undo, bridge types, integrity-gate shell, placement operations, chunk-transposition
service) and is wired into the viewer via `ViewerApp_Editor.cs`/`EnsureEditorHost()`. But it sits
**beside**, not **in place of**, the god-object state the epic measured (166-178's whole reason for
being). `_chunkClipboard*`/`_selectedChunks` in `ViewerApp.cs` measured **124** today — the epic's
"Today" baseline, unchanged. `_stagedPlacementEdits`/`_selectedPlacement*` measured **116** (close to
the baseline 112) and is still the file-write path `DrawPlacementAuthoringPanel` calls
(`StageAuthoringPlacementEdit`/`DrawPlacementSaveQueueActions`, both in `ViewerApp.cs`), with the new
`EditorSession.RecordApplied` call bolted on beside it for undo bookkeeping only. `EditorSession.SaveAll()`
does not write any file — it only clears the dirty set and logs. `EditorApplierAdapter.Apply()` only
reverses `ReconciliationApplyOperation` and `PlacementMoveOperation`; `PlacementRotateOperation`/
`PlacementScaleOperation`/`PlacementDeleteOperation` and every chunk-transposition operation hit the
`default: break` case — Undo silently no-ops for them today.

Separately, the chunk-manipulator/multi-tile-transposition UI in `ViewerApp_Editor.cs` is **not** the
169 migration: it is a new capability built under spec 195 (superseded by 219/222 per
`memory-bank/archive/2026-09-06-pre-context-cleanup-progress.md:28`) alongside the untouched old
sidebar clipboard — a second implementation, not the promised one.

## Datastore epic (179-183): zero landed code

Both measured "two implementations" facts from the epic are unchanged: `MpqArchiveCatalog.cs` and
`NativeMpqService.cs` both still exist (179 not started); `data-harvester/src/harvester/zarr_io.py:33`
still defaults to `zstd clevel=5 shuffle=bitshuffle` while `v25/dataset.py:62` still defaults to
`lz4 clevel=1` (182 not started). `InputSha256` is still only written into the harvest manifest
(`tools/harvest/.../Program.cs:748`), never compared (181 not started). No multi-build content-addressed
store exists (180). No "Load Zarr Datastore" entry point exists in the viewer (183) — the closest
relative, `RosettaDatastoreTerrainAdapter`/`_showRosettaDatastoreDialog`, is a distinct, narrower
object-library-based feature, not full multi-build client-equivalent loading through `IDataSource`.

---

### 166 Editor Plugin Host
- Stated status: Draft | Tasks: no tasks.md
- Scope: An Editor destination listing registered plugins, host/registry/lifecycle only; reference plugin proves registration.
- Verified implemented: `EditorHost` (register/activate/deactivate/update/draw/dispose, fault containment, era-based availability recompute) at `src/core/WowViewer.Core.Editor/EditorHost.cs`; `EditorPluginRegistry` (duplicate-id fails at registration) at `Plugins/EditorPluginRegistry.cs`; `EditorPluginAvailability`/`EditorPluginDescriptor`/`IEditorPlugin`/`ReferenceEditorPlugin`; era resolution at `Eras/EraHandlerResolver.cs`; wired into viewer at `ViewerApp_Editor.cs:EnsureEditorHost()`/`DrawEditorContent()` with 3 plugins registered (`ReferenceEditorPlugin`, `TerrainTemplateEditorPlugin`, `ChunkManipulatorEditorPlugin`); tests at `tests/WowViewer.Core.Editor.Tests/EditorHostTests.cs` (178 lines), `Eras/EraHandlerResolverTests.cs`.
- Partial: FR-006 (plugin input scoping vs existing viewer bindings) not independently verified; FR-005 (CASC-ready `IDataSource` abstraction) not exercised by a stub source.
- Not implemented: none found missing from the core contract.
- Checkbox accuracy: no tasks.md to check.
- Operator gates owed: none blocking — this is pure code/logic, testable without a real client.
- Open residue (spec-stated only): FR-006 input-scoping verification; SC-003 (stub data-source-kind consumed with no plugin changes) unverified.
- Superseded by / overlaps: none.
- Disposition: ARCHIVE-COMPLETE
- Confidence: high

### 167 Editor ↔ Runtime Bridge
- Stated status: Draft | Tasks: no tasks.md
- Scope: One boundary for reading live scene state and applying operations; proven via translation-only placement moves through `AdtPlacementWriter`, retiring `ViewerApp`'s 112-ref staging.
- Verified implemented: `IEditorSceneReader`/`EditorSceneSnapshot`/`EditorCamera`/`EditorLoadedTile`/`EditorSelectionEntry` at `Bridge/`; `IEditorOperationApplier` at `Session/`; viewer-side adapters `EditorSceneReaderAdapter`/`EditorApplierAdapter` in `ViewerApp_Editor.cs`; `PlacementMoveOperation` applies through the bridge and reverses via `_worldScene.TryUpdateSelectedPlacementPosition`.
- Partial: `EditorSceneReaderAdapter.Capture()` always passes `LoadedTiles: []` — FR-001's "loaded tiles" is not actually surfaced. `EditorApplierAdapter.Apply()` only handles 2 of 5+ operation types found in the operations folder (move, reconciliation) — everything else silently no-ops on apply/undo.
- Not implemented: **FR-007 "ViewerApp's parallel staging implementation is removed, not wrapped" is violated** — `_stagedPlacementEdits`/`_selectedPlacement*` still measured at 116 references in `ViewerApp.cs`/`ViewerApp_Workspaces.cs` (baseline was 112) and is still the actual write path `DrawPlacementAuthoringPanel` calls.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001 (move+undo restoring file+viewport together), SC-004 (build/run with Editor projects removed) — neither exercised in this audit.
- Open residue (spec-stated only): FR-007 retirement of `_stagedPlacementEdits`; SC-002 (112→0, currently 116); wiring `LoadedTiles` into the snapshot; completing `EditorApplierAdapter` for all operation kinds.
- Superseded by / overlaps: 175 (placement authoring UI actually built on top of this half-wired bridge), 176 (reconciliation apply is the only operation type fully wired end-to-end).
- Disposition: FOLD
- Confidence: high

### 168 Editor Session — Undo, Dirty State, Save Arbitration
- Stated status: Draft | Tasks: no tasks.md
- Scope: One cross-plugin undo/redo history, aggregated dirty state, save policy, write-guarding against game installs/containers.
- Verified implemented: `EditorSession` at `Session/EditorSession.cs` — `Undo()`/`Redo()`/`RecordApplied()`/`DirtyPlugins`/`HasUnsavedChanges`/`GuardWritablePath()`/`IsContainerPath()` (`.mpq` refusal)/`ResolveOutputPath()`; wired into UI (`Undo`/`Redo`/`Save All` buttons in `DrawEditorContent`); tests at `Session/EditorSessionTests.cs` (148 lines).
- Partial: the mechanism is solid and tested in isolation, but production usage is narrow — `_editorSession.RecordApplied()` is called from only two call sites (`RecordAuthoringSessionOperation` for placement moves/rotate/scale/delete, and `ApplyAcceptedReconciliation`). The chunk-manipulator plugin never calls it, so its edits are outside the "one cross-plugin history" this spec requires. `SaveAll()` clears dirty flags and logs but performs **no file write** — actual writes for placement edits still go through the old `DrawPlacementSaveQueueActions`/staged-edit system in `ViewerApp.cs`, not through this session's save path.
- Not implemented: FR-003 (exit warning listing plugins with pending changes) not found; close-with-unsaved-changes handling not located.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001 (10-op mixed undo across two plugins), SC-004 (install-tree hash-invariance).
- Open residue (spec-stated only): FR-002/FR-003 (exit warning), wiring chunk-manipulator and all placement-edit kinds through the same session/save path so undo/save are real for every operation, not just move+reconciliation.
- Superseded by / overlaps: 167 (same gap: `EditorApplierAdapter` doesn't reverse most operation types).
- Disposition: FOLD
- Confidence: high

### 169 Chunk Clipboard Plugin (Migration)
- Stated status: Draft | Tasks: no tasks.md
- Scope: Migrate the **existing** terrain chunk copy/paste/selection/save tooling out of `ViewerApp`/`ViewerApp_Sidebars` into a plugin, byte-identical, zero behavior change. Explicitly called out as "the host contract's only real validation."
- Verified implemented: none of this spec's actual scope. A **different**, new capability — a multi-tile/sub-cell chunk transposition tool (`ChunkManipulatorEditorPlugin`, `ChunkTranspositionService`, `Operations/ChunkTransposition*`) — was built instead, under spec 195 (superseded by 219/222 per `memory-bank/archive/2026-09-06-pre-context-cleanup-progress.md:28`), with its own tests (`Operations/ChunkTranspositionServiceTests.cs`, `Operations/ChunkSelectionRegionTests.cs`).
- Partial: none — the migration itself was never attempted.
- Not implemented: **FR-001–FR-006 entirely.** SC-001 measured directly: `_chunkClipboard*`/`_selectedChunks` refs in `ViewerApp.cs` = **124** (identical to the epic's "Today" baseline, target was 0). The old Terrain Lab > Clipboard entry point was not checked for removal/routing but the backing state clearly persists.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-002 (byte-identical real-tile output across rotations/toggles) — moot, migration not started.
- Open residue (spec-stated only): the entire spec — FR-001 through FR-006 remain open scope, distinct from the chunk-manipulator capability that landed under a different number.
- Superseded by / overlaps: the *capability space* overlaps with the unrelated chunk-manipulator plugin (ex-195/219/222), which does not satisfy this spec's migration requirement and risks becoming the "third implementation" the epic warns against if the old clipboard is never retired.
- Disposition: FOLD
- Confidence: high

### 170 DBC/DB2 Table Browser
- Stated status: Draft | Tasks: no tasks.md
- Scope: Read-only typed browsing/search/sort/foreign-key nav over client tables via a plugin+grid, reusing the already-plumbed DBCD/WoWDBDefs/`ArchiveReaderDbcProvider` path.
- Verified implemented: the underlying read path the spec assumes ("already plumbed") is real — DBCD/WoWDBDefs vendored, `MpqDBCProvider` at `src/viewer/WoWViewer/DataSources/MpqDBCProvider.cs`. No browser plugin or grid UI exists anywhere in `src/viewer` or `src/core/WowViewer.Core.Editor` (grep for `DbcBrowser`/`DbcTableBrowser`/`DBCTableBrowser` across `src` returned nothing).
- Partial: none.
- Not implemented: FR-001 through FR-007 — no plugin, no grid, no search/sort/filter, no foreign-key navigation.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-002 (20-table verification across two build eras) — moot.
- Open residue (spec-stated only): entire spec.
- Superseded by / overlaps: none found.
- Disposition: FOLD
- Confidence: high

### 171 DBC/DB2 Table Editing and Loose Save
- Stated status: Draft | Tasks: no tasks.md
- Scope: Cell editing, row add/delete, saving a modified table as loose `.dbc`/`.db2` via `DBCDStorage.Save(string)`.
- Verified implemented: none. Depends on 170, which has no UI to extend.
- Partial: none.
- Not implemented: FR-001 through FR-009 entirely.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001/SC-003 (byte-identical round-trip, install-tree hash invariance) — moot.
- Open residue (spec-stated only): entire spec.
- Superseded by / overlaps: none found.
- Disposition: FOLD
- Confidence: high

### 172 Editor Edit Journal — Crash Recovery and Resumable Sessions
- Stated status: Draft | Tasks: no tasks.md
- Scope: Every completed Editor Operation durably written to a Zarr-backed journal before being reported staged; resumable/nameable/deletable sessions.
- Verified implemented: none. No journal writer, no Zarr-backed session persistence, no resume UI found under `src/core/WowViewer.Core.Editor` or `src/viewer`.
- Partial: the prerequisite (168's in-memory undo/dirty model) exists to make durable, but nothing persists it.
- Not implemented: FR-001 through FR-011 entirely.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001/SC-002 (10 kill-points, no partial-edit-as-complete) — moot, nothing to kill.
- Open residue (spec-stated only): entire spec.
- Superseded by / overlaps: none found.
- Disposition: FOLD
- Confidence: high

### 173 Asset Integrity Gate
- Stated status: Draft | Tasks: no tasks.md
- Scope: Validate on read, refuse writes from quarantined/unverified inputs, re-verify every written file, record provenance including losses; publish a census over `H:\CLIENTS\WoW335\modernwow\`.
- Verified implemented: the gate shell — `AssetIntegrityGate.ValidateOnRead/AuthorizeWrite/VerifyWrittenFile/HasCompleteProvenance/RecordedLosses` at `Integrity/AssetIntegrityGate.cs`; `IAssetValidator`, `AssetValidationResult`, `ValidationVerdict`, `AssetProvenance`, `IntegrityWriteDecision`; tests at `tests/WowViewer.Core.Editor.Tests/Integrity/AssetIntegrityGateTests.cs`. The gate correctly treats "unverified" as blocking (FR-002) per its logic.
- Partial: `IAssetValidator` is only an interface — **no concrete structural validator** (WMO/ADT/M2 constraint checking) was found anywhere in `src`, so `ValidateOnRead` has nothing real to delegate to yet; the gate's plumbing is real but the actual FR-001 validation logic is not.
- Not implemented: FR-007 (sweep `modernwow` and publish a census) — no census artifact found anywhere in the repo. FR-008 (384-group WMO merge real-data/render verification) — not done; the only coverage remains the pre-existing synthetic unit test `Convert_WhenSourceExceedsLegacyGroupLimit_MergesOverflowIntoFinalLegacyGroup` in `tests/WowViewer.Core.Tests/WmoV17ToV14ConverterTests.cs`, exactly as the spec's own Assumptions section already described before this spec started.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001/SC-002 (zero crashes on modernwow + published census) is the spec's central, entirely-unmet, operator-owned (real client corpus) gate.
- Open residue (spec-stated only): FR-001 concrete validators, FR-007 census (SC-001/SC-002), FR-008 384-group render verification (SC-006).
- Superseded by / overlaps: 174 depends on 173's census, which doesn't exist, so 174 is fully blocked.
- Disposition: FOLD
- Confidence: high

### 174 Asset Repair Patterns
- Stated status: Draft | Tasks: no tasks.md
- Scope: Named, opt-in, per-pattern repairs for defects the 173 census identifies; never modifies source; must pass the validation its input failed; a render-verified 384-group repair example.
- Verified implemented: none. No repair-pattern code, no census to drive coverage (see 173).
- Partial: none.
- Not implemented: FR-001 through FR-008 entirely; explicitly blocked on 173's unmet FR-007.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-003 (real >384-group WMO repaired, loads and renders in target client) — moot.
- Open residue (spec-stated only): entire spec, gated on 173's census landing first.
- Superseded by / overlaps: hard-blocked by 173.
- Disposition: FOLD
- Confidence: high

### 175 Placement Authoring in the Viewport
- Stated status: Draft | Tasks: no tasks.md
- Scope: Full placement editing (move/rotate/scale/add/delete) in the viewport, saved through existing core writers, zero new serializers.
- Verified implemented: a real, working UI panel — `DrawPlacementAuthoringPanel()` in `ViewerApp_Editor.cs` — lets the user move/rotate/scale/delete a selected placement, with each action both staging the edit (`StageAuthoringPlacementEdit`) and recording an `EditorOperation` (`PlacementMoveOperation`/`PlacementRotateOperation`/`PlacementScaleOperation`/`PlacementDeleteOperation`) into `_editorSession`. Core writer reuse confirmed: `AdtPlacementEditor`/`AdtPlacementWriter` at `src/core/WowViewer.Core.IO/Maps/`, with unit tests `tests/WowViewer.Core.Editor.Tests/Operations/AdtPlacementEditorTests.cs` (260 lines) and `PlacementWriteServiceTests.cs` (161 lines). FR-006 (zero new serializers) holds — no new placement serializer found.
- Partial: the actual **save** path (`DrawPlacementSaveQueueActions`) and the staging store (`_stagedPlacementEdits`) are the pre-existing `ViewerApp.cs` implementation, not routed through `EditorSession.SaveAll()` — so undo for rotate/scale/delete is recorded but, per the 167/168 findings above, `EditorApplierAdapter` doesn't reverse those operation kinds. New-placement add (FR-002) was not located in the panel shown (only move/rotate/scale/delete).
- Not implemented: FR-002 (add a new placement by choosing model/WMO + position) not found in the panel; SC-002 (byte-identical unedited chunks, chunk-level verified), SC-006 (loads in an independent external tool) unverified.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-002, SC-005 (install-tree hash invariance), SC-006 (independent-tool load) — all real-data/real-tool gates, unwitnessed here.
- Open residue (spec-stated only): FR-002 (add placement), full routing through 168's session/save/undo rather than the legacy staging path, SC-002/SC-006 verification.
- Superseded by / overlaps: 167/168 (shared undo-wiring gap), 176 (reconciliation apply already demonstrates the *correct*, fully-session-routed pattern this panel should follow).
- Disposition: FOLD
- Confidence: high

### 176 PM4-Guided Object Transfer and Museum Placement Repair
- Stated status: **Implementing** — "Phases 1–2 and a usable Reconcile apply loop are source-proven; in-scene overlay, P1 transfer, and real-client proof remain" (self-reported in spec.md) | Tasks: 8 checked / 13 total (5 open T001–T005, 3 parked T006–T008)
- Scope: (a) PM4-guided reconciliation/repair of existing Museum placements — align/substitute/clone proposals, previewed and explicitly accepted; (b) general cross-tile/cross-era object transfer (FR-001–FR-006) — the spec's stated P1 user story.
- Verified implemented: reconciliation half (a) is real and wired end-to-end: `Pm4ReconciliationEngine`/`Pm4ReconciliationInputAdapter` (`src/core/WowViewer.Core.PM4/Reconciliation/`), `ReconciliationApplyOperation`/`ReconciliationApplyService` (`src/core/WowViewer.Core.Editor/Operations/`), full UI in `ViewerApp_Editor.cs` (`DrawReconciliationPanel`, preview/accept/reject/apply, guarded writes via `_editorSession.ResolveOutputPath`/`GuardWritablePath`, provenance sidecar `.reconciliation.json`), tests `Operations/ReconciliationApplyServiceTests.cs` (305 lines). This is the one place in the batch where session-guarded write + undo + provenance is fully wired as the spec intends.
- Partial: T001 ("remove leftover `PrefillReconciliationPathsFromScene()`/unused fields") is checked `[ ]` in tasks.md but verified **already done** in code — grep for `PrefillReconciliationPathsFromScene`/`DrawReconciliationPathField` across `src/viewer` returns nothing.
- Not implemented: (b) general object transfer — **FR-001 through FR-006 are explicitly parked** (tasks.md T006: "P1 cross-tile / cross-era transfer... Parked until T001–T005... Separate phase"). In-scene overlay of proposals (T003) and moving preview off the render thread (T004) are open. SC-004 (atomic multi-file failure leaves nothing modified, T007) unverified.
- Checkbox accuracy: 1 unchecked-but-present (T001).
- Operator gates owed: T005 (real PM4/Museum pair preview→accept→apply→reload with hash record) is explicitly marked user-owned in tasks.md; SC-006–SC-009 all require real-client/real-corpus proof.
- Open residue (spec-stated only): T001 (mark done)/T002–T005 (in-scene overlay, off-render-thread preview, real-pair proof), and the entire parked FR-001–FR-006 transfer capability (T006–T008).
- Superseded by / overlaps: none — this is the batch's most mature, actively-tracked spec (has its own plan/tasks/data-model/research/quickstart).
- Disposition: KEEP-ACTIVE
- Confidence: high

### 177 ADT Tile Creation
- Stated status: Draft | Tasks: no tasks.md
- Scope: Create a new ADT tile **within an existing map** (empty or seeded), update the map's tile index, correct per-era form. Explicitly out-of-scope: creating new maps.
- Verified implemented: none of this spec's in-scope capability found under `Operations/` or the Editor plugins. A related-but-explicitly-out-of-scope capability exists instead — `Workbench.Services.NewMapCreatorService`/`LoadGeneratedNewMap` in `ViewerApp_Editor.cs`, which creates a **whole new map** (this is spec 234/236 territory per the brief, matching 177's own "Out of Scope: Creating new maps" line almost word for word).
- Partial: none.
- Not implemented: FR-001 through FR-007 (tile creation inside an existing map, tile-index registration, era-correct form) not found.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001 (loads in viewer and an independent tool) — moot.
- Open residue (spec-stated only): entire spec; depends on 176's still-parked object-transfer half.
- Superseded by / overlaps: `NewMapCreatorService`/spec 234 covers the adjacent, explicitly-out-of-scope "new map" capability, not this spec's "new tile in existing map" capability — do not conflate them when reconciling.
- Disposition: FOLD
- Confidence: medium (did not exhaustively search every Workbench page for a tile-creation control beyond the Editor Operations/plugins survey)

### 178 MCP Automation Surface (External, Optional)
- Stated status: Draft | Tasks: no tasks.md
- Scope: An optional, disabled-by-default MCP adapter that remote-controls whatever Editor operations already exist; strictly downstream, no vote on Editor design, deletable with zero effect on the Editor.
- Verified implemented: none. No `ModelContextProtocol`/MCP server package reference, no MCP tool/type anywhere in `src` (grep for MCP-related class/namespace patterns across `src` returned nothing). A related spec, `213-mcp-tooling-harness`, targets exposing the **CLI tools** (not Editor operations) over MCP and is also spec-only (Draft, no tasks.md, no code).
- Partial: none.
- Not implemented: FR-001 through FR-019 entirely.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001/SC-005 (real pipeline script, stock third-party MCP client) — moot.
- Open residue (spec-stated only): entire spec; explicitly last/optional per the epic's own ordering.
- Superseded by / overlaps: 213 (mcp-tooling-harness) covers a different surface (CLI tools) and does not substitute for this spec; both are unimplemented.
- Disposition: FOLD
- Confidence: high

### 179 Canonical MPQ Patch-Chain Resolver
- Stated status: Draft | Tasks: no tasks.md
- Scope: One patch-priority implementation shared by viewer and tools, replacing the two that exist today, with recorded patch/delete provenance.
- Verified implemented: none — this is a consolidation spec and no consolidation has happened.
- Partial: none.
- Not implemented: FR-001 through FR-008. **Directly measured**: `MpqArchiveCatalog.cs` and `NativeMpqService.cs` both still exist in `src/core/WowViewer.Core.IO/Files/` — the exact two-implementation state the epic's baseline describes as "Today: 2" (target 1).
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-002 (9→0 tied archives on `H:\CLIENTS\WoW335\modernwow\`) requires real client access, not re-verified here.
- Open residue (spec-stated only): entire spec; this is the datastore epic's explicit landing-first gate — everything downstream (180-183) inherits its absence.
- Superseded by / overlaps: none.
- Disposition: FOLD
- Confidence: high

### 180 Multi-Build Content-Addressed Datastore
- Stated status: Draft | Tasks: no tasks.md
- Scope: One Zarr datastore holding several client builds, content-addressed, builds added/removed cheaply, verified against source.
- Verified implemented: none — no multi-build/content-addressed store builder found in `data-harvester/` or `src`. Blocked on 179 (patch resolution must be canonical before a store can trust what it dedupes).
- Partial: none.
- Not implemented: FR-001 through FR-014 entirely.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001/SC-003 (≥3-build byte-identical round-trip, remove-reclaims-only-unique) — moot, nothing built.
- Open residue (spec-stated only): entire spec.
- Superseded by / overlaps: hard-blocked by 179.
- Disposition: FOLD
- Confidence: high

### 181 Incremental Processing (Derivation Dedupe)
- Stated status: Draft | Tasks: no tasks.md
- Scope: Reuse derived artifacts whose complete input set + processing version are unchanged; the epic calls this "the datastore's actual payoff."
- Verified implemented: none. **Directly measured**: `InputSha256` is computed and written into the harvest manifest (`tools/harvest/WowViewer.Tool.Harvest/Program.cs:748`) and, per grep across all of `src/*.cs`, is never read back or compared anywhere — exactly the epic's "0 uses of InputSha256 to avoid work" baseline, unchanged.
- Partial: none.
- Not implemented: FR-001 through FR-008 entirely.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001/SC-002 (measured reuse ratio, byte-identical incremental-vs-scratch) — moot.
- Open residue (spec-stated only): entire spec; depends on 180.
- Superseded by / overlaps: none.
- Disposition: FOLD
- Confidence: high

### 182 Adaptive Per-Type Storage Encoding
- Stated status: Draft | Tasks: no tasks.md
- Scope: Per-array-type codec/level selection from measured ratio + decompression throughput; converge the repo's two current codec defaults into one mechanism.
- Verified implemented: none. **Directly measured, unchanged from the epic's baseline**: `data-harvester/src/harvester/zarr_io.py:33` — `BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")`; `data-harvester/src/harvester/v25/dataset.py:62` — `BloscCodec(cname="lz4", clevel=1)`. Two defaults, still two.
- Partial: none.
- Not implemented: FR-001 through FR-006 entirely.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001 (beat the measured ~4:1 baseline) — moot, no selection mechanism exists.
- Open residue (spec-stated only): entire spec; best measured after 181 per the spec's own dependency note.
- Superseded by / overlaps: none.
- Disposition: FOLD
- Confidence: high

### 183 Viewer: Load Zarr Datastore
- Stated status: Draft | Tasks: no tasks.md
- Scope: A "Load Zarr Datastore" entry point: pick a datastore + build, viewer behaves as if that client were installed, no extraction to disk, switches without restart.
- Verified implemented: none of this spec's capability. No "Load Zarr Datastore" menu/entry point found (grep across `src/viewer` for the phrase and for `LoadZarrDatastore` returned nothing). `ZarrTileDatasetLoader` (`src/viewer/WoWViewer/Terrain/ZarrTileDatasetLoader.cs`) exists but is a narrower, pre-existing per-tile-dataset loader (spec 041, per its own doc comment) unrelated to this spec's full-client-equivalence goal. A separate `RosettaDatastoreTerrainAdapter`/`_showRosettaDatastoreDialog` ("Rosetta Datastore" dialog, `ViewerApp.cs:844,1944-1945,13037-13066`) loads terrain from a PM4-object-library-flavored dataset — also a distinct, narrower feature, not multi-build client-equivalent loading through `IDataSource`.
- Partial: the ingredients this spec would need (an `IDataSource` abstraction, a terrain-adapter seam) exist generically, but nothing implements "pick a datastore build, viewer behaves like the original client" as this spec requires.
- Not implemented: FR-001 through FR-008 entirely.
- Checkbox accuracy: no tasks.md.
- Operator gates owed: SC-001 (≥3-map render match against original client) — moot.
- Open residue (spec-stated only): entire spec; explicitly gated on 180 being trustworthy first, which it is not.
- Superseded by / overlaps: `ZarrTileDatasetLoader` (spec 041) and the Rosetta Datastore dialog are adjacent but narrower — do not treat either as satisfying this spec when reconciling.
- Disposition: FOLD
- Confidence: high

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 166 | ARCHIVE-COMPLETE | 2 (minor, unverified SCs) | editor-plugin-host |
| 167 | FOLD | 4 | editor-runtime-bridge |
| 168 | FOLD | 3 | editor-session-undo |
| 169 | FOLD | 6 (FR-001–006) | chunk-clipboard-migration |
| 170 | FOLD | 7 (FR-001–007) | dbc-table-browser |
| 171 | FOLD | 9 (FR-001–009) | dbc-table-editing |
| 172 | FOLD | 11 (FR-001–011) | editor-edit-journal |
| 173 | FOLD | 3 (FR-001 validators, FR-007 census, FR-008 render-proof) | asset-integrity-gate |
| 174 | FOLD | 8 (FR-001–008, blocked on 173) | asset-repair-patterns |
| 175 | FOLD | 4 (FR-002 add, session/save routing, SC-002/SC-006) | placement-authoring |
| 176 | KEEP-ACTIVE | 8 (T001–T008) | object-transfer |
| 177 | FOLD | 7 (FR-001–007) | adt-tile-creation |
| 178 | FOLD | 19 (FR-001–019, optional/last) | mcp-automation-surface |
| 179 | FOLD | 8 (FR-001–008, must-land-first) | client-datastore-patch-resolver |
| 180 | FOLD | 14 (FR-001–014, blocked on 179) | client-datastore-multi-build |
| 181 | FOLD | 8 (FR-001–008, blocked on 180) | client-datastore-incremental |
| 182 | FOLD | 6 (FR-001–006, blocked on 180) | client-datastore-encoding |
| 183 | FOLD | 8 (FR-001–008, blocked on 180) | client-datastore-viewer-load |
