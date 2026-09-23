# Batch F2 — Reconstruction & Map Editing (specs 190, 191, 192, 203, 208, 219, 220, 222, 230, 232, 234)

Audited read-only against AUDIT-BRIEF.md. Code roots checked: `src/core/WowViewer.Core.Editor`,
`src/core/WowViewer.Core.IO` (writers, `Maps/`, `Terrain/`, `Procedural/`, `Dbc/`), `src/core/WowViewer.Core`
(`Maps/PhaseComposition.cs`, `TileContentTransform.cs`), `src/core/WowViewer.Core.PM4`,
`src/core/WowViewer.Core.Runtime/World/Terrain/Stratigraphy`, `src/viewer/WoWViewer/Terrain/*`,
`src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs`, `tools/inspect/WowViewer.Tool.Inspect/Program.cs`,
`tests/WowViewer.Core.Tests/*`. Note up front: 236 (`v0.5.4-dev`, out of this batch's number range but
directly relevant) is the live carrier for 234's save pipeline and for residue from 191/219/232 — flagged
under each spec below and included as context, not as one of the 11 audited specs.

---

### 190 Rosetta Calibration Corpus for PM4 Object Identification
- Stated status: Draft, but the spec body is an extensive checkpoint log (31 checkpoints) claiming full
  delivery | Tasks: 30/30 checked (T001-T030 across Phases 1, 1.5, 2, 2.5, 3, 4)
- Scope: synthesize a labelled "designkit" tileset placing every client object once, build a PM4
  reference-signature library from it, do deterministic PM4 lookup against it, synthesize companion
  ADTs for orphan PM4 tiles (US1-US4).
- Verified implemented: `RosettaTilesetGenerator`, `RosettaDbcGenerator`, `RosettaMinimapPainter`,
  `RosettaObjectLibrary`, `RosettaDatastoreWriter`, `RosettaPm4LookupEngine` (`WowViewer.Core.PM4/Matching`),
  `RosettaCompanionAdtSynthesizer` all exist as claimed -> CLI verbs `rosetta-generate`,
  `rosetta-datastore-info/query/diff`, `rosetta-pm4-match`, `rosetta-synthesize-companions` in
  `WowViewer.Tool.Inspect/Program.cs`; tests `RosettaTilesetGeneratorTests`, `RosettaDatastoreTests`,
  `RosettaPm4LookupEngineTests`, `RosettaReferenceLibraryTests`, `RosettaCompanionAdtSynthesizerTests`;
  viewer "Load from Rosetta Datastore" wiring in `ViewerApp.cs`/`ViewerApp_ClientDialogs.cs` and
  `RosettaDatastoreTerrainAdapter.cs`; `Pm4ReconciliationInputAdapter.BuildRosettaCorpusReferences`
  connects lookup to Spec 176.
- Partial: none found beyond what the spec itself already flags as operator-owned (real 0.5.3 client
  visual proof of legibility/minimap rendering — repeatedly named "operator-owned" in the checkpoint log
  itself, not hidden).
- Not implemented: none found against the spec's stated FRs.
- Checkbox accuracy: accurate (every checked task has a corresponding class/CLI verb/test).
- Operator gates owed: real 0.5.3/1.12.1/3.3.5 client boot + visual legibility proof (already named
  in-spec, not new residue).
- Open residue (spec-stated only): none — Out of Scope section explicitly excludes real-client loading.
- Superseded by / overlaps: feeds 191 (dense layout follow-on) and 230 US1 (Rosetta-indexed placement,
  itself not yet implemented — see 230).
- Disposition: ARCHIVE-COMPLETE
- Confidence: high

---

### 191 Procedural Garden Museum Map Generator & Dense Calibration Corpus
- Stated status: "Complete" | Tasks: 21/24 checked (T001-T021 Phases 1-5 checked; T022-T024 Phase 6
  unchecked)
- Scope: dense adaptive-cell layout packing, M2 scaling, garden terrain sculpting, organic multi-layer
  texture painting, a generic procedural map surface, semantic asset classification (US1-US6).
- Verified implemented: `SemanticAssetClassifier`, `AdaptiveLayoutPacker`, `ProceduralTerrainSculptor`,
  `ProceduralTexturePainter`, `IGenerativeMapSurface` all present under
  `src/core/WowViewer.Core.IO/Procedural/` (spec text says `WowViewer.Core.Editor.Procedural` — actual
  namespace is `WowViewer.Core.IO.Procedural`, a doc/code naming drift, not a functional gap); tests
  `ProceduralTerrainSculptorTests.cs` present; `rosetta-generate` CLI carries `--density`/`--m2-scale`/
  `--theme`/`--noise-roughness` per T016.
- Partial: AC5.3 ("Procedural Map Generator UI panel" in the WoWViewer Editor with live parameter tuning)
  — no dedicated panel found under that name in `src/viewer/WoWViewer`; the CLI/engine surface (AC5.1/5.2)
  is real but the standalone editor UI panel described by US5 was not located.
- Not implemented (spec-stated, own Phase 6): **T022** live multi-tileset ADT emission overhaul (3-4
  layer garden/cobblestone/marble painting instead of 2-layer rectangular masks), **T023** organic terrain
  curvature in the live generator, **T024** real-client visual verification in Alpha 0.5.3 and 3.3.5.
- Checkbox accuracy: accurate for Phases 1-5; Phase 6 correctly left unchecked.
- Operator gates owed: T024's real-client visual verification.
- Open residue (spec-stated only): T022 (live multi-tileset texturing), T023 (organic curvature in the
  live generator), T024 (real-client verification). **Note**: T022's complaint ("random assets from
  random expansions even for 0.5.3 generation") is restated near-verbatim as 236 US4/FR-011-013 — the two
  specs describe the same open defect.
- Superseded by / overlaps: 192 (built a more general brush/template system after this), 236 US4/Phase 4
  (client-constrained generator, i.e. this spec's Phase 6 residue, actively being worked there).
- Disposition: FOLD (residue = T022-T024, carries into the 236/reconstruction epic)
- Confidence: medium (AC5.3 panel absence not exhaustively confirmed; everything else high-confidence)

---

### 192 Terrain Template Brush & Paste Library with Interactive In-Viewer Map Generator
- Stated status: "Complete" | Tasks: 23/23 checked (T001-T023, Phases 1-6)
- Scope: `TerrainBrushPaste` data model + curated library, ADT paste extraction, stamp operation with
  feathering/undo, templated procedural map generator, in-viewer editor plugin, CLI (US1-US5).
- Verified implemented: `TerrainBrushPaste.cs`, `TerrainBrushLibrary.cs`, `TerrainLayerAllocator.cs`,
  `CuratedTerrainBrushLibrary.cs`, `AdtPasteExtractor.cs` (all `WowViewer.Core.IO/Terrain/`);
  `TerrainStampOperation.cs` (`WowViewer.Core.Editor/Operations/`); `TerrainTemplateEditorPlugin.cs`
  (`WowViewer.Core.Editor/Plugins/`, implements `IEditorPlugin`); `TemplatedTerrainGenerator.cs`; CLI verb
  `terrain-generate-templated` confirmed wired in `Program.cs` (help text + case both present); plugin
  registration referenced in `ViewerApp_Editor.cs`.
- Partial: none found.
- Not implemented: none found against spec-stated FRs.
- Checkbox accuracy: accurate.
- Operator gates owed: none named in this spec (no explicit real-client visual gate in the spec text).
- Open residue (spec-stated only): none.
- Superseded by / overlaps: functionally the generalization that supersedes 191's ad hoc texture/terrain
  code; both remain live side by side today.
- Disposition: ARCHIVE-COMPLETE
- Confidence: high

---

### 203 Multi-Phase Map Composition
- Stated status: Draft, no tasks.md — spec.md is itself an evidence-heavy problem writeup with several
  "confirmed" sub-findings already resolved inline | Tasks: no tasks.md
- Scope: FR-001 patch phase chunks field-by-field instead of wholesale replace; FR-002/003 reconcile
  placements by `uniqueId` with collision reporting; FR-004/005 support >1 simultaneous phase with
  deterministic order; FR-009 fix Alpha MDNM/MONM name-table cross-talk.
- Verified implemented: **FR-001** — `PhaseChunkMerger.Merge` (`src/viewer/WoWViewer/Terrain/PhaseChunkMerger.cs`)
  composes channel-by-channel per `PhaseDataChannel`, replacing the old whole-chunk assignment described
  in the spec's "Context" section as the bug; **FR-004** — `ITerrainAdapter.PhaseLayers` is now
  `IList<PhaseLayerSettings>` (not a single string) and `StandardTerrainAdapter`/`AlphaTerrainAdapter`
  both `foreach` over `ActivePhaseLayers`, calling `MergePhaseTile` per layer — multiple simultaneous
  phases confirmed working; **FR-009** — `AlphaTerrainAdapter.cs` (~line 867-931) converts a phase WDT's
  local MDNM/MONM indices into the base adapter's combined table with a
  `[AlphaADT] Phase ... name map` diagnostic line, exactly as specced; **SC-006** — `DbcMapPhaseTable.cs`
  answers the Map.dbc parent/child question (user memory confirms: `ParentMapID`, 18/239 rows in 5.0.1).
- Partial: **FR-002/FR-003** — placement merging in `MergePhasePlacements`
  (`StandardTerrainAdapter.cs` ~line 881) is NOT `uniqueId`-based reconciliation. It is
  "if the phase owns placements (`PhaseOwnsPlacements` — presence-gated), clear and replace the whole
  list; else keep the base list wholesale." There is no per-`uniqueId` walk, no "phase wins on the same id,
  base-only retained" merge, and no collision report/count. This is a real gap against the literal FR text,
  though it is a deliberate, documented design (`OnlyTakeWhatThePhaseCarries` presence gating) rather than
  an oversight.
- Not implemented: **FR-002** (uniqueId-keyed placement merge), **FR-003** (collision report with
  differing fields), **SC-002** (uniqueId collision count across a corpus, "including zero") — the
  hypothesis this FR/SC pair was written to test is still untested at the uniqueId-comparison level the
  spec describes.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: SC-001 (base content survives after phase load, pixel-capture), SC-003 (three Jade
  Forest phases, capture), SC-004/SC-005 (pixel comparisons).
- Open residue (spec-stated only): FR-002/FR-003/SC-002 (uniqueId-based placement reconciliation +
  collision reporting — currently presence-gated whole-list replace instead).
- Superseded by / overlaps: 219 (rotation extends `PhaseLayerSettings` this spec introduced), 222/232
  (the composition UI and cell/Z/edge-blend refinements built on this spec's channel model).
- Disposition: FOLD (residue: uniqueId placement reconciliation + collision report, carries into the
  active Cartography/232 epic since it owns `MergePhasePlacements` today)
- Confidence: high

---

### 208 Cross-Map Tile Transplant (Pre-Alpha Restoration)
- Stated status: Draft | Tasks: Phase 0 (T001-T005) checked with evidence links; Phase 1-4
  (T101-T405, 24 tasks) and Operator gates (T501-T503) all unchecked
- Scope: read tiles from a source map different from the target, select an arbitrary tile subset via a
  64x64 minimap-guided picker, rotate/mirror/offset at chunk granularity, record provenance per tile
  (US1-US4).
- Verified implemented: the underlying engine this spec asked for has been built, but under specs 219/222/
  232 rather than as 208's own code. `PhaseLayerSettings.MapName` already names a donor map different from
  the base (cross-map sourcing, FR-001); `PhaseTilePlacement`/`ResolveTileSource` give per-tile donor-to-
  target mapping (FR-002); a "Donor Tile Grid Picker (Single-Tile Placement)" collapsing panel exists in
  `ViewerApp_PhaseLayers.cs` (labelled "Spec 222-T108"), giving a browsable donor grid (partial FR-009);
  `TileContentTransform.cs` + `AlphaTileData.RotateQuarterTurn` give exact-grid rotation/mirror of terrain
  + placements (FR-005); `PhaseDataChannel` is the one channel model in use, matching FR-003's "map one
  onto the other, do not introduce a third."
- Not implemented (as this spec's own explicit shape): no dedicated `TileTransplant`/`CrossMapTransplant`
  class exists; no standalone pop-out 64x64 window with the three distinct visual states (no-tile /
  tile-no-preview / unselected) the spec requires (US2) — the picker found is an inline collapsing header,
  not a pop-out grid with drag-select; no explicit **provenance record** entity (source map, source tile,
  channels, transform, order) distinct from the layer's own settings — 232's project JSON persists layer
  config but that is not the same as a per-target-tile audit trail (FR-007); undo through `EditorSession`
  specifically (FR-008) not confirmed — composition lives in the viewer's phase-layer session, not
  editor-session undo.
- Checkbox accuracy: accurate for what's literally under this spec's own code (Phase 1-4 genuinely never
  built as 208 artifacts) but understates capability delivered by sibling specs.
- Operator gates owed: T501-T503 (all).
- Open residue (spec-stated only): FR-007 provenance record as a first-class, order-preserving entity;
  US2's pop-out picker with the three named visual states; FR-008 EditorSession-specific undo.
- Superseded by / overlaps: 219 (transform seam), 222/232 (the actual UI and cross-map composition this
  spec asked for, now shipped there under "Cartography" naming instead of "Transplant").
- Disposition: ARCHIVE-SUPERSEDED (the feature this spec specified now exists, built inside 219/222/232;
  remaining gaps are narrow and belong in the active 232 epic, not a revived 208)
- Confidence: medium (did not verify EditorSession undo wiring exhaustively)

---

### 219 Map Composition Selection & Transform Workbench (branch name says "phase-layer-rotation")
- Stated status: "Implementing — Phase 1 Core transform seam validated; workbench UI/runtime/export scope
  open" | Tasks: Phase 1 (T001-T006, Gate 1) checked = 6/38; Phases 2-8 (T007-T038) all unchecked
- Scope: this is the umbrella spec — rotation (90/45/free), mirror, per-tile placement, a shared tile/
  chunk/cell selection model used by both an orthographic whole-map canvas AND the 3D renderer, retiring
  the old Chunk Manipulator, cell-granular translation, and a "Save Transformed Map" writer path
  (US1-US10, FR-001..035).
- Verified implemented (beyond what tasks.md admits): `TileContentTransform.cs` (Core seam, T001) and
  `PhaseLayerSettings.RotationDegrees/RotationOriginTileX/Y/MirrorHorizontal/MirrorVertical` +
  `PhaseTilePlacement`/`PhaseTileSource`/`PhaseCompositionPolicy.ResolveTileSource` (T002-T004) all exist.
  Phase 2 adapter wiring (T007/T008, marked unchecked) is in fact **done**: `StandardTerrainAdapter` and
  `AlphaTerrainAdapter` both route through `ResolveTileSource` and apply rotation/mirror via
  `AlphaTileData.RotateQuarterTurn`/`TransformChunksForTarget` (confirmed working end-to-end per 232's
  T015 receipts — Shadowfang-over-Azeroth composes with real rotated terrain). UI rotation/mirror controls
  exist in `ViewerApp_PhaseLayers.cs` (90/180/270 dropdown, Mirror H/V checkboxes) — T010's exact-90°/
  mirror portion is live; free-angle and 45° are NOT exposed (dropdown only offers the four quarter-turn
  states) confirming T028/Phase 6 genuinely open.
- Partial: Phase 2's base-layer-as-first-composition-layer-with-channel-gates (T011) — base channel gates
  exist per 232 FR-11 residue tracking, not fully confirmed wired.
- Not implemented: **Phase 3** (T013-T017) — no `MapContentSelection` Core contract or full 64x64
  orthographic canvas found (Cartography's minimap-based interaction, built under 222, is a materially
  smaller "drag a footprint on the minimap" model, not this phase's tile/chunk/cell paint+lasso canvas);
  **Phase 4** (T018-T021) — no shared 2D/3D selection instance, no confirmed Chunk Manipulator retirement;
  **Phase 5** (T022-T027) — canonical `CellOffsetX/Y` exists as a *232* task (T010, itself still open per
  232's own tracking), not as 219's own deliverable; **Phase 6** (T028-T030) — 45°/free-angle rotation
  confirmed absent from the UI; **Phase 7** (T031-T035) — "Save Transformed Map" / `MapSaveService` does
  not exist anywhere in the codebase (independently confirmed missing — see 234/236 below); **Phase 8**
  (T036-T038) — no pixel-regression suite or cross-reference cleanup found.
- Checkbox accuracy: **6 unchecked-but-present** (T007, T008, and functionally most of T010's 90°/mirror
  portion, T002-T004 already counted in Phase 1 but re-confirmed live in the adapters) — tasks.md
  materially understates progress on Phases 1-2; Phases 3-8 checkbox-accurate (genuinely not started as
  this spec's own artifacts, though 232 is quietly doing some of Phase 5's job under a different task ID).
- Operator gates owed: Gate 2 through Gate 8, all.
- Open residue (spec-stated only): Phase 3 (orthographic whole-map selection canvas), Phase 4 (3D
  selector + Chunk Manipulator retirement), Phase 6 (45°/free-angle with reported approximation), Phase 7
  (Save Transformed Map — now also tracked as 234/236's job), Phase 8 (regression + doc cleanup).
- Superseded by / overlaps: 222 and 232 are actively delivering slices of this spec's scope (rotation,
  per-tile placement, cell offset) under the "Cartography" name rather than this spec's "Workbench" name;
  234/236 are delivering Phase 7 (save).
- Disposition: KEEP-ACTIVE (too large and too actively being worked piecemeal by 222/232/236 to archive;
  should be the anchor doc for the new epic rather than folded into it)
- Confidence: medium (broad spec; verified the load-bearing claims, not every FR)

---

### 220 WMO Doodad Placement Editing, Custom Doodad Sets & WMO Writing
- Stated status: no explicit status line (tasks-only spec) | Tasks: 0/22 checked (T001-T406 across
  Phase 0-4, all unchecked)
- Scope: edit MODD placements inside a loaded WMO (move/rotate/scale/delete/duplicate), author custom
  MODS doodad sets, write the edited WMO back in its source version (V14 first), gated by a round-trip
  safety test (US1-US5).
- Verified implemented: none. `WmoDoodadEditOperations`, `WmoDoodadUndoStack`, and
  `tests/.../Editor/WmoDoodadRoundTripTests.cs` do not exist anywhere in the tree. `WmoV17ToV14Converter`
  does exist (two implementations, in `WowViewer.Core.IO.Converters` and `WowViewer.Core.IO.Wmo` — an
  unrelated pre-existing duplication worth a separate look, not investigated further here) but nothing
  calls it for a doodad-editing save path.
- Partial: none.
- Not implemented: all of Phase 0 (round-trip gate), Phase 1 (core edit ops), Phase 2 (set authoring),
  Phase 3 (renderer hook/live editing), Phase 4 (add-from-model + versioned save) — every T-id in the
  file.
- Checkbox accuracy: accurate (0 checked, 0 present).
- Operator gates owed: T406 (all of it, since nothing to verify exists yet).
- Open residue (spec-stated only): the entire spec — US1-US5 in full.
- Superseded by / overlaps: none found; still a live, unstarted, self-contained proposal that depends on
  Spec 211's picking (confirmed to exist) but has no code of its own.
- Disposition: FOLD (residue = entire spec, unstarted; carries forward as-is into the reconstruction epic)
- Confidence: high

---

### 222 Cartography — Multi-Map, Multi-Tile Composition Workbench
- Stated status: "Draft v2 — awaiting operator sign-off" but heavily implemented in practice |
  Tasks: 8 checked / ~17 total (Phase 1: 3/3 + gate unchecked; Phase 2: 2/2 + gate unchecked;
  Phase 0.5: 3/4; Phase 3: 0/5 checked, T107 explicitly marked in-progress; Phase 4: 0/3)
- Scope: consolidate three drifted selection/transform systems (phase panel, chunk manipulator, scene
  click-select) into one right-sidebar "Cartography" surface: add maps/tiles as layers, drag-align on the
  minimap, transform tools, retire the old surfaces (US1-US6).
- Verified implemented: `MapFootprint.cs` (Core, `GetOccupiedTiles`/`TryResolveMap` on `ITerrainAdapter`),
  minimap footprint draw pass (`MinimapHelpers.RenderPhaseFootprints`), drag-to-align
  (`ViewerApp_MinimapAndStatus.cs`, `MapFootprint.ApplyDragDelta`), inline resolution-state badges in
  `ViewerApp_PhaseLayers.cs`, and the donor-tile-grid picker ("Spec 222-T108" — code present, though the
  task itself is still marked unchecked in tasks.md, a genuine checkbox miss). `MapFootprintTests.cs`
  20/20 per the task notes.
- Partial: T107 (transform toolbar wrapping `TileContentTransform`) — tasks.md itself documents this was
  blocked on a viewer<->Core `TerrainChunkData` type split and the naive wiring attempt was reverted; 232's
  later work (T015a-e) resolved this via a different mechanism (`AlphaTileData.RotateQuarterTurn`
  full-tile-lattice rotation) rather than by finishing T107 as originally scoped — so T107 is
  superseded-in-place, not completed as written.
- Not implemented: T109 (rebuild layer stack as its own right-sidebar panel + delete left-sidebar panel —
  tasks.md notes this was itself superseded by Spec 223's Archaeology-merge sequencing), T110 (DBC
  child-map suggestions in add flow), T111 (Chunk Manipulator parity checklist + retirement — not found),
  T112-T114 (cleanup/docs/operator verification).
- Checkbox accuracy: **1 unchecked-but-present** (T108, the donor tile picker, exists in code but tasks.md
  still shows it unchecked).
- Operator gates owed: Gate 1, Gate 2, Gate 3 (30-second add-align criterion), T114.
- Open residue (spec-stated only): T109/T110/T111/T112-T114 — old-surface retirement and DBC
  suggestion wiring never finished.
- Superseded by / overlaps: 219 (this spec explicitly narrows 219's scope to "minimap is the alignment
  canvas... 219's full orthographic workbench remains a separate future feature"), 232 (direct sequel that
  took over active development, including re-solving T107's blocker).
- Disposition: KEEP-ACTIVE (232 is its direct, actively-worked continuation under the same UI surface)
- Confidence: high

---

### 230 Reconstruction Editor — Rosetta Placement, Map Generator & Multi-Era Save
- Stated status: "Draft — authored verbatim from operator directive; not planned"; spec itself carries a
  2026-09-09 amendment: "US2/US3 superseded by Spec 234" | Tasks: no tasks.md
- Scope (post-amendment): only **US1** remains owned here — pick an asset from the Rosetta 3D object
  library, copy/paste it into a loaded or generated map at a chosen position, staged through the existing
  placement-edit/save pipeline.
- Verified implemented: none found. No `RosettaObjectPicker`, Rosetta-library-to-placement-queue UI, or
  similar was located under `src/viewer`. The Rosetta library itself (190) and generic placement-editing
  primitives exist, but nothing wires "pick from Rosetta library -> paste into map" specifically.
- Partial: the building blocks US1 depends on (Rosetta library, placement-edit pipeline) all exist per
  190's audit; only the connecting UI/workflow is missing.
- Not implemented: US1 in full (the only story this spec still owns after its own amendment).
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: SC-1 (generate, place, save without touching a CLI) — untestable, since US1 isn't
  built and US2/US3 belong to 234 now.
- Open residue (spec-stated only): US1 — Rosetta-indexed object placement (the spec's own in-place
  amendment already disposes of US2/US3 by pointing at 234).
- Superseded by / overlaps: US2/US3 explicitly superseded by 234 (self-declared in the spec text, dated
  2026-09-09); US1 stands alone and is not touched by any other spec in this batch.
- Disposition: FOLD (residue = US1 only; US2/US3 portion is ARCHIVE-SUPERSEDED per the spec's own note)
- Confidence: high

---

### 232 Cartography Composition: Cell-Level Alignment, Project Persistence, Full-Map Export & Layer UI
- Stated status: "Draft" but under very active, heavily-receipted development, including its own
  governance self-audit | Tasks: 4/~45 checked at face value (T015a, T015b, T015c, T015e); however this
  spec was already re-audited by the project's own 224-T201 governance pass on 2026-09-10, which walked
  through T050/T051/T053/T054/T056-T059/T064/T066 and corrected each from `[x]` to `[ ]` in place, with a
  documented reason per item (self-contradicting text, missing/phantom receipt files, or "operator visual
  witness" still open) — so the file's current checkbox state already reflects that correction.
- Scope: cell-level (1/16-tile) alignment fine-tune, project-file persistence with locking, full-map
  export through the real client writers, plus a long tail of operator-reported defects/directives
  (Phase 5-7) layered on afterward.
- Verified implemented: `PhaseEdgeBlender.cs` (`WowViewer.Core.Runtime/World/Terrain/Stratigraphy`,
  T064's WDL magnetic edge-snap), `CellOffsetX`/`EdgeBlendWdl`/`RotateQuarterTurn` all present in
  `PhaseComposition.cs`/`AlphaTileData.cs` and both adapters; 11 evidence files under `evidence/`
  (T010-T013, T015a/b, T015c, T015e, T050, T051, T053, T054, T056-T059, T064, T066) — each receipt
  independently documents its own build/test status AND explicitly names the outstanding operator visual
  witness, matching the governance audit's findings.
- Partial: nearly everything in Phase 1/4/5 — code-level and unit-test-verified per the receipts, but the
  decisive visual/alignment criteria are consistently marked "Open, operator-owned" in the receipts
  themselves (e.g. T066's cell-shift fix: 0 of 2 acceptance criteria have evidence beyond code review).
- Not implemented: Phase 2 (T020-T022, project persistence/locking — unchecked, no receipt found for it
  specifically beyond what T051/T052 partially cover), Phase 3 (T030-T032, full-map export — unchecked,
  no export pipeline code found), Phase 6 (T060-T063: MDX fire/water light regression, composed-map
  minimap synthesis, texture restoration for stripped phase maps), Phase 7 (T067-T071, all explicitly
  "None of these have been investigated yet" per the spec's own text).
- Checkbox accuracy: accurate as currently written (post-self-correction); the spec's own governance note
  is reliable and matches independent code verification here.
- Operator gates owed: essentially every SC in the spec (SC-1 through SC-5) — all explicitly marked open
  pending operator visual witness in the receipts.
- Open residue (spec-stated only): T020-T022 (persistence/locking), T030-T032 (full-map export), T060-T063
  (Phase 6), T067-T071 (Phase 7 operator-reported gaps, uninvestigated).
- Superseded by / overlaps: continuation of 219/222; T030 (full-map export) is the same deliverable as
  219's Phase 7 "Save Transformed Map" and 234's US1 — three specs currently point at one unbuilt export
  pipeline.
- Disposition: KEEP-ACTIVE (the live epic; already self-governing per its own 224-T201 audit)
- Confidence: high

---

### 234 Map Save (Merged ADT / Alpha WDT) & New Map Creator
- Stated status: "Draft — authored verbatim from operator directive; not planned" | Tasks: no tasks.md
  (only `checklists/requirements.md`)
- Scope: US1/US2 — a "Save Map" action in both Archaeology (Cartography) and the Editor's Data I/O page,
  writing the live composed/edited map to Alpha 0.5.3 WDT or LK v18 ADT through one shared pipeline; US3 —
  a New Map creator (blank or template-generated) in the Editor tab.
- Verified implemented: **US3 partial** — `NewMapCreatorService.cs`
  (`src/viewer/WoWViewer/Workbench/Services/`) exists and is referenced from `ViewerApp_Editor.cs`.
  **US1/US2 — not implemented**: no `MapSaveService`, no "Save Map" pipeline, no save action found in
  either Archaeology or Editor code. This is independently confirmed by Spec 236's own operator-directive
  text (2026-09-15): *"none of the map merge saving shit works, nothing... no files in the output folder"*
  — the operator himself reports this spec's core deliverable as unbuilt five days after it was authored,
  and 236 Phase 5 (T041) explicitly scopes `MapSaveService.cs` as "implementing Spec 234," itself still
  unchecked in 236's own tasks.md.
- Partial: US3 (New Map creator exists at the service level; whether it satisfies AC3 fully — refusal on
  name collision, immediate loadability — not exhaustively verified).
- Not implemented: US1 (Save Map from Archaeology), US2 (Save Map from Editor Data I/O), the shared save
  pipeline (FR-001 through FR-007), and by extension every SC in the spec.
- Checkbox accuracy: n/a (no tasks.md; the spec's own dependent, 236, is the more reliable status source
  and confirms non-implementation).
- Operator gates owed: all of SC-001 through SC-004 — none are reachable since the save pipeline doesn't
  exist.
- Open residue (spec-stated only): FR-001..FR-007 (the entire save pipeline) — explicitly still owed;
  FR-008/009 (New Map creator) partially covered by existing `NewMapCreatorService`.
- Superseded by / overlaps: **236 Phase 5 (T040-T043) is the current active owner of this spec's save
  pipeline**, per 236's own text ("implementing Spec 234"); 219 Phase 7 ("Save Transformed Map") and 232
  T030 (full-map export) are two more specs converging on the same unbuilt deliverable.
- Disposition: FOLD (residue = the entire save pipeline, US1/US2; carries into 236, which is already the
  de facto owner and itself still open on this point)
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 190 | ARCHIVE-COMPLETE | 0 | Rosetta calibration corpus |
| 191 | FOLD | 3 (T022-T024) | Procedural garden museum generator |
| 192 | ARCHIVE-COMPLETE | 0 | Terrain template brush library |
| 203 | FOLD | 3 (FR-002/FR-003/SC-002) | Multi-phase composition — placement reconciliation |
| 208 | ARCHIVE-SUPERSEDED | 3 (provenance record, pop-out picker, EditorSession undo) | Cross-map tile transplant |
| 219 | KEEP-ACTIVE | 5 phases (3,4,6,7,8) | Map composition transform workbench (epic anchor) |
| 220 | FOLD | entire spec (unstarted) | WMO doodad placement editing |
| 222 | KEEP-ACTIVE | 5 (T109-T114) | Cartography consolidation |
| 230 | FOLD | 1 (US1 Rosetta placement) | Reconstruction editor |
| 232 | KEEP-ACTIVE | ~4 phases (persistence, export, Phase 6/7) | Cartography composition (live epic) |
| 234 | FOLD | entire save pipeline (US1/US2) | Map save / new map — folds into 236 |

**Cross-cutting finding**: three separate specs (219 Phase 7, 232 T030, 234 US1/US2) each independently
describe the same unbuilt deliverable — a "compose the full map and write real client files" export
pipeline — and a fourth (236 Phase 5) is now the active implementer. Any new epic should collapse these
into one save/export task, not carry three duplicate descriptions forward.
