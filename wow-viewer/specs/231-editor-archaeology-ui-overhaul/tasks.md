# Tasks — Spec 231 Editor & Archaeology Workspace UI Overhaul

Phases match [plan.md](plan.md). A checkbox may be set `[x]` only with a receipt in this
directory (`evidence/`) per AGENTS.md §9.2. Implementation session: fresh chat (operator
directive, 2026-09-07).

## Phase 0 — Inventory v3 baseline (Spec 227 gate folded in)

- [x] T001 Capture Editor + Archaeology tab screenshots as they exist today into
      `evidence/inventory-v3-baseline.md` (operator-assisted per Spec 227 T003).
      Done 2026-09-07 via the viewer's built-in capture automation (operator directive);
      10-page matrix + run fingerprint in
      [evidence/inventory-v3-baseline.md](evidence/inventory-v3-baseline.md)
- [x] T002 Record the 4-page IA scope freeze: operator acknowledges Placement & Objects /
      Terrain Tools / Data I/O / Converters and the Archaeology analysis-only page list.
      Receipt: [evidence/t002-scope-freeze-ack.md](evidence/t002-scope-freeze-ack.md)
- [x] T003 Write the dated supersession note in `227-ui-reaudit/tasks.md` for the Editor-tab
      item ("sane Editor tabs" → Spec 231). Note written 2026-09-07 in that file's header

## Phase 1 — Page skeletons + migration map

- [x] T010 Create `Workbench/Pages/` service classes: `EditorDataIoPage`,
      `EditorPlacementObjectsPage`, `EditorTerrainToolsPage`, `EditorConvertersPage` with a
      narrow `ViewerAppContext` (constructor-injected; no god-class reach-back)
- [x] T011 `WorkbenchNavigator.GetEditorWorkbenchLabels()` → the 4 new labels; update
      `DrawEditorWorkbenchSubTabContent` to delegate; update
      `NormalizeWorkbenchStateAfterLoad` for old-index remap (0→Placement, 1→Converters,
      2→Placement, 3→Data I/O, 4→Terrain Tools, 5→Placement) and verify with a pre-change
      settings file. Remap implemented at the settings-load boundary gated on
      `WorkbenchNavigationVersion` (see receipt note 1); pre-change settings runtime check
      remains operator-owned
- [x] T012 Gate: `dotnet build wow-viewer/WowViewer.slnx -c Debug` 0 errors; old content
      reachable from new pages; zero behavior change receipt.
      Receipt: [evidence/t010-t012-phase1-receipt.md](evidence/t010-t012-phase1-receipt.md)

## Phase 2 — Data I/O consolidation (D1, D6)

- [x] T020 Move all export/import content into `EditorDataIoPage` (PM4 JSON/OBJ, correlation
      JSON, GLB scene/collision/map tiles, terrain layer export/import, synthesized minimap,
      VLM/ML harvest launchers) — groups collapsed by default. PM4 export command set moved to a
      collapsed "PM4 Exports" section; VLM/ML harvest launchers do not exist (retired workflow —
      see receipt note 1)
- [x] T021 Delete the 5 duplicate export-button sites (ViewerApp_Pm4Utilities ~378/448/1201/
      1782/3900) and the Archaeology-hosted imports page; replace with `Open Data I/O` links
      carrying current selection (selection lives on WorldScene, preserved across navigation).
      Archaeology-tab de-hosting of the imports route is T042 (receipt note 2)
- [x] T022 Background-execution audit on the new page: every potentially >1s command uses the
      `ExportPm4ObjectsObjSet` pattern (minimum: correlation JSON build, object-match report
      build, PM4 JSON dump). Both JSON dumps backgrounded; object-match report refresh is an
      uncalled dead path (receipt D6 table)
- [x] T023 Gate: SC-1 grep receipt for D1 (each label = 1 draw site); build/test clean; export
      smoke on a small map with the UI responsive.
      Receipt: [evidence/t020-t023-phase2-receipt.md](evidence/t020-t023-phase2-receipt.md).
      Interactive export execution (path picker) remains operator-owned

## Phase 3 — Placement & Objects (D2, D4, D5a)

- [x] T030 Diff the two PM4/WMO correlation panels; fold unique behavior into the survivor.
      Workbench copy was a strict subset of `DrawPm4WmoCorrelationContent` (no unique behavior;
      shared state fields) — survivor kept
- [x] T031 Consolidate transform/match/reconcile/collection/graph tools + surviving correlation
      panel into `EditorPlacementObjectsPage` (new collapsed "PM4 Placement Tools" section);
      delete the original workbench correlation panel (149 lines). The PM4 tab Correlation page
      is the surviving full panel. Tasks-panel-internal terrain blocks still to relocate (see
      receipt "Known remaining work" 1)
- [x] T032 Doodad-set: object page owns the full combo; toolbar keeps the hovered-WMO quick
      combo (FR-6); delete the other two combos (Model Info + Actions copies deleted)
- [x] T033 Gate: SC-1 receipts D2/D4; SC-5 proof that the duplicated correlation code block is
      gone; build/test clean. Receipt:
      [evidence/t030-t043-phase3-4-receipt.md](evidence/t030-t043-phase3-4-receipt.md)

## Phase 4 — Terrain Tools, Converters, Archaeology de-hosting (D3, D5b)

- [x] T040 Terrain Tools page: single "Clipboard + Save" copy (both dead duplicate hosts
      deleted — `DrawTerrainToolsWindow` was unreachable, `DrawTerrainToolsSubTab` had no
      callers); terrain lab + stratigraphy/weak-signal entries already routed
- [ ] T041 Converters page: map/WMO/M2-MDX converters, ADT utilities, round-trip validation,
      Spec 221 harness (converter content routed in Phase 1; Spec 221 harness surfaces unverified)
- [x] T042 Remove Archaeology editor hosting: the dead `DrawArchaeologyEditorContent`
      dispatcher and `_archaeologyEditorSubTab` field deleted (0 references remain); Archaeology
      = analysis pages only. Content method renames deferred
- [x] T043 Gate: SC-1 receipts D3/D5; build clean (0 errors); line counts recorded —
      `ViewerApp_Sidebars.cs` 6,173 → 6,013; `ViewerApp_Pm4Utilities.cs` 4,451 → 4,368.
      Receipt: [evidence/t030-t043-phase3-4-receipt.md](evidence/t030-t043-phase3-4-receipt.md)

## Phase 7 — Wire whole-layer rotation/mirror end-to-end (operator directive 2026-09-07)

> Operator: "where the fuck is the ability to rotate the whole fucking map layer? you only
> half-assed the tile/map rotation and never extended it to the layers like I fucking asked."

**Audit finding (2026-09-07)**: the entire core engine already exists but was never wired:

- `PhaseLayerSettings` carries `RotationDegrees`, `RotationOriginTileX/Y`, `MirrorHorizontal`,
  `MirrorVertical` (`WowViewer.Core/Maps/PhaseComposition.cs:94-114`)
- `PhaseComposition.ResolveTileSource` does the rotation/mirror-aware inverse donor-tile lookup;
  `ComposeTileTransforms` + `ResolveRotationApproximation` classify exact-grid vs free-rotate
  (`PhaseComposition.cs:296-400`)
- `TileContentTransform` fully transforms chunk content — heightmap vertices, normals, holes,
  alpha maps, liquid, chunk slots, placements, yaw (`WowViewer.Core/Maps/TileContentTransform.cs`)

**Missing wiring** (the half-assed part):
1. Layers panel UI exposes only offsets — no rotation/mirror controls (`ViewerApp_PhaseLayers.cs:294-337`)
2. `AlphaTerrainAdapter` / `StandardTerrainAdapter` compute `source = target - offset` directly and
   never call `ResolveTileSource` nor apply `TileContentTransform` to loaded donor tiles
3. `TranslatePhasePlacements` (both adapters) translates only — no rotation/mirror
4. `GetLayerFootprints` / `MapFootprint.OverlapsBase` and the minimap layer rendering ignore rotation
5. `PhaseLayerSettings.Clone`-equivalent in `PhaseComposition.cs:159-170` does not copy the
   rotation/mirror fields

- [ ] T070 Layers panel: rotation combo (0/90/180/270 + free-angle input), mirror H/V checkboxes,
      footprint-centered rotation origin; clone/persistence fix (missing field copies)
- [ ] T071 Adapters: resolve donor tiles through `PhaseComposition.ResolveTileSource`; apply
      `TileContentTransform.TransformChunks` with the returned `Transforms` for exact-grid layers;
      free-angle layers surface the approximation warning
- [ ] T072 Placements: route `TranslatePhasePlacements` through
      `TileContentTransform.TransformMddf/Modf` with the layer transform
- [ ] T073 Footprints + minimap: rotate layer footprints and minimap tile placement through the
      same transform; `MapFootprint.OverlapsBase` rotation overload
- [ ] T074 Gate: build/test clean; visual proof of a quarter-turn layer on a real map (operator);
      receipt naming every wired consumer

## Phase 6 — Tools curation & removal pass (operator amendment 2026-09-07)

Model surfaces: Editor → Data I/O page and the Quick panel (simple tools, one location,
SharedUiWidgets, no one-off styling). Everything below removes UI surfaces only — MPQ/ADT/WMO/M2
readers stay untouched (AGENTS.md §4).

- [x] T060 Wiring audit: enumerate every private draw/handler method in `ViewerApp*.cs` and its
      reference count; classify each Editor/Archaeology surface as live / dead / operator-named
      for removal. Output: [evidence/p6-wiring-audit.md](evidence/p6-wiring-audit.md).
      Batch 1 executed: 27 dead draw methods removed (~880 lines), build 0 errors. T061–T064
      scoped with wiring status, not yet executed
- [ ] T061 Remove the old weak-signal amplifier UI (`_terrainWeakSignalRestore*` family and its
      draw surfaces) — never fully wired per operator
- [ ] T062 Remove "weird terrain tools" that do not work / do not fit (task-inspector terrain
      blocks, one-off tool windows) after per-surface wiring check
- [ ] T063 Remove stale selection-tools surfaces that duplicate Inspector/Placement functionality
- [ ] T064 Converge remaining Editor/Archaeology pages on the Data I/O / Quick pattern
      (SharedUiWidgets sections, collapsed groups, links between pages)
- [ ] T065 Gate: build 0 errors; line-count delta recorded for every touched `ViewerApp_*.cs`;
      removal receipt with per-surface evidence; no reader code touched

## Phase 5 — Navigation test, inventory, closure

- [ ] T050 Operator navigation smoke: US1 (PM4 OBJ export), US2 (placement edit + reconcile),
      US3 (Archaeology analysis-only) each ≤2 navigations; SC-3 1600×900 no-scroll check
- [ ] T051 Inventory v3 rows for every moved/removed surface (Spec 223 FR-9); STATUS.md +
      epic updates
- [ ] T052 Final `dotnet build` + `dotnet test` receipts; spec status → Implemented with user
      gates (→ Complete on operator acceptance)
