# Epic 250 — Map Reconstruction, Composition & Editor Platform

**Created**: 2026-09-23 (spec reconciliation) · **Branch**: `v0.6.0-dev` · **Status**: Triage pending

> Primary successor of 27 archived specs (split specs also contribute items; see the ledger), including the old `epic-editor-platform`.
> **No new scope** (§9.1): every backlog item is scope a source spec already stated. Evidence:
> [reconciliation ledger](../archived/reconciliation-2026-09-23/README.md), audits
> [F1](../archived/reconciliation-2026-09-23/audit/batch-F1.md) · [F2](../archived/reconciliation-2026-09-23/audit/batch-F2.md) ·
> [A2](../archived/reconciliation-2026-09-23/audit/batch-A2.md) · [G1](../archived/reconciliation-2026-09-23/audit/batch-G1.md).
> Editing model (operator, 2026-09-06): **reconstruction from existing data**, not sculpting/hand-painting.

## Goal

Compose new and restored maps from existing client data — multi-map, multi-tile, rotated, aligned —
and **save them as real client files** (LK ADT/WDT and Alpha 0.5.3 WDT) with undoable, journaled edits.

## Delivered baseline (verified in code 2026-09-23 — do not re-plan)

| Capability | Where | Source |
|---|---|---|
| Editor plugin host, registry, lifecycle, era resolution, session + bridge types, integrity-gate shell, placement / chunk-transposition / reconciliation operations; tested | `Core.Editor/`, `ViewerApp_Editor.cs`, `WowViewer.Core.Editor.Tests` | 166 (167/168 partial) |
| Multi-phase composition: per-channel `PhaseChunkMerger`, N phase layers, Alpha name-map remap, Map.dbc parent/child | `PhaseChunkMerger`, `PhaseLayers`, `DbcMapPhaseTable` | 203 (135) |
| Rotation/mirror (exact quarter-turns) of terrain + placements, per-tile donor→target mapping | `TileContentTransform`, `AlphaTileData.RotateQuarterTurn`, `PhaseCompositionPolicy` | 219 Ph 1/2 (208) |
| Cartography: map footprints, minimap drag-to-align, resolution badges, donor-tile grid picker | `MapFootprint`, `ViewerApp_PhaseLayers.cs` | 222 |
| Cell offsets, WDL magnetic edge-snap, layer-rigid fine-tune, per-tile lock, placed-tiles-only, seam repair, MCAL repair, placement coordinates/layer Z (source landed; **witnesses owed**) | `PhaseEdgeBlender`, `PhaseComposition` | 232 (11 receipts) |
| Minimap donor-tile tool (navigate vs donor mode) | `MinimapDonorToolService` | 236 |
| New Map creator service; generator WDL output + fractal relief | `NewMapCreatorService`, `WdlWriter`, `TemplatedTerrainGenerator` | 234 US3 (partial), 236 |
| Rosetta calibration corpus; procedural garden engine + `rosetta-generate` CLI | `Core.IO/Procedural/` | 190, 191 |
| Terrain template brush + paste library, in-viewer generator | | 192 |
| Temporal stratigraphy; WDL lattice magnetization + neighbour-mesh auto-fit | `Stratigraphy` namespace | 194, 196 |
| PM4-guided object transfer: preview/reconciliation half (8/13 tasks) | | 176 |

## Known "we thought we had it" gaps (measured)

- **No map save exists.** `MapSaveService` is absent; `EditorSession.SaveAll()` writes no file. Four
  specs described this one deliverable (219 Ph 7, 232 T030, 234 US1/US2, 236 Ph 5).
- **Undo is partial.** `EditorApplierAdapter.Apply()` reverses 2 of 5+ operation kinds; rotate, scale,
  delete and chunk transposition no-op on undo.
- The god-object edit state was never migrated: `_chunkClipboard*`/`_selectedChunks` = 124 refs,
  `_stagedPlacementEdits`/`_selectedPlacement*` = 116 refs (epic target was 0).
- Phase placement merge is presence-gated whole-list replace, not `uniqueId` reconciliation.

## Backlog (spec-stated residue — each item awaits operator triage in [TRIAGE.md](../TRIAGE.md))

### A. Save & round-trip (the missing deliverable)

| ID | Item | Source |
|---|---|---|
| E-01 | One map save pipeline: composed/merged map → LK ADT/WDT + Alpha WDT, from Archaeology **and** Editor Data I/O | 234 US1/US2 FR-001–007; 219 Ph 7; 232 T030–T032; 236 Ph 5 |
| E-02 | Cartography project persistence: save / lock / autoload the layer stack | 232 T020–T022 |
| E-03 | Export path resolves under the workspace, not `bin/Debug` | 236 (operator complaint, confirmed) |
| E-04 | New Map creator acceptance: name-collision refusal, immediate loadability | 234 US3 |

### B. Editor platform completion

| ID | Item | Source |
|---|---|---|
| E-10 | Complete `EditorApplierAdapter` for every operation kind; loaded tiles in the snapshot; retire `_stagedPlacementEdits` | 167 FR-001/FR-007, SC-002 |
| E-11 | Session: exit warning; route chunk-manipulator + every placement edit through one undo/save path | 168 FR-002/FR-003 |
| E-12 | Chunk clipboard migrated onto the plugin host | 169 |
| E-13 | Add-placement in the viewport, routed through the session | 175 FR-002 |
| E-14 | Edit journal: crash recovery + resumable sessions | 172 |
| E-15 | DBC/DB2 table browser; table editing + loose save | 170, 171 |
| E-16 | Asset integrity validators + census; repair patterns (after census) | 173 FR-001/007/008; 174 |
| E-17 | ADT tile creation | 177 |
| E-18 | PM4-guided object transfer: in-scene overlay, off-render-thread preview, real-pair proof, parked transfer half | 176 T002–T008 |

### C. Composition workbench

| ID | Item | Source |
|---|---|---|
| E-20 | `uniqueId` placement reconciliation + collision report | 203 FR-002/003, SC-002 |
| E-21 | Orthographic whole-map selection canvas; shared 2D/3D selection; Chunk Manipulator retirement | 219 Ph 3/4; 222 T111 |
| E-22 | 45° / free-angle rotation with reported approximation | 219 Ph 6 |
| E-23 | Composition pixel-regression suite + doc cleanup | 219 Ph 8; 222 T112–T114 |
| E-24 | Layer stack as its own panel; DBC child-map suggestions in the add flow | 222 T109/T110 |
| E-25 | Operator-reported cartography defects: MCSH bake option, broken "Include WMO geometry", no-water minimap shading, cell fine-tune granularity, offset counter sizing | 232 T067–T071 |
| E-26 | Composed-map minimap synthesis; texture restoration for stripped phase maps; MDX fire/water light regression | 232 T060–T063 |
| E-27 | UniqueId colour-coding; chunk off-by-one re-audit | 232 T055, T065 |
| E-28 | Transplant provenance record; pop-out donor picker with three visual states | 208 FR-007, US2 |

### D. Generation & authoring

| ID | Item | Source |
|---|---|---|
| E-30 | Client-constrained generator: only assets from the loaded client/era (no cross-expansion assets) | 236 US4/FR-011–013; 191 T022 |
| E-31 | Live multi-tileset texturing + organic curvature in the generator; generator UI panel | 191 T022–T023, AC5.3 |
| E-32 | Rosetta-indexed object placement | 230 US1 |
| E-33 | WMO doodad placement editing, custom doodad sets, WMO writing | 220 US1–US5 |

## Operator verification owed on shipped code

232 witnesses (T015d rotated seam, T051 lock badge, T053, T054, T056–T059, T064, T066); 191 T024 real
client; 194 NFR-001/003; 196 AC-002/AC-005; 222 Gate 3 (30-second add-align).
