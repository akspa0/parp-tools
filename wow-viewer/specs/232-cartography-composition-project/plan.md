# Implementation Plan: Cartography Composition

**Branch**: `223-ui-consolidation-audit` | **Date**: 2026-09-08 | **Spec**: [spec.md](spec.md)

## Summary

Make the viewer's phase-layer composition usable as restoration tooling: cell-granular alignment, durable per-map layer projects, and complete loose client-format map export. The active repair is the seam defect in quarter-turn layers: rotate the full donor-tile lattices once, then slice them into MCNKs. This replaces the unsafe per-chunk rotation without changing MPQ, ADT, WMO, M2/MDX readers, or `AlphaWdtWriter`.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary dependencies**: Silk.NET.OpenGL viewer; `WowViewer.Core` and `WowViewer.Core.IO`

**Storage**: human-editable JSON under `output/projects/cartography/`; exported loose ADT/WDT files only. MPQ/CASC remain read-only inputs.

**Testing**: xUnit Maps tests; `dotnet build` and `dotnet test` on `WowViewer.slnx`; operator visual/client-load gates where the spec requires them.

**Target platform**: Windows desktop viewer, Alpha 0.x and LK 3.3.5 client content.

**Project type**: desktop application plus shared core libraries.

**Performance goals**: no render-thread export work; live layer refresh remains bounded to the affected 64x64 grid tiles.

**Constraints**: one `PhaseCompositionPolicy` path for live and exported content; transform order is rotation then mirrors; 64x64 bounds are never crossed; locked layers reject edits; output stays inside the project-managed output root.

**Scale/scope**: one base map, ordered unresolved/enabled/disabled layer stack, 64x64 tiles and 16x16 MCNKs per tile. Free-angle content rotation and direct terrain sculpting remain out of scope.

## Constitution Check

| Gate | Result | Evidence / decision |
|---|---|---|
| Repo independence | Pass | Work is contained within `wow-viewer/` source, tests, specs, and output conventions. |
| Library first / one format owner | Pass | Transform math and project DTOs live in `WowViewer.Core`; adapters consume them. No parallel reader/writer is introduced. |
| Real-data validation | Pending operator gate | Unit tests establish invariants; the Deadmines/Azeroth screenshot and client load remain operator-owned. |
| No client-path assumptions | Pass | Client root is runtime configuration; plans name no source default. |
| Loose output only | Pass | Export emits loose ADT/WDT files, never MPQ/CASC. |
| Frozen-format-reader boundary | Pass | `AlphaWdtWriter` and proven readers are not modified by the seam-repair slice. |
| Source-decomposition rule | Pass | No new `ViewerApp` or `WorldScene` members are proposed; UI delegates to terrain/core owners. |

Re-check after each implemented phase. Any change to a working reader or `AlphaWdtWriter` needs a separately evidenced regression reason before implementation.

## Existing implementation recovery

Phase 1 code (cell offset, cross-border source resolution, adapter application, panel controls) and Phase 2 code (JSON project persistence, auto-load, save/load controls, locks) were already present when this plan was restored. Their focused source/test evidence is recorded in [evidence/t010-t013-phase1-receipt.md](evidence/t010-t013-phase1-receipt.md); their visual restart and lock receipts remain open. The task checklist must be reconciled only with a receipt—never by treating commit text or green compilation as runtime evidence.

## Implementation sequence

### Phase A — Reconcile completed core work and operator gates

1. Re-run the Maps suite for existing cell-resolution and project serialization tests.
2. Record source/test receipts for T010–T012 and T020–T021; do not close T013 or T022 without the operator's visual/restart witnesses.
3. Keep the exact DeadminesInstance-to-Azeroth Moonbrook alignment and save/restart scripts in [quickstart.md](quickstart.md).

### Phase B — T015 full-tile seam repair (highest priority)

1. Add a pure `AlphaTileData.RotateQuarterTurn(int, bool, bool)` operation that creates rotated full-tile channel lattices using one transform map.
2. Rotate height, normals (including XY direction), shadows, alpha planes, chunk-index metadata, and liquid-local grids; preserve data absent from the source as absent.
3. Add a synthetic 257x257 test proving the rotated lattice slices into the exact chunk slot that `TransformChunkSlot` selects; cover mirrors and identity.
4. Change the Alpha transformed load path to parse one donor tile, rotate at tile level, then call `ToTileLoadResult` for the target coordinates. Cell shifts keep their existing neighboring source policy.
5. Run focused Maps tests and solution build. Leave T015d open for the operator seam screenshot.

### Phase C — Remaining composition UX

1. Implement exclusive placed-tile picker mode and its reversible panel toggle (T050).
2. Implement persisted per-tile ownership locks and minimap badges (T051).
3. Repair the WL* inspection fall-through before unrelated renderer changes (T053).
4. Default Archaeology to Map Layers (T054), then add explicit UniqueId era tint data and rendering (T055).

### Phase D — Output and remaining renderer work

1. Diagnose and restore MDX fire/water/light behavior with an operator-visible regression witness (T060).
2. Route minimap synthesis through the loaded composition (T061).
3. Implement full-map export using the same policy path, background progress and integrity report (T030/T031/T062); operator validates client load and parity (T032).
4. Add persisted texture remapping only after the export parity path is stable (T063).

## Project Structure

```text
specs/232-cartography-composition-project/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/composition-boundary.md
├── tasks.md
└── evidence/

src/core/WowViewer.Core/Maps/
├── PhaseComposition.cs
├── PhaseLayerProjectFile.cs
└── AlphaTileData.cs

src/viewer/WoWViewer/Terrain/
├── AlphaTerrainAdapter.cs
├── StandardTerrainAdapter.cs
├── TerrainManager.cs
└── CartographyProjectStore.cs

tests/WowViewer.Core.Tests/Maps/
```

**Structure decision**: pure composition rules and serializable project state belong in Core; viewer terrain adapters perform configured I/O and UI only delegates to the terrain manager.

## Complexity Tracking

No constitution exception is proposed.

