# Tasks: Spec 195 — Overhead Chunk Manipulator & Multi-Tile Sub-Cell Transposition Engine

## Phase 1: Global Coordinate Space & Selection Region Model (US1)

- [x] T001 Implement `GlobalChunkCoordinate` struct in `WowViewer.Core.Editor/Operations/GlobalChunkCoordinate.cs` with tile/chunk conversions, world coordinate translation, and decomposition helpers
- [x] T002 Implement `ChunkSelectionRegion` class in `WowViewer.Core.Editor/Operations/ChunkSelectionRegion.cs` supporting rectangle ranges, whole-tile additions, individual toggles, and bounding box queries
- [x] T003 Author unit tests in `WowViewer.Core.Editor.Tests/Operations/ChunkSelectionRegionTests.cs` validating coordinate conversions, multi-tile rectangular box selections, and containment checks (Gate 1)

---

## Phase 2: Core Chunk Transposition & Texture Palette Engine (US2 & US3)

- [x] T004 Define `ChunkTranspositionPayload` and `ChunkTranspositionOptions` in `WowViewer.Core.Editor/Operations/`
- [x] T005 Implement `ChunkTranspositionService.cs` in `WowViewer.Core.Editor/Operations/` with height scaling, normal recalculation, texture layer re-indexing (MTEX palette merge), hole mask transfer, and object placement (MDDF/MODF) translation
- [x] T006 Implement `ChunkTranspositionOperation.cs` implementing `IEditorOperation` with snapshot capture for `EditorSession` undo/redo
- [x] T007 Author unit tests in `WowViewer.Core.Editor.Tests/Operations/ChunkTranspositionServiceTests.cs` verifying boundary crossing, texture harmonization, rotation ($90^\circ, 180^\circ, 270^\circ$), and placement shifting (Gate 2)

---

## Phase 3: Editor Plugin & Interactive Overhead Canvas (US4)

- [x] T008 Implement `ChunkManipulatorEditorPlugin.cs` in `WowViewer.Core.Editor/Plugins/` and register in `EditorHost`
- [x] T009 Implement the interactive overhead 2D selection canvas in `ViewerApp_Editor.cs` with tile borders ($16\text{ chunks} = 533.334\text{m}$), chunk grid lines ($33.334\text{m}$), and click-drag selection box
- [x] T010 Unify legacy chunk clipboard UI in `ViewerApp_Editor.cs` and `ViewerApp_Sidebars.cs` to delegate to `ChunkManipulatorEditorPlugin`
- [x] T011 Wire transposition execution and in-place replacement via `ReplaceTileChunksAndRebuild` (Gate 3)

---

## Phase 4: Full Validation & Live Integration (US5)

- [x] T012 Verify multi-tile cut/copy/paste, rotation, and placement translation in the live viewer
- [x] T013 Verify `EditorSession` Undo/Redo reversibility on multi-tile chunk transposition
- [x] T014 Execute full test suite `dotnet test` and update `specs/STATUS.md` and `memory-bank/activeContext.md` (Gate 4)
