# Technical Plan: Spec 195 — Overhead Chunk Manipulator & Multi-Tile Sub-Cell Transposition Engine

## 1. Architectural Overview & Component Structure

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ViewerApp UI Layer                              │
│                                                                        │
│   ┌────────────────────────┐         ┌─────────────────────────────┐   │
│   │ 2D Overhead Canvas     │         │ 3D Perspective Viewport     │   │
│   │ (Minimap texture base, │         │ (Bounding Box Wireframe,    │   │
│   │  533m tile / 33m chunk │         │  Gizmo Drag Preview,        │   │
│   │  selection rectangle)  │         │  Shift Offset Controls)     │   │
│   └───────────┬────────────┘         └──────────────┬──────────────┘   │
└───────────────┼─────────────────────────────────────┼──────────────────┘
                │                                     │
                ▼                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  ChunkSelectionRegion (Global Coordinates)             │
│   - GlobalChunkCoordinate (Gx, Gy) ∈ [0..1023]                         │
│   - SelectedChunks HashSet<(int Gx, int Gy)>                           │
│   - SelectionBounds (MinGx, MinGy, MaxGx, MaxGy)                       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│           ChunkTranspositionService (WowViewer.Core.Editor)            │
│   - ExtractSelectionPayload(sources, includeObjects, includeTextures)  │
│   - TransformPayload(payload, deltaGx, deltaGy, deltaZ, rotation)      │
│   - HarmonizeTextures(sourceChunk, targetTileMtex)                     │
│   - ApplyTransposition(payload, destinationTiles)                      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               EditorSession & Undo/Redo Operations                     │
│   - ChunkTranspositionOperation (IEditorOperation)                     │
│   - ReplaceTileChunksAndRebuild (TerrainManager / VlmTerrainManager)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. File Changes & New Additions

### Layer 1: Core Data Models & Operations (`WowViewer.Core.Editor`)
1. **`GlobalChunkCoordinate.cs`** [NEW]:
   - `public readonly record struct GlobalChunkCoordinate(int Gx, int Gy)`
   - Conversion helpers: `FromTileAndChunk(int tileX, int tileY, int chunkX, int chunkY)` $\to$ `(Gx, Gy)`.
   - Decomposition: `TileX`, `TileY`, `ChunkX`, `ChunkY`, `WorldCenter`.
2. **`ChunkSelectionRegion.cs`** [NEW]:
   - Tracks a set of `GlobalChunkCoordinate` items.
   - Methods: `Add`, `Remove`, `Toggle`, `AddRectangle(from, to)`, `AddTile(tileX, tileY)`, `Clear`, `Bounds`.
3. **`ChunkTranspositionPayload.cs`** [NEW]:
   - Data structure holding extracted chunk states (Heights, Normals, HoleMask, Layers, AlphaMaps, ShadowMap, Mccv, Liquid, ModelPlacements, WmoPlacements) relative to an origin global coordinate $(G_{x0}, G_{y0})$.
4. **`ChunkTranspositionOptions.cs`** [NEW]:
   - Options: `IncludeHeights`, `RelativeHeights`, `IncludeTextures`, `IncludeHoles`, `IncludeLiquid`, `IncludeM2Placements`, `IncludeWmoPlacements`, `RotationDegrees` ($0, 90, 180, 270$), `MirrorX`, `MirrorY`, `ElevationOffset`.
5. **`ChunkTranspositionService.cs`** [NEW]:
   - Performs the transposition, texture layer palette mapping, placement offset calculation, and boundary normal recalculation.
6. **`ChunkTranspositionOperation.cs`** [NEW]:
   - Implements `IEditorOperation` for complete reversibility in `EditorSession`.

---

### Layer 2: Editor Plugin & UI Integration (`WoWViewer`)
1. **`ChunkManipulatorEditorPlugin.cs`** [NEW]:
   - Registered in `_editorHost` with `Id = "chunk-manipulator"`.
   - Exposes selection state, active tool mode (Select, Move, Rotate, Erase), and transform parameters.
2. **`ViewerApp_Editor.cs`** [MODIFY]:
   - Add `DrawChunkManipulatorPluginPanel()` rendering the overhead canvas, coordinate readouts, selection statistics, transposition buttons, and texture harmonization options.
3. **`ViewerApp_MinimapAndStatus.cs` / `MinimapHelpers.cs`** [MODIFY]:
   - Support rendering the `ChunkSelectionRegion` directly on top of the interactive minimap and overhead viewports.
4. **`ViewerApp_Sidebars.cs`** [MODIFY]:
   - Unify and modernize the legacy Chunk Clipboard panel to use `ChunkManipulatorEditorPlugin`.

---

## 3. Phased Implementation Roadmap

- **Phase 1: Global Coordinate Space & Selection Region Model**
  - Implement `GlobalChunkCoordinate`, `ChunkSelectionRegion`, and unit tests.
- **Phase 2: Core Chunk Transposition & Texture Palette Engine**
  - Implement `ChunkTranspositionPayload`, `ChunkTranspositionOptions`, `ChunkTranspositionService`, `ChunkTranspositionOperation`, and unit tests.
- **Phase 3: Interactive 2D Overhead Map Canvas & In-Viewport 3D Gizmo**
  - Wire interactive overhead canvas with zoom/pan and box-selection in `ViewerApp_Editor.cs` and `MinimapHelpers.cs`.
- **Phase 4: Multi-Tile Persistence, Verification & Undo/Redo**
  - Verify multi-tile pasting, placement translation, and `EditorSession` undo/redo across loaded tiles.
