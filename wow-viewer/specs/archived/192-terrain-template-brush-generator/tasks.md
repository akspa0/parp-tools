# Actionable Task Ledger: Spec 192 — Terrain Template Brush & Paste Library with Interactive In-Viewer Map Generator

**Spec ID**: `192`  
**Feature**: Terrain Template Brush & Paste Library with Interactive In-Viewer Map Generator  
**Status**: Complete  
**Created**: 2026-08-30  

---

## Phase 1: Core Data Models, Layer Allocator & Curated Paste Library (US1, US2)

- [x] **T001**: Implement `TerrainBrushPaste` and `TerrainPasteLayer` in `WowViewer.Core.IO.Terrain` representing relative height deltas, multi-layer alpha masks, texture paths, dimensions, and tags.
- [x] **T002**: Implement `TerrainBrushLibrary` in `WowViewer.Core.IO.Terrain` supporting categorization, tagging, tag-based filtering, and JSON serialization.
- [x] **T003**: Implement `TerrainLayerAllocator` in `WowViewer.Core.IO.Terrain` for intelligent multi-layer texture merging, weight normalization, and strict $\le 4$ layer-per-chunk budget enforcement.
- [x] **T004**: Implement `CuratedTerrainBrushLibrary` in `WowViewer.Core.IO.Terrain` providing built-in procedural and archetypal terrain motifs (cobblestone roads, grass knolls, stone plazas, riverbeds, mountain ridges, garden courtyards).
- [x] **T005**: Authored unit tests in `TerrainBrushPasteTests.cs` and `TerrainLayerAllocatorTests.cs` verifying serialization, layer budget enforcement, and tag searching.

---

## Phase 2: ADT Extraction & Serialization Engine (US2)

- [x] **T006**: Implement `AdtPasteExtractor` in `WowViewer.Core.IO.Terrain` extracting bounded terrain paste sub-regions directly from loaded `LkMcnkData` / `AlphaMcnk` structures.
- [x] **T007**: Add height delta baseline zeroing and alpha splat extraction per chunk in `AdtPasteExtractor`.
- [x] **T008**: Authored unit tests in `AdtPasteExtractorTests.cs` verifying extraction of height and alpha from mock/synthetic ADT chunk data.

---

## Phase 3: Terrain Stamp Operation with Seamless Feathering (US3, US5)

- [x] **T009**: Implement `TerrainStampOptions` (scale, rotation, height multiplier, feather radius, blend mode).
- [x] **T010**: Implement `TerrainStampOperation` in `WowViewer.Core.Editor.Operations` applying stamps with `SmoothStep`/cosine edge feathering, height blending, and normal recalculation.
- [x] **T011**: Implement Undo/Redo state capture in `TerrainStampOperation` for integration with `EditorSession`.
- [x] **T012**: Authored unit tests in `TerrainStampOperationTests.cs` verifying height delta application, feathering boundaries, and undo state restoration.

---

## Phase 4: Templated Procedural Map Generator (US4)

- [x] **T013**: Implement `TerrainMapTemplate` and `BiomePalette` defining themes (Garden Museum, Elwynn Forest, Barrens, Cobblestone City, Dun Morogh).
- [x] **T014**: Implement `TemplatedTerrainGenerator` in `WowViewer.Core.IO.Terrain` supporting procedural road network splines, courtyard placement, and terrain feature scattering.
- [x] **T015**: Enforce slope constraints ($\le 25^\circ$) and flat courtyard zones ($Z = 0$) in `TemplatedTerrainGenerator` to ensure 100% obstruction-free navigation.
- [x] **T016**: Connect `TemplatedTerrainGenerator` to `LkAdtWriter`, `AlphaTerrainAdapter`, and `WdlWriter` for multi-era map file output.
- [x] **T017**: Authored unit tests in `TemplatedTerrainGeneratorTests.cs` verifying multi-tile generation, texture layer limits, and slope bounds.

---

## Phase 5: Interactive In-Viewer Editor Plugin Surface (US3)

- [x] **T018**: Implement `TerrainTemplateEditorPlugin` in `WowViewer.Core.Editor.Plugins` implementing `IEditorPlugin`.
- [x] **T019**: Implement ImGui UI in `ViewerApp_Editor.cs` with:
  - Categorized Paste Catalog Browser with thumbnail previews.
  - Interactive Brush Stamping controls (Radius, Height Scale, Rotation, Feathering).
  - Viewport 3D stamping cursor and click-to-stamp trigger.
  - "New Map from Template" generation wizard dialog.
- [x] **T020**: Register `TerrainTemplateEditorPlugin` in `EditorHost` and verify catalog availability.

---

## Phase 6: CLI Tooling, Verification & Memory Bank Ledger (US1–US5)

- [x] **T021**: Add `terrain-generate-templated` command in `WowViewer.Tool.Inspect` (`Program.cs`).
- [x] **T022**: Run full solution test suite ensuring 100% green pass across all Core and Editor test suites.
- [x] **T023**: Commit changes to git and update `STATUS.md`, `activeContext.md`, and `progress.md`.
