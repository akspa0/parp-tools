# Task Checklist: Spec 288 — PM4 Overlay Modernization, Color-Coding & Sidebar Refactor

## Phase 1: Color-Coding Service & Palettes

- [x] **T001**: Create `Pm4ColorCodingService.cs` in `WoWViewer/Terrain/Pm4/Services/` implementing instance clustering, semantic roles, elevation ramp, and match status.
- [x] **T002**: Implement `Pm4ColorPalette` enum and color mapping logic in `Pm4ColorCodingService`.
- [x] **T003**: Implement dynamic legend generator `GetLegendEntries(palette)` returning color swatches, labels, and entity counts.
- [x] **T004**: Wire `Pm4ColorCodingService` into `Pm4OverlayScene.GetPm4ObjectColor()`.
- [x] **T005**: Add unit tests in `tests/WowViewer.Core.PM4.Tests/Pm4ColorCodingServiceTests.cs` validating non-zero instance color variation for doodads (`Ck24 == 0`) and elevation monotonicity.
- [x] **Gate A (Phase 1 Verification)**:
  - `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/ -c Debug --filter "FullyQualifiedName~Pm4ColorCodingService"` passes.
  - Receipt written to `specs/288-pm4-overlay-refactor-and-color-coding/evidence/phase1-color-coding.md`.

---

## Phase 2: Spatial Index & Fast Ray Picking

- [x] **T006**: Create `Pm4SpatialIndex.cs` in `WoWViewer/Terrain/Pm4/Services/` storing per-tile and per-object AABBs.
- [x] **T007**: Implement hierarchical three-stage ray-cast picking (Tile AABB -> Object AABB -> Möller-Trumbore triangle test).
- [x] **T008**: Implement $O(1)$ 3x3 tile neighborhood query in `Pm4SpatialIndex`.
- [x] **T009**: Refactor `Pm4OverlayScene.Selection.cs` to delegate ray-picking and spatial lookups to `Pm4SpatialIndex`.
- [x] **T010**: Add unit tests in `tests/WowViewer.Core.PM4.Tests/Pm4SpatialIndexTests.cs` verifying ray picking hit detection and tile culling.
- [x] **Gate B (Phase 2 Verification)**:
  - `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/ -c Debug --filter "FullyQualifiedName~Pm4SpatialIndex"` passes.
  - Receipt written to `specs/288-pm4-overlay-refactor-and-color-coding/evidence/phase2-spatial-index.md`.

---

## Phase 3: Mesh Renderer & Buffer Decomposition

- [x] **T011**: Create `Pm4OverlayMeshRenderer.cs` in `WoWViewer/Terrain/Pm4/Services/` managing OpenGL shader programs, VAOs, VBOs, and draw dispatches.
- [x] **T012**: Extract wireframe, solid-fill, and MSPV wall rendering from `Pm4OverlayScene.cs` into `Pm4OverlayMeshRenderer`.
- [x] **T013**: Implement per-tile geometry buffer caching to avoid per-frame CPU vertex copies for static tiles.
- [x] **T014**: Verify `Pm4OverlayScene.cs` and all extracted files are $\le 2,000$ lines conforming to AGENTS.md §10.
- [x] **Gate C (Phase 3 Verification)**:
  - `dotnet build I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore` compiles with 0 errors.
  - Line count audit confirms all files $\le 2,000$ lines.
  - Receipt written to `specs/288-pm4-overlay-refactor-and-color-coding/evidence/phase3-renderer-decomposition.md`.

---

## Phase 4: Modernized PM4 Sidebar Workbench

- [x] **T015**: Create `Pm4WorkbenchSidebarService.cs` in `WoWViewer/Terrain/Pm4/UI/` replacing legacy tab sections with `SharedUiWidgets` layout.
- [x] **T016**: Add interactive **Color Palette & Visual Legend** widget with clickable color swatches and active count badges.
- [x] **T017**: Add streamlined **Surface Class Filter Matrix** with individual class toggles.
- [x] **T018**: Integrate selected object inspection and one-click model duplication actions.
- [x] **T019**: Register all new sidebar controls in `eng/ui/ui-reference-inventory.json` and generate updated `UI-REFERENCE.md`.
- [x] **T020**: Run `Verify-UiReference.ps1` and ensure 0 errors and 0 warnings.
- [x] **Gate D (Phase 4 Verification)**:
  - Solution builds cleanly with 0 errors.
  - `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` passes with 0 errors, 0 warnings.
  - Receipt written to `specs/288-pm4-overlay-refactor-and-color-coding/evidence/phase4-sidebar-workbench.md`.
