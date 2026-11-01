# Spec 289 Tasks: PM4 3D Footprint Extrusion & Direct MPQ Matching

## Phase 1: Top-Alignment Fix in Doodad Magnetization & Duplication
- [x] **T001**: Update `AttractPositionToPm4Mesh` in `Pm4UnauthoredPlacementResolver.cs` and `FitModelToTarget` in `Pm4DoodadDuplicationEngine.cs` so that doodad placement position $Z$ places the model's top surface at the PM4 collision height: $P_Z = Z_{\text{pm4\_top}} - (H_{\text{local\_top}} \cdot s)$.
- [x] **T002**: Update `MagnetizePlacementToPm4` and `MagnetizeTilePlacementsToPm4` in `Pm4DoodadDuplicationService.cs` to calculate `modelTopRelativeZ` via `WorldPlacementTransform` and place the object so its top reaches the PM4 surface.
- [x] **T003**: Update existing and add new unit tests in `Pm4DoodadDuplicationTests.cs` verifying top-aligned placement positions and scale preservation.
- [x] **GATE-1**: Run `dotnet test` on `WowViewer.Core.PM4.Tests` ensuring 100% pass rate. (Receipt: `evidence/phase1-top-alignment.md`)

## Phase 2: Core 3D Footprint & Synthetic Wall Extrusion Engine
- [ ] **T004**: Create `Pm4FootprintExtruder.cs` in `WowViewer.Core.PM4/Footprint/` with 2D convex hull / boundary perimeter extraction and $Z_{\text{base}}$ resolution (MPRL terrain contact vs Placement-Z).
- [ ] **T005**: Implement wall quad generation in `Pm4FootprintExtruder.cs` connecting top boundary edges $[v_i, v_{i+1}]$ to base edges at $Z_{\text{base}}$, producing a closed 3D envelope.
- [ ] **T006**: Create unit tests in `Pm4FootprintExtrusionTests.cs` verifying convex hull extraction, $Z_{\text{base}}$ resolution, and extruded wall triangle generation.
- [ ] **GATE-2**: Run `dotnet test` on `WowViewer.Core.PM4.Tests` ensuring Phase 2 tests pass.

## Phase 3: Scale-Invariant Geometric Signature & MPQ Matcher
- [ ] **T007**: Create `Pm4GeometricShapeDescriptor.cs` in `WowViewer.Core.PM4/Footprint/` calculating $W, D, H$, aspect ratio $W/D$, height ratio $H/\max(W,D)$, and planar orientation.
- [ ] **T008**: Create `Pm4DirectModelMatcher.cs` in `WowViewer.Core.PM4/Matching/` ranking candidate model bounding boxes by geometric similarity score without donor map dependencies.
- [ ] **T009**: Add unit tests in `Pm4DirectModelMatcherTests.cs` testing scale-invariant matching against known assets (e.g. `STRANGLETHORNRUINS05.M2`).
- [ ] **GATE-3**: Run `dotnet test` on `WowViewer.Core.PM4.Tests` ensuring Phase 3 tests pass.

## Phase 4: Viewer Workbench Integration & UI Reference
- [ ] **T010**: Wire extruded wall wireframe rendering into `Pm4OverlayMeshRenderer.cs`.
- [ ] **T011**: Add "Show PM4 Extruded Walls" checkbox and "Match to Client Assets" action button in `Pm4WorkbenchSidebarService.cs`.
- [ ] **T012**: Run `Verify-UiReference.ps1` and verify all new controls pass with 0 errors/warnings.
- [ ] **GATE-4**: Full build and verification across `WoWViewer.csproj` and `Verify-UiReference.ps1`.

