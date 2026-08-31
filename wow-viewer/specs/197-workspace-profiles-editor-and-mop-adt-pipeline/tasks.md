# Spec 197 Tasks: UI Workspace Profiles, Editor Mode Integration, PM4 Mouse Inspection, Multi-Client Map Staging & MoP 5.0.1 ADT Pipeline

Task status below distinguishes implemented reader/runtime and conversion-boundary
work from native-evidence and native-writer work that is still open.

## Phase 1: Workspace Profile Switcher & Editor Tab Integration
- [ ] **T101**: Define `ViewerWorkspaceMode` (`Viewer`, `Editor`, `Archaeology`) in `ViewerApp.cs`.
- [ ] **T102**: Add the Top-Bar Mode Switcher button group (`[Viewer]`, `[Editor]`, `[Archaeology]`) in `DrawMenuBar()` / `ViewerApp.cs`.
- [ ] **T103**: Expose the missing `Editor` tab button (`DrawTopTabButton(WorkbenchTab.Editor, "Editor")`) in `ViewerApp_Sidebars.cs`.
- [ ] **T104**: Filter right-sidebar tabs (`Quick`, `Inspect`, `Scene`, `Utilities`, `Editor`, `Archaeology`) according to active `ViewerWorkspaceMode`.
- [ ] **T105**: Test profile switching across Viewer, Editor, and Archaeology modes.

## Phase 2: Menu Audit & "MK Dataset" Legacy Purge
- [ ] **T106**: Audit all items in `DrawMainMenu()` and remove dead "MK Dataset" / `MkDatasetHarvester` references.
- [ ] **T107**: Consolidate ML and offline export options under `Tools > Offline Data / Conversion`.
- [ ] **T108**: Verify clean compilation and zero leftover references to `MkDataset`.

## Phase 3: Direct PM4 Viewport Mouse Raycast & Selection
- [ ] **T109**: Implement bounding-box and triangle raycast helper for PM4 geometry in `WorldScene.cs`.
- [ ] **T110**: Wire PM4 candidates into `TryHandleSceneClickSelection` in `ViewerApp_ClickSelection.cs`.
- [ ] **T111**: Enable Inspector synchronization when a PM4 object is picked via viewport click.
- [ ] **T112**: Test mouse picking of PM4 objects, surfaces, and collision bounding boxes.

## Phase 4: Multi-Client Map Staging & Restoration Library
- [ ] **T113**: Create `MultiClientMapStagingService.cs` supporting multiple simultaneous `IDataSource` client mounts.
- [ ] **T114**: Implement cross-client chunk slice extraction and staging into a unified restoration target.
- [ ] **T115**: Write unit tests for multi-client mounting and chunk copying in `MultiClientMapStagingTests.cs`.

## Phase 5: Ghidra Analysis & 4.3.4–5.1 MoP ADT Pipeline
- [x] **T116**: Connect to the GhidraMCP HTTP session (`127.0.0.1:8089`) on `Wow.exe` 5.0.1.15464; install the 6.0.0 Python bridge with `uv` and validate stdio auto-connect. Full binary extraction remains T117.
- [-] **T117**: Decompile and extract `CMapChunk`, `CMapTile`, `MHID`, `MDID`, `MCXH`, WMO terrain seam blending, and dormant/developer paths. Evidence is recorded in [`research-ghidra-5.0.1.md`](research-ghidra-5.0.1.md:1), [`5.0.1-dead-dormant-partial-rendering.md`](evidence/5.0.1-dead-dormant-partial-rendering.md), and [`wow-5.0.1-adt-wdt-definitive.md`](../../docs/architecture/wow-5.0.1-adt-wdt-definitive.md). Confirmed so far: `MVER == 0x12`, root + selected `_obj0`/`_obj1` and `_tex0`/`_tex1`, 256 outer MCNK records per file-data object, root-only MCNK header consumption, map-table `Flag_Exists` admission, the native `MCBB`/`MCDD`/`MCMT`/`MCRD`/`MCRW` dispatcher entries, and a partial liquid factory default branch. `MHID`/`MDID`/`MCXH`, alternate `_lod.adt` use, complete blend/seam shader details, indirect reachability, and one path-builder address remain open.
- [ ] **T117a**: Complete the native WDT/MAIN map-table and LOD-band extraction, including the population of `Flag_Exists`, the relationship between suffix bands and internal `LOD_COUNT`, and any alternate 5.0.1 `_lod.adt` path. Documentation now records the consumer-side `Flag_Exists` gate and proven `_obj0`/`_obj1` + `_tex0`/`_tex1` selection; population semantics remain open.
- [ ] **T117b**: Complete the native terrain blend/render extraction for `MCBB`, alpha/height inputs, WMO-to-terrain seam ownership, and optional/dead/partial renderer paths. The initial dead/dormant inventory and definitive guide are written; numeric blend/seam consumers and indirect reachability remain open.
- [x] **T118**: Implement the shared split-ADT reader/runtime slice: family and file-kind detection for `_obj0`/`_obj1` and `_tex0`/`_tex1`, root-plus-selected-band loading, headerless companion MCNK routing, and sparse MCIN slot preservation. The existing `MopAdtChunkParser` contracts remain evidence-gated for native semantics.
- [x] **T119**: Wire the supported 5.0.1 split-family discovery/loading path through `StandardTerrainAdapter` and the runtime data flow. Height-texture shader parity and WMO seam blending remain open and are not claimed complete.
- [x] **T120**: Add focused regression tests for band-1 discovery/family resolution, root/headered and split/headerless MCNK handling, sparse physical slots, and object-reference retention.
- [x] **T121**: Define explicit map-conversion source/target formats in Core, expose target selection in the viewer, report lossy routes, route split-to-Alpha correctly, and guard target-aware `LkAdtWriter` calls to LK v18 only. Keep native MoP split output disabled because no split writer exists.
- [ ] **T122**: Audit remaining compact MCIN consumers and the core runtime convenience path; extend merger/texture-transfer/converter inputs to `_obj1`/`_tex1` and add sparse merger coverage.
- [ ] **T123**: Define a slot-aware canonical ADT document and per-field loss policy, then implement/test a native-aligned MoP split writer before enabling `MopSplitAdt` as a supported target.
