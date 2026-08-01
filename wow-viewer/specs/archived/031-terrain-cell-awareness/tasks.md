# Tasks: Terrain Cell Awareness and Sub-Chunk Addressing
**Input**: Design documents from `wow-viewer/specs/031-terrain-cell-awareness/`

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US3)

---

## Phase 1: Terrain Data Structures in Core.IO (US1 — P1)
**Goal**: Parse 145-vertex layout, build cell grid, face planes, hole mask.

**Independent Test**: Load MCNK from staged client, verify 145 vertices, 256 face planes, correct hole mask.

- [ ] T001 [US1] Extend `AdtChunkReader` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtChunkReader.cs` — expose 145 vertices (81 outer entries 0-80, 64 inner entries 81-144 in interleaved MCVT order)
- [ ] T002 [US1] Build `AdtCellGrid` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtCellGrid.cs` — 8x8 cells, per-cell diagonal split variant (from Ghidra `0x008a10b8`/`0x008a10bc` lookup tables or height-gradient inference)
- [ ] T003 [US1] Build `AdtFacePlaneBuilder` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtFacePlaneBuilder.cs` — 256 face planes (8x8 cells × 4 triangles per cell), plane equation from cross product
- [ ] T004 [US1] Build `AdtHoleMask` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtHoleMask.cs` — 16-bit holes field, 4x4 grouping, test: `(g_holeMask[(subX>>1) + (subY>>1)*4] & chunk->holes) == 0`
- [ ] T005 [P] [US1] Build `AdtCellAddress` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtCellAddress.cs` — 13-bit packed: sub-chunk (3 bits) + chunk (4 bits) + ADT (6 bits)
- [ ] T006 [US1] Extend `AdtTerrainData` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTerrainData.cs` — carry cell grid, face planes, hole mask, cell address
- [ ] T007 [P] [US1] Add unit tests in `wow-viewer/tests/WowViewer.Core.Tests/AdtCellGridTests.cs` — vertex count (145), face plane count (256), hole mask test per cell
- [ ] T008 [US1] Validate against staged client `3_3_5_12340` — load MCNK, verify 145 vertices, 256 face planes
- [ ] T009 [US1] Port normal unpacking: Y/Z/X order, scale ~0.251388, add to `AdtChunkReader`
- [ ] T010 [P] [US1] Validate normal count (145), correct byte order, add to unit tests

**Checkpoint**: Core.IO reads 145-vertex layout, builds cell grid + face planes + hole mask. Build passes. Tests green.

---

## Phase 2: Runtime Cell Awareness (US1 + US2 — P1)
**Goal**: Runtime carries cell grid + spatial queries work.

**Independent Test**: Spatial query for known world position (e.g., Deadmines entrance) returns correct ADT/chunk/cell indices.

- [ ] T011 [US2] Extend `WorldTerrainChunkData` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainChunkData.cs` — 145 vertices + cell grid + face planes
- [ ] T012 [US2] Build `WorldTerrainCellGrid` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainCellGrid.cs` — 64 cells, diagonal split, 4 triangles each
- [ ] T013 [P] [US2] Build `WorldTerrainFacePlanes` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainFacePlanes.cs` — 256 planes, per-cell access
- [ ] T014 [US2] Build `WorldTerrainHoleMask` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainHoleMask.cs` — hole test per cell, 2x2 block granularity
- [ ] T015 [US2] Build `WorldCellAddress` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Spatial/WorldCellAddress.cs` — world (X,Y) → (ADT, chunk, cell, is_holed); reverse: (ADT, chunk, cell) → world AABB
- [ ] T016 [P] [US2] Wire into `WorldTerrainLodBuilder` — LOD respects cell boundaries, far terrain uses low-detail 17x17
- [ ] T017 [P] [US2] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WorldCellAddressTests.cs` — spatial query, reverse query, hole mask query
- [ ] T018 [US2] Validate against native client: query Deadmines entrance, verify ADT/chunk/cell indices
- [ ] T019 [US2] Validate hole mask: holed cell returns `is_holed=true`, native vs reconstructive mode
- [ ] T020 [US2] Validate cell AABB: reverse query produces correct world-space box

**Checkpoint**: Runtime has cell awareness + spatial queries. Build passes. Tests green.

---

## Phase 3: Mesh Builder with 145-Vertex Topology (US3 — P2)
**Goal**: Mesh uses correct 145-vertex topology with per-cell diagonal splits.

**Independent Test**: Render chunk, compare against 9x9 flat grid — inner vertices add mid-cell detail.

- [ ] T021 [US3] Extend `WorldTerrainTileBuilder` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainTileBuilder.cs` — 145-vertex index buffer (256 triangles)
- [ ] T022 [US3] Implement per-cell diagonal split using `WorldTerrainCellGrid` — 4 triangles per cell, correct vertex indices
- [ ] T023 [US3] Implement hole mask handling: native mode (exclude holed cells) vs reconstructive mode (render all + toggleable overlay mask)
- [ ] T024 [P] [US3] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/TerrainMeshBuilderTests.cs` — triangle count (256), diagonal split correctness, hole handling
- [ ] T025 [US3] Validate mesh: render chunk, verify mid-cell inner vertices visible
- [ ] T026 [US3] Validate diagonal splits: compare against native client screenshot
- [ ] T027 [US3] Validate hole mask: holed cells excluded or overlaid correctly in both modes
- [ ] T028 [P] [US3] Performance test: mesh build < 1ms per chunk

**Checkpoint**: Mesh renders with correct 145-vertex topology. Inner vertices visible. Build passes. Tests green.

---

## Phase 4: Cell-Level Raycast/Collision (US2 — P2)
**Goal**: Efficient ray vs terrain collision using cell face planes.

**Independent Test**: Cast ray at known terrain position, verify hit cell is correct.

- [ ] T029 [US2] Build `WorldCellRaycast` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Spatial/WorldCellRaycast.cs` — ray vs 256 face planes, early out on first hit
- [ ] T030 [US2] Optimize: use `WorldCellAddress` to narrow search to cells along ray path
- [ ] T031 [P] [US2] Add unit tests in `wow-viewer/tests/WowViewer.Core.Runtime.Tests/WorldCellRaycastTests.cs` — ray hits correct cell, miss returns null, performance < 10us
- [ ] T032 [US2] Validate against native client: raycast at known position (Deadmines entrance), verify cell index
- [ ] T033 [P] [US2] Wire into world scene collision system — terrain picking uses cell raycast
- [ ] T034 [US2] Performance test: raycast < 10us per ray for typical scene

**Checkpoint**: Cell-level raycast works. Collision uses cell face planes. Build passes. Tests green.

---

## Dependencies & Execution Order

### Phase Dependencies
- **Phase 1** → **Phase 2**: Data structures must exist before runtime awareness
- **Phase 2** → **Phase 3**: Runtime cell awareness must exist before mesh builder
- **Phase 2** → **Phase 4**: Face planes must exist before raycast

### Parallel Opportunities
- **Phase 1**: T005 + T007 can run in parallel (different files)
- **Phase 2**: T013 + T016 + T017 can run in parallel (different files)
- **Phase 3**: T024 + T028 can run in parallel
- **Phase 4**: T031 + T033 can run in parallel

### Execution Strategy
1. **Phase 1** first (foundation — data structures)
2. **Phase 2** after Phase 1 (runtime awareness + spatial queries)
3. **Phase 3** after Phase 2 (mesh builder with correct topology)
4. **Phase 4** after Phase 2 (raycast/collision)

---

## Task Count
- **Total**: 34 tasks
- **Phase 1**: 10 tasks (data structures)
- **Phase 2**: 10 tasks (runtime awareness + spatial queries)
- **Phase 3**: 8 tasks (mesh builder)
- **Phase 4**: 6 tasks (raycast/collision)
- **Parallel tasks**: 8 tasks marked [P]
