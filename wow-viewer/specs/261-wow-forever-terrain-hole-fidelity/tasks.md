# Tasks: Spec 261 — WoW Forever Terrain Hole Fidelity & Format Conformance

- [x] **Phase 1: Format Audit & Mathematical Analysis**
  - [x] T001: Audit `wowdev.wiki/ADT/v18` in its entirety (sections 1..50) for modern hole format specifications and auxiliary chunks.
  - [x] T002: Inspect split ADT companion chunks (root, obj0, obj1, tex0, lod) via CASC storage to verify no alternative hole streams exist.
  - [x] T003: Formulate and solve the coordinate mapping equations (row vs col, Little-Endian bit order) using seam continuity across adjacent chunks.
  - [x] T004: Validate hole boundary continuity against real WMO (Church crypt 111538) and M2 (Open Graves) placements in Deathknell.

- [x] **Phase 2: Universal Implementation & Consumer Alignment**
  - [x] T010: Verify `TerrainHoleMath.cs` canonical bit testing (`IsCellHoled64`, `IsCellHoled16`) and up/down-sampling.
  - [x] T011: Verify `Mcnk.cs` reading of 64-bit mask from offset 0x14 when flag 0x10000 is present.
  - [x] T012: Verify `StandardTerrainAdapter.cs` propagation of `HolesHighRes` into `TerrainChunkData.HoleMask64`.
  - [x] T013: Verify mesh builders (`TerrainTileMeshBuilder.cs`, `TerrainMeshBuilder.cs`) skip triangles in holed cells using `TerrainHoleMath.IsCellHoled64`.
  - [x] T014: Verify exporters and workbench tools (`MapGlbExporter.cs`, `TerrainChunkMath.cs`, `TerrainHeightmapIo.cs`) use consistent hole logic.
  - [x] T015: Verify `GroundEffectPlacementModels.cs` and `WorldTerrainHoleMask.cs` honor 64-bit hole masks.

- [x] **Phase 3: Test Suite & Real CASC Data Verification**
  - [x] T020: Add real CASC chunk tests in `TerrainHoleMathTests.cs` using WoW Forever 1.60.1 chunk masks (Deathknell chunks [5,8] and [5,9]).
  - [x] T021: Add seam continuity regression test verifying adjacent chunk boundary matching.
  - [x] T022: Run test suite via `dotnet test` and confirm 100% pass rate.

- [x] **Phase 4: Governance & Documentation**
  - [x] T030: Compile verification receipt under `specs/261-wow-forever-terrain-hole-fidelity/evidence/receipt-spec261.md` per `AGENTS.md` §9.2.
  - [x] T031: Register Spec 261 in `specs/STATUS.md`.
  - [x] T032: Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.
