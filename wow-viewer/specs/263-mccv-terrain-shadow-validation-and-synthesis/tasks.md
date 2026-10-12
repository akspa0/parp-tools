# Tasks: Spec 263 — 1.60 MCCV Terrain Shadow Ground-Truth Validation & Bidirectional Renderer Synthesis

- [x] **Phase 1: 1.60 MCCV Extraction & 145-Vertex Lattice Geometry**
  - [x] T001: Implement `MccvTerrainShadowService.cs` in `WowViewer.Core.IO/Maps/` extracting and packing 145-vertex MCCV BGRA byte chunks (580 bytes) and interpolating to 2D elevation/shadow matrices.
  - [x] T002: Add unit tests in `MccvTerrainShadowServiceTests.cs` verifying 145-vertex lattice coordinates (9x9 outer + 8x8 inner) and BGRA serialization.
  - [x] T003: Implement `mccv_shadow_comparator.py` in `data-harvester/src/harvester/v60/` with `rasterize_mccv_tile_to_grid` and `synthesize_mccv_from_residual`.
  - [x] T004: Add test coverage in `data-harvester/tests/v60/test_mccv_shadow_comparator.py`.

- [x] **Phase 2: Comparative Correlation Engine & Residual Cross-Validation**
  - [x] T010: Implement `compare_residual_to_mccv()` calculating NCC, MAE, SSIM, and dynamic range ratio.
  - [x] T011: Implement CLI script `scripts/v60_compare_mccv_residuals.py` evaluating 1.12.1 minimap residuals against 1.60 MCCV data and producing diagnostic comparison sheets with zone tier stratification.
  - [x] T012: Verify NCC >= 0.70 and ridge alignment coincidence >= 75% on Tier 1 trusted baseline zones (Westfall, STV, Elwynn) (AC-002, AC-003).

- [x] **Phase 3: 1.60 MCCV Synthesis & ADT Injection Bridge**
  - [x] T020: Implement synthesis of 145-vertex BGRA MCCV arrays from 2D residual shadows in `MccvTerrainShadowService.cs` and `mccv_shadow_comparator.py`.
  - [x] T021: Verify roundtrip compatibility into `LkMcnkData` / `AdtChunkWriter` (AC-004).

- [x] **Phase 4: Renderer Modulation Verification in WoW: Forever**
  - [x] T030: Verify `TerrainShader.cs` and `TerrainMinimapCompositor.cs` evaluate MCCV modulation (2.0 * mccv) cleanly without color clipping or distortion (AC-005).

- [x] **Phase 5: Governance, Receipts & Documentation**
  - [x] T040: Register Spec 263 in `wow-viewer/specs/STATUS.md`.
  - [x] T041: Compile verification evidence into `specs/263-mccv-terrain-shadow-validation-and-synthesis/evidence/receipt-spec263.md` per `AGENTS.md` §9.2.
  - [x] T042: Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.
