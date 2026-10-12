# Spec 261: WoW Forever (11.2.7 / 1.60.1) Terrain Hole Fidelity & Format Conformance

**Owner**: Epic 248 (Formats, Readers, Writers & Conversion) & Epic 249 (Renderer Performance, Lighting & Correctness)  
**Origin**: Operator prompt (2026-10-04) regarding WoW Forever (client build `1.60.1.70205`, branched from 11.2.7 client in Jan 2025) terrain hole decoding accuracy, ADT v18 conformance, and coordinate alignment.  
**Status**: In Progress  

---

## 1. Executive Summary

When inspecting terrain in the WoW Forever client (`wow_classic_beta`, build `1.60.1.70205`, 11.2.7 engine line from January 2025), terrain holes over subterranean entrances (such as the Deathknell church crypt stairs, caves, and open graves) appeared missing or improperly cut in earlier viewer releases (`v0.6.0-alpha1`).

This specification resolves and solidifies terrain hole fidelity across all client eras, with specific focus on modern 11.2.7 / 1.60.1 split ADTs:
1. **Format Conformance Verification**: Thoroughly audited the format specification against `wowdev.wiki/ADT/v18` (all 50 sections) and validated all split ADT chunk streams (root, obj0, obj1, tex0, lod) via CASC FileDataIDs. Confirmed that no auxiliary or alternative hole chunks exist (sections 49 `MASD` and 50 `MALP` are dedicated solely to shoreline distance fields and LOD liquid quadtree patches).
2. **64-bit High-Res Holes Decoding**: Confirmed the 8-byte payload structure at MCNK offset `0x14` when MCNK flag `0x10000` (`high_res_holes`) is present. In modern clients, the legacy 16-bit field at offset `0x3C` is set to `0x0000`. In older viewer builds (`v0.6.0-alpha1`), reading only offset `0x3C` caused all modern holes to be completely missed.
3. **Little-Endian Coordinate Translation**: Verified the bit-to-coordinate mapping. Reading 8 bytes as a Little-Endian `uint64` maps row `cellY` (0..7, along World X) to byte index `r = 0..7` and column `cellX` (0..7, along World Y) to bit index `c = 0..7`:
   $$\text{bit} = \text{cellY} \times 8 + \text{cellX}$$
   Seam continuity testing across tile and chunk boundaries in multiple zones (Deathknell, Raven Hill, Elwynn) proves 100% seam continuity across boundaries with 0.00 yard gap.
4. **World Geometry Alignment**: Verified holed bounding boxes against real WMO and M2 placements in Deathknell (WMO 111538 Church crypt stairs, Open Grave M2s), proving sub-yard alignment with entrance geometry.
5. **Universal Consumer Consistency**: Ensured identical hole testing and mesh index skipping across all rendering, export, and editing subsystems (`TerrainTileMeshBuilder`, `TerrainMeshBuilder`, `MapGlbExporter`, `TerrainChunkMath`, `TerrainHeightmapIo`, `GroundEffectPlacementModels`, `WorldTerrainHoleMask`, `AlphaTerrainAdapter`).
6. **Backward Compatibility**: Guaranteed 100% fidelity for legacy 16-bit hole masks (4×4 groups of 2×2 cells) used in pre-5.3 clients (Alpha 0.5.3, 1.12, 3.3.5) with lossless upsampling and downsampling.

---

## 2. Requirements & Acceptance Criteria

### User Stories

- **US1: Modern Client Terrain Hole Precision**: As a user viewing maps from modern clients (such as WoW Forever 1.60.1 / 11.2.7), I see terrain holes properly cut out for crypt entrances, dungeons, open graves, and cave openings without solid terrain blocking interior geometry.
- **US2: Subsystem Consistency**: As a user exporting terrain to GLB, editing chunks in the workbench, or generating ground effects/flora, terrain holes are consistently recognized and respected across all subsystems without discrepancies.
- **US3: Legacy Era Backward Compatibility**: As a user viewing or converting legacy Alpha 0.5.3, Vanilla 1.12, or WotLK 3.3.5 maps, 16-bit 4×4 hole groups continue to render identically without regressions.

### Acceptance Criteria

- **AC-001**: `TerrainHoleMath.ReadHoleMasks` correctly identifies `high_res_holes` (flag `0x10000`) and reads the 64-bit mask from offset `0x14` when present, while reading 16-bit low-res mask from offset `0x3C` for legacy chunks.
- **AC-002**: `TerrainHoleMath.IsCellHoled64` implements the canonical bit test `((holeMask64 >> (cellY * 8 + cellX)) & 1UL) != 0UL`, maintaining seamless chunk-boundary continuity.
- **AC-003**: `TerrainHoleMath.UpsampleLowResToHighRes` losslessly converts 16-bit masks (4×4 groups) into 64-bit masks (8×8 cells) where each group bit expands into its corresponding 2×2 quad of cells.
- **AC-004**: `TerrainHoleMath.DownsampleHighResToLowRes` correctly aggregates 64-bit high-res masks into 16-bit representations by setting the group bit if any of the 4 subcells are holed.
- **AC-005**: All consumers (`TerrainTileMeshBuilder`, `TerrainMeshBuilder`, `MapGlbExporter`, `TerrainChunkMath`, `TerrainHeightmapIo`, `GroundEffectPlacementModels`, `WorldTerrainHoleMask`, `AlphaTerrainAdapter`) utilize `TerrainHoleMath` consistently.
- **AC-006**: Unit tests in `TerrainHoleMathTests.cs` validate synthetic bit patterns, real CASC chunk samples from WoW Forever 1.60.1, round-trip conversions, and geometry alignment against real client assets.
- **AC-007**: Strict compliance with `AGENTS.md`: §9 Governance receipts, §10 God-Class freeze (zero additions to `WorldScene` or `ViewerApp`), and §4 Core library architectural boundaries.
