# Plan: Spec 261 — WoW Forever Terrain Hole Fidelity & Format Conformance

## 1. Architecture & Bit-Mapping Specification

### 1.1 MCNK Header Layout across Eras

In legacy pre-5.3 ADTs (Vanilla through Mists 5.2):
- Header offset `0x14`: Sub-chunk offset table (`ofsMcvt`, `ofsMcnr`, etc.)
- Header offset `0x3C`: 16-bit `holes_low_res` bitmask. Each bit represents a 2×2 group of cells across the 4×4 chunk grid ($4 \times 4 = 16\text{ bits}$).
- MCNK flags bit `0x10000` is `0`.

In modern 5.3+ ADTs (including WoW Forever 1.60.1 / 11.2.7):
- MCNK flags has bit `0x10000` (`high_res_holes`) set.
- Header offset `0x14`: 64-bit `holes_high_res` bitmask (8 bytes). Each bit represents an individual cell across the 8×8 chunk grid ($8 \times 8 = 64\text{ bits}$).
- Header offset `0x3C`: Explicitly `0x0000`.

### 1.2 Endianness and Coordinate Mapping

When reading the 8 bytes from offset `0x14` into an unsigned 64-bit integer (`BinaryPrimitives.ReadUInt64LittleEndian`):
- Byte 0 (offset `0x14`) forms bits 0..7
- Byte 1 (offset `0x15`) forms bits 8..15
- ...
- Byte 7 (offset `0x1B`) forms bits 56..63

In the terrain mesh builder:
- Row index `cellY` (0..7) runs along the chunk's local Y direction (which corresponds to World $-X$, north-to-south).
- Column index `cellX` (0..7) runs along the chunk's local X direction (which corresponds to World $-Y$, east-to-west).
- The bit index within the 64-bit integer is:
  $$\text{bitIndex} = \text{cellY} \times 8 + \text{cellX}$$
- Testing cell hole status:
  $$\text{isHoled} = ((\text{holeMask64} \gg (\text{cellY} \times 8 + \text{cellX})) \ \& \ 1\text{UL}) \neq 0\text{UL}$$

### 1.3 Seam Continuity & Alignment Validation

In tile [29, 28] (Deathknell):
- Chunk [5, 8] bounds: $X \in [1300.00, 1333.33]$, $Y \in [1933.33, 1966.67]$
- Chunk [5, 9] bounds: $X \in [1266.67, 1300.00]$, $Y \in [1933.33, 1966.67]$
- Chunk [5, 8] raw hole bytes: `00 00 00 00 00 00 0F 0F`
  - Rows 6 and 7 are holed in cols 0..3 ($Y \in [1950.00, 1966.67]$).
  - Row 7 extends to local boundary $X = 1300.00$.
- Chunk [5, 9] raw hole bytes: `3F 3F 00 00 00 00 00 00`
  - Rows 0 and 1 are holed in cols 0..5 ($Y \in [1941.67, 1966.67]$).
  - Row 0 starts at local boundary $X = 1300.00$.
- Continuity verification:
  - $\text{minX8} = 1300.00$ meets $\text{maxX9} = 1300.00$ with zero gap.
  - Across the $X=1300.00$ seam, holed columns match across the boundary, creating a continuous crypt entrance corridor.

---

## 2. Implementation Approach & File Changes

### 2.1 Core Bitmath (`WowViewer.Core/Maps/TerrainHoleMath.cs`)
- Provide canonical static utilities:
  - `IsCellHoled64(ulong holeMask64, int cellX, int cellY)`
  - `IsCellHoled16(ushort holeMask16, int cellX, int cellY)`
  - `UpsampleLowResToHighRes(ushort lowRes)`
  - `DownsampleHighResToLowRes(ulong highRes)`
  - `ReadHoleMasks(ReadOnlySpan<byte> mcnkHeaderOrPayload, uint mcnkFlags)`

### 2.2 Consumers Across Repository
- Verify all consumers use `TerrainHoleMath` consistently:
  - `TerrainTileMeshBuilder.cs`: uses `TerrainHoleMath.IsCellHoled64` / `IsCellHoled16` to skip cell index generation.
  - `TerrainMeshBuilder.cs`: uses `TerrainHoleMath.IsCellHoled64` / `IsCellHoled16` for legacy or un-batched chunk mesh generation.
  - `MapGlbExporter.cs`: skips holed cells during 3D GLB export.
  - `TerrainChunkMath.cs`: skips holed cells during normal vector generation and workbench index building.
  - `TerrainHeightmapIo.cs`: skips holed cells during heightmap OBJ/mesh generation.
  - `GroundEffectPlacementModels.cs`: queries `TerrainHoleMath.IsCellHoled64` to prevent flora/grass from spawning in hole cells.
  - `WorldTerrainHoleMask.cs`: runtime representation with 64-bit and 16-bit views.
  - `AlphaTerrainAdapter.cs`: upsamples Alpha 0.5.3 16-bit hole masks to 64-bit for universal downstream handling.

### 2.3 Unit & Real-Data Tests (`WowViewer.Core.Tests/Maps/TerrainHoleMathTests.cs`)
- Add unit tests for:
  - Real CASC chunk hole masks from WoW Forever 1.60.1 (Deathknell [5, 8] and [5, 9]).
  - Boundary continuity checks.
  - Model alignment checks (Church WMO and Open Grave M2s).
  - Round-trip fidelity between 16-bit legacy upsampling and downsampling.

---

## 3. Governance & Quality Gates

- **Gate 1**: `dotnet build` passes with 0 errors across the solution.
- **Gate 2**: `dotnet test` passes with 100% success on all `TerrainHoleMathTests`.
- **Gate 3**: Compliance with `AGENTS.md` §9 (receipt in `evidence/receipt-spec261.md`), §10 (no new members in `WorldScene` or `ViewerApp`), and §4 (working format readers preserved).
