# Alpha 0.5.3 Terrain Format Specification

**Source of truth**: `MdxViewer/Terrain/AlphaTerrainAdapter.cs`, `gillijimproject-csharp/WowFiles/Alpha/`
**Verified by**: Ghidra reverse engineering of WoWClient.exe 0.5.3.3368 (with PDB)

---

## WDT (World Data Table)

Alpha WDTs are **monolithic** files containing all embedded ADT data. They are typically **≥64KB** (distinguishing them from LK WDTs which are ~33KB).

### Detection
```
if (fileSize >= 65536) → Alpha WDT
if (fileSize < 65536)  → Standard/LK WDT
```

### Structure

The WDT contains these top-level chunks (FourCC stored in **forward** order, not reversed like LK):

| Chunk | Description |
|-------|-------------|
| MVER  | Version (always 18) |
| MPHD  | Map header flags |
| MAIN  | 64×64 tile existence grid with embedded ADT offsets |
| MDNM  | MDX model name table (null-separated strings) |
| MONM  | WMO model name table (null-separated strings) |
| MODF  | WMO placements (WDT-level, for WMO-only maps) |

### MAIN Chunk — Tile Grid

The MAIN chunk is a 64×64 grid. **Column-major** ordering:

```
tileIndex = tileX * 64 + tileY
```

Each entry contains an **offset** into the WDT file where the embedded ADT data begins. Offset = 0 means no tile.

### WMO-Only Maps

If `IsWmoBased` is true (no terrain tiles exist), the WDT contains WMO placements directly in its MODF chunk. The WMO is positioned in absolute WoW world coordinates (no MapOrigin subtraction needed for WMO-only maps).

---

## ADT (Embedded in WDT)

Each ADT is embedded at the offset given by the MAIN chunk. Parsed via `AdtAlpha`.

### ADT Top-Level Chunks

| Chunk | Description |
|-------|-------------|
| MCIN  | MCNK offset table (256 entries, one per terrain chunk) |
| MTEX  | Texture name table (null-separated BLP paths) |
| MDDF  | MDX doodad placement entries |
| MODF  | WMO placement entries |

### Tile Coordinate System

Each tile is identified by `(tileX, tileY)` derived from:
```
tileX = tileIndex / 64    (column, north-south axis in WoW)
tileY = tileIndex % 64    (row, east-west axis in WoW)
```

**World position** of a tile corner:
```
wowX = MapOrigin - tileY * ChunkSize     (east-west, decreasing eastward)
wowY = MapOrigin - tileX * ChunkSize     (north-south, decreasing northward)
```

Where:
- `MapOrigin = 17066.66666`
- `ChunkSize = 533.33333` (one tile = one "chunk" in WDT terms = 16×16 terrain chunks)

---

## MCNK (Terrain Chunk)

Each ADT contains 256 MCNK chunks arranged in a 16×16 grid. Each MCNK represents a ~33.33 unit square of terrain.

### MCNK Header (Alpha Format)

The Alpha MCNK header layout differs from LK. Key fields:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00   | 4    | IndexX | Chunk X index within tile (0-15) |
| 0x04   | 4    | IndexY | Chunk Y index within tile (0-15) |
| 0x08   | 4    | NLayers | Number of texture layers (1-4) |
| 0x0C   | 4    | Holes | 16-bit hole mask |

Subchunk offsets are relative to the MCNK data start.

### World Position Calculation

For a chunk at `(chunkX, chunkY)` within tile `(tileX, tileY)`:
```
chunkSmall = ChunkSize / 16    (= 33.33333)
worldX = MapOrigin - tileX * ChunkSize - chunkY * chunkSmall
worldY = MapOrigin - tileY * ChunkSize - chunkX * chunkSmall
```

**CRITICAL**: Note the **swap** — `chunkY` affects `worldX` and `chunkX` affects `worldY`. This is because WoW's coordinate system has X=north and Y=west, but the chunk grid is indexed differently.

---

## MCVT (Height Map)

### Alpha Format (Non-Interleaved)

Alpha MCVT stores heights in **non-interleaved** format:

```
[81 outer vertex heights][64 inner vertex heights]
Total: 145 × 4 bytes = 580 bytes
```

- **Outer vertices**: 9×9 grid = 81 floats (corners and edges of cells)
- **Inner vertices**: 8×8 grid = 64 floats (centers of cells)

### Reinterleaving to Standard Layout

The standard interleaved layout alternates outer and inner rows:
```
Row 0: 9 outer vertices (outerRow 0)
Row 1: 8 inner vertices (innerRow 0)
Row 2: 9 outer vertices (outerRow 1)
Row 3: 8 inner vertices (innerRow 1)
...
Row 16: 9 outer vertices (outerRow 8)
Total: 9×9 + 8×8 = 81 + 64 = 145
```

**Reinterleave algorithm**:
```
destIdx = 0
for row = 0..16:
    if row is even:
        outerRow = row / 2
        for col = 0..8:
            srcIdx = outerRow * 9 + col
            dest[destIdx++] = outer[srcIdx]
    else:
        innerRow = row / 2
        for col = 0..7:
            srcIdx = innerRow * 8 + col
            dest[destIdx++] = inner[81 + srcIdx]    (offset by 81 outer)
```

### LK Format (Already Interleaved)

LK MCVT stores heights in already-interleaved format. No reordering needed.

---

## MCNR (Normals)

### Alpha Format (Non-Interleaved)

Same non-interleaved layout as MCVT:

```
[81 outer normals × 3 bytes][64 inner normals × 3 bytes]
Total: 145 × 3 = 435 bytes
```

Each normal is 3 **signed bytes**: `(X, Z, Y)` in WoW convention.

### Decoding a Normal
```
nx = (sbyte)data[offset + 0] / 127.0     → X component
nz = (sbyte)data[offset + 1] / 127.0     → Z component (up in WoW)
ny = (sbyte)data[offset + 2] / 127.0     → Y component
normalize(nx, ny, nz)
```

### Reinterleaving

Uses the **exact same** reinterleave algorithm as MCVT, but with stride 3 (bytes per normal) instead of 4 (bytes per float).

---

## MCLY (Texture Layers)

Up to 4 texture layers per chunk. Each entry is 16 bytes:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00   | 4    | TextureIndex | Index into MTEX texture name table |
| 0x04   | 4    | Flags | Layer flags |
| 0x08   | 4    | AlphaOffset | Offset into MCAL data for this layer's alpha |
| 0x0C   | 4    | EffectId | Ground effect ID |

### Flags
| Bit | Name | Description |
|-----|------|-------------|
| 0x100 | UseAlpha | Layer has alpha map data |
| 0x200 | CompressedAlpha | Alpha is RLE compressed (rare in Alpha) |

---

## MCAL (Alpha Maps)

Layer 0 is always fully opaque (no alpha map). Layers 1-3 have alpha maps.

### Alpha 0.5.3: 4-bit Packed (2048 bytes per layer)

**Ghidra-verified** via `CMapChunk::UnpackAlphaBits` @ `0x0069a621`.

Default format. **Row-major** pixel ordering (confirmed by Ghidra — single linear loop 0→4095).

Each byte encodes 2 pixels:
```
low_nibble  = byte & 0x0F   → pixel at position j*2    (even index)
high_nibble = byte >> 4      → pixel at position j*2+1  (odd index)
```

The client shifts nibbles into ARGB format: `(nibble << 28) | 0x00FFFFFF` for 4-bit alpha in high byte.

Expand to 8-bit for GPU: `value_8bit = nibble * 17` (maps 0x0→0, 0xF→255)

Total: 2048 bytes → 64×64 = 4096 pixel values.

**Mip levels**: `CWorld::alphaMipLevel` controls resolution:
- Level 0: 64×64 (2048 bytes packed) — standard
- Level 1: 32×32 (reads from 33-wide source with +0x21 stride)

### 8-bit Uncompressed (4096 bytes per layer)

If flag `CompressedAlpha (0x200)` is set in MCLY, alpha data is 4096 bytes of direct 8-bit values.

### RLE Compressed

Rare in Alpha. Header byte: bit 7 = fill flag, bits 0-6 = count.
- Fill: next byte repeated `count` times
- Copy: next `count` bytes copied literally

---

## MCSH (Shadow Map)

64×64 bit shadow map stored as 512 bytes (64 rows × 8 bytes/row).

### Bit Layout
```
for y = 0..63:
    for x = 0..63:
        byteIndex = y * 8 + (x / 8)
        bitIndex  = x % 8
        isShadowed = (data[byteIndex] >> bitIndex) & 1
```

### Polarity
- Bit = 1 → **shadowed** (output 255 for GPU upload)
- Bit = 0 → **lit** (output 0)

---

## MDDF (MDX/Doodad Placements)

Entry size: **36 bytes**

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00   | 4    | NameIndex | Index into MDNM name table |
| 0x04   | 4    | UniqueId | Unique ID for deduplication across tiles |
| 0x08   | 4    | PosX | WoW X position (north-south) |
| 0x0C   | 4    | PosZ | WoW Z position (height/up) |
| 0x10   | 4    | PosY | WoW Y position (east-west) |
| 0x14   | 4    | RotX | Rotation X (degrees) |
| 0x18   | 4    | RotZ | Rotation Z (degrees) |
| 0x1C   | 4    | RotY | Rotation Y (degrees) |
| 0x20   | 2    | Scale | Scale factor (1024 = 1.0) |
| 0x22   | 2    | Flags | Placement flags |

### Position Layout in File

**CRITICAL**: The file stores positions as `(X, Z, Y)` — the **middle** component is height.
```
file bytes 0x08-0x14: posX, posZ, posY   (Z is height, in the middle)
```

### Coordinate Conversion (WoW → Renderer)
```
rendererX = MapOrigin - wowY    (posY from file)
rendererY = MapOrigin - wowX    (posX from file)
rendererZ = wowZ                (posZ from file, height)
```

### Scale
```
actualScale = scale / 1024.0
```

---

## MODF (WMO Placements)

Entry size: **64 bytes**

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00   | 4    | NameIndex | Index into MONM name table |
| 0x04   | 4    | UniqueId | Unique ID for deduplication |
| 0x08   | 4    | PosX | WoW X position |
| 0x0C   | 4    | PosZ | WoW Z position (height) |
| 0x10   | 4    | PosY | WoW Y position |
| 0x14   | 4    | RotX | Rotation X (degrees) |
| 0x18   | 4    | RotZ | Rotation Z |
| 0x1C   | 4    | RotY | Rotation Y |
| 0x20   | 4    | BoundsMinX | AABB min X |
| 0x24   | 4    | BoundsMinZ | AABB min Z (height) |
| 0x28   | 4    | BoundsMinY | AABB min Y |
| 0x2C   | 4    | BoundsMaxX | AABB max X |
| 0x30   | 4    | BoundsMaxZ | AABB max Z (height) |
| 0x34   | 4    | BoundsMaxY | AABB max Y |
| 0x38   | 2    | Flags | Placement flags |
| 0x3A   | 2    | DoodadSet | Doodad set index |
| 0x3C   | 2    | NameSet | Name set index |
| 0x3E   | 2    | Padding | Unused |

### Position Layout

Same as MDDF: file stores `(X, Z, Y)` with Z as height in the middle.

### WMO-Only Maps

For WMO-only maps (no terrain tiles), positions and bounding boxes are used **directly** as absolute coordinates. No MapOrigin subtraction.

### Terrain Maps

Same coordinate conversion as MDDF:
```
rendererX = MapOrigin - posY
rendererY = MapOrigin - posX
rendererZ = posZ
```

Bounding boxes must be converted and min/max swapped (MapOrigin-min > MapOrigin-max):
```
rBBMinX = MapOrigin - bbMaxY
rBBMaxX = MapOrigin - bbMinY
rBBMinY = MapOrigin - bbMaxX
rBBMaxY = MapOrigin - bbMinX
rBBMinZ = bbMinZ
rBBMaxZ = bbMaxZ
```

---

## Hole Mask

16-bit mask stored in MCNK header at offset 0x0C. Each bit controls visibility of a 2×2 cell quad.

The 16 bits map to a 4×4 grid of "holes" covering the 8×8 cell grid:
```
Bit 0  → cells (0,0)-(1,1)
Bit 1  → cells (2,0)-(3,1)
Bit 2  → cells (4,0)-(5,1)
Bit 3  → cells (6,0)-(7,1)
Bit 4  → cells (0,2)-(1,3)
...
Bit 15 → cells (6,6)-(7,7)
```

If a bit is set, the corresponding 2×2 cell area is a "hole" (invisible/passable).

---

## Key Constants (Ghidra-Verified)

```
MapOrigin     = 17066.66666    // World coord of tile (0,0) corner
ChunkSize     = 533.33333      // World units per tile (16 chunks)
CellSize      = 66.66667       // World units per cell (ChunkSize / 8)
ChunkOffset   = 266.66667      // Half a chunk (ChunkSize / 2)
ChunkScale    = 0.001875       // 1 / ChunkSize

TilesPerMapEdge    = 64        // 64×64 tile grid
ChunksPerTileEdge  = 16        // 16×16 chunks per tile
ChunksPerTile      = 256       // 16 × 16
VerticesPerChunk   = 145       // 81 outer + 64 inner
TrianglesPerChunk  = 256       // 8×8 cells × 4 triangles each
```
