# Alpha 0.5.3 Coordinate Systems

**Source of truth**: `WoWConstants.cs`, `AlphaTerrainAdapter.cs`, `StandardTerrainAdapter.cs`

---

## WoW World Coordinate System

WoW uses a **left-handed** coordinate system:

```
X = North (positive) → South (negative)
Y = West (positive) → East (negative)
Z = Up (positive) → Down (negative)
```

The world is a 64×64 tile grid. Tile (0,0) is at the **northwest** corner.

### Map Origin

```
MapOrigin = 17066.66666
```

This is the world coordinate of tile (0,0)'s corner. Coordinates **decrease** as tile indices increase.

### Tile to World Conversion

```
wowX = MapOrigin - tileY * ChunkSize     // tileY maps to X (north-south)
wowY = MapOrigin - tileX * ChunkSize     // tileX maps to Y (east-west)
```

**CRITICAL**: `tileX` maps to `wowY` and `tileY` maps to `wowX`. This is because:
- The tile grid X axis runs east-west (which is WoW Y)
- The tile grid Y axis runs north-south (which is WoW X)

### Chunk Within Tile

Each tile has 16×16 chunks. Chunk position:

```
chunkSmall = ChunkSize / 16 = 33.33333
wowX_chunk = MapOrigin - tileY * ChunkSize - chunkX * chunkSmall
wowY_chunk = MapOrigin - tileX * ChunkSize - chunkY * chunkSmall
```

**ALSO SWAPPED**: `chunkX` affects `wowX` (via tileY axis) and `chunkY` affects `wowY` (via tileX axis).

---

## Renderer Coordinate System

Our viewer uses a **right-handed** coordinate system with Y-up conventions mapped as:

```
rendererX = MapOrigin - wowY    (east-west axis, screen horizontal)
rendererY = MapOrigin - wowX    (north-south axis, screen vertical)
rendererZ = wowZ                (height, up)
```

This is equivalent to:
```
rendererX = tileX * ChunkSize + chunkY * chunkSmall    (relative to origin)
rendererY = tileY * ChunkSize + chunkX * chunkSmall    (relative to origin)
```

### Why the Swap?

WoW's coordinate system has X=north and Y=west. Our renderer has X going right (east) and Y going up (north on the screen). So:
- **Renderer X** = east direction = WoW's -Y direction = `MapOrigin - wowY`
- **Renderer Y** = north direction = WoW's X direction... but inverted since WoW X decreases southward = `MapOrigin - wowX`

This swap introduces a **handedness change**. The terrain works fine because we just remap vertices. But for models (WMO/MDX), the handedness flip causes a **mirror** that must be corrected with `Scale(1, -1, 1)` applied to model geometry.

---

## File Format Position Layouts

### Common Pattern: (X, Z, Y) in File

All placement entries (MDDF, MODF) store positions as:

```
Byte 0x08: posX    (WoW X = north-south)
Byte 0x0C: posZ    (WoW Z = height/up)
Byte 0x10: posY    (WoW Y = east-west)
```

The **middle** float is always **height**. This is consistent across Alpha and LK formats.

### Converting File Position to Renderer

```csharp
rendererX = MapOrigin - posY;    // file posY → renderer X
rendererY = MapOrigin - posX;    // file posX → renderer Y
rendererZ = posZ;                // file posZ → renderer Z (height)
```

### WMO-Only Maps (No Terrain)

For WMO-only maps, positions and bounding boxes are used **directly** without MapOrigin subtraction. The WMO model geometry is in its own local space.

---

## WDT MAIN Chunk Indexing

### Alpha WDT (0.5.3)

**Column-major** ordering:
```
tileIndex = tileX * 64 + tileY
```

To recover tile coordinates:
```
tileX = tileIndex / 64
tileY = tileIndex % 64
```

### Standard WDT (LK 3.3.5+)

**Row-major** ordering:
```
tileIndex = tileY * 64 + tileX
```

To recover tile coordinates:
```
tileX = tileIndex % 64
tileY = tileIndex / 64
```

**IMPORTANT**: Alpha and LK use **different** MAIN indexing!

---

## VLM Dataset Tile Naming

The VLM exporter writes filenames as `MapName_{fileX}_{fileY}` where:
```
fileX = tileIndex % 64 = WoW tileY    (for Alpha column-major)
fileY = tileIndex / 64 = WoW tileX    (for Alpha column-major)
```

So when loading VLM files:
```
tileX = fileY    (second number in filename)
tileY = fileX    (first number in filename)
```

---

## Scale Factors

### MDDF Scale (Doodad)
```
scale_uint16 in file
actual_scale = scale_uint16 / 1024.0
```
1024 = 1.0× scale.

### MODF Scale
WMO placements in Alpha don't have an explicit scale field (unlike LK). Scale is assumed 1.0.

---

## Rotation

All rotations in placement entries are in **degrees** (not radians).

```
file: rotX, rotZ, rotY  (degrees, same XZY order as position)
```

Convert to radians for rendering:
```
radX = rotX * π / 180
radY = rotY * π / 180
radZ = rotZ * π / 180
```

---

## Summary Table

| Concept | WoW | Renderer | File (MDDF/MODF) |
|---------|-----|----------|-------------------|
| North-South | X (+ = north) | Y (+ = north) | posX (byte 0x08) |
| East-West | Y (+ = west) | X (+ = east) | posY (byte 0x10) |
| Height | Z (+ = up) | Z (+ = up) | posZ (byte 0x0C) |
| Tile grid col | tileX (east-west) | - | - |
| Tile grid row | tileY (north-south) | - | - |
