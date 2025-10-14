# Mesh Extractor Rewrite Plan

## Problem Statement

Current `AdtMeshExtractor.cs` produces invalid OBJ files:
- ❌ Wrong vertex order (should be z,x,y not x,y,z)
- ❌ Missing proper UV bounds calculation
- ❌ Incorrect face winding (should be 4 triangles per quad)
- ❌ Missing hole detection
- ❌ Single-threaded (too slow for large maps)
- ❌ Face format includes normals we don't calculate properly

## Working Reference

**Source:** `old_projects/WDT-ADT-to-OBJ-GLB/ADTPreFabTool.Console/Program.cs` (lines 1060-1228)

**Output Example:** `Kalimdor_39_27.obj` - 37,120 vertices, proper UVs, correct faces

## Key Differences (Working vs Current)

| Feature | Working Code | Current Code | Fix Needed |
|---------|--------------|--------------|------------|
| Vertex order | `v z x y` | `v x y z` | ✅ Swap to z,x,y |
| UV calculation | Bounds-based | Tile-size modulo | ✅ Use actual bounds |
| Face format | `f v/vt` | `f v/vt/vn` | ✅ Remove normals |
| Triangles per quad | 4 | 4 (but wrong indices) | ✅ Fix indices |
| Hole detection | ✅ Full | ❌ Incomplete | ✅ Add proper logic |
| Threading | ❌ Single | ❌ Single | ✅ Add Parallel.ForEach |

## Correct Implementation Details

### 1. Vertex Extraction (Lines 1109-1126)

```csharp
for (int row = 0; row < 17; row++)
{
    bool isShort = (row % 2) == 1;
    int colCount = isShort ? 8 : 9;
    
    for (int col = 0; col < colCount; col++)
    {
        float vx = chunk.header.position.Y - (col * UNIT_SIZE);
        if (isShort) vx -= UNIT_SIZE_HALF;
        float vy = chunk.vertices.vertices[idx] + chunk.header.position.Z;
        float vz = chunk.header.position.X - (row * UNIT_SIZE_HALF);
        
        positions.Add((vx, vy, vz));
        
        // Track bounds for UV calculation
        if (vx < minX) minX = vx;
        if (vx > maxX) maxX = vx;
        if (vz < minZ) minZ = vz;
        if (vz > maxZ) maxZ = vz;
        
        idx++;
    }
}
```

### 2. UV Calculation (Lines 1132-1147)

```csharp
float spanX = Math.Max(1e-6f, maxX - minX);
float spanZ = Math.Max(1e-6f, maxZ - minZ);
const float eps = 2.5e-3f;

for (int i = 0; i < positions.Count; i++)
{
    var p = positions[i];
    // Normalize to tile extents
    float u = (p.x - minX) / spanX;
    float v = (maxZ - p.z) / spanZ;
    
    // Optional flips for different orientations
    if (yFlip) v = 1f - v;
    if (xFlip) u = 1f - u;
    
    // Clamp to avoid edge artifacts
    u = Math.Clamp(u, eps, 1f - eps);
    v = Math.Clamp(v, eps, 1f - eps);
    
    uvs.Add((u, v));
}
```

### 3. OBJ Writing (Lines 1150-1224)

```csharp
// Write vertices in z,x,y order
for (int i = 0; i < positions.Count; i++)
{
    var p = positions[i];
    writer.WriteLine($"v {p.z:F6} {p.x:F6} {p.y:F6}");
}

// Write UVs
for (int i = 0; i < uvs.Count; i++)
{
    var t = uvs[i];
    writer.WriteLine($"vt {t.u} {t.v}");
}

// Write faces with hole detection
for (int j = 9, xx = 0, yy = 0; j < 145; j++, xx++)
{
    if (xx >= 8) { xx = 0; yy++; }
    
    bool isHole = CheckHole(chunk, xx, yy);
    
    if (!isHole)
    {
        int baseIndex = chunkStartIndex[chunkIndex];
        int i0 = j;
        int a = baseIndex + i0 + 1;      // +1 for OBJ 1-based indexing
        int b = baseIndex + (i0 - 9) + 1;
        int c = baseIndex + (i0 + 8) + 1;
        int d = baseIndex + (i0 - 8) + 1;
        int e = baseIndex + (i0 + 9) + 1;
        
        // 4 triangles per quad
        writer.WriteLine($"f {a}/{a} {b}/{b} {c}/{c}");
        writer.WriteLine($"f {a}/{a} {d}/{d} {b}/{b}");
        writer.WriteLine($"f {a}/{a} {e}/{e} {d}/{d}");
        writer.WriteLine($"f {a}/{a} {c}/{c} {e}/{e}");
    }
    
    if (((j + 1) % (9 + 8)) == 0) j += 9;
}
```

### 4. Hole Detection (Lines 1180-1200)

```csharp
bool isHole = true;

if (((uint)chunk.header.flags & 0x10000u) == 0)
{
    // Low-res holes (4x4 grid, 16 bits)
    int current = 1 << ((xx / 2) + (yy / 2) * 4);
    if ((chunk.header.holesLowRes & current) == 0)
        isHole = false;
}
else
{
    // High-res holes (8x8 grid, 8 bytes)
    byte holeByte = yy switch
    {
        0 => chunk.header.holesHighRes_0,
        1 => chunk.header.holesHighRes_1,
        2 => chunk.header.holesHighRes_2,
        3 => chunk.header.holesHighRes_3,
        4 => chunk.header.holesHighRes_4,
        5 => chunk.header.holesHighRes_5,
        6 => chunk.header.holesHighRes_6,
        _ => chunk.header.holesHighRes_7,
    };
    if (((holeByte >> xx) & 1) == 0)
        isHole = false;
}
```

## Multi-Threading Strategy

```csharp
var options = new ParallelOptions
{
    MaxDegreeOfParallelism = Environment.ProcessorCount
};

var manifestEntries = new ConcurrentBag<MeshManifestEntry>();

Parallel.ForEach(adtTiles, options, tile =>
{
    try
    {
        var result = ExtractSingleTile(source, mapName, tile.x, tile.y, meshDir);
        if (result != null)
            manifestEntries.Add(result);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Error] Tile [{tile.x},{tile.y}]: {ex.Message}");
    }
});
```

## Implementation Steps

1. ✅ **Analyze working code** - Documented above
2. ⏳ **Fix vertex order** - Change to z,x,y in ExportOBJ
3. ⏳ **Fix UV calculation** - Use bounds-based approach
4. ⏳ **Fix face format** - Remove normals, use v/vt only
5. ⏳ **Fix face indices** - Use correct 4-triangle pattern
6. ⏳ **Add hole detection** - Implement full logic
7. ⏳ **Add multi-threading** - Parallel.ForEach for tiles
8. ⏳ **Remove normal calculation** - Not needed for v/vt format
9. ⏳ **Test with sample tiles** - Compare output to working version

## Testing Checklist

- [ ] Vertex count matches (37,120 for Kalimdor_39_27)
- [ ] UV range is 0-1 (not outside bounds)
- [ ] Faces reference valid vertex indices
- [ ] OBJ loads in Blender without errors
- [ ] Minimap texture maps correctly
- [ ] Holes appear in correct locations
- [ ] Multi-threading doesn't cause corruption
- [ ] Performance: 100+ tiles in reasonable time

## Expected Performance

**Single-threaded:** ~2-5 seconds per tile
**Multi-threaded (24 cores):** ~0.1-0.2 seconds per tile effective

**For 256 tiles (full map):**
- Single: ~8-20 minutes
- Multi: ~30-60 seconds ✅

## References

- Working code: `old_projects/WDT-ADT-to-OBJ-GLB/ADTPreFabTool.Console/Program.cs`
- Working output: `old_projects/WDT-ADT-to-OBJ-GLB/project_output/Kalimdor-20250829-113026/minimap_obj/Kalimdor/`
- wow.export: `lib/wow.export/src/js/3D/exporters/ADTExporter.js` (reference for advanced features)
