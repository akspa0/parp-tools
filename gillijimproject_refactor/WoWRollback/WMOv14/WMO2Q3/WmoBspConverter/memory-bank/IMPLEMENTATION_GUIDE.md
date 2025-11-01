# WMO v14 → Q3 Converter Implementation Guide

## Critical Discovery: Data Extraction Bug

### Root Cause
The current `WmoV14Parser` is NOT reading the **MOVI chunk** (vertex indices). Instead, it falls back to sequential triangle assembly, which produces incorrect geometry.

### What Works
- ✅ Coordinate transformation: WMO(X,Y,Z) → Q3(X,Z,-Y)
- ✅ Scaling: 100x uniform with Z inverted
- ✅ Brush generation: 5-plane thin slab brushes
- ✅ Plane winding: Correct CCW normals pointing outward
- ✅ Texture extraction and shader generation

### What's Broken
- ❌ MOVI chunk not being read
- ❌ Fallback uses sequential triples instead of actual indices
- ❌ MOPY metadata not properly associated with triangles

## Implementation Fix

### Step 1: Understand the Chunk Structure

**MOVI Chunk** (Vertex Indices):
- Format: Array of `ushort` values
- 2 bytes per index
- 3 indices per triangle
- Total size = (number of triangles × 3) × 2 bytes

**MOPY Chunk** (Face Metadata):
- Format: Array of `MOPY` structs (2 bytes each)
- Byte 0: Flags (render, collision, detail, etc.)
- Byte 1: Material ID
- 1 entry per triangle

**MOVT Chunk** (Vertex Positions):
- Format: Array of Vector3 (12 bytes each)
- 3 floats (X, Y, Z) per vertex
- 4 bytes per float

### Step 2: Update WmoV14Parser

Reference implementation: `FullV14Converter.AnalyzeAndBuildGroup()` (lines 165-294)

Key changes:
1. Read MOVI chunk as array of ushorts
2. Read MOPY chunk as array of structs (flags + material ID)
3. Build triangles using MOVI indices instead of sequential assembly
4. Associate MOPY metadata with each triangle

### Step 3: Triangle Assembly Logic

```csharp
// Pseudocode from POC
if (subChunks.TryGetValue("MOVI", out var moviData))
{
    var indices = MOVIParser.Parse(moviData);  // Returns List<ushort>
    mesh.Indices = indices;
    // Build triangles from indices
    for (int i = 0; i < indices.Count; i += 3)
    {
        int i0 = indices[i];
        int i1 = indices[i + 1];
        int i2 = indices[i + 2];
        // Create triangle with these indices
    }
}

if (subChunks.TryGetValue("MOPY", out var mopyData))
{
    var flags = new List<byte>();
    var mats = new List<ushort>();
    MOPYParser.Parse(mopyData, flags, mats);
    // Associate with triangles
    for (int i = 0; i < triangles.Count; i++)
    {
        triangles[i].Flags = flags[i];
        triangles[i].MaterialId = mats[i];
    }
}
```

### Step 4: Coordinate Transformation (Already Correct)

```csharp
// WMO(X,Y,Z) → Q3(X,Z,-Y) with 100x scale
const float scale = 100.0f;
v0 = new Vector3(v0.X * scale, v0.Z * scale, v0.Y * scale * -1.0f);
```

### Step 5: Brush Generation (Already Correct)

5-plane thin slab brushes with:
- Bottom plane: original triangle
- Top plane: extruded triangle (1 unit up)
- 3 side planes: vertical faces connecting edges
- All planes with CCW winding, normals pointing outward

## POC Reference Files

Located in: `old_sources/src/WoWToolbox/WoWToolbox.Core.v2/Services/WMO/Legacy/`

- `FullV14Converter.cs` - Main converter logic
- `WmoGroupMesh.cs` - Mesh data structures
- Parsers:
  - `MOVIParser.cs` - Vertex indices
  - `MOPYParser.cs` - Face metadata
  - `MOVTParser.cs` - Vertex positions
  - `MONRParser.cs` - Vertex normals
  - `MOTVParser.cs` - Texture UVs
  - `MOMTParser.cs` - Material indices
  - `MOTXParser.cs` - Texture names

## Testing Strategy

1. **Data Extraction Test**:
   - Verify MOVI chunk is read correctly
   - Check triangle count matches MOVI size / 3
   - Validate all indices are within vertex bounds

2. **Geometry Test**:
   - Load `.map` in GtkRadiant
   - Verify geometry matches original WMO shape
   - Check texture mapping is correct

3. **Compilation Test**:
   - Compile `.map` with Q3Map2 via GtkRadiant
   - Verify no "bad normal" or "degenerate plane" errors
   - Load in ioquake3 and verify visual correctness

## Next Session Checklist

- [ ] Locate and review MOVIParser implementation
- [ ] Update WmoV14Parser to read MOVI chunk
- [ ] Remove fallback sequential triangle assembly
- [ ] Test with test.wmo (simple cube)
- [ ] Verify geometry renders correctly in GtkRadiant
- [ ] Compile with Q3Map2 and verify no errors
- [ ] Test with larger WMO files
