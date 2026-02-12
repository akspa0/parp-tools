# Ghidra Analysis: WoW Alpha 0.5.3 MDX/WMO File Reading

## Executive Summary

This document summarizes findings from analyzing the original WoW Alpha 0.5.3 client (`wowclient.exe`) using Ghidra to understand how MDX and WMO files are properly read. The goal is to identify why ~50% of MDX files fail to load and why geosets with different textures don't render correctly.

## Key Findings

### 1. Hardcoded Development Path

The original WoW client contains a hardcoded development path:
```
"I@\\Guldan\\Drive2\\Projects\\WoW\\Data\\"
```

**Location:** `wowclient.exe` @ `0x0080350a`

**Implication:** This is a debug/development build path that was never changed for release. The client prepends this path to relative texture paths when flag `0x4000` is set.

### 2. Texture Loading Process

The client uses the following chain for texture loading:

```
MdxReadTextures (0x0044e310)
  → ProcessTextures (0x0044c2e0)
    → LoadModelTexture (0x00447f50)
      → TextureCreate (0x00470b60)
        → TextureAllocGxTex
```

**Key Code Path:** `LoadModelTexture` @ `0x00447f50`

```c
if ((((param_2 & 0x4000) != 0) && (param_1[1] != ':') && (*param_1 != '\\')) {
    SStrCopy(path,"\\\\Guldan\\Drive2\\Projects\\WoW\\Data\\",0x104);
    SStrPack(path,param_1,0x104);
    pHVar1 = TextureCreate(path,(CGxTexFlags)param_4,(CStatus *)param_3,0);
    return pHVar1;
}
pHVar1 = TextureCreate(param_1,(CGxTexFlags)param_4,(CStatus *)param_3,0);
```

**Critical Finding:** When texture path is relative (doesn't start with `:` or `\`) AND flag `0x4000` is set, the client prepends the hardcoded development path.

### 3. Material/Geoset Loading Process

```
MdxReadMaterials (0x0044e550)
  → LoadMaterialData (0x0044e6b0)
    → LoadLayerData (0x0044e900)
```

**Material Layer Processing:**
- `LoadLayerData` processes texture flags and sets blend modes
- It calls `GetTextureShader` to determine shader settings
- Layer structure includes: BlendMode, Flags, TextureId, TransformId, CoordId, StaticAlpha

### 4. Geoset Data Loading

```
MdxReadGeosets (0x0044eba0)
  → LoadGeosetData (0x0044eec0)
    → LoadGeosetPrimitiveData
    → LoadGeosetTransformGroups
```

**Geoset Structure:**
- VERTX: Vertices (C3Vector)
- NRMS: Normals (C3Vector)
- PTYP: Primitive types
- PCNT: Primitive counts
- PVTX: Primitive vertices (indices)
- GNDX: Group indices
- MTGC: Matrix group counts
- MATS: Matrix indices
- UVAS: UV set count
- UVBS: UV coordinates

### 5. MDX File Format (from MDX-L_Tool)

**Chunk Structure:**
```
MDLX (magic)
├── VERS (version)
├── MODL (model info)
├── SEQS (sequences/animations)
├── GLBS (global sequences)
├── MTLS (materials)
├── TEXS (textures)
├── GEOS (geosets)
├── BONE (bones)
├── HELP (helpers)
├── PIVT (pivot points)
├── ATCH (attachments)
├── LITE (lights)
├── PREM/PRE2 (particle emitters)
├── RIBB (ribbon emitters)
├── EVTS (events)
├── CAMS (cameras)
├── CLID (collision)
├── HTST (hit test shapes)
├── TXAN (texture animations)
└── CORN (PopcornFX emitters)
```

**Texture Path Reading:** [`MdxFile.cs:255`](../MDX-L_Tool/Formats/Mdx/MdxFile.cs:255)
```csharp
tex.Path = ReadFixedString(br, 0x104);  // 260-byte fixed string
```

**Alpha 0.5.3 Special Case:** [`MdxFile.cs:345-360`](../MDX-L_Tool/Formats/Mdx/MdxFile.cs:345-360)
```csharp
case "UVAS":
    if (mdxVersion == 1300) // Alpha 0.5.3 Optimization
    {
        // Alpha 0.5.3 Optimization:
        // If Version is 1300, UVAS block seemingly contains data directly
        // If Count is 1 (which means 1 UV set).
        // The data length corresponds to the number of vertices.
        // We must read this data to maintain alignment.
        int nVerts = geo.Vertices.Count;
        if (nVerts > 0)
        {
            for (int k = 0; k < nVerts; k++)
                geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
        }
    }
    // If not version 1300, it's a standard container and we continue to inner chunks (UVBS)
    break;
```

**Critical Finding:** Alpha 0.5.3 (version 1300) has a special UVAS chunk format where UV data is stored directly instead of being a container for UVBS chunks.

### 6. Smart Seek for Alignment/Padding Recovery

**Location:** [`MdxFile.cs:391-424`](../MDX-L_Tool/Formats/Mdx/MdxFile.cs:391-424)

```csharp
// Smart Seek for Alignment/Padding Recovery
// Alpha 0.5.3 often puts padding bytes between chunks (e.g. 8 bytes after UVAS data).
// Instead of aborting, we scan forward a short distance to find the next valid tag.
long currentPos = br.BaseStream.Position - 8; // Start of unknown tag
long limit = Math.Min(currentPos + 64, geoEnd);

// Start scan from 1 byte ahead
br.BaseStream.Position = currentPos + 1; 
bool recovered = false;

while (br.BaseStream.Position < limit - 4)
{
    long p = br.BaseStream.Position;
    byte[] tagBytes = br.ReadBytes(4);
    br.BaseStream.Position = p; // Rewind
    
    string possibleTag = Encoding.ASCII.GetString(tagBytes);
    if (IsValidGeosetTag(possibleTag))
    {
        Console.WriteLine($"      [RECOVERY] Skipped {p - currentPos} bytes. Resuming at valid tag '{possibleTag}' (Pos: {p}).");
        br.BaseStream.Position = p; // Align to new tag
        recovered = true;
        break;
    }
    br.BaseStream.Position = p + 1;
}

if (!recovered)
{
    Console.WriteLine($"      [WARN] Unknown GEOS sub-tag: {tag} at {currentPos}. Scan failed.");
    // Restore position to continue blindly or let loop finish
    br.BaseStream.Position = currentPos + 8;
}
```

**Critical Finding:** The MDX-L_Tool has a sophisticated recovery mechanism to handle padding bytes between chunks by scanning forward to find the next valid tag.

## Potential Issues in Current Implementation

### Issue 1: Texture Path Resolution

**Current Code:** [`Rendering/ModelRenderer.cs:360-450`](Rendering/ModelRenderer.cs:360-450)

The current code tries multiple fallback paths:
1. Original path from MDX
2. Normalized path (forward slash to backslash)
3. Model directory + filename
4. Local BLP file
5. Local PNG file

**Potential Problems:**
1. **MDX-L_Tool might not be correctly parsing texture paths** - The 260-byte fixed string might not be read correctly
2. **Replaceable texture resolution might not be working** - The DBC lookup might not be finding the correct entries
3. **Hardcoded path issue** - If the MDX file has relative texture paths with flag `0x4000`, the original client would prepend the hardcoded development path, but our viewer doesn't handle this
4. **Alpha 0.5.3 UVAS special case** - The MDX-L_Tool has special handling for version 1300, but there might be edge cases not covered

### Issue 2: Material/Geoset Mapping

**Current Code:** [`Rendering/ModelRenderer.cs:43-64`](Rendering/ModelRenderer.cs:43-64)

The code logs material→texture mapping, but there might be issues with:
1. **Material ID references being incorrect** - The material ID in the geoset might not match the actual material array index
2. **Layer texture IDs being out of bounds** - The texture ID in a layer might be >= the texture count
3. **Geoset material references being invalid** - The geoset might reference a material that doesn't exist

### Issue 3: Replaceable Texture Resolution

**Current Code:** [`Rendering/ReplaceableTextureResolver.cs`](Rendering/ReplaceableTextureResolver.cs)

The resolver uses DBC files to resolve replaceable textures, but:
1. **Only resolves texture paths** - It doesn't handle other referenced files
2. **DBC lookup might not be finding the correct entries** - The model path normalization might not match the DBC entries
3. **Build version inference might be incorrect** - The path-based inference might not work for all cases

### Issue 4: Geoset Visibility

**Current Code:** [`Rendering/ModelRenderer.cs:100-157`](Rendering/ModelRenderer.cs:100-157)

The code has visibility toggles for geosets, but:
1. **No validation of geoset count** - If the geoset count is 0, the visibility toggles won't work
2. **No validation of material count** - If the material count is 0, the rendering might fail

## Recommendations

### 1. Fix Camera Controls

**Status:** ✅ Already implemented in [`ViewerApp.cs:117-130`](ViewerApp.cs:117-130)

**Changes Made:**
- Inverted yaw: `_camera.Yaw -= dx * 0.5f;` (was `+=`)
- Inverted pitch: `_camera.Pitch += dy * 0.5f;` (was `-=`)
- Added debug logging to track mouse movement

**Expected Result:** Dragging left should now look left, dragging up should now look up.

### 2. Enhance Texture Loading Diagnostics

**Status:** ✅ Already implemented in [`Rendering/ModelRenderer.cs:360-450`](Rendering/ModelRenderer.cs:360-450)

**Changes Made:**
- Added comprehensive texture loading summary
- Added per-texture load source tracking (MPQ, local BLP, PNG)
- Added replaceable texture resolution tracking
- Added error logging for BLP decoding and PNG loading failures

**Expected Result:** Console output will show detailed information about why textures fail to load.

### 3. Improve MDX File Format Parsing

**Potential Issues:**
1. **Texture path parsing** - The 260-byte fixed string might not be read correctly for all MDX files
2. **Alpha 0.5.3 UVAS special case** - There might be edge cases not covered
3. **Padding recovery** - The smart seek mechanism might not work correctly for all MDX files

**Recommendations:**
1. Add validation of texture path length and content
2. Add more robust handling of the Alpha 0.5.3 UVAS special case
3. Improve the padding recovery mechanism to handle more edge cases

### 4. Improve Replaceable Texture Resolution

**Potential Issues:**
1. **DBC lookup failures** - The model path normalization might not match the DBC entries
2. **Build version inference** - The path-based inference might not work for all cases
3. **Missing DBC files** - The DBC files might not exist in the MPQ

**Recommendations:**
1. Add more robust model path normalization
2. Add fallback mechanisms for DBC lookup
3. Add validation of DBC file existence before attempting to load
4. Add logging of DBC lookup attempts and failures

### 5. Improve Material/Geoset Validation

**Potential Issues:**
1. **Material ID out of bounds** - The material ID in the geoset might not match the actual material array index
2. **Layer texture ID out of bounds** - The texture ID in a layer might be >= the texture count
3. **Geoset material reference invalid** - The geoset might reference a material that doesn't exist

**Recommendations:**
1. Add validation of material ID before accessing the material array
2. Add validation of layer texture ID before accessing the texture array
3. Add validation of geoset material reference before rendering
4. Add error logging when validation fails

## Next Steps

1. **Test the application** with the enhanced debug logging to identify which MDX files fail to load and why
2. **Analyze the console output** to determine the root cause of the failures
3. **Fix the identified issues** based on the analysis
4. **Verify the fixes** by testing with the same MDX files that previously failed

## Appendix: MDX Chunk Identifiers

From [`MdxHeaders.cs`](../MDX-L_Tool/Formats/Mdx/MdxHeaders.cs):

| Chunk | Description |
|--------|-------------|
| VERS | Version |
| MODL | Model info |
| SEQS | Sequences/animations |
| GLBS | Global sequences |
| MTLS | Materials |
| TEXS | Textures |
| GEOS | Geosets |
| BONE | Bones |
| HELP | Helpers |
| PIVT | Pivot points |
| ATCH | Attachments |
| LITE | Lights |
| PREM/PRE2 | Particle emitters |
| RIBB | Ribbon emitters |
| EVTS | Events |
| CAMS | Cameras |
| CLID | Collision |
| HTST | Hit test shapes |
| TXAN | Texture animations |
| CORN | PopcornFX emitters |

| Sub-chunk | Description |
|-----------|-------------|
| VRTX | Vertices |
| NRMS | Normals |
| PTYP | Primitive types |
| PCNT | Primitive counts |
| PVTX | Primitive vertices (indices) |
| GNDX | Group indices |
| MTGC | Matrix group counts |
| MATS | Matrix indices |
| UVAS | UV set count |
| UVBS | UV coordinates |
| BIDX | Bone indices (skinning) |
| BWGT | Bone weights (skinning) |
| ATSQ | Geoset animation tracks (alpha/color) - WoW Alpha 0.5.3 specific |

### 7. ATSQ Chunk - Geoset Animation Tracks (Alpha 0.5.3)

**Location:** Found in WoW Alpha 0.5.3 MDX files as a sub-chunk within GEOS (geosets).

**Ghidra Analysis:**
- Function: `IReadBinGeosetAnim` @ `0x007aa390`
- Function: `ReadBinGeosetAnim` @ `0x007aa1f0`
- Function: `AnimAddGeoset` @ `0x00755db0`

**Chunk Structure:**

```
ATSQ (Geoset Animation Section)
├── Geoset ID (uint32) - Which geoset this animation applies to
├── Default Alpha (float32) - Default opacity value (0.0 to 1.0)
├── Default Color (3x float32) - Default RGB color values
├── Unknown (uint32) - Possibly flags or padding
├── KGAO (Keyframe Group Alpha Opacity) - Alpha animation keys
│   ├── Keyframe Count (uint32)
│   ├── Interpolation Type (uint32) - 0=Linear, 1=Hermite, 2=Bezier, 3=Bezier
│   ├── Global Sequence ID (uint32) - 0xFFFFFFFF if not used
│   └── Keyframes (variable)
│       ├── Time (int32)
│       ├── Value (float32) - Alpha value
│       └── [Optional] Tangent In/Out (2x float32) - For Hermite/Bezier
└── KGAC (Keyframe Group Alpha Color) - Color animation keys
    ├── Keyframe Count (uint32)
    ├── Interpolation Type (uint32)
    ├── Global Sequence ID (uint32)
    └── Keyframes (variable)
        ├── Time (int32)
        ├── Value (3x float32) - RGB color
        └── [Optional] Tangent In/Out (6x float32) - For Hermite/Bezier
```

**Key Findings:**

1. **Alpha Animation:** Controls geoset visibility/opacity over time
2. **Color Animation:** Controls geoset color tint over time
3. **Interpolation Types:**
   - Type 0: Linear interpolation
   - Type 1: Hermite interpolation (with tangents)
   - Type 2: Bezier interpolation
   - Type 3: Bezier interpolation (variant)

4. **Global Sequences:** Can reference global sequences for synchronized animations

5. **Default Values:** If no keyframes are present, the default alpha/color values are used

**Implications for MDX Viewer:**
- The ATSQ chunk is specific to WoW Alpha 0.5.3 and may not exist in later versions
- Geoset animations need to be evaluated at runtime based on current animation time
- Alpha animations can be used for fade-in/fade-out effects
- Color animations can be used for tinting or color cycling effects

### 8. Terrain and Liquid File Reading

The Alpha 0.5.3 client handles world data loading through several core classes:

#### `CMapArea` (Map Area Loading)
**Constructor:** 0x006aa880
**Chunk Count:** 256 (16x16)
**File Metadata:** Stores `SMChunkInfo` (offset, size, asyncId) for each chunk.

#### `CMapChunk` (Map Chunk Loading)
**Constructor:** 0x00698510
**Data Chunks:**
- `VRTX`: 145 vertices (9x9)
- `NRMS`: 145 normals
- `PLNS`: 256 planes
- `MCLY`: Texture layers (up to 4)
- `MCSH`: Shadow maps
- `MCAL`: Alpha maps
- `MCLQ`: Liquid data

#### `CChunkLiquid` (Liquid Loading)
**Entry Point:** `CChunkLiquid::CChunkLiquid` @ 0x00696950
**Types:** Water (0), Ocean (1), Magma (2), Slime (3).

---

## References to Extended Analysis

- **[01_Terrain_System.md](World_Rendering_Analysis/01_Terrain_System.md)**: Deep dive into the terrain loading hierarchy.
- **[05_Liquid_Rendering.md](World_Rendering_Analysis/05_Liquid_Rendering.md)**: Details on liquid type flags and particle setup.
- **[World_Rendering_and_Mesh_Traversal.md](WoW_Alpha_0.5.3_World_Rendering_and_Mesh_Traversal.md)**: High-level overview of world rendering logic.

---

## Next Steps
...
