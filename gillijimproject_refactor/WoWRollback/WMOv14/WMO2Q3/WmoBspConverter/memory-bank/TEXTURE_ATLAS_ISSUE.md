# WMO Texture Atlas Mapping Issue

## Problem Statement

The castle01.wmo exports geometry correctly, but **textures are not mapped to the correct surfaces**. For example, parts that should be wood are showing the wrong texture.

## Root Cause

WMO v14 uses **texture atlases** where multiple material regions are packed into single texture files. The MOMT (material) chunk contains information about which region of the texture atlas each material uses, but we're currently only extracting the texture filename, not the UV offset/scale data.

### Current Implementation (Incomplete)

```csharp
// WmoV14Parser.cs line 74-105
private List<uint> ParseMomtTextureIndices(byte[] momtData, List<string> textures, ...)
{
    const int ENTRY_SIZE = 64;  // ✅ Correct size
    var list = new List<uint>();
    for (int off = 0; off + ENTRY_SIZE <= momtData.Length; off += ENTRY_SIZE)
    {
        // ❌ Only extracting texture index, not UV transform data
        uint idx0 = BitConverter.ToUInt32(momtData, off + 0);
        // ...
        list.Add(idx0);
    }
    return list;
}
```

### What's Missing

Each 64-byte MOMT entry likely contains:
1. **Texture index** (which texture file) - ✅ We extract this
2. **UV offset** (X,Y offset into atlas) - ❌ NOT extracted
3. **UV scale** (width,height of region) - ❌ NOT extracted
4. **Shader flags** - ❌ NOT extracted
5. **Blend mode** - ❌ NOT extracted
6. **Other material properties** - ❌ NOT extracted

## Evidence

From castle01.wmo output:
```
[OBJ] Group 0: mat histogram = [0:184] (mapping=MOPYx2)
[OBJ] Group 1: mat histogram = [0:1154, 1:315] (mapping=MOPYx2)
```

- Group 0: All 184 triangles assigned to material 0
- Group 1: 1154 triangles to material 0, 315 to material 1

But the WMO has multiple distinct visual materials (stone, wood, metal, etc.) that should map to different regions of the texture atlas.

## MOMT Structure (64 bytes)

Based on WoW file format documentation and similar formats:

```
Offset  Size  Field
------  ----  -----
0x00    4     Flags (shader type, blending, etc.)
0x04    4     Shader ID
0x08    4     Blend mode
0x0C    4     Texture1 offset (MOTX string offset)
0x10    4     Color1 (emissive?)
0x14    4     Flags2
0x18    4     Texture2 offset (MOTX string offset)
0x1C    4     Color2 (diffuse?)
0x20    4     Texture3 offset (MOTX string offset)
0x24    4     Color3 (?)
0x28    4     Flags3
0x2C    4     Runtime data...
...     ...   (more fields to 64 bytes)
```

**Key insight**: The texture offset at 0x0C points to a string in MOTX, but there may be additional UV transform data elsewhere in the structure.

## Comparison with MDX Format

Looking at the MDX viewer code (`modelreader.ts` lines 165-206):

```typescript
private parseMaterials(model: Model, reader: Reader): void {
    const count = reader.int32();
    for (let i = 0; i < count; i++) {
        const material: Material = { Layers: [] } as Material;
        reader.int32(); // material size inclusive
        material.PriorityPlane = reader.int32();
        const layersCount = reader.int32();
        
        for (let i = 0; i < layersCount; i++) {
            const layer: Layer = {} as Layer;
            layer.FilterMode = reader.int32();
            layer.Shading = reader.int32();
            layer.TextureID = reader.int32();
            layer.TVertexAnimId = reader.int32();
            layer.CoordId = reader.int32();  // ← UV coordinate set!
            layer.Alpha = reader.float();
            // ...
        }
    }
}
```

MDX materials have:
- **Multiple layers** per material
- **CoordId** - which UV coordinate set to use
- **TVertexAnimId** - animated UV transforms
- **FilterMode** and **Shading** - rendering properties

## Solution Approach

### Phase 1: Parse Complete MOMT Structure

1. Create `WmoMaterial` class with all 64 bytes of data:
   ```csharp
   public class WmoMaterial
   {
       public uint Flags { get; set; }
       public uint ShaderType { get; set; }
       public uint BlendMode { get; set; }
       public uint Texture1Offset { get; set; }
       public uint Texture2Offset { get; set; }
       public uint Texture3Offset { get; set; }
       public Vector4 Color1 { get; set; }
       public Vector4 Color2 { get; set; }
       // ... more fields
   }
   ```

2. Parse all fields from MOMT chunk
3. Store in `WmoV14Data.Materials` list

### Phase 2: Understand UV Coordinate Mapping

The MOTV chunk contains UV coordinates, but we need to understand:
- Are there multiple UV sets per vertex?
- Do materials reference specific UV coordinate sets?
- Are UV transforms applied per-material or per-batch?

### Phase 3: Apply UV Transforms in Export

1. When writing OBJ faces, apply material-specific UV transforms:
   ```csharp
   // Pseudo-code
   var material = materials[faceMatId];
   var uv = baseUV * material.UVScale + material.UVOffset;
   ```

2. Or generate separate texture files per material region (extract from atlas)

### Phase 4: Test and Validate

1. Export castle01.wmo with corrected UV mapping
2. Verify wood appears on wooden parts
3. Verify stone appears on stone parts
4. Check in Blender/3D viewer

## References

- **WMO Format Docs**: wowdev.wiki WMO page
- **MDX Viewer**: `wow-mdx-viewer/src/formats/mdx/modelreader.ts`
- **Current Parser**: `WmoV14Parser.cs` lines 74-105
- **Current Exporter**: `WmoObjExporter.cs` lines 100-249

## Next Steps

1. ✅ Document the issue (this file)
2. ⏳ Research MOMT structure from wowdev.wiki or hex dumps
3. ⏳ Implement full MOMT parsing
4. ⏳ Wire UV transforms into OBJ export
5. ⏳ Test with castle01.wmo
