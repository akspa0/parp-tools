# Phase 6B: MDX Support & Texture Baking 🎨✨

**Goal**: Add Alpha MDX model support and texture baking/generation capabilities.

---

## 📋 Overview

Extend Phase 6 with:
1. **Alpha MDX support** (fallback for assets not in M2 format)
2. **Texture baking** (generate terrain textures from alpha layers)
3. **FileDataID tracking** (all assets used in maps)
4. **High-resolution minimaps** (like wow.export)

### Learning from TypeScript Projects

#### wow-mdx-viewer
**Path**: `lib/wow-mdx-viewer/src/`

**Key Files**:
- `formats/reader.ts` - MDX file format reading
- `managers/filemanager.ts` - Asset management
- `managers/itexturemanager.ts` - Texture handling

**What to Learn**:
- MDX chunk structure (Alpha model format)
- Vertex/bone/animation data layout
- Texture coordinate mapping
- Legacy format quirks

#### wow.export
**Path**: `lib/wow.export/src/js/`

**Key Files**:
- `3D/exporters/ADTExporter.js` - ADT terrain export
- `3D/renderers/WMORenderer.js` - WMO rendering with textures
- `3D/renderers/M2Renderer.js` - M2 model rendering
- `components/map-viewer.js` - Map tile rendering
- Uses **Three.js** for 3D rendering

**What to Learn**:
- Texture baking pipeline (alpha layers → final texture)
- High-res minimap generation
- Material/shader setup
- Texture atlas packing

---

## 🏗️ Extended Architecture

### New Namespace: Alpha Models
```
WoWRollback.Core/Formats/Alpha/
├── Models/                      ← NEW: MDX support
│   ├── MdxReader.cs             // Read Alpha MDX files
│   ├── MdxConverter.cs          // Convert MDX → M2 (if possible)
│   └── MdxMeshBuilder.cs        // Build mesh from MDX
└── (existing WDT/ADT readers)
```

### New Namespace: Texture Processing
```
WoWRollback.Core/Export/
├── Textures/
│   ├── AlphaLayerProcessor.cs   ← NEW: Process MCAL alpha layers
│   ├── TextureBaker.cs          ← NEW: Bake final textures
│   ├── MinimapGenerator.cs      ← NEW: High-res minimaps
│   └── FileDataTracker.cs       ← NEW: Track all asset references
```

---

## 📦 Phase 6B Tasks

### Task 1: MDX Format Support
**New File**: `WoWRollback.Core/Formats/Alpha/Models/MdxReader.cs`

#### MDX Format Overview (Alpha 0.5.3)
```
MDX File Structure:
├── VERS (version)
├── MODL (model info)
├── SEQS (sequences/animations)
├── MTLS (materials)
├── TEXS (texture names)
├── GEOS (geometry - vertices, normals, UVs)
├── BONE (bone definitions)
└── HELP (helper objects)
```

#### Implementation
```csharp
namespace WoWRollback.Core.Formats.Alpha.Models;

public class MdxReader
{
    public static MdxModel ReadMdx(string mdxPath)
    {
        using var fs = File.OpenRead(mdxPath);
        using var br = new BinaryReader(fs);
        
        var model = new MdxModel();
        
        while (fs.Position < fs.Length)
        {
            var chunkId = new string(br.ReadChars(4));
            var chunkSize = br.ReadUInt32();
            
            switch (chunkId)
            {
                case "VERS":
                    model.Version = br.ReadUInt32();
                    break;
                    
                case "MODL":
                    model.Name = ReadString(br, 80);
                    model.BoundingBox = ReadBoundingBox(br);
                    break;
                    
                case "GEOS":
                    model.Geometry = ReadGeometry(br, chunkSize);
                    break;
                    
                case "TEXS":
                    model.Textures = ReadTextures(br, chunkSize);
                    break;
                    
                // ... other chunks
                    
                default:
                    fs.Seek(chunkSize, SeekOrigin.Current);
                    break;
            }
        }
        
        return model;
    }
    
    private static MdxGeometry ReadGeometry(BinaryReader br, uint size)
    {
        var geo = new MdxGeometry();
        
        // Read vertices
        var vertexCount = br.ReadUInt32();
        geo.Vertices = new Vector3[vertexCount];
        for (int i = 0; i < vertexCount; i++)
        {
            geo.Vertices[i] = new Vector3(
                br.ReadSingle(),
                br.ReadSingle(),
                br.ReadSingle()
            );
        }
        
        // Read normals
        var normalCount = br.ReadUInt32();
        geo.Normals = new Vector3[normalCount];
        for (int i = 0; i < normalCount; i++)
        {
            geo.Normals[i] = new Vector3(
                br.ReadSingle(),
                br.ReadSingle(),
                br.ReadSingle()
            );
        }
        
        // Read UVs
        var uvCount = br.ReadUInt32();
        geo.UVs = new Vector2[uvCount];
        for (int i = 0; i < uvCount; i++)
        {
            geo.UVs[i] = new Vector2(
                br.ReadSingle(),
                br.ReadSingle()
            );
        }
        
        // Read triangles
        var triangleCount = br.ReadUInt32();
        geo.Triangles = new int[triangleCount * 3];
        for (int i = 0; i < triangleCount * 3; i++)
        {
            geo.Triangles[i] = br.ReadUInt16();
        }
        
        return geo;
    }
}

public record MdxModel
{
    public uint Version { get; init; }
    public string Name { get; init; }
    public BoundingBox BoundingBox { get; init; }
    public MdxGeometry Geometry { get; init; }
    public List<string> Textures { get; init; } = new();
    public List<MdxMaterial> Materials { get; init; } = new();
}

public record MdxGeometry
{
    public Vector3[] Vertices { get; init; }
    public Vector3[] Normals { get; init; }
    public Vector2[] UVs { get; init; }
    public int[] Triangles { get; init; }
}
```

---

### Task 2: Texture Baking
**New File**: `WoWRollback.Core/Export/Textures/TextureBaker.cs`

#### Alpha Layer Processing (MCAL)
```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WoWRollback.Core.Export.Textures;

public class TextureBaker
{
    private readonly TextureExtractor _extractor;
    
    public async Task<Image<Rgba32>> BakeTerrainTextureAsync(
        McnkChunk chunk,
        List<string> texturePaths,
        CancellationToken ct = default)
    {
        // 64×64 texture per chunk (can upscale to 512×512)
        var size = 512;
        var output = new Image<Rgba32>(size, size);
        
        // Load base texture (layer 0)
        var baseTexture = await LoadTextureAsync(texturePaths[0], ct);
        
        // Apply as base layer
        output.Mutate(ctx => ctx.DrawImage(baseTexture, 1.0f));
        
        // Apply alpha layers (MCAL data)
        for (int i = 1; i < Math.Min(4, texturePaths.Count); i++)
        {
            var layerTexture = await LoadTextureAsync(texturePaths[i], ct);
            var alphaMask = CreateAlphaMask(chunk.AlphaLayers[i - 1], size);
            
            // Blend using alpha mask
            output.Mutate(ctx => ctx.DrawImage(layerTexture, alphaMask, 1.0f));
        }
        
        return output;
    }
    
    private Image<Rgba32> CreateAlphaMask(byte[] alphaData, int size)
    {
        var mask = new Image<Rgba32>(size, size);
        
        // Alpha data is 64×64, upscale to desired size
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var alpha = alphaData[y * 64 + x];
                var color = new Rgba32(255, 255, 255, alpha);
                
                // Upscale by drawing a block
                int blockSize = size / 64;
                for (int by = 0; by < blockSize; by++)
                {
                    for (int bx = 0; bx < blockSize; bx++)
                    {
                        mask[x * blockSize + bx, y * blockSize + by] = color;
                    }
                }
            }
        }
        
        return mask;
    }
    
    private async Task<Image<Rgba32>> LoadTextureAsync(string path, CancellationToken ct)
    {
        // Extract BLP → PNG/RGBA
        var blpData = await _extractor.ExtractBlpAsync(path, ct);
        return Image.Load<Rgba32>(blpData);
    }
}
```

---

### Task 3: High-Resolution Minimap Generation
**New File**: `WoWRollback.Core/Export/Textures/MinimapGenerator.cs`

#### Learning from wow.export
```javascript
// wow.export: js/components/map-viewer.js
// They render tiles at high resolution, then composite into minimap

// Our C# approach:
```

```csharp
public class MinimapGenerator
{
    public async Task<Image<Rgba32>> GenerateMinimapAsync(
        string wdtPath,
        int resolution = 2048, // 2K or 4K
        CancellationToken ct = default)
    {
        var wdtInfo = WdtAlphaReader.ReadWdt(wdtPath);
        
        // Determine map bounds (which tiles exist)
        var bounds = CalculateBounds(wdtInfo.AdtTiles);
        var tileWidth = bounds.MaxX - bounds.MinX + 1;
        var tileHeight = bounds.MaxY - bounds.MinY + 1;
        
        // Resolution per tile
        var tileRes = resolution / Math.Max(tileWidth, tileHeight);
        
        // Create output image
        var output = new Image<Rgba32>(tileWidth * tileRes, tileHeight * tileRes);
        
        // Render each tile
        await Parallel.ForEachAsync(wdtInfo.AdtTiles, 
            new ParallelOptions { CancellationToken = ct },
            async (adtNum, token) =>
        {
            var tileX = adtNum % 64;
            var tileY = adtNum / 64;
            
            // Read ADT
            var adt = AdtAlphaReader.ReadAdt(wdtPath, adtNum, wdtInfo.AdtOffsets[adtNum]);
            
            // Render tile to texture
            var tileImage = await RenderTileToTextureAsync(adt, tileRes, token);
            
            // Composite into output
            var offsetX = (tileX - bounds.MinX) * tileRes;
            var offsetY = (tileY - bounds.MinY) * tileRes;
            
            output.Mutate(ctx => ctx.DrawImage(tileImage, new Point(offsetX, offsetY), 1.0f));
        });
        
        return output;
    }
    
    private async Task<Image<Rgba32>> RenderTileToTextureAsync(
        AdtData adt,
        int resolution,
        CancellationToken ct)
    {
        var tileImage = new Image<Rgba32>(resolution, resolution);
        
        // For each chunk (16×16 = 256 chunks per tile)
        var chunkRes = resolution / 16;
        
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                var chunk = adt.Chunks[cy * 16 + cx];
                
                // Bake chunk texture
                var chunkTexture = await _textureBaker.BakeTerrainTextureAsync(
                    chunk, 
                    chunk.TexturePaths,
                    ct);
                
                // Resize to chunk resolution
                chunkTexture.Mutate(ctx => ctx.Resize(chunkRes, chunkRes));
                
                // Composite into tile
                tileImage.Mutate(ctx => ctx.DrawImage(
                    chunkTexture,
                    new Point(cx * chunkRes, cy * chunkRes),
                    1.0f));
            }
        }
        
        return tileImage;
    }
}
```

---

### Task 4: FileDataID Tracking
**New File**: `WoWRollback.Core/Export/Textures/FileDataTracker.cs`

```csharp
public class FileDataTracker
{
    private readonly HashSet<AssetReference> _trackedAssets = new();
    
    public void TrackAdt(AdtData adt)
    {
        // Track textures (BLP files)
        foreach (var chunk in adt.Chunks)
        {
            foreach (var texturePath in chunk.TexturePaths)
            {
                TrackAsset(AssetType.Texture, texturePath);
            }
        }
        
        // Track doodads (M2 models)
        foreach (var doodad in adt.Doodads)
        {
            TrackAsset(AssetType.M2Model, doodad.ModelPath);
            
            // Check if M2 exists, fallback to MDX
            if (!File.Exists(doodad.ModelPath))
            {
                var mdxPath = doodad.ModelPath.Replace(".m2", ".mdx");
                if (File.Exists(mdxPath))
                {
                    TrackAsset(AssetType.MdxModel, mdxPath);
                }
            }
        }
        
        // Track WMOs (world objects)
        foreach (var wmo in adt.WorldObjects)
        {
            TrackAsset(AssetType.WMO, wmo.WmoPath);
        }
    }
    
    private void TrackAsset(AssetType type, string path)
    {
        _trackedAssets.Add(new AssetReference
        {
            Type = type,
            Path = NormalizePath(path),
            FileDataId = ResolveFileDataId(path) // From listfile or CASC
        });
    }
    
    public async Task ExportManifestAsync(string outputPath)
    {
        var manifest = new AssetManifest
        {
            GeneratedDate = DateTime.UtcNow,
            TotalAssets = _trackedAssets.Count,
            Assets = _trackedAssets
                .GroupBy(a => a.Type)
                .ToDictionary(
                    g => g.Key.ToString(),
                    g => g.Select(a => new
                    {
                        a.Path,
                        a.FileDataId,
                        Exists = File.Exists(a.Path)
                    }).ToList()
                )
        };
        
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        
        await File.WriteAllTextAsync(outputPath, json);
    }
}

public record AssetReference
{
    public AssetType Type { get; init; }
    public string Path { get; init; }
    public uint? FileDataId { get; init; }
}

public enum AssetType
{
    Texture,    // BLP files
    M2Model,    // M2 models (LK)
    MdxModel,   // MDX models (Alpha)
    WMO,        // World objects
    Sound,      // Audio files
    Music       // Music files
}
```

---

## 🎨 Enhanced Export Workflow

### Complete Tile Export with Textures
```csharp
public class TileExporter
{
    private readonly TerrainMeshBuilder _meshBuilder;
    private readonly TextureBaker _textureBaker;
    private readonly GlbExporter _glbExporter;
    private readonly FileDataTracker _assetTracker;
    
    public async Task<ExportResult> ExportTileWithTexturesAsync(
        string wdtPath,
        int adtNum,
        string outputDir,
        ExportOptions options,
        CancellationToken ct = default)
    {
        // 1. Read ADT
        var adt = AdtAlphaReader.ReadAdt(wdtPath, adtNum, offset);
        
        // 2. Track all assets
        _assetTracker.TrackAdt(adt);
        
        // 3. Build mesh
        var mesh = _meshBuilder.BuildTerrainMesh(adt, options);
        
        // 4. Bake textures for each chunk
        var bakedTextures = new List<Image<Rgba32>>();
        foreach (var chunk in adt.Chunks)
        {
            var texture = await _textureBaker.BakeTerrainTextureAsync(
                chunk,
                chunk.TexturePaths,
                ct);
            bakedTextures.Add(texture);
        }
        
        // 5. Create texture atlas (combine all chunk textures)
        var atlas = CreateTextureAtlas(bakedTextures);
        
        // 6. Build material with atlas
        var material = new Material
        {
            BaseColorTexture = atlas,
            Name = $"Terrain_{adtNum}"
        };
        
        // 7. Export as GLB with embedded texture
        var outputPath = Path.Combine(outputDir, $"tile_{adtNum}.glb");
        await _glbExporter.ExportAsync(outputPath, mesh, material, ct);
        
        return new ExportResult { Success = true, OutputPath = outputPath };
    }
    
    private Image<Rgba32> CreateTextureAtlas(List<Image<Rgba32>> textures)
    {
        // 16×16 chunks = 4×4 atlas layout (4096×4096 total)
        var atlasSize = 4096;
        var chunkSize = 256; // 256×256 per chunk
        
        var atlas = new Image<Rgba32>(atlasSize, atlasSize);
        
        for (int i = 0; i < textures.Count; i++)
        {
            var x = (i % 16) * chunkSize;
            var y = (i / 16) * chunkSize;
            
            textures[i].Mutate(ctx => ctx.Resize(chunkSize, chunkSize));
            atlas.Mutate(ctx => ctx.DrawImage(textures[i], new Point(x, y), 1.0f));
        }
        
        return atlas;
    }
}
```

---

## 🎯 CLI Commands

### Export with Texture Baking
```powershell
# Export tiles with baked textures
wowrollback export-3d path/to/Azeroth.wdt \
  --output exports/Azeroth \
  --format glb \
  --bake-textures \
  --texture-resolution 512 \
  --threads 8

# Generate high-res minimap
wowrollback generate-minimap path/to/Azeroth.wdt \
  --output Azeroth_minimap_4K.png \
  --resolution 4096

# Export asset manifest (FileDataIDs)
wowrollback export-3d path/to/Azeroth.wdt \
  --output exports/Azeroth \
  --track-assets \
  --manifest exports/Azeroth_manifest.json
```

### MDX Fallback Support
```powershell
# Prefer M2, fallback to MDX
wowrollback export-3d path/to/Azeroth.wdt \
  --output exports/Azeroth \
  --include-doodads \
  --mdx-fallback \
  --prefer-m2
```

---

## 📊 Output Formats

### Asset Manifest (JSON)
```json
{
  "generatedDate": "2025-10-04T07:20:00Z",
  "totalAssets": 1543,
  "assets": {
    "Texture": [
      {
        "path": "Tileset\\Generic\\Grass01.blp",
        "fileDataId": 123456,
        "exists": true
      },
      {
        "path": "Tileset\\Dungeon\\Stone02.blp",
        "fileDataId": 123457,
        "exists": true
      }
    ],
    "M2Model": [
      {
        "path": "World\\Generic\\Human\\Male\\HumanMale.m2",
        "fileDataId": 234567,
        "exists": true
      }
    ],
    "MdxModel": [
      {
        "path": "World\\Generic\\Barrel\\Barrel.mdx",
        "fileDataId": null,
        "exists": true
      }
    ]
  }
}
```

### GLB with Baked Textures
```
Azeroth_29_40.glb (single file, ~15-25 MB)
├── Terrain Mesh (4,352 vertices)
├── Embedded Texture Atlas (4096×4096 PNG)
│   └── Baked from 256 chunk textures (alpha layers applied)
└── Material (PBR)
    ├── BaseColor → Texture Atlas
    ├── Roughness → 0.8 (terrain default)
    └── Metallic → 0.0
```

---

## 🔧 Dependencies

### New NuGet Packages
```xml
<!-- Already have from Phase 6 -->
<PackageReference Include="SharpGLTF.Toolkit" Version="1.0.0-alpha0031" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.1.0" />

<!-- NEW for Phase 6B -->
<PackageReference Include="SixLabors.ImageSharp.Drawing" Version="2.1.0" />
```

---

## 📅 Implementation Timeline

### Week 1: MDX Support
- [ ] Study wow-mdx-viewer TypeScript code
- [ ] Implement MdxReader.cs
- [ ] Test with Alpha doodads
- [ ] Fallback logic (M2 → MDX)

### Week 2: Texture Baking
- [ ] Study wow.export texture pipeline
- [ ] Implement AlphaLayerProcessor
- [ ] Implement TextureBaker
- [ ] Test on various terrain types

### Week 3: Minimap Generation
- [ ] Implement MinimapGenerator
- [ ] Parallel tile rendering
- [ ] Test high-resolution outputs (2K, 4K)

### Week 4: FileDataID Tracking
- [ ] Implement FileDataTracker
- [ ] Asset manifest export
- [ ] Integration with existing export pipeline

### Week 5: Polish & Testing
- [ ] Complete export pipeline
- [ ] Performance optimization
- [ ] Documentation
- [ ] Example outputs

---

## ✅ Success Criteria

- [ ] Export Azeroth with baked textures (128 tiles) in <15 minutes
- [ ] GLB files include embedded 4K texture atlases
- [ ] MDX fallback works for Alpha-only assets
- [ ] Generate 4K minimap of Azeroth in <5 minutes
- [ ] Asset manifest tracks all 1500+ referenced files
- [ ] Textures display correctly in Blender/Unity

---

## 🎯 Integration with Phase 6

### Update TileExporter
```csharp
// Add texture baking to existing export
if (options.BakeTextures)
{
    var bakedTextures = await BakeTexturesAsync(adt, ct);
    material.BaseColorTexture = CreateAtlas(bakedTextures);
}
else
{
    // Use original texture references
    material.TextureReferences = adt.GetTextureReferences();
}
```

### Update CLI
```powershell
wowrollback export-3d path/to/map.wdt \
  --bake-textures              # ← NEW: Bake alpha layers
  --texture-resolution 512     # ← NEW: Per-chunk resolution
  --mdx-fallback               # ← NEW: Use MDX if M2 missing
  --track-assets               # ← NEW: Generate manifest
  --generate-minimap           # ← NEW: High-res minimap
```

---

## 🌟 Why This Matters

### Complete Asset Pipeline
- ✅ **Alpha MDX** → Handle legacy models
- ✅ **Texture Baking** → Self-contained exports (no external textures)
- ✅ **FileDataID Tracking** → Full asset inventory
- ✅ **High-Res Minimaps** → Better than Blizzard's originals!

### Data Science Applications
- ✅ Texture analysis (terrain type classification)
- ✅ Asset usage statistics
- ✅ Procedural generation training data

### 3D Asset Creation
- ✅ Export entire maps for custom projects
- ✅ Extract specific areas with baked textures
- ✅ Create high-quality minimaps for tools/addons

---

**Phase 6B makes WoWRollback the most comprehensive WoW terrain tool ever built!** 🚀
