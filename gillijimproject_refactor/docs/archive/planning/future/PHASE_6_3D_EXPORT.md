# Phase 6: 3D Export & Visualization 🎨

**Goal**: Consolidate ADTPrefabTool 3D export functionality into WoWRollback with modern formats (GLB, glTF).

---

## 📋 Overview

ADTPrefabTool.poc contains valuable 3D export and terrain analysis functionality that should be integrated into the unified WoWRollback tool.

### Current Capabilities (ADTPrefabTool)
- ✅ OBJ export for ADT terrain meshes
- ✅ Pattern mining (prefab detection)
- ✅ Multi-tile processing
- ✅ Local-space export (normalized coordinates)
- ✅ Terrain feature analysis (height, slope, curvature)

### Target Capabilities (WoWRollback)
- ✅ All above features
- ✨ **GLB/glTF export** (modern 3D formats)
- ✨ **Texture embedding** (embedded in GLB)
- ✨ **LOD generation** (multiple detail levels)
- ✨ **WMO/M2 export** (doodads and objects)
- ✨ **Multi-threaded export** (parallel tile processing)

---

## 🏗️ Architecture

### New Namespace Structure
```
WoWRollback.Core/
├── Formats/
│   ├── Alpha/         ← Already planned
│   ├── Lk/            ← Already planned
│   └── Dbc/           ← Already planned
├── Export/            ← NEW: 3D export functionality
│   ├── Mesh/
│   │   ├── TerrainMeshBuilder.cs       // Build terrain mesh from MCNK
│   │   ├── DoodadMeshBuilder.cs        // M2 model placement
│   │   ├── WmoMeshBuilder.cs           // WMO object placement
│   │   └── MeshOptimizer.cs            // Simplification, LOD
│   ├── Formats/
│   │   ├── ObjExporter.cs              // Legacy OBJ export
│   │   ├── GlbExporter.cs              // GLB/glTF 2.0 export
│   │   └── ColladaExporter.cs          // DAE export (optional)
│   ├── Textures/
│   │   ├── TextureExtractor.cs         // Extract BLP → PNG/JPG
│   │   ├── TextureAtlasBuilder.cs      // Combine textures
│   │   └── MaterialBuilder.cs          // PBR material setup
│   └── Analysis/
│       ├── TerrainFeatures.cs          // Height, slope, curvature
│       ├── PrefabMiner.cs              // Pattern detection
│       └── ChunkTokenizer.cs           // Chunk feature hashing
└── Processing/
    └── TileExporter.cs                 ← NEW: Multi-threaded tile export
```

---

## 📦 Migration from ADTPrefabTool

### Files to Migrate

#### 1. Core Terrain Export
**Source**: `ADTPrefabTool.poc/src/AlphaWDTReader/`

```
Migrate:
├── Readers/AlphaTerrainDecoder.cs     → TerrainMeshBuilder.cs
├── ReferencePort/McnkHeader.cs        → (merge into existing McnkReader)
└── ReferencePort/McnkReaders.cs       → (merge into existing McnkReader)

Keep Pattern (refactor):
└── Program.cs (prefab mining logic)   → PrefabMiner.cs
```

#### 2. Feature Analysis
**Source**: ADTPrefabTool `ComputeChunkFeatures()`

```csharp
// Migrate to WoWRollback.Core/Export/Analysis/TerrainFeatures.cs

public class TerrainFeatures
{
    public static ChunkFeatures Analyze(float[,] heightMap)
    {
        return new ChunkFeatures
        {
            HeightRange = ComputeHeightRange(heightMap),
            AvgSlope = ComputeAverageSlope(heightMap),
            CurvatureMean = ComputeCurvature(heightMap),
            GradientHistogram = ComputeGradientHistogram(heightMap)
        };
    }
    
    // Migrate existing algorithms from ADTPrefabTool
}

public record ChunkFeatures
{
    public float HeightRange { get; init; }
    public float AvgSlope { get; init; }
    public float CurvatureMean { get; init; }
    public float[] GradientHistogram { get; init; }
}
```

---

## 🎯 Implementation Tasks

### Task 1: Terrain Mesh Building
**New File**: `WoWRollback.Core/Export/Mesh/TerrainMeshBuilder.cs`

```csharp
namespace WoWRollback.Core.Export.Mesh;

public class TerrainMeshBuilder
{
    public Mesh BuildTerrainMesh(AdtData adt, ExportOptions options)
    {
        var mesh = new Mesh();
        
        // For each MCNK chunk
        foreach (var chunk in adt.Chunks)
        {
            // Build vertex grid (9×9 outer + 8×8 inner = 145 vertices)
            var vertices = BuildChunkVertices(chunk, options);
            mesh.Vertices.AddRange(vertices);
            
            // Build triangles (8×8×2 = 128 triangles per chunk)
            var triangles = BuildChunkTriangles(chunk, mesh.Vertices.Count);
            mesh.Triangles.AddRange(triangles);
            
            // UVs from texture coordinates
            var uvs = BuildChunkUVs(chunk);
            mesh.UVs.AddRange(uvs);
            
            // Normals (computed or from MCNR)
            var normals = BuildChunkNormals(chunk, options);
            mesh.Normals.AddRange(normals);
        }
        
        return mesh;
    }
    
    private List<Vector3> BuildChunkVertices(McnkChunk chunk, ExportOptions options)
    {
        var vertices = new List<Vector3>();
        
        // Outer vertices (9×9 grid)
        for (int y = 0; y < 9; y++)
        {
            for (int x = 0; x < 9; x++)
            {
                var height = chunk.Heights.Outer[y * 9 + x];
                var worldPos = ChunkToWorld(chunk.X, chunk.Y, x, y, height);
                vertices.Add(options.LocalSpace ? ToLocal(worldPos) : worldPos);
            }
        }
        
        // Inner vertices (8×8 grid, offset)
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                var height = chunk.Heights.Inner[y * 8 + x];
                var worldPos = ChunkToWorld(chunk.X, chunk.Y, x + 0.5f, y + 0.5f, height);
                vertices.Add(options.LocalSpace ? ToLocal(worldPos) : worldPos);
            }
        }
        
        return vertices;
    }
}

public class Mesh
{
    public List<Vector3> Vertices { get; } = new();
    public List<int> Triangles { get; } = new();
    public List<Vector2> UVs { get; } = new();
    public List<Vector3> Normals { get; } = new();
    public List<Vector4> Colors { get; } = new(); // Vertex colors
}

public record ExportOptions
{
    public bool LocalSpace { get; init; } = false;
    public bool IncludeHoles { get; init; } = false;
    public bool IncludeLiquids { get; init; } = false;
    public bool IncludeDoodads { get; init; } = false;
    public bool IncludeWMOs { get; init; } = false;
    public int LodLevel { get; init; } = 0;
}
```

---

### Task 2: GLB/glTF Export
**New File**: `WoWRollback.Core/Export/Formats/GlbExporter.cs`

**NuGet Package**: `SharpGLTF.Toolkit` (excellent C# glTF library)

```csharp
using SharpGLTF.Scenes;
using SharpGLTF.Schema2;

namespace WoWRollback.Core.Export.Formats;

public class GlbExporter
{
    public async Task ExportAsync(
        string outputPath, 
        Mesh mesh, 
        Material material,
        CancellationToken ct = default)
    {
        var scene = new SceneBuilder();
        
        // Create material with embedded textures
        var gltfMaterial = CreateMaterial(material);
        
        // Create mesh node
        var meshBuilder = scene.AddRigidMesh(mesh.Vertices, mesh.Triangles, gltfMaterial);
        
        // Add UVs
        meshBuilder.SetVertexUVs(0, mesh.UVs);
        
        // Add normals
        meshBuilder.SetVertexNormals(mesh.Normals);
        
        // Add vertex colors (for shadow maps, etc.)
        if (mesh.Colors.Count > 0)
            meshBuilder.SetVertexColors(0, mesh.Colors);
        
        // Build and save as GLB (binary glTF)
        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);
    }
    
    private MaterialBuilder CreateMaterial(Material material)
    {
        var builder = new MaterialBuilder("Terrain")
            .WithMetallicRoughnessShader()
            .WithBaseColor(material.BaseColorTexture)
            .WithMetallicRoughness(material.MetallicRoughnessTexture)
            .WithNormal(material.NormalTexture);
        
        return builder;
    }
}
```

---

### Task 3: Multi-Threaded Tile Export
**New File**: `WoWRollback.Core/Processing/TileExporter.cs`

```csharp
namespace WoWRollback.Core.Processing;

public class TileExporter
{
    private readonly TerrainMeshBuilder _meshBuilder;
    private readonly GlbExporter _glbExporter;
    private readonly TextureExtractor _textureExtractor;
    
    public async Task<ExportResult> ExportTilesAsync(
        string wdtPath,
        string outputDir,
        ExportOptions options,
        CancellationToken ct = default)
    {
        var wdtInfo = WdtAlphaReader.ReadWdt(wdtPath);
        var results = new ConcurrentBag<TileExportResult>();
        var progress = 0;
        
        var parallelOptions = new ParallelOptions 
        { 
            MaxDegreeOfParallelism = options.MaxThreads,
            CancellationToken = ct
        };
        
        await Parallel.ForEachAsync(wdtInfo.AdtTiles, parallelOptions, 
            async (adtNum, token) =>
        {
            try
            {
                // Read ADT
                var adt = AdtAlphaReader.ReadAdt(wdtPath, adtNum, wdtInfo.AdtOffsets[adtNum]);
                
                // Build mesh
                var mesh = _meshBuilder.BuildTerrainMesh(adt, options);
                
                // Extract textures
                var textures = await _textureExtractor.ExtractTexturesAsync(adt, token);
                
                // Build material
                var material = BuildMaterial(adt, textures);
                
                // Export as GLB
                var outputPath = Path.Combine(outputDir, 
                    $"{wdtInfo.MapName}_{adtNum / 64}_{adtNum % 64}.glb");
                await _glbExporter.ExportAsync(outputPath, mesh, material, token);
                
                results.Add(new TileExportResult 
                { 
                    AdtNum = adtNum, 
                    Success = true,
                    OutputPath = outputPath,
                    VertexCount = mesh.Vertices.Count,
                    TriangleCount = mesh.Triangles.Count / 3
                });
                
                // Progress
                var current = Interlocked.Increment(ref progress);
                if (current % 10 == 0)
                {
                    _logger.LogInformation("Exported {Current}/{Total} tiles", 
                        current, wdtInfo.AdtTiles.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to export tile {AdtNum}", adtNum);
                results.Add(new TileExportResult { AdtNum = adtNum, Success = false });
            }
        });
        
        return new ExportResult
        {
            MapName = wdtInfo.MapName,
            TotalTiles = wdtInfo.AdtTiles.Count,
            SuccessfulTiles = results.Count(r => r.Success),
            Results = results.ToList()
        };
    }
}
```

---

### Task 4: Prefab Pattern Mining
**New File**: `WoWRollback.Core/Export/Analysis/PrefabMiner.cs`

Migrate the excellent pattern detection logic from ADTPrefabTool:

```csharp
public class PrefabMiner
{
    public async Task<PrefabMiningResult> MinePatternsAsync(
        string wdtPath,
        PrefabMiningOptions options,
        CancellationToken ct = default)
    {
        // 1. Tokenize chunks (k-means clustering on terrain features)
        var tokens = await TokenizeChunksAsync(wdtPath, options.CodebookSize, ct);
        
        // 2. Build neighborhood index (2x2, 3x2, 3x3, etc.)
        var neighborhoods = BuildNeighborhoodIndex(tokens, options.PatternSizes);
        
        // 3. Mine frequent patterns
        var patterns = FindFrequentPatterns(neighborhoods, options.MinFrequency);
        
        // 4. Region growing (expand patterns across tiles)
        var grown = GrowRegions(patterns, tokens);
        
        // 5. Non-maximum suppression (remove duplicates)
        var final = ApplyNMS(grown, options.NmsThreshold);
        
        return new PrefabMiningResult
        {
            Patterns = final,
            Stats = new PrefabStats
            {
                TotalChunks = tokens.Count,
                UniquePatterns = final.Count,
                AveragePatternSize = final.Average(p => p.ChunkCount)
            }
        };
    }
}
```

---

## 🎨 CLI Commands

### Export 3D Tiles
```powershell
# Export all tiles as GLB files (multi-threaded)
wowrollback export-3d path/to/Azeroth.wdt \
  --output exports/Azeroth \
  --format glb \
  --threads 8 \
  --include-textures \
  --include-doodads

# Export specific tiles
wowrollback export-3d path/to/Azeroth.wdt \
  --output exports/Azeroth \
  --tiles 29_40,30_40,31_40 \
  --format glb

# Export with LOD levels
wowrollback export-3d path/to/Azeroth.wdt \
  --output exports/Azeroth \
  --format glb \
  --lod 0,1,2 \
  --lod-distance 100,500,1000
```

### Mine Prefab Patterns
```powershell
# Find recurring terrain patterns
wowrollback mine-prefabs path/to/Azeroth.wdt \
  --output prefabs/Azeroth \
  --pattern-sizes 2x2,3x2,3x3,4x2,4x3 \
  --codebook-size 512 \
  --min-frequency 5 \
  --export-top 50

# Export pattern instances as GLB
wowrollback mine-prefabs path/to/Azeroth.wdt \
  --output prefabs/Azeroth \
  --pattern-sizes 3x3 \
  --export-format glb \
  --local-space
```

### Batch Export All Maps
```powershell
# Export all maps during conversion
wowrollback compare-versions \
  --alpha-root ../test_data \
  --versions 0.5.3.3368 \
  --maps Azeroth,Kalimdor \
  --threads 8 \
  --export-3d \
  --export-format glb
```

---

## 📊 Output Formats

### GLB/glTF (Primary)
**Benefits**:
- ✅ Industry standard (Unity, Unreal, Blender, web viewers)
- ✅ Embedded textures (single file)
- ✅ PBR materials (physically-based rendering)
- ✅ Efficient binary format
- ✅ LOD support

**Structure**:
```
Azeroth_29_40.glb
├── Scene
│   └── Terrain Mesh
│       ├── Vertices (4,352 for 16×16 chunks)
│       ├── Normals
│       ├── UVs (texture coordinates)
│       └── Material
│           ├── BaseColor (terrain textures)
│           ├── Normal Map
│           └── Metallic/Roughness
└── Embedded Textures
    ├── tileset_01.png
    ├── tileset_02.png
    └── ...
```

### OBJ (Legacy, for compatibility)
```
Azeroth_29_40.obj
Azeroth_29_40.mtl
textures/
├── tileset_01.png
└── ...
```

---

## 🔧 Dependencies

### New NuGet Packages
```xml
<PackageReference Include="SharpGLTF.Toolkit" Version="1.0.0-alpha0031" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.1.0" />
<PackageReference Include="System.Numerics.Vectors" Version="4.5.0" />
```

---

## 📅 Implementation Timeline

### Week 1: Mesh Building
- [ ] Migrate terrain decoder from ADTPrefabTool
- [ ] Implement TerrainMeshBuilder
- [ ] Add OBJ export (simple test)
- [ ] Unit tests for mesh generation

### Week 2: GLB Export
- [ ] Add SharpGLTF.Toolkit
- [ ] Implement GlbExporter
- [ ] Texture extraction and embedding
- [ ] Material setup (PBR)

### Week 3: Multi-Threading
- [ ] Implement TileExporter (parallel)
- [ ] Progress reporting
- [ ] Error handling
- [ ] Performance testing

### Week 4: Prefab Mining
- [ ] Migrate pattern detection
- [ ] Implement PrefabMiner
- [ ] CLI integration
- [ ] Export pattern instances

### Week 5: Polish
- [ ] LOD generation
- [ ] WMO/M2 export (optional)
- [ ] Documentation
- [ ] Example outputs

---

## ✅ Success Criteria

- [ ] Export Azeroth (128 tiles) as GLB files in <10 minutes
- [ ] GLB files load correctly in Blender/Unity
- [ ] Textures embedded and display properly
- [ ] Prefab mining finds 50+ recurring patterns
- [ ] Pattern exports load as expected
- [ ] Multi-threaded export uses 60-80% CPU
- [ ] Documentation with example outputs

---

## 🎯 Future Enhancements

### Phase 7: Advanced Features (Optional)
- [ ] **WMO/M2 Export**: Full doodad and object support
- [ ] **Collision Mesh**: Separate collision geometry
- [ ] **LOD Auto-Generation**: Mesh simplification
- [ ] **Texture Atlas**: Combine all textures into one
- [ ] **glTF Draco Compression**: Smaller file sizes
- [ ] **Web Viewer**: Interactive 3D viewer in browser
- [ ] **Unity/Unreal Plugins**: Direct import

---

## 📚 Integration with Existing Phases

### Update Phase 2 (MapConverter)
Add optional 3D export during conversion:

```csharp
public async Task<ConversionResult> ConvertMapAsync(
    string wdtPath,
    string outputDir,
    ConversionOptions options,
    CancellationToken ct = default)
{
    // ... existing conversion ...
    
    // Optional: Export as 3D
    if (options.Export3D)
    {
        await _tileExporter.ExportTilesAsync(wdtPath, 
            Path.Combine(outputDir, "3d"), 
            options.ExportOptions, 
            ct);
    }
    
    return result;
}
```

### Update CLI (CompareVersionsCommand)
```powershell
wowrollback compare-versions \
  --alpha-root ../test_data \
  --versions 0.5.3.3368 \
  --maps Azeroth \
  --threads 8 \
  --viewer-report \
  --export-3d              # ← NEW FLAG
  --export-format glb      # ← NEW FLAG
  --export-prefabs         # ← NEW FLAG
```

---

## 🎉 Why This Matters

### For Data Scientists
- ✅ Analyze terrain patterns across entire continents
- ✅ Export for ML training (terrain classification)
- ✅ Quantitative landscape analysis

### For 3D Artists
- ✅ Import into Blender/Unity/Unreal
- ✅ Extract prefabs for custom maps
- ✅ Texture analysis and extraction

### For Researchers
- ✅ Procedural generation studies
- ✅ Pattern recognition in game design
- ✅ Historical preservation (Alpha → modern formats)

### For the Project
- ✅ Comprehensive tool (all use cases covered)
- ✅ Modern formats (future-proof)
- ✅ Excellent documentation (examples, tutorials)

---

**This phase makes WoWRollback the definitive tool for WoW terrain analysis!** 🚀
