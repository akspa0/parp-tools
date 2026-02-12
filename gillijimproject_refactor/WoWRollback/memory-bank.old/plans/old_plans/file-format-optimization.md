# File Format Optimization Plan

## Quick Reference

**Goal:** Reduce file sizes by 75% and improve viewer performance by 4-5x

**Changes:**
1. PNG → JPG for minimaps (70% smaller, faster decode)
2. OBJ → GLB for meshes (75% smaller, native browser support)

**Implementation Phases:**
0. **PHASE 0:** Fix current OBJ texture bugs (URGENT - 3 critical fixes)
1. **PHASE 1:** JPG minimap support (3 files to modify)
2. **PHASE 2:** GLB export implementation (1 file, complete rewrite of ExportGLB)
3. **PHASE 3:** Coordinate minimap + mesh export (ensure correct order)
4. **PHASE 4:** Viewer updates for GLB (optional, for 3D viewer)

**Key Files:**
- `WoWRollback.Cli/Program.cs` - Minimap extraction (line 656)
- `WoWRollback.AnalysisModule/AdtMeshExtractor.cs` - GLB export (line 288)
- `WoWRollback.Core/Services/Minimap/MinimapComposer.cs` - Minimap saving
- Viewer HTML/JS files - Tile URL patterns

**Critical Issues to Fix:**
1. Minimaps must be extracted BEFORE meshes (GLB needs JPG texture)
2. **CURRENT BUG:** OBJ MTL files reference `.png` that doesn't exist → checkerboard texture
3. Minimap files must be copied to mesh directory for both OBJ and GLB
4. **CURRENT BUG:** UV coordinates are flipped - texture appears mirrored on X and Y axes

---

## Problem Statement

### Current Bug: Missing Textures in OBJ

**Issue:** The current OBJ export creates MTL files that correctly reference `{mapName}_{x}_{y}.png`, but:
- The minimap PNG files are in `minimaps/` directory
- The OBJ/MTL files are in `{mapName}_mesh/` directory
- **The texture files are never copied to the mesh directory**
- Result: Blender/viewers can't find the textures → checkerboard "missing texture" pattern
- Makes the meshes useless for visualization

**Root Cause:** Workflow issue - minimap files are extracted to one directory, meshes to another, but files are never copied together.

**Current directory structure:**
```
output/
├── minimaps/
│   ├── Kalimdor_39_27.png  ← Texture is here
│   └── ...
└── Kalimdor_mesh/
    ├── Kalimdor_39_27.obj  ← Mesh is here
    ├── Kalimdor_39_27.mtl  ← References "Kalimdor_39_27.png" (relative path)
    └── ...                  ← But texture is NOT here!
```

**Fix Required:**
1. Change `.png` to `.jpg` in MTL (we're switching to JPG anyway)
2. **Copy minimap files to mesh directory** BEFORE or DURING mesh export
3. This makes textures available as relative paths for OBJ/MTL/GLB

### Current Bug: UV Coordinates Flipped

**Issue:** Textures appear mirrored/flipped on both X and Y axes when loaded in Blender.

**Root Cause:** Missing UV flip operations. The working implementation applies:
```csharp
if (yFlip) v = 1f - v;  // Flip V coordinate
if (xFlip) u = 1f - u;  // Flip U coordinate
```

**Current code** (AdtMeshExtractor.cs lines 237-241):
```csharp
float u = (p.x - minX) / spanX;
float v = (maxZ - p.z) / spanZ;

u = Math.Clamp(u, eps, 1f - eps);
v = Math.Clamp(v, eps, 1f - eps);
```

**Missing:** The flip operations!

**Fix Required:** Add UV flips with defaults `yFlip = true, xFlip = true` (matching working code).

---

## Original Problem Statement

**Current State:**
- OBJ files: 2-6MB each × 256 tiles = **512MB - 1.5GB per map** ❌
- PNG minimaps: Large file size, slow browser decode
- Viewer performance: Slow tile loading and rendering

**Target State:**
- GLB files: ~500KB-1MB each × 256 tiles = **128MB - 256MB per map** ✅ (75% reduction)
- JPG minimaps: 50-100KB each (vs 200-500KB PNG) ✅ (70% reduction)
- Faster viewer: Quicker decode, lower memory usage

## File Format Comparison

### Terrain Meshes

| Format | Size per Tile | Pros | Cons | Browser Support |
|--------|---------------|------|------|-----------------|
| **OBJ** | 2-6MB | Human-readable, universal | Huge, no compression, no textures | Via loader |
| **GLB** | 500KB-1MB | Binary, compressed, embedded textures | Binary format | Native (WebGL) ✅ |

**Decision: GLB** - 75% smaller, native browser support, embedded textures

### Minimap Textures

| Format | Size | Decode Speed | Quality | Browser Support |
|--------|------|--------------|---------|-----------------|
| **PNG** | 200-500KB | Slow | Lossless | Native ✅ |
| **JPG** | 50-100KB | Fast | Good enough | Native ✅ |
| **WebP** | 40-80KB | Medium | Excellent | Modern browsers ✅ |

**Decision: JPG** - 70% smaller than PNG, fastest decode, universal support

**Alternative: WebP** - Best compression, but slightly slower decode (consider as option)

## Implementation Strategy

### 1. GLB Export (Priority: HIGH)

**Current GLB code** (lines 288-324 in AdtMeshExtractor.cs):
```csharp
private void ExportGLB(
    List<(float x, float y, float z)> positions,
    List<int> indices,
    string outputPath)
{
    var scene = new SceneBuilder();
    var material = new MaterialBuilder("TerrainMaterial")
        .WithMetallicRoughnessShader()
        .WithChannelParam(KnownChannel.BaseColor, new Vector4(0.6f, 0.6f, 0.6f, 1.0f));
    
    var mesh = new MeshBuilder<VertexPosition>("TerrainMesh");
    // ... add triangles ...
}
```

**Needs:**
- ✅ Already has SharpGLTF
- ❌ Missing: UV coordinates
- ❌ Missing: Texture embedding
- ❌ Missing: Proper vertex format (VertexPositionTexture)

**Updated signature:**
```csharp
private void ExportGLB(
    List<(float x, float y, float z)> positions,
    List<(float u, float v)> uvs,
    List<int> chunkStartIndices,
    dynamic adt,
    string glbPath,
    string texturePath) // JPG minimap
```

**Key changes:**
1. Use `VertexPositionTexture` instead of `VertexPosition`
2. Embed JPG texture in GLB
3. Generate faces with hole detection (same as OBJ)
4. Use proper material with texture channel

### 2. JPG Minimap Conversion (Priority: HIGH)

**Current PNG code** (Program.cs line 701):
```csharp
image.Save(outStream, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
```

**Updated to JPG:**
```csharp
using SixLabors.ImageSharp.Formats.Jpeg;

image.Save(outStream, new JpegEncoder 
{ 
    Quality = 85  // 85 = good balance (50-100 scale)
});
```

**Quality recommendations:**
- **85**: Default - good balance of size/quality
- **90**: High quality - for detailed areas
- **75**: Lower quality - for distant/background tiles

### 3. Update Default Export Settings

**AdtMeshExtractor.cs** (line 41):
```csharp
// OLD:
bool exportGlb = false,
bool exportObj = true,

// NEW:
bool exportGlb = true,   // ✅ Default to GLB
bool exportObj = false,  // ❌ Disable OBJ by default
```

### 4. Minimap Extraction Updates

**Program.cs ExtractMinimapsFromMpq** (lines 656-739):
```csharp
// OLD:
var outputPath = Path.Combine(minimapOutDir, $"{mapName}_{x}_{y}.png");
image.Save(outStream, new PngEncoder());

// NEW:
var outputPath = Path.Combine(minimapOutDir, $"{mapName}_{x}_{y}.jpg");
image.Save(outStream, new JpegEncoder { Quality = 85 });
```

### 5. Viewer Updates

**2D Viewer** (index.html):
```javascript
// OLD:
const tileUrl = `minimap/${version}/${map}/${map}_${x}_${y}.png`;

// NEW:
const tileUrl = `minimap/${version}/${map}/${map}_${x}_${y}.jpg`;
```

**3D Viewer** (if implemented):
```javascript
// Load GLB instead of OBJ
const loader = new GLTFLoader();
loader.load(`mesh/${map}_${x}_${y}.glb`, (gltf) => {
    scene.add(gltf.scene);
});
```

## Expected Improvements

### File Size Comparison (256 tiles)

| Component | Current (PNG+OBJ) | Optimized (JPG+GLB) | Savings |
|-----------|-------------------|---------------------|---------|
| Meshes | 512MB - 1.5GB | 128MB - 256MB | **75%** |
| Minimaps | 51MB - 128MB | 13MB - 26MB | **75%** |
| **Total** | **563MB - 1.6GB** | **141MB - 282MB** | **75%** |

### Performance Improvements

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Tile load time | 200-500ms | 50-100ms | **4-5x faster** |
| Memory usage | High (uncompressed) | Low (compressed) | **60-70% less** |
| Browser decode | Slow (PNG) | Fast (JPG) | **2-3x faster** |
| Initial page load | 5-10s | 1-2s | **5x faster** |

## Implementation Order

1. ✅ **Add JPG support to minimap extraction** (easiest, immediate benefit)
   - Update `ExtractMinimapsFromMpq` in Program.cs
   - Add quality parameter (default 85)
   - Update file extensions

2. ✅ **Update 2D viewer to use JPG** (required for step 1)
   - Update tile URL generation
   - Test in browser

3. ✅ **Implement proper GLB export** (more complex)
   - Add VertexPositionTexture support
   - Embed JPG texture in GLB
   - Generate faces with hole detection
   - Test in Blender/Three.js

4. ✅ **Switch default to GLB** (after step 3 works)
   - Change default parameters
   - Update documentation

5. ⏳ **Optional: Add WebP support** (future optimization)
   - Even better compression than JPG
   - Requires fallback for older browsers

## Code Locations

### Files to Modify:

1. **WoWRollback.Cli/Program.cs**
   - `ExtractMinimapsFromMpq` (line 656): Add JPG encoder
   - Add quality parameter

2. **WoWRollback.AnalysisModule/AdtMeshExtractor.cs**
   - `ExtractFromArchive` (line 41): Switch defaults
   - `ExportGLB` (line 288): Implement properly with UVs and texture

3. **WoWRollback.Core/Services/Minimap/MinimapComposer.cs**
   - Update to output JPG instead of PNG

4. **Viewer HTML/JS files**
   - Update tile URL patterns (.png → .jpg)

### New Dependencies:

```xml
<!-- Already have these -->
<PackageReference Include="SixLabors.ImageSharp" Version="..." />
<PackageReference Include="SharpGLTF.Toolkit" Version="..." />
```

No new dependencies needed! ✅

## Testing Checklist

- [ ] JPG minimap extraction works
- [ ] JPG quality is acceptable (85)
- [ ] 2D viewer loads JPG tiles correctly
- [ ] GLB export includes UVs
- [ ] GLB export embeds texture
- [ ] GLB export respects holes
- [ ] GLB loads in Blender
- [ ] GLB loads in Three.js viewer
- [ ] File sizes reduced by ~75%
- [ ] Load times improved by 4-5x

## CLI Options & Flexibility

**Default behavior (optimized):**
```bash
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor
# Outputs: GLB meshes + JPG minimaps (75% smaller, 4-5x faster)
```

**Debug with OBJ export:**
```bash
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor --export-obj
# Outputs: OBJ + GLB meshes + JPG minimaps
```

**OBJ only (for debugging):**
```bash
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor --export-obj --no-glb
# Outputs: OBJ meshes + JPG minimaps
```

**High quality JPG:**
```bash
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor --jpg-quality 90
# Outputs: GLB meshes + high-quality JPG minimaps (larger files)
```

**Legacy PNG format:**
```bash
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor --minimap-format png
# Outputs: GLB meshes + PNG minimaps (for compatibility)
```

---

## Rollback Plan

If issues arise:
1. OBJ export available via `--export-obj` flag (kept for debugging)
2. PNG export available via `--minimap-format png` flag (kept for compatibility)
3. Quality adjustable via `--jpg-quality 85` parameter

## Detailed Implementation Guide

### PHASE 0: Fix Current OBJ Texture Bug (URGENT)

**Problem:** MTL files correctly reference texture filenames, but the texture files are in a different directory (`minimaps/`) than the mesh files (`{mapName}_mesh/`). Blender/viewers can't find them.

**Solution:** Copy minimap files to the mesh directory so they're co-located with OBJ/MTL files.

#### Step 0.1: Fix MTL Texture Reference

**Location:** `WoWRollback.AnalysisModule/AdtMeshExtractor.cs` line 340

**Current:**
```csharp
mtl.AppendLine($"map_Kd {Path.GetFileNameWithoutExtension(baseName)}.png");
```

**Change to:**
```csharp
mtl.AppendLine($"map_Kd {Path.GetFileNameWithoutExtension(baseName)}.jpg");
```

**Why:** We're switching to JPG format (Phase 1), so update the extension now.

#### Step 0.2: Copy Minimap Files to Mesh Directory (CRITICAL FIX)

**Add this code in the analysis workflow BEFORE mesh extraction:**

```csharp
// Copy minimap files to mesh directory so OBJ/GLB can reference them
var minimapSourceDir = Path.Combine(outDir, "minimaps");
var meshDir = Path.Combine(outDir, $"{mapName}_mesh");
Directory.CreateDirectory(meshDir);

if (Directory.Exists(minimapSourceDir))
{
    int copied = 0;
    foreach (var minimapFile in Directory.GetFiles(minimapSourceDir, $"{mapName}_*.jpg"))
    {
        var fileName = Path.GetFileName(minimapFile);
        var destPath = Path.Combine(meshDir, fileName);
        File.Copy(minimapFile, destPath, overwrite: true);
        copied++;
    }
    Console.WriteLine($"[info] Copied {copied} minimap textures to mesh directory");
}
else
{
    Console.WriteLine($"[warn] Minimap directory not found: {minimapSourceDir}");
}
```

**Location to add:** In `Program.cs`, in commands like `analyze-map-adts-mpq`, after minimap extraction but before mesh extraction.

**Result after fix:**
```
output/
├── minimaps/
│   ├── Kalimdor_39_27.jpg  ← Original location
│   └── ...
└── Kalimdor_mesh/
    ├── Kalimdor_39_27.obj
    ├── Kalimdor_39_27.mtl  ← References "Kalimdor_39_27.jpg"
    ├── Kalimdor_39_27.jpg  ← COPIED HERE! ✅
    └── ...
```

Now Blender/viewers can find the texture using the relative path in the MTL file!

#### Step 0.3: Fix UV Coordinate Flipping (CRITICAL FIX)

**Problem:** Textures appear mirrored on X and Y axes because UV coordinates aren't flipped.

**Location:** `WoWRollback.AnalysisModule/AdtMeshExtractor.cs` lines 227-244 (UV calculation section)

**Current code:**
```csharp
// Second pass: compute UVs from bounds
float spanX = Math.Max(1e-6f, maxX - minX);
float spanZ = Math.Max(1e-6f, maxZ - minZ);
const float eps = 2.5e-3f;

var uvs = new List<(float u, float v)>(positions.Count);
for (int i = 0; i < positions.Count; i++)
{
    var p = positions[i];
    // UV mapping: normalized to tile extents
    float u = (p.x - minX) / spanX;
    float v = (maxZ - p.z) / spanZ;
    
    u = Math.Clamp(u, eps, 1f - eps);
    v = Math.Clamp(v, eps, 1f - eps);
    
    uvs.Add((u, v));
}
```

**Fixed code (add flips BEFORE clamping):**
```csharp
// Second pass: compute UVs from bounds
float spanX = Math.Max(1e-6f, maxX - minX);
float spanZ = Math.Max(1e-6f, maxZ - minZ);
const float eps = 2.5e-3f;

// UV flip flags (matching working implementation defaults)
bool yFlip = true;  // Flip V coordinate
bool xFlip = true;  // Flip U coordinate

var uvs = new List<(float u, float v)>(positions.Count);
for (int i = 0; i < positions.Count; i++)
{
    var p = positions[i];
    // UV mapping: normalized to tile extents
    float u = (p.x - minX) / spanX;
    float v = (maxZ - p.z) / spanZ;
    
    // Apply flips (CRITICAL - matches working code)
    if (yFlip) v = 1f - v;
    if (xFlip) u = 1f - u;
    
    u = Math.Clamp(u, eps, 1f - eps);
    v = Math.Clamp(v, eps, 1f - eps);
    
    uvs.Add((u, v));
}
```

**Why these defaults?**
- `yFlip = true`: WoW minimap V coordinate increases downward, but texture V increases upward
- `xFlip = true`: WoW coordinate system vs texture coordinate system mismatch

**Result:** Texture will appear correctly oriented in Blender/viewers! ✅

---

### PHASE 1: JPG Minimap Support

#### Step 1.1: Update ExtractMinimapsFromMpq in Program.cs

**Location:** `WoWRollback.Cli/Program.cs` line 656

**Add using statement at top:**
```csharp
using SixLabors.ImageSharp.Formats.Jpeg;
```

**Find this code (line 691):**
```csharp
var outputPath = Path.Combine(minimapOutDir, $"{mapName}_{x}_{y}.png");
using var outStream = File.Create(outputPath);
image.Save(outStream, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
```

**Replace with:**
```csharp
var outputPath = Path.Combine(minimapOutDir, $"{mapName}_{x}_{y}.jpg");
using var outStream = File.Create(outputPath);
image.Save(outStream, new JpegEncoder { Quality = 85 });
```

**Also update the info message (line 706):**
```csharp
// OLD:
Console.WriteLine($"[info] Extracted and converted {extracted} minimap tiles (BLP→PNG)");

// NEW:
Console.WriteLine($"[info] Extracted and converted {extracted} minimap tiles (BLP→JPG)");
```

#### Step 1.2: Update MinimapComposer.cs

**Location:** `WoWRollback.Core/Services/Minimap/MinimapComposer.cs`

**Find the Save call (around line 50-60):**
```csharp
// Look for code like:
image.Save(outputPath, new PngEncoder());
// or
await image.SaveAsPngAsync(outputPath);
```

**Replace with:**
```csharp
image.Save(outputPath, new JpegEncoder { Quality = 85 });
```

**Update file extension in path generation:**
```csharp
// OLD:
var outputPath = Path.Combine(outputDir, $"{fileName}.png");

// NEW:
var outputPath = Path.Combine(outputDir, $"{fileName}.jpg");
```

#### Step 1.3: Update Viewer Tile URLs

**Location:** Search for all viewer HTML/JS files that reference minimap tiles

**Files to check:**
- `WoWRollback.ViewerModule/wwwroot/index.html`
- `WoWRollback.ViewerModule/wwwroot/js/*.js`
- Any viewer template files

**Find patterns like:**
```javascript
`minimap/${version}/${map}/${map}_${x}_${y}.png`
`${map}_${x}_${y}.png`
".png"
```

**Replace with:**
```javascript
`minimap/${version}/${map}/${map}_${x}_${y}.jpg`
`${map}_${x}_${y}.jpg`
".jpg"
```

**CSS background-image updates:**
```css
/* OLD */
background-image: url('minimap/0.5.3/Kalimdor/Kalimdor_32_32.png');

/* NEW */
background-image: url('minimap/0.5.3/Kalimdor/Kalimdor_32_32.jpg');
```

---

### PHASE 2: GLB Export Implementation

#### Step 2.1: Update ExportGLB Method Signature

**Location:** `WoWRollback.AnalysisModule/AdtMeshExtractor.cs` line 288

**Current signature:**
```csharp
private void ExportGLB(
    List<(float x, float y, float z)> positions,
    List<int> indices,
    string outputPath)
```

**New signature:**
```csharp
private void ExportGLB(
    List<(float x, float y, float z)> positions,
    List<(float u, float v)> uvs,
    List<int> chunkStartIndices,
    dynamic adt,
    string glbPath,
    string texturePath)
```

#### Step 2.2: Complete GLB Export Implementation

**Replace entire ExportGLB method with:**

```csharp
private void ExportGLB(
    List<(float x, float y, float z)> positions,
    List<(float u, float v)> uvs,
    List<int> chunkStartIndices,
    dynamic adt,
    string glbPath,
    string texturePath)
{
    using var scene = new SceneBuilder();
    
    // Load texture image (JPG)
    MaterialBuilder material;
    if (File.Exists(texturePath))
    {
        var textureImage = SixLabors.ImageSharp.Image.Load<Rgba32>(texturePath);
        var memoryImage = new MemoryImage(
            textureImage.Width,
            textureImage.Height,
            textureImage.GetPixelMemoryGroup().ToArray()[0]);
        
        material = new MaterialBuilder("TerrainMaterial")
            .WithMetallicRoughnessShader()
            .WithChannelImage(KnownChannel.BaseColor, memoryImage);
    }
    else
    {
        // Fallback: no texture
        material = new MaterialBuilder("TerrainMaterial")
            .WithMetallicRoughnessShader()
            .WithChannelParam(KnownChannel.BaseColor, new Vector4(0.6f, 0.6f, 0.6f, 1.0f));
    }

    var mesh = new MeshBuilder<VertexPositionTexture1>("TerrainMesh");
    var prim = mesh.UsePrimitive(material);

    // Build vertex array with UVs
    var vertices = new VertexPositionTexture1[positions.Count];
    for (int i = 0; i < positions.Count; i++)
    {
        var p = positions[i];
        var uv = uvs[i];
        
        // CRITICAL: Use z,x,y order for correct orientation (same as OBJ)
        vertices[i] = new VertexPositionTexture1(
            new Vector3(p.z, p.x, p.y),
            new Vector2(uv.u, uv.v));
    }

    // Generate faces with hole detection (same logic as OBJ export)
    int chunkIndex = 0;
    foreach (var chunk in adt.chunks)
    {
        if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
        {
            chunkIndex++;
            continue;
        }

        for (int j = 9, xx = 0, yy = 0; j < 145; j++, xx++)
        {
            if (xx >= 8) { xx = 0; yy++; }

            bool isHole = IsHole(chunk, xx, yy);

            if (!isHole)
            {
                int baseIndex = chunkStartIndices[chunkIndex];
                int i0 = j;
                int a = baseIndex + i0;
                int b = baseIndex + (i0 - 9);
                int c = baseIndex + (i0 + 8);
                int d = baseIndex + (i0 - 8);
                int e = baseIndex + (i0 + 9);
                
                // 4 triangles per quad
                // Note: GLB uses 0-based indexing (unlike OBJ's 1-based)
                prim.AddTriangle(vertices[a], vertices[b], vertices[c]);
                prim.AddTriangle(vertices[a], vertices[d], vertices[b]);
                prim.AddTriangle(vertices[a], vertices[e], vertices[d]);
                prim.AddTriangle(vertices[a], vertices[c], vertices[e]);
            }

            if (((j + 1) % (9 + 8)) == 0) j += 9;
        }

        chunkIndex++;
    }

    scene.AddRigidMesh(mesh, Matrix4x4.Identity);
    
    var model = scene.ToGltf2();
    model.SaveGLB(glbPath);
}
```

**Required using statements at top of file:**
```csharp
using System.Numerics;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using SharpGLTF.Schema2;
using SharpGLTF.Memory;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
```

#### Step 2.3: Update ExtractTileMesh to Call GLB Export

**Location:** `AdtMeshExtractor.cs` line 246-253

**Find this code:**
```csharp
// Export GLB if requested (TODO: implement properly)
string? glbFile = null;
if (exportGlb && positions.Count > 0)
{
    glbFile = $"{mapName}_{tileX}_{tileY}.glb";
    // var glbPath = Path.Combine(meshDir, glbFile);
    // ExportGLB(positions, uvs, chunkStartIndices, adt, glbPath);
}
```

**Replace with:**
```csharp
// Export GLB if requested
string? glbFile = null;
if (exportGlb && positions.Count > 0)
{
    glbFile = $"{mapName}_{tileX}_{tileY}.glb";
    var glbPath = Path.Combine(meshDir, glbFile);
    
    // Texture path (JPG minimap - must exist or GLB will use fallback color)
    var texturePath = Path.Combine(meshDir, $"{mapName}_{tileX}_{tileY}.jpg");
    
    ExportGLB(positions, uvs, chunkStartIndices, adt, glbPath, texturePath);
}
```

#### Step 2.4: Switch Default Export to GLB (Keep OBJ as Option)

**Location:** `AdtMeshExtractor.cs` line 41-42

**Change defaults:**
```csharp
// OLD:
bool exportGlb = false,
bool exportObj = true,

// NEW:
bool exportGlb = true,   // Default to GLB (smaller, faster)
bool exportObj = false,  // Keep as option for debugging
```

**Keep OBJ export available via CLI flag:**
- Users can still export OBJ with `--export-obj` flag
- Useful for debugging, Blender import, manual inspection
- Both formats can be exported simultaneously if needed

**When to use OBJ:**
- ✅ Debugging mesh issues (human-readable format)
- ✅ Importing into Blender for editing
- ✅ Manual inspection of vertex/face data
- ✅ Compatibility with older tools

**When to use GLB (default):**
- ✅ Production viewer (75% smaller, faster loading)
- ✅ Web-based 3D visualization (native browser support)
- ✅ Distributing large datasets (much smaller file sizes)
- ✅ Mobile/low-bandwidth scenarios

---

### PHASE 3: Coordinate Minimap + Mesh Export

#### Step 3.1: Ensure Minimap Extracted Before Mesh

**Problem:** GLB export needs JPG texture to exist first.

**Solution:** In the analysis workflow, ensure minimaps are extracted before meshes.

**Location:** Check `Program.cs` analysis commands (e.g., `analyze-map-adts-mpq`)

**Ensure order is:**
1. Extract ADT placements
2. **Extract minimaps** ← Must come first
3. Extract meshes (GLB references minimap JPG)
4. Generate viewer

**Example order (around line 1120-1130):**
```csharp
// Step 1: Extract placements
var extractResult = extractor.ExtractFromArchive(...);

// Step 2: Extract minimaps FIRST
var minimapDir = ExtractMinimapsFromMpq(src, mapName, outDir);

// Step 3: Extract meshes (can now reference minimap JPGs)
var meshExtractor = new AdtMeshExtractor();
var meshResult = meshExtractor.ExtractFromArchive(src, mapName, outDir, 
    exportGlb: true, exportObj: false);
```

#### Step 3.2: CRITICAL - Copy Minimaps to Mesh Directory

**Problem:** 
- Minimap JPGs are in `minimaps/` directory
- Both OBJ and GLB export look for them in `{mapName}_mesh/` directory
- **Current bug:** OBJ MTL files reference textures that don't exist → checkerboard pattern

**Solution Option A:** Copy minimap JPGs to mesh directory before mesh export (RECOMMENDED).

```csharp
// After minimap extraction, before mesh extraction:
var minimapDir = Path.Combine(outDir, "minimaps");
var meshDir = Path.Combine(outDir, $"{mapName}_mesh");
Directory.CreateDirectory(meshDir);

// Copy JPG files to mesh directory
if (Directory.Exists(minimapDir))
{
    foreach (var jpgFile in Directory.GetFiles(minimapDir, $"{mapName}_*.jpg"))
    {
        var fileName = Path.GetFileName(jpgFile);
        var destPath = Path.Combine(meshDir, fileName);
        File.Copy(jpgFile, destPath, overwrite: true);
    }
    Console.WriteLine($"[info] Copied {Directory.GetFiles(minimapDir, $"{mapName}_*.jpg").Length} minimap JPGs to mesh directory");
}
```

**Solution Option B:** Update GLB export to look in minimap directory.

```csharp
// In ExtractTileMesh method, line 502:
// OLD:
var texturePath = Path.Combine(meshDir, $"{mapName}_{tileX}_{tileY}.jpg");

// NEW:
var minimapPath = Path.Combine(Path.GetDirectoryName(meshDir)!, "minimaps", $"{mapName}_{tileX}_{tileY}.jpg");
var texturePath = File.Exists(minimapPath) ? minimapPath : Path.Combine(meshDir, $"{mapName}_{tileX}_{tileY}.jpg");
```

**Recommendation:** Use Option A (copy files) - simpler, keeps all mesh-related files together, easier for distribution.

---

### PHASE 4: Viewer Updates for GLB

#### Step 4.1: Add Three.js GLTFLoader (if not present)

**Location:** Viewer HTML file (e.g., `wwwroot/index.html`)

**Add script tag:**
```html
<script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.150.0/examples/js/loaders/GLTFLoader.js"></script>
```

#### Step 4.2: Load GLB Meshes in Viewer

**Example JavaScript code:**
```javascript
const loader = new THREE.GLTFLoader();

function loadTileMesh(mapName, tileX, tileY) {
    const glbUrl = `mesh/${mapName}_${tileX}_${tileY}.glb`;
    
    loader.load(glbUrl, (gltf) => {
        // Add to scene
        scene.add(gltf.scene);
        
        // Optional: position based on tile coordinates
        gltf.scene.position.set(tileX * 533.33, 0, tileY * 533.33);
    }, undefined, (error) => {
        console.error(`Failed to load ${glbUrl}:`, error);
    });
}
```

---

## Edge Cases & Error Handling

### 1. Missing Minimap Texture

**Problem:** GLB export called but JPG doesn't exist yet.

**Solution:** Fallback to solid color material (already in code above).

### 2. Corrupt BLP Files

**Problem:** BLP→JPG conversion fails.

**Solution:** Catch exception, log warning, continue with other tiles.

```csharp
try
{
    var blp = new Warcraft.NET.Files.BLP.BLP(blpData);
    var image = blp.GetMipMap(0);
    image.Save(outStream, new JpegEncoder { Quality = 85 });
}
catch (Exception ex)
{
    Console.WriteLine($"[warn] Failed to convert tile [{x},{y}]: {ex.Message}");
    continue; // Skip this tile
}
```

### 3. Large Tile Coordinates

**Problem:** Tiles outside 0-63 range.

**Solution:** Already handled by loop bounds (0-63).

### 4. Empty Chunks

**Problem:** Chunk has no vertices.

**Solution:** Already handled - skip and continue.

```csharp
if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
{
    chunkStartIndices.Add(positions.Count);
    continue;
}
```

### 5. UV Out of Bounds

**Problem:** UV coordinates < 0 or > 1.

**Solution:** Already clamped with epsilon.

```csharp
const float eps = 2.5e-3f;
u = Math.Clamp(u, eps, 1f - eps);
v = Math.Clamp(v, eps, 1f - eps);
```

---

## Testing Commands

### Test JPG Minimap Extraction
```bash
dotnet run --project WoWRollback.Cli -- probe-minimap \
  --client-path "G:\WoW\...\0.6.0.3592\" \
  --map Kalimdor \
  --limit 5
```

**Expected:** Files named `Kalimdor_X_Y.jpg` (not .png)

### Test GLB Mesh Export
```bash
dotnet run --project WoWRollback.Cli -- analyze-map-adts-mpq \
  --client-path "G:\WoW\...\0.6.0.3592\" \
  --map Kalimdor \
  --version "0.6.0.3592" \
  --max-tiles 5
```

**Expected:** Files named `Kalimdor_X_Y.glb` in `{output}/Kalimdor_mesh/`

### Verify GLB in Blender
1. Open Blender
2. File → Import → glTF 2.0 (.glb/.gltf)
3. Select `Kalimdor_39_27.glb`
4. Should see textured terrain mesh

### Verify in Browser
1. Start viewer: `dotnet run --project WoWRollback.Cli -- serve-viewer`
2. Open http://localhost:8080
3. Check browser console for errors
4. Verify tiles load as JPG (Network tab)

---

## File Size Verification

### Before (PNG + OBJ):
```bash
# Check minimap size
ls -lh output/minimaps/*.png | head -5
# Expected: 200-500KB each

# Check mesh size
ls -lh output/Kalimdor_mesh/*.obj | head -5
# Expected: 2-6MB each
```

### After (JPG + GLB):
```bash
# Check minimap size
ls -lh output/minimaps/*.jpg | head -5
# Expected: 50-100KB each (70% reduction)

# Check mesh size
ls -lh output/Kalimdor_mesh/*.glb | head -5
# Expected: 500KB-1MB each (75% reduction)
```

---

## Rollback Instructions

If GLB export has issues, revert to OBJ:

```csharp
// In AdtMeshExtractor.cs line 41:
bool exportGlb = false,  // ← Disable GLB
bool exportObj = true,   // ← Re-enable OBJ
```

If JPG has quality issues, revert to PNG:

```csharp
// In Program.cs line 691:
var outputPath = Path.Combine(minimapOutDir, $"{mapName}_{x}_{y}.png");
image.Save(outStream, new PngEncoder());
```

Or adjust quality:
```csharp
new JpegEncoder { Quality = 90 }  // Higher quality (larger files)
```

---

## Future Optimizations

1. **WebP support**: Even better compression
2. **Texture atlasing**: Combine multiple tiles into one texture
3. **LOD meshes**: Lower detail for distant tiles
4. **Mesh simplification**: Reduce vertex count for flat areas
5. **Draco compression**: Further compress GLB geometry
