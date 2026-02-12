# MdxViewer Architecture Enhancement Plan

## Current State

### Working Components
- MPQ archive loading with internal listfile extraction
- Loose file scanning (Dungeons, Textures, World folders)
- MDX model loading and basic rendering
- WMO (World Map Object) loading and basic rendering
- Camera controls (mouse look, WASD movement)
- File browser with extension filters and search
- Basic shader-based rendering with directional lighting

### Known Issues
- Animation data shows garbage values in Info panel
- No animation playback system
- Limited texture support (replaceable IDs, minimal loose file support)
- No transparent texture rendering (alpha blending)
- No terrain rendering system
- No detail doodad system
- No day/night cycle / terrain lighting
- No frustum culling optimization
- Build version inference fails for 0.5.3

---

## 1. Animation System

### Current Issue
Animation data displays garbage values in the Info panel because the animation structures aren't properly parsed or displayed.

### Required Changes

#### Data Layer (`MdxLTool.Formats.Mdx`)
- [ ] Fix `AnimationSequence` structure parsing
- [ ] Implement `Animation` track parsing (translation, rotation, scale)
- [ ] Parse `KeyframeTrack<T>` data correctly
- [ ] Implement animation interpolation (linear, bezier, hermite)

#### UI Layer (`ViewerApp`)
- [ ] Add animation selector dropdown
- [ ] Add animation playback controls (play, pause, scrub)
- [ ] Display correct animation timing (non-garbage values)
- [ ] Show animation sequence names and durations

#### Rendering Layer (`ModelRenderer`)
- [ ] Implement bone transformation matrix calculation
- [ ] Apply animation keyframes to bone hierarchy
- [ ] Update vertex positions based on animated bones
- [ ] Support multiple animation sequences per model

### Key Files
- `src/MdxViewer/Formats/Mdx/GeosetAnimation.cs` - Animation track structures
- `src/MdxViewer/Formats/Mdx/Animation.cs` - To be created
- `src/MdxViewer/Rendering/AnimationController.cs` - To be created

---

## 2. Texture System

### Current Issue
Limited texture support - loose files not properly searched, replaceable IDs not resolved, no alpha transparency.

### Required Changes

#### Data Layer
- [ ] Implement `ReplaceableTextureResolver` for 0.5.3 (fix build version lookup)
- [ ] Add loose file texture search in model directory
- [ ] Support for Alpha texture formats (BLP2, etc.)

#### UI Layer (`ViewerApp`)
- [ ] Add texture browser panel
- [ ] Show currently loaded textures
- [ ] Allow texture override selection

#### Rendering Layer (`ModelRenderer`, `Shader`)
- [ ] Implement alpha testing for transparent textures
- [ ] Support alpha blending for translucent materials
- [ ] Parse material blend modes (opaque, alpha, add, modulate)
- [ ] Multi-pass rendering for complex transparency

### Shader Changes
```glsl
// Fragment shader needs:
uniform sampler2D uSampler;
uniform int uBlendMode;  // 0=opaque, 1=alpha, 2=add, 3=modulate
uniform float uAlphaTest; // Alpha cutoff value

void main() {
    vec4 texColor = texture(uSampler, vTexCoord);
    
    if (uBlendMode == 1) { // Alpha blended
        if (texColor.a < uAlphaTest) discard;
    }
    // ... rest of lighting calculation
}
```

---

## 3. Rendering Pipeline

### Current Issue
No proper render sorting for transparent objects, no depth sorting.

### Required Changes

#### Render Queue System
- [ ] Create `RenderQueue` to sort objects by depth
- [ ] Implement painter's algorithm for transparency
- [ ] Separate opaque and transparent render passes
- [ ] Add render priority/layer system

#### Frustum Culling (from documentation)
- [ ] Extract frustum planes from view-projection matrix
- [ ] Implement sphere-AABB intersection tests
- [ ] Add culling to WMO and model rendering
- [ ] Only render visible objects

```csharp
// Frustum culling approach
public class Frustum
{
    public Plane[] Planes { get; } = new Plane[6];
    
    public void ExtractPlanes(Matrix4x4 viewProj)
    {
        // Extract all 6 frustum planes
    }
    
    public bool Contains(BoundingSphere sphere)
    {
        // Sphere-frustum test
    }
}
```

---

## 4. Terrain Rendering System

### From Documentation
The documentation covers a complete Alpha 0.5.3 terrain system:

#### Data Structures
- [ ] Implement `MapArea` parser (16x16 chunks)
- [ ] Implement `MapChunk` parser (terrain cells)
- [ ] Parse terrain vertices and indices
- [ ] Implement heightmap calculation

#### Rendering Pipeline
- [ ] Create terrain shader with texture layering
- [ ] Implement alpha map blending
- [ ] Support up to 4 texture layers
- [ ] Parse and render liquid (water, ocean, magma, slime)

#### Key Constants
```
CHUNK_SCALE = 0.001875 (1/533.3333)
CHUNK_OFFSET = 266.6667
CHUNK_SIZE = 533.3333
CELL_SIZE = 66.6667
NUM_CELLS = 8
NUM_VERTICES = 9
```

### Key Files to Create
- `src/MdxViewer/Rendering/TerrainRenderer.cs`
- `src/MdxViewer/Formats/Terrain/MapArea.cs`
- `src/MdxViewer/Formats/Terrain/MapChunk.cs`
- `src/MdxViewer/Formats/Terrain/LiquidData.cs`

---

## 5. Detail Doodad System

### From Documentation
Detail doodads are background vegetation/rocks rendered around the camera.

#### Implementation
- [ ] Parse detail doodad definitions from ADT files
- [ ] Implement detail doodad placement
- [ ] Create LOD system (fade in/out based on distance)
- [ ] Random rotation and scale variations

```csharp
public class DetailDoodadManager
{
    public void Render(Vector3 cameraPos, float distance)
    {
        // Only render within DETAIL_DOODAD_DISTANCE (100.0f)
        // Scale/fade based on distance
    }
}
```

---

## 6. Lighting System

### From Documentation
Alpha 0.5.3 has a sophisticated day/night cycle.

#### Implementation
- [ ] Implement sun position calculation based on time
- [ ] Calculate ambient, diffuse, and fog colors
- [ ] Support day/night color transitions
- [ ] Apply lighting to terrain and models

```csharp
public class LightingSystem
{
    public Vector3 SunPosition { get; private set; }
    public Vector3 AmbientColor { get; private set; }
    public Vector3 SunColor { get; private set; }
    public Vector3 FogColor { get; private set; }
    
    public void Update(float timeOfDay) // 0-24 hours
    {
        // Calculate sun position based on time
        // Interpolate colors for day/night transition
    }
}
```

---

## 7. Model Info Panel Fixes

### Current Issues
- Animation data shows garbage
- Build version inference fails

### Required Changes
- [ ] Fix animation sequence parsing and display
- [ ] Implement WoWDBDefs lookup for 0.5.3 specifically
- [ ] Add bone hierarchy visualization
- [ ] Show material/texture information

```csharp
// Fix build version lookup - search WoWDBDefs for 0.5.3
private void LoadBuildVersion()
{
    var dbdFiles = Directory.GetFiles(_dbDefPath, "*.dbd");
    foreach (var file in dbdFiles)
    {
        var content = File.ReadAllText(file);
        if (content.Contains("0.5.3") || content.Contains("build = 0.5.3"))
        {
            _buildVersion = "0.5.3";
            break;
        }
    }
}
```

---

## 8. WMO Loading from Listfile-less MPQs

### Working Feature (Added)
The `NativeMpqService` now has:
- `ExtractInternalListfiles()` - Extracts files from MPQ listfiles
- `ScanForWmoFiles()` - Scans for WMO files by 'MOHD' magic
- `ReadScannedFile()` - Reads scanned files by placeholder path

### Next Steps
- [ ] Add UI to browse scanned WMO files
- [ ] Implement file loading from placeholder paths
- [ ] Support other scanned file types (M2/MDX, textures)

---

## Implementation Priority

### Phase 1: Critical Fixes
1. Build version lookup for 0.5.3
2. Animation data parsing and display
3. Basic animation playback

### Phase 2: Visual Improvements
1. Transparent texture rendering
2. Loose file texture support
3. Lighting system

### Phase 3: Terrain & Doodads
1. Terrain renderer
2. Detail doodad system
3. Frustum culling

### Phase 4: Polish
1. Day/night cycle
2. UI improvements
3. Performance optimization

---

## File Structure

```
src/MdxViewer/
├── DataSources/
│   ├── MpqDataSource.cs
│   └── LooseFileDataSource.cs
├── Formats/
│   ├── Mdx/
│   │   ├── Model.cs
│   │   ├── Animation.cs (new)
│   │   ├── Geoset.cs
│   │   └── Bone.cs (new)
│   └── Terrain/ (new)
│       ├── MapArea.cs
│       ├── MapChunk.cs
│       └── LiquidData.cs
├── Rendering/
│   ├── ModelRenderer.cs
│   ├── WmoRenderer.cs
│   ├── TerrainRenderer.cs (new)
│   ├── AnimationController.cs (new)
│   ├── LightingSystem.cs (new)
│   ├── Frustum.cs (new)
│   ├── RenderQueue.cs (new)
│   └── Shaders/
│       ├── Model.frag
│       ├── Model.vert
│       ├── Terrain.frag (new)
│       └── Terrain.vert (new)
└── ViewerApp.cs
```

---

## Dependencies

### NuGet Packages
- Silk.NET.OpenGL 3.0.0+ (rendering)
- Silk.NET.Input 3.0.0+ (input)
- ImGui.NET 1.89+ (UI)
- SixLabors.ImageSharp 3.1.6 (texture loading)

### Internal Libraries
- WoWMapConverter.Core (MPQ reading, format converters)
- Warcraft.NET (file format definitions)
- SereniaBLPLib (BLP texture format)
