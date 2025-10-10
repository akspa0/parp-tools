# Three.js 3D Viewer Implementation Plan

## Goal
Replace broken Leaflet 2D viewer with simple Three.js 3D viewer that:
- Uses existing WDL→GLB terrain exports
- Displays object placements (M2/WMO) at correct positions
- Supports custom minimaps/textures
- No coordinate transformation hell

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ C# Pipeline                                             │
├─────────────────────────────────────────────────────────┤
│ 1. ADT Parser → Read terrain + placements              │
│ 2. WDL Regenerator → Convert Alpha WDL → 3.3.5 format  │
│ 3. GLB Exporter → Terrain mesh with textures           │
│ 4. Minimap Generator → Render ADT textures → PNG/WebP  │
│ 5. Master Index → { placements, worldCoords }          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Three.js Viewer                                         │
├─────────────────────────────────────────────────────────┤
│ 1. Load GLB terrain mesh                               │
│ 2. Apply minimap textures to mesh                      │
│ 3. Place objects at worldWest/North/Up coords          │
│ 4. Render with OrbitControls                           │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: WDL Regeneration (Alpha → 3.3.5)

### Problem
- Alpha 0.5.3/0.5.5 WDL format not compatible with existing GLB exporter
- Need to regenerate to 3.3.5 format

### Solution: Use Noggit-Red WDL Generation Code

**Files to port from noggit-red:**
- `src/noggit/Red/PreloadWDL/` - WDL generation logic
- Reads ADT heights and generates low-res WDL mesh
- Outputs 3.3.5 compatible WDL format

**Implementation:**

```csharp
// File: WoWRollback.WdlRegeneration/WdlRegenerator.cs
public class WdlRegenerator
{
    /// <summary>
    /// Regenerate WDL file from ADT data in 3.3.5 format
    /// Ports noggit-red WDL generation logic
    /// </summary>
    public void RegenerateWdl(string mapName, List<AdtAlpha> adts, string outputPath)
    {
        // 1. Extract heightmap from each ADT (MCVT chunks)
        // 2. Downsample to WDL resolution (64x64 per tile → 17x17 WDL grid)
        // 3. Generate MARE chunk (low-res heightmap)
        // 4. Generate MAOF chunk (area offsets)
        // 5. Generate MAHO chunk (hole data)
        // 6. Write as 3.3.5 WDT format
        
        var wdl = new WdlFile();
        
        foreach (var adt in adts)
        {
            var heights = ExtractHeightmap(adt);
            var downsampled = DownsampleTo17x17(heights);
            wdl.AddTileHeights(adt.TileX, adt.TileY, downsampled);
        }
        
        wdl.WriteTo335Format(outputPath);
    }
    
    private float[,] ExtractHeightmap(AdtAlpha adt)
    {
        // Extract MCVT data from each MCNK chunk
        // Alpha format: MCVT = 145 floats (9x9 + 8x8)
        // Return 129x129 heightmap for full ADT
    }
    
    private float[,] DownsampleTo17x17(float[,] heights)
    {
        // Downsample 129x129 → 17x17 for WDL
        // Simple averaging or bilinear filtering
    }
}
```

**Reference:**
- noggit-red repo: https://github.com/Marlamin/noggit-red
- WDL spec: https://wowdev.wiki/WDL

---

## Phase 2: Minimap Texture Generation

### Problem
- Alpha 0.5.3/0.5.5 missing many minimap tiles
- Need to generate from ADT texture data

### Solution: Render ADT Textures to Minimap

**Implementation:**

```csharp
// File: WoWRollback.MinimapGenerator/AdtTextureRenderer.cs
public class AdtTextureRenderer
{
    /// <summary>
    /// Render ADT texture layers to minimap image
    /// Ports texture blending logic from wow.export or noggit
    /// </summary>
    public async Task<Image> RenderMinimap(AdtAlpha adt, string blpCachePath)
    {
        var minimap = new Image<Rgba32>(512, 512);
        
        // For each MCNK chunk (16x16 grid)
        foreach (var chunk in adt.GetMcnkChunks())
        {
            // 1. Get texture layers (MCLY)
            var layers = chunk.GetTextureLayers();
            
            // 2. Get alpha maps (MCAL)
            var alphaMaps = chunk.GetAlphaMaps();
            
            // 3. Load BLP textures
            var textures = await LoadTextures(layers, blpCachePath);
            
            // 4. Blend layers using alpha maps
            var blended = BlendLayers(textures, alphaMaps);
            
            // 5. Write to minimap at chunk position
            int chunkX = chunk.IndexX * 32; // Each chunk = 32x32 pixels
            int chunkY = chunk.IndexY * 32;
            WriteChunkToMinimap(minimap, blended, chunkX, chunkY);
        }
        
        return minimap;
    }
    
    private Image BlendLayers(List<Image> textures, List<AlphaMap> alphaMaps)
    {
        // Start with base layer (layer 0)
        var result = textures[0].Clone();
        
        // Blend additional layers using alpha maps
        for (int i = 1; i < textures.Count; i++)
        {
            BlendLayer(result, textures[i], alphaMaps[i - 1]);
        }
        
        return result;
    }
    
    private void BlendLayer(Image dest, Image src, AlphaMap alpha)
    {
        // For each pixel, blend src over dest using alpha
        for (int y = 0; y < dest.Height; y++)
        {
            for (int x = 0; x < dest.Width; x++)
            {
                float a = alpha.GetAlpha(x, y) / 255f;
                dest[x, y] = Blend(dest[x, y], src[x, y], a);
            }
        }
    }
}
```

**Libraries:**
- **SixLabors.ImageSharp** - Image manipulation (already in project)
- **SereniaBLPLib** - BLP texture loading
- **CASCLib** - Extract textures from MPQ/CASC

**Reference:**
- wow.export minimap renderer: https://github.com/Kruithne/wow.export
- ADT texture blending: https://wowdev.wiki/ADT/v18#MCLY_chunk

---

## Phase 3: GLB Terrain with Textures

### Current State
- `WdlGltfExporter.cs` exports terrain mesh
- No textures applied (grey mesh only)

### Enhancement: Apply Minimap Textures

```csharp
// Enhance WdlGltfExporter.cs
public class WdlGltfExporter
{
    public void ExportWithTextures(string wdlPath, string minimapPath, string outputGlb)
    {
        // 1. Generate mesh from WDL (existing code)
        var mesh = GenerateMeshFromWdl(wdlPath);
        
        // 2. Generate UV coordinates
        mesh.UVs = GenerateUVCoordinates(mesh.Vertices);
        
        // 3. Load minimap texture
        var texture = Image.Load(minimapPath);
        
        // 4. Create GLTF with texture
        var gltf = new ModelRoot();
        var material = gltf.CreateMaterial("terrain")
            .WithPBRMetallicRoughness(
                baseColorTexture: texture,
                metallicFactor: 0,
                roughnessFactor: 1
            );
        
        var meshNode = gltf.CreateMesh("terrain");
        meshNode.CreatePrimitive()
            .WithVertexAccessor("POSITION", mesh.Vertices)
            .WithVertexAccessor("TEXCOORD_0", mesh.UVs)
            .WithIndicesAccessor(PrimitiveType.TRIANGLES, mesh.Indices)
            .WithMaterial(material);
        
        gltf.SaveGLB(outputGlb);
    }
    
    private Vector2[] GenerateUVCoordinates(Vector3[] vertices)
    {
        // Map vertices to 0-1 UV space
        // Vertices are in WoW world coords, map to texture space
        var uvs = new Vector2[vertices.Length];
        
        for (int i = 0; i < vertices.Length; i++)
        {
            // Normalize to 0-1 based on map bounds
            uvs[i] = new Vector2(
                (vertices[i].X + 17066.66f) / (17066.66f * 2),
                (vertices[i].Z + 17066.66f) / (17066.66f * 2)
            );
        }
        
        return uvs;
    }
}
```

---

## Phase 4: Three.js Viewer

### Simplified Viewer (No Coordinate Hell)

```html
<!DOCTYPE html>
<html>
<head>
    <title>WoW Rollback 3D Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: monospace;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>
</head>
<body>
    <div id="info">
        <strong>WoW Rollback 3D Viewer</strong><br>
        Map: <span id="mapName">Loading...</span><br>
        Objects: <span id="objectCount">0</span><br>
        <br>
        <strong>Controls:</strong><br>
        Left Mouse: Rotate<br>
        Right Mouse: Pan<br>
        Scroll: Zoom<br>
        Click Object: Details
    </div>
    
    <script type="module">
        import * as THREE from 'three';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        
        // Setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x87CEEB); // Sky blue
        
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            50000
        );
        camera.position.set(0, 2000, 2000);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 0.5);
        scene.add(directionalLight);
        
        // Load terrain
        const gltfLoader = new GLTFLoader();
        const terrainPath = '../../terrain/Kalidar_terrain.glb';
        
        gltfLoader.load(terrainPath, (gltf) => {
            scene.add(gltf.scene);
            console.log('Terrain loaded');
        });
        
        // Load placements
        const response = await fetch('../../04_analysis/0.5.3/master/Kalidar_master_index.json');
        const data = await response.json();
        
        document.getElementById('mapName').textContent = data.map;
        let objectCount = 0;
        
        // Raycaster for clicking
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const clickableObjects = [];
        
        // Add placements
        data.tiles.forEach(tile => {
            tile.placements.forEach(p => {
                const isWmo = p.kind === 'WMO';
                
                // Create geometry
                const geometry = isWmo
                    ? new THREE.BoxGeometry(20, 20, 20)
                    : new THREE.SphereGeometry(10);
                
                // Create material
                const material = new THREE.MeshPhongMaterial({
                    color: isWmo ? 0xFF9800 : 0x2196F3,
                    emissive: isWmo ? 0x442200 : 0x001144,
                    shininess: 30
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                
                // Direct position from pipeline data (NO TRANSFORMATION)
                mesh.position.set(
                    p.worldWest,  // X
                    p.worldUp,    // Y (elevation)
                    p.worldNorth  // Z
                );
                
                // Store placement data for popup
                mesh.userData = {
                    uniqueId: p.uniqueId,
                    kind: p.kind,
                    assetPath: p.assetPath,
                    worldWest: p.worldWest,
                    worldNorth: p.worldNorth,
                    worldUp: p.worldUp,
                    tileX: tile.tileX,
                    tileY: tile.tileY
                };
                
                scene.add(mesh);
                clickableObjects.push(mesh);
                objectCount++;
            });
        });
        
        document.getElementById('objectCount').textContent = objectCount;
        
        // Click handler
        function onClick(event) {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(clickableObjects);
            
            if (intersects.length > 0) {
                const obj = intersects[0].object.userData;
                alert(`
                    UID: ${obj.uniqueId}
                    Type: ${obj.kind}
                    Path: ${obj.assetPath}
                    Position: (${obj.worldWest.toFixed(2)}, ${obj.worldNorth.toFixed(2)}, ${obj.worldUp.toFixed(2)})
                    Tile: ${obj.tileX}_${obj.tileY}
                `);
            }
        }
        
        window.addEventListener('click', onClick);
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
```

---

## Implementation Order

### Week 1: WDL Regeneration
1. Port noggit-red WDL generation code to C#
2. Test with Kalidar Alpha ADTs
3. Verify 3.3.5 WDL output

### Week 2: Minimap Generation
1. Implement ADT texture extraction
2. Implement layer blending (MCLY + MCAL)
3. Generate minimap tiles for missing areas
4. Allow custom minimap override

### Week 3: Enhanced GLB Export
1. Add UV coordinates to WdlGltfExporter
2. Apply minimap textures to terrain mesh
3. Test with Three.js viewer

### Week 4: Three.js Viewer
1. Create simple-3d-viewer.html
2. Test object placement (NO transformations)
3. Add click handlers and popups
4. Polish UI and controls

---

## Success Criteria

- ✅ WDL regenerated from Alpha ADTs to 3.3.5 format
- ✅ Custom minimaps generated from ADT textures
- ✅ GLB terrain has textures applied
- ✅ Objects render at correct positions in 3D
- ✅ Click to see object details
- ✅ Smooth 60fps with 1000+ objects
- ✅ **ZERO coordinate transformation bugs**

---

## Dependencies

### C# Libraries
- **SharpGLTF** - GLB export (already in project)
- **SixLabors.ImageSharp** - Image manipulation
- **SereniaBLPLib** - BLP texture loading
- **Newtonsoft.Json** - JSON (already in project)

### JS Libraries (CDN)
- **Three.js** - 3D rendering
- **OrbitControls** - Camera controls
- **GLTFLoader** - Load GLB terrain

### Reference Code
- **noggit-red**: WDL generation
- **wow.export**: Minimap rendering
- **wowdev.wiki**: File format specs

---

## Next Steps

1. Create `WoWRollback.WdlRegeneration` project
2. Port noggit WDL code
3. Create `WoWRollback.MinimapGenerator` project
4. Implement texture blending
5. Enhance `WdlGltfExporter` with UVs/textures
6. Create `simple-3d-viewer.html`
7. Test end-to-end

**Estimated Time: 3-4 weeks for complete working 3D viewer**
