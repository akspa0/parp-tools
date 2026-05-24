# Phase 0 Research: Data Pipeline + MdxViewer Rendering Reference

## 1. Input Data Pipeline (Proven — from V16)

### IArchiveCatalog (NativeMpqService)
- `wow-viewer/src/core/WowViewer.Core.IO/Files/NativeMpqService.cs`
- `ReadFile(string virtualPath) → byte[]?`
- `FileExists(string virtualPath) → bool`
- Also: `MpqArchiveCatalog` with `ScanMapMpqArchives()` for per-map MPQs

### WorldTerrainTileBuilder (Runtime)
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainTileBuilder.cs`
- `Read(Stream, MapFileSummary, AdtTextureFile?) → WorldTerrainTileData`
- Internally: reads MCNK headers from MCIN → parses MCVT heights, MCLY layers, MCAL alpha via AdtTextureChunkReader/AdtMcalDecoder
- AdtTextureFile loaded from companion `_tex0.adt` via `AdtTextureReader.Read()`

### WorldTerrainTileData
- `SourcePath`, `Kind` (MapFileKind), `Chunks[]`, `Heightmap?`
- 256 chunks per tile (16x16 grid)

### WorldTerrainChunkData
- `IndexX`, `IndexY` (0-15), `AreaId`, `Flags`, `HoleMask` (16-bit)
- `Heights[]` (145 float MCVT samples)
- `TextureLayers[]` (IReadOnlyList<AdtTextureChunkLayer>)

### AdtTextureChunkLayer
- `Index` (0-3), `TextureId`, `TexturePath`, `Flags`, `AlphaOffset`, `EffectId`
- `DecodedAlpha` (AdtMcalDecodedLayer with Alpha[] 4096-byte 64x64 array)
- Layer 0 has NO alpha (base layer); layers 1-3 may have alpha

## 2. MdxViewer Rendering Reference

### Vertex Format (Batched Tile Path)
- Position: 3 floats (world X/Y, height Z)
- Normal: 3 floats
- UV: 2 floats (0-1 per chunk)
- VertexColor: 4 floats (MCCV tint, BGRA stored, 127/255 = neutral)
- ChunkSlice: 1 uint (which of 256 chunks)
- TexIdx: 4 uints (texture array layer indices, 0xFFFF = unused)

### Texture Approach
- **Diffuse**: GL_TEXTURE_2D_ARRAY of all unique BLP textures, resampled to 256x256, with mipmaps
- **Alpha+Shadow**: GL_TEXTURE_2D_ARRAY 64x64x256 RGBA8 (R=alpha1, G=alpha2, B=alpha3, A=shadow)
- Per-chunk slice in alpha/shadow array

### Shader Math
- UV: `vec2(-vWorldPos.y, -vWorldPos.x) * (8.0/33.333)` for world-UV
- Lighting: `abs(dot(normal, normalize(lightDir)))` with ambient + light color
- Alpha blend: layer 0 base, layers 1-3 mix via alpha
- MCCV: `clamp(vVertexColor.rgb * 2.0, 0.0, 2.0)` tint; `clamp(vVertexColor.a * 2.0 - 1.0, 0.0, 1.0)` strength
- Shadow: alphaShadow.a * 0.4 darkening
- Fog: linear mix based on distance(start,end)
- Output alpha: 1.0 when layer 0 visible; computed when hidden

### Mesh Building
- 145 vertices, 256 triangles (768 indices) per chunk minus holes
- All chunks packed into one VAO per tile
- Alpha+shadow per-chunk as TexImage3D slices

## 3. File Mapping for Implementation

| New File | Purpose | Reference |
|----------|---------|-----------|
| HeadlessContext.cs | Hidden Silk.NET window + GL context | MdxViewer ViewerApp.cs window creation |
| RenderSurface.cs | FBO with color/depth attachments | MdxViewer ViewerApp.cs framebuffer |
| SceneCamera.cs | Camera math | MdxViewer Rendering/Camera.cs |
| TerrainMeshBuilder.cs | WorldTerrainTileData → GL mesh | MdxViewer Terrain/TerrainTileMeshBuilder.cs |
| TerrainShader.cs | GLSL vertex+fragment (tile path) | MdxViewer Terrain/TerrainRenderer.cs L1589-1777 |
| TextureCache.cs | BLP→GL texture + array | MdxViewer Terrain/TerrainRenderer.cs L918-976 |
| FrameCapture.cs | Framebuffer readback | MdxViewer ViewerApp.cs capture code |
