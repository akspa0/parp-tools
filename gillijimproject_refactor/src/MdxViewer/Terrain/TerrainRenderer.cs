using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using SereniaBLPLib;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders terrain chunks using single-pass texture layering.
/// Up to 4 layers (base + 3 overlays) are blended in the fragment shader using MCAL alpha maps.
/// </summary>
public class TerrainRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ShaderProgram _shader;
    private readonly ShaderProgram _tileShader;
    private readonly IDataSource? _dataSource;
    private readonly TerrainLighting _lighting;

    private readonly int _uUseWorldUvLoc;
    private readonly int _uTileUseWorldUvLoc;

    private readonly int _uAlphaDebugChannelLoc;
    private readonly int _uTileAlphaDebugChannelLoc;

    private readonly int _uTileDiffuseLayerCountLoc;

    private readonly int[] _uHasTexLoc = new int[4];
    private readonly int[] _uHasAlphaLoc = new int[4]; // 1..3
    private readonly int _uHasShadowLoc;

    // Track GL state to avoid redundant driver calls (big win when drawing hundreds/thousands of chunks).
    private readonly uint[] _boundTexture2DByUnit = new uint[8]; // units 0..7
    private int _activeTextureUnit = -1;

    // Optional resolver: given a texture name (e.g. "Tileset\Ashenvale\..."), return
    // a local file path to a PNG. Used by VLM projects that export textures as PNGs.
    private readonly Func<string, string?>? _texturePathResolver;

    // Texture cache: texture name → GL handle
    private readonly Dictionary<string, uint> _textureCache = new(StringComparer.OrdinalIgnoreCase);

    // Loaded chunk meshes
    private readonly List<TerrainChunkMesh> _chunks = new();

    // Loaded tile meshes (batched)
    private readonly List<TerrainTileMesh> _tiles = new();
    private readonly Dictionary<(int, int), TerrainTileMesh> _tileMap = new();
    private readonly Dictionary<(int, int), List<TerrainChunkInfo>> _chunkInfosByTile = new();
    private readonly Dictionary<(int tileX, int tileY, int chunkX, int chunkY), TerrainChunkInfo> _chunkInfoByKey = new();
    private readonly Dictionary<(int tileX, int tileY, int chunkX, int chunkY), TerrainChunkMesh> _chunkMeshByKey = new();
    private int _loadedTileChunkCount;

    // Tile texture name tables
    private readonly Dictionary<(int, int), List<string>> _tileTextures = new();

    private bool _wireframe;

    // Layer visibility toggles (exposed for UI)
    public bool ShowLayer0 { get; set; } = true;
    public bool ShowLayer1 { get; set; } = true;
    public bool ShowLayer2 { get; set; } = true;
    public bool ShowLayer3 { get; set; } = true;

    // Diffuse UV mode: world-space (seamless across chunks) vs per-chunk local UV.
    // Alpha (0.5.3) expects the historical per-chunk mapping.
    public bool UseWorldUvForDiffuse { get; set; } = true;

    // Grid overlay
    public bool ShowChunkGrid { get; set; } = false;
    public bool ShowTileGrid { get; set; } = false;

    // Debug: show alpha masks as grayscale on white (no diffuse texture)
    public bool ShowAlphaMask { get; set; } = false;

    // When ShowAlphaMask is enabled, select which overlay alpha channel to visualize.
    // 0 = legacy behavior (first enabled overlay via L1/L2/L3 toggles), 1..3 = Alpha1..Alpha3.
    public int AlphaMaskChannel { get; set; } = 1;

    // MCSH shadow map overlay
    public bool ShowShadowMap { get; set; } = false;

    // Alpha/shadow sampling: WoW-like linear by default; optional nearest for crisper mask edges.
    private bool _useNearestForAlphaSampling;
    private bool _alphaSamplingDirty = true;
    public bool UseNearestForAlphaSampling
    {
        get => _useNearestForAlphaSampling;
        set
        {
            if (_useNearestForAlphaSampling == value) return;
            _useNearestForAlphaSampling = value;
            _alphaSamplingDirty = true;
        }
    }

    // Topographical contour lines
    public bool ShowContours { get; set; } = false;
    public float ContourInterval { get; set; } = 2.0f;

    public int LoadedChunkCount => _loadedTileChunkCount > 0 ? _loadedTileChunkCount : _chunks.Count;
    public TerrainLighting Lighting => _lighting;

    /// <summary>
    /// Find the chunk mesh closest to the given world XY position (for area lookup).
    /// Returns null if no chunks are loaded.
    /// </summary>
    public TerrainChunkMesh? GetChunkAt(float worldX, float worldY)
    {
        if (TryGetChunkKey(worldX, worldY, out var key) && _chunkMeshByKey.TryGetValue(key, out var exact))
            return exact;

        // Robust fallback: even if chunk indices are mirrored/swapped (or float rounding hits an edge),
        // select the chunk whose bounds contain the point. This is cheap (≤ 9*256 checks) and
        // matches the terrain's actual placement in renderer coordinates.
        if (TryGetTileKey(worldX, worldY, out var tileKey) && TryFindChunkMeshByBounds(tileKey, worldX, worldY, out var byBounds))
            return byBounds;

        TerrainChunkMesh? best = null;
        float bestDist = float.MaxValue;
        foreach (var chunk in _chunks)
        {
            var center = (chunk.BoundsMin + chunk.BoundsMax) * 0.5f;
            float dx = center.X - worldX;
            float dy = center.Y - worldY;
            float dist = dx * dx + dy * dy;
            if (dist < bestDist)
            {
                bestDist = dist;
                best = chunk;
            }
        }
        return best;
    }

    public TerrainChunkInfo? GetChunkInfoAt(float worldX, float worldY)
    {
        if (TryGetChunkKey(worldX, worldY, out var key) && _chunkInfoByKey.TryGetValue(key, out var exact))
            return exact;

        // Robust fallback: search for the chunk whose bounds contain the point within the
        // computed tile and immediate neighbors.
        if (TryGetTileKey(worldX, worldY, out var tileKey) && TryFindChunkInfoByBounds(tileKey, worldX, worldY, out var byBounds))
            return byBounds;

        TerrainChunkInfo? best = null;
        float bestDist = float.MaxValue;
        foreach (var tileInfos in _chunkInfosByTile.Values)
        {
            foreach (var chunk in tileInfos)
            {
                var center = (chunk.BoundsMin + chunk.BoundsMax) * 0.5f;
                float dx = center.X - worldX;
                float dy = center.Y - worldY;
                float dist = dx * dx + dy * dy;
                if (dist < bestDist)
                {
                    bestDist = dist;
                    best = chunk;
                }
            }
        }
        return best;
    }

    public bool TryGetChunkInfo(int tileX, int tileY, int chunkX, int chunkY, out TerrainChunkInfo info)
    {
        return _chunkInfoByKey.TryGetValue((tileX, tileY, chunkX, chunkY), out info);
    }

    private static bool TryGetTileKey(float worldX, float worldY, out (int tileX, int tileY) tileKey)
    {
        tileKey = default;

        float dx = WoWConstants.MapOrigin - worldX;
        float dy = WoWConstants.MapOrigin - worldY;

        if (float.IsNaN(dx) || float.IsNaN(dy) || float.IsInfinity(dx) || float.IsInfinity(dy))
            return false;

        int tileX = (int)MathF.Floor(dx / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor(dy / WoWConstants.ChunkSize);
        if (tileX < 0 || tileX >= 64 || tileY < 0 || tileY >= 64)
            return false;

        tileKey = (tileX, tileY);
        return true;
    }

    private static bool ContainsXY(Vector3 min, Vector3 max, float x, float y)
    {
        // Inclusive edges with a tiny epsilon to reduce boundary flicker.
        const float eps = 1e-4f;
        float minX = MathF.Min(min.X, max.X) - eps;
        float maxX = MathF.Max(min.X, max.X) + eps;
        float minY = MathF.Min(min.Y, max.Y) - eps;
        float maxY = MathF.Max(min.Y, max.Y) + eps;
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }

    private bool TryFindChunkInfoByBounds((int tileX, int tileY) tileKey, float worldX, float worldY, out TerrainChunkInfo info)
    {
        info = default;

        for (int ox = -1; ox <= 1; ox++)
        {
            int tx = tileKey.tileX + ox;
            if (tx < 0 || tx >= 64) continue;

            for (int oy = -1; oy <= 1; oy++)
            {
                int ty = tileKey.tileY + oy;
                if (ty < 0 || ty >= 64) continue;

                if (!_chunkInfosByTile.TryGetValue((tx, ty), out var infos))
                    continue;

                foreach (var ci in infos)
                {
                    if (ContainsXY(ci.BoundsMin, ci.BoundsMax, worldX, worldY))
                    {
                        info = ci;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private bool TryFindChunkMeshByBounds((int tileX, int tileY) tileKey, float worldX, float worldY, out TerrainChunkMesh mesh)
    {
        mesh = default!;

        // We don't currently keep a per-tile list for non-batched meshes. Filter by tile and test bounds.
        for (int ox = -1; ox <= 1; ox++)
        {
            int tx = tileKey.tileX + ox;
            if (tx < 0 || tx >= 64) continue;

            for (int oy = -1; oy <= 1; oy++)
            {
                int ty = tileKey.tileY + oy;
                if (ty < 0 || ty >= 64) continue;

                foreach (var c in _chunks)
                {
                    if (c.TileX != tx || c.TileY != ty)
                        continue;

                    if (ContainsXY(c.BoundsMin, c.BoundsMax, worldX, worldY))
                    {
                        mesh = c;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private static bool TryGetChunkKey(float worldX, float worldY, out (int tileX, int tileY, int chunkX, int chunkY) key)
    {
        key = default;

        // Renderer/world coords are defined such that:
        //   wowX = MapOrigin - worldY
        //   wowY = MapOrigin - worldX
        // and terrain chunk origin is:
        //   worldX = MapOrigin - tileX*ChunkSize - chunkY*(ChunkSize/16)
        //   worldY = MapOrigin - tileY*ChunkSize - chunkX*(ChunkSize/16)
        float dx = WoWConstants.MapOrigin - worldX;
        float dy = WoWConstants.MapOrigin - worldY;

        if (float.IsNaN(dx) || float.IsNaN(dy) || float.IsInfinity(dx) || float.IsInfinity(dy))
            return false;

        int tileX = (int)MathF.Floor(dx / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor(dy / WoWConstants.ChunkSize);
        if (tileX < 0 || tileX >= 64 || tileY < 0 || tileY >= 64)
            return false;

        float localX = dx - tileX * WoWConstants.ChunkSize;
        float localY = dy - tileY * WoWConstants.ChunkSize;

        float chunkSmall = WoWConstants.ChunkSize / 16f;
        int chunkY = (int)MathF.Floor(localX / chunkSmall);
        int chunkX = (int)MathF.Floor(localY / chunkSmall);

        // Boundary clamp (can happen when exactly on tile edge due to float precision)
        chunkX = Math.Clamp(chunkX, 0, 15);
        chunkY = Math.Clamp(chunkY, 0, 15);

        key = (tileX, tileY, chunkX, chunkY);
        return true;
    }

    public TerrainRenderer(GL gl, IDataSource? dataSource, TerrainLighting lighting,
        Func<string, string?>? texturePathResolver = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _lighting = lighting;
        _texturePathResolver = texturePathResolver;
        _shader = CreateTerrainShader();
        _tileShader = CreateTileTerrainShader();
        _uTileDiffuseLayerCountLoc = _tileShader.GetUniformLocation("uDiffuseLayerCount");

        _uUseWorldUvLoc = _shader.GetUniformLocation("uUseWorldUV");
        _uTileUseWorldUvLoc = _tileShader.GetUniformLocation("uUseWorldUV");

        _uAlphaDebugChannelLoc = _shader.GetUniformLocation("uAlphaDebugChannel");
        _uTileAlphaDebugChannelLoc = _tileShader.GetUniformLocation("uAlphaDebugChannel");

        _uHasShadowLoc = _shader.GetUniformLocation("uHasShadowMap");
        for (int i = 0; i < 4; i++)
            _uHasTexLoc[i] = _shader.GetUniformLocation($"uHasTex{i}");
        for (int i = 1; i < 4; i++)
            _uHasAlphaLoc[i] = _shader.GetUniformLocation($"uHasAlpha{i}");

        InitializeSamplerUniforms();
    }

    /// <summary>
    /// Add chunk meshes and their texture tables for rendering.
    /// </summary>
    public void AddChunks(List<TerrainChunkMesh> chunks, IDictionary<(int, int), List<string>> tileTextures)
    {
        foreach (var chunk in chunks)
        {
            chunk.Gl = _gl;
            UploadAlphaTextures(chunk);
            UploadShadowTexture(chunk);
        }
        _chunks.AddRange(chunks);

        foreach (var chunk in chunks)
            _chunkMeshByKey[(chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY)] = chunk;

        foreach (var kvp in tileTextures)
        {
            _tileTextures[kvp.Key] = kvp.Value;
            // Pre-load textures for this tile
            foreach (var texName in kvp.Value)
                GetOrLoadTexture(texName);
        }

        ViewerLog.Trace($"[TerrainRenderer] Now rendering {_chunks.Count} chunks, {_textureCache.Count} textures cached");
    }

    /// <summary>
    /// Add a batched tile mesh. Caller owns the mesh lifetime and should Dispose() it after removal.
    /// </summary>
    public void AddTile(TerrainTileMesh tileMesh, List<string> tileTextureNames, List<TerrainChunkInfo> chunkInfos)
    {
        var key = (tileMesh.TileX, tileMesh.TileY);
        if (_tileMap.ContainsKey(key))
            return;

        tileMesh.Gl = _gl;
        CreateDiffuseArrayTexture(tileMesh, tileTextureNames);

        _tiles.Add(tileMesh);
        _tileMap[key] = tileMesh;
        _chunkInfosByTile[key] = chunkInfos;
        foreach (var ci in chunkInfos)
            _chunkInfoByKey[(ci.TileX, ci.TileY, ci.ChunkX, ci.ChunkY)] = ci;
        _loadedTileChunkCount += chunkInfos.Count;

        ViewerLog.Trace($"[TerrainRenderer] Now rendering {_tiles.Count} batched tiles ({_loadedTileChunkCount} chunks)");
    }

    /// <summary>
    /// Remove a batched tile mesh from the render list. The mesh itself is disposed by the caller.
    /// </summary>
    public void RemoveTile(int tileX, int tileY)
    {
        var key = (tileX, tileY);
        if (_tileMap.TryGetValue(key, out var mesh))
        {
            _tiles.Remove(mesh);
            _tileMap.Remove(key);
        }

        if (_chunkInfosByTile.TryGetValue(key, out var infos))
        {
            _loadedTileChunkCount -= infos.Count;
            foreach (var ci in infos)
                _chunkInfoByKey.Remove((ci.TileX, ci.TileY, ci.ChunkX, ci.ChunkY));
            _chunkInfosByTile.Remove(key);
        }
    }

    /// <summary>
    /// Remove specific chunk meshes from the render list (called when tiles unload).
    /// The meshes themselves are disposed by TerrainManager.
    /// </summary>
    public void RemoveChunks(List<TerrainChunkMesh> chunks)
    {
        foreach (var chunk in chunks)
        {
            _chunks.Remove(chunk);
            _chunkMeshByKey.Remove((chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY));
        }
    }

    // Culling stats (updated each frame)
    public int ChunksRendered { get; private set; }
    public int ChunksCulled { get; private set; }

    // Per-frame perf counters (updated each Render call)
    public int LastFrameDrawCalls { get; private set; }
    public int LastFrameUniform1Calls { get; private set; }
    public int LastFrameActiveTextureCalls { get; private set; }
    public int LastFrameActiveTextureSkips { get; private set; }
    public int LastFrameBindTextureCalls { get; private set; }
    public int LastFrameBindTextureSkips { get; private set; }

    /// <summary>
    /// Render all loaded terrain chunks with optional frustum culling.
    /// </summary>
    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, FrustumCuller? frustum = null)
    {
        if (_tiles.Count == 0 && _chunks.Count == 0) return;

        if (_alphaSamplingDirty)
            ApplyAlphaSamplingMode();

        LastFrameDrawCalls = 0;
        LastFrameUniform1Calls = 0;
        LastFrameActiveTextureCalls = 0;
        LastFrameActiveTextureSkips = 0;
        LastFrameBindTextureCalls = 0;
        LastFrameBindTextureSkips = 0;

        // IMPORTANT: invalidate cached texture-unit bindings each frame.
        // Other renderers (models, UI, minimap) may change GL texture state between frames.
        // If we keep a cross-frame cache, we can incorrectly skip binds and sample random textures.
        _activeTextureUnit = -1;
        Array.Clear(_boundTexture2DByUnit, 0, _boundTexture2DByUnit.Length);

        _lighting.Update();

        // Choose render path
        if (_tiles.Count > 0)
        {
            RenderTiles(view, proj, cameraPos, frustum);
            return;
        }

        _shader.Use();
        _shader.SetMat4("uView", view);
        _shader.SetMat4("uProj", proj);
        _shader.SetMat4("uModel", Matrix4x4.Identity);
        _shader.SetVec3("uLightDir", _lighting.LightDirection);
        _shader.SetVec3("uLightColor", _lighting.LightColor);
        _shader.SetVec3("uAmbientColor", _lighting.AmbientColor);
        _shader.SetVec3("uFogColor", _lighting.FogColor);
        _shader.SetFloat("uFogStart", _lighting.FogStart);
        _shader.SetFloat("uFogEnd", _lighting.FogEnd);
        _shader.SetVec3("uCameraPos", cameraPos);

        // Disable face culling for terrain (winding varies across terrain adapters/versions)
        _gl.Disable(EnableCap.CullFace);

        if (_wireframe)
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);
        else
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // Pass grid uniforms
        _shader.SetInt("uShowChunkGrid", ShowChunkGrid ? 1 : 0);
        _shader.SetInt("uShowTileGrid", ShowTileGrid ? 1 : 0);
        _shader.SetInt("uShowAlphaMask", ShowAlphaMask ? 1 : 0);
        _shader.SetInt("uShowShadowMap", ShowShadowMap ? 1 : 0);
        _shader.SetInt("uShowContours", ShowContours ? 1 : 0);
        _shader.SetFloat("uContourInterval", ContourInterval);

        Uniform1Counted(_uUseWorldUvLoc, UseWorldUvForDiffuse ? 1 : 0);
        Uniform1Counted(_uAlphaDebugChannelLoc, Math.Clamp(AlphaMaskChannel, 0, 3));

        // Layer visibility toggles
        _shader.SetInt("uShowLayer0", ShowLayer0 ? 1 : 0);
        _shader.SetInt("uShowLayer1", ShowLayer1 ? 1 : 0);
        _shader.SetInt("uShowLayer2", ShowLayer2 ? 1 : 0);
        _shader.SetInt("uShowLayer3", ShowLayer3 ? 1 : 0);

        // Render each chunk (with frustum + distance culling)
        // Chunks fully beyond fog are invisible — skip them to save GPU work
        float chunkCullDist = _lighting.FogEnd + 200f; // Small buffer past fog end
        float chunkCullDistSq = chunkCullDist * chunkCullDist;
        ChunksRendered = 0;
        ChunksCulled = 0;
        foreach (var chunk in _chunks)
        {
            // Distance cull: skip chunks entirely beyond fog
            var chunkCenter = (chunk.BoundsMin + chunk.BoundsMax) * 0.5f;
            float distSq = Vector3.DistanceSquared(cameraPos, chunkCenter);
            if (distSq > chunkCullDistSq)
            { ChunksCulled++; continue; }
            if (frustum != null && !frustum.TestAABB(chunk.BoundsMin, chunk.BoundsMax))
            { ChunksCulled++; continue; }
            RenderChunk(chunk);
            ChunksRendered++;
        }

        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
        _gl.Enable(EnableCap.CullFace);
    }

    private void ApplyAlphaSamplingMode()
    {
        _alphaSamplingDirty = false;
        int filter = _useNearestForAlphaSampling ? (int)TextureMinFilter.Nearest : (int)TextureMinFilter.Linear;
        int magFilter = _useNearestForAlphaSampling ? (int)TextureMagFilter.Nearest : (int)TextureMagFilter.Linear;

        // Per-chunk path
        foreach (var chunk in _chunks)
        {
            foreach (var tex in chunk.AlphaTextures.Values)
            {
                if (tex == 0) continue;
                _gl.BindTexture(TextureTarget.Texture2D, tex);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, filter);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, magFilter);
            }
            if (chunk.ShadowTexture != 0)
            {
                _gl.BindTexture(TextureTarget.Texture2D, chunk.ShadowTexture);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, filter);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, magFilter);
            }
        }

        // Batched tile path
        foreach (var tile in _tiles)
        {
            if (tile.AlphaShadowArrayTexture == 0) continue;
            _gl.BindTexture(TextureTarget.Texture2DArray, tile.AlphaShadowArrayTexture);
            _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, filter);
            _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, magFilter);
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);
    }

    public unsafe void ReplaceTileAlphaShadowArray(int tileX, int tileY, byte[] alphaShadowRgba)
    {
        if (!_tileMap.TryGetValue((tileX, tileY), out var tile))
            return;
        if (tile.AlphaShadowArrayTexture == 0)
            return;
        if (alphaShadowRgba == null || alphaShadowRgba.Length < 64 * 64 * 4 * 256)
            return;

        _gl.BindTexture(TextureTarget.Texture2DArray, tile.AlphaShadowArrayTexture);
        fixed (byte* ptr = alphaShadowRgba)
        {
            _gl.TexSubImage3D(TextureTarget.Texture2DArray, 0,
                0, 0, 0,
                64, 64, 256,
                PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);

        // Keep sampling mode consistent if user toggled it.
        _alphaSamplingDirty = true;
    }

    private unsafe void RenderTiles(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, FrustumCuller? frustum)
    {
        _tileShader.Use();
        _tileShader.SetMat4("uView", view);
        _tileShader.SetMat4("uProj", proj);
        _tileShader.SetMat4("uModel", Matrix4x4.Identity);
        _tileShader.SetVec3("uLightDir", _lighting.LightDirection);
        _tileShader.SetVec3("uLightColor", _lighting.LightColor);
        _tileShader.SetVec3("uAmbientColor", _lighting.AmbientColor);
        _tileShader.SetVec3("uFogColor", _lighting.FogColor);
        _tileShader.SetFloat("uFogStart", _lighting.FogStart);
        _tileShader.SetFloat("uFogEnd", _lighting.FogEnd);
        _tileShader.SetVec3("uCameraPos", cameraPos);

        _gl.Disable(EnableCap.CullFace);
        if (_wireframe)
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);
        else
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        _tileShader.SetInt("uShowChunkGrid", ShowChunkGrid ? 1 : 0);
        _tileShader.SetInt("uShowTileGrid", ShowTileGrid ? 1 : 0);
        _tileShader.SetInt("uShowAlphaMask", ShowAlphaMask ? 1 : 0);
        _tileShader.SetInt("uShowShadowMap", ShowShadowMap ? 1 : 0);
        _tileShader.SetInt("uShowContours", ShowContours ? 1 : 0);
        _tileShader.SetFloat("uContourInterval", ContourInterval);

        _tileShader.SetInt("uShowLayer0", ShowLayer0 ? 1 : 0);
        _tileShader.SetInt("uShowLayer1", ShowLayer1 ? 1 : 0);
        _tileShader.SetInt("uShowLayer2", ShowLayer2 ? 1 : 0);
        _tileShader.SetInt("uShowLayer3", ShowLayer3 ? 1 : 0);

        Uniform1Counted(_uTileUseWorldUvLoc, UseWorldUvForDiffuse ? 1 : 0);
        Uniform1Counted(_uTileAlphaDebugChannelLoc, Math.Clamp(AlphaMaskChannel, 0, 3));

        float cullDist = _lighting.FogEnd + 200f;
        float cullDistSq = cullDist * cullDist;
        ChunksRendered = 0;
        ChunksCulled = 0;
        LastFrameDrawCalls = 0;

        bool baseVisible = ShowLayer0;
        bool anyOverlayVisible = ShowLayer1 || ShowLayer2 || ShowLayer3;
        bool blendForOverlaysOnly = !baseVisible && anyOverlayVisible;

        foreach (var tile in _tiles)
        {
            var center = (tile.BoundsMin + tile.BoundsMax) * 0.5f;
            float distSq = Vector3.DistanceSquared(cameraPos, center);
            if (distSq > cullDistSq)
            { ChunksCulled += tile.ChunkCount; continue; }
            if (frustum != null && !frustum.TestAABB(tile.BoundsMin, tile.BoundsMax))
            { ChunksCulled += tile.ChunkCount; continue; }

            if (blendForOverlaysOnly)
            {
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                _gl.DepthMask(false);
            }
            else
            {
                _gl.Disable(EnableCap.Blend);
                _gl.DepthMask(true);
            }
            _gl.DepthFunc(DepthFunction.Lequal);

            BindTextureArray(0, tile.DiffuseArrayTexture);
            BindTextureArray(1, tile.AlphaShadowArrayTexture);
            _gl.Uniform1(_uTileDiffuseLayerCountLoc, tile.DiffuseLayerCount);
            LastFrameUniform1Calls++;

            _gl.BindVertexArray(tile.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, tile.IndexCount, DrawElementsType.UnsignedShort, null);
            _gl.BindVertexArray(0);

            LastFrameDrawCalls++;
            ChunksRendered += tile.ChunkCount;
        }

        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
        _gl.Enable(EnableCap.CullFace);
    }

    private void Uniform1Counted(int location, int value)
    {
        _gl.Uniform1(location, value);
        LastFrameUniform1Calls++;
    }

    private unsafe void RenderChunk(TerrainChunkMesh chunk)
    {
        var texNames = _tileTextures.GetValueOrDefault((chunk.TileX, chunk.TileY));
        if (texNames == null || texNames.Count == 0 || chunk.Layers.Length == 0)
        {
            // No textures — render with flat color
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
            _gl.DepthFunc(DepthFunction.Lequal);
            Uniform1Counted(_uHasTexLoc[0], 0);
            Uniform1Counted(_uHasTexLoc[1], 0);
            Uniform1Counted(_uHasTexLoc[2], 0);
            Uniform1Counted(_uHasTexLoc[3], 0);
            Uniform1Counted(_uHasAlphaLoc[1], 0);
            Uniform1Counted(_uHasAlphaLoc[2], 0);
            Uniform1Counted(_uHasAlphaLoc[3], 0);

            bool hasShadow = chunk.ShadowTexture != 0;
            Uniform1Counted(_uHasShadowLoc, hasShadow ? 1 : 0);
            if (hasShadow)
                BindTexture2D(7, chunk.ShadowTexture);
            _gl.BindVertexArray(chunk.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, chunk.IndexCount, DrawElementsType.UnsignedShort, null);
            LastFrameDrawCalls++;
            _gl.BindVertexArray(0);
            return;
        }

        bool baseVisible = ShowLayer0;
        bool anyOverlayVisible = (ShowLayer1 && chunk.Layers.Length > 1)
            || (ShowLayer2 && chunk.Layers.Length > 2)
            || (ShowLayer3 && chunk.Layers.Length > 3);

        // Preserve prior debug behavior: if base is hidden, render overlays with blending and no depth writes.
        if (!baseVisible && anyOverlayVisible)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.DepthMask(false);
        }
        else
        {
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
        }
        _gl.DepthFunc(DepthFunction.Lequal);

        BindChunkLayerTextures(chunk, texNames);

        // Draw once
        _gl.BindVertexArray(chunk.Vao);
        _gl.DrawElements(PrimitiveType.Triangles, chunk.IndexCount, DrawElementsType.UnsignedShort, null);
        LastFrameDrawCalls++;
        _gl.BindVertexArray(0);

        // Restore defaults
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
    }

    private void InitializeSamplerUniforms()
    {
        // Bind sampler uniforms to fixed texture units once.
        // Units:
        // 0-3 = diffuse layers 0..3
        // 4-6 = alpha maps for layers 1..3
        // 7   = shadow map
        _shader.Use();
        _shader.SetInt("uDiffuse0", 0);
        _shader.SetInt("uDiffuse1", 1);
        _shader.SetInt("uDiffuse2", 2);
        _shader.SetInt("uDiffuse3", 3);
        _shader.SetInt("uAlpha1", 4);
        _shader.SetInt("uAlpha2", 5);
        _shader.SetInt("uAlpha3", 6);
        _shader.SetInt("uShadowSampler", 7);
    }

    private void SetActiveTextureUnit(int unit)
    {
        if (_activeTextureUnit == unit)
        {
            LastFrameActiveTextureSkips++;
            return;
        }
        _gl.ActiveTexture((TextureUnit)((int)TextureUnit.Texture0 + unit));
        _activeTextureUnit = unit;
        LastFrameActiveTextureCalls++;
    }

    private void BindTexture2D(int unit, uint texture)
    {
        if ((uint)unit >= (uint)_boundTexture2DByUnit.Length) return;
        if (_boundTexture2DByUnit[unit] == texture)
        {
            LastFrameBindTextureSkips++;
            return;
        }
        SetActiveTextureUnit(unit);
        _gl.BindTexture(TextureTarget.Texture2D, texture);
        _boundTexture2DByUnit[unit] = texture;
        LastFrameBindTextureCalls++;
    }

    private void BindTextureArray(int unit, uint texture)
    {
        if ((uint)unit >= (uint)_boundTexture2DByUnit.Length) return;
        if (_boundTexture2DByUnit[unit] == texture)
        {
            LastFrameBindTextureSkips++;
            return;
        }
        SetActiveTextureUnit(unit);
        _gl.BindTexture(TextureTarget.Texture2DArray, texture);
        _boundTexture2DByUnit[unit] = texture;
        LastFrameBindTextureCalls++;
    }

    private unsafe void CreateDiffuseArrayTexture(TerrainTileMesh tileMesh, List<string> textureNames)
    {
        int maxLayers = 256;
        try { maxLayers = _gl.GetInteger(GetPName.MaxArrayTextureLayers); } catch { }

        int requestedLayers = textureNames.Count;
        int layerCount = Math.Max(1, Math.Min(requestedLayers, maxLayers));

        // Determine a target size based on available textures (cap at 256 to keep memory reasonable).
        int maxDim = 0;
        for (int i = 0; i < Math.Min(textureNames.Count, layerCount); i++)
        {
            if (TryLoadTerrainTexturePixels(textureNames[i], out int w, out int h, out _))
                maxDim = Math.Max(maxDim, Math.Max(w, h));
        }
        int targetDim = maxDim switch
        {
            <= 0 => 256,
            <= 64 => 64,
            <= 128 => 128,
            <= 256 => 256,
            _ => 256
        };

        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, tex);
        _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, (uint)targetDim, (uint)targetDim, (uint)layerCount, 0,
            PixelFormat.Rgba, PixelType.UnsignedByte, (void*)0);

        for (int layer = 0; layer < layerCount; layer++)
        {
            byte[] pixels;
            int w, h;
            if (layer < textureNames.Count && TryLoadTerrainTexturePixels(textureNames[layer], out w, out h, out pixels))
            {
                if (w != targetDim || h != targetDim)
                    pixels = ResampleRgbaNearest(pixels, w, h, targetDim, targetDim);
            }
            else
            {
                pixels = CreateSolidRgba(targetDim, targetDim, 255, 255, 255, 255);
            }

            fixed (byte* ptr = pixels)
            {
                _gl.TexSubImage3D(TextureTarget.Texture2DArray, 0, 0, 0, layer, (uint)targetDim, (uint)targetDim, 1,
                    PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
        _gl.GenerateMipmap(TextureTarget.Texture2DArray);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);

        tileMesh.DiffuseArrayTexture = tex;
        tileMesh.DiffuseLayerCount = layerCount;
    }

    private static byte[] CreateSolidRgba(int w, int h, byte r, byte g, byte b, byte a)
    {
        var data = new byte[w * h * 4];
        for (int i = 0; i < data.Length; i += 4)
        {
            data[i + 0] = r;
            data[i + 1] = g;
            data[i + 2] = b;
            data[i + 3] = a;
        }
        return data;
    }

    private static byte[] ResampleRgbaNearest(byte[] src, int srcW, int srcH, int dstW, int dstH)
    {
        var dst = new byte[dstW * dstH * 4];
        for (int y = 0; y < dstH; y++)
        {
            int sy = (int)((long)y * srcH / dstH);
            for (int x = 0; x < dstW; x++)
            {
                int sx = (int)((long)x * srcW / dstW);
                int s = (sy * srcW + sx) * 4;
                int d = (y * dstW + x) * 4;
                dst[d + 0] = src[s + 0];
                dst[d + 1] = src[s + 1];
                dst[d + 2] = src[s + 2];
                dst[d + 3] = src[s + 3];
            }
        }
        return dst;
    }

    private bool TryLoadTerrainTexturePixels(string textureName, out int w, out int h, out byte[] pixels)
    {
        w = 0;
        h = 0;
        pixels = Array.Empty<byte>();
        if (string.IsNullOrEmpty(textureName)) return false;

        byte[]? blpData = null;
        if (_dataSource != null)
        {
            blpData = _dataSource.ReadFile(textureName);
            if (blpData == null)
                blpData = _dataSource.ReadFile(textureName.Replace('/', '\\'));
        }

        if (blpData == null || blpData.Length == 0)
        {
            if (_texturePathResolver != null)
            {
                var pngPath = _texturePathResolver(textureName);
                if (pngPath != null)
                    return TryLoadPngPixels(pngPath, out w, out h, out pixels);
            }
            return false;
        }

        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new BlpFile(ms);
            using var bmp = blp.GetBitmap(0);
            w = bmp.Width;
            h = bmp.Height;

            pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var srcBytes = new byte[data.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, srcBytes, 0, srcBytes.Length);
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = y * data.Stride + x * 4;
                        int dstIdx = (y * w + x) * 4;
                        pixels[dstIdx + 0] = srcBytes[srcIdx + 2];
                        pixels[dstIdx + 1] = srcBytes[srcIdx + 1];
                        pixels[dstIdx + 2] = srcBytes[srcIdx + 0];
                        pixels[dstIdx + 3] = srcBytes[srcIdx + 3];
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(data);
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryLoadPngPixels(string pngPath, out int w, out int h, out byte[] pixels)
    {
        w = 0;
        h = 0;
        pixels = Array.Empty<byte>();

        try
        {
            using var bmp = new System.Drawing.Bitmap(pngPath);
            w = bmp.Width;
            h = bmp.Height;
            pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var srcBytes = new byte[data.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, srcBytes, 0, srcBytes.Length);
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = y * data.Stride + x * 4;
                        int dstIdx = (y * w + x) * 4;
                        pixels[dstIdx + 0] = srcBytes[srcIdx + 2];
                        pixels[dstIdx + 1] = srcBytes[srcIdx + 1];
                        pixels[dstIdx + 2] = srcBytes[srcIdx + 0];
                        pixels[dstIdx + 3] = srcBytes[srcIdx + 3];
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(data);
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private void BindChunkLayerTextures(TerrainChunkMesh chunk, List<string> texNames)
    {
        // Diffuse textures 0..3
        for (int layer = 0; layer < 4; layer++)
        {
            bool hasTex = false;
            if (layer < chunk.Layers.Length)
            {
                int texIdx = chunk.Layers[layer].TextureIndex;
                if (texIdx >= 0 && texIdx < texNames.Count)
                {
                    uint glTex = GetOrLoadTexture(texNames[texIdx]);
                    if (glTex != 0)
                    {
                        BindTexture2D(layer, glTex);
                        hasTex = true;
                    }
                }
            }
            Uniform1Counted(_uHasTexLoc[layer], hasTex ? 1 : 0);
        }

        // Alpha maps for overlay layers 1..3
        for (int layer = 1; layer < 4; layer++)
        {
            bool hasAlpha = chunk.AlphaTextures.TryGetValue(layer, out uint alphaTex) && alphaTex != 0;
            if (hasAlpha)
            {
                BindTexture2D(3 + layer, alphaTex); // layer 1->unit4, 2->unit5, 3->unit6
            }
            Uniform1Counted(_uHasAlphaLoc[layer], hasAlpha ? 1 : 0);
        }

        // Shadow map
        bool hasShadow = chunk.ShadowTexture != 0;
        if (hasShadow)
        {
            BindTexture2D(7, chunk.ShadowTexture);
        }
        Uniform1Counted(_uHasShadowLoc, hasShadow ? 1 : 0);
    }

    public void ToggleWireframe()
    {
        _wireframe = !_wireframe;
    }

    private uint GetOrLoadTexture(string textureName)
    {
        if (string.IsNullOrEmpty(textureName)) return 0;
        if (_textureCache.TryGetValue(textureName, out uint cached)) return cached;

        uint glTex = LoadTerrainTexture(textureName);
        _textureCache[textureName] = glTex;
        return glTex;
    }

    private unsafe uint LoadTerrainTexture(string textureName)
    {
        byte[]? blpData = null;

        // Try data source (MPQ)
        if (_dataSource != null)
        {
            blpData = _dataSource.ReadFile(textureName);
            if (blpData == null)
            {
                // Try with normalized path
                blpData = _dataSource.ReadFile(textureName.Replace('/', '\\'));
            }
        }

        if (blpData == null || blpData.Length == 0)
        {
            // Try PNG from texture path resolver (VLM projects)
            if (_texturePathResolver != null)
            {
                var pngPath = _texturePathResolver(textureName);
                if (pngPath != null)
                    return LoadPngTexture(pngPath);
            }
            return 0;
        }

        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new BlpFile(ms);
            var bmp = blp.GetBitmap(0);

            int w = bmp.Width, h = bmp.Height;
            var pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var srcBytes = new byte[data.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, srcBytes, 0, srcBytes.Length);

                // BGRA → RGBA
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = y * data.Stride + x * 4;
                        int dstIdx = (y * w + x) * 4;
                        pixels[dstIdx + 0] = srcBytes[srcIdx + 2]; // R
                        pixels[dstIdx + 1] = srcBytes[srcIdx + 1]; // G
                        pixels[dstIdx + 2] = srcBytes[srcIdx + 0]; // B
                        pixels[dstIdx + 3] = srcBytes[srcIdx + 3]; // A
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(data);
            }
            bmp.Dispose();

            // Upload to GPU with repeat wrapping (terrain tiles)
            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            return tex;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TerrainRenderer] Failed to load texture {textureName}: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Load a PNG file from disk and upload as a GL texture.
    /// Used by VLM projects where textures are exported as PNGs.
    /// </summary>
    private unsafe uint LoadPngTexture(string pngPath)
    {
        try
        {
            using var bmp = new System.Drawing.Bitmap(pngPath);
            int w = bmp.Width, h = bmp.Height;
            var pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var srcBytes = new byte[data.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, srcBytes, 0, srcBytes.Length);

                // BGRA → RGBA
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = y * data.Stride + x * 4;
                        int dstIdx = (y * w + x) * 4;
                        pixels[dstIdx + 0] = srcBytes[srcIdx + 2]; // R
                        pixels[dstIdx + 1] = srcBytes[srcIdx + 1]; // G
                        pixels[dstIdx + 2] = srcBytes[srcIdx + 0]; // B
                        pixels[dstIdx + 3] = srcBytes[srcIdx + 3]; // A
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(data);
            }

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            return tex;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TerrainRenderer] Failed to load PNG texture {pngPath}: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Upload alpha map byte arrays as GL textures (R8, 64x64).
    /// </summary>
    private unsafe void UploadAlphaTextures(TerrainChunkMesh chunk)
    {
        foreach (var kvp in chunk.AlphaMaps)
        {
            int layer = kvp.Key;
            byte[] alphaData = kvp.Value;

            // Alpha maps are 64x64 (4096 bytes after expansion)
            int size = 64;
            if (alphaData.Length < size * size)
                continue;

            // Noggit fix: duplicate last row/column so edge texels have valid data
            for (int i = 0; i < 64; i++)
            {
                alphaData[i * 64 + 63] = alphaData[i * 64 + 62];
                alphaData[63 * 64 + i] = alphaData[62 * 64 + i];
            }
            alphaData[63 * 64 + 63] = alphaData[62 * 64 + 62];

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);

            fixed (byte* ptr = alphaData)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.R8,
                    (uint)size, (uint)size, 0, PixelFormat.Red, PixelType.UnsignedByte, ptr);

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            chunk.AlphaTextures[layer] = tex;
        }
    }

    /// <summary>
    /// Upload MCSH shadow map as a GL texture (R8, 64x64).
    /// </summary>
    private unsafe void UploadShadowTexture(TerrainChunkMesh chunk)
    {
        if (chunk.ShadowMap == null || chunk.ShadowMap.Length < 64 * 64)
            return;

        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, tex);

        fixed (byte* ptr = chunk.ShadowMap)
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.R8,
                64, 64, 0, PixelFormat.Red, PixelType.UnsignedByte, ptr);

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2D, 0);

        chunk.ShadowTexture = tex;
    }

    private ShaderProgram CreateTerrainShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(uModel) * aNormal;
    vTexCoord = aTexCoord;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

uniform sampler2D uDiffuse0;
uniform sampler2D uDiffuse1;
uniform sampler2D uDiffuse2;
uniform sampler2D uDiffuse3;
uniform sampler2D uAlpha1;
uniform sampler2D uAlpha2;
uniform sampler2D uAlpha3;
uniform sampler2D uShadowSampler;

uniform int uUseWorldUV;
uniform int uAlphaDebugChannel;

uniform int uHasTex0;
uniform int uHasTex1;
uniform int uHasTex2;
uniform int uHasTex3;
uniform int uHasAlpha1;
uniform int uHasAlpha2;
uniform int uHasAlpha3;
uniform int uHasShadowMap;

uniform int uShowLayer0;
uniform int uShowLayer1;
uniform int uShowLayer2;
uniform int uShowLayer3;

uniform int uShowChunkGrid;
uniform int uShowTileGrid;
uniform int uShowAlphaMask;
uniform int uShowShadowMap;
uniform int uShowContours;
uniform float uContourInterval;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;

out vec4 FragColor;

vec4 SampleDiffuse(int hasTex, sampler2D s, vec2 uv)
{
    if (hasTex == 1 && uShowAlphaMask == 0)
        return texture(s, uv);
    return vec4(1.0, 1.0, 1.0, 1.0);
}

void main() {
    // Diffuse texture: use WORLD-SPACE UVs for seamless tiling across chunks.
    // WoW terrain textures tile ~8 times per chunk (chunk = ChunkSize/16 = 33.333 units).
    // So texture repeats every 33.333/8 = ~4.167 units.
    float texScale = 8.0 / 33.333;
    vec2 diffuseUV = (uUseWorldUV == 1) ? (vWorldPos.xy * texScale) : (vTexCoord * 8.0);

    // Sample alpha maps.
    // Alpha maps are 64x64 with Noggit edge fix applied, sampled with ClampToEdge.
    float a1Raw = (uHasAlpha1 == 1) ? texture(uAlpha1, vTexCoord).r : 1.0;
    float a2Raw = (uHasAlpha2 == 1) ? texture(uAlpha2, vTexCoord).r : 1.0;
    float a3Raw = (uHasAlpha3 == 1) ? texture(uAlpha3, vTexCoord).r : 1.0;

    // Apply layer toggles to blending only.
    float a1 = (uShowLayer1 == 1) ? a1Raw : 0.0;
    float a2 = (uShowLayer2 == 1) ? a2Raw : 0.0;
    float a3 = (uShowLayer3 == 1) ? a3Raw : 0.0;

    // Alpha mask debug mode: show the first enabled overlay alpha as grayscale.
    // (Matches prior behavior when users toggled layers to isolate a single overlay pass.)
    if (uShowAlphaMask == 1) {
        float dbg = 1.0;
        if (uAlphaDebugChannel == 1) dbg = a1Raw;
        else if (uAlphaDebugChannel == 2) dbg = a2Raw;
        else if (uAlphaDebugChannel == 3) dbg = a3Raw;
        else {
            // Legacy: use layer toggles to pick the first enabled overlay.
            if (uShowLayer1 == 1) dbg = a1Raw;
            else if (uShowLayer2 == 1) dbg = a2Raw;
            else if (uShowLayer3 == 1) dbg = a3Raw;
        }
        FragColor = vec4(dbg, dbg, dbg, 1.0);
        return;
    }

    // Lighting
    vec3 norm = normalize(vNormal);
    float diff = abs(dot(norm, normalize(uLightDir)));
    vec3 lighting = uAmbientColor + uLightColor * diff;
    vec3 result = vec3(1.0);

    // Base layer
    if (uShowLayer0 == 1) {
        vec4 c0 = SampleDiffuse(uHasTex0, uDiffuse0, diffuseUV);
        result = c0.rgb * lighting;
    }

    // Overlay layers (sequential blend)
    if (uShowLayer1 == 1 && uHasTex1 == 1) {
        vec4 c1 = texture(uDiffuse1, diffuseUV);
        vec3 l1 = c1.rgb * lighting;
        result = mix(result, l1, a1);
    }
    if (uShowLayer2 == 1 && uHasTex2 == 1) {
        vec4 c2 = texture(uDiffuse2, diffuseUV);
        vec3 l2 = c2.rgb * lighting;
        result = mix(result, l2, a2);
    }
    if (uShowLayer3 == 1 && uHasTex3 == 1) {
        vec4 c3 = texture(uDiffuse3, diffuseUV);
        vec3 l3 = c3.rgb * lighting;
        result = mix(result, l3, a3);
    }

    // MCSH shadow map overlay (all layers - shadows must persist through alpha-blended overlays)
    if (uShowShadowMap == 1 && uHasShadowMap == 1) {
        float shadow = texture(uShadowSampler, vTexCoord).r;
        // Darken shadowed areas: shadow=1.0 means shadowed, 0.0 means lit
        result *= mix(1.0, 0.4, shadow);
    }

    // Fog
    float dist = length(vWorldPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 finalColor = mix(uFogColor, result, fogFactor);

    // Topographical contour lines (base visible only)
    if (uShowContours == 1 && uShowLayer0 == 1) {
        float height = vWorldPos.z;
        float interval = max(uContourInterval, 0.5);
        // Use fwidth for screen-space anti-aliased contour lines
        float dh = fwidth(height);
        float modH = mod(height, interval);
        // Very thin line at each contour interval
        float lineWidth = max(dh * 0.8, 0.02);
        float dist = min(modH, interval - modH);
        float contourLine = 1.0 - smoothstep(0.0, lineWidth, dist);
        // Threshold: only draw if clearly on a contour line
        contourLine = contourLine > 0.1 ? contourLine : 0.0;
        // Major contour every 5 intervals - slightly thicker
        float majorInterval = interval * 5.0;
        float modMajor = mod(height, majorInterval);
        float majorLineWidth = max(dh * 1.2, 0.04);
        float majorDist = min(modMajor, majorInterval - modMajor);
        float majorLine = 1.0 - smoothstep(0.0, majorLineWidth, majorDist);
        majorLine = majorLine > 0.1 ? majorLine : 0.0;
        // Color: minor=dark overlay, major=darker overlay
        vec3 minorColor = mix(finalColor, vec3(0.0), 0.4);
        vec3 majorColor = mix(finalColor, vec3(0.0), 0.6);
        finalColor = mix(finalColor, minorColor, contourLine);
        finalColor = mix(finalColor, majorColor, majorLine);
    }

    // Grid overlays (drawn only when base is visible)
    if (uShowLayer0 == 1) {
        // Chunk grid: chunk size = 33.333 units
        if (uShowChunkGrid == 1) {
            float chunkSize = 33.333;
            vec2 chunkFrac = fract(vWorldPos.xy / chunkSize);
            float chunkLine = step(chunkFrac.x, 0.005) + step(1.0 - chunkFrac.x, 0.005)
                            + step(chunkFrac.y, 0.005) + step(1.0 - chunkFrac.y, 0.005);
            chunkLine = clamp(chunkLine, 0.0, 1.0);
            finalColor = mix(finalColor, vec3(0.0, 1.0, 1.0), chunkLine * 0.6);
        }
        // Tile grid: tile size = 533.333 units
        if (uShowTileGrid == 1) {
            float tileSize = 533.333;
            vec2 tileFrac = fract(vWorldPos.xy / tileSize);
            float tileLine = step(tileFrac.x, 0.001) + step(1.0 - tileFrac.x, 0.001)
                           + step(tileFrac.y, 0.001) + step(1.0 - tileFrac.y, 0.001);
            tileLine = clamp(tileLine, 0.0, 1.0);
            finalColor = mix(finalColor, vec3(1.0, 0.3, 0.0), tileLine * 0.8);
        }
    }

    // Output alpha: if base is visible, terrain is opaque.
    // If base is hidden, preserve prior debug behavior by making the chunk alpha depend on overlays.
    float outAlpha = 1.0;
    if (uShowLayer0 == 0) {
        float inv = 1.0;
        if (uShowLayer1 == 1) inv *= (1.0 - a1);
        if (uShowLayer2 == 1) inv *= (1.0 - a2);
        if (uShowLayer3 == 1) inv *= (1.0 - a3);
        outAlpha = 1.0 - inv;
    }

    FragColor = vec4(finalColor, outAlpha);
}
";

        var program = ShaderProgram.Create(_gl, vertSrc, fragSrc);

        // Bind samplers to fixed texture units
        program.Use();
        program.SetInt("uDiffuse0", 0);
        program.SetInt("uDiffuse1", 1);
        program.SetInt("uDiffuse2", 2);
        program.SetInt("uDiffuse3", 3);
        program.SetInt("uAlpha1", 4);
        program.SetInt("uAlpha2", 5);
        program.SetInt("uAlpha3", 6);
        program.SetInt("uShadowSampler", 7);

        return program;
    }

    private ShaderProgram CreateTileTerrainShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in uint aChunkSlice;
layout(location = 4) in uvec4 aTexIdx;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;
flat out uint vChunkSlice;
flat out uvec4 vTexIdx;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(uModel) * aNormal;
    vTexCoord = aTexCoord;
    vChunkSlice = aChunkSlice;
    vTexIdx = aTexIdx;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;
flat in uint vChunkSlice;
flat in uvec4 vTexIdx;

uniform sampler2DArray uDiffuseArray;
uniform sampler2DArray uAlphaShadowArray;
uniform int uDiffuseLayerCount;

uniform int uUseWorldUV;
uniform int uAlphaDebugChannel;

uniform int uShowLayer0;
uniform int uShowLayer1;
uniform int uShowLayer2;
uniform int uShowLayer3;

uniform int uShowChunkGrid;
uniform int uShowTileGrid;
uniform int uShowAlphaMask;
uniform int uShowShadowMap;
uniform int uShowContours;
uniform float uContourInterval;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;

out vec4 FragColor;

bool HasLayer(uint idx) {
    return (idx != 65535u) && (int(idx) >= 0) && (int(idx) < uDiffuseLayerCount);
}

void main() {
    float texScale = 8.0 / 33.333;
    vec2 diffuseUV = (uUseWorldUV == 1) ? (vWorldPos.xy * texScale) : (vTexCoord * 8.0);

    vec4 alphaShadow = texture(uAlphaShadowArray, vec3(vTexCoord, float(vChunkSlice)));
    bool has0 = HasLayer(vTexIdx.x);
    bool has1 = HasLayer(vTexIdx.y);
    bool has2 = HasLayer(vTexIdx.z);
    bool has3 = HasLayer(vTexIdx.w);

    float a1Raw = alphaShadow.r;
    float a2Raw = alphaShadow.g;
    float a3Raw = alphaShadow.b;

    float a1 = (uShowLayer1 == 1 && has1) ? a1Raw : 0.0;
    float a2 = (uShowLayer2 == 1 && has2) ? a2Raw : 0.0;
    float a3 = (uShowLayer3 == 1 && has3) ? a3Raw : 0.0;

    if (uShowAlphaMask == 1) {
        float dbg = 1.0;
        if (uAlphaDebugChannel == 1) dbg = a1Raw;
        else if (uAlphaDebugChannel == 2) dbg = a2Raw;
        else if (uAlphaDebugChannel == 3) dbg = a3Raw;
        else {
            if (uShowLayer1 == 1 && has1) dbg = a1Raw;
            else if (uShowLayer2 == 1 && has2) dbg = a2Raw;
            else if (uShowLayer3 == 1 && has3) dbg = a3Raw;
        }
        FragColor = vec4(dbg, dbg, dbg, 1.0);
        return;
    }

    vec3 norm = normalize(vNormal);
    float diff = abs(dot(norm, normalize(uLightDir)));
    vec3 lighting = uAmbientColor + uLightColor * diff;
    vec3 result = vec3(1.0);

    if (uShowLayer0 == 1 && has0) {
        vec4 c0 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.x)));
        result = c0.rgb * lighting;
    }
    if (uShowLayer1 == 1 && has1) {
        vec4 c1 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.y)));
        result = mix(result, c1.rgb * lighting, a1);
    }
    if (uShowLayer2 == 1 && has2) {
        vec4 c2 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.z)));
        result = mix(result, c2.rgb * lighting, a2);
    }
    if (uShowLayer3 == 1 && has3) {
        vec4 c3 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.w)));
        result = mix(result, c3.rgb * lighting, a3);
    }

    if (uShowShadowMap == 1) {
        float shadow = alphaShadow.a;
        result *= mix(1.0, 0.4, shadow);
    }

    float dist = length(vWorldPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 finalColor = mix(uFogColor, result, fogFactor);

    if (uShowContours == 1 && uShowLayer0 == 1) {
        float height = vWorldPos.z;
        float interval = max(uContourInterval, 0.5);
        float dh = fwidth(height);
        float modH = mod(height, interval);
        float lineWidth = max(dh * 0.8, 0.02);
        float d0 = min(modH, interval - modH);
        float contourLine = 1.0 - smoothstep(0.0, lineWidth, d0);
        contourLine = contourLine > 0.1 ? contourLine : 0.0;
        float majorInterval = interval * 5.0;
        float modMajor = mod(height, majorInterval);
        float majorLineWidth = max(dh * 1.2, 0.04);
        float d1 = min(modMajor, majorInterval - modMajor);
        float majorLine = 1.0 - smoothstep(0.0, majorLineWidth, d1);
        majorLine = majorLine > 0.1 ? majorLine : 0.0;
        vec3 minorColor = mix(finalColor, vec3(0.0), 0.4);
        vec3 majorColor = mix(finalColor, vec3(0.0), 0.6);
        finalColor = mix(finalColor, minorColor, contourLine);
        finalColor = mix(finalColor, majorColor, majorLine);
    }

    if (uShowLayer0 == 1) {
        if (uShowChunkGrid == 1) {
            float chunkSize = 33.333;
            vec2 chunkFrac = fract(vWorldPos.xy / chunkSize);
            float chunkLine = step(chunkFrac.x, 0.005) + step(1.0 - chunkFrac.x, 0.005)
                            + step(chunkFrac.y, 0.005) + step(1.0 - chunkFrac.y, 0.005);
            chunkLine = clamp(chunkLine, 0.0, 1.0);
            finalColor = mix(finalColor, vec3(0.0, 1.0, 1.0), chunkLine * 0.6);
        }
        if (uShowTileGrid == 1) {
            float tileSize = 533.333;
            vec2 tileFrac = fract(vWorldPos.xy / tileSize);
            float tileLine = step(tileFrac.x, 0.001) + step(1.0 - tileFrac.x, 0.001)
                           + step(tileFrac.y, 0.001) + step(1.0 - tileFrac.y, 0.001);
            tileLine = clamp(tileLine, 0.0, 1.0);
            finalColor = mix(finalColor, vec3(1.0, 0.3, 0.0), tileLine * 0.8);
        }
    }

    float outAlpha = 1.0;
    if (uShowLayer0 == 0) {
        float inv = 1.0;
        if (uShowLayer1 == 1 && has1) inv *= (1.0 - a1);
        if (uShowLayer2 == 1 && has2) inv *= (1.0 - a2);
        if (uShowLayer3 == 1 && has3) inv *= (1.0 - a3);
        outAlpha = 1.0 - inv;
    }

    FragColor = vec4(finalColor, outAlpha);
}
";

        var program = ShaderProgram.Create(_gl, vertSrc, fragSrc);
        program.Use();
        program.SetInt("uDiffuseArray", 0);
        program.SetInt("uAlphaShadowArray", 1);
        return program;
    }

    public void Dispose()
    {
        _tiles.Clear();
        _tileMap.Clear();
        _chunkInfosByTile.Clear();
        _loadedTileChunkCount = 0;

        foreach (var chunk in _chunks)
            chunk.Dispose();
        _chunks.Clear();

        foreach (var tex in _textureCache.Values)
        {
            if (tex != 0) _gl.DeleteTexture(tex);
        }
        _textureCache.Clear();

        _shader.Dispose();
        _tileShader.Dispose();
    }
}
