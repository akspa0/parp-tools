using System.Diagnostics;
using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using SereniaBLPLib;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders terrain chunks using a batched tile path when available, while preserving the legacy per-chunk path for consumers that still upload chunk meshes directly.
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
    private readonly int _uUseMccvLoc;
    private readonly int _uTileUseMccvLoc;
    private readonly int _uAlphaDebugChannelLoc;
    private readonly int _uTileAlphaDebugChannelLoc;
    private readonly int _uTileDiffuseLayerCountLoc;
    private readonly int _uTileOpacityLoc;
    private readonly int[] _uHasTexLoc = new int[4];
    private readonly int[] _uHasAlphaLoc = new int[4];
    private readonly int[] _uImplicitAlphaLoc = new int[4];
    private readonly int _uHasShadowLoc;

    private readonly uint[] _boundTexture2DByUnit = new uint[8];
    private int _activeTextureUnit = -1;

    private readonly Func<string, string?>? _texturePathResolver;
    private readonly Dictionary<string, uint> _textureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<TerrainChunkMesh> _chunks = new();
    private readonly List<TerrainTileMesh> _tiles = new();
    private readonly Dictionary<(int, int), TerrainTileMesh> _tileMap = new();
    private readonly Dictionary<(int, int), List<global::MdxViewer.Terrain.TerrainChunkInfo>> _chunkInfosByTile = new();
    private readonly Dictionary<(int tileX, int tileY, int chunkX, int chunkY), global::MdxViewer.Terrain.TerrainChunkInfo> _chunkInfoByKey = new();
    private readonly Dictionary<(int tileX, int tileY, int chunkX, int chunkY), TerrainChunkMesh> _chunkMeshByKey = new();
    private int _loadedTileChunkCount;

    private readonly Dictionary<(int, int), List<string>> _tileTextures = new();
    private readonly Dictionary<(int, int), float> _tileAlphas = new();
    private readonly Dictionary<(int, int), float> _tileTargetAlphas = new();
    private long _lastTileFadeTimestamp;

    private const float TileFadeInDurationSeconds = 0.28f;

    private bool _wireframe;

    public bool ShowLayer0 { get; set; } = true;
    public bool ShowLayer1 { get; set; } = true;
    public bool ShowLayer2 { get; set; } = true;
    public bool ShowLayer3 { get; set; } = true;
    public bool UseWorldUvForDiffuse { get; set; } = true;
    public bool ShowChunkGrid { get; set; }
    public bool ShowTileGrid { get; set; }
    public bool ShowAlphaMask { get; set; }
    public int AlphaMaskChannel { get; set; } = 1;
    public bool ShowShadowMap { get; set; }
    public bool UseMccv { get; set; } = true;

    private bool _useNearestForAlphaSampling;
    private bool _alphaSamplingDirty = true;
    public bool UseNearestForAlphaSampling
    {
        get => _useNearestForAlphaSampling;
        set
        {
            if (_useNearestForAlphaSampling == value)
                return;

            _useNearestForAlphaSampling = value;
            _alphaSamplingDirty = true;
        }
    }

    public bool ShowContours { get; set; }
    public float ContourInterval { get; set; } = 2.0f;

    public int LoadedChunkCount => _loadedTileChunkCount > 0 ? _loadedTileChunkCount : _chunks.Count;
    public TerrainLighting Lighting => _lighting;

    public readonly struct TerrainChunkInfo
    {
        public int TileX { get; }
        public int TileY { get; }
        public int ChunkX { get; }
        public int ChunkY { get; }
        public Vector3 BoundsMin { get; }
        public Vector3 BoundsMax { get; }
        public int AreaId { get; }

        public TerrainChunkInfo(int tileX, int tileY, int chunkX, int chunkY, Vector3 boundsMin, Vector3 boundsMax, int areaId)
        {
            TileX = tileX;
            TileY = tileY;
            ChunkX = chunkX;
            ChunkY = chunkY;
            BoundsMin = boundsMin;
            BoundsMax = boundsMax;
            AreaId = areaId;
        }

        public TerrainChunkInfo(global::MdxViewer.Terrain.TerrainChunkInfo info)
            : this(info.TileX, info.TileY, info.ChunkX, info.ChunkY, info.BoundsMin, info.BoundsMax, info.AreaId)
        {
        }
    }

    public TerrainChunkMesh? GetChunkAt(float worldX, float worldY)
    {
        if (TryGetChunkKey(worldX, worldY, out var key) && _chunkMeshByKey.TryGetValue(key, out var exact))
            return exact;

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
            return new TerrainChunkInfo(exact);

        if (TryGetTileKey(worldX, worldY, out var tileKey) && TryFindChunkInfoByBounds(tileKey, worldX, worldY, out var byBounds))
            return new TerrainChunkInfo(byBounds);

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
                    best = new TerrainChunkInfo(chunk);
                }
            }
        }

        return best;
    }

    public bool TryGetChunkInfo(int tileX, int tileY, int chunkX, int chunkY, out TerrainChunkInfo info)
    {
        if (_chunkInfoByKey.TryGetValue((tileX, tileY, chunkX, chunkY), out var stored))
        {
            info = new TerrainChunkInfo(stored);
            return true;
        }

        info = default;
        return false;
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

    private static bool TryGetChunkKey(float worldX, float worldY, out (int tileX, int tileY, int chunkX, int chunkY) key)
    {
        key = default;

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
        float chunkSize = WoWConstants.ChunkSize / 16f;

        int chunkY = Math.Clamp((int)MathF.Floor(localX / chunkSize), 0, 15);
        int chunkX = Math.Clamp((int)MathF.Floor(localY / chunkSize), 0, 15);

        key = (tileX, tileY, chunkX, chunkY);
        return true;
    }

    private static bool ContainsXY(Vector3 min, Vector3 max, float x, float y)
    {
        const float epsilon = 1e-4f;
        float minX = MathF.Min(min.X, max.X) - epsilon;
        float maxX = MathF.Max(min.X, max.X) + epsilon;
        float minY = MathF.Min(min.Y, max.Y) - epsilon;
        float maxY = MathF.Max(min.Y, max.Y) + epsilon;
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }

    private bool TryFindChunkInfoByBounds((int tileX, int tileY) tileKey, float worldX, float worldY, out global::MdxViewer.Terrain.TerrainChunkInfo info)
    {
        info = default;

        for (int offsetX = -1; offsetX <= 1; offsetX++)
        {
            int tx = tileKey.tileX + offsetX;
            if (tx < 0 || tx >= 64)
                continue;

            for (int offsetY = -1; offsetY <= 1; offsetY++)
            {
                int ty = tileKey.tileY + offsetY;
                if (ty < 0 || ty >= 64)
                    continue;

                if (!_chunkInfosByTile.TryGetValue((tx, ty), out var infos))
                    continue;

                foreach (var chunk in infos)
                {
                    if (ContainsXY(chunk.BoundsMin, chunk.BoundsMax, worldX, worldY))
                    {
                        info = chunk;
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

        for (int offsetX = -1; offsetX <= 1; offsetX++)
        {
            int tx = tileKey.tileX + offsetX;
            if (tx < 0 || tx >= 64)
                continue;

            for (int offsetY = -1; offsetY <= 1; offsetY++)
            {
                int ty = tileKey.tileY + offsetY;
                if (ty < 0 || ty >= 64)
                    continue;

                foreach (var chunk in _chunks)
                {
                    if (chunk.TileX != tx || chunk.TileY != ty)
                        continue;

                    if (ContainsXY(chunk.BoundsMin, chunk.BoundsMax, worldX, worldY))
                    {
                        mesh = chunk;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    public TerrainRenderer(GL gl, IDataSource? dataSource, TerrainLighting lighting, Func<string, string?>? texturePathResolver = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _lighting = lighting;
        _texturePathResolver = texturePathResolver;
        _shader = CreateTerrainShader();
        _tileShader = CreateTileTerrainShader();

        _uTileDiffuseLayerCountLoc = _tileShader.GetUniformLocation("uDiffuseLayerCount");
        _uTileOpacityLoc = _tileShader.GetUniformLocation("uOpacity");
        _uUseWorldUvLoc = _shader.GetUniformLocation("uUseWorldUV");
        _uTileUseWorldUvLoc = _tileShader.GetUniformLocation("uUseWorldUV");
        _uUseMccvLoc = _shader.GetUniformLocation("uUseMccv");
        _uTileUseMccvLoc = _tileShader.GetUniformLocation("uUseMccv");
        _uAlphaDebugChannelLoc = _shader.GetUniformLocation("uAlphaDebugChannel");
        _uTileAlphaDebugChannelLoc = _tileShader.GetUniformLocation("uAlphaDebugChannel");
        _uHasShadowLoc = _shader.GetUniformLocation("uHasShadowMap");

        for (int i = 0; i < 4; i++)
            _uHasTexLoc[i] = _shader.GetUniformLocation($"uHasTex{i}");
        for (int i = 1; i < 4; i++)
            _uImplicitAlphaLoc[i] = _shader.GetUniformLocation($"uImplicitAlpha{i}");
        for (int i = 1; i < 4; i++)
            _uHasAlphaLoc[i] = _shader.GetUniformLocation($"uHasAlpha{i}");

        InitializeSamplerUniforms();
        _lastTileFadeTimestamp = Stopwatch.GetTimestamp();
    }

    public void AddChunks(List<TerrainChunkMesh> chunks, IDictionary<(int, int), List<string>> tileTextures)
    {
        foreach (var chunk in chunks)
        {
            chunk.Gl = _gl;
            UploadAlphaTextures(chunk);
            UploadShadowTexture(chunk);
            _chunkMeshByKey[(chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY)] = chunk;
        }

        _chunks.AddRange(chunks);

        foreach (var kvp in tileTextures)
        {
            _tileTextures[kvp.Key] = kvp.Value;
            foreach (var textureName in kvp.Value)
                GetOrLoadTexture(textureName);
        }

        ViewerLog.Trace($"[TerrainRenderer] Now rendering {_chunks.Count} chunks, {_textureCache.Count} textures cached");
    }

    public void AddTile(TerrainTileMesh tileMesh, List<string> tileTextureNames, List<global::MdxViewer.Terrain.TerrainChunkInfo> chunkInfos, bool fadeIn = true)
    {
        var key = (tileMesh.TileX, tileMesh.TileY);
        if (_tileMap.ContainsKey(key))
            return;

        tileMesh.Gl = _gl;
        CreateDiffuseArrayTexture(tileMesh, tileTextureNames);

        _tiles.Add(tileMesh);
        _tileMap[key] = tileMesh;
        _tileAlphas[key] = fadeIn ? 0.0f : 1.0f;
        _tileTargetAlphas[key] = 1.0f;
        _chunkInfosByTile[key] = chunkInfos;
        foreach (var chunkInfo in chunkInfos)
            _chunkInfoByKey[(chunkInfo.TileX, chunkInfo.TileY, chunkInfo.ChunkX, chunkInfo.ChunkY)] = chunkInfo;

        _loadedTileChunkCount += chunkInfos.Count;

        ViewerLog.Trace($"[TerrainRenderer] Now rendering {_tiles.Count} batched tiles ({_loadedTileChunkCount} chunks)");
    }

    public void RemoveTile(int tileX, int tileY)
    {
        var key = (tileX, tileY);
        if (_tileMap.TryGetValue(key, out var mesh))
        {
            _tiles.Remove(mesh);
            _tileMap.Remove(key);
        }

        _tileAlphas.Remove(key);
        _tileTargetAlphas.Remove(key);

        if (_chunkInfosByTile.TryGetValue(key, out var infos))
        {
            _loadedTileChunkCount -= infos.Count;
            foreach (var chunkInfo in infos)
                _chunkInfoByKey.Remove((chunkInfo.TileX, chunkInfo.TileY, chunkInfo.ChunkX, chunkInfo.ChunkY));
            _chunkInfosByTile.Remove(key);
        }
    }

    public void RemoveChunks(List<TerrainChunkMesh> chunks)
    {
        foreach (var chunk in chunks)
        {
            _chunks.Remove(chunk);
            _chunkMeshByKey.Remove((chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY));
        }
    }

    public void ApplyTextureSamplingSettings()
    {
        foreach (var textureId in _textureCache.Values)
        {
            if (textureId == 0)
                continue;

            _gl.BindTexture(TextureTarget.Texture2D, textureId);
            RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: true, TextureWrapMode.Repeat, TextureWrapMode.Repeat);
        }

        foreach (var chunk in _chunks)
        {
            foreach (var textureId in chunk.AlphaTextures.Values)
            {
                if (textureId == 0)
                    continue;

                _gl.BindTexture(TextureTarget.Texture2D, textureId);
                RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: false, TextureWrapMode.ClampToEdge, TextureWrapMode.ClampToEdge);
            }

            if (chunk.ShadowTexture != 0)
            {
                _gl.BindTexture(TextureTarget.Texture2D, chunk.ShadowTexture);
                RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: false, TextureWrapMode.ClampToEdge, TextureWrapMode.ClampToEdge);
            }
        }

        foreach (var tile in _tiles)
        {
            if (tile.DiffuseArrayTexture != 0)
            {
                _gl.BindTexture(TextureTarget.Texture2DArray, tile.DiffuseArrayTexture);
                RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2DArray, hasMipmaps: true, TextureWrapMode.Repeat, TextureWrapMode.Repeat);
            }

            if (tile.AlphaShadowArrayTexture != 0)
            {
                _gl.BindTexture(TextureTarget.Texture2DArray, tile.AlphaShadowArrayTexture);
                RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2DArray, hasMipmaps: false, TextureWrapMode.ClampToEdge, TextureWrapMode.ClampToEdge);
            }
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);
    }

    public int ChunksRendered { get; private set; }
    public int ChunksCulled { get; private set; }
    public int LastFrameDrawCalls { get; private set; }
    public int LastFrameUniform1Calls { get; private set; }
    public int LastFrameActiveTextureCalls { get; private set; }
    public int LastFrameActiveTextureSkips { get; private set; }
    public int LastFrameBindTextureCalls { get; private set; }
    public int LastFrameBindTextureSkips { get; private set; }

    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, FrustumCuller? frustum = null)
    {
        if (_tiles.Count == 0 && _chunks.Count == 0)
            return;

        if (_alphaSamplingDirty)
            ApplyAlphaSamplingMode();

        LastFrameDrawCalls = 0;
        LastFrameUniform1Calls = 0;
        LastFrameActiveTextureCalls = 0;
        LastFrameActiveTextureSkips = 0;
        LastFrameBindTextureCalls = 0;
        LastFrameBindTextureSkips = 0;

        _activeTextureUnit = -1;
        Array.Clear(_boundTexture2DByUnit, 0, _boundTexture2DByUnit.Length);

        _lighting.Update();

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

        _gl.Disable(EnableCap.CullFace);
        _gl.PolygonMode(TriangleFace.FrontAndBack, _wireframe ? PolygonMode.Line : PolygonMode.Fill);

        _shader.SetInt("uShowChunkGrid", ShowChunkGrid ? 1 : 0);
        _shader.SetInt("uShowTileGrid", ShowTileGrid ? 1 : 0);
        _shader.SetInt("uShowAlphaMask", ShowAlphaMask ? 1 : 0);
        _shader.SetInt("uShowShadowMap", ShowShadowMap ? 1 : 0);
        _shader.SetInt("uShowContours", ShowContours ? 1 : 0);
        _shader.SetFloat("uContourInterval", ContourInterval);
        _shader.SetInt("uShowLayer0", ShowLayer0 ? 1 : 0);
        _shader.SetInt("uShowLayer1", ShowLayer1 ? 1 : 0);
        _shader.SetInt("uShowLayer2", ShowLayer2 ? 1 : 0);
        _shader.SetInt("uShowLayer3", ShowLayer3 ? 1 : 0);

        Uniform1Counted(_uUseWorldUvLoc, UseWorldUvForDiffuse ? 1 : 0);
        Uniform1Counted(_uUseMccvLoc, UseMccv ? 1 : 0);
        Uniform1Counted(_uAlphaDebugChannelLoc, Math.Clamp(AlphaMaskChannel, 0, 3));

        float chunkCullDistance = _lighting.FogEnd + 200f;
        float chunkCullDistanceSq = chunkCullDistance * chunkCullDistance;
        ChunksRendered = 0;
        ChunksCulled = 0;

        foreach (var chunk in _chunks)
        {
            var center = (chunk.BoundsMin + chunk.BoundsMax) * 0.5f;
            float distanceSq = Vector3.DistanceSquared(cameraPos, center);
            if (distanceSq > chunkCullDistanceSq)
            {
                ChunksCulled++;
                continue;
            }

            if (frustum != null && !frustum.TestAABB(chunk.BoundsMin, chunk.BoundsMax))
            {
                ChunksCulled++;
                continue;
            }

            RenderChunk(chunk);
            ChunksRendered++;
        }

        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
        _gl.Enable(EnableCap.CullFace);
    }

    private void ApplyAlphaSamplingMode()
    {
        _alphaSamplingDirty = false;
        int minFilter = _useNearestForAlphaSampling ? (int)TextureMinFilter.Nearest : (int)TextureMinFilter.Linear;
        int magFilter = _useNearestForAlphaSampling ? (int)TextureMagFilter.Nearest : (int)TextureMagFilter.Linear;

        foreach (var chunk in _chunks)
        {
            foreach (var texture in chunk.AlphaTextures.Values)
            {
                if (texture == 0)
                    continue;

                _gl.BindTexture(TextureTarget.Texture2D, texture);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, minFilter);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, magFilter);
            }

            if (chunk.ShadowTexture != 0)
            {
                _gl.BindTexture(TextureTarget.Texture2D, chunk.ShadowTexture);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, minFilter);
                _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, magFilter);
            }
        }

        foreach (var tile in _tiles)
        {
            if (tile.AlphaShadowArrayTexture == 0)
                continue;

            _gl.BindTexture(TextureTarget.Texture2DArray, tile.AlphaShadowArrayTexture);
            _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, minFilter);
            _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, magFilter);
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);
    }

    public unsafe void ReplaceTileAlphaShadowArray(int tileX, int tileY, byte[] alphaShadowRgba)
    {
        if (_tileMap.TryGetValue((tileX, tileY), out var tile))
        {
            if (tile.AlphaShadowArrayTexture == 0 || alphaShadowRgba == null || alphaShadowRgba.Length < 64 * 64 * 4 * 256)
                return;

            _gl.BindTexture(TextureTarget.Texture2DArray, tile.AlphaShadowArrayTexture);
            fixed (byte* ptr = alphaShadowRgba)
            {
                _gl.TexSubImage3D(TextureTarget.Texture2DArray, 0, 0, 0, 0, 64, 64, 256, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }
            _gl.BindTexture(TextureTarget.Texture2DArray, 0);
            _alphaSamplingDirty = true;
            return;
        }

        const int chunkSize = 64 * 64;
        const int chunkStride = chunkSize * 4;
        const int tileChunks = 16;
        int expectedLength = chunkStride * tileChunks * tileChunks;
        if (alphaShadowRgba == null || alphaShadowRgba.Length < expectedLength)
            throw new ArgumentException($"Expected alpha-shadow RGBA array length {expectedLength}.");

        foreach (var chunk in _chunks)
        {
            if (chunk.TileX != tileX || chunk.TileY != tileY)
                continue;

            int slice = chunk.ChunkY * tileChunks + chunk.ChunkX;
            int sliceBase = slice * chunkStride;
            var alpha1 = new byte[chunkSize];
            var alpha2 = new byte[chunkSize];
            var alpha3 = new byte[chunkSize];
            var shadow = new byte[chunkSize];

            for (int i = 0; i < chunkSize; i++)
            {
                int source = sliceBase + i * 4;
                alpha1[i] = alphaShadowRgba[source + 0];
                alpha2[i] = alphaShadowRgba[source + 1];
                alpha3[i] = alphaShadowRgba[source + 2];
                shadow[i] = alphaShadowRgba[source + 3];
            }

            chunk.AlphaMaps[1] = alpha1;
            chunk.AlphaMaps[2] = alpha2;
            chunk.AlphaMaps[3] = alpha3;
            chunk.ShadowMap = shadow;

            for (int layer = 1; layer <= 3; layer++)
            {
                if (chunk.AlphaTextures.TryGetValue(layer, out uint oldTexture) && oldTexture != 0)
                    _gl.DeleteTexture(oldTexture);
                chunk.AlphaTextures.Remove(layer);
            }

            if (chunk.ShadowTexture != 0)
            {
                _gl.DeleteTexture(chunk.ShadowTexture);
                chunk.ShadowTexture = 0;
            }

            chunk.AlphaTextures[1] = UploadSingleChannelTexture(alpha1);
            chunk.AlphaTextures[2] = UploadSingleChannelTexture(alpha2);
            chunk.AlphaTextures[3] = UploadSingleChannelTexture(alpha3);
            chunk.ShadowTexture = UploadSingleChannelTexture(shadow);
        }
    }

    private unsafe void RenderTiles(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, FrustumCuller? frustum)
    {
        UpdateTileFades();

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
        _gl.PolygonMode(TriangleFace.FrontAndBack, _wireframe ? PolygonMode.Line : PolygonMode.Fill);

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
        Uniform1Counted(_uTileUseMccvLoc, UseMccv ? 1 : 0);
        Uniform1Counted(_uTileAlphaDebugChannelLoc, Math.Clamp(AlphaMaskChannel, 0, 3));

        float cullDistance = _lighting.FogEnd + 200f;
        float cullDistanceSq = cullDistance * cullDistance;
        ChunksRendered = 0;
        ChunksCulled = 0;
        LastFrameDrawCalls = 0;

        bool baseVisible = ShowLayer0;
        bool overlayVisible = ShowLayer1 || ShowLayer2 || ShowLayer3;
        bool blendOverlaysOnly = !baseVisible && overlayVisible;

        foreach (var tile in _tiles)
        {
            var tileKey = (tile.TileX, tile.TileY);
            float tileOpacity = _tileAlphas.TryGetValue(tileKey, out float storedAlpha) ? storedAlpha : 1.0f;

            var center = (tile.BoundsMin + tile.BoundsMax) * 0.5f;
            float distanceSq = Vector3.DistanceSquared(cameraPos, center);
            if (distanceSq > cullDistanceSq)
            {
                ChunksCulled += tile.ChunkCount;
                continue;
            }

            if (frustum != null && !frustum.TestAABB(tile.BoundsMin, tile.BoundsMax))
            {
                ChunksCulled += tile.ChunkCount;
                continue;
            }

            if (blendOverlaysOnly)
            {
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                _gl.DepthMask(false);
            }
            else if (tileOpacity < 0.999f)
            {
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                _gl.DepthMask(true);
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
            _tileShader.SetFloat("uOpacity", tileOpacity);

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

    private void UpdateTileFades()
    {
        long now = Stopwatch.GetTimestamp();
        float deltaSeconds = (float)(now - _lastTileFadeTimestamp) / Stopwatch.Frequency;
        _lastTileFadeTimestamp = now;
        if (deltaSeconds <= 0f)
            return;

        float blend = Math.Clamp(deltaSeconds / TileFadeInDurationSeconds, 0f, 1f);
        blend = blend * blend * (3f - 2f * blend);

        foreach (var tileKey in _tileMap.Keys)
        {
            float currentAlpha = _tileAlphas.TryGetValue(tileKey, out float storedAlpha) ? storedAlpha : 1.0f;
            float targetAlpha = _tileTargetAlphas.TryGetValue(tileKey, out float storedTarget) ? storedTarget : 1.0f;
            if (MathF.Abs(currentAlpha - targetAlpha) <= 0.001f)
            {
                _tileAlphas[tileKey] = targetAlpha;
                continue;
            }

            _tileAlphas[tileKey] = currentAlpha + (targetAlpha - currentAlpha) * blend;
        }
    }

    private void Uniform1Counted(int location, int value)
    {
        _gl.Uniform1(location, value);
        LastFrameUniform1Calls++;
    }

    private unsafe void RenderChunk(TerrainChunkMesh chunk)
    {
        var textureNames = _tileTextures.GetValueOrDefault((chunk.TileX, chunk.TileY));
        if (textureNames == null || textureNames.Count == 0 || chunk.Layers.Length == 0)
        {
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
        bool overlayVisible = (ShowLayer1 && chunk.Layers.Length > 1) || (ShowLayer2 && chunk.Layers.Length > 2) || (ShowLayer3 && chunk.Layers.Length > 3);

        if (!baseVisible && overlayVisible)
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
        BindChunkLayerTextures(chunk, textureNames);

        _gl.BindVertexArray(chunk.Vao);
        _gl.DrawElements(PrimitiveType.Triangles, chunk.IndexCount, DrawElementsType.UnsignedShort, null);
        LastFrameDrawCalls++;
        _gl.BindVertexArray(0);

        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
    }

    private void InitializeSamplerUniforms()
    {
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
        if ((uint)unit >= (uint)_boundTexture2DByUnit.Length)
            return;
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
        if ((uint)unit >= (uint)_boundTexture2DByUnit.Length)
            return;
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

        int maxDimension = 0;
        for (int i = 0; i < Math.Min(textureNames.Count, layerCount); i++)
        {
            if (TryLoadTerrainTexturePixels(textureNames[i], out int width, out int height, out _))
                maxDimension = Math.Max(maxDimension, Math.Max(width, height));
        }

        int targetDimension = maxDimension switch
        {
            <= 0 => 256,
            <= 64 => 64,
            <= 128 => 128,
            <= 256 => 256,
            _ => 256,
        };

        uint texture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, texture);
        _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, (uint)targetDimension, (uint)targetDimension, (uint)layerCount, 0, PixelFormat.Rgba, PixelType.UnsignedByte, (void*)0);

        for (int layer = 0; layer < layerCount; layer++)
        {
            byte[] pixels;
            int width;
            int height;
            if (layer < textureNames.Count && TryLoadTerrainTexturePixels(textureNames[layer], out width, out height, out pixels))
            {
                if (width != targetDimension || height != targetDimension)
                    pixels = ResampleRgbaNearest(pixels, width, height, targetDimension, targetDimension);
            }
            else
            {
                pixels = CreateSolidRgba(targetDimension, targetDimension, 255, 255, 255, 255);
            }

            fixed (byte* ptr = pixels)
            {
                _gl.TexSubImage3D(TextureTarget.Texture2DArray, 0, 0, 0, layer, (uint)targetDimension, (uint)targetDimension, 1, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
        _gl.GenerateMipmap(TextureTarget.Texture2DArray);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);

        tileMesh.DiffuseArrayTexture = texture;
        tileMesh.DiffuseLayerCount = layerCount;
    }

    private static byte[] CreateSolidRgba(int width, int height, byte red, byte green, byte blue, byte alpha)
    {
        var data = new byte[width * height * 4];
        for (int i = 0; i < data.Length; i += 4)
        {
            data[i + 0] = red;
            data[i + 1] = green;
            data[i + 2] = blue;
            data[i + 3] = alpha;
        }
        return data;
    }

    private static byte[] ResampleRgbaNearest(byte[] source, int sourceWidth, int sourceHeight, int destWidth, int destHeight)
    {
        var dest = new byte[destWidth * destHeight * 4];
        for (int y = 0; y < destHeight; y++)
        {
            int srcY = (int)((long)y * sourceHeight / destHeight);
            for (int x = 0; x < destWidth; x++)
            {
                int srcX = (int)((long)x * sourceWidth / destWidth);
                int sourceIndex = (srcY * sourceWidth + srcX) * 4;
                int destIndex = (y * destWidth + x) * 4;
                dest[destIndex + 0] = source[sourceIndex + 0];
                dest[destIndex + 1] = source[sourceIndex + 1];
                dest[destIndex + 2] = source[sourceIndex + 2];
                dest[destIndex + 3] = source[sourceIndex + 3];
            }
        }

        return dest;
    }

    private bool TryLoadTerrainTexturePixels(string textureName, out int width, out int height, out byte[] pixels)
    {
        width = 0;
        height = 0;
        pixels = Array.Empty<byte>();
        if (string.IsNullOrEmpty(textureName))
            return false;

        byte[]? blpData = null;
        if (_dataSource != null)
        {
            blpData = _dataSource.ReadFile(textureName) ?? _dataSource.ReadFile(textureName.Replace('/', '\\'));
        }

        if (blpData == null || blpData.Length == 0)
        {
            if (_texturePathResolver != null)
            {
                var pngPath = _texturePathResolver(textureName);
                if (pngPath != null)
                    return TryLoadPngPixels(pngPath, out width, out height, out pixels);
            }

            return false;
        }

        try
        {
            using var stream = new MemoryStream(blpData);
            using var blp = new BlpFile(stream);
            using var bitmap = blp.GetBitmap(0);
            width = bitmap.Width;
            height = bitmap.Height;

            pixels = new byte[width * height * 4];
            var rect = new System.Drawing.Rectangle(0, 0, width, height);
            var data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var sourceBytes = new byte[data.Stride * height];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, sourceBytes, 0, sourceBytes.Length);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int sourceIndex = y * data.Stride + x * 4;
                        int destIndex = (y * width + x) * 4;
                        pixels[destIndex + 0] = sourceBytes[sourceIndex + 2];
                        pixels[destIndex + 1] = sourceBytes[sourceIndex + 1];
                        pixels[destIndex + 2] = sourceBytes[sourceIndex + 0];
                        pixels[destIndex + 3] = sourceBytes[sourceIndex + 3];
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(data);
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryLoadPngPixels(string pngPath, out int width, out int height, out byte[] pixels)
    {
        width = 0;
        height = 0;
        pixels = Array.Empty<byte>();

        try
        {
            using var bitmap = new System.Drawing.Bitmap(pngPath);
            width = bitmap.Width;
            height = bitmap.Height;
            pixels = new byte[width * height * 4];
            var rect = new System.Drawing.Rectangle(0, 0, width, height);
            var data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var sourceBytes = new byte[data.Stride * height];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, sourceBytes, 0, sourceBytes.Length);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int sourceIndex = y * data.Stride + x * 4;
                        int destIndex = (y * width + x) * 4;
                        pixels[destIndex + 0] = sourceBytes[sourceIndex + 2];
                        pixels[destIndex + 1] = sourceBytes[sourceIndex + 1];
                        pixels[destIndex + 2] = sourceBytes[sourceIndex + 0];
                        pixels[destIndex + 3] = sourceBytes[sourceIndex + 3];
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(data);
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private void BindChunkLayerTextures(TerrainChunkMesh chunk, List<string> textureNames)
    {
        for (int layer = 0; layer < 4; layer++)
        {
            bool hasTexture = false;
            if (layer < chunk.Layers.Length)
            {
                int textureIndex = chunk.Layers[layer].TextureIndex;
                if (textureIndex >= 0 && textureIndex < textureNames.Count)
                {
                    uint texture = GetOrLoadTexture(textureNames[textureIndex]);
                    if (texture != 0)
                    {
                        BindTexture2D(layer, texture);
                        hasTexture = true;
                    }
                }
            }

            Uniform1Counted(_uHasTexLoc[layer], hasTexture ? 1 : 0);
        }

        for (int layer = 1; layer < 4; layer++)
        {
            bool hasLayer = layer < chunk.Layers.Length;
            bool usesAlphaMap = hasLayer && (chunk.Layers[layer].Flags & 0x100u) != 0;
            bool implicitFullAlpha = hasLayer && !usesAlphaMap;

            bool hasAlpha = chunk.AlphaTextures.TryGetValue(layer, out uint alphaTexture) && alphaTexture != 0;
            if (hasAlpha)
                BindTexture2D(3 + layer, alphaTexture);

            Uniform1Counted(_uHasAlphaLoc[layer], hasAlpha ? 1 : 0);
            Uniform1Counted(_uImplicitAlphaLoc[layer], implicitFullAlpha ? 1 : 0);
        }

        bool hasShadow = chunk.ShadowTexture != 0;
        if (hasShadow)
            BindTexture2D(7, chunk.ShadowTexture);

        Uniform1Counted(_uHasShadowLoc, hasShadow ? 1 : 0);
    }

    public void ToggleWireframe()
    {
        _wireframe = !_wireframe;
    }

    private uint GetOrLoadTexture(string textureName)
    {
        if (string.IsNullOrEmpty(textureName))
            return 0;
        if (_textureCache.TryGetValue(textureName, out uint cached))
            return cached;

        uint texture = LoadTerrainTexture(textureName);
        _textureCache[textureName] = texture;
        return texture;
    }

    private unsafe uint LoadTerrainTexture(string textureName)
    {
        byte[]? blpData = null;
        if (_dataSource != null)
        {
            blpData = _dataSource.ReadFile(textureName) ?? _dataSource.ReadFile(textureName.Replace('/', '\\'));
        }

        if (blpData == null || blpData.Length == 0)
        {
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
            using var stream = new MemoryStream(blpData);
            using var blp = new BlpFile(stream);
            using var bitmap = blp.GetBitmap(0);
            int width = bitmap.Width;
            int height = bitmap.Height;
            var pixels = new byte[width * height * 4];
            var rect = new System.Drawing.Rectangle(0, 0, width, height);
            var data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var sourceBytes = new byte[data.Stride * height];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, sourceBytes, 0, sourceBytes.Length);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int sourceIndex = y * data.Stride + x * 4;
                        int destIndex = (y * width + x) * 4;
                        pixels[destIndex + 0] = sourceBytes[sourceIndex + 2];
                        pixels[destIndex + 1] = sourceBytes[sourceIndex + 1];
                        pixels[destIndex + 2] = sourceBytes[sourceIndex + 0];
                        pixels[destIndex + 3] = sourceBytes[sourceIndex + 3];
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(data);
            }

            uint texture = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, texture);
            fixed (byte* ptr = pixels)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, (uint)width, (uint)height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            _gl.BindTexture(TextureTarget.Texture2D, 0);
            return texture;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TerrainRenderer] Failed to load texture {textureName}: {ex.Message}");
            return 0;
        }
    }

    private unsafe uint LoadPngTexture(string pngPath)
    {
        try
        {
            using var bitmap = new System.Drawing.Bitmap(pngPath);
            int width = bitmap.Width;
            int height = bitmap.Height;
            var pixels = new byte[width * height * 4];
            var rect = new System.Drawing.Rectangle(0, 0, width, height);
            var data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var sourceBytes = new byte[data.Stride * height];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, sourceBytes, 0, sourceBytes.Length);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int sourceIndex = y * data.Stride + x * 4;
                        int destIndex = (y * width + x) * 4;
                        pixels[destIndex + 0] = sourceBytes[sourceIndex + 2];
                        pixels[destIndex + 1] = sourceBytes[sourceIndex + 1];
                        pixels[destIndex + 2] = sourceBytes[sourceIndex + 0];
                        pixels[destIndex + 3] = sourceBytes[sourceIndex + 3];
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(data);
            }

            uint texture = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, texture);
            fixed (byte* ptr = pixels)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, (uint)width, (uint)height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            _gl.BindTexture(TextureTarget.Texture2D, 0);
            return texture;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[TerrainRenderer] Failed to load PNG texture {pngPath}: {ex.Message}");
            return 0;
        }
    }

    private unsafe void UploadAlphaTextures(TerrainChunkMesh chunk)
    {
        foreach (var kvp in chunk.AlphaMaps)
        {
            int layer = kvp.Key;
            byte[] alphaData = kvp.Value;
            if (alphaData.Length < 64 * 64)
                continue;

            uint texture = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, texture);
            fixed (byte* ptr = alphaData)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.R8, 64, 64, 0, PixelFormat.Red, PixelType.UnsignedByte, ptr);
            }

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            _gl.BindTexture(TextureTarget.Texture2D, 0);
            chunk.AlphaTextures[layer] = texture;
        }
    }

    private unsafe void UploadShadowTexture(TerrainChunkMesh chunk)
    {
        if (chunk.ShadowMap == null || chunk.ShadowMap.Length < 64 * 64)
            return;

        uint texture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, texture);
        fixed (byte* ptr = chunk.ShadowMap)
        {
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.R8, 64, 64, 0, PixelFormat.Red, PixelType.UnsignedByte, ptr);
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        chunk.ShadowTexture = texture;
    }

    private unsafe uint UploadSingleChannelTexture(byte[] data)
    {
        uint texture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, texture);
        fixed (byte* ptr = data)
        {
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.R8, 64, 64, 0, PixelFormat.Red, PixelType.UnsignedByte, ptr);
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return texture;
    }

    private ShaderProgram CreateTerrainShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 5) in vec4 aVertexColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;
out vec4 vVertexColor;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(uModel) * aNormal;
    vTexCoord = aTexCoord;
    vVertexColor = aVertexColor;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;
in vec4 vVertexColor;

uniform sampler2D uDiffuse0;
uniform sampler2D uDiffuse1;
uniform sampler2D uDiffuse2;
uniform sampler2D uDiffuse3;
uniform sampler2D uAlpha1;
uniform sampler2D uAlpha2;
uniform sampler2D uAlpha3;
uniform sampler2D uShadowSampler;

uniform int uUseWorldUV;
uniform int uUseMccv;
uniform int uAlphaDebugChannel;
uniform int uHasTex0;
uniform int uHasTex1;
uniform int uHasTex2;
uniform int uHasTex3;
uniform int uHasAlpha1;
uniform int uHasAlpha2;
uniform int uHasAlpha3;
uniform int uImplicitAlpha1;
uniform int uImplicitAlpha2;
uniform int uImplicitAlpha3;
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
uniform float uOpacity;

out vec4 FragColor;

vec4 SampleDiffuse(int hasTex, sampler2D s, vec2 uv)
{
    if (hasTex == 1 && uShowAlphaMask == 0)
        return texture(s, uv);
    return vec4(1.0, 1.0, 1.0, 1.0);
}

void main() {
    float texScale = 8.0 / 33.333;
    vec2 diffuseUV = (uUseWorldUV == 1) ? (vWorldPos.xy * texScale) : (vTexCoord * 8.0);

    float a1Raw = (uHasAlpha1 == 1) ? texture(uAlpha1, vTexCoord).r : ((uImplicitAlpha1 == 1) ? 1.0 : 0.0);
    float a2Raw = (uHasAlpha2 == 1) ? texture(uAlpha2, vTexCoord).r : ((uImplicitAlpha2 == 1) ? 1.0 : 0.0);
    float a3Raw = (uHasAlpha3 == 1) ? texture(uAlpha3, vTexCoord).r : ((uImplicitAlpha3 == 1) ? 1.0 : 0.0);

    float a1 = (uShowLayer1 == 1) ? a1Raw : 0.0;
    float a2 = (uShowLayer2 == 1) ? a2Raw : 0.0;
    float a3 = (uShowLayer3 == 1) ? a3Raw : 0.0;

    if (uShowAlphaMask == 1) {
        float dbg = 1.0;
        if (uAlphaDebugChannel == 1) dbg = a1Raw;
        else if (uAlphaDebugChannel == 2) dbg = a2Raw;
        else if (uAlphaDebugChannel == 3) dbg = a3Raw;
        else {
            if (uShowLayer1 == 1) dbg = a1Raw;
            else if (uShowLayer2 == 1) dbg = a2Raw;
            else if (uShowLayer3 == 1) dbg = a3Raw;
        }
        FragColor = vec4(dbg, dbg, dbg, 1.0);
        return;
    }

    vec3 norm = normalize(vNormal);
    float diff = abs(dot(norm, normalize(uLightDir)));
    vec3 lighting = uAmbientColor + uLightColor * diff;
    vec3 result = vec3(1.0);

    if (uShowLayer0 == 1) {
        vec4 c0 = SampleDiffuse(uHasTex0, uDiffuse0, diffuseUV);
        result = c0.rgb * lighting;
    }
    if (uShowLayer1 == 1 && uHasTex1 == 1) {
        vec4 c1 = texture(uDiffuse1, diffuseUV);
        result = mix(result, c1.rgb * lighting, a1);
    }
    if (uShowLayer2 == 1 && uHasTex2 == 1) {
        vec4 c2 = texture(uDiffuse2, diffuseUV);
        result = mix(result, c2.rgb * lighting, a2);
    }
    if (uShowLayer3 == 1 && uHasTex3 == 1) {
        vec4 c3 = texture(uDiffuse3, diffuseUV);
        result = mix(result, c3.rgb * lighting, a3);
    }

    vec3 vertexTint = (uUseMccv == 1) ? clamp(vVertexColor.rgb * 2.0, 0.0, 2.0) : vec3(1.0);
    result *= vertexTint;

    if (uShowShadowMap == 1 && uHasShadowMap == 1) {
        float shadow = texture(uShadowSampler, vTexCoord).r;
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
        if (uShowLayer1 == 1) inv *= (1.0 - a1);
        if (uShowLayer2 == 1) inv *= (1.0 - a2);
        if (uShowLayer3 == 1) inv *= (1.0 - a3);
        outAlpha = 1.0 - inv;
    }

    FragColor = vec4(finalColor, outAlpha * uOpacity);
}
";

        var program = ShaderProgram.Create(_gl, vertSrc, fragSrc);
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
layout(location = 5) in vec4 aVertexColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;
out vec4 vVertexColor;
flat out uint vChunkSlice;
flat out uvec4 vTexIdx;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(uModel) * aNormal;
    vTexCoord = aTexCoord;
    vVertexColor = aVertexColor;
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
in vec4 vVertexColor;
flat in uint vChunkSlice;
flat in uvec4 vTexIdx;

uniform sampler2DArray uDiffuseArray;
uniform sampler2DArray uAlphaShadowArray;
uniform int uDiffuseLayerCount;
uniform int uUseWorldUV;
uniform int uUseMccv;
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

    vec3 vertexTint = (uUseMccv == 1) ? clamp(vVertexColor.rgb * 2.0, 0.0, 2.0) : vec3(1.0);
    result *= vertexTint;

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

        foreach (var texture in _textureCache.Values)
        {
            if (texture != 0)
                _gl.DeleteTexture(texture);
        }
        _textureCache.Clear();

        _shader.Dispose();
        _tileShader.Dispose();
    }
}