using System.Diagnostics;
using System.Numerics;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;

namespace WoWViewer.Terrain;

/// <summary>
/// Renders a low-resolution 3D terrain mesh from WDL (World Detail Level) data.
/// Each WDL tile has a 17×17 outer + 16×16 inner height grid — same layout as MCNK
/// but at tile scale (each WDL "cell" = one ADT chunk = 533.33 world units).
/// 
/// Used as background/far terrain: parsed map-wide as compact CPU data, promoted to GPU
/// meshes inside the horizon window, and kept as an underlay while detailed ADT meshes stream.
/// </summary>
public class WdlTerrainRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ShaderProgram _shader;
    private readonly MinimapRenderer? _minimapRenderer;
    private string? _mapName;

    private const float TileFadeOutDurationSeconds = 0.18f;
    private const float TileFadeInDurationSeconds = 0.32f;
    private const float TileShowDelaySeconds = 0.12f;
    private const float DistanceHazeStartFactor = 0.45f;
    private const float DistanceHazeEndFactor = 0.96f;
    private const float DistanceHazeColorBlend = 0.92f;
    private const float DistanceHazeOpacityFloor = 0.18f;
    private const float HorizonDistancePadding = 2500f;

    // Per-tile GPU mesh data
    private readonly Dictionary<int, WdlTileMesh> _tileMeshes = new(); // tileIndex → mesh
    private readonly Dictionary<int, WdlParser.WdlTile> _tileData = new(); // parsed CPU height data
    private readonly Dictionary<int, float> _tileAlphas = new();
    private readonly Dictionary<int, float> _tileTargetAlphas = new();
    private readonly Dictionary<int, long> _tileShowReadyTimestamps = new();
    private readonly HashSet<int> _detailedTileIndices = new();
    private readonly List<(int index, float distanceSq)> _tileResidencyScratch = new();
    private readonly List<int> _tileEvictionScratch = new();
    private long _lastFadeTimestamp;

    private const int MaxTileMeshBuildsPerFrame = 8;
    private const float TileResidencyLoadPadding = 1.0f;
    private const float TileResidencyUnloadPadding = 2.0f;

    // Stats
    public int TotalTiles => _tileMeshes.Count;
    public int VisibleTiles => _tileAlphas.Values.Count(static alpha => alpha > 0.01f);
    public int HiddenTiles => _tileAlphas.Values.Count(static alpha => alpha <= 0.01f);

    public WdlTerrainRenderer(GL gl, MinimapRenderer? minimapRenderer = null)
    {
        _gl = gl;
        _shader = CreateShader();
        _minimapRenderer = minimapRenderer;
        _lastFadeTimestamp = Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// Load WDL data and retain compact CPU height data. GPU tile meshes are streamed
    /// around the camera during Render instead of being built for the whole map here.
    /// </summary>
    public bool Load(IDataSource dataSource, string mapDirectory)
    {
        _mapName = mapDirectory;

        if (!WdlDataSourceResolver.TryReadWdlBytes(dataSource, mapDirectory, out byte[]? wdlBytes, out string? resolvedPath)
            || wdlBytes == null
            || wdlBytes.Length == 0)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL 3D] No WDL data for {mapDirectory}");
            return false;
        }

        var wdlData = WdlParser.Parse(wdlBytes);
        if (wdlData == null)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[WDL 3D] Failed to parse WDL for {mapDirectory}");
            return false;
        }

        if (!string.IsNullOrWhiteSpace(resolvedPath))
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL 3D] Loaded {mapDirectory} from {resolvedPath}");

        int indexed = 0;
        for (int tileY = 0; tileY < 64; tileY++)
        {
            for (int tileX = 0; tileX < 64; tileX++)
            {
                int idx = GetTileIndex(tileX, tileY);
                var tile = wdlData.Tiles[idx];
                if (tile?.HasData != true) continue;

                _tileData[idx] = tile;
                indexed++;
            }
        }

        ViewerLog.Important(ViewerLog.Category.Terrain, $"[WDL 3D] Indexed {indexed} low-res terrain tiles for {mapDirectory}; GPU meshes stream by fog range");
        return indexed > 0;
    }

    /// <summary>
    /// Replaces the WDL suppression set with the detailed ADT tiles that will
    /// actually be submitted this frame. GPU residency alone is not enough:
    /// retained neighboring tiles remain loaded for streaming, but the terrain
    /// selector may intentionally omit them from detailed rendering.
    /// </summary>
    public void SetDetailedTileSubmission(
        IReadOnlyList<(int tileX, int tileY)> selectedTiles,
        Func<int, int, bool> isTileResident)
    {
        ArgumentNullException.ThrowIfNull(selectedTiles);
        ArgumentNullException.ThrowIfNull(isTileResident);

        _detailedTileIndices.Clear();
        foreach (var (tileX, tileY) in selectedTiles)
        {
            if (isTileResident(tileX, tileY))
                _detailedTileIndices.Add(GetTileIndex(tileX, tileY));
        }
    }

    /// <summary>
    /// Render all visible WDL tiles.
    /// </summary>
    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        TerrainLighting lighting, FrustumCuller? frustum = null, bool opaqueFallback = false,
        float? horizonDistance = null)
    {
        if (_tileData.Count == 0) return;

        UpdateTileResidency(cameraPos, lighting.FogEnd);
        if (_tileMeshes.Count == 0)
            return;

        long now = Stopwatch.GetTimestamp();
        float deltaSeconds = (float)(now - _lastFadeTimestamp) / Stopwatch.Frequency;
        _lastFadeTimestamp = now;
        UpdateTileFades(deltaSeconds);

        _shader.Use();
        _shader.SetMat4("uView", view);
        _shader.SetMat4("uProj", proj);
        _shader.SetVec3("uLightDir", lighting.LightDirection);
        _shader.SetVec3("uLightColor", lighting.LightColor);
        _shader.SetVec3("uAmbientColor", lighting.AmbientColor);
        _shader.SetVec3("uFogColor", lighting.FogColor);
        _shader.SetFloat("uFogStart", lighting.FogStart);
        _shader.SetFloat("uFogEnd", lighting.FogEnd);
        _shader.SetFloat("uDistanceHazeStartFactor", DistanceHazeStartFactor);
        _shader.SetFloat("uDistanceHazeEndFactor", DistanceHazeEndFactor);
        _shader.SetFloat("uDistanceHazeColorBlend", DistanceHazeColorBlend);
        _shader.SetFloat("uDistanceHazeOpacityFloor", DistanceHazeOpacityFloor);
        _shader.SetVec3("uCameraPos", cameraPos);
        _shader.SetInt("uMinimapTexture", 0);
        _shader.SetInt("uForceOpaque", opaqueFallback ? 1 : 0);

        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.DepthTest);
        if (opaqueFallback)
        {
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
        }
        else
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.DepthMask(false);
        }
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // Push WDL slightly behind real terrain to prevent z-fighting at tile edges
        _gl.Enable(EnableCap.PolygonOffsetFill);
        _gl.PolygonOffset(1.0f, 1.0f);

        // WDL is a far-field fallback, not a second full-map render pass. The WDL
        // loader intentionally has cheap height data for every populated tile, but
        // tiles beyond the active fog range cannot contribute visible pixels. Keep
        // the frustum test and add the same distance admission the detailed terrain
        // streamer uses so a wide camera frustum does not turn 839 tiles into 839
        // draw calls every frame.
        float fogDistance = MathF.Max(
            horizonDistance ?? (lighting.FogEnd + HorizonDistancePadding),
            WoWConstants.ChunkSize * 1.5f);
        float fogDistanceSq = fogDistance * fogDistance;
        float detailedOverlapDistance = MathF.Max(lighting.FogEnd, WoWConstants.ChunkSize * 0.5f);
        float detailedOverlapDistanceSq = detailedOverlapDistance * detailedOverlapDistance;

        foreach (var (idx, mesh) in _tileMeshes)
        {
            if (!_tileAlphas.TryGetValue(idx, out float alpha) || alpha <= 0.01f)
                continue;

            // Frustum cull
            if (frustum != null && !frustum.TestAABB(mesh.BoundsMin, mesh.BoundsMax))
                continue;

            float distanceSq = DistanceSquaredPointToAabb(cameraPos, mesh.BoundsMin, mesh.BoundsMax);
            if (distanceSq > fogDistanceSq)
                continue;

            if (_detailedTileIndices.Contains(idx) && distanceSq <= detailedOverlapDistanceSq)
                continue;

            uint minimapTexture = 0;
            if (_minimapRenderer != null && !string.IsNullOrWhiteSpace(_mapName))
                minimapTexture = _minimapRenderer.GetTileTexture(_mapName, mesh.TileY, mesh.TileX);

            _shader.SetFloat("uOpacity", alpha);
            _shader.SetInt("uUseMinimapTexture", minimapTexture != 0 ? 1 : 0);
            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, minimapTexture);
            _gl.BindVertexArray(mesh.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, mesh.IndexCount, DrawElementsType.UnsignedInt, null);
        }

        _gl.Disable(EnableCap.PolygonOffsetFill);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.BindVertexArray(0);
    }

    private void UpdateTileResidency(Vector3 cameraPos, float fogEnd)
    {
        float fogDistance = MathF.Max(fogEnd + HorizonDistancePadding, WoWConstants.ChunkSize * 1.5f);
        float loadDistance = fogDistance + (TileResidencyLoadPadding * WoWConstants.ChunkSize);
        float unloadDistance = fogDistance + (TileResidencyUnloadPadding * WoWConstants.ChunkSize);
        float loadDistanceSq = loadDistance * loadDistance;
        float unloadDistanceSq = unloadDistance * unloadDistance;

        _tileResidencyScratch.Clear();
        foreach (int tileIndex in _tileData.Keys)
        {
            GetTileCoordinates(tileIndex, out int tileX, out int tileY);
            GetTileGridBounds(tileX, tileY, out Vector3 boundsMin, out Vector3 boundsMax);
            float distanceSq = DistanceSquaredPointToAabb(cameraPos, boundsMin, boundsMax);
            if (distanceSq <= loadDistanceSq)
                _tileResidencyScratch.Add((tileIndex, distanceSq));
        }

        _tileResidencyScratch.Sort(static (left, right) => left.distanceSq.CompareTo(right.distanceSq));

        int builtThisFrame = 0;
        foreach ((int tileIndex, _) in _tileResidencyScratch)
        {
            if (_tileMeshes.ContainsKey(tileIndex)
                || !_tileData.TryGetValue(tileIndex, out WdlParser.WdlTile? tile))
            {
                continue;
            }

            GetTileCoordinates(tileIndex, out int tileX, out int tileY);
            WdlTileMesh? mesh = BuildTileMesh(tile, tileX, tileY);
            if (mesh == null)
                continue;

            _tileMeshes[tileIndex] = mesh;
            _tileAlphas[tileIndex] = 1.0f;
            _tileTargetAlphas[tileIndex] = 1.0f;

            builtThisFrame++;
            if (builtThisFrame >= MaxTileMeshBuildsPerFrame)
                break;
        }

        _tileEvictionScratch.Clear();
        foreach ((int tileIndex, WdlTileMesh mesh) in _tileMeshes)
        {
            float distanceSq = DistanceSquaredPointToAabb(cameraPos, mesh.BoundsMin, mesh.BoundsMax);
            if (distanceSq > unloadDistanceSq)
                _tileEvictionScratch.Add(tileIndex);
        }

        foreach (int tileIndex in _tileEvictionScratch)
        {
            if (!_tileMeshes.Remove(tileIndex, out WdlTileMesh? mesh))
                continue;

            DisposeTileMesh(mesh);
            _tileAlphas.Remove(tileIndex);
            _tileTargetAlphas.Remove(tileIndex);
            _tileShowReadyTimestamps.Remove(tileIndex);
        }
    }

    private static float DistanceSquaredPointToAabb(Vector3 point, Vector3 min, Vector3 max)
    {
        float dx = point.X < min.X ? min.X - point.X : point.X > max.X ? point.X - max.X : 0f;
        float dy = point.Y < min.Y ? min.Y - point.Y : point.Y > max.Y ? point.Y - max.Y : 0f;
        float dz = point.Z < min.Z ? min.Z - point.Z : point.Z > max.Z ? point.Z - max.Z : 0f;
        return dx * dx + dy * dy + dz * dz;
    }

    private void SetTileTargetAlpha(int tileIndex, float targetAlpha)
    {
        if (!_tileMeshes.ContainsKey(tileIndex))
            return;

        if (!_tileAlphas.ContainsKey(tileIndex))
            _tileAlphas[tileIndex] = targetAlpha;

        _tileTargetAlphas[tileIndex] = targetAlpha;
        if (targetAlpha >= 0.999f)
            _tileShowReadyTimestamps[tileIndex] = Stopwatch.GetTimestamp() + (long)(TileShowDelaySeconds * Stopwatch.Frequency);
        else
            _tileShowReadyTimestamps.Remove(tileIndex);
    }

    private void UpdateTileFades(float deltaSeconds)
    {
        if (deltaSeconds <= 0f)
            return;

        long now = _lastFadeTimestamp;
        foreach (int tileIndex in _tileMeshes.Keys)
        {
            float currentAlpha = _tileAlphas.TryGetValue(tileIndex, out float storedAlpha) ? storedAlpha : 1.0f;
            float targetAlpha = _tileTargetAlphas.TryGetValue(tileIndex, out float storedTarget) ? storedTarget : 1.0f;

            if (targetAlpha > currentAlpha
                && _tileShowReadyTimestamps.TryGetValue(tileIndex, out long showReadyTimestamp)
                && now < showReadyTimestamp)
            {
                continue;
            }

            if (MathF.Abs(currentAlpha - targetAlpha) <= 0.001f)
            {
                _tileAlphas[tileIndex] = targetAlpha;
                continue;
            }

            float duration = targetAlpha > currentAlpha ? TileFadeInDurationSeconds : TileFadeOutDurationSeconds;
            float blend = Math.Clamp(deltaSeconds / duration, 0f, 1f);
            blend = blend * blend * (3f - 2f * blend);
            _tileAlphas[tileIndex] = currentAlpha + (targetAlpha - currentAlpha) * blend;
        }
    }

    // ── Mesh building ────────────────────────────────────────────────────

    private static int GetTileIndex(int tileX, int tileY) => tileX * 64 + tileY;

    private static void GetTileCoordinates(int tileIndex, out int tileX, out int tileY)
    {
        tileX = tileIndex / 64;
        tileY = tileIndex % 64;
    }

    private static void GetTileGridBounds(int tileX, int tileY, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        float worldX = WoWConstants.MapOrigin - (tileX * WoWConstants.ChunkSize);
        float worldY = WoWConstants.MapOrigin - (tileY * WoWConstants.ChunkSize);
        float minX = worldX - WoWConstants.ChunkSize;
        float maxX = worldX;
        float minY = worldY - WoWConstants.ChunkSize;
        float maxY = worldY;

        // Z is intentionally broad for residency admission; the actual mesh AABB is
        // used for draw-time distance culling after the tile is built.
        boundsMin = new Vector3(minX, minY, -10000f);
        boundsMax = new Vector3(maxX, maxY, 10000f);
    }

    private unsafe WdlTileMesh? BuildTileMesh(WdlParser.WdlTile tile, int tileX, int tileY)
    {
        // WDL uses the same 17×17 outer + 16×16 inner layout as MCNK, but each WDL cell
        // maps onto the viewer's 64×64 chunk grid. The terrain manager, minimap, and WDL
        // preview all use ChunkSize spacing for these coordinates.
        // 17×17 = 289 outer vertices, 16×16 = 256 inner vertices = 545 total.
        const int outerEdge = 17;
        const int innerEdge = 16;
        int totalVerts = outerEdge * outerEdge + innerEdge * innerEdge; // 545

        // Cell world origin (top-left corner in renderer space).
        // A WDL cell spans one ChunkSize step, not one full ADT tile.
        float tileWorldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize;
        float tileWorldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize;
        float cellSize = WoWConstants.ChunkSize / innerEdge;

        // Vertex data: position(3) + normal(3) + uv(2) = 8 floats per vertex
        float[] vertices = new float[totalVerts * 8];
        var boundsMin = new Vector3(float.MaxValue);
        var boundsMax = new Vector3(float.MinValue);

        // Outer vertices (17×17) — indices 0..288
        for (int r = 0; r < outerEdge; r++)
        {
            for (int c = 0; c < outerEdge; c++)
            {
                int vi = r * outerEdge + c;
                float x = tileWorldX - r * cellSize;
                float y = tileWorldY - c * cellSize;
                float z = tile.Heights[vi]; // Use flat Heights array
                float u = c / (float)innerEdge;
                float v = r / (float)innerEdge;

                int offset = vi * 8;
                vertices[offset + 0] = x;
                vertices[offset + 1] = y;
                vertices[offset + 2] = z;
                // Normal computed later
                vertices[offset + 3] = 0;
                vertices[offset + 4] = 0;
                vertices[offset + 5] = 1;
                vertices[offset + 6] = u;
                vertices[offset + 7] = v;

                boundsMin = Vector3.Min(boundsMin, new Vector3(x, y, z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(x, y, z));
            }
        }

        // Inner vertices (16×16) — indices 289..544
        int innerBase = outerEdge * outerEdge;
        for (int r = 0; r < innerEdge; r++)
        {
            for (int c = 0; c < innerEdge; c++)
            {
                int vi = innerBase + r * innerEdge + c;
                float x = tileWorldX - (r + 0.5f) * cellSize;
                float y = tileWorldY - (c + 0.5f) * cellSize;
                float z = tile.Heights[vi]; // Use flat Heights array
                float u = (c + 0.5f) / innerEdge;
                float v = (r + 0.5f) / innerEdge;

                int offset = vi * 8;
                vertices[offset + 0] = x;
                vertices[offset + 1] = y;
                vertices[offset + 2] = z;
                vertices[offset + 3] = 0;
                vertices[offset + 4] = 0;
                vertices[offset + 5] = 1;
                vertices[offset + 6] = u;
                vertices[offset + 7] = v;

                boundsMin = Vector3.Min(boundsMin, new Vector3(x, y, z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(x, y, z));
            }
        }

        // Build index buffer: 16×16 cells, each split into 4 triangles via center vertex
        // Client topology: 1024 triangles = 3072 indices per cell (0xC00)
        var indices = new List<uint>(3072);
        for (int r = 0; r < innerEdge; r++)
        {
            for (int c = 0; c < innerEdge; c++)
            {
                // Outer corner indices
                uint v00 = (uint)(r * outerEdge + c);
                uint v10 = (uint)(r * outerEdge + c + 1);
                uint v01 = (uint)((r + 1) * outerEdge + c);
                uint v11 = (uint)((r + 1) * outerEdge + c + 1);
                // Inner center vertex
                uint center = (uint)(innerBase + r * innerEdge + c);

                // 4 triangles around center (matching client CreateAreaLowDetailIndices)
                indices.Add(v00); indices.Add(v10); indices.Add(center);
                indices.Add(v10); indices.Add(v11); indices.Add(center);
                indices.Add(v11); indices.Add(v01); indices.Add(center);
                indices.Add(v01); indices.Add(v00); indices.Add(center);
            }
        }

        // Compute per-vertex normals from triangle faces
        var normals = new Vector3[totalVerts];
        for (int i = 0; i < indices.Count; i += 3)
        {
            int i0 = (int)indices[i], i1 = (int)indices[i + 1], i2 = (int)indices[i + 2];
            var v0 = new Vector3(vertices[i0 * 8], vertices[i0 * 8 + 1], vertices[i0 * 8 + 2]);
            var v1 = new Vector3(vertices[i1 * 8], vertices[i1 * 8 + 1], vertices[i1 * 8 + 2]);
            var v2 = new Vector3(vertices[i2 * 8], vertices[i2 * 8 + 1], vertices[i2 * 8 + 2]);
            var normal = Vector3.Cross(v1 - v0, v2 - v0);
            normals[i0] += normal;
            normals[i1] += normal;
            normals[i2] += normal;
        }
        for (int i = 0; i < totalVerts; i++)
        {
            var n = Vector3.Normalize(normals[i]);
            if (float.IsNaN(n.X)) n = Vector3.UnitZ;
            vertices[i * 8 + 3] = n.X;
            vertices[i * 8 + 4] = n.Y;
            vertices[i * 8 + 5] = n.Z;
        }

        // Upload to GPU
        uint vao = _gl.GenVertexArray();
        _gl.BindVertexArray(vao);

        uint vbo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* ptr = vertices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        uint ebo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        var idxArray = indices.ToArray();
        fixed (uint* ptr = idxArray)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(idxArray.Length * sizeof(uint)), ptr, BufferUsageARB.StaticDraw);

        uint stride = 8 * sizeof(float);
        // Position (location 0)
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        // Normal (location 1)
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        // UV (location 2)
        _gl.EnableVertexAttribArray(2);
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));

        _gl.BindVertexArray(0);

        return new WdlTileMesh
        {
            Vao = vao, Vbo = vbo, Ebo = ebo,
            IndexCount = (uint)idxArray.Length,
            BoundsMin = boundsMin, BoundsMax = boundsMax,
            TileX = tileX, TileY = tileY
        };
    }

    // ── Shader ───────────────────────────────────────────────────────────

    private ShaderProgram CreateShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vFragPos;
out vec2 vTexCoord;

void main() {
    vFragPos = aPos;
    vNormal = aNormal;
    vTexCoord = aTexCoord;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
";

        string fragSrc = @"
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;
in vec2 vTexCoord;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform float uDistanceHazeStartFactor;
uniform float uDistanceHazeEndFactor;
uniform float uDistanceHazeColorBlend;
uniform float uDistanceHazeOpacityFloor;
uniform vec3 uCameraPos;
uniform float uOpacity;
uniform sampler2D uMinimapTexture;
uniform int uUseMinimapTexture;
uniform int uForceOpaque;

out vec4 FragColor;

vec3 ComputeHeightColor(float height) {
    if (height < 50.0) {
        return mix(vec3(0.2, 0.35, 0.15), vec3(0.3, 0.5, 0.2), clamp(height / 50.0, 0.0, 1.0));
    }

    if (height < 200.0) {
        float t = (height - 50.0) / 150.0;
        return mix(vec3(0.3, 0.5, 0.2), vec3(0.5, 0.4, 0.25), t);
    }

    float t = clamp((height - 200.0) / 300.0, 0.0, 1.0);
    return mix(vec3(0.5, 0.4, 0.25), vec3(0.6, 0.6, 0.55), t);
}

void main() {
    float height = vFragPos.z;
    vec3 baseColor = ComputeHeightColor(height);
    if (uUseMinimapTexture != 0) {
        vec3 minimapColor = texture(uMinimapTexture, vTexCoord).rgb;
        baseColor = mix(baseColor, minimapColor, 0.9);
    }

    // Lighting
    vec3 norm = normalize(vNormal);
    float diff = max(dot(norm, normalize(uLightDir)), 0.0);
    vec3 litColor = baseColor * (uAmbientColor + uLightColor * diff);

    // WDL-only haze: start the fade earlier than the main fog and let distant tiles
    // keep a faint silhouette by reducing opacity instead of snapping to solid dark terrain.
    float dist = length(vFragPos - uCameraPos);
    float hazeStart = max(0.0, uFogStart * uDistanceHazeStartFactor);
    float hazeEnd = max(hazeStart + 1.0, uFogEnd * uDistanceHazeEndFactor);
    float haze = smoothstep(hazeStart, hazeEnd, dist);
    vec3 finalColor = mix(litColor, uFogColor, haze * uDistanceHazeColorBlend);
    float finalOpacity = uForceOpaque != 0
        ? uOpacity
        : uOpacity * mix(1.0, uDistanceHazeOpacityFloor, haze);

    FragColor = vec4(finalColor, finalOpacity);
}
";

        return ShaderProgram.Create(_gl, vertSrc, fragSrc);
    }

    // ── Cleanup ──────────────────────────────────────────────────────────

    private void DisposeTileMesh(WdlTileMesh mesh)
    {
        _gl.DeleteVertexArray(mesh.Vao);
        _gl.DeleteBuffer(mesh.Vbo);
        _gl.DeleteBuffer(mesh.Ebo);
    }

    public void Dispose()
    {
        foreach (var mesh in _tileMeshes.Values)
            DisposeTileMesh(mesh);
        _tileMeshes.Clear();
        _tileData.Clear();
        _tileAlphas.Clear();
        _tileTargetAlphas.Clear();
        _tileShowReadyTimestamps.Clear();
        _detailedTileIndices.Clear();
        _shader.Dispose();
    }

    private class WdlTileMesh
    {
        public uint Vao, Vbo, Ebo;
        public uint IndexCount;
        public Vector3 BoundsMin, BoundsMax;
        public int TileX, TileY;
    }
}
