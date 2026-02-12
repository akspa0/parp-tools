using System.Collections.Concurrent;
using System.Numerics;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Manages terrain loading and rendering from a VLM dataset project folder.
/// Mirrors <see cref="TerrainManager"/> but reads from VLM JSON instead of Alpha WDT/ADT.
/// Tile JSON parsing runs on a background thread; GPU mesh upload happens on the render thread.
/// </summary>
public class VlmTerrainManager : ISceneRenderer
{
    private readonly GL _gl;
    private readonly VlmProjectLoader _loader;
    private readonly TerrainMeshBuilder _meshBuilder;
    private readonly TerrainRenderer _terrainRenderer;
    private readonly LiquidRenderer _liquidRenderer;

    // Loaded tiles: (tileX, tileY) → list of chunk meshes
    private readonly Dictionary<(int, int), List<TerrainChunkMesh>> _loadedTiles = new();

    // Async streaming: background-parsed tiles waiting for GPU upload
    private readonly ConcurrentQueue<(int tx, int ty, TileLoadResult result)> _pendingTiles = new();
    // Tiles currently being loaded on background thread
    private readonly ConcurrentDictionary<(int, int), byte> _loadingTiles = new();

    // AOI
    private const int AoiRadius = 3;
    private const int MaxGpuUploadsPerFrame = 2;

    /// <summary>Called when a tile is loaded, with per-tile placement data.</summary>
    public event Action<int, int, TileLoadResult>? OnTileLoaded;
    /// <summary>Called when a tile is unloaded.</summary>
    public event Action<int, int>? OnTileUnloaded;

    private int _lastCameraTileX = -1;
    private int _lastCameraTileY = -1;
    private Vector3 _cameraPos;
    private bool _disposed;

    public int LoadedTileCount => _loadedTiles.Count;
    public int LoadedChunkCount => _terrainRenderer.LoadedChunkCount;
    public bool IsTileLoaded(int tileX, int tileY) => _loadedTiles.ContainsKey((tileX, tileY));
    public TerrainLighting Lighting => _terrainRenderer.Lighting;
    public TerrainRenderer Renderer => _terrainRenderer;
    public LiquidRenderer LiquidRenderer => _liquidRenderer;
    public string MapName => _loader.MapName;
    public VlmProjectLoader Loader => _loader;

    public VlmTerrainManager(GL gl, string projectRoot)
    {
        _gl = gl;
        _loader = new VlmProjectLoader(projectRoot);
        _meshBuilder = new TerrainMeshBuilder(gl);
        _terrainRenderer = new TerrainRenderer(gl, null, new TerrainLighting(),
            texturePathResolver: _loader.ResolveTexturePath);
        _liquidRenderer = new LiquidRenderer(gl);

        FindInitialCameraPosition(out _cameraPos);
    }

    /// <summary>
    /// Update terrain AOI based on camera position. Call each frame before Render.
    /// Queues new tiles for background loading and submits completed tiles to GPU.
    /// </summary>
    public void UpdateAOI(Vector3 cameraPos)
    {
        _cameraPos = cameraPos;

        // Submit any background-loaded tiles to GPU (render thread only)
        SubmitPendingTiles();

        int tileX = (int)((WoWConstants.MapOrigin - cameraPos.X) / WoWConstants.ChunkSize);
        int tileY = (int)((WoWConstants.MapOrigin - cameraPos.Y) / WoWConstants.ChunkSize);
        tileX = Math.Clamp(tileX, 0, 63);
        tileY = Math.Clamp(tileY, 0, 63);

        if (tileX == _lastCameraTileX && tileY == _lastCameraTileY)
            return;

        _lastCameraTileX = tileX;
        _lastCameraTileY = tileY;

        var desiredTiles = new HashSet<(int, int)>();
        for (int dy = -AoiRadius; dy <= AoiRadius; dy++)
        for (int dx = -AoiRadius; dx <= AoiRadius; dx++)
        {
            int tx = tileX + dx;
            int ty = tileY + dy;
            if (tx >= 0 && tx < 64 && ty >= 0 && ty < 64 &&
                _loader.TileCoords.Contains((tx, ty)))
                desiredTiles.Add((tx, ty));
        }

        // Unload tiles no longer in AOI
        var toUnload = _loadedTiles.Keys.Where(k => !desiredTiles.Contains(k)).ToList();
        foreach (var key in toUnload)
        {
            var meshes = _loadedTiles[key];
            _terrainRenderer.RemoveChunks(meshes);
            _liquidRenderer.RemoveChunksForTile(key.Item1, key.Item2);
            foreach (var chunk in meshes)
                chunk.Dispose();
            _loadedTiles.Remove(key);
            OnTileUnloaded?.Invoke(key.Item1, key.Item2);
        }

        // Queue new tiles for background loading
        foreach (var (tx, ty) in desiredTiles)
        {
            if (!_loadedTiles.ContainsKey((tx, ty)) && _loadingTiles.TryAdd((tx, ty), 0))
            {
                var capturedTx = tx;
                var capturedTy = ty;
                ThreadPool.QueueUserWorkItem(_ =>
                {
                    if (_disposed) return;
                    try
                    {
                        var result = _loader.LoadTile(capturedTx, capturedTy);
                        if (!_disposed)
                            _pendingTiles.Enqueue((capturedTx, capturedTy, result));
                    }
                    catch (Exception ex)
                    {
                        ViewerLog.Trace($"[VlmTerrainManager] Background load ({capturedTx},{capturedTy}) failed: {ex.Message}");
                    }
                    finally
                    {
                        _loadingTiles.TryRemove((capturedTx, capturedTy), out byte _);
                    }
                });
            }
        }
    }

    /// <summary>
    /// Submit background-loaded tiles to GPU. Must be called on the render thread.
    /// Limits uploads per frame to avoid stalls.
    /// </summary>
    private void SubmitPendingTiles()
    {
        int uploaded = 0;
        while (uploaded < MaxGpuUploadsPerFrame && _pendingTiles.TryDequeue(out var pending))
        {
            var (tx, ty, result) = pending;

            // Skip if tile was already loaded (e.g. duplicate queue) or no longer desired
            if (_loadedTiles.ContainsKey((tx, ty)))
                continue;

            var meshes = new List<TerrainChunkMesh>();
            foreach (var chunkData in result.Chunks)
            {
                var mesh = _meshBuilder.BuildChunkMesh(chunkData);
                if (mesh != null)
                    meshes.Add(mesh);
            }

            _loadedTiles[(tx, ty)] = meshes;
            _terrainRenderer.AddChunks(meshes, _loader.TileTextures);
            _liquidRenderer.AddChunks(result.Chunks);
            OnTileLoaded?.Invoke(tx, ty, result);
            uploaded++;
        }
    }

    private void FindInitialCameraPosition(out Vector3 cameraPos)
    {
        if (_loader.TileCoords.Count == 0)
        {
            cameraPos = Vector3.Zero;
            return;
        }

        float sumX = 0, sumY = 0;
        foreach (var (tx, ty) in _loader.TileCoords)
        {
            sumX += WoWConstants.MapOrigin - tx * WoWConstants.ChunkSize;
            sumY += WoWConstants.MapOrigin - ty * WoWConstants.ChunkSize;
        }

        cameraPos = new Vector3(
            sumX / _loader.TileCoords.Count,
            sumY / _loader.TileCoords.Count,
            200f);
    }

    public Vector3 GetInitialCameraPosition() => _cameraPos;

    // ── ISceneRenderer ──

    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        _terrainRenderer.Render(view, proj, _cameraPos);
        // Liquid rendered separately via RenderLiquid() after all opaque geometry
    }

    public void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos)
    {
        _cameraPos = cameraPos;
        _terrainRenderer.Render(view, proj, cameraPos);
    }

    /// <summary>
    /// Render liquid surfaces. Call AFTER all opaque geometry.
    /// </summary>
    public void RenderLiquid(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, float deltaTime = 0.016f)
    {
        _liquidRenderer.Render(view, proj, cameraPos, _terrainRenderer.Lighting, deltaTime);
    }

    public void ToggleWireframe() => _terrainRenderer.ToggleWireframe();

    public int SubObjectCount => _loadedTiles.Count;

    public string GetSubObjectName(int index)
    {
        var keys = _loadedTiles.Keys.ToList();
        return index < keys.Count ? $"Tile ({keys[index].Item1},{keys[index].Item2})" : "";
    }

    public bool GetSubObjectVisible(int index) => true;
    public void SetSubObjectVisible(int index, bool visible) { }

    public void Dispose()
    {
        _disposed = true;
        // Drain pending queue
        while (_pendingTiles.TryDequeue(out _)) { }
        _liquidRenderer.Dispose();
        _terrainRenderer.Dispose();
        foreach (var meshes in _loadedTiles.Values)
            foreach (var mesh in meshes)
                mesh.Dispose();
        _loadedTiles.Clear();
    }
}
