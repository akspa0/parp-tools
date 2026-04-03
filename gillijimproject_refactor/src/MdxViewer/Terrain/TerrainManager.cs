using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Manages terrain loading and rendering for an Alpha WDT map.
/// Handles AOI-based tile loading/unloading as the camera moves.
/// ADT parsing runs on a background thread; GPU mesh upload happens on the render thread.
/// Implements <see cref="ISceneRenderer"/> so it can be used as the active renderer in ViewerApp.
/// </summary>
public class TerrainManager : ISceneRenderer
{
    private readonly GL _gl;
    private readonly ITerrainAdapter _adapter;
    private readonly TerrainMeshBuilder _meshBuilder;
    private readonly TerrainRenderer _terrainRenderer;
    private readonly LiquidRenderer _liquidRenderer;
    private readonly IDataSource? _dataSource;

    // Loaded tiles: (tileX, tileY) → list of chunk meshes (GPU-resident)
    private readonly Dictionary<(int, int), List<TerrainChunkMesh>> _loadedTiles = new();

    // Persistent cache: parsed tile data stays in memory forever to avoid re-parsing from disk
    private readonly ConcurrentDictionary<(int, int), TileLoadResult> _tileCache = new();

    // Async streaming: background-parsed tiles waiting for GPU upload
    private readonly ConcurrentQueue<(int tx, int ty, TileLoadResult result)> _pendingTiles = new();
    // Tiles currently being loaded on background thread
    private readonly ConcurrentDictionary<(int, int), byte> _loadingTiles = new();
    private readonly List<(int tileX, int tileY)> _unloadScratch = new();
    private readonly List<(int tx, int ty, float priority)> _tilesToLoadScratch = new();
    private readonly HashSet<(int tileX, int tileY)> _ignoreTerrainHolesTiles = new();

    // AOI: how many tiles around the camera to keep loaded
    private const int AoiRadius = 4; // Load 9×9 tiles around camera for smoother streaming
    private const int UnloadRadius = AoiRadius + 1; // Keep one extra ring to prevent aggressive edge pop-out
    private const int MaxGpuUploadsPerFrame = 4;
    private const double MaxGpuUploadBudgetMs = 5.0;
    private const int MaxConcurrentMpqReads = 4; // Limit concurrent MPQ reads to avoid frame drops
    private readonly SemaphoreSlim _mpqReadSemaphore = new(MaxConcurrentMpqReads);

    /// <summary>Called when a tile is loaded, with per-tile placement data.</summary>
    public event Action<int, int, TileLoadResult>? OnTileLoaded;
    /// <summary>Called when a tile is unloaded.</summary>
    public event Action<int, int>? OnTileUnloaded;

    // When true, all tiles are pre-loaded and AOI streaming is disabled.
    // UpdateAOI still tracks camera position but skips tile load/unload.
    private bool _allTilesResident;

    // Camera tracking for AOI updates
    private int _lastCameraTileX = -1;
    private int _lastCameraTileY = -1;
    private Vector3 _cameraPos;
    private bool _disposed;

    // Stats
    public int LoadedTileCount => _loadedTiles.Count;
    public int LoadedChunkCount => _terrainRenderer.LoadedChunkCount;
    public bool IsTileLoaded(int tileX, int tileY) => _loadedTiles.ContainsKey((tileX, tileY));
    public IEnumerable<(int tileX, int tileY)> LoadedTiles => _loadedTiles.Keys;
    /// <summary>True while background tile loads or pending GPU uploads remain.</summary>
    public bool IsStreaming => !_loadingTiles.IsEmpty || !_pendingTiles.IsEmpty;
    public TerrainLighting Lighting => _terrainRenderer.Lighting;
    public TerrainRenderer Renderer => _terrainRenderer;
    public LiquidRenderer LiquidRenderer => _liquidRenderer;
    public string MapName { get; }
    public bool IgnoreTerrainHolesGlobally
    {
        get => _ignoreTerrainHolesGlobally;
        set
        {
            if (_ignoreTerrainHolesGlobally == value)
                return;

            _ignoreTerrainHolesGlobally = value;
            RebuildLoadedTilesForHoleVisibility();
        }
    }

    /// <summary>Exposes the terrain adapter for WorldScene to access placement data.</summary>
    public ITerrainAdapter Adapter => _adapter;

    private bool _ignoreTerrainHolesGlobally;

    /// <summary>
    /// Replace a tile's parsed chunk data and rebuild its loaded GPU meshes.
    /// Call on the render thread.
    /// </summary>
    public void ReplaceTileChunksAndRebuild(int tileX, int tileY, IReadOnlyList<TerrainChunkData> newChunks)
    {
            var replacementChunks = newChunks.ToList();

        if (!_tileCache.TryGetValue((tileX, tileY), out var cached))
        {
            cached = _adapter.LoadTileWithPlacements(tileX, tileY);
            _tileCache[(tileX, tileY)] = cached;
        }

        cached.Chunks.Clear();
            cached.Chunks.AddRange(replacementChunks);

            var meshes = new List<TerrainChunkMesh>(cached.Chunks.Count);
            foreach (var chunkData in cached.Chunks)
            {
                var mesh = BuildChunkMesh(chunkData);
                if (mesh != null)
                    meshes.Add(mesh);
            }

            if (cached.Chunks.Count == 0)
                return;

        var key = (tileX, tileY);
        if (_loadedTiles.TryGetValue(key, out var oldMeshes))
        {
            _terrainRenderer.RemoveChunks(oldMeshes);
            _liquidRenderer.RemoveChunksForTile(tileX, tileY);
            foreach (var mesh in oldMeshes)
                mesh.Dispose();
            _loadedTiles.Remove(key);
        }

        _loadedTiles[key] = meshes;
        _terrainRenderer.AddChunks(meshes, _adapter.TileTextures);
        _liquidRenderer.AddChunks(cached.Chunks);
    }

    public bool IsIgnoringTerrainHolesForTile(int tileX, int tileY)
        => _ignoreTerrainHolesGlobally || _ignoreTerrainHolesTiles.Contains((tileX, tileY));

    public bool SetIgnoreTerrainHolesForTile(int tileX, int tileY, bool enabled)
    {
        bool changed = enabled
            ? _ignoreTerrainHolesTiles.Add((tileX, tileY))
            : _ignoreTerrainHolesTiles.Remove((tileX, tileY));

        if (changed && !_ignoreTerrainHolesGlobally)
            RebuildLoadedTileForHoleVisibility(tileX, tileY);

        return changed;
    }

    private TerrainChunkMesh? BuildChunkMesh(TerrainChunkData chunkData)
        => _meshBuilder.BuildChunkMesh(chunkData, IsIgnoringTerrainHolesForTile(chunkData.TileX, chunkData.TileY));

    private void RebuildLoadedTilesForHoleVisibility()
    {
        foreach (var (tileX, tileY) in _loadedTiles.Keys.ToList())
            RebuildLoadedTileForHoleVisibility(tileX, tileY);
    }

    private void RebuildLoadedTileForHoleVisibility(int tileX, int tileY)
    {
        if (!_loadedTiles.ContainsKey((tileX, tileY)))
            return;

        if (!_tileCache.TryGetValue((tileX, tileY), out var result))
        {
            result = _adapter.LoadTileWithPlacements(tileX, tileY);
            _tileCache[(tileX, tileY)] = result;
        }

        ReplaceTileChunksAndRebuild(tileX, tileY, result.Chunks);
    }

    /// <summary>
    /// Try to get cached parsed tile data.
    /// </summary>
    public bool TryGetTileLoadResult(int tileX, int tileY, out TileLoadResult result)
    {
        if (_tileCache.TryGetValue((tileX, tileY), out var cached))
        {
            result = cached;
            return true;
        }

        result = new TileLoadResult();
        return false;
    }

    /// <summary>
    /// Get parsed tile data from cache, loading it if needed.
    /// </summary>
    public TileLoadResult GetOrLoadTileLoadResult(int tileX, int tileY)
    {
        if (_tileCache.TryGetValue((tileX, tileY), out var cached))
            return cached;

        if (!_adapter.TileExists(tileX, tileY))
            return new TileLoadResult();

        var loaded = _adapter.LoadTileWithPlacements(tileX, tileY);
        _tileCache[(tileX, tileY)] = loaded;
        return loaded;
    }

    public TerrainManager(GL gl, string wdtPath, IDataSource? dataSource)
    {
        _gl = gl;
        _dataSource = dataSource;
        MapName = Path.GetFileNameWithoutExtension(wdtPath);

        _adapter = new AlphaTerrainAdapter(wdtPath);
        _meshBuilder = new TerrainMeshBuilder(gl);
        _terrainRenderer = new TerrainRenderer(gl, dataSource, new TerrainLighting());
        _liquidRenderer = new LiquidRenderer(gl);

        // Find the center of populated tiles for initial camera placement
        FindInitialCameraPosition(out _cameraPos);
    }

    /// <summary>
    /// Create a TerrainManager with a pre-built terrain adapter (for Standard WDT, etc.).
    /// </summary>
    public TerrainManager(GL gl, ITerrainAdapter adapter, string mapName, IDataSource? dataSource)
    {
        _gl = gl;
        _dataSource = dataSource;
        MapName = mapName;

        _adapter = adapter;
        _meshBuilder = new TerrainMeshBuilder(gl);
        _terrainRenderer = new TerrainRenderer(gl, dataSource, new TerrainLighting());
        _liquidRenderer = new LiquidRenderer(gl);

        FindInitialCameraPosition(out _cameraPos);
    }

    /// <summary>
    /// Update terrain AOI based on camera position. Call each frame before Render.
    /// Queues new tiles for background loading and submits completed tiles to GPU.
    /// Uses a square AOI with one-ring unload hysteresis.
    /// </summary>
    public void UpdateAOI(Vector3 cameraPos)
    {
        _cameraPos = cameraPos;

        // If all tiles are pre-loaded, skip AOI streaming entirely
        if (_allTilesResident)
            return;

        // Submit any background-loaded tiles to GPU (render thread only)
        SubmitPendingTiles();

        // Convert camera world position to tile coordinates
        int tileX = (int)((WoWConstants.MapOrigin - cameraPos.X) / WoWConstants.ChunkSize);
        int tileY = (int)((WoWConstants.MapOrigin - cameraPos.Y) / WoWConstants.ChunkSize);

        tileX = Math.Clamp(tileX, 0, 63);
        tileY = Math.Clamp(tileY, 0, 63);

        // Only update if camera moved to a different tile
        if (tileX == _lastCameraTileX && tileY == _lastCameraTileY)
            return;

        _lastCameraTileX = tileX;
        _lastCameraTileY = tileY;

        // Determine which tiles should be loaded: a square AOI around the current camera tile.
        var desiredTiles = new HashSet<(int, int)>();
        for (int dy = -AoiRadius; dy <= AoiRadius; dy++)
        {
            for (int dx = -AoiRadius; dx <= AoiRadius; dx++)
            {
                int tx = tileX + dx;
                int ty = tileY + dy;
                if (tx >= 0 && tx < 64 && ty >= 0 && ty < 64 && _adapter.TileExists(tx, ty))
                    desiredTiles.Add((tx, ty));
            }
        }

        // Build a wider retention set for unloading. This hysteresis avoids dropping tiles
        // that are still near/in-view right when camera crosses tile boundaries.
        var unloadKeepTiles = new HashSet<(int, int)>();
        for (int dy = -UnloadRadius; dy <= UnloadRadius; dy++)
        {
            for (int dx = -UnloadRadius; dx <= UnloadRadius; dx++)
            {
                int tx = tileX + dx;
                int ty = tileY + dy;
                if (tx >= 0 && tx < 64 && ty >= 0 && ty < 64 && _adapter.TileExists(tx, ty))
                    unloadKeepTiles.Add((tx, ty));
            }
        }

        // Unload tiles outside retention radius — dispose GPU meshes but keep parsed data in cache.
        // Reuse a scratch list to avoid per-update LINQ/ToList allocations.
        _unloadScratch.Clear();
        foreach (var key in _loadedTiles.Keys)
        {
            if (!unloadKeepTiles.Contains(key))
                _unloadScratch.Add(key);
        }

        foreach (var key in _unloadScratch)
        {
            var meshes = _loadedTiles[key];
            _terrainRenderer.RemoveChunks(meshes);
            _liquidRenderer.RemoveChunksForTile(key.Item1, key.Item2);
            foreach (var chunk in meshes)
                chunk.Dispose();
            _loadedTiles.Remove(key);
            // NOTE: _tileCache retains the parsed data so re-entry is instant
            OnTileUnloaded?.Invoke(key.Item1, key.Item2);
        }

        // Queue new tiles for background loading.
        _tilesToLoadScratch.Clear();
        foreach (var (tx, ty) in desiredTiles)
        {
            if (_loadedTiles.ContainsKey((tx, ty)) || !_loadingTiles.TryAdd((tx, ty), 0))
                continue;
            float priority = MathF.Abs(tx - tileX) + MathF.Abs(ty - tileY);
            _tilesToLoadScratch.Add((tx, ty, priority));
            _loadingTiles.TryRemove((tx, ty), out _); // will re-add below
        }
        _tilesToLoadScratch.Sort((a, b) => a.priority.CompareTo(b.priority));

        foreach (var (tx, ty, _) in _tilesToLoadScratch)
        {
            if (!_loadingTiles.TryAdd((tx, ty), 0)) continue;

            // Check cache first — if we already parsed this tile, skip the expensive disk read
            if (_tileCache.TryGetValue((tx, ty), out var cached))
            {
                _pendingTiles.Enqueue((tx, ty, cached));
                _loadingTiles.TryRemove((tx, ty), out _);
                continue;
            }

            var capturedTx = tx;
            var capturedTy = ty;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                if (_disposed) return;
                // Throttle concurrent MPQ reads to avoid saturating I/O and causing frame drops
                _mpqReadSemaphore.Wait();
                try
                {
                    var result = _adapter.LoadTileWithPlacements(capturedTx, capturedTy);
                    _tileCache[(capturedTx, capturedTy)] = result; // Cache for future re-entry
                    if (!_disposed)
                        _pendingTiles.Enqueue((capturedTx, capturedTy, result));
                }
                catch (Exception ex)
                {
                    ViewerLog.Trace($"[TerrainManager] Background load ({capturedTx},{capturedTy}) failed: {ex.Message}");
                }
                finally
                {
                    _mpqReadSemaphore.Release();
                    _loadingTiles.TryRemove((capturedTx, capturedTy), out byte _);
                }
            });
        }
    }

    /// <summary>
    /// Submit background-loaded tiles to GPU. Must be called on the render thread.
    /// Limits uploads per frame to avoid stalls.
    /// </summary>
    private void SubmitPendingTiles()
    {
        int uploaded = 0;
        var uploadBudget = Stopwatch.StartNew();
        while (uploaded < MaxGpuUploadsPerFrame)
        {
            if (uploaded > 0 && uploadBudget.Elapsed.TotalMilliseconds >= MaxGpuUploadBudgetMs)
                break;

            if (!_pendingTiles.TryDequeue(out var pending))
                break;

            var (tx, ty, result) = pending;

            if (_loadedTiles.ContainsKey((tx, ty)))
                continue;

            var meshes = new List<TerrainChunkMesh>();
            foreach (var chunkData in result.Chunks)
            {
                var mesh = BuildChunkMesh(chunkData);
                if (mesh != null)
                    meshes.Add(mesh);
            }

            _loadedTiles[(tx, ty)] = meshes;
            _terrainRenderer.AddChunks(meshes, _adapter.TileTextures);
            _liquidRenderer.AddChunks(result.Chunks);

            // Notify listeners (WorldScene) about the new tile's placements
            OnTileLoaded?.Invoke(tx, ty, result);
            uploaded++;
        }
    }

    /// <summary>
    /// Load all tiles at once (for small maps or initial load). Synchronous.
    /// </summary>
    public void LoadAllTiles(Action<int, int, string>? onProgress = null)
    {
        // Disable AOI streaming immediately so UpdateAOI doesn't unload tiles
        // while we're loading them on the render thread.
        _allTilesResident = true;

        int total = _adapter.ExistingTiles.Count;
        int loaded = 0;
        ViewerLog.Trace($"[TerrainManager] Loading all {total} tiles...");
        foreach (int tileIdx in _adapter.ExistingTiles)
        {
            int tx = tileIdx / 64;
            int ty = tileIdx % 64;
            if (!_loadedTiles.ContainsKey((tx, ty)))
                LoadTileSynchronous(tx, ty);
            loaded++;
            onProgress?.Invoke(loaded, total, $"Tile ({tx},{ty})");
        }
        ViewerLog.Trace($"[TerrainManager] All tiles loaded: {_loadedTiles.Count} tiles, {LoadedChunkCount} chunks");
        _allTilesResident = true;
    }

    private void LoadTileSynchronous(int tileX, int tileY)
    {
        var result = _adapter.LoadTileWithPlacements(tileX, tileY);
        _tileCache[(tileX, tileY)] = result; // Cache for consistency with AOI path
        var meshes = new List<TerrainChunkMesh>();

        foreach (var chunkData in result.Chunks)
        {
            var mesh = BuildChunkMesh(chunkData);
            if (mesh != null)
                meshes.Add(mesh);
        }

        _loadedTiles[(tileX, tileY)] = meshes;
        _terrainRenderer.AddChunks(meshes, _adapter.TileTextures);
        _liquidRenderer.AddChunks(result.Chunks);

        OnTileLoaded?.Invoke(tileX, tileY, result);
    }

    private void FindInitialCameraPosition(out Vector3 cameraPos)
    {
        // Find the center of all existing tiles in WoW world coordinates
        // rendererX = wowY = MapOrigin - tileX * ChunkSize
        // rendererY = wowX = MapOrigin - tileY * ChunkSize
        if (_adapter.ExistingTiles.Count == 0)
        {
            cameraPos = Vector3.Zero;
            return;
        }

        if (MapName.Equals("development", StringComparison.OrdinalIgnoreCase) && _adapter.TileExists(0, 0))
        {
            float tileCenter = WoWConstants.MapOrigin - (WoWConstants.ChunkSize * 0.5f);
            cameraPos = new Vector3(tileCenter, tileCenter, 200f);
            return;
        }

        float sumX = 0, sumY = 0;
        foreach (int idx in _adapter.ExistingTiles)
        {
            // Alpha WDT MAIN is column-major: index = tileX*64+tileY
            int tx = idx / 64; // column (east-west)
            int ty = idx % 64; // row (north-south)
            sumX += WoWConstants.MapOrigin - tx * WoWConstants.ChunkSize;
            sumY += WoWConstants.MapOrigin - ty * WoWConstants.ChunkSize;
        }

        float avgX = sumX / _adapter.ExistingTiles.Count;
        float avgY = sumY / _adapter.ExistingTiles.Count;

        cameraPos = new Vector3(avgX, avgY, 200f);
    }

    /// <summary>
    /// Get the initial camera position for this map.
    /// </summary>
    public Vector3 GetInitialCameraPosition() => _cameraPos;

    // ── ISceneRenderer implementation ────────────────────────────────────

    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        _terrainRenderer.Render(view, proj, _cameraPos);
        // Liquid is rendered separately AFTER all opaque geometry (WMOs, MDX)
        // so objects below the water surface are visible through the transparent water.
        // See WorldScene.Render() or call RenderLiquid() explicitly.
    }

    /// <summary>
    /// Render with explicit camera position and optional frustum culler.
    /// </summary>
    public void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, FrustumCuller? frustum = null)
    {
        _cameraPos = cameraPos;
        _terrainRenderer.Render(view, proj, cameraPos, frustum);
    }

    /// <summary>
    /// Render liquid surfaces. Call AFTER all opaque geometry (terrain, WMOs, MDX)
    /// so objects below the water surface are visible through transparent water.
    /// </summary>
    public void RenderLiquid(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, float deltaTime = 0.016f)
    {
        _liquidRenderer.Render(view, proj, cameraPos, _terrainRenderer.Lighting, deltaTime);
    }

    public void ToggleWireframe()
    {
        _terrainRenderer.ToggleWireframe();
    }

    public int SubObjectCount => _loadedTiles.Count;

    public string GetSubObjectName(int index)
    {
        var keys = _loadedTiles.Keys.ToList();
        if (index < keys.Count)
            return $"Tile ({keys[index].Item1},{keys[index].Item2})";
        return "";
    }

    public bool GetSubObjectVisible(int index) => true; // All tiles always visible for now
    public void SetSubObjectVisible(int index, bool visible) { } // TODO: per-tile visibility

    public void Dispose()
    {
        _disposed = true;
        while (_pendingTiles.TryDequeue(out _)) { }
        _mpqReadSemaphore.Dispose();
        _liquidRenderer.Dispose();
        _terrainRenderer.Dispose();
        foreach (var meshes in _loadedTiles.Values)
            foreach (var mesh in meshes)
                mesh.Dispose();
        _loadedTiles.Clear();
    }
}
