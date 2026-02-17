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
    private readonly TerrainTileMeshBuilder _tileMeshBuilder;
    private readonly TerrainRenderer _terrainRenderer;
    private readonly LiquidRenderer _liquidRenderer;
    private readonly IDataSource? _dataSource;

    // Loaded tiles: (tileX, tileY) → batched tile mesh (GPU-resident)
    private readonly Dictionary<(int, int), TerrainTileMesh> _loadedTiles = new();

    // Persistent cache: parsed tile data stays in memory forever to avoid re-parsing from disk
    private readonly ConcurrentDictionary<(int, int), TileLoadResult> _tileCache = new();

    // Async streaming: background-parsed tiles waiting for GPU upload
    private readonly ConcurrentQueue<(int tx, int ty, TileLoadResult result)> _pendingTiles = new();
    // Tiles currently being loaded on background thread
    private readonly ConcurrentDictionary<(int, int), byte> _loadingTiles = new();
    private readonly List<(int tileX, int tileY)> _unloadScratch = new();
    private readonly List<(int tx, int ty, float priority)> _tilesToLoadScratch = new();

    // AOI: how many tiles around the camera to keep loaded
    private const int AoiRadius = 4; // Load 9×9 tiles around camera for smoother streaming
    private const int UnloadRadius = AoiRadius + 1; // Keep one extra ring to prevent aggressive edge pop-out
    private const int AoiForwardExtra = 3; // Extra tiles ahead of camera heading
    private const int MaxGpuUploadsPerFrame = 4; // Avoid large upload bursts that can stall render thread
    private const double MaxGpuUploadBudgetMs = 5.0; // Native-like per-frame prep budget for non-priority work
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
    private Vector3 _lastCameraPos; // For computing movement direction
    private Vector2 _cameraHeading; // Normalized XY movement direction
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

    /// <summary>Exposes the terrain adapter for WorldScene to access placement data.</summary>
    public ITerrainAdapter Adapter => _adapter;

    /// <summary>
    /// Try to get parsed tile data from the persistent cache (does not require the tile to be GPU-loaded).
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

    public TerrainManager(GL gl, string wdtPath, IDataSource? dataSource)
    {
        _gl = gl;
        _dataSource = dataSource;
        MapName = Path.GetFileNameWithoutExtension(wdtPath);

        _adapter = new AlphaTerrainAdapter(wdtPath);
        _tileMeshBuilder = new TerrainTileMeshBuilder(gl);
        _terrainRenderer = new TerrainRenderer(gl, dataSource, new TerrainLighting());
        _terrainRenderer.UseWorldUvForDiffuse = false;
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
        _tileMeshBuilder = new TerrainTileMeshBuilder(gl);
        _terrainRenderer = new TerrainRenderer(gl, dataSource, new TerrainLighting());
        _terrainRenderer.UseWorldUvForDiffuse = adapter is not AlphaTerrainAdapter;
        _liquidRenderer = new LiquidRenderer(gl);

        FindInitialCameraPosition(out _cameraPos);
    }

    /// <summary>
    /// Update terrain AOI based on camera position. Call each frame before Render.
    /// Queues new tiles for background loading and submits completed tiles to GPU.
    /// Uses directional loading: tiles ahead of camera heading are prioritized,
    /// tiles behind camera are unloaded first.
    /// </summary>
    public void UpdateAOI(Vector3 cameraPos)
    {
        _cameraPos = cameraPos;

        // If all tiles are pre-loaded, skip AOI streaming entirely
        if (_allTilesResident)
            return;

        // Submit any background-loaded tiles to GPU (render thread only)
        SubmitPendingTiles();

        // Track camera movement direction for directional loading
        var delta = cameraPos - _lastCameraPos;
        if (delta.LengthSquared() > 1f) // Only update heading if camera moved meaningfully
        {
            var dir2d = new Vector2(delta.X, delta.Y);
            if (dir2d.LengthSquared() > 0.01f)
                _cameraHeading = Vector2.Normalize(dir2d);
        }
        _lastCameraPos = cameraPos;

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

        // Determine which tiles should be loaded:
        // Base AOI (AoiRadius square) + extra tiles ahead of camera heading
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

        // Add extra tiles ahead of camera heading (directional lookahead)
        if (_cameraHeading.LengthSquared() > 0.5f)
        {
            // World X maps to tile via (MapOrigin - X) / ChunkSize, so moving +X = decreasing tileX
            // Heading is in world coords, so negate for tile coords
            int headDx = _cameraHeading.X > 0.3f ? -1 : _cameraHeading.X < -0.3f ? 1 : 0;
            int headDy = _cameraHeading.Y > 0.3f ? -1 : _cameraHeading.Y < -0.3f ? 1 : 0;

            for (int step = 1; step <= AoiForwardExtra; step++)
            {
                int fx = tileX + headDx * (AoiRadius + step);
                int fy = tileY + headDy * (AoiRadius + step);
                if (fx >= 0 && fx < 64 && fy >= 0 && fy < 64 && _adapter.TileExists(fx, fy))
                    desiredTiles.Add((fx, fy));
                // Also add diagonal neighbors for smoother coverage
                if (headDx != 0 && headDy != 0)
                {
                    int fx2 = tileX + headDx * (AoiRadius + step);
                    int fy2 = tileY;
                    if (fx2 >= 0 && fx2 < 64 && fy2 >= 0 && fy2 < 64 && _adapter.TileExists(fx2, fy2))
                        desiredTiles.Add((fx2, fy2));
                    fx2 = tileX;
                    fy2 = tileY + headDy * (AoiRadius + step);
                    if (fx2 >= 0 && fx2 < 64 && fy2 >= 0 && fy2 < 64 && _adapter.TileExists(fx2, fy2))
                        desiredTiles.Add((fx2, fy2));
                }
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
            var tileMesh = _loadedTiles[key];
            _terrainRenderer.RemoveTile(key.Item1, key.Item2);
            _liquidRenderer.RemoveChunksForTile(key.Item1, key.Item2);
            tileMesh.Dispose();
            _loadedTiles.Remove(key);
            // NOTE: _tileCache retains the parsed data so re-entry is instant
            OnTileUnloaded?.Invoke(key.Item1, key.Item2);
        }

        // Queue new tiles for background loading, prioritized by direction
        // Sort: tiles ahead of camera heading load first, tiles behind load last
        _tilesToLoadScratch.Clear();
        foreach (var (tx, ty) in desiredTiles)
        {
            if (_loadedTiles.ContainsKey((tx, ty)) || !_loadingTiles.TryAdd((tx, ty), 0))
                continue;
            // Priority: lower = load first. Tiles in heading direction get lower priority values.
            float priority = 0f;
            if (_cameraHeading.LengthSquared() > 0.5f)
            {
                // Tile offset from camera in world-ish direction
                float offX = -(tx - tileX); // negate because tile coords are inverted from world
                float offY = -(ty - tileY);
                var off2d = new Vector2(offX, offY);
                if (off2d.LengthSquared() > 0.01f)
                {
                    float dot = Vector2.Dot(Vector2.Normalize(off2d), _cameraHeading);
                    priority = -dot; // ahead = negative priority = loads first
                }
            }
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

            var (tileMesh, chunkInfos) = _tileMeshBuilder.BuildTileMesh(tx, ty, result.Chunks);
            if (tileMesh == null)
                continue;

            _loadedTiles[(tx, ty)] = tileMesh;
            if (!_adapter.TileTextures.TryGetValue((tx, ty), out var texNames))
                texNames = new List<string>();
            _terrainRenderer.AddTile(tileMesh, texNames, chunkInfos);
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

        var (tileMesh, chunkInfos) = _tileMeshBuilder.BuildTileMesh(tileX, tileY, result.Chunks);
        if (tileMesh == null)
            return;

        _loadedTiles[(tileX, tileY)] = tileMesh;
        if (!_adapter.TileTextures.TryGetValue((tileX, tileY), out var texNames))
            texNames = new List<string>();
        _terrainRenderer.AddTile(tileMesh, texNames, chunkInfos);
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
        foreach (var mesh in _loadedTiles.Values)
            mesh.Dispose();
        _loadedTiles.Clear();
    }
}
