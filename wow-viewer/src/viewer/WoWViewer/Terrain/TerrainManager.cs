using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using WowViewer.Core.Runtime.World;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using Silk.NET.OpenGL;

namespace WoWViewer.Terrain;

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
    private readonly DirectionalTileSelector _directionalTileSelector;
    private readonly CameraTileWindowSelector _cameraTileWindowSelector;
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
    private readonly HashSet<(int tileX, int tileY)> _ignoreTerrainHolesTiles = new();
    // Explicit capture-path pins extend the normal AOI retention lease without
    // changing the camera-driven detailed-tile budget. They are cleared when
    // capture preparation ends.
    private readonly HashSet<(int tileX, int tileY)> _capturePreloadTiles = new();

    // Visibility remains strict and directional. Residency is a separate,
    // camera-centered bounded window so nearby tiles can stream in without
    // expanding the per-frame object/terrain submission set.
    private const int MinDetailedTileCandidateRadius = 1;
    private const int MaxDetailedTileCandidateRadius = 1;
    private const int MinDetailedTileCount = 1;
    private const int MaxDetailedTileCount = 4;
    public const int MinRetainedTileRadius = 1;
    private const int DefaultRetainedTileRadius = 2;
    public const int MaxRetainedTileRadius = 3;
    public const int MaxManualDetailedTileCount = 4;
    public const float DirectionalTileFovDegrees = 45f;
    private const int MaxGpuUploadsPerFrame = 6;
    private const double MaxGpuUploadBudgetMs = 7.0;
    private const int MaxConcurrentMpqReads = 4; // Limit concurrent MPQ reads to avoid frame drops
    private readonly SemaphoreSlim _mpqReadSemaphore = new(MaxConcurrentMpqReads);
    // Terrain adapters retain parse-time lookup/reporting state. Keep the adapter boundary
    // serialized while allowing the surrounding streaming scheduler to remain asynchronous.
    private readonly object _adapterLoadLock = new();

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
    private int _lastDetailedTileCandidateRadius = -1;
    private int _lastTargetDetailedTileCount = -1;
    private int _lastTargetRetainedTileRadius = -1;
    private int _detailedTileCountOverride;
    private int _effectiveDetailedTileCount = MaxDetailedTileCount;
    private int _effectiveRetainedTileCount;
    private int _retainedTileRadius = DefaultRetainedTileRadius;
    private Vector3 _cameraPos;
    private Vector3 _lastCameraPos;
    private Vector2 _cameraHeading;
    private Vector2 _lastSelectedHeading;
    private readonly List<(int tileX, int tileY)> _lastSelectedTiles = new(4);
    private readonly List<(int tileX, int tileY)> _lastRetainedTiles = new(25);
    private bool _disposed;

    // Stats
    public int LoadedTileCount => _loadedTiles.Count;
    public int LoadedChunkCount => _terrainRenderer.LoadedChunkCount;
    public bool IsTileLoaded(int tileX, int tileY) => _loadedTiles.ContainsKey((tileX, tileY));
    public IEnumerable<(int tileX, int tileY)> LoadedTiles => _loadedTiles.Keys;
    public int CameraTileX => _lastCameraTileX;
    public int CameraTileY => _lastCameraTileY;
    public int LastUnloadedTileX { get; private set; } = -1;
    public int LastUnloadedTileY { get; private set; } = -1;
    public int TileUnloadEventCount { get; private set; }
    /// <summary>True while background tile loads or pending GPU uploads remain.</summary>
    public bool IsStreaming => !_loadingTiles.IsEmpty || !_pendingTiles.IsEmpty;
    public int BackgroundTileLoadCount => _loadingTiles.Count;
    public int PendingGpuTileUploadCount => _pendingTiles.Count;
    public int PendingTerrainLoadCount => _loadingTiles.Count + _pendingTiles.Count;
    public int CapturePreloadTileCount => _capturePreloadTiles.Count;
    public IReadOnlyCollection<(int tileX, int tileY)> CapturePreloadTiles => _capturePreloadTiles;
    public IReadOnlyList<(int tileX, int tileY)> LastSelectedTiles => _lastSelectedTiles;
    public IReadOnlyList<(int tileX, int tileY)> LastRetainedTiles => _lastRetainedTiles;
    public int LastFrameActiveTileCount => _lastSelectedTiles.Count;
    public int LastFrameRetainedTileCount => _lastRetainedTiles.Count;
    public int LastFrameDetailedTileDrawCalls => _terrainRenderer.LastFrameDrawCalls;
    public bool LastDirectionalTileInvariantPassed
        => LastFrameActiveTileCount <= 4
            && (_capturePreloadTiles.Count > 0 || LastFrameDetailedTileDrawCalls <= 4);
    public TerrainLighting Lighting => _terrainRenderer.Lighting;
    public TerrainRenderer Renderer => _terrainRenderer;
    public LiquidRenderer LiquidRenderer => _liquidRenderer;
    public bool TerrainVisible { get; set; } = true;
    public string MapName { get; }
    public int DetailedTileCountOverride
    {
        get => _detailedTileCountOverride;
        set
        {
            int clamped = Math.Clamp(value, 0, MaxManualDetailedTileCount);
            if (_detailedTileCountOverride == clamped)
                return;

            _detailedTileCountOverride = clamped;
            InvalidateStreamingTargets();
        }
    }

    /// <summary>
    /// Pins a bounded set of tiles for an explicit capture-path preload. The
    /// normal AOI still controls the camera-facing detailed set; pinned tiles
    /// are only added to the load/retention sets while the lease is active.
    /// </summary>
    public void SetCapturePreloadTiles(IEnumerable<(int tileX, int tileY)> tiles)
    {
        ArgumentNullException.ThrowIfNull(tiles);

        _capturePreloadTiles.Clear();
        foreach (var (tileX, tileY) in tiles)
        {
            if (tileX >= 0 && tileX < 64 && tileY >= 0 && tileY < 64 && _adapter.TileExists(tileX, tileY))
                _capturePreloadTiles.Add((tileX, tileY));
        }

        InvalidateStreamingTargets();
    }

    /// <summary>Release the current capture-path tile lease.</summary>
    public void ClearCapturePreloadTiles()
    {
        if (_capturePreloadTiles.Count == 0)
            return;

        _capturePreloadTiles.Clear();
        InvalidateStreamingTargets();
    }

    public int EffectiveDetailedTileCount => _effectiveDetailedTileCount;
    public int EffectiveRetainedTileCount => _effectiveRetainedTileCount;
    public int EffectiveRetainedTileRadius => _retainedTileRadius;
    public int RetainedTileRadius
    {
        get => _retainedTileRadius;
        set
        {
            int clamped = Math.Clamp(value, MinRetainedTileRadius, MaxRetainedTileRadius);
            if (_retainedTileRadius == clamped)
                return;

            _retainedTileRadius = clamped;
            InvalidateStreamingTargets();
        }
    }
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
            cached = LoadTileWithPlacementsSerialized(tileX, tileY);
            _tileCache[(tileX, tileY)] = cached;
        }

        cached.Chunks.Clear();
        cached.Chunks.AddRange(replacementChunks);

        var key = (tileX, tileY);
        if (_loadedTiles.TryGetValue(key, out var oldTileMesh))
        {
            _terrainRenderer.RemoveTile(tileX, tileY);
            _liquidRenderer.RemoveChunksForTile(tileX, tileY);
            oldTileMesh.Dispose();
            _loadedTiles.Remove(key);
        }

        if (cached.Chunks.Count == 0)
            return;

        var (tileMesh, chunkInfos) = BuildTileMesh(tileX, tileY, cached.Chunks);
        if (tileMesh == null)
            return;

        _loadedTiles[key] = tileMesh;
        if (!_adapter.TileTextures.TryGetValue(key, out var textureNames))
            textureNames = new List<string>();
        _terrainRenderer.AddTile(tileMesh, textureNames, chunkInfos, fadeIn: false);
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

    private (TerrainTileMesh? tileMesh, List<TerrainChunkInfo> chunkInfos) BuildTileMesh(int tileX, int tileY, IReadOnlyList<TerrainChunkData> chunks)
    {
        IReadOnlyList<TerrainChunkData> chunksToBuild = chunks;
        if (IsIgnoringTerrainHolesForTile(tileX, tileY))
        {
            var adjustedChunks = new List<TerrainChunkData>(chunks.Count);
            foreach (var chunk in chunks)
            {
                adjustedChunks.Add(chunk.HoleMask == 0 ? chunk : CloneChunkWithHoleMask(chunk, 0));
            }

            chunksToBuild = adjustedChunks;
        }

        return _tileMeshBuilder.BuildTileMesh(tileX, tileY, chunksToBuild);
    }

    private static TerrainChunkData CloneChunkWithHoleMask(TerrainChunkData chunk, int holeMask)
        => new()
        {
            McinIndex = chunk.McinIndex,
            TileX = chunk.TileX,
            TileY = chunk.TileY,
            ChunkX = chunk.ChunkX,
            ChunkY = chunk.ChunkY,
            Heights = chunk.Heights,
            Normals = chunk.Normals,
            HoleMask = holeMask,
            Layers = chunk.Layers,
            AlphaMaps = chunk.AlphaMaps,
            ShadowMap = chunk.ShadowMap,
            MccvColors = chunk.MccvColors,
            Liquid = chunk.Liquid,
            WorldPosition = chunk.WorldPosition,
            AreaId = chunk.AreaId,
            McnkFlags = chunk.McnkFlags,
            AlphaSourceFlags = chunk.AlphaSourceFlags,
        };

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
            result = LoadTileWithPlacementsSerialized(tileX, tileY);
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

    public bool TryUpdateCachedPlacementPosition(ObjectType objectType, int tileX, int tileY, int placementEntryIndex, Vector3 newPosition)
    {
        if (!_tileCache.TryGetValue((tileX, tileY), out TileLoadResult? cached))
            return false;

        switch (objectType)
        {
            case ObjectType.Mdx:
                if (placementEntryIndex < 0 || placementEntryIndex >= cached.MddfPlacements.Count)
                    return false;

                MddfPlacement mddf = cached.MddfPlacements[placementEntryIndex];
                mddf.Position = newPosition;
                cached.MddfPlacements[placementEntryIndex] = mddf;
                return true;

            case ObjectType.Wmo:
                if (placementEntryIndex < 0 || placementEntryIndex >= cached.ModfPlacements.Count)
                    return false;

                ModfPlacement modf = cached.ModfPlacements[placementEntryIndex];
                Vector3 delta = newPosition - modf.Position;
                modf.Position = newPosition;
                modf.BoundsMin += delta;
                modf.BoundsMax += delta;
                cached.ModfPlacements[placementEntryIndex] = modf;
                return true;

            default:
                return false;
        }
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

        var loaded = LoadTileWithPlacementsSerialized(tileX, tileY);
        _tileCache[(tileX, tileY)] = loaded;
        return loaded;
    }

    public TerrainManager(GL gl, string wdtPath, IDataSource? dataSource)
    {
        _gl = gl;
        _dataSource = dataSource;
        MapName = Path.GetFileNameWithoutExtension(wdtPath);

        _adapter = new AlphaTerrainAdapter(wdtPath);
        _directionalTileSelector = new DirectionalTileSelector(
            WoWConstants.MapOrigin,
            WoWConstants.ChunkSize,
            64,
            (tileX, tileY) => _adapter.TileExists(tileX, tileY));
        _cameraTileWindowSelector = new CameraTileWindowSelector(
            WoWConstants.MapOrigin,
            WoWConstants.ChunkSize,
            64,
            (tileX, tileY) => _adapter.TileExists(tileX, tileY));
        _tileMeshBuilder = new TerrainTileMeshBuilder(gl);
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
        _directionalTileSelector = new DirectionalTileSelector(
            WoWConstants.MapOrigin,
            WoWConstants.ChunkSize,
            64,
            (tileX, tileY) => _adapter.TileExists(tileX, tileY));
        _cameraTileWindowSelector = new CameraTileWindowSelector(
            WoWConstants.MapOrigin,
            WoWConstants.ChunkSize,
            64,
            (tileX, tileY) => _adapter.TileExists(tileX, tileY));
        _tileMeshBuilder = new TerrainTileMeshBuilder(gl);
        _terrainRenderer = new TerrainRenderer(gl, dataSource, new TerrainLighting());
        _liquidRenderer = new LiquidRenderer(gl);

        FindInitialCameraPosition(out _cameraPos);
    }

    /// <summary>
    /// Get or set the current overlay map name on the terrain adapter.
    /// </summary>
    public string? OverlayMapName => _adapter.OverlayMapName;

    /// <summary>
    /// Set or clear the secondary overlay map. Evicts and re-streams affected tiles.
    /// </summary>
    public void SetOverlayMap(string? overlayMapName)
    {
        string? current = _adapter.OverlayMapName;
        string? incoming = string.IsNullOrWhiteSpace(overlayMapName) ? null : overlayMapName.Trim();
        if (string.Equals(current, incoming, StringComparison.OrdinalIgnoreCase))
            return;

        _adapter.OverlayMapName = incoming;

        // Evict all cached tiles so they reload with overlay resolution
        var keysToEvict = _tileCache.Keys.ToList();
        foreach (var key in keysToEvict)
        {
            if (_loadedTiles.TryGetValue(key, out var oldMesh))
            {
                _terrainRenderer.RemoveTile(key.Item1, key.Item2);
                _liquidRenderer.RemoveChunksForTile(key.Item1, key.Item2);
                oldMesh.Dispose();
                _loadedTiles.Remove(key);
                OnTileUnloaded?.Invoke(key.Item1, key.Item2);
            }

            _tileCache.TryRemove(key, out _);
        }

        InvalidateStreamingTargets();
        ViewerLog.Important(ViewerLog.Category.Terrain,
            incoming != null
                ? $"[TerrainManager] Secondary overlay map set to '{incoming}'. Evicted {keysToEvict.Count} cached tiles."
                : $"[TerrainManager] Secondary overlay map cleared. Evicted {keysToEvict.Count} cached tiles.");
    }

    /// <summary>
    /// Update terrain AOI based on camera position. Call each frame before Render.
    /// Queues new tiles for background loading and submits completed tiles to GPU.
    /// Uses the strict directional baseline: active tile plus at most three
    /// adjacent tiles in the camera-facing 45-degree cone.
    /// </summary>
    public void UpdateAOI(Vector3 cameraPos, Vector3? cameraForward = null)
    {
        _cameraPos = cameraPos;

        // Submit any background-loaded tiles to GPU (render thread only)
        SubmitPendingTiles();

        if (cameraForward.HasValue)
        {
            Vector2 forward2d = new(cameraForward.Value.X, cameraForward.Value.Y);
            if (forward2d.LengthSquared() > 1e-4f)
                _cameraHeading = Vector2.Normalize(forward2d);
        }
        else
        {
            Vector3 delta = cameraPos - _lastCameraPos;
            Vector2 delta2d = new(delta.X, delta.Y);
            if (delta2d.LengthSquared() > 1f)
                _cameraHeading = Vector2.Normalize(delta2d);
        }

        _lastCameraPos = cameraPos;

        // Convert camera world position to tile coordinates
        int tileX = (int)((WoWConstants.MapOrigin - cameraPos.X) / WoWConstants.ChunkSize);
        int tileY = (int)((WoWConstants.MapOrigin - cameraPos.Y) / WoWConstants.ChunkSize);

        tileX = Math.Clamp(tileX, 0, 63);
        tileY = Math.Clamp(tileY, 0, 63);

        ComputeStreamingTargets(
            _terrainRenderer.Lighting.FogEnd,
            out int detailedTileCandidateRadius,
            out int targetDetailedTileCount,
            out int targetRetainedTileRadius);
        _effectiveDetailedTileCount = targetDetailedTileCount;

        // Re-evaluate when the camera crosses a tile boundary, the requested
        // budget changes, or the camera turns far enough to change admission.
        if (tileX == _lastCameraTileX
            && tileY == _lastCameraTileY
            && detailedTileCandidateRadius == _lastDetailedTileCandidateRadius
            && targetDetailedTileCount == _lastTargetDetailedTileCount
            && targetRetainedTileRadius == _lastTargetRetainedTileRadius
            && Vector2.DistanceSquared(_cameraHeading, _lastSelectedHeading) <= 1e-4f)
            return;

        _lastCameraTileX = tileX;
        _lastCameraTileY = tileY;
        _lastDetailedTileCandidateRadius = detailedTileCandidateRadius;
        _lastTargetDetailedTileCount = targetDetailedTileCount;
        _lastTargetRetainedTileRadius = targetRetainedTileRadius;

        float yaw = _cameraHeading.LengthSquared() > 1e-4f
            ? MathF.Atan2(_cameraHeading.Y, _cameraHeading.X) * (180f / MathF.PI)
            : 0f;
        List<DirectionalTileCoord> selectedTiles = _directionalTileSelector.GetVisibleTiles(
            cameraPos,
            yaw,
            DirectionalTileFovDegrees);

        _lastSelectedTiles.Clear();
        foreach (DirectionalTileCoord selected in selectedTiles.Take(targetDetailedTileCount))
            _lastSelectedTiles.Add((selected.TileX, selected.TileY));
        _lastSelectedHeading = _cameraHeading;

        _lastRetainedTiles.Clear();
        foreach (DirectionalTileCoord retained in _cameraTileWindowSelector.GetTiles(cameraPos, targetRetainedTileRadius))
            _lastRetainedTiles.Add((retained.TileX, retained.TileY));
        _effectiveRetainedTileCount = _lastRetainedTiles.Count;

        // Full-load is an explicit residency stress mode, not a visibility
        // mode. Keep the camera-facing selection current so object admission
        // can still reject inactive resident tiles.
        if (_allTilesResident)
            return;

        var desiredTiles = new HashSet<(int, int)>();
        foreach (var retained in _lastRetainedTiles)
            desiredTiles.Add(retained);

        // Capture-path tiles are deliberately loaded in addition to the
        // camera-facing set. This is an explicit, user-requested residency
        // lease; it must not turn ordinary world navigation into full-map load.
        foreach (var tile in _capturePreloadTiles)
            desiredTiles.Add(tile);

        var unloadKeepTiles = new HashSet<(int, int)>();
        foreach (var retained in _lastRetainedTiles)
            unloadKeepTiles.Add(retained);

        foreach (var tile in _capturePreloadTiles)
            unloadKeepTiles.Add(tile);

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
            LastUnloadedTileX = key.Item1;
            LastUnloadedTileY = key.Item2;
            TileUnloadEventCount++;
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
            if (_cameraHeading.LengthSquared() > 0.25f)
            {
                Vector2 offset = new(-(tx - tileX), -(ty - tileY));
                if (offset.LengthSquared() > 0.01f)
                    priority -= Vector2.Dot(Vector2.Normalize(offset), _cameraHeading) * 2f;
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
                    var result = LoadTileWithPlacementsSerialized(capturedTx, capturedTy);
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

    private void InvalidateStreamingTargets()
    {
        _lastDetailedTileCandidateRadius = -1;
        _lastTargetDetailedTileCount = -1;
        _lastTargetRetainedTileRadius = -1;
    }

    private void ComputeStreamingTargets(
        float fogEnd,
        out int detailedTileCandidateRadius,
        out int targetDetailedTileCount,
        out int targetRetainedTileRadius)
    {
        // The strict baseline intentionally ignores fog distance. Fog remains
        // a render effect; it does not enlarge the normal tile admission set.
        _ = fogEnd;

        if (_detailedTileCountOverride > 0)
        {
            targetDetailedTileCount = Math.Clamp(_detailedTileCountOverride, 1, MaxManualDetailedTileCount);
            detailedTileCandidateRadius = MaxDetailedTileCandidateRadius;
            targetRetainedTileRadius = _retainedTileRadius;
            return;
        }

        targetDetailedTileCount = Math.Clamp(MaxDetailedTileCount, MinDetailedTileCount, MaxDetailedTileCount);
        detailedTileCandidateRadius = MinDetailedTileCandidateRadius;
        targetRetainedTileRadius = _retainedTileRadius;
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

            var (tileMesh, chunkInfos) = BuildTileMesh(tx, ty, result.Chunks);
            if (tileMesh == null)
                continue;

            _loadedTiles[(tx, ty)] = tileMesh;
            if (!_adapter.TileTextures.TryGetValue((tx, ty), out var textureNames))
                textureNames = new List<string>();
            _terrainRenderer.AddTile(tileMesh, textureNames, chunkInfos, fadeIn: true);
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
        var result = LoadTileWithPlacementsSerialized(tileX, tileY);
        _tileCache[(tileX, tileY)] = result; // Cache for consistency with AOI path

        var (tileMesh, chunkInfos) = BuildTileMesh(tileX, tileY, result.Chunks);
        if (tileMesh == null)
            return;

        _loadedTiles[(tileX, tileY)] = tileMesh;
        if (!_adapter.TileTextures.TryGetValue((tileX, tileY), out var textureNames))
            textureNames = new List<string>();
        _terrainRenderer.AddTile(tileMesh, textureNames, chunkInfos, fadeIn: false);
        _liquidRenderer.AddChunks(result.Chunks);

        OnTileLoaded?.Invoke(tileX, tileY, result);
    }

    private TileLoadResult LoadTileWithPlacementsSerialized(int tileX, int tileY)
    {
        lock (_adapterLoadLock)
            return _adapter.LoadTileWithPlacements(tileX, tileY);
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
        if (!TerrainVisible)
            return;

        _terrainRenderer.Render(view, proj, _cameraPos, visibleTileKeys: _lastSelectedTiles);
        TraceDirectionalFrameDiagnostics();
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
        if (!TerrainVisible)
            return;

        _terrainRenderer.Render(view, proj, cameraPos, frustum, _lastSelectedTiles);
        TraceDirectionalFrameDiagnostics();
    }

    /// <summary>
    /// Render liquid surfaces. Call AFTER all opaque geometry (terrain, WMOs, MDX)
    /// so objects below the water surface are visible through transparent water.
    /// </summary>
    public void RenderLiquid(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, float deltaTime = 0.016f)
    {
        _liquidRenderer.Render(view, proj, cameraPos, _terrainRenderer.Lighting, deltaTime, _lastSelectedTiles);
    }

    public bool IsWireframe => _terrainRenderer.IsWireframe;

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
    public void SetSubObjectVisible(int index, bool visible) { } // per-tile visibility is a future feature; tracked but not yet specced

    private void TraceDirectionalFrameDiagnostics()
    {
        ViewerLog.Trace($"[TerrainSelection] Active Tiles: {LastFrameActiveTileCount}; Retained Tiles: {LastFrameRetainedTileCount}; Detailed Draw Calls: {LastFrameDetailedTileDrawCalls}; Invariant: {LastDirectionalTileInvariantPassed}");
    }

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
