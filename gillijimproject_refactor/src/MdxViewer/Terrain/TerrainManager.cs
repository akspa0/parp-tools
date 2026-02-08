using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Manages terrain loading and rendering for an Alpha WDT map.
/// Handles AOI-based tile loading/unloading as the camera moves.
/// Implements <see cref="ISceneRenderer"/> so it can be used as the active renderer in ViewerApp.
/// </summary>
public class TerrainManager : ISceneRenderer
{
    private readonly GL _gl;
    private readonly AlphaTerrainAdapter _adapter;
    private readonly TerrainMeshBuilder _meshBuilder;
    private readonly TerrainRenderer _terrainRenderer;
    private readonly IDataSource? _dataSource;

    // Loaded tiles: (tileX, tileY) → list of chunk meshes
    private readonly Dictionary<(int, int), List<TerrainChunkMesh>> _loadedTiles = new();

    // AOI: how many tiles around the camera to keep loaded
    private const int AoiRadius = 2; // Load 5×5 tiles around camera (max ~25, but limited by existing tiles)

    // Camera tracking for AOI updates
    private int _lastCameraTileX = -1;
    private int _lastCameraTileY = -1;
    private Vector3 _cameraPos;

    // Stats
    public int LoadedTileCount => _loadedTiles.Count;
    public int LoadedChunkCount => _terrainRenderer.LoadedChunkCount;
    public bool IsTileLoaded(int tileX, int tileY) => _loadedTiles.ContainsKey((tileX, tileY));
    public TerrainLighting Lighting => _terrainRenderer.Lighting;
    public TerrainRenderer Renderer => _terrainRenderer;
    public string MapName { get; }

    /// <summary>Exposes the terrain adapter for WorldScene to access placement data.</summary>
    public AlphaTerrainAdapter Adapter => _adapter;

    public TerrainManager(GL gl, string wdtPath, IDataSource? dataSource)
    {
        _gl = gl;
        _dataSource = dataSource;
        MapName = Path.GetFileNameWithoutExtension(wdtPath);

        _adapter = new AlphaTerrainAdapter(wdtPath);
        _meshBuilder = new TerrainMeshBuilder(gl);
        _terrainRenderer = new TerrainRenderer(gl, dataSource, new TerrainLighting());

        // Find the center of populated tiles for initial camera placement
        FindInitialCameraPosition(out _cameraPos);
    }

    /// <summary>
    /// Update terrain AOI based on camera position. Call each frame before Render.
    /// </summary>
    public void UpdateAOI(Vector3 cameraPos)
    {
        _cameraPos = cameraPos;

        // Convert camera world position to tile coordinates
        // rendererX = wowY = MapOrigin - tileX * ChunkSize → tileX = (MapOrigin - rendererX) / ChunkSize
        // rendererY = wowX = MapOrigin - tileY * ChunkSize → tileY = (MapOrigin - rendererY) / ChunkSize
        int tileX = (int)((WoWConstants.MapOrigin - cameraPos.X) / WoWConstants.ChunkSize);
        int tileY = (int)((WoWConstants.MapOrigin - cameraPos.Y) / WoWConstants.ChunkSize);

        tileX = Math.Clamp(tileX, 0, 63);
        tileY = Math.Clamp(tileY, 0, 63);

        // Only update if camera moved to a different tile
        if (tileX == _lastCameraTileX && tileY == _lastCameraTileY)
            return;

        _lastCameraTileX = tileX;
        _lastCameraTileY = tileY;

        // Determine which tiles should be loaded
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

        // Unload tiles no longer in AOI
        var toUnload = _loadedTiles.Keys.Where(k => !desiredTiles.Contains(k)).ToList();
        foreach (var key in toUnload)
        {
            foreach (var chunk in _loadedTiles[key])
                chunk.Dispose();
            _loadedTiles.Remove(key);
        }

        // Load new tiles
        foreach (var (tx, ty) in desiredTiles)
        {
            if (!_loadedTiles.ContainsKey((tx, ty)))
                LoadTile(tx, ty);
        }
    }

    /// <summary>
    /// Load all tiles at once (for small maps or initial load).
    /// </summary>
    public void LoadAllTiles(Action<int, int, string>? onProgress = null)
    {
        int total = _adapter.ExistingTiles.Count;
        int loaded = 0;
        Console.WriteLine($"[TerrainManager] Loading all {total} tiles...");
        foreach (int tileIdx in _adapter.ExistingTiles)
        {
            // Alpha WDT MAIN is column-major: index = x*64+y
            int tx = tileIdx / 64;
            int ty = tileIdx % 64;
            if (!_loadedTiles.ContainsKey((tx, ty)))
                LoadTile(tx, ty);
            loaded++;
            onProgress?.Invoke(loaded, total, $"Tile ({tx},{ty})");
        }
        Console.WriteLine($"[TerrainManager] All tiles loaded: {_loadedTiles.Count} tiles, {LoadedChunkCount} chunks");
    }

    private void LoadTile(int tileX, int tileY)
    {
        var chunkDataList = _adapter.LoadTile(tileX, tileY);
        var meshes = new List<TerrainChunkMesh>();

        foreach (var chunkData in chunkDataList)
        {
            var mesh = _meshBuilder.BuildChunkMesh(chunkData);
            if (mesh != null)
                meshes.Add(mesh);
        }

        _loadedTiles[(tileX, tileY)] = meshes;
        _terrainRenderer.AddChunks(meshes, _adapter.TileTextures);
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
    }

    /// <summary>
    /// Render with explicit camera position (called from WorldScene or ViewerApp).
    /// </summary>
    public void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos)
    {
        _cameraPos = cameraPos;
        _terrainRenderer.Render(view, proj, cameraPos);
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
        _terrainRenderer.Dispose();
        foreach (var meshes in _loadedTiles.Values)
            foreach (var mesh in meshes)
                mesh.Dispose();
        _loadedTiles.Clear();
    }
}
