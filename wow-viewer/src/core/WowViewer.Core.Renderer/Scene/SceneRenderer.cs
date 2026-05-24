using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Maps;
using WowViewer.Core.Renderer.Liquid;
using WowViewer.Core.Renderer.Sky;
using WowViewer.Core.Renderer.Terrain;
using WowViewer.Core.Renderer.Wmo;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Renderer.Scene;

public sealed class SceneRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly TerrainRenderer _terrainRenderer;
    private readonly SkyRenderer _skyRenderer;
    private readonly LiquidRenderer _liquidRenderer;
    private readonly WmoRenderer _wmoRenderer;
    private readonly Dictionary<string, MeshCacheEntry> _meshCache = new();

    private sealed record MeshCacheEntry(TerrainMesh Mesh);

    public SceneRenderer(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        var terrainShader = new TerrainShader(gl);
        _terrainRenderer = new TerrainRenderer(gl, terrainShader);
        _skyRenderer = new SkyRenderer(gl);
        _liquidRenderer = new LiquidRenderer(gl);
        _wmoRenderer = new WmoRenderer(gl);
    }

    public TerrainRenderer Terrain => _terrainRenderer;
    public SkyRenderer Sky => _skyRenderer;
    public LiquidRenderer Liquid => _liquidRenderer;
    public WmoRenderer Wmo => _wmoRenderer;

    public WorldTerrainTileData? CurrentTileData { get; set; }

    public void SetTileData(WorldTerrainTileData tileData)
    {
        CurrentTileData = tileData;
        GetOrCreateMesh(tileData);
    }

    public void RenderFull(SceneCamera camera, RenderVariant variant)
    {
        _skyRenderer.Render(camera);

        if (!variant.HideTerrain && CurrentTileData is not null)
            RenderTile(camera, CurrentTileData, variant);

        if (!variant.HideLiquids)
            _liquidRenderer.Render(camera, variant);

        if (!variant.HideObjects)
            _wmoRenderer.Render(camera, variant);
    }

    public void RenderTile(SceneCamera camera, WorldTerrainTileData tileData, RenderVariant variant)
    {
        var mesh = GetOrCreateMesh(tileData);
        if (variant.HideTerrain)
            return;
        _terrainRenderer.Render(camera, [mesh], variant);
    }

    private TerrainMesh GetOrCreateMesh(WorldTerrainTileData tileData)
    {
        string key = $"{tileData.SourcePath}_{tileData.ChunkCount}";
        if (_meshCache.TryGetValue(key, out var entry))
            return entry.Mesh;

        ParseTileCoords(tileData.SourcePath, out int tileX, out int tileY);
        int maxLayers = tileData.Chunks.Max(c => c.TextureLayers.Count);
        var builder = new TerrainMeshBuilder(_gl);
        var mesh = builder.Build(tileX, tileY, tileData, maxLayers);
        _meshCache[key] = new MeshCacheEntry(mesh);
        return mesh;
    }

    internal static void ParseTileCoords(string sourcePath, out int tileX, out int tileY)
    {
        string name = Path.GetFileNameWithoutExtension(sourcePath);
        int idx = name.LastIndexOf('_');
        if (idx < 0 || !int.TryParse(name.AsSpan(idx + 1), out tileY))
        { tileX = 0; tileY = 0; return; }
        name = name[..idx];
        idx = name.LastIndexOf('_');
        if (idx < 0 || !int.TryParse(name.AsSpan(idx + 1), out tileX))
        { tileX = 0; tileY = 0; }
    }

    public void Dispose()
    {
        foreach (var entry in _meshCache.Values)
            entry.Mesh.Dispose();
        _meshCache.Clear();
        _terrainRenderer.Dispose();
        _skyRenderer.Dispose();
        _liquidRenderer.Dispose();
        _wmoRenderer.Dispose();
    }
}
