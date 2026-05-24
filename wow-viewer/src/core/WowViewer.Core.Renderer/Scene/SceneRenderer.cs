using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Renderer.Terrain;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Renderer.Scene;

public sealed class SceneRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly TerrainRenderer _terrainRenderer;
    private readonly Dictionary<string, MeshCacheEntry> _meshCache = new();

    private sealed record MeshCacheEntry(TerrainMesh Mesh);

    public SceneRenderer(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        var shader = new TerrainShader(gl);
        _terrainRenderer = new TerrainRenderer(gl, shader);
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
        {
            tileX = 0; tileY = 0;
            return;
        }
        name = name[..idx];
        idx = name.LastIndexOf('_');
        if (idx < 0 || !int.TryParse(name.AsSpan(idx + 1), out tileX))
        {
            tileX = 0; tileY = 0;
        }
    }

    public void Dispose()
    {
        foreach (var entry in _meshCache.Values)
            entry.Mesh.Dispose();

        _terrainRenderer.Dispose();
    }
}
