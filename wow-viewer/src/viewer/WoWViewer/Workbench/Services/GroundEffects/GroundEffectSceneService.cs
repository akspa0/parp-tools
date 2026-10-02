using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Runtime.DetailDoodads;

namespace WoWViewer;

/// <summary>
/// Workbench service managing ground effect doodads (flora, grass, clutter) across loaded terrain tiles.
/// Evaluates ADT layer effects and WMO detail doodads and renders them with distance culling and GPU instancing.
/// </summary>
internal sealed partial class GroundEffectSceneService : IDisposable
{
    private readonly IViewerAppHost _host;
    private GroundEffectLookup _lookup = new();
    private readonly Dictionary<(int TileX, int TileY), List<DetailDoodadInstance>> _tileDoodads = new();
    private TerrainManager? _hookedTerrainManager;
    private IDataSource? _hookedDataSource;

    public bool EnableGroundEffects { get; set; } = true;
    public float GroundEffectDensity { get; set; } = 1.0f;
    public float GroundEffectDistance { get; set; } = 120.0f;

    public int ActiveDetailDoodadCount
    {
        get
        {
            lock (_tileDoodads)
            {
                return _tileDoodads.Values.Sum(list => list.Count);
            }
        }
    }

    public int LoadedTileDoodadCount
    {
        get
        {
            lock (_tileDoodads)
            {
                return _tileDoodads.Count;
            }
        }
    }

    internal GroundEffectSceneService(IViewerAppHost host)
    {
        _host = host;
    }

    public void Dispose()
    {
        UnhookEvents();
        lock (_tileDoodads)
        {
            _tileDoodads.Clear();
        }
    }

    public void InvalidateDoodads()
    {
        lock (_tileDoodads)
        {
            _tileDoodads.Clear();
        }

        if (_hookedTerrainManager != null)
        {
            foreach (var kvp in _hookedTerrainManager.TileCache)
            {
                ProcessTile(kvp.Key.Item1, kvp.Key.Item2, kvp.Value);
            }
        }
    }

    private void SyncState()
    {
        var currentDataSource = _host.DataSource;
        if (currentDataSource != _hookedDataSource)
        {
            _hookedDataSource = currentDataSource;
            _lookup = new GroundEffectLookup();
            lock (_tileDoodads)
            {
                _tileDoodads.Clear();
            }

            if (currentDataSource != null)
            {
                _lookup.Load(currentDataSource.ReadFile);
            }
        }

        var currentTerrainManager = _host.TerrainManager;
        if (currentTerrainManager != _hookedTerrainManager)
        {
            UnhookEvents();
            _hookedTerrainManager = currentTerrainManager;
            if (_hookedTerrainManager != null)
            {
                _hookedTerrainManager.OnTileLoaded += OnTileLoaded;
                _hookedTerrainManager.OnTileUnloaded += OnTileUnloaded;

                foreach (var kvp in _hookedTerrainManager.TileCache)
                {
                    ProcessTile(kvp.Key.Item1, kvp.Key.Item2, kvp.Value);
                }
            }
        }
    }

    private void UnhookEvents()
    {
        if (_hookedTerrainManager != null)
        {
            _hookedTerrainManager.OnTileLoaded -= OnTileLoaded;
            _hookedTerrainManager.OnTileUnloaded -= OnTileUnloaded;
            _hookedTerrainManager = null;
        }
    }

    private void EnsureLookupLoaded()
    {
        if (!_lookup.IsLoaded && _host.DataSource != null)
        {
            _lookup.Load(_host.DataSource.ReadFile);
        }
    }

    private void OnTileLoaded(int tileX, int tileY, TileLoadResult result)
    {
        ProcessTile(tileX, tileY, result);
    }

    private void OnTileUnloaded(int tileX, int tileY)
    {
        lock (_tileDoodads)
        {
            _tileDoodads.Remove((tileX, tileY));
        }
    }

    private void ProcessTile(int tileX, int tileY, TileLoadResult result)
    {
        if (result.Chunks == null || result.Chunks.Count == 0)
            return;

        EnsureLookupLoaded();
        if (!_lookup.IsLoaded)
            return;

        var tileInstances = new List<DetailDoodadInstance>();
        foreach (var chunk in result.Chunks)
        {
            if (chunk.Heights == null || chunk.Heights.Length < 145)
                continue;

            var input = new TerrainChunkPlacementInput
            {
                WorldPosition = chunk.WorldPosition,
                Heights = chunk.Heights,
                Normals = chunk.Normals,
                MccvColors = chunk.MccvColors,
                ShadowMap = chunk.ShadowMap,
                HoleMask64 = chunk.HoleMask64,
                HoleMask = chunk.HoleMask,
                Layers = chunk.Layers.Select(l => new TerrainChunkLayerInput
                {
                    EffectId = l.EffectId,
                    TextureIndex = l.TextureIndex,
                    AlphaMap = chunk.AlphaMaps.TryGetValue(l.TextureIndex, out var alpha) ? alpha : null,
                }).ToList(),
                ChunkX = chunk.ChunkX,
                ChunkY = chunk.ChunkY,
                TileX = tileX,
                TileY = tileY,
            };

            var chunkInstances = GroundEffectPlacementGenerator.GenerateChunkDoodads(
                input,
                _lookup.GetTextureRecord,
                _lookup.GetDoodadRecord,
                GroundEffectDensity);

            if (chunkInstances.Count > 0)
                tileInstances.AddRange(chunkInstances);
        }

        lock (_tileDoodads)
        {
            _tileDoodads[(tileX, tileY)] = tileInstances;
        }
    }

    public void Render(
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor,
        Vector3 fogColor,
        float fogStart,
        float fogEnd)
    {
        if (!EnableGroundEffects)
            return;

        SyncState();

        if (_tileDoodads.Count == 0)
            return;

        var worldScene = _host.WorldScene;
        if (worldScene == null)
            return;

        var assetManager = ((IWorldSceneHost)worldScene).Assets;
        if (assetManager == null)
            return;

        float maxDistSq = GroundEffectDistance * GroundEffectDistance;
        float fadeStartDist = Math.Max(10.0f, GroundEffectDistance - 25.0f);

        var batches = new Dictionary<string, List<(Matrix4x4 Matrix, float Fade)>>(StringComparer.OrdinalIgnoreCase);

        lock (_tileDoodads)
        {
            foreach (var ((tx, ty), instances) in _tileDoodads)
            {
                for (int i = 0; i < instances.Count; i++)
                {
                    var inst = instances[i];
                    float distSq = Vector3.DistanceSquared(cameraPos, inst.Position);
                    if (distSq > maxDistSq)
                        continue;

                    string? modelPath = inst.ModelPath;
                    if (string.IsNullOrEmpty(modelPath) && inst.FileDataId.HasValue && inst.FileDataId.Value > 0)
                    {
                        modelPath = FileDataIdPaths.Resolve(inst.FileDataId.Value);
                    }
                    if (string.IsNullOrEmpty(modelPath))
                        continue;

                    string modelKey = WorldAssetManager.NormalizeKey(modelPath);

                    float dist = MathF.Sqrt(distSq);
                    float fade = dist > fadeStartDist
                        ? Math.Clamp(1.0f - (dist - fadeStartDist) / (GroundEffectDistance - fadeStartDist), 0.0f, 1.0f)
                        : 1.0f;

                    var matrix = Matrix4x4.CreateScale(inst.Scale)
                        * Matrix4x4.CreateFromQuaternion(inst.Orientation)
                        * Matrix4x4.CreateTranslation(inst.Position);

                    if (!batches.TryGetValue(modelKey, out var list))
                    {
                        list = new List<(Matrix4x4 Matrix, float Fade)>();
                        batches[modelKey] = list;
                    }
                    list.Add((matrix, fade));
                }
            }
        }

        foreach (var (modelKey, transforms) in batches)
        {
            var renderer = assetManager.GetMdx(modelKey);
            if (renderer == null)
                continue;

            renderer.BeginBatch(
                view,
                proj,
                fogColor,
                fogStart,
                fogEnd,
                cameraPos,
                lightDir,
                lightColor,
                ambientColor);

            for (int i = 0; i < transforms.Count; i++)
            {
                renderer.RenderInstance(transforms[i].Matrix, RenderPass.Opaque, transforms[i].Fade);
            }
        }
    }
}
