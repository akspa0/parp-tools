using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Runtime.DetailDoodads;
using WowViewer.Core.Runtime.World;

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
    private readonly List<DetailDoodadInstance> _globalWmoDoodads = new();
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
                return _tileDoodads.Values.Sum(list => list.Count) + _globalWmoDoodads.Count;
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
            _globalWmoDoodads.Clear();
        }
    }

    public void InvalidateDoodads()
    {
        lock (_tileDoodads)
        {
            _tileDoodads.Clear();
            _globalWmoDoodads.Clear();
        }

        if (_hookedTerrainManager != null)
        {
            ProcessGlobalWmos();
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
                _globalWmoDoodads.Clear();
            }

            EnsureLookupLoaded();
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

                ProcessGlobalWmos();
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
        if (!_lookup.IsLoaded)
        {
            if (_host.DbcProvider != null && !string.IsNullOrWhiteSpace(_host.DbdDir) && !string.IsNullOrWhiteSpace(_host.DbcBuild))
            {
                _lookup.Load(_host.DbcProvider, _host.DbdDir, _host.DbcBuild);
            }

            if (!_lookup.IsLoaded && _host.DataSource != null)
            {
                _lookup.Load(_host.DataSource.ReadFile);
            }
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
        EnsureLookupLoaded();
        if (!_lookup.IsLoaded)
            return;

        var tileInstances = new List<DetailDoodadInstance>();

        if (result.Chunks != null && result.Chunks.Count > 0)
        {
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
        }

        // Process WMO placements on this tile
        if (result.ModfPlacements != null && result.ModfPlacements.Count > 0 && _hookedTerrainManager != null)
        {
            var wmoNames = _hookedTerrainManager.Adapter.WmoModelNames;
            var assets = ((IWorldSceneHost?)_host.WorldScene)?.Assets;

            if (assets != null)
            {
                foreach (var p in result.ModfPlacements)
                {
                    if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count)
                        continue;

                    string modelKey = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
                    var transform = WorldPlacementTransform.Build(p.Position, p.Rotation);
                    var wmoInstances = GenerateWmoDoodads(modelKey, transform, assets);
                    if (wmoInstances.Count > 0)
                        tileInstances.AddRange(wmoInstances);
                }
            }
        }

        lock (_tileDoodads)
        {
            _tileDoodads[(tileX, tileY)] = tileInstances;
        }
    }

    private void ProcessGlobalWmos()
    {
        if (_hookedTerrainManager == null)
            return;

        var adapter = _hookedTerrainManager.Adapter;
        if (adapter.ModfPlacements.Count == 0)
            return;

        EnsureLookupLoaded();
        if (!_lookup.IsLoaded)
            return;

        var wmoNames = adapter.WmoModelNames;
        var assets = ((IWorldSceneHost?)_host.WorldScene)?.Assets;
        if (assets == null)
            return;

        var globalInstances = new List<DetailDoodadInstance>();
        foreach (var p in adapter.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count)
                continue;

            string modelKey = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            var transform = WorldPlacementTransform.Build(p.Position, p.Rotation);
            var wmoInstances = GenerateWmoDoodads(modelKey, transform, assets);
            if (wmoInstances.Count > 0)
                globalInstances.AddRange(wmoInstances);
        }

        lock (_tileDoodads)
        {
            _globalWmoDoodads.Clear();
            if (globalInstances.Count > 0)
                _globalWmoDoodads.AddRange(globalInstances);
        }
    }

    private List<DetailDoodadInstance> GenerateWmoDoodads(
        string modelKey,
        Matrix4x4 worldTransform,
        WorldAssetManager assets)
    {
        var instances = new List<DetailDoodadInstance>();

        if (!assets.TryGetLoadedWmo(modelKey, out var renderer) || renderer == null)
        {
            renderer = assets.GetWmo(modelKey);
            if (renderer == null)
                return instances;
        }

        var wmo = renderer.WmoData;
        if (wmo == null)
            return instances;

        // 1. Process MDDL detail doodad chunk if parsed
        if (wmo.DetailDoodads != null && wmo.DetailDoodads.Groups.Count > 0)
        {
            var groupDocMap = wmo.DetailDoodads.Groups.ToDictionary(g => (int)g.GroupIndex);
            for (int gi = 0; gi < wmo.Groups.Count; gi++)
            {
                var group = wmo.Groups[gi];
                if (!groupDocMap.TryGetValue(gi, out var groupData))
                    continue;

                var input = new WmoGroupPlacementInput
                {
                    GroupIndex = gi,
                    Vertices = group.Vertices.ToArray(),
                    Normals = group.Normals.Count == group.Vertices.Count
                        ? group.Normals.ToArray()
                        : ComputeVertexNormals(group.Vertices, group.Indices),
                    Indices = group.Indices.ToArray(),
                    Batches = group.Batches.Select(b => (b.FirstIndex, b.IndexCount, b.FirstVertex, b.LastVertex)).ToArray(),
                    WorldTransform = worldTransform,
                    Layers = wmo.DetailDoodads.Layers,
                    Commands = groupData.Commands,
                };

                var groupInstances = WmoDetailDoodadDecoder.DecodeGroupDoodads(input, _lookup.GetDoodadRecord);
                if (groupInstances.Count > 0)
                    instances.AddRange(groupInstances);
            }
        }

        // 2. Process material GroundType references (surfaces/roofs textured with grass/foliage)
        var matGroundTypes = new Dictionary<int, GroundEffectTextureRecord>();
        for (int mi = 0; mi < wmo.Materials.Count; mi++)
        {
            uint groundType = wmo.Materials[mi].GroundType;
            if (groundType > 0)
            {
                var rec = _lookup.GetTextureRecord(groundType);
                if (rec != null && rec.DoodadIds.Count > 0)
                    matGroundTypes[mi] = rec;
            }
        }

        if (matGroundTypes.Count > 0)
        {
            var rng = new Random(modelKey.GetHashCode());
            for (int gi = 0; gi < wmo.Groups.Count; gi++)
            {
                var group = wmo.Groups[gi];
                if (group.Batches.Count == 0 || group.Vertices.Count == 0 || group.Indices.Count == 0)
                    continue;

                var vertNormals = group.Normals.Count == group.Vertices.Count
                    ? group.Normals.ToArray()
                    : ComputeVertexNormals(group.Vertices, group.Indices);

                foreach (var batch in group.Batches)
                {
                    if (!matGroundTypes.TryGetValue(batch.MaterialId, out var texRec))
                        continue;

                    int density = (int)MathF.Ceiling(texRec.Density * GroundEffectDensity * 0.15f);
                    if (density <= 0)
                        density = 1;

                    int indexEnd = (int)(batch.FirstIndex + batch.IndexCount);
                    int step = Math.Max(3, 12 / density * 3);
                    for (int ii = (int)batch.FirstIndex; ii + 2 < indexEnd && ii + 2 < group.Indices.Count; ii += step)
                    {
                        int i0 = group.Indices[ii];
                        int i1 = group.Indices[ii + 1];
                        int i2 = group.Indices[ii + 2];

                        if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                            continue;

                        Vector3 localPos = (group.Vertices[i0] + group.Vertices[i1] + group.Vertices[i2]) / 3.0f;
                        Vector3 localNorm = (vertNormals[i0] + vertNormals[i1] + vertNormals[i2]) / 3.0f;
                        if (localNorm.LengthSquared() > 0.001f)
                            localNorm = Vector3.Normalize(localNorm);
                        else
                            localNorm = Vector3.UnitZ;

                        Vector3 worldPos = Vector3.Transform(localPos, worldTransform);
                        Vector3 worldNorm = Vector3.Normalize(Vector3.TransformNormal(localNorm, worldTransform));

                        // Slope check: normal Z >= 0.4
                        if (worldNorm.Z < GroundEffectPlacementGenerator.SlopeNormalZThreshold)
                            continue;

                        uint chosenDoodadId = texRec.DoodadIds[rng.Next(texRec.DoodadIds.Count)];
                        var doodad = _lookup.GetDoodadRecord(chosenDoodadId);
                        var flags = doodad?.Flags ?? GroundEffectDoodadFlags.None;

                        float yaw = rng.NextSingle() * MathF.PI * 2.0f;
                        var rotYaw = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, yaw);
                        Quaternion orientation = (flags & GroundEffectDoodadFlags.AlignToNormal) != 0
                            ? GroundEffectPlacementGenerator.ComputeNormalAlignment(worldNorm) * rotYaw
                            : rotYaw;

                        float scale = 0.9f + rng.NextSingle() * 0.2f;
                        if (doodad?.AnimScale > 0)
                            scale *= doodad.AnimScale;

                        instances.Add(new DetailDoodadInstance(
                            worldPos,
                            orientation,
                            scale,
                            0xFFFFFFFF,
                            chosenDoodadId,
                            doodad?.FileDataId,
                            doodad?.ModelPath,
                            flags));
                    }
                }
            }
        }

        return instances;
    }

    private static Vector3[] ComputeVertexNormals(IReadOnlyList<Vector3> vertices, IReadOnlyList<ushort> indices)
    {
        var normals = new Vector3[vertices.Count];
        for (int i = 0; i + 2 < indices.Count; i += 3)
        {
            int i0 = indices[i];
            int i1 = indices[i + 1];
            int i2 = indices[i + 2];
            if (i0 < vertices.Count && i1 < vertices.Count && i2 < vertices.Count)
            {
                Vector3 e1 = vertices[i1] - vertices[i0];
                Vector3 e2 = vertices[i2] - vertices[i0];
                Vector3 fn = Vector3.Normalize(Vector3.Cross(e1, e2));
                normals[i0] += fn;
                normals[i1] += fn;
                normals[i2] += fn;
            }
        }
        for (int i = 0; i < normals.Length; i++)
        {
            if (normals[i].LengthSquared() > 0.001f)
                normals[i] = Vector3.Normalize(normals[i]);
            else
                normals[i] = Vector3.UnitZ;
        }
        return normals;
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

        if (_tileDoodads.Count == 0 && _globalWmoDoodads.Count == 0)
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
            void AccumulateInstances(List<DetailDoodadInstance> instances)
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

            foreach (var ((tx, ty), instances) in _tileDoodads)
            {
                AccumulateInstances(instances);
            }

            if (_globalWmoDoodads.Count > 0)
            {
                AccumulateInstances(_globalWmoDoodads);
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
