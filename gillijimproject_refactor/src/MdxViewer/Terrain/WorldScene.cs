using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Combines terrain (WDT/ADT), WMO placements (MODF), and MDX placements (MDDF)
/// into a single world scene — the same way the game client renders a map.
/// 
/// Uses <see cref="WorldAssetManager"/> to ensure each model is loaded exactly once.
/// Instances are lightweight structs holding only a model key + transform.
/// </summary>
public class WorldScene : ISceneRenderer
{
    private readonly TerrainManager _terrainManager;
    private readonly WorldAssetManager _assets;

    // Lightweight instance lists — just a key + transform, no renderer reference
    private readonly List<ObjectInstance> _mdxInstances = new();
    private readonly List<ObjectInstance> _wmoInstances = new();

    private bool _objectsVisible = true;
    private bool _wmosVisible = true;
    private bool _doodadsVisible = true;

    // Stats
    public int MdxInstanceCount => _mdxInstances.Count;
    public int WmoInstanceCount => _wmoInstances.Count;
    public int UniqueMdxModels => _assets.MdxModelsLoaded;
    public int UniqueWmoModels => _assets.WmoModelsLoaded;
    public TerrainManager Terrain => _terrainManager;
    public WorldAssetManager Assets => _assets;

    public WorldScene(GL gl, string wdtPath, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null)
    {
        _assets = new WorldAssetManager(gl, dataSource, texResolver);

        // Create terrain manager (loads terrain chunks + collects MDDF/MODF placements)
        _terrainManager = new TerrainManager(gl, wdtPath, dataSource);
        _terrainManager.LoadAllTiles();

        var adapter = _terrainManager.Adapter;

        // Build manifest of unique assets referenced by this map
        var manifest = _assets.BuildManifest(
            adapter.MdxModelNames, adapter.WmoModelNames,
            adapter.MddfPlacements, adapter.ModfPlacements);

        // Load each unique model exactly once
        _assets.LoadManifest(manifest);

        // Build lightweight instance lists (just key + transform)
        BuildInstances(adapter);
    }

    private void BuildInstances(AlphaTerrainAdapter adapter)
    {
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // MDX (doodad) placements
        foreach (var p in adapter.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;
            var transform = Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationZ(p.Rotation.Y * MathF.PI / 180f)
                * Matrix4x4.CreateRotationX(p.Rotation.X * MathF.PI / 180f)
                * Matrix4x4.CreateRotationY(p.Rotation.Z * MathF.PI / 180f)
                * Matrix4x4.CreateTranslation(p.Position);

            _mdxInstances.Add(new ObjectInstance { ModelKey = key, Transform = transform });
        }

        // WMO placements
        foreach (var p in adapter.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            var transform = Matrix4x4.CreateRotationZ(p.Rotation.Y * MathF.PI / 180f)
                * Matrix4x4.CreateRotationX(p.Rotation.X * MathF.PI / 180f)
                * Matrix4x4.CreateRotationY(p.Rotation.Z * MathF.PI / 180f)
                * Matrix4x4.CreateTranslation(p.Position);

            _wmoInstances.Add(new ObjectInstance { ModelKey = key, Transform = transform });
        }

        Console.WriteLine($"[WorldScene] Instances: {_mdxInstances.Count} MDX, {_wmoInstances.Count} WMO");
        Console.WriteLine($"[WorldScene] Unique models: {UniqueMdxModels} MDX, {UniqueWmoModels} WMO");

        // Diagnostic: show first few placement positions vs terrain center
        var camPos = _terrainManager.GetInitialCameraPosition();
        Console.WriteLine($"[WorldScene] Terrain camera center: ({camPos.X:F1}, {camPos.Y:F1}, {camPos.Z:F1})");
        for (int i = 0; i < Math.Min(5, adapter.MddfPlacements.Count); i++)
        {
            var p = adapter.MddfPlacements[i];
            string name = p.NameIndex < mdxNames.Count ? Path.GetFileName(mdxNames[p.NameIndex]) : "?";
            Console.WriteLine($"[WorldScene]   MDDF[{i}] raw=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) rot=({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1}) scale={p.Scale:F2} model={name}");
        }
        for (int i = 0; i < Math.Min(5, adapter.ModfPlacements.Count); i++)
        {
            var p = adapter.ModfPlacements[i];
            string name = p.NameIndex < wmoNames.Count ? Path.GetFileName(wmoNames[p.NameIndex]) : "?";
            Console.WriteLine($"[WorldScene]   MODF[{i}] raw=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) rot=({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1}) model={name}");
        }
    }

    // ── ISceneRenderer ──────────────────────────────────────────────────

    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        // 1. Render terrain
        _terrainManager.Render(view, proj);

        if (!_objectsVisible) return;

        // 2. Render WMO instances (each instance references a shared renderer)
        if (_wmosVisible)
        {
            foreach (var inst in _wmoInstances)
            {
                var renderer = _assets.GetWmo(inst.ModelKey);
                if (renderer == null) continue;
                renderer.RenderWithTransform(inst.Transform, view, proj);
            }
        }

        // 3. Render MDX instances
        if (_doodadsVisible)
        {
            foreach (var inst in _mdxInstances)
            {
                var renderer = _assets.GetMdx(inst.ModelKey);
                if (renderer == null) continue;
                renderer.RenderWithTransform(inst.Transform, view, proj);
            }
        }
    }

    public void ToggleWireframe()
    {
        _terrainManager.ToggleWireframe();
    }

    public void ToggleObjects() => _objectsVisible = !_objectsVisible;
    public void ToggleWmos() => _wmosVisible = !_wmosVisible;
    public void ToggleDoodads() => _doodadsVisible = !_doodadsVisible;

    public int SubObjectCount => 3;

    public string GetSubObjectName(int index) => index switch
    {
        0 => $"Terrain ({_terrainManager.LoadedChunkCount} chunks)",
        1 => $"WMOs ({_wmoInstances.Count} instances, {UniqueWmoModels} unique)",
        2 => $"Doodads ({_mdxInstances.Count} instances, {UniqueMdxModels} unique)",
        _ => ""
    };

    public bool GetSubObjectVisible(int index) => index switch
    {
        0 => true,
        1 => _wmosVisible,
        2 => _doodadsVisible,
        _ => false
    };

    public void SetSubObjectVisible(int index, bool visible)
    {
        switch (index)
        {
            case 1: _wmosVisible = visible; break;
            case 2: _doodadsVisible = visible; break;
        }
    }

    public void Dispose()
    {
        _terrainManager.Dispose();
        _assets.Dispose();
        _mdxInstances.Clear();
        _wmoInstances.Clear();
    }
}

/// <summary>
/// Lightweight placement instance — just a model key and world transform.
/// The actual renderer is looked up from WorldAssetManager at render time.
/// </summary>
public struct ObjectInstance
{
    public string ModelKey;
    public Matrix4x4 Transform;
}
