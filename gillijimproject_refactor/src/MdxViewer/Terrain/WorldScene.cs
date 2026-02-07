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
    private readonly GL _gl;
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
    public bool IsWmoBased => _terrainManager.Adapter.IsWmoBased;

    // Expose raw placement data for UI object list
    public IReadOnlyList<MddfPlacement> MddfPlacements => _terrainManager.Adapter.MddfPlacements;
    public IReadOnlyList<ModfPlacement> ModfPlacements => _terrainManager.Adapter.ModfPlacements;
    public IReadOnlyList<string> MdxModelNames => _terrainManager.Adapter.MdxModelNames;
    public IReadOnlyList<string> WmoModelNames => _terrainManager.Adapter.WmoModelNames;

    // Bounding box debug rendering
    private bool _showBoundingBoxes = false;
    private BoundingBoxRenderer? _bbRenderer;
    public bool ShowBoundingBoxes { get => _showBoundingBoxes; set => _showBoundingBoxes = value; }

    public WorldScene(GL gl, string wdtPath, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _assets = new WorldAssetManager(gl, dataSource, texResolver);
        _bbRenderer = new BoundingBoxRenderer(gl);

        // Create terrain manager (uses AOI-based lazy loading — tiles load as camera moves)
        onStatus?.Invoke("Loading WDT...");
        _terrainManager = new TerrainManager(gl, wdtPath, dataSource);
        // Initial AOI load happens on first UpdateAOI call from ViewerApp

        var adapter = _terrainManager.Adapter;

        // Build manifest of unique assets referenced by this map
        onStatus?.Invoke("Building asset manifest...");
        var manifest = _assets.BuildManifest(
            adapter.MdxModelNames, adapter.WmoModelNames,
            adapter.MddfPlacements, adapter.ModfPlacements);

        // Load each unique model exactly once
        onStatus?.Invoke($"Loading {manifest.ReferencedMdx.Count} MDX + {manifest.ReferencedWmo.Count} WMO models...");
        _assets.LoadManifest(manifest);

        // Build lightweight instance lists (just key + transform)
        BuildInstances(adapter);

        // For WMO-only maps, compute camera from MODF placement
        if (adapter.IsWmoBased && adapter.ModfPlacements.Count > 0)
        {
            var wmoPos = adapter.ModfPlacements[0].Position;
            _wmoCameraOverride = new Vector3(wmoPos.X, wmoPos.Y, wmoPos.Z + 50f);
            Console.WriteLine($"[WorldScene] WMO-only map, camera at WMO position: ({wmoPos.X:F1}, {wmoPos.Y:F1}, {wmoPos.Z:F1})");
        }

        onStatus?.Invoke("World loaded.");
    }

    private Vector3? _wmoCameraOverride;
    /// <summary>For WMO-only maps, returns the WMO position as camera start. Otherwise null.</summary>
    public Vector3? WmoCameraOverride => _wmoCameraOverride;

    private void BuildInstances(AlphaTerrainAdapter adapter)
    {
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // MDX (doodad) placements
        // Rotation stored as (wowRotX, wowRotY, wowRotZ) in degrees
        // Our renderer swaps X↔Y: rendererX=wowY, rendererY=wowX, rendererZ=wowZ
        // So: rotate around rendererY (=wowX) by rotX, rendererX (=wowY) by rotY, rendererZ by rotZ
        foreach (var p in adapter.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;
            float rx = p.Rotation.X * MathF.PI / 180f; // wowRotX → around rendererY
            float ry = p.Rotation.Y * MathF.PI / 180f; // wowRotY → around rendererX
            float rz = p.Rotation.Z * MathF.PI / 180f; // wowRotZ → around rendererZ (heading)
            var transform = Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationY(rx)
                * Matrix4x4.CreateRotationX(ry)
                * Matrix4x4.CreateRotationZ(-rz)
                * Matrix4x4.CreateTranslation(p.Position);

            _mdxInstances.Add(new ObjectInstance { ModelKey = key, Transform = transform });
        }

        // WMO placements
        foreach (var p in adapter.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;
            var transform = Matrix4x4.CreateRotationY(rx)
                * Matrix4x4.CreateRotationX(ry)
                * Matrix4x4.CreateRotationZ(-rz)
                * Matrix4x4.CreateTranslation(p.Position);

            _wmoInstances.Add(new ObjectInstance { ModelKey = key, Transform = transform });
        }

        Console.WriteLine($"[WorldScene] Instances: {_mdxInstances.Count} MDX, {_wmoInstances.Count} WMO");
        Console.WriteLine($"[WorldScene] Unique models: {UniqueMdxModels} MDX, {UniqueWmoModels} WMO");

        // Diagnostic: terrain chunk WorldPosition range
        var camPos = _terrainManager.GetInitialCameraPosition();
        Console.WriteLine($"[WorldScene] Camera: ({camPos.X:F1}, {camPos.Y:F1}, {camPos.Z:F1})");

        // Compute terrain bounding box from chunk WorldPositions
        float tMinX = float.MaxValue, tMinY = float.MaxValue, tMinZ = float.MaxValue;
        float tMaxX = float.MinValue, tMaxY = float.MinValue, tMaxZ = float.MinValue;
        foreach (var chunk in _terrainManager.Adapter.LastLoadedChunkPositions)
        {
            tMinX = Math.Min(tMinX, chunk.X); tMaxX = Math.Max(tMaxX, chunk.X);
            tMinY = Math.Min(tMinY, chunk.Y); tMaxY = Math.Max(tMaxY, chunk.Y);
            tMinZ = Math.Min(tMinZ, chunk.Z); tMaxZ = Math.Max(tMaxZ, chunk.Z);
        }
        Console.WriteLine($"[WorldScene] TERRAIN  X:[{tMinX:F1} .. {tMaxX:F1}]  Y:[{tMinY:F1} .. {tMaxY:F1}]  Z:[{tMinZ:F1} .. {tMaxZ:F1}]");

        // Compute object bounding box (from stored positions, which are already transformed)
        float oMinX = float.MaxValue, oMinY = float.MaxValue, oMinZ = float.MaxValue;
        float oMaxX = float.MinValue, oMaxY = float.MinValue, oMaxZ = float.MinValue;
        foreach (var p in adapter.MddfPlacements)
        {
            oMinX = Math.Min(oMinX, p.Position.X); oMaxX = Math.Max(oMaxX, p.Position.X);
            oMinY = Math.Min(oMinY, p.Position.Y); oMaxY = Math.Max(oMaxY, p.Position.Y);
            oMinZ = Math.Min(oMinZ, p.Position.Z); oMaxZ = Math.Max(oMaxZ, p.Position.Z);
        }
        foreach (var p in adapter.ModfPlacements)
        {
            oMinX = Math.Min(oMinX, p.Position.X); oMaxX = Math.Max(oMaxX, p.Position.X);
            oMinY = Math.Min(oMinY, p.Position.Y); oMaxY = Math.Max(oMaxY, p.Position.Y);
            oMinZ = Math.Min(oMinZ, p.Position.Z); oMaxZ = Math.Max(oMaxZ, p.Position.Z);
        }
        Console.WriteLine($"[WorldScene] OBJECTS  X:[{oMinX:F1} .. {oMaxX:F1}]  Y:[{oMinY:F1} .. {oMaxY:F1}]  Z:[{oMinZ:F1} .. {oMaxZ:F1}]");
        Console.WriteLine($"[WorldScene] DELTA    X:{(tMinX+tMaxX)/2 - (oMinX+oMaxX)/2:F1}  Y:{(tMinY+tMaxY)/2 - (oMinY+oMaxY)/2:F1}  Z:{(tMinZ+tMaxZ)/2 - (oMinZ+oMaxZ)/2:F1}");

        // Print first 3 MDDF raw values for manual inspection
        for (int i = 0; i < Math.Min(3, adapter.MddfPlacements.Count); i++)
        {
            var p = adapter.MddfPlacements[i];
            string name = p.NameIndex < mdxNames.Count ? Path.GetFileName(mdxNames[p.NameIndex]) : "?";
            Console.WriteLine($"[WorldScene]   MDDF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
        for (int i = 0; i < Math.Min(3, adapter.ModfPlacements.Count); i++)
        {
            var p = adapter.ModfPlacements[i];
            string name = p.NameIndex < wmoNames.Count ? Path.GetFileName(wmoNames[p.NameIndex]) : "?";
            Console.WriteLine($"[WorldScene]   MODF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
    }

    // ── ISceneRenderer ──────────────────────────────────────────────────

    private bool _renderDiagPrinted = false;
    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        // 1. Render terrain
        _terrainManager.Render(view, proj);

        // Reset GL state after terrain
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.Enable(EnableCap.DepthTest);
        _gl.UseProgram(0); // unbind terrain shader

        if (!_objectsVisible) return;

        // One-time render diagnostic
        if (!_renderDiagPrinted)
        {
            int wmoFound = 0, wmoMissing = 0;
            foreach (var inst in _wmoInstances)
            {
                if (_assets.GetWmo(inst.ModelKey) != null) wmoFound++;
                else { wmoMissing++; if (wmoMissing <= 3) Console.WriteLine($"[WorldScene] WMO NOT FOUND: \"{inst.ModelKey}\""); }
            }
            int mdxFound = 0, mdxMissing = 0;
            foreach (var inst in _mdxInstances)
            {
                if (_assets.GetMdx(inst.ModelKey) != null) mdxFound++;
                else { mdxMissing++; if (mdxMissing <= 3) Console.WriteLine($"[WorldScene] MDX NOT FOUND: \"{inst.ModelKey}\""); }
            }
            Console.WriteLine($"[WorldScene] Render check: WMO {wmoFound} found / {wmoMissing} missing, MDX {mdxFound} found / {mdxMissing} missing");
        }

        // 2. Render WMO instances (each instance references a shared renderer)
        if (_wmosVisible)
        {
            int wmoRendered = 0;
            foreach (var inst in _wmoInstances)
            {
                var renderer = _assets.GetWmo(inst.ModelKey);
                if (renderer == null) continue;
                // Reset blend state before each model
                _gl.Disable(EnableCap.Blend);
                _gl.DepthMask(true);
                renderer.RenderWithTransform(inst.Transform, view, proj);
                wmoRendered++;
            }
            if (!_renderDiagPrinted) Console.WriteLine($"[WorldScene] WMO render loop: {wmoRendered} rendered");
        }

        // 3. Render MDX instances
        if (_doodadsVisible)
        {
            int mdxRendered = 0;
            foreach (var inst in _mdxInstances)
            {
                var renderer = _assets.GetMdx(inst.ModelKey);
                if (renderer == null) continue;
                // Reset blend state before each model
                _gl.Disable(EnableCap.Blend);
                _gl.DepthMask(true);
                renderer.RenderWithTransform(inst.Transform, view, proj);
                mdxRendered++;
            }
            if (!_renderDiagPrinted) Console.WriteLine($"[WorldScene] MDX render loop: {mdxRendered} rendered");
            if (!_renderDiagPrinted) _renderDiagPrinted = true;
        }

        // Reset GL state before bounding boxes
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.UseProgram(0);
        _gl.BindVertexArray(0);

        // 4. Debug bounding boxes for all placements
        if (_showBoundingBoxes && _bbRenderer != null)
        {
            // Disable depth test so BBs always draw on top
            _gl.Disable(EnableCap.DepthTest);

            var adapter = _terrainManager.Adapter;
            if (!_renderDiagPrinted)
                Console.WriteLine($"[WorldScene] BB render: {adapter.MddfPlacements.Count} MDDF + {adapter.ModfPlacements.Count} MODF markers");
            // MDDF markers (yellow)
            foreach (var p in adapter.MddfPlacements)
                _bbRenderer.DrawMarker(p.Position, 5f, view, proj, new Vector3(1f, 1f, 0f));
            // MODF markers (cyan)
            foreach (var p in adapter.ModfPlacements)
                _bbRenderer.DrawMarker(p.Position, 10f, view, proj, new Vector3(0f, 1f, 1f));

            _gl.Enable(EnableCap.DepthTest);
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
        _bbRenderer?.Dispose();
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
