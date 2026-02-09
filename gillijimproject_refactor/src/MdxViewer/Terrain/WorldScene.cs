using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
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
    // These are rebuilt from _tileMdxInstances/_tileWmoInstances when tiles change
    private List<ObjectInstance> _mdxInstances = new();
    private List<ObjectInstance> _wmoInstances = new();

    // Per-tile instance storage for lazy load/unload
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileMdxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileWmoInstances = new();
    private bool _instancesDirty = false;

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

    // Sky dome
    private readonly SkyDomeRenderer _skyDome;
    public SkyDomeRenderer SkyDome => _skyDome;

    // Bounding box debug rendering
    private bool _showBoundingBoxes = false;
    private BoundingBoxRenderer? _bbRenderer;
    public bool ShowBoundingBoxes { get => _showBoundingBoxes; set => _showBoundingBoxes = value; }

    // Area POI
    private AreaPoiLoader? _poiLoader;
    private bool _showPoi = true;
    public bool ShowPoi { get => _showPoi; set => _showPoi = value; }
    public AreaPoiLoader? PoiLoader => _poiLoader;

    /// <summary>
    /// Load AreaPOI entries for this map from DBC. Call after construction when DBC provider is available.
    /// </summary>
    public void LoadAreaPoi(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build)
    {
        _poiLoader = new AreaPoiLoader();
        _poiLoader.Load(dbcProvider, dbdDir, build, _terrainManager.MapName);
    }

    public WorldScene(GL gl, string wdtPath, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _assets = new WorldAssetManager(gl, dataSource, texResolver);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);

        // Create terrain manager (uses AOI-based lazy loading — tiles load as camera moves)
        onStatus?.Invoke("Loading WDT...");
        _terrainManager = new TerrainManager(gl, wdtPath, dataSource);

        InitFromAdapter(onStatus);
    }

    /// <summary>
    /// Create a WorldScene with a pre-built TerrainManager (for Standard WDT, etc.).
    /// </summary>
    public WorldScene(GL gl, TerrainManager terrainManager, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _assets = new WorldAssetManager(gl, dataSource, texResolver);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);
        _terrainManager = terrainManager;

        InitFromAdapter(onStatus);
    }

    private void InitFromAdapter(Action<string>? onStatus)
    {
        var adapter = _terrainManager.Adapter;

        // For WMO-only maps, pre-load the WDT-level placements + models
        if (adapter.IsWmoBased && adapter.ModfPlacements.Count > 0)
        {
            var manifest = _assets.BuildManifest(
                adapter.MdxModelNames, adapter.WmoModelNames,
                adapter.MddfPlacements, adapter.ModfPlacements);
            _assets.LoadManifest(manifest);
            BuildInstances(adapter);

            var p = adapter.ModfPlacements[0];
            var bbCenter = (p.BoundsMin + p.BoundsMax) * 0.5f;
            var bbExtent = p.BoundsMax - p.BoundsMin;
            float dist = MathF.Max(bbExtent.Length() * 0.5f, 100f);
            _wmoCameraOverride = bbCenter + new Vector3(dist, 0, bbExtent.Z * 0.3f);
            ViewerLog.Info(ViewerLog.Category.Terrain, $"WMO-only map, camera at BB center: ({bbCenter.X:F1}, {bbCenter.Y:F1}, {bbCenter.Z:F1}), dist={dist:F0}");
        }

        _terrainManager.OnTileLoaded += OnTileLoaded;
        _terrainManager.OnTileUnloaded += OnTileUnloaded;

        onStatus?.Invoke("World loaded (tiles stream as you move).");
    }

    private Vector3? _wmoCameraOverride;
    /// <summary>For WMO-only maps, returns the WMO position as camera start. Otherwise null.</summary>
    public Vector3? WmoCameraOverride => _wmoCameraOverride;

    private void BuildInstances(ITerrainAdapter adapter)
    {
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // Placement transform for terrain maps.
        // Positions are already converted to renderer coords in AlphaTerrainAdapter:
        //   rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX, rendererZ = wowZ
        // X negation is baked into vertex data at upload time (WmoRenderer/ModelRenderer),
        // so placement transforms use identity — just scale, rotation, translation.
        bool wmoBased = adapter.IsWmoBased;

        // MDX (doodad) placements
        foreach (var p in adapter.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            // MDX geometry is offset from origin — pre-translate by -boundsCenter
            // so the bounding box center aligns with origin before scale/rotation/translation.
            Matrix4x4 pivotCorrection = Matrix4x4.Identity;
            if (_assets.TryGetMdxPivotOffset(key, out var pivot))
                pivotCorrection = Matrix4x4.CreateTranslation(-pivot);

            var transform = pivotCorrection
                * Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Use actual model bounds if available, transformed to world space
            Vector3 bbMin, bbMax;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
            {
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            }
            else
            {
                bbMin = p.Position - new Vector3(2f);
                bbMax = p.Position + new Vector3(2f);
            }
            _mdxInstances.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = bbMin,
                BoundsMax = bbMax
            });
        }

        // WMO placements
        foreach (var p in adapter.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            var transform = Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Get MOHD local bounds from the loaded WMO model and transform to world space.
            // The WMO is a container — its internal geometry has its own local bounding box
            // (MOHD bounds) around the WMO's local origin. We transform that through the
            // placement matrix to get the correct world-space AABB.
            // Falls back to MODF file bounds if model isn't loaded.
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
            {
                TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
            }
            else
            {
                localMin = localMax = Vector3.Zero;
                worldMin = p.BoundsMin;
                worldMax = p.BoundsMax;
            }

            _wmoInstances.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = worldMin,
                BoundsMax = worldMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax
            });
        }

        ViewerLog.Important(ViewerLog.Category.Terrain, $"Instances: {_mdxInstances.Count} MDX, {_wmoInstances.Count} WMO");
        ViewerLog.Important(ViewerLog.Category.Terrain, $"Unique models: {UniqueMdxModels} MDX, {UniqueWmoModels} WMO");

        // Diagnostic: terrain chunk WorldPosition range
        var camPos = _terrainManager.GetInitialCameraPosition();
        ViewerLog.Info(ViewerLog.Category.Terrain, $"Camera: ({camPos.X:F1}, {camPos.Y:F1}, {camPos.Z:F1})");

        // Compute terrain bounding box from chunk WorldPositions
        float tMinX = float.MaxValue, tMinY = float.MaxValue, tMinZ = float.MaxValue;
        float tMaxX = float.MinValue, tMaxY = float.MinValue, tMaxZ = float.MinValue;
        foreach (var chunk in _terrainManager.Adapter.LastLoadedChunkPositions)
        {
            tMinX = Math.Min(tMinX, chunk.X); tMaxX = Math.Max(tMaxX, chunk.X);
            tMinY = Math.Min(tMinY, chunk.Y); tMaxY = Math.Max(tMaxY, chunk.Y);
            tMinZ = Math.Min(tMinZ, chunk.Z); tMaxZ = Math.Max(tMaxZ, chunk.Z);
        }
        ViewerLog.Info(ViewerLog.Category.Terrain, $"TERRAIN  X:[{tMinX:F1} .. {tMaxX:F1}]  Y:[{tMinY:F1} .. {tMaxY:F1}]  Z:[{tMinZ:F1} .. {tMaxZ:F1}]");

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
        ViewerLog.Info(ViewerLog.Category.Terrain, $"OBJECTS  X:[{oMinX:F1} .. {oMaxX:F1}]  Y:[{oMinY:F1} .. {oMaxY:F1}]  Z:[{oMinZ:F1} .. {oMaxZ:F1}]");
        ViewerLog.Info(ViewerLog.Category.Terrain, $"DELTA    X:{(tMinX+tMaxX)/2 - (oMinX+oMaxX)/2:F1}  Y:{(tMinY+tMaxY)/2 - (oMinY+oMaxY)/2:F1}  Z:{(tMinZ+tMaxZ)/2 - (oMinZ+oMaxZ)/2:F1}");

        // Print first 3 MDDF raw values for manual inspection
        for (int i = 0; i < Math.Min(3, adapter.MddfPlacements.Count); i++)
        {
            var p = adapter.MddfPlacements[i];
            string name = p.NameIndex < mdxNames.Count ? Path.GetFileName(mdxNames[p.NameIndex]) : "?";
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  MDDF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
        for (int i = 0; i < Math.Min(3, adapter.ModfPlacements.Count); i++)
        {
            var p = adapter.ModfPlacements[i];
            string name = p.NameIndex < wmoNames.Count ? Path.GetFileName(wmoNames[p.NameIndex]) : "?";
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  MODF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
    }

    /// <summary>
    /// Called by TerrainManager when a new tile enters the AOI.
    /// Builds object instances for the tile and lazy-loads any new models.
    /// </summary>
    private void OnTileLoaded(int tileX, int tileY, TileLoadResult result)
    {
        var adapter = _terrainManager.Adapter;
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // Build MDX instances for this tile
        var tileMdx = new List<ObjectInstance>();
        foreach (var p in result.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            _assets.EnsureMdxLoaded(key);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            // MDX geometry is offset from origin — pre-translate by -boundsCenter
            Matrix4x4 pivotCorrection = Matrix4x4.Identity;
            if (_assets.TryGetMdxPivotOffset(key, out var pivot))
                pivotCorrection = Matrix4x4.CreateTranslation(-pivot);

            // X negation baked into vertex data — placement is just scale/rot/translate
            var transform = pivotCorrection
                * Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);
            Vector3 bbMin, bbMax;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            else
            { bbMin = p.Position - new Vector3(2f); bbMax = p.Position + new Vector3(2f); }
            tileMdx.Add(new ObjectInstance { ModelKey = key, Transform = transform, BoundsMin = bbMin, BoundsMax = bbMax });
        }

        // Build WMO instances for this tile
        var tileWmo = new List<ObjectInstance>();
        foreach (var p in result.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            _assets.EnsureWmoLoaded(key);
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            // X negation baked into vertex data — placement is just rot/translate
            var transform = Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Get MOHD local bounds and transform to world space
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
            {
                TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
            }
            else
            {
                localMin = localMax = Vector3.Zero;
                worldMin = p.BoundsMin;
                worldMax = p.BoundsMax;
            }

            tileWmo.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = worldMin,
                BoundsMax = worldMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax
            });
        }

        _tileMdxInstances[(tileX, tileY)] = tileMdx;
        _tileWmoInstances[(tileX, tileY)] = tileWmo;
        _instancesDirty = true;

        if (tileMdx.Count > 0 || tileWmo.Count > 0)
            ViewerLog.Info(ViewerLog.Category.Terrain, $"Tile ({tileX},{tileY}) loaded: {tileMdx.Count} MDX, {tileWmo.Count} WMO instances");
    }

    /// <summary>
    /// Called by TerrainManager when a tile leaves the AOI.
    /// </summary>
    private void OnTileUnloaded(int tileX, int tileY)
    {
        _tileMdxInstances.Remove((tileX, tileY));
        _tileWmoInstances.Remove((tileX, tileY));
        _instancesDirty = true;
    }

    /// <summary>
    /// Rebuild flat instance lists from per-tile dictionaries.
    /// Called lazily before rendering when _instancesDirty is true.
    /// </summary>
    private void RebuildInstanceLists()
    {
        _mdxInstances.Clear();
        foreach (var list in _tileMdxInstances.Values)
            _mdxInstances.AddRange(list);
        _wmoInstances.Clear();
        foreach (var list in _tileWmoInstances.Values)
            _wmoInstances.AddRange(list);
        _instancesDirty = false;
    }

    /// <summary>
    /// Transform an axis-aligned bounding box through a matrix by transforming all 8 corners
    /// and computing the new AABB that encloses them.
    /// </summary>
    private static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax)
    {
        outMin = new Vector3(float.MaxValue);
        outMax = new Vector3(float.MinValue);
        Span<float> xs = stackalloc float[] { min.X, max.X };
        Span<float> ys = stackalloc float[] { min.Y, max.Y };
        Span<float> zs = stackalloc float[] { min.Z, max.Z };
        foreach (var x in xs)
        foreach (var y in ys)
        foreach (var z in zs)
        {
            var p = Vector3.Transform(new Vector3(x, y, z), m);
            outMin = Vector3.Min(outMin, p);
            outMax = Vector3.Max(outMax, p);
        }
    }

    // ── ISceneRenderer ──────────────────────────────────────────────────

    private bool _renderDiagPrinted = false;
    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        // Rebuild flat instance lists if tiles changed
        if (_instancesDirty)
            RebuildInstanceLists();

        // Extract camera position for sky dome
        Matrix4x4.Invert(view, out var viewInvSky);
        var camPos = new Vector3(viewInvSky.M41, viewInvSky.M42, viewInvSky.M43);

        // 0. Render sky dome (before terrain, no depth write)
        _skyDome.UpdateFromLighting(_terrainManager.Lighting.GameTime);
        _skyDome.Render(view, proj, camPos);

        // Also set clear color to horizon color so any gaps match the sky
        _gl.ClearColor(_skyDome.HorizonColor.X, _skyDome.HorizonColor.Y, _skyDome.HorizonColor.Z, 1f);

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
                else { wmoMissing++; if (wmoMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Wmo, $"NOT FOUND: \"{inst.ModelKey}\""); }
            }
            int mdxFound = 0, mdxMissing = 0;
            foreach (var inst in _mdxInstances)
            {
                if (_assets.GetMdx(inst.ModelKey) != null) mdxFound++;
                else { mdxMissing++; if (mdxMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Mdx, $"NOT FOUND: \"{inst.ModelKey}\""); }
            }
            ViewerLog.Info(ViewerLog.Category.Terrain, $"Render check: WMO {wmoFound} found / {wmoMissing} missing, MDX {mdxFound} found / {mdxMissing} missing");
        }

        // Extract camera position from view matrix (inverse of view translation)
        Matrix4x4.Invert(view, out var viewInv);
        var cameraPos = new Vector3(viewInv.M41, viewInv.M42, viewInv.M43);

        // ── PASS 1: OPAQUE ──────────────────────────────────────────────
        // Render all opaque geometry first with depth write ON.
        // This ensures correct depth buffer before any transparent rendering.
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Less);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);

        // 2a. WMO opaque pass
        if (_wmosVisible)
        {
            int wmoRendered = 0;
            foreach (var inst in _wmoInstances)
            {
                var renderer = _assets.GetWmo(inst.ModelKey);
                if (renderer == null) continue;
                _gl.Disable(EnableCap.Blend);
                _gl.DepthMask(true);
                renderer.RenderWithTransform(inst.Transform, view, proj);
                wmoRendered++;
            }
            if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Wmo, $"WMO render loop: {wmoRendered} rendered");
        }

        // 3a. MDX opaque pass
        if (_doodadsVisible)
        {
            int mdxRendered = 0;
            foreach (var inst in _mdxInstances)
            {
                var renderer = _assets.GetMdx(inst.ModelKey);
                if (renderer == null) continue;
                _gl.Disable(EnableCap.Blend);
                _gl.DepthMask(true);
                renderer.RenderWithTransform(inst.Transform, view, proj, RenderPass.Opaque);
                mdxRendered++;
            }
            if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Mdx, $"MDX opaque pass: {mdxRendered} rendered");
        }

        // ── PASS 2: TRANSPARENT (back-to-front) ────────────────────────
        // Render transparent/blended layers sorted by distance to camera.
        // Depth test ON but depth write OFF so transparent objects don't
        // occlude each other incorrectly.
        if (_doodadsVisible)
        {
            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthFunc(DepthFunction.Lequal);

            // Sort instances back-to-front by distance to camera
            var sorted = new List<(int idx, float dist)>(_mdxInstances.Count);
            for (int i = 0; i < _mdxInstances.Count; i++)
            {
                var inst = _mdxInstances[i];
                if (_assets.GetMdx(inst.ModelKey) == null) continue;
                var center = (inst.BoundsMin + inst.BoundsMax) * 0.5f;
                float dist = Vector3.DistanceSquared(cameraPos, center);
                sorted.Add((i, dist));
            }
            sorted.Sort((a, b) => b.dist.CompareTo(a.dist)); // back-to-front

            foreach (var (idx, _) in sorted)
            {
                var inst = _mdxInstances[idx];
                var renderer = _assets.GetMdx(inst.ModelKey);
                renderer!.RenderWithTransform(inst.Transform, view, proj, RenderPass.Transparent);
            }
            if (!_renderDiagPrinted) _renderDiagPrinted = true;
        }
        else
        {
            if (!_renderDiagPrinted) _renderDiagPrinted = true;
        }

        // ── PASS 3: LIQUID ──────────────────────────────────────────────
        // Render liquid surfaces LAST so all opaque geometry (terrain, WMOs, MDX)
        // is already in the framebuffer. Liquid uses alpha blending + depth mask off,
        // so objects below the water surface are visible through the transparent water.
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _terrainManager.RenderLiquid(view, proj, cameraPos);

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
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"BB render: {adapter.MddfPlacements.Count} MDDF + {adapter.ModfPlacements.Count} MODF markers");
            // MDDF bounding boxes (yellow)
            foreach (var inst in _mdxInstances)
                _bbRenderer.DrawBoxMinMax(inst.BoundsMin, inst.BoundsMax, view, proj, new Vector3(1f, 1f, 0f));
            // MODF bounding boxes (cyan = world-space transformed bounds)
            foreach (var inst in _wmoInstances)
            {
                _bbRenderer.DrawBoxMinMax(inst.BoundsMin, inst.BoundsMax, view, proj, new Vector3(0f, 1f, 1f));
                // If local MOHD bounds are available, also draw them in green
                if (inst.LocalBoundsMin != Vector3.Zero || inst.LocalBoundsMax != Vector3.Zero)
                {
                    TransformBounds(inst.LocalBoundsMin, inst.LocalBoundsMax, inst.Transform,
                        out var lWorldMin, out var lWorldMax);
                    _bbRenderer.DrawBoxMinMax(lWorldMin, lWorldMax, view, proj, new Vector3(0f, 1f, 0f));
                }
            }

            _gl.Enable(EnableCap.DepthTest);
        }

        // 5. POI markers (magenta, always on top)
        if (_showPoi && _poiLoader != null && _bbRenderer != null && _poiLoader.Entries.Count > 0)
        {
            _gl.Disable(EnableCap.DepthTest);
            foreach (var poi in _poiLoader.Entries)
                _bbRenderer.DrawMarker(poi.Position, 8f, view, proj, new Vector3(1f, 0f, 1f));
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
        _terrainManager.OnTileLoaded -= OnTileLoaded;
        _terrainManager.OnTileUnloaded -= OnTileUnloaded;
        _terrainManager.Dispose();
        _assets.Dispose();
        _bbRenderer?.Dispose();
        _skyDome.Dispose();
        _mdxInstances.Clear();
        _wmoInstances.Clear();
        _tileMdxInstances.Clear();
        _tileWmoInstances.Clear();
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
    /// <summary>World-space AABB (local bounds transformed through placement matrix).</summary>
    public Vector3 BoundsMin;
    /// <summary>World-space AABB (local bounds transformed through placement matrix).</summary>
    public Vector3 BoundsMax;
    /// <summary>Model-local bounding box min (MOHD for WMO, model extents for MDX). Zero if unavailable.</summary>
    public Vector3 LocalBoundsMin;
    /// <summary>Model-local bounding box max (MOHD for WMO, model extents for MDX). Zero if unavailable.</summary>
    public Vector3 LocalBoundsMax;
}
