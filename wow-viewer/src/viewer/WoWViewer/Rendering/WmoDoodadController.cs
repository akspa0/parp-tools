using System.Diagnostics;
using System.Numerics;
using System.Text;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.World;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Terrain;

namespace WoWViewer.Rendering;

/// <summary>
/// Controls loading, lifecycle, visibility, culling, ray picking, and rendering of WMO doodad instances.
/// </summary>
internal sealed class WmoDoodadController : IDisposable
{
    public const int DefaultDeferredDoodadLoads = 1;
    public const double DefaultDeferredDoodadBudgetMs = 2.0;

    private readonly GL _gl;
    private readonly WmoV14ToV17Converter.WmoV14Data _wmo;
    private readonly string _modelDir;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;
    private readonly string? _buildVersion;
    private readonly bool _deferInitialDoodadLoads;

    private bool _doodadsVisible = true;
    private bool _runtimeDoodadsVisible = true;
    private bool _supportsGpuInstancedOpaque = true;

    private readonly HashSet<int> _runtimeVisibleDoodadDefIndices = new();
    private readonly HashSet<string> _updatedDoodadModelsScratch = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<(int idx, float distSq)> _visibleDoodadsScratch = new();

    private readonly Dictionary<IModelRenderer, List<int>> _opaqueDoodadBatchGroups = new();
    private readonly List<IModelRenderer> _opaqueDoodadBatchRenderers = new();
    private bool _doodadAnimationsPreparedForWorldFrame;

    private readonly Dictionary<string, IModelRenderer?> _doodadModelCache = new(StringComparer.OrdinalIgnoreCase);
    internal WmoDoodadModelShare? DoodadModelShare { get; set; }
    private bool _lastDoodadLoadWasCacheHit;
    private readonly Dictionary<string, M2RouteDecision?> _doodadRouteDecisions = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<DoodadInstance> _doodadInstances = new();
    private readonly List<string> _doodadNames = new(); // resolved from MODN
    private readonly Dictionary<string, string> _canonicalDoodadPathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string?> _bestSkinPathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _loggedMissingDoodadSkinPaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly Queue<string> _pendingDoodadModelLoads = new();
    private readonly HashSet<string> _queuedDoodadModelLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<int>> _doodadInstanceIndicesByModel = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _doodadSourceModelPaths = new(StringComparer.OrdinalIgnoreCase);

    private int _activeDoodadSet;
    private enum DoodadLoadResult { Loaded, NotFound, ParseError }
    private DoodadLoadResult _lastLoadResult;

    public float DoodadCullDistance { get; set; } = 4000f;
    public uint DoodadMaxRenderCount { get; set; } = 1024;

    public M2RouteDecision? GetDoodadRouteDecision(string normalizedPath)
        => _doodadRouteDecisions.TryGetValue(normalizedPath, out var decision) ? decision : null;

    public bool DoodadsVisible
    {
        get => _doodadsVisible;
        set => _doodadsVisible = value;
    }

    public bool RuntimeDoodadsVisible
    {
        get => _runtimeDoodadsVisible;
        set => _runtimeDoodadsVisible = value;
    }

    public bool SupportsGpuInstancedOpaque
    {
        get => _supportsGpuInstancedOpaque;
        set => _supportsGpuInstancedOpaque = value;
    }

    public bool DoodadAnimationsPreparedForWorldFrame
    {
        get => _doodadAnimationsPreparedForWorldFrame;
        set => _doodadAnimationsPreparedForWorldFrame = value;
    }

    public int DoodadSetCount => _wmo.DoodadSets.Count;
    public int ActiveDoodadSet => _activeDoodadSet;
    public int DoodadInstanceCount => _doodadInstances.Count;
    public int DoodadDefCount => _wmo.DoodadDefs.Count;
    public int PendingDoodadModelLoadsCount => _pendingDoodadModelLoads.Count;
    public int CachedDoodadModelCount => _doodadModelCache.Count;
    public IReadOnlyList<DoodadInstance> DoodadInstances => _doodadInstances;

    public WmoDoodadController(
        GL gl,
        WmoV14ToV17Converter.WmoV14Data wmo,
        string modelDir,
        IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver,
        string? buildVersion,
        bool deferInitialDoodadLoads)
    {
        _gl = gl;
        _wmo = wmo;
        _modelDir = modelDir;
        _dataSource = dataSource;
        _texResolver = texResolver;
        _buildVersion = buildVersion;
        _deferInitialDoodadLoads = deferInitialDoodadLoads;

        ResolveDoodadNames();
        if (_wmo.DoodadSets.Count > 0)
            LoadActiveDoodadSet();
    }

    public void ResolveDoodadNames()
    {
        _doodadNames.Clear();
        if (_wmo.DoodadNamesRaw.Length == 0) return;

        var raw = _wmo.DoodadNamesRaw;
        int start = 0;
        for (int i = 0; i <= raw.Length; i++)
        {
            if (i == raw.Length || raw[i] == 0)
            {
                if (i > start)
                {
                    // Names resolved on demand by offset in GetDoodadName
                }
                start = i + 1;
            }
        }
    }

    public string GetDoodadName(uint nameOffset)
    {
        if (nameOffset >= _wmo.DoodadNamesRaw.Length) return "";
        int end = (int)nameOffset;
        while (end < _wmo.DoodadNamesRaw.Length && _wmo.DoodadNamesRaw[end] != 0)
            end++;
        if (end == (int)nameOffset) return "";
        return Encoding.UTF8.GetString(_wmo.DoodadNamesRaw, (int)nameOffset, end - (int)nameOffset);
    }

    public void LoadActiveDoodadSet()
    {
        _doodadInstances.Clear();
        _pendingDoodadModelLoads.Clear();
        _queuedDoodadModelLoads.Clear();
        _doodadInstanceIndicesByModel.Clear();
        _doodadSourceModelPaths.Clear();

        if (_wmo.DoodadSets.Count == 0 || _wmo.DoodadDefs.Count == 0)
            return;

        if (_activeDoodadSet >= _wmo.DoodadSets.Count)
            _activeDoodadSet = 0;

        var set = _wmo.DoodadSets[_activeDoodadSet];
        ViewerLog.Trace($"[WmoRenderer] Loading DoodadSet [{_activeDoodadSet}] \"{set.Name}\": {set.Count} doodads (start={set.StartIndex}), DoodadDefs.Count={_wmo.DoodadDefs.Count}, DoodadNamesRaw.Length={_wmo.DoodadNamesRaw.Length}");

        int loaded = 0, failed = 0, emptyName = 0, notFound = 0, parseError = 0, deferredUniqueModels = 0;
        for (uint i = set.StartIndex; i < set.StartIndex + set.Count && i < (uint)_wmo.DoodadDefs.Count; i++)
        {
            var def = _wmo.DoodadDefs[(int)i];
            string modelPath = GetDoodadName(def.NameIndex);

            if (string.IsNullOrEmpty(modelPath))
            {
                emptyName++;
                failed++;
                continue;
            }

            var transform = Matrix4x4.CreateScale(def.Scale)
                          * Matrix4x4.CreateFromQuaternion(def.Orientation)
                          * Matrix4x4.CreateTranslation(def.Position);

            string normalizedModelPath = NormalizeDoodadPath(modelPath).ToLowerInvariant();
            _doodadSourceModelPaths[normalizedModelPath] = modelPath;
            if (!_doodadInstanceIndicesByModel.TryGetValue(normalizedModelPath, out List<int>? instanceIndices))
            {
                instanceIndices = new List<int>();
                _doodadInstanceIndicesByModel[normalizedModelPath] = instanceIndices;
            }

            IModelRenderer? renderer = null;
            if (_deferInitialDoodadLoads)
            {
                if (_queuedDoodadModelLoads.Add(normalizedModelPath))
                {
                    _pendingDoodadModelLoads.Enqueue(normalizedModelPath);
                    deferredUniqueModels++;
                }
            }
            else
            {
                renderer = GetOrLoadDoodadModel(modelPath);
            }

            _doodadInstances.Add(new DoodadInstance
            {
                ModelPath = modelPath,
                NormalizedModelPath = normalizedModelPath,
                Renderer = renderer,
                Transform = transform,
                Visible = true,
                DoodadDefIndex = (int)i,
                LocalPosition = def.Position
            });
            instanceIndices.Add(_doodadInstances.Count - 1);

            if (_deferInitialDoodadLoads)
                continue;

            if (renderer != null)
                loaded++;
            else
            {
                failed++;
                if (_lastLoadResult == DoodadLoadResult.NotFound) notFound++;
                else if (_lastLoadResult == DoodadLoadResult.ParseError) parseError++;
            }
        }

        if (_deferInitialDoodadLoads)
        {
            ViewerLog.Trace($"[WmoRenderer] Doodads queued for deferred loading: {_doodadInstances.Count} instances, {deferredUniqueModels} unique models");
        }
        else
        {
            ViewerLog.Trace($"[WmoRenderer] Doodads: {loaded} loaded, {failed} failed ({emptyName} empty names, {notFound} not found, {parseError} parse errors), {_doodadModelCache.Count} unique models cached");
        }
    }

    public void SetActiveDoodadSet(int index)
    {
        if (index == _activeDoodadSet || index < 0 || index >= _wmo.DoodadSets.Count) return;
        _activeDoodadSet = index;
        LoadActiveDoodadSet();
    }

    public void SetDoodadVisible(int index, bool visible)
    {
        if (index >= 0 && index < _doodadInstances.Count)
            _doodadInstances[index].Visible = visible;
    }

    public void SetRuntimeDoodadsVisible(bool visible)
    {
        _runtimeDoodadsVisible = visible;
    }

    public string GetDoodadSetName(int index) =>
        index < _wmo.DoodadSets.Count ? (_wmo.DoodadSets[index].Name ?? $"Set {index}") : "";

    public bool TryGetDoodadSetRange(int index, out string name, out int startIndex, out int count)
    {
        if (index >= 0 && index < _wmo.DoodadSets.Count)
        {
            var set = _wmo.DoodadSets[index];
            name = set.Name ?? $"Set {index}";
            startIndex = (int)set.StartIndex;
            count = (int)set.Count;
            return true;
        }
        name = string.Empty;
        startIndex = 0;
        count = 0;
        return false;
    }

    public bool TryGetDoodadInfo(int index, out WmoDoodadInfo info)
    {
        if (index >= 0 && index < _doodadInstances.Count)
        {
            DoodadInstance doodad = _doodadInstances[index];
            Quaternion orientation = Quaternion.Identity;
            float scale = 1f;
            uint nameIndex = 0;
            if (TryGetDoodadDef(doodad.DoodadDefIndex, out WmoV14ToV17Converter.WmoDoodadDef def))
            {
                orientation = def.Orientation;
                scale = def.Scale;
                nameIndex = def.NameIndex;
            }

            info = new WmoDoodadInfo(
                index,
                doodad.ModelPath,
                doodad.DoodadDefIndex,
                doodad.LocalPosition,
                doodad.Visible,
                doodad.Renderer != null,
                orientation,
                scale,
                nameIndex);
            return true;
        }

        info = default;
        return false;
    }

    public bool TryGetDoodadBounds(int index, in Matrix4x4 modelMatrix, out Vector3 boundsMin, out Vector3 boundsMax, out bool boundsResolved)
    {
        if (index >= 0 && index < _doodadInstances.Count)
        {
            DoodadInstance doodad = _doodadInstances[index];
            Matrix4x4 doodadWorld = doodad.Transform * modelMatrix;
            if (doodad.Renderer is IModelRenderer modelRenderer)
            {
                WmoGeometryHelper.TransformAabb(modelRenderer.BoundsMin, modelRenderer.BoundsMax, doodadWorld, out boundsMin, out boundsMax);
                boundsResolved = true;
                return true;
            }

            Vector3 worldPosition = Vector3.Transform(doodad.LocalPosition, modelMatrix);
            boundsMin = worldPosition - new Vector3(2f);
            boundsMax = worldPosition + new Vector3(2f);
            boundsResolved = false;
            return true;
        }

        boundsMin = boundsMax = Vector3.Zero;
        boundsResolved = false;
        return false;
    }

    public bool TryGetDoodadWorldTransform(int index, in Matrix4x4 modelMatrix, out Matrix4x4 transform)
    {
        if (index >= 0 && index < _doodadInstances.Count)
        {
            transform = _doodadInstances[index].Transform * modelMatrix;
            return true;
        }

        transform = Matrix4x4.Identity;
        return false;
    }

    public bool TryGetDoodadLocalBounds(int index, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (index >= 0 && index < _doodadInstances.Count
            && _doodadInstances[index].Renderer is IModelRenderer modelRenderer)
        {
            boundsMin = modelRenderer.BoundsMin;
            boundsMax = modelRenderer.BoundsMax;
            return true;
        }

        boundsMin = boundsMax = Vector3.Zero;
        return false;
    }

    public void RequestDoodadModelLoad(int index)
    {
        if (!_deferInitialDoodadLoads || index < 0 || index >= _doodadInstances.Count)
            return;

        DoodadInstance doodad = _doodadInstances[index];
        if (doodad.Renderer != null)
            return;

        if (!_queuedDoodadModelLoads.Add(doodad.NormalizedModelPath))
            return;

        _pendingDoodadModelLoads.Enqueue(doodad.NormalizedModelPath);
    }

    public bool TryPickDoodadsByRay(
        Vector3 rayOrigin,
        Vector3 rayDir,
        in Matrix4x4 modelMatrix,
        List<(int index, float distance, Vector3 hitPoint, Vector3 boundsMin, Vector3 boundsMax, WmoDoodadInfo info)> hits)
    {
        if (!_doodadsVisible || _doodadInstances.Count == 0)
            return false;

        bool found = false;
        Vector3 padding = new Vector3(0.5f);
        for (int i = 0; i < _doodadInstances.Count; i++)
        {
            var d = _doodadInstances[i];
            if (!d.Visible)
                continue;

            if (!TryGetDoodadBounds(i, modelMatrix, out Vector3 bMin, out Vector3 bMax, out _))
                continue;

            float dist = Terrain.WorldScene.RayAABBIntersect(rayOrigin, rayDir, bMin - padding, bMax + padding);
            if (dist >= 0f)
            {
                Vector3 hitPoint = rayOrigin + rayDir * dist;
                TryGetDoodadInfo(i, out WmoDoodadInfo info);
                hits.Add((i, dist, hitPoint, bMin, bMax, info));
                found = true;
            }
        }

        return found;
    }

    public bool TryGetDoodadDef(int doodadDefIndex, out WmoV14ToV17Converter.WmoDoodadDef def)
    {
        if (doodadDefIndex >= 0 && doodadDefIndex < _wmo.DoodadDefs.Count)
        {
            def = _wmo.DoodadDefs[doodadDefIndex];
            return true;
        }
        def = default;
        return false;
    }

    public string GetDoodadDefName(int doodadDefIndex)
    {
        if (doodadDefIndex >= 0 && doodadDefIndex < _wmo.DoodadDefs.Count)
        {
            var def = _wmo.DoodadDefs[doodadDefIndex];
            return GetDoodadName(def.NameIndex);
        }
        return "";
    }

    public void ClearRuntimeVisibleDoodadDefs() => _runtimeVisibleDoodadDefIndices.Clear();
    public void AddRuntimeVisibleDoodadDef(int defIndex) => _runtimeVisibleDoodadDefIndices.Add(defIndex);

    public void BeginWorldFrame()
    {
        _doodadAnimationsPreparedForWorldFrame = true;
        UpdateDoodadAnimations();
    }

    public void EndWorldFrame()
    {
        _doodadAnimationsPreparedForWorldFrame = false;
    }

    public int ProcessDeferredDoodadLoads(
        int maxLoads = DefaultDeferredDoodadLoads,
        double maxBudgetMs = DefaultDeferredDoodadBudgetMs)
    {
        if (!_deferInitialDoodadLoads || _pendingDoodadModelLoads.Count == 0)
            return 0;

        if (maxLoads <= 0 || maxBudgetMs <= 0)
            return 0;

        var stopwatch = Stopwatch.StartNew();
        int loadsCompleted = 0;
        while (loadsCompleted < maxLoads
            && stopwatch.Elapsed.TotalMilliseconds < maxBudgetMs
            && _pendingDoodadModelLoads.TryDequeue(out string? normalizedModelPath))
        {
            _queuedDoodadModelLoads.Remove(normalizedModelPath);
            if (!_doodadInstanceIndicesByModel.TryGetValue(normalizedModelPath, out List<int>? indices) || indices.Count == 0)
                continue;

            string modelPath = _doodadSourceModelPaths.TryGetValue(normalizedModelPath, out string? sourceModelPath)
                ? sourceModelPath
                : _doodadInstances[indices[0]].ModelPath;
            IModelRenderer? renderer = GetOrLoadDoodadModel(modelPath);
            foreach (int idx in indices)
                _doodadInstances[idx].Renderer = renderer;

            if (!_lastDoodadLoadWasCacheHit)
                loadsCompleted++;
        }

        return loadsCompleted;
    }

    private IModelRenderer? GetOrLoadDoodadModel(string modelPath)
    {
        string normalized = NormalizeDoodadPath(modelPath).ToLowerInvariant();

        _lastDoodadLoadWasCacheHit = true;
        if (_doodadModelCache.TryGetValue(normalized, out var cached))
        {
            _lastLoadResult = cached != null ? DoodadLoadResult.Loaded : DoodadLoadResult.NotFound;
            return cached;
        }

        if (DoodadModelShare != null && DoodadModelShare.TryAcquire(normalized, out IModelRenderer? shared))
        {
            _doodadModelCache[normalized] = shared;
            _lastLoadResult = shared != null ? DoodadLoadResult.Loaded : DoodadLoadResult.NotFound;
            return shared;
        }

        _lastDoodadLoadWasCacheHit = false;

        IModelRenderer? renderer = null;
        _lastLoadResult = DoodadLoadResult.NotFound;
        try
        {
            string resolvedModelPath;
            byte[]? modelData;
            string normalizedModelPath = NormalizeDoodadPath(modelPath);

            if (!TryReadPreferredClassicDoodadData(normalizedModelPath, out resolvedModelPath, out modelData))
            {
                resolvedModelPath = ResolveCanonicalDoodadPath(modelPath);
                modelData = ReadDoodadFileData(resolvedModelPath);
                if ((modelData == null || modelData.Length == 0) && !resolvedModelPath.Equals(modelPath, StringComparison.OrdinalIgnoreCase))
                    modelData = ReadDoodadFileData(modelPath);
            }

            if (modelData == null || modelData.Length == 0)
            {
                if (_doodadModelCache.Count < 30) // only log first 30 unique misses
                    ViewerLog.Trace($"  Doodad not found: {modelPath}");

                _doodadModelCache[normalized] = null;
                DoodadModelShare?.Add(normalized, null);
                return null;
            }

            bool isM2Family = WarcraftNetM2Adapter.IsM2FamilyContainer(modelData);

            if (isM2Family)
            {
                renderer = LoadM2DoodadRenderer(modelPath, resolvedModelPath, modelData);
            }
            else
            {
                using var stream = new MemoryStream(modelData);
                var mdx = MdxFile.Load(stream);
                string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;
                renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver, resolvedModelPath);
            }

            if (renderer != null)
            {
                _lastLoadResult = DoodadLoadResult.Loaded;
                ViewerLog.Trace($"  Doodad loaded: {Path.GetFileName(modelPath)}");
            }
        }
        catch (Exception ex)
        {
            _lastLoadResult = DoodadLoadResult.ParseError;
            ViewerLog.Trace($"  Doodad load failed: {modelPath} — {ex.Message}");
        }

        _doodadModelCache[normalized] = renderer;
        DoodadModelShare?.Add(normalized, renderer);
        return renderer;
    }

    private IModelRenderer? LoadM2DoodadRenderer(string originalModelPath, string resolvedModelPath, byte[] modelData)
    {
        WarcraftNetM2Adapter.ValidateModelProfile(modelData, resolvedModelPath, _buildVersion);
        string buildProfileId = FormatProfileRegistry.ResolveModelProfile(_buildVersion)?.ProfileId ?? "unknown";

        var candidatePaths = new List<string>(WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath));
        string? bestSkinPath = ResolveBestSkinPath(resolvedModelPath);
        if (!string.IsNullOrWhiteSpace(bestSkinPath))
            candidatePaths.Add(bestSkinPath);

        Exception? lastSkinError = null;
        bool anySkinFound = false;

        foreach (string skinPath in candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            byte[]? skinBytes = ReadDoodadFileData(skinPath);
            if (skinBytes == null || skinBytes.Length == 0)
                continue;

            anySkinFound = true;

            try
            {
                ViewerLog.Trace($"[M2] Trying WMO doodad skin for {Path.GetFileName(originalModelPath)}: {skinPath} ({skinBytes.Length} bytes)");
                M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(modelData, skinBytes, resolvedModelPath, skinPath);
                MdxFile? adapted = null;
                try
                {
                    if (!WowViewerM2RuntimeBridge.PreferNativeStaticRenderer)
                        adapted = WarcraftNetM2Adapter.BuildRuntimeModel(modelData, skinBytes, resolvedModelPath, _buildVersion);
                }
                catch (Exception adapterEx)
                {
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] M2->MDX adapter fallback failed for {Path.GetFileName(resolvedModelPath)}: {adapterEx.Message} (native renderer will be used)");
                }

                var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterSkin, M2RouteType.AdapterSkin, skinPath);
                _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                M2RouteDiagnostics.LogRouteDecision(route);

                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Selected WMO doodad skin for {Path.GetFileName(originalModelPath)}: {skinPath} ({skinBytes.Length} bytes)");
                return WowViewerM2RuntimeBridge.CreateRenderer(
                    _gl,
                    runtimeModel,
                    adapted,
                    Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir,
                    _dataSource,
                    _texResolver,
                    _buildVersion,
                    resolvedModelPath,
                    deferInitialTextureLoads: _deferInitialDoodadLoads);
            }
            catch (Exception ex)
            {
                lastSkinError = ex;
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] WMO doodad skin candidate failed for {Path.GetFileName(originalModelPath)}: {skinPath} ({ex.Message})");
            }
        }

        if (!anySkinFound)
        {
            if (WarcraftNetM2Adapter.SupportsEmbeddedNativeRoute(_buildVersion))
            {
                try
                {
                    M2StaticRenderModel runtimeModel = WarcraftNetM2Adapter.BuildEmbeddedStaticRenderModel(modelData, resolvedModelPath, _buildVersion);
                    var route = M2RouteDecision.Create(
                        originalModelPath,
                        buildProfileId,
                        M2RouteType.NativeEmbeddedProfile,
                        M2RouteType.NativeEmbeddedProfile,
                        fallbackReason: "No external .skin resolved for WMO doodad; using native embedded root-profile geometry");
                    _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                    M2RouteDiagnostics.LogRouteDecision(route);

                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Loaded native embedded root-profile geometry for WMO doodad {Path.GetFileName(originalModelPath)} after no external .skin resolved");
                    return WowViewerM2RuntimeBridge.CreateRenderer(
                        _gl,
                        runtimeModel,
                        adaptedMdx: null,
                        modelDir: Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir,
                        dataSource: _dataSource,
                        texResolver: _texResolver,
                        buildVersion: _buildVersion,
                        sourceModelPath: resolvedModelPath,
                        deferInitialTextureLoads: _deferInitialDoodadLoads);
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Native embedded root-profile WMO doodad route failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }

            if (string.Equals(FormatProfileRegistry.ResolveModelProfile(_buildVersion)?.ProfileId, FormatProfileRegistry.M2Profile3018303.ProfileId, StringComparison.Ordinal))
            {
                try
                {
                    var adapted = WarcraftNetM2Adapter.BuildRuntimeModel(modelData, null, resolvedModelPath, _buildVersion);
                    string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;

                    var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterEmbeddedProfile, M2RouteType.AdapterEmbeddedProfile, fallbackReason: "No external .skin resolved for WMO doodad, using embedded root-profile");
                    _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                    M2RouteDiagnostics.LogRouteDecision(route);

                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Loaded embedded root-profile geometry for WMO doodad {Path.GetFileName(originalModelPath)} after no external .skin resolved");
                    return new M2Renderer(
                        new MdxRenderer(_gl, adapted, modelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion),
                        resolvedModelPath);
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Embedded root-profile WMO doodad fallback failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }

            M2Era1121EraTag detectedEra = M2ModelReaderDispatcher.DetectEra(modelData.AsSpan(), resolvedModelPath);
            if (detectedEra is M2Era1121EraTag.Md20_1X_V100 or M2Era1121EraTag.Md20_1X_V101)
            {
                try
                {
                    var adapted = WarcraftNetM2Adapter.BuildRuntimeModel(modelData, null, resolvedModelPath, _buildVersion);
                    string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;

                    var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterEmbeddedProfile, M2RouteType.AdapterEmbeddedProfile, fallbackReason: $"1.12.1 WMO doodad (era={detectedEra.ToDisplayString()}), no external .skin needed");
                    _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                    M2RouteDiagnostics.LogRouteDecision(route);

                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Loaded embedded 1.12.1 geometry for WMO doodad {Path.GetFileName(originalModelPath)} (era={detectedEra.ToDisplayString()})");
                    return new M2Renderer(
                        new MdxRenderer(_gl, adapted, modelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion),
                        resolvedModelPath);
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Embedded 1.12.1 WMO doodad fallback failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }

            if (_loggedMissingDoodadSkinPaths.Add(resolvedModelPath))
                ViewerLog.Important(ViewerLog.Category.Mdx, $"[M2] Missing WMO doodad .skin for: {Path.GetFileName(originalModelPath)}");
        }

        if (WarcraftNetM2Adapter.IsMd20(modelData))
        {
            byte[]? convertedBytes = ConvertM2ToMdx(modelData, resolvedModelPath);
            if (convertedBytes != null && convertedBytes.Length > 0)
            {
                try
                {
                    using var convertedStream = new MemoryStream(convertedBytes);
                    var convertedMdx = MdxFile.Load(convertedStream);
                    if (WarcraftNetM2Adapter.HasRenderableGeometry(convertedMdx))
                    {
                        string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;

                        var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterSkin, M2RouteType.ConversionFallback, fallbackReason: "Adapter/skin path failed for WMO doodad, fell back to M2->MDX conversion");
                        _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                        M2RouteDiagnostics.LogRouteDecision(route);

                        ViewerLog.Info(ViewerLog.Category.Mdx,
                            $"[M2] Falling back to M2->MDX conversion for WMO doodad {Path.GetFileName(originalModelPath)} after adapter failure");
                        return new M2Renderer(
                            new MdxRenderer(_gl, convertedMdx, modelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion),
                            resolvedModelPath);
                    }

                    lastSkinError = new InvalidDataException(
                        $"M2->MDX fallback produced no renderable geometry for WMO doodad {Path.GetFileName(originalModelPath)} ({WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)})");
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Rejecting converted WMO doodad fallback for {Path.GetFileName(originalModelPath)}: {WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)}");
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Converted WMO doodad fallback load failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }
        }

        if (lastSkinError != null)
            throw new InvalidDataException($"All .skin candidates failed for WMO doodad M2: {Path.GetFileName(originalModelPath)}", lastSkinError);

        return null;
    }

    private byte[]? ConvertM2ToMdx(byte[] modelData, string resolvedModelPath)
    {
        try
        {
            byte[]? skinBytes = null;
            foreach (string skinPath in WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath).Distinct(StringComparer.OrdinalIgnoreCase))
            {
                skinBytes = ReadDoodadFileData(skinPath);
                if (skinBytes != null && skinBytes.Length > 0)
                    break;
            }

            var converter = new WoWViewer.Transfer.M2ToMdxConverter();
            return converter.ConvertToBytes(modelData, skinBytes, _buildVersion);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] WMO doodad M2->MDX converter fallback failed for {Path.GetFileName(resolvedModelPath)}: {ex.Message}");
            return null;
        }
    }

    private string ResolveCanonicalDoodadPath(string modelPath)
    {
        string normalizedPath = NormalizeDoodadPath(modelPath);
        if (_canonicalDoodadPathCache.TryGetValue(normalizedPath, out string? cachedPath))
            return cachedPath;

        string resolvedPath = normalizedPath;
        if (_dataSource is MpqDataSource mpqDataSource)
        {
            string? found = mpqDataSource.FindInFileSet(normalizedPath);
            if (!string.IsNullOrWhiteSpace(found))
            {
                resolvedPath = NormalizeDoodadPath(found);
            }
            else
            {
                foreach (string alternatePath in EnumerateAlternateDoodadPaths(normalizedPath))
                {
                    found = mpqDataSource.FindInFileSet(alternatePath);
                    if (string.IsNullOrWhiteSpace(found))
                        continue;

                    resolvedPath = NormalizeDoodadPath(found);
                    break;
                }
            }
        }

        if (resolvedPath.Equals(normalizedPath, StringComparison.OrdinalIgnoreCase) && _dataSource != null)
        {
            if (!_dataSource.FileExists(normalizedPath))
            {
                foreach (string alternatePath in EnumerateAlternateDoodadPaths(normalizedPath))
                {
                    if (_dataSource.FileExists(alternatePath))
                    {
                        resolvedPath = NormalizeDoodadPath(alternatePath);
                        break;
                    }
                }
            }
        }

        _canonicalDoodadPathCache[normalizedPath] = resolvedPath;
        return resolvedPath;
    }

    private string? ResolveBestSkinPath(string resolvedModelPath)
    {
        if (_bestSkinPathCache.TryGetValue(resolvedModelPath, out string? cachedPath))
            return cachedPath;

        IReadOnlyList<string> skinFiles = _dataSource?.GetFileList(".skin") ?? Array.Empty<string>();
        string? bestSkinPath = DoodadModelShare != null
            ? DoodadModelShare.SkinIndex.FindBestSkin(resolvedModelPath, skinFiles)
            : WarcraftNetM2Adapter.FindSkinInFileList(resolvedModelPath, skinFiles);

        _bestSkinPathCache[resolvedModelPath] = bestSkinPath;
        return bestSkinPath;
    }

    private byte[]? ReadDoodadFileData(string path)
    {
        string normalizedPath = NormalizeDoodadPath(path);

        if (_dataSource != null)
        {
            byte[]? data = _dataSource.ReadFile(path);
            if ((data == null || data.Length == 0) && !normalizedPath.Equals(path, StringComparison.OrdinalIgnoreCase))
                data = _dataSource.ReadFile(normalizedPath);

            if ((data == null || data.Length == 0) && _dataSource is MpqDataSource mpqDataSource)
            {
                string? found = mpqDataSource.FindInFileSet(normalizedPath);
                if (!string.IsNullOrWhiteSpace(found))
                    data = _dataSource.ReadFile(found);
            }

            if (data != null && data.Length > 0)
                return data;
        }

        string diskPath = path;
        if (!Path.IsPathRooted(diskPath))
            diskPath = Path.Combine(_modelDir, normalizedPath);

        if (File.Exists(diskPath))
            return File.ReadAllBytes(diskPath);

        string fallbackPath = Path.Combine(_modelDir, Path.GetFileName(normalizedPath));
        if (!fallbackPath.Equals(diskPath, StringComparison.OrdinalIgnoreCase) && File.Exists(fallbackPath))
            return File.ReadAllBytes(fallbackPath);

        return null;
    }

    private static string NormalizeDoodadPath(string path) => path.Replace('/', '\\');

    private static bool IsClassicDoodadRequest(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase);
    }

    private bool TryReadPreferredClassicDoodadData(string normalizedPath, out string resolvedPath, out byte[]? data)
    {
        resolvedPath = normalizedPath;
        data = null;

        if (!IsClassicDoodadRequest(normalizedPath))
            return false;

        foreach (string candidate in EnumeratePreferredClassicDoodadPaths(normalizedPath))
        {
            data = ReadDoodadFileData(candidate);
            if (data == null || data.Length == 0)
                continue;

            resolvedPath = NormalizeDoodadPath(candidate);
            return true;
        }

        data = null;
        resolvedPath = normalizedPath;
        return false;
    }

    private static IEnumerable<string> EnumeratePreferredClassicDoodadPaths(string normalizedPath)
    {
        yield return normalizedPath;

        if (normalizedPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return normalizedPath[..^4] + ".mdl";
            yield break;
        }

        if (normalizedPath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
            yield return normalizedPath[..^4] + ".mdx";
    }

    private static IEnumerable<string> EnumerateAlternateDoodadPaths(string normalizedPath)
    {
        if (normalizedPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return normalizedPath[..^4] + ".m2";
            yield return normalizedPath[..^4] + ".mdl";
            yield break;
        }

        if (normalizedPath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return normalizedPath[..^4] + ".mdx";
            yield return normalizedPath[..^4] + ".m2";
            yield break;
        }

        if (normalizedPath.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return normalizedPath[..^3] + ".mdx";
            yield return normalizedPath[..^3] + ".mdl";
        }
    }

    public int PrepareVisibleDoodads(
        Matrix4x4 modelMatrix,
        Vector3 cameraPos,
        float fogEnd,
        bool updateAnimation,
        bool sortByDistance = true,
        bool respectRuntimeDoodadVisibility = true)
    {
        _updatedDoodadModelsScratch.Clear();
        _visibleDoodadsScratch.Clear();

        if (updateAnimation && !_doodadAnimationsPreparedForWorldFrame)
            UpdateDoodadAnimations();

        float doodadCullDistance = MathF.Max(DoodadCullDistance, MathF.Min(fogEnd + 800f, 6000f));
        float cullDistSq = doodadCullDistance * doodadCullDistance;
        for (int di = 0; di < _doodadInstances.Count; di++)
        {
            DoodadInstance inst = _doodadInstances[di];
            if (!inst.Visible || inst.Renderer == null)
                continue;
            if (respectRuntimeDoodadVisibility
                && _runtimeVisibleDoodadDefIndices.Count > 0
                && !_runtimeVisibleDoodadDefIndices.Contains(inst.DoodadDefIndex))
                continue;

            Vector3 worldPos = Vector3.Transform(inst.LocalPosition, modelMatrix);
            float distSq = Vector3.DistanceSquared(cameraPos, worldPos);
            if (distSq > cullDistSq)
                continue;

            _visibleDoodadsScratch.Add((di, distSq));
        }

        if (sortByDistance && _visibleDoodadsScratch.Count > 1)
            _visibleDoodadsScratch.Sort((a, b) => a.distSq.CompareTo(b.distSq));

        return Math.Min(_visibleDoodadsScratch.Count, (int)DoodadMaxRenderCount);
    }

    public void UpdateDoodadAnimations()
    {
        _updatedDoodadModelsScratch.Clear();
        foreach (DoodadInstance inst in _doodadInstances)
        {
            if (inst.Renderer != null && _updatedDoodadModelsScratch.Add(inst.ModelPath))
                inst.Renderer.UpdateAnimation();
        }
    }

    public unsafe void RenderOpaqueDoodads(
        int visibleDoodadRenderCount,
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor,
        ref int doodadSubmissions,
        SceneLightManager? sceneLights = null)
    {
        _opaqueDoodadBatchGroups.Clear();
        _opaqueDoodadBatchRenderers.Clear();

        for (int vi = 0; vi < visibleDoodadRenderCount; vi++)
        {
            int doodadIndex = _visibleDoodadsScratch[vi].idx;
            DoodadInstance inst = _doodadInstances[doodadIndex];
            IModelRenderer renderer = inst.Renderer!;
            if (renderer.RequiresUnbatchedWorldRender)
            {
                doodadSubmissions++;
                renderer.RenderWithTransform(inst.Transform * modelMatrix, view, proj, RenderPass.Opaque, 1.0f,
                    fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor, sceneLights);
                continue;
            }

            if (!_opaqueDoodadBatchGroups.TryGetValue(renderer, out List<int>? indices))
            {
                indices = new List<int>();
                _opaqueDoodadBatchGroups.Add(renderer, indices);
                _opaqueDoodadBatchRenderers.Add(renderer);
            }

            indices.Add(doodadIndex);
        }

        foreach (IModelRenderer renderer in _opaqueDoodadBatchRenderers)
        {
            List<int> indices = _opaqueDoodadBatchGroups[renderer];
            if (renderer is IGpuInstancedModelRenderer gpuRenderer
                && gpuRenderer.SupportsGpuInstancedOpaque)
            {
                gpuRenderer.BeginGpuInstanceBatch(view, proj, fogColor, fogStart, fogEnd,
                    cameraPos, lightDir, lightColor, ambientColor);
                foreach (int doodadIndex in indices)
                {
                    DoodadInstance inst = _doodadInstances[doodadIndex];
                    gpuRenderer.QueueGpuInstance(inst.Transform * modelMatrix, 1.0f);
                    doodadSubmissions++;
                }

                gpuRenderer.EndGpuInstanceBatch();
            }
            else
            {
                renderer.BeginBatch(view, proj, fogColor, fogStart, fogEnd,
                    cameraPos, lightDir, lightColor, ambientColor, sceneLights);
                foreach (int doodadIndex in indices)
                {
                    DoodadInstance inst = _doodadInstances[doodadIndex];
                    renderer.RenderInstance(inst.Transform * modelMatrix, RenderPass.Opaque, 1.0f);
                    doodadSubmissions++;
                }
            }
        }
    }

    public void CollectOpaqueDoodadsForPlacement(
        Matrix4x4 modelMatrix,
        Vector3 cameraPos,
        float fogEnd,
        Action<WmoOpaqueDoodadBatchItem> collect,
        ref int doodadSubmissions)
    {
        ArgumentNullException.ThrowIfNull(collect);
        if (!_supportsGpuInstancedOpaque || !_doodadsVisible || !_runtimeDoodadsVisible || _doodadInstances.Count == 0)
            return;

        int visibleDoodadRenderCount = PrepareVisibleDoodads(
            modelMatrix,
            cameraPos,
            fogEnd,
            updateAnimation: false,
            sortByDistance: false,
            respectRuntimeDoodadVisibility: false);

        for (int vi = 0; vi < visibleDoodadRenderCount; vi++)
        {
            DoodadInstance inst = _doodadInstances[_visibleDoodadsScratch[vi].idx];
            if (inst.Renderer is not IModelRenderer renderer)
                continue;

            doodadSubmissions++;
            collect(new WmoOpaqueDoodadBatchItem(renderer, inst.Transform * modelMatrix));
        }
    }

    public unsafe void RenderOpaqueDoodadsForPlacement(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor,
        ref int doodadSubmissions)
    {
        if (!_supportsGpuInstancedOpaque || !_doodadsVisible || !_runtimeDoodadsVisible || _doodadInstances.Count == 0)
            return;

        int visibleDoodadRenderCount = PrepareVisibleDoodads(
            modelMatrix,
            cameraPos,
            fogEnd,
            updateAnimation: false,
            sortByDistance: false,
            respectRuntimeDoodadVisibility: false);

        RenderOpaqueDoodads(visibleDoodadRenderCount, modelMatrix, view, proj,
            fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor, ref doodadSubmissions);
    }

    public void RenderTransparentDoodads(
        int visibleDoodadRenderCount,
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor,
        ref int doodadSubmissions,
        SceneLightManager? sceneLights = null)
    {
        if (visibleDoodadRenderCount <= 0)
            return;

        for (int vi = visibleDoodadRenderCount - 1; vi >= 0; vi--)
        {
            var inst = _doodadInstances[_visibleDoodadsScratch[vi].idx];
            if (!inst.Renderer!.HasTransparentWorldPass)
                continue;

            var doodadWorld = inst.Transform * modelMatrix;
            doodadSubmissions++;
            inst.Renderer.RenderWithTransform(doodadWorld, view, proj, RenderPass.Transparent, 1.0f,
                fogColor, fogStart, fogEnd, cameraPos,
                lightDir, lightColor, ambientColor, sceneLights);
        }
    }

    public void CollectSceneLights(Matrix4x4 modelMatrix, ICollection<SceneLight> lights, string sourceKey)
    {
        if (!_doodadsVisible || !_runtimeDoodadsVisible || _doodadInstances.Count == 0)
            return;

        for (int i = 0; i < _doodadInstances.Count; i++)
        {
            DoodadInstance doodad = _doodadInstances[i];
            if (!doodad.Visible || doodad.Renderer is not ISceneLightEmitter emitter)
                continue;

            Matrix4x4 doodadWorld = doodad.Transform * modelMatrix;
            string doodadSourceKey = string.IsNullOrWhiteSpace(doodad.NormalizedModelPath)
                ? sourceKey
                : doodad.NormalizedModelPath;
            emitter.CollectSceneLights(doodadWorld, lights, doodadSourceKey);
        }
    }

    public void ApplyTextureSamplingSettings()
    {
        foreach (var renderer in _doodadModelCache.Values)
            renderer?.ApplyTextureSamplingSettings();
    }

    public void Dispose()
    {
        foreach (var pair in _doodadModelCache)
        {
            if (DoodadModelShare != null)
                DoodadModelShare.Release(pair.Key);
            else
                pair.Value?.Dispose();
        }

        _doodadModelCache.Clear();
        _doodadInstances.Clear();
        _loggedMissingDoodadSkinPaths.Clear();
        _pendingDoodadModelLoads.Clear();
        _queuedDoodadModelLoads.Clear();
        _doodadInstanceIndicesByModel.Clear();
        _doodadSourceModelPaths.Clear();
    }
}

internal sealed class DoodadInstance
{
    public string ModelPath = "";
    public string NormalizedModelPath = "";
    public IModelRenderer? Renderer;
    public Matrix4x4 Transform;
    public bool Visible = true;
    public int DoodadDefIndex;
    public Vector3 LocalPosition; // WMO-local position for fast culling
}
