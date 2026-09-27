using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Population;
using WoWViewer.Rendering;
using WoWViewer.Audio;
using WowViewer.Core.Audio;
using WowViewer.Core.Maps;
using Silk.NET.OpenGL;
using CorePm4AxisConvention = WowViewer.Core.PM4.Models.Pm4AxisConvention;
using CorePm4CorrelationCandidateScore = WowViewer.Core.PM4.Models.Pm4CorrelationCandidateScore;
using CorePm4CorrelationMetrics = WowViewer.Core.PM4.Models.Pm4CorrelationMetrics;
using CorePm4CorrelationObjectDescriptor = WowViewer.Core.PM4.Models.Pm4CorrelationObjectDescriptor;
using CorePm4CorrelationGeometryInput = WowViewer.Core.PM4.Models.Pm4CorrelationGeometryInput;
using CorePm4CorrelationObjectInput = WowViewer.Core.PM4.Models.Pm4CorrelationObjectInput;
using CorePm4CorrelationObjectState = WowViewer.Core.PM4.Models.Pm4CorrelationObjectState;
using CorePm4CorrelationMath = WowViewer.Core.PM4.Services.Pm4CorrelationMath;
using CorePm4ConnectorKey = WowViewer.Core.PM4.Models.Pm4ConnectorKey;
using CorePm4ConnectorMergeCandidate = WowViewer.Core.PM4.Models.Pm4ConnectorMergeCandidate;
using CorePm4CoordinateMode = WowViewer.Core.PM4.Models.Pm4CoordinateMode;
using CorePm4GeometryLineSegment = WowViewer.Core.PM4.Models.Pm4GeometryLineSegment;
using CorePm4GeometryTriangle = WowViewer.Core.PM4.Models.Pm4GeometryTriangle;
using CorePm4LinkedPositionRefSummary = WowViewer.Core.PM4.Models.Pm4LinkedPositionRefSummary;
using CorePm4MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using CorePm4MshdGroupingService = WowViewer.Core.PM4.Services.Pm4MshdGroupingService;
using CorePm4MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using CorePm4MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using CorePm4CoordinateModeResolution = WowViewer.Core.PM4.Models.Pm4CoordinateModeResolution;
using CorePm4ObjectGroupKey = WowViewer.Core.PM4.Models.Pm4ObjectGroupKey;
using CorePm4CachedTile = WowViewer.Core.PM4.Caching.Pm4CachedTile;
using CorePm4CachedObject = WowViewer.Core.PM4.Caching.Pm4CachedObject;
using CorePm4CachedConnectorKey = WowViewer.Core.PM4.Caching.Pm4CachedConnectorKey;
using CorePm4CachedLineSegment = WowViewer.Core.PM4.Caching.Pm4CachedLineSegment;
using CorePm4CachedTriangle = WowViewer.Core.PM4.Caching.Pm4CachedTriangle;
using CorePm4PerFileCacheEntry = WowViewer.Core.PM4.Caching.Pm4PerFileCacheEntry;
using CorePm4PerFileCache = WowViewer.Core.PM4.Caching.Pm4PerFileCache;
using CorePm4PerFileCacheService = WowViewer.Core.PM4.Caching.Pm4PerFileCacheService;
using CorePm4PlacementContract = WowViewer.Core.PM4.Services.Pm4PlacementContract;
using CorePm4PlacementMath = WowViewer.Core.PM4.Services.Pm4PlacementMath;
using CorePm4PlacementSolution = WowViewer.Core.PM4.Models.Pm4PlacementSolution;
using Pm4PlanarTransform = WowViewer.Core.PM4.Models.Pm4PlanarTransform;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using CorePm4DecodeAuditReport = WowViewer.Core.PM4.Models.Pm4DecodeAuditReport;
using CorePm4ExplorationSnapshot = WowViewer.Core.PM4.Models.Pm4ExplorationSnapshot;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using CorePm4ObjectHypothesis = WowViewer.Core.PM4.Models.Pm4ObjectHypothesis;
using MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using Pm4VersionFormatter = WowViewer.Core.PM4.Services.Pm4VersionFormatter;
using MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using Pm4File = WowViewer.Core.PM4.Research.Pm4ResearchDocument;
using CorePm4ReferenceAudit = WowViewer.Core.PM4.Models.Pm4ReferenceAudit;
using CorePm4ResearchAuditAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchAuditAnalyzer;
using CorePm4ResearchHierarchyAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchHierarchyAnalyzer;
using CorePm4ResearchSnapshotBuilder = WowViewer.Core.PM4.Research.Pm4ResearchSnapshotBuilder;
using CorePm4TileObjectHypothesisReport = WowViewer.Core.PM4.Models.Pm4TileObjectHypothesisReport;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WorldFramePassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldFramePassCoordinator;
using WorldFramePassOptions = WowViewer.Core.Runtime.World.Passes.WorldFramePassOptions;
using WorldFramePasses = WowViewer.Core.Runtime.World.Passes.WorldFramePasses;
using WorldObjectPassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldObjectPassCoordinator;
using WorldObjectPassFrame = WowViewer.Core.Runtime.World.Passes.WorldObjectPassFrame;
using WorldModelBatchGate = WowViewer.Core.Runtime.World.Passes.WorldModelBatchGate;
using WorldModelRenderPath = WowViewer.Core.Runtime.World.Passes.WorldModelRenderPath;
using WorldModelSubmissionOutcome = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionOutcome;
using WorldModelSubmissionTally = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionTally;
using VisibleMdxInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleMdxEntry;
using VisibleWmoInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleWmoEntry;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.World;
using static WoWViewer.Terrain.Pm4OverlayScene;
using static WoWViewer.Terrain.Pm4OverlayMatching;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;
using WowViewer.Core.Runtime.World.Selection;

namespace WoWViewer.Terrain;

/// <summary>
/// World-scene atmosphere: Light.dbc/LIT lighting and fallback, fog-range override and restore, and the light-driven skybox backdrop with its stars fallback.
/// Moved verbatim from <see cref="WorldScene"/> (Spec 255). Scene state it still needs comes
/// only through <see cref="IWorldSceneHost"/>; the bridge members below keep the names the moved
/// code used inside WorldScene, so no moved body was edited.
/// </summary>
public sealed class SceneAtmosphere
{
    private readonly IWorldSceneHost _host;

    internal SceneAtmosphere(IWorldSceneHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the WorldScene members).
    private WorldAssetManager _assets => _host.Assets;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private SkyDomeRenderer _skyDome => _host.SkyDome;
    private ref List<ObjectInstance> _skyboxInstances => ref _host.SkyboxInstances;
    private TerrainManager _terrainManager => _host.TerrainManager;
    private IModelRenderer? TryGetQueuedMdx(string modelKey) => _host.TryGetQueuedMdx(modelKey);

    private bool _clientStarsProbeComplete;
    private string? _clientStarsFallbackModelPath;
    private string? _activeLightSkyboxSourcePath;
    private string? _activeLightSkyboxModelKey;

    // DBC Lighting
    internal LightService? _lightService;
    public LightService? LightService => _lightService;

    // Alpha LIT lighting (lazy-loaded on first request)
    internal LitLoader? _litLoader;
    internal bool _showLitLights;
    private bool _showLitMinimapMarkers;
    private bool _litLoadAttempted;
    internal bool _useLitFogOverride;
    private bool _litAutoFallback;
    private string _litAutoFallbackReason = string.Empty;
    // Local Light* spatial selection is retained for diagnostics, but its
    // renderer application remains opt-in until the native local-zone
    // transform/falloff contract is proven for the active build.
    internal bool _useLocalDbcLightingOverlay;
    private bool _hasGlobalViewerFogRange;
    private float _globalViewerFogStart;
    private float _globalViewerFogEnd;
    private bool _hasPreLitFogRange;
    private float _preLitFogStart;
    private float _preLitFogEnd;
    private bool _hasUserFogRangeOverride;
    private float _userFogStart = TerrainLightingMath.DefaultFogStart;
    private float _userFogEnd = TerrainLightingMath.DefaultFogEnd;
    private float _activeFogStart = TerrainLightingMath.DefaultFogStart;
    private float _activeFogEnd = TerrainLightingMath.DefaultFogEnd;
    private string _activeFogRangeSource = "Fallback";
    private bool _activeFogRangeAdjusted;
    private string _litStatus = "LIT not loaded.";
    internal int _selectedLitLightIndex = -1;
    private string? _selectedLitSourcePath;
    internal LitLoader.LitLightingSample? _lastLitSample;
    public bool ShowLitLights
    {
        get => _showLitLights;
        set
        {
            _showLitLights = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    /// <summary>Shows loaded positional LIT entries on shared minimap surfaces without changing lighting.</summary>
    public bool ShowLitMinimapMarkers
    {
        get => _showLitMinimapMarkers;
        set
        {
            _showLitMinimapMarkers = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    public bool UseLitFogOverride
    {
        get => _useLitFogOverride;
        set
        {
            if (_useLitFogOverride == value)
                return;
            if (!value)
                RestorePreLitFogRange(_terrainManager?.Lighting);
            _useLitFogOverride = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    public bool UseLocalDbcLightingOverlay
    {
        get => _useLocalDbcLightingOverlay;
        set => _useLocalDbcLightingOverlay = value;
    }

    public LitLoader? LitLoader => _litLoader;
    public bool LitLoadAttempted => _litLoadAttempted;
    public string LitStatus => _litStatus;
    public bool LitAutoFallbackActive => _litAutoFallback;
    public string LitAutoFallbackReason => _litAutoFallbackReason;
    public int SelectedLitLightIndex { get => _selectedLitLightIndex; set => _selectedLitLightIndex = value; }
    public string? SelectedLitSourcePath => _selectedLitSourcePath ?? _litLoader?.SourcePath;
    public IReadOnlyList<string> AvailableLitSourcePaths => _litLoader?.AvailableSourcePaths ?? Array.Empty<string>();
    public LitLoader.LitLightingSample? LastLitSample => _lastLitSample;

    /// <summary>User-selected fog range that is intentionally independent from lighting recommendations.</summary>
    public bool HasUserFogRangeOverride => _hasUserFogRangeOverride;

    public float UserFogStart => _userFogStart;

    public float UserFogEnd => _userFogEnd;

    public float ActiveFogStart => _activeFogStart;

    public float ActiveFogEnd => _activeFogEnd;

    public string ActiveFogRangeSource => _activeFogRangeSource;

    public bool ActiveFogRangeAdjusted => _activeFogRangeAdjusted;

    public void SetUserFogRangeOverride(float fogStart, float fogEnd)
    {
        (_userFogStart, _userFogEnd) = TerrainLightingMath.NormalizeFogRange(fogStart, fogEnd);
        _hasUserFogRangeOverride = true;
    }

    public void ClearUserFogRangeOverride()
    {
        _hasUserFogRangeOverride = false;
    }

    internal void CapturePreLitFogRange(TerrainLighting lighting)
    {
        if (_hasPreLitFogRange)
            return;

        _preLitFogStart = lighting.FogStart;
        _preLitFogEnd = lighting.FogEnd;
        _hasPreLitFogRange = true;
    }

    internal void RestoreGlobalViewerFogRange(TerrainLighting lighting)
    {
        if (!_hasGlobalViewerFogRange)
        {
            _globalViewerFogStart = lighting.FogStart;
            _globalViewerFogEnd = lighting.FogEnd;
            _hasGlobalViewerFogRange = true;
        }

        lighting.FogStart = _globalViewerFogStart;
        lighting.FogEnd = _globalViewerFogEnd;
    }

    private void RestorePreLitFogRange(TerrainLighting? lighting)
    {
        if (!_hasPreLitFogRange)
            return;
        if (lighting != null)
        {
            lighting.FogStart = _preLitFogStart;
            lighting.FogEnd = _preLitFogEnd;
        }
        _hasPreLitFogRange = false;
    }

    internal void ResolveActiveFogRange(TerrainLighting lighting, string recommendationSource)
    {
        float rawRecommendedStart = lighting.FogStart;
        float rawRecommendedEnd = lighting.FogEnd;
        float fallbackStart = _hasPreLitFogRange ? _preLitFogStart : TerrainLightingMath.DefaultFogStart;
        float fallbackEnd = _hasPreLitFogRange ? _preLitFogEnd : TerrainLightingMath.DefaultFogEnd;
        (float recommendedStart, float recommendedEnd) = TerrainLightingMath.NormalizeFogRange(
            rawRecommendedStart,
            rawRecommendedEnd,
            fallbackStart,
            fallbackEnd);

        (float activeStart, float activeEnd) = _hasUserFogRangeOverride
            ? TerrainLightingMath.NormalizeFogRange(_userFogStart, _userFogEnd, recommendedStart, recommendedEnd)
            : (recommendedStart, recommendedEnd);

        _activeFogRangeAdjusted = !FogRangesEqual(rawRecommendedStart, rawRecommendedEnd, recommendedStart, recommendedEnd)
            || (_hasUserFogRangeOverride && !FogRangesEqual(_userFogStart, _userFogEnd, activeStart, activeEnd));
        _activeFogRangeSource = _hasUserFogRangeOverride ? "User override" : recommendationSource;
        _activeFogStart = activeStart;
        _activeFogEnd = activeEnd;
        lighting.FogStart = activeStart;
        lighting.FogEnd = activeEnd;
    }

    private static bool FogRangesEqual(float leftStart, float leftEnd, float rightStart, float rightEnd)
        => MathF.Abs(leftStart - rightStart) < 0.001f && MathF.Abs(leftEnd - rightEnd) < 0.001f;

    private void LazyLoadLit()
    {
        _litLoadAttempted = true;
        _lastLitSample = null;

        if (_dataSource == null)
        {
            _litStatus = "LIT unavailable: no data source.";
            return;
        }

        _litLoader = new LitLoader(_dataSource, _terrainManager.MapName, _selectedLitSourcePath);
        if (_litLoader.Load())
        {
            _litStatus = _litLoader.Status;
            _selectedLitSourcePath = _litLoader.SourcePath;
            if (_selectedLitLightIndex < 0 && _litLoader.Lights.Count > 0)
                _selectedLitLightIndex = 0;
            return;
        }

        _litStatus = _litLoader.Status;
    }

    public void ReloadLit(string? sourcePath = null)
    {
        _selectedLitSourcePath = string.IsNullOrWhiteSpace(sourcePath) ? null : sourcePath;
        _selectedLitLightIndex = -1;
        _litLoader = null;
        _litLoadAttempted = false;
        _litStatus = "LIT reload queued.";
        LazyLoadLit();
    }

    /// <summary>
    /// Activates the map's LIT lighting source when no usable map-scoped Light DBC profile exists.
    /// This is an automatic default only; the user can still turn the override off in the UI.
    /// </summary>
    public void EnableLitFallback(string reason)
    {
        _litAutoFallback = true;
        _litAutoFallbackReason = string.IsNullOrWhiteSpace(reason)
            ? "No usable map-scoped Light DBC profile is available."
            : reason.Trim();

        if (!_useLitFogOverride)
            UseLitFogOverride = true;
        else if (!_litLoadAttempted)
            LazyLoadLit();
    }

    /// <summary>
    /// Load the exact-build Light* DBC chain for zone-based lighting, with the flattened
    /// LightData table retained only as a later-build compatibility fallback.
    /// </summary>
    public void LoadLighting(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _lightService = new LightService();
        _lightService.Load(dbcProvider, dbdDir, build, mapId);
        if (!_lightService.HasUsableLightingForMap)
        {
            EnableLitFallback(
                $"No usable Light DBC profile exists for map {mapId}; LIT is enabled automatically.");
        }
    }

    internal void RenderSkyboxBackdrop(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        bool renderedActiveClientSky = false;
        if (_skyDome.NightVisibility > 0.001f
            && TryGetQueuedMdx(_activeLightSkyboxModelKey ?? string.Empty) is { } lightSkyboxRenderer)
        {
            lightSkyboxRenderer.UpdateAnimation();
            lightSkyboxRenderer.RenderBackdrop(Matrix4x4.CreateTranslation(cameraPos), view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            renderedActiveClientSky = true;
        }

        if (_skyboxInstances.Count == 0)
            return;

        ObjectInstance? nearestSkybox = null;
        float nearestDistSq = float.MaxValue;
        foreach (var inst in _skyboxInstances)
        {
            float distSq = Vector3.DistanceSquared(cameraPos, inst.PlacementPosition);
            if (distSq >= nearestDistSq)
                continue;

            nearestDistSq = distSq;
            nearestSkybox = inst;
        }

        if (!nearestSkybox.HasValue)
            return;

        var skybox = nearestSkybox.Value;
        if (renderedActiveClientSky
            && string.Equals(skybox.ModelKey, _activeLightSkyboxModelKey, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        var renderer = TryGetQueuedMdx(skybox.ModelKey);
        if (renderer == null)
            return;

        renderer.UpdateAnimation();
        renderer.RenderBackdrop(CreateSkyboxBackdropTransform(skybox.Transform, cameraPos), view, proj,
            fogColor, fogStart, fogEnd, cameraPos,
            lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
    }

    private static Matrix4x4 CreateSkyboxBackdropTransform(Matrix4x4 placementTransform, Vector3 cameraPos)
    {
        placementTransform.M41 = cameraPos.X;
        placementTransform.M42 = cameraPos.Y;
        placementTransform.M43 = cameraPos.Z;
        return placementTransform;
    }

    internal static bool IsSkyboxModelPath(string modelPath)
    {
        return WorldSkyboxBackdropClassifier.IsBackdropModelPath(modelPath);
    }

    internal void UpdateActiveSkyboxModel()
    {
        string? sourcePath = _lightService?.ActiveSkyboxModelPath;
        sourcePath = ResolveClientSkyboxPath(sourcePath);
        if (string.IsNullOrWhiteSpace(sourcePath))
            sourcePath = ResolveClientStarsFallback();

        if (string.Equals(sourcePath, _activeLightSkyboxSourcePath, StringComparison.OrdinalIgnoreCase))
            return;

        _activeLightSkyboxSourcePath = sourcePath;
        _activeLightSkyboxModelKey = string.IsNullOrWhiteSpace(sourcePath)
            ? null
            : WorldAssetManager.NormalizeKey(sourcePath);

        if (string.IsNullOrWhiteSpace(_activeLightSkyboxModelKey))
            return;

        _assets.PrioritizeMdxLoad(_activeLightSkyboxModelKey);
        ViewerLog.Info(
            ViewerLog.Category.Mdx,
            $"[Sky] Active client sky model: {sourcePath} (source={(_lightService?.ActiveSkyboxModelPath is null ? "client-stars fallback" : "LightSkybox DBC")})");
    }

    private string? ResolveClientSkyboxPath(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath) || _dataSource == null)
            return null;

        if (_dataSource.FileExists(sourcePath))
            return sourcePath;

        if (!string.IsNullOrWhiteSpace(Path.GetExtension(sourcePath)))
            return null;

        foreach (string extension in new[] { ".m2", ".mdx", ".mdl" })
        {
            string candidate = sourcePath + extension;
            if (_dataSource.FileExists(candidate))
                return candidate;
        }

        return null;
    }

    private string? ResolveClientStarsFallback()
    {
        if (_clientStarsProbeComplete)
            return _clientStarsFallbackModelPath;

        _clientStarsProbeComplete = true;
        if (_dataSource == null)
            return null;

        string[] candidates =
        [
            @"Environments\Stars\Stars.m2",
            @"Environments\Stars\Stars.mdx",
            @"Environments\Stars\Stars.mdl",
        ];
        foreach (string path in candidates)
        {
            if (!_dataSource.FileExists(path))
                continue;

            _clientStarsFallbackModelPath = path;
            ViewerLog.Info(ViewerLog.Category.Mdx, $"[Sky] Discovered client stars fallback: {path}");
            return path;
        }

        // Some extracted clients retain a World prefix around the same asset.
        foreach (string path in new[]
        {
            @"World\Environments\Stars\Stars.m2",
            @"World\Environments\Stars\Stars.mdx",
            @"World\Environments\Stars\Stars.mdl",
        })
        {
            if (_dataSource.FileExists(path))
            {
                _clientStarsFallbackModelPath = path;
                ViewerLog.Info(ViewerLog.Category.Mdx, $"[Sky] Discovered client stars fallback: {path}");
                return path;
            }
        }

        ViewerLog.Debug(ViewerLog.Category.Mdx, "[Sky] Client stars fallback was not present in the data source file list.");
        return null;
    }
}
