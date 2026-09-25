using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;

namespace WoWViewer;

/// <summary>
/// Main viewer application. Owns window, GL context, ImGui, camera, renderer.
/// Provides menu bar, file browser, model info panel, and 3D viewport.
/// </summary>
public partial class ViewerApp : IDisposable, Workbench.Pages.IEditorPageHost, IViewerAppHost
{
    internal enum Pm4WorkbenchTab
    {
        Overlay,
        Selection,
        Correlation,
    }

    internal enum WorkspaceMode
    {
        Viewer,
        Editor,
        Archaeology,
    }

    [Obsolete("Shell panel system deprecated in 069. Use tab system (View > Tab System). Will be removed in 070.")]
    internal enum ShellPanelId
    {
        Navigator,
        Inspector,
        Pm4Workbench,
        TerrainControls,
        RuntimeStats,
        WorldObjects,
        ModelInfo,
        Minimap,
        WorkspaceBars,
        Pm4Info,
        Pm4SceneGraph,
    }



    internal enum ShellPanelLane
    {
        Left,
        Right,
        Floating,
    }

    internal enum EditorWorkspaceTask
    {
        Terrain,
        Objects,
        Pm4Evidence,
        Inspect,
        Publish,
    }

    internal enum FixedBottomDrawerTab
    {
        Workspace,
        Terrain,
        Pm4,
        World,
        Diagnostics,
    }

    internal readonly record struct ShellPanelDefinition(
        ShellPanelId Id,
        string WindowName,
        ShellPanelLane Lane,
        float DefaultWidth,
        float MinWidth,
        float CompactMinWidth,
        float MaxWidth);

    internal sealed class SavedShellPanelLayout
    {
        public int PanelId { get; set; }
        public float NormalizedX { get; set; }
        public float NormalizedY { get; set; }
        public float NormalizedWidth { get; set; }
        public float NormalizedHeight { get; set; }
    }

    private const string ViewerProductTitle = "WoWViewer";
    internal static readonly string ViewerDisplayVersion = GetViewerDisplayVersion();
    internal static string ViewerProductName => $"{ViewerProductTitle} v{ViewerDisplayVersion}";

    private IWindow _window = null!;
    private GL _gl = null!;
    private IInputContext _input = null!;
    private ImGuiController _imGui = null!;
    private Camera _camera = new();
    private ISceneRenderer? _renderer;

    // Data source
    private IDataSource? _dataSource;
    private ReplaceableTextureResolver? _texResolver;
    private DBCD.Providers.IDBCProvider? _dbcProvider;
    private string? _dbdDir;
    private string? _dbcBuild;
    internal const float MaxTerrainFogDistance = 20000f;

    private readonly List<WoWViewer.Terrain.ClientBuildOption> _clientBuildOptions = new();
    private string? _lastVirtualPath; // Virtual path of last loaded file (for DBC lookup)
    private string _statusMessage = "No data source loaded. Use File > Open Game Folder (MPQ) first, then Open File for standalone assets.";
    private AreaTableService? _areaTableService;
    private string _currentAreaName = "";
    private WowViewer.Core.World.AreaLookupResult? _currentAreaLookup;
    private int _currentMapId = -1; // MapID of the currently loaded world
    private string? _lastWorldSceneWdtPath;
    private Vector3 _lastWorldSceneCameraPosition;
    private float _lastWorldSceneCameraYaw = 180f;
    private float _lastWorldSceneCameraPitch = -20f;
    private readonly Dictionary<string, Dictionary<int, string>> _savedTaxiActorModelOverridesByMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, SavedObjectPathFilterMap> _savedObjectPathFiltersByMap = new(StringComparer.OrdinalIgnoreCase);

    // Map discovery
    private List<MapDefinition> _discoveredMaps = new();

    /// <summary>
    /// Ordering for every map list. Defaults to Map ID because every row is labelled
    /// <c>[id] Name</c> and the id is how the DBC, phase relationships and external tooling
    /// refer to maps; name ordering left that leading number in no order at all.
    /// </summary>
    private MapListSortMode _mapListSortMode = MapListSortMode.MapId;
    private Md5TranslateIndex? _md5Index;
    private MinimapRenderer? _minimapRenderer;
    private WdlPreviewRenderer? _wdlPreviewRenderer;
    private WdlPreviewCacheService? _wdlPreviewCacheService;
    private bool _showWdlPreview = false;
    private MapDefinition? _selectedMapForPreview;
    private Vector2? _selectedSpawnTile; // WDL tile coordinates (0-63)
    private Vector3? _pendingWorldSpawnOverride;
    private float _minimapZoom = 4f; // Number of tiles visible in each direction from camera
    private bool _fullscreenMinimap = false; // M key toggles fullscreen minimap
    private Vector2 _minimapPanOffset = Vector2.Zero; // Pan offset for click-and-drag
    private bool _minimapDragging = false;
    private (int tileX, int tileY)? _pendingMinimapTeleportTile;
    private int _pendingMinimapTeleportClickCount;
    private DateTime _pendingMinimapTeleportLastClickUtc = DateTime.MinValue;
    private Rendering.LoadingScreen? _loadingScreen;

    // Output directories (next to the executable)
    internal static readonly string OutputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");
    internal static readonly string CacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache");
    internal static readonly string ExportDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "export");
    internal static readonly string ProjectsDir = Path.Combine(OutputDir, "projects");
    internal static readonly string SettingsDir = Path.Combine(OutputDir, "settings");
    internal const int MinimapTeleportConfirmClicks = 3;

    // File browser state
    private List<string> _filteredFiles = new();
    private string _searchFilter = "";
    private string _extensionFilter = ".mdx";
    private int _selectedFileIndex = -1;
    private string? _loadedFilePath;
    private string? _loadedFileName;

    // Model info
    private string _modelInfo = "";
    private readonly Dictionary<string, string?> _standaloneSkinPathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _loggedStandaloneMissingSkinPaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _reportedAreaDiagnostics = new(StringComparer.Ordinal);
    
    // Stored loaded model data for export (avoids re-parsing from disk)
    private WmoV14ToV17Converter.WmoV14Data? _loadedWmo;

    internal static string GetViewerDisplayVersion()
    {
        var informational = typeof(ViewerApp).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion;
        if (!string.IsNullOrWhiteSpace(informational))
        {
            // The .NET SDK appends "+<git-commit>" build metadata to
            // InformationalVersion; display only the bare version number.
            int metadataStart = informational.IndexOf('+');
            return metadataStart >= 0 ? informational[..metadataStart] : informational;
        }

        return typeof(ViewerApp).Assembly.GetName().Version?.ToString(3) ?? "unknown";
    }
    private MdxFile? _loadedMdx;
    private M2StaticRenderModel? _loadedM2Runtime;

    // Mouse state
    private float _lastMouseX, _lastMouseY;
    private bool _mouseDown;
    private float _pendingSceneMouseWheelDelta;

    // UI state
    private bool _showFileBrowser = true;
    private bool _showModelInfo = true;
    private bool _showWorkspaceBarsPanel = true;
    private bool _hideUiChrome;
    private bool _showLogViewer = false;
    private bool _showMinimapWindow = false;
    private bool _showPerfWindow = false;
    private WorkspaceMode _workspaceMode = WorkspaceMode.Viewer;
    private EditorWorkspaceTask _editorWorkspaceTask = EditorWorkspaceTask.Terrain;
    private FixedBottomDrawerTab _activeBottomDrawerTab = FixedBottomDrawerTab.Workspace;
    private FixedBottomDrawerTab? _pendingRightSidebarSection;
    private bool _useDockspaceUi = true;

    // 069 Phase 1: tab system state. On by default; can toggle off via View > Legacy Sidebar UI.
    private bool _useTabUi = true;
    private WorkbenchTab _activeTopTab = WorkbenchTab.Quick;
    private int _activeBottomTabIndex = 0;
    private int _activeUtilitiesTabIndex = 0;
    private int _activePm4TabIndex = 0;

    // Workbench popout (069 Phase 14: single resizable panel, no window sprawl)
    private bool _workbenchOpen = true;

    // 069 Phase 6: sticky archeology settings (persist across viewer restarts).
    private int _archeologyMinUniqueId = -1; // -1 = unset (use first detected value)
    private int _archeologyMaxUniqueId = -1; // -1 = unset
    private int _archeologyScopeIndex = 0;   // 0 = PerMap, 1 = CameraTile

    // 069 Phase 7: archeology playback (animate Visible Range End over time).
    private bool _archeologyPlaybackActive = false;
    private float _archeologyPlaybackSpeed = 50f; // uniqueIds per second
    private bool _archeologyPlaybackLoop = false;

    // 069 Phase 7: capture integration flags
    private bool _archeologyApplyToNextCapture = false;
    private bool _archeologyApplyToVideoRecording = false;
    private bool _autoOpenWorldMapsPanel;
    private Vector2 _dockspaceHostPosition;
    private Vector2 _dockspaceHostSize;
    private AssetCatalogView? _catalogView;
    private bool _wantOpenFile = false;
    private bool _wantExportGlb = false;
    private bool _wantExportGlbCollision = false;
    private bool _wantExportMapGlbTiles = false;
    internal string _projectOutputRootDir = ProjectsDir;
    private string _editorProjectOutputDir = string.Empty;
    private string? _selectedPlacementSaveTargetPath;
    private string _selectedPlacementSaveStatus = "Select a tile-backed world object to stage a translation-only save.";

    internal static readonly ShellPanelDefinition[] ShellPanelDefinitions =
    {
        new(ShellPanelId.Navigator, "Navigator", ShellPanelLane.Left, DefaultSidebarWidth, SidebarMinWidth, SidebarCompactMinWidth, SidebarMaxWidth),
        new(ShellPanelId.Inspector, "Selection", ShellPanelLane.Right, DefaultSidebarWidth, SidebarMinWidth, SidebarCompactMinWidth, SidebarMaxWidth),
        new(ShellPanelId.Pm4Workbench, "PM4 Workbench", ShellPanelLane.Right, 420f, 300f, 220f, SidebarMaxWidth),
        new(ShellPanelId.TerrainControls, "Terrain Controls", ShellPanelLane.Right, DefaultSidebarWidth, SidebarMinWidth, SidebarCompactMinWidth, SidebarMaxWidth),
        new(ShellPanelId.RuntimeStats, "Runtime Stats", ShellPanelLane.Right, DefaultSidebarWidth, SidebarMinWidth, SidebarCompactMinWidth, SidebarMaxWidth),
        new(ShellPanelId.WorldObjects, "World Objects", ShellPanelLane.Right, 420f, 300f, 220f, SidebarMaxWidth),
        new(ShellPanelId.ModelInfo, "Model Info", ShellPanelLane.Right, DefaultSidebarWidth, SidebarMinWidth, SidebarCompactMinWidth, SidebarMaxWidth),
        new(ShellPanelId.Minimap, "Minimap", ShellPanelLane.Floating, 360f, 300f, 260f, 520f),
        new(ShellPanelId.WorkspaceBars, "Workspace Bars", ShellPanelLane.Left, 360f, 280f, 240f, 520f),
        new(ShellPanelId.Pm4Info, "PM4 Info", ShellPanelLane.Right, 400f, 280f, 200f, SidebarMaxWidth),
        new(ShellPanelId.Pm4SceneGraph, "PM4 Scene Graph", ShellPanelLane.Right, 420f, 300f, 220f, SidebarMaxWidth),
    };
    private readonly Dictionary<ShellPanelId, SavedShellPanelLayout> _savedShellPanelLayouts = new();
    private readonly HashSet<ShellPanelId> _pendingShellPanelLayoutRestore = new();
    private bool _forceApplyShellPanelLayout;

    internal enum TerrainTileScope
    {
        CurrentTile = 0,
        LoadedTiles = 1,
        WholeMap = 2,
        CustomList = 3,
        RectRange = 4,
    }

    internal enum TerrainExportKind
    {
        None = 0,
        AlphaCurrentTileAtlas = 1,
        AlphaCurrentTileChunksFolder = 2,
        AlphaLoadedTilesFolder = 3,
        AlphaWholeMapFolder = 4,
        Heightmap257CurrentTilePerTile = 10,
        Heightmap257LoadedTilesFolderPerTile = 11,
        Heightmap257WholeMapFolderPerMap = 12,
        MccvCurrentTilePng = 20,
        MccvLoadedTilesFolder = 21,
        MccvWholeMapFolder = 22,
    }

    internal enum TerrainImportKind
    {
        None = 0,
        AlphaFolder = 1,
        Heightmap257Folder = 10,
        MccvFolder = 20,
    }

    private bool _wantTerrainExport;
    private TerrainExportKind _terrainExportKind = TerrainExportKind.None;
    private bool _wantTerrainImport;
    private TerrainImportKind _terrainImportKind = TerrainImportKind.None;
    private bool _showAlphaFolderImportScope;
    private bool _showHeightmapFolderImportScope;
    private bool _showMccvFolderImportScope;
    private TerrainTileScope _terrainTileScope = TerrainTileScope.LoadedTiles;
    private TerrainTileScope _mapGlbScope = TerrainTileScope.CurrentTile;
    private int _terrainTileRangeStartX;
    private int _terrainTileRangeStartY;
    private int _terrainTileRangeEndX = 63;
    private int _terrainTileRangeEndY = 63;

    private bool _chunkToolEnabled;
    private ChunkClipboard? _chunkClipboard;
    private string _chunkClipboardStatus = "";
    private (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardCopiedKey;
    private (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardLockedTargetKey;
    private bool _chunkClipboardCtrlCWasPressed;
    private bool _chunkClipboardCtrlVWasPressed;
    private readonly HashSet<(int tileX, int tileY, int chunkX, int chunkY)> _selectedChunks = new();
    private ChunkClipboardSet? _chunkClipboardSet;
    private bool _chunkClipboardShowOverlay = true;
    private TerrainAnalysisPreviewTexture? _terrainAnalysisLocalTexture;
    private TerrainAnalysisPreviewTexture? _terrainAnalysisGlobalTexture;
    private TerrainAnalysisPreviewTexture? _terrainAnalysisAlphaTexture;
    internal const float TerrainWeakSignalRestoreDefaultMinZ = -10f;
    internal const float TerrainWeakSignalRestoreDefaultMaxZ = 10f;
    internal const float TerrainWeakSignalRestoreMaxFactor = 512f;
    private bool _terrainWeakSignalRestoreEnabled;
    private bool _terrainWeakSignalRestoreAllLoadedTiles = true;
    private bool _terrainWeakSignalRestoreUseTextureSubdivisions = true;
    private bool _terrainWeakSignalRestoreUseAutoFactor = true;
    private float _terrainWeakSignalRestoreManualFactor = 16f;
    internal float _terrainWeakSignalRestoreCandidateMinHeight = TerrainWeakSignalRestoreDefaultMinZ;
    internal float _terrainWeakSignalRestoreCandidateMaxHeight = TerrainWeakSignalRestoreDefaultMaxZ;
    private string _terrainWeakSignalRestoreStatus = string.Empty;
    private readonly Dictionary<(int tileX, int tileY), WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyTileAnalysis> _stratigraphyTileAnalyses = new();
    private bool _stratigraphyUnhideDevMeshes = true;
    private bool _stratigraphyPreserveNegativeFloor = true;
    private bool _stratigraphyPolarityInverted = false;
    private WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode _stratigraphyAnchorMode = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode.LowestZ_Floor;
    private bool _stratigraphyUseNeighborAutoFit = false;
    private bool _stratigraphyUseWdlMagnetization = false;
    private float _stratigraphyWdlMagnetizationStrength = 1.0f;
    private (int tileX, int tileY)? _terrainAnalysisPreviewCompareTile;
    private float? _terrainAnalysisPreviewSimilarity;
    private readonly List<TerrainHiddenTileCandidate> _terrainAnalysisHiddenCandidates = new();
    private int _terrainAnalysisHiddenSelectedIndex = -1;
    private string _terrainAnalysisHiddenStatus = string.Empty;
    private Terrain.BoundingBoxRenderer? _editorOverlayBb;
    private bool _standaloneWmoGroupOverlayEnabled = true;
    private bool _standaloneWmoGroupLabelsAllEnabled = true;
    private bool _standaloneWmoOverlayIncludeHiddenGroups = true;
    private int _hoveredStandaloneWmoGroupIndex = -1;
    private int _selectedStandaloneWmoGroupIndex = -1;
    private readonly HashSet<int> _highlightedStandaloneWmoGroupIndices = new();

    private sealed class TerrainShadowStudyResult
    {
        public int SampleCount { get; init; }
        public int ShadowedSampleCount { get; init; }
        public float ShadowedAverageHeight { get; init; }
        public float LitAverageHeight { get; init; }
        public float BestAgreement { get; init; }
        public float BestPrecision { get; init; }
        public float BestRecall { get; init; }
        public float BestAzimuthDegrees { get; init; }
        public float BestSlopePerWorldUnit { get; init; }
        public int BestRaySteps { get; init; }
    }

    internal sealed class ChunkClipboard
    {
        public float[] Heights { get; }
        public Vector3[] Normals { get; }
        public int HoleMask { get; }
        public WoWViewer.Terrain.TerrainLayer[] Layers { get; }
        public Dictionary<int, byte[]> AlphaMaps { get; }
        public byte[]? ShadowMap { get; }
        public byte[]? MccvColors { get; }

        public ChunkClipboard(WoWViewer.Terrain.TerrainChunkData chunk)
        {
            Heights = (float[])chunk.Heights.Clone();
            Normals = (Vector3[])chunk.Normals.Clone();
            HoleMask = chunk.HoleMask;
            Layers = chunk.Layers.ToArray();
            AlphaMaps = TerrainChunkMath.CloneAlphaMaps(chunk.AlphaMaps);
            ShadowMap = chunk.ShadowMap != null ? (byte[])chunk.ShadowMap.Clone() : null;
            MccvColors = chunk.MccvColors != null ? (byte[])chunk.MccvColors.Clone() : null;
        }
    }

    internal sealed class ChunkClipboardSet
    {
        public int OriginGlobalChunkX { get; }
        public int OriginGlobalChunkY { get; }
        public Dictionary<(int dx, int dy), ChunkClipboard> Chunks { get; } = new();

        public ChunkClipboardSet(int originGlobalChunkX, int originGlobalChunkY)
        {
            OriginGlobalChunkX = originGlobalChunkX;
            OriginGlobalChunkY = originGlobalChunkY;
        }
    }

    // Sidebar layout
    private bool _showLeftSidebar = true;
    private bool _showRightSidebar = true;
    internal const float DefaultSidebarWidth = 360f;
    internal const float DefaultRightSidebarWidth = 480f;
    internal const float SidebarMinWidth = 280f;
    internal const float SidebarCompactMinWidth = 240f;
    internal const float SidebarMaxWidth = 1080f;
    internal const float DefaultBottomDrawerHeight = 280f;
    private const float BottomDrawerSplitterHeight = 8f;
    internal const float SceneViewportHardMinWidth = 240f;
    internal float _leftSidebarWidth = DefaultSidebarWidth;
    internal float _rightSidebarWidth = DefaultRightSidebarWidth;
    internal float _bottomDrawerHeight = DefaultBottomDrawerHeight;
    internal const float MenuBarHeight = 22f;
    internal const float ToolbarHeight = 32f;
    internal const float BottomBarHeight = 36f;
    internal const float StatusBarHeight = 24f;

    private float GetActiveToolbarHeight()
    {
        return _hideUiChrome ? 0f : ToolbarHeight;
    }

    private float GetTopChromeHeight()
    {
        return MenuBarHeight + GetActiveToolbarHeight();
    }

    private bool IsBottomDrawerVisible()
    {
        return false;
    }

    /// <summary>When true, load all tiles at startup instead of AOI streaming. Default: false (stream tiles as camera moves).</summary>
    public bool FullLoadMode { get; set; } = false;

    // Terrain/World state
    private TerrainManager? _terrainManager;
    private VlmTerrainManager? _vlmTerrainManager;
    private WorldScene? _worldScene;
    private SceneCursorRenderer? _sceneCursorRenderer;
    private SceneClusterSelector3D? _sceneClusterSelector3D;
    private CameraHudRig? _cameraHudRig;
    private float _uiFontScale = 1.0f;

    // Object picking state
    private int _selectedObjectIndex = -1; // -1=none, 0..modf-1=WMO, modf..modf+mddf-1=MDX
    private string _selectedObjectType = "";
    private string _selectedObjectInfo = "";
        private int _selectedAreaPoiId = -1;
    private string _taxiActorModelOverrideInput = "";
    private int _taxiActorModelOverrideInputRouteId = -1;
    private int _taxiActorModelOverrideTargetRouteId = -1;
    private bool _layoutObjectPreviewMode;
    private bool _layoutObjectPreviewStateCaptured;
    private bool _layoutObjectPreviewSavedObjectsVisible = true;
    private bool _layoutObjectPreviewSavedWmosVisible = true;
    private bool _layoutObjectPreviewSavedDoodadsVisible = true;
    private WorldObjectVisibilityProfile _layoutObjectPreviewSavedVisibilityProfile = WorldObjectVisibilityProfile.Performance;
    private SqlWorldPopulationService? _sqlPopulationService;
    private bool _sqlForceStreamRefresh;
    private string _wlLayerSelectedBodyKey = "";
    private bool _wlLayerListIsolationEnabled;
    private Vector3 _pm4SavedOverlayTranslation = Vector3.Zero;
    private Vector3 _pm4SavedOverlayRotationDegrees = Vector3.Zero;
    private Vector3 _pm4SavedOverlayScale = Vector3.One;
    private ShellPanelId? _pendingFocusedShellPanel;
    private Pm4ObjectMatchObject? _hoveredPm4ObjectMatch;
    private (int tileX, int tileY, uint ck24, int objectPart)? _hoveredPm4ObjectMatchKey;
    private int _hoveredPm4ObjectMatchCacheMaxMatches = -1;
    private int _pm4ObjectMatchMaxMatchesPerObject = 5;
    private readonly Dictionary<string, SavedPm4ObjectMatchSelection> _savedPm4ObjectMatches = new(StringComparer.OrdinalIgnoreCase);
    private Pm4WmoMatchStore? _pm4WmoMatchStore;
    private Dictionary<string, Pm4WmoMatchEntry> _pm4WmoMatchEntries = new(StringComparer.OrdinalIgnoreCase);
    private bool _showCaptureAutomationWindow = false;
    private bool _showCameraPathWindow;
    private bool _showUniqueIdArchaeologyWindow;
    private bool _showWeakSignalWindow;

    // Camera speed (adjustable via UI)
    private float _cameraSpeed = 50f;
    // Field of view in degrees (adjustable via UI)
    private float _fovDegrees = 45f;
    private int _savedDetailedAdtTileCountOverride;

    private bool _autoFrameModelOnLoad = true;
    private bool _hasExplicitWmoMliqRotationOverride;

    // Sky gradient for standalone model viewing
    private uint _skyVao, _skyVbo, _skyShader;
    private bool _skyReady;

    // Folder dialog workaround (ImGui doesn't have native dialogs)
    private bool _showFolderInput = false;
    private string _folderInputBuf = "";
    private bool _showBuildSelectionDialog;
    private int _selectedBuildOptionIndex;
    private bool _showListfileInput = false;
    private bool _showRosettaDatastoreDialog = false;
    private string _lastGameFolderPath = "";
    private string _lastLooseOverlayPath = "";
    private List<KnownGoodClientPath> _knownGoodClientPaths = new();
    private string? _pendingKnownGoodClientPath;
    private string? _pendingKnownGoodClientBuildVersion;
    private bool _pendingKnownGoodClientAttachLooseFolder;
    private bool _openForgetKnownGoodClientConfirm;
    private string? _pendingForgetKnownGoodClientPath;
    private string? _pendingForgetKnownGoodClientDisplayName;

    // FPS counter
    private int _frameCount;
    private double _fpsTimer;
    private double _currentFps;
    private double _frameTimeMs;

    // Map Converter state
    private bool _showMapConverterDialog = false;
    private string _mapConvertSourcePath = "";
    private string _mapConvertOutputDir = "";
    private string _mapConvertProjectSourceKey = string.Empty;
    private string _mapConvertLkMapDir = ""; // Split ADT → Alpha: directory containing split ADT files

    // WMO Converter state
    private bool _showWmoConverterDialog = false;
    private string _wmoConvertSourcePath = "";

    // Terrain-derived minimap export state. This is intentionally separate from the retired
    // VLM/MK dataset workflow: it invokes the direct client terrain synthesis command.
    private bool _showSynthesizedMinimapExportDialog;

    // ML Dataset build state
    private bool _showVlmExportDialog = false;
    private string _vlmClientPath = "";
    private string _vlmMapName = "development";
    private string _vlmOutputDir = "";

    // ML Dataset manifest and validation state
    private string _mkHarvestDatasetRoot = "";
    private int _mkHarvestViewerValidationQueued = 0;
    private int _mkHarvestViewerValidationCompleted = 0;
    private int _mkHarvestViewerValidationFailed = 0;

    // Terrain texture transfer state
    private bool _showTerrainTextureTransferDialog = false;
    private string _terrainTransferSourceDir = Pm4CoordinateService.DefaultDevelopmentMapDirectory;
    private string _terrainTransferTargetDir = Pm4CoordinateService.DefaultDevelopmentMapDirectory;
    private string _terrainTransferOutputDir = Path.Combine("output", "terrain-texture-transfer-ui");

    // U-01 composition root (AGENTS.md §10): features extracted from ViewerApp live in owned
    // services; each gets this instance only as IViewerAppHost.
    private readonly ConverterDialogsService _converterDialogs;
    private readonly DatasetExportDialogsService _datasetExportDialogs;
    private readonly TerrainWeakSignalRestoreService _terrainWeakSignalRestore;
    private readonly TerrainTileIoService _terrainTileIo;
    private readonly ChunkEditService _chunkEdit;
    private readonly PlacementEditService _placementEditing;
    private readonly SqlSpawnStreamingService _sqlSpawnStreaming;
    private readonly TaxiAndAreaPoiSelectionService _taxiAndAreaPoi;
    private readonly SceneHoverAndPickService _sceneHoverPick;
    private readonly ShellLayoutService _shellLayout;
    private readonly WdlPreviewService _wdlPreview;
    private readonly AreaContextService _areaContext;
    private readonly StandaloneModelLoaderService _modelLoader;
    private readonly WorldLoaderService _worldLoader;
    private readonly DataSourceSessionService _dataSourceSession;
    private readonly TerrainQueryService _terrainQuery;
    private readonly StratigraphyService _stratigraphy;
    private readonly ProjectOutputService _projectOutput;
    private readonly WorldObjectsPanelService _worldObjectsPanel;
    private readonly ViewerSettingsService _settings;
    private readonly CascAhdrSourceService _cascAhdrSource;
    private readonly ClientDialogsService _clientDialogs;
    private readonly MainMenuBarService _mainMenuBar;
    private readonly Pm4WorkbenchService _pm4Workbench;
    private readonly TaxiPanelService _taxiPanel;
    private readonly ArchaeologyPanelService _archaeologyPanel;
    private readonly ModelInspectorPanelService _modelInspector;
    private readonly TerrainControlsPanelService _terrainControlsPanel;
    private readonly NavigatorPanelService _navigatorPanel;
    private readonly ViewerChromeService _viewerChrome;
    private readonly LightingPanelService _lightingPanel;
    private readonly AudioPanelService _audioPanel;
    private readonly ThemesService _themes;
    private readonly TerrainInspectionPanelService _terrainInspection;
    private readonly InvestigationService _investigation;
    private readonly InspectorPayloadsService _inspectorPayloads;
    private readonly WorkbenchPanelsService _workbenchPanels;
    private readonly LogViewerService _logViewer;
    private readonly RenderQualityService _renderQuality;
    private readonly DatasetCatalogService _datasetCatalog;
    private readonly WmoGroupsPanelService _wmoGroupsPanel;
    private readonly SettingsWindowService _settingsWindow;
    private readonly SynthesizedMinimapExportService _synthesizedMinimapExport;
    private readonly MlTrainingService _mlTraining;
    private readonly TerrainAnalysisService _terrainAnalysis;

    public ViewerApp()
    {
        _converterDialogs = new ConverterDialogsService(this);
        _datasetExportDialogs = new DatasetExportDialogsService(this);
        _terrainWeakSignalRestore = new TerrainWeakSignalRestoreService(this);
        _terrainTileIo = new TerrainTileIoService(this);
        _chunkEdit = new ChunkEditService(this);
        _placementEditing = new PlacementEditService(this);
        _sqlSpawnStreaming = new SqlSpawnStreamingService(this);
        _taxiAndAreaPoi = new TaxiAndAreaPoiSelectionService(this);
        _sceneHoverPick = new SceneHoverAndPickService(this);
        _shellLayout = new ShellLayoutService(this);
        _wdlPreview = new WdlPreviewService(this);
        _areaContext = new AreaContextService(this);
        _modelLoader = new StandaloneModelLoaderService(this);
        _worldLoader = new WorldLoaderService(this);
        _dataSourceSession = new DataSourceSessionService(this);
        _terrainQuery = new TerrainQueryService(this);
        _stratigraphy = new StratigraphyService(this);
        _projectOutput = new ProjectOutputService(this);
        _worldObjectsPanel = new WorldObjectsPanelService(this);
        _settings = new ViewerSettingsService(this);
        _cascAhdrSource = new CascAhdrSourceService(this);
        _clientDialogs = new ClientDialogsService(this);
        _mainMenuBar = new MainMenuBarService(this);
        _pm4Workbench = new Pm4WorkbenchService(this);
        _taxiPanel = new TaxiPanelService(this);
        _archaeologyPanel = new ArchaeologyPanelService(this);
        _modelInspector = new ModelInspectorPanelService(this);
        _terrainControlsPanel = new TerrainControlsPanelService(this);
        _navigatorPanel = new NavigatorPanelService(this);
        _viewerChrome = new ViewerChromeService(this);
        _lightingPanel = new LightingPanelService(this);
        _audioPanel = new AudioPanelService(this);
        _themes = new ThemesService(this);
        _terrainInspection = new TerrainInspectionPanelService(this);
        _investigation = new InvestigationService(this);
        _inspectorPayloads = new InspectorPayloadsService(this);
        _workbenchPanels = new WorkbenchPanelsService(this);
        _logViewer = new LogViewerService(this);
        _renderQuality = new RenderQualityService(this);
        _datasetCatalog = new DatasetCatalogService(this);
        _wmoGroupsPanel = new WmoGroupsPanelService(this);
        _settingsWindow = new SettingsWindowService(this);
        _synthesizedMinimapExport = new SynthesizedMinimapExportService(this);
        _mlTraining = new MlTrainingService(this);
        _terrainAnalysis = new TerrainAnalysisService(this);
    }

    // IViewerAppHost: the ViewerApp state and behaviour the extracted services may use.
    ref IDataSource? IViewerAppHost.DataSource => ref _dataSource;
    ref string? IViewerAppHost.LoadedFilePath => ref _loadedFilePath;
    ref string IViewerAppHost.MapConvertLkMapDir => ref _mapConvertLkMapDir;
    ref string IViewerAppHost.MapConvertOutputDir => ref _mapConvertOutputDir;
    ref string IViewerAppHost.MapConvertProjectSourceKey => ref _mapConvertProjectSourceKey;
    ref string IViewerAppHost.MapConvertSourcePath => ref _mapConvertSourcePath;
    ref string IViewerAppHost.ProjectOutputRootDir => ref _projectOutputRootDir;
    ref bool IViewerAppHost.ShowMapConverterDialog => ref _showMapConverterDialog;
    ref bool IViewerAppHost.ShowWmoConverterDialog => ref _showWmoConverterDialog;
    ref string IViewerAppHost.WmoConvertSourcePath => ref _wmoConvertSourcePath;
    string IViewerAppHost.GetProjectOutputRootDirectory() => _projectOutput.GetProjectOutputRootDirectory();
    void IViewerAppHost.HandleProjectOutputRootChanged() => _projectOutput.HandleProjectOutputRootChanged();
    void IViewerAppHost.LoadWdtTerrain(string wdtPath) => _worldLoader.LoadWdtTerrain(wdtPath);
    ref string IViewerAppHost.MkHarvestDatasetRoot => ref _mkHarvestDatasetRoot;
    ref int IViewerAppHost.MkHarvestViewerValidationCompleted => ref _mkHarvestViewerValidationCompleted;
    ref int IViewerAppHost.MkHarvestViewerValidationFailed => ref _mkHarvestViewerValidationFailed;
    ref int IViewerAppHost.MkHarvestViewerValidationQueued => ref _mkHarvestViewerValidationQueued;
    ref MkHarvestViewerValidationCapturePlan? IViewerAppHost.PendingMkHarvestViewerValidationCapturePlan => ref _pendingMkHarvestViewerValidationCapturePlan;
    ref bool IViewerAppHost.ShowTerrainTextureTransferDialog => ref _showTerrainTextureTransferDialog;
    ref bool IViewerAppHost.ShowVlmExportDialog => ref _showVlmExportDialog;
    ref string IViewerAppHost.TerrainTransferOutputDir => ref _terrainTransferOutputDir;
    ref string IViewerAppHost.TerrainTransferSourceDir => ref _terrainTransferSourceDir;
    ref string IViewerAppHost.TerrainTransferTargetDir => ref _terrainTransferTargetDir;
    ref string IViewerAppHost.VlmClientPath => ref _vlmClientPath;
    ref string IViewerAppHost.VlmMapName => ref _vlmMapName;
    ref string IViewerAppHost.VlmOutputDir => ref _vlmOutputDir;
    ref VlmTerrainManager? IViewerAppHost.VlmTerrainManager => ref _vlmTerrainManager;
    void IViewerAppHost.LoadVlmProject(string projectRoot) => _worldLoader.LoadVlmProject(projectRoot);
    void IViewerAppHost.StitchMkHarvestViewerValidationOutputs(string mapName, string outputDirectory, string noLiquidsOutputDirectory, string noObjectsOutputDirectory, string objectsOnlyOutputDirectory, int requestedResolution) => StitchMkHarvestViewerValidationOutputs(mapName, outputDirectory, noLiquidsOutputDirectory, noObjectsOutputDirectory, objectsOnlyOutputDirectory, requestedResolution);
    HashSet<(int tileX, int tileY, int chunkX, int chunkY)> IViewerAppHost.SelectedChunks => _selectedChunks;
    ref WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode IViewerAppHost.StratigraphyAnchorMode => ref _stratigraphyAnchorMode;
    ref bool IViewerAppHost.StratigraphyPolarityInverted => ref _stratigraphyPolarityInverted;
    ref bool IViewerAppHost.StratigraphyPreserveNegativeFloor => ref _stratigraphyPreserveNegativeFloor;
    ref bool IViewerAppHost.StratigraphyUnhideDevMeshes => ref _stratigraphyUnhideDevMeshes;
    ref bool IViewerAppHost.StratigraphyUseNeighborAutoFit => ref _stratigraphyUseNeighborAutoFit;
    ref bool IViewerAppHost.StratigraphyUseWdlMagnetization => ref _stratigraphyUseWdlMagnetization;
    ref float IViewerAppHost.StratigraphyWdlMagnetizationStrength => ref _stratigraphyWdlMagnetizationStrength;
    ref TerrainManager? IViewerAppHost.TerrainManager => ref _terrainManager;
    ref TerrainTileScope IViewerAppHost.TerrainTileScope => ref _terrainTileScope;
    ref bool IViewerAppHost.TerrainWeakSignalRestoreAllLoadedTiles => ref _terrainWeakSignalRestoreAllLoadedTiles;
    ref float IViewerAppHost.TerrainWeakSignalRestoreCandidateMaxHeight => ref _terrainWeakSignalRestoreCandidateMaxHeight;
    ref float IViewerAppHost.TerrainWeakSignalRestoreCandidateMinHeight => ref _terrainWeakSignalRestoreCandidateMinHeight;
    ref bool IViewerAppHost.TerrainWeakSignalRestoreEnabled => ref _terrainWeakSignalRestoreEnabled;
    ref float IViewerAppHost.TerrainWeakSignalRestoreManualFactor => ref _terrainWeakSignalRestoreManualFactor;
    ref string IViewerAppHost.TerrainWeakSignalRestoreStatus => ref _terrainWeakSignalRestoreStatus;
    ref bool IViewerAppHost.TerrainWeakSignalRestoreUseAutoFactor => ref _terrainWeakSignalRestoreUseAutoFactor;
    ref bool IViewerAppHost.TerrainWeakSignalRestoreUseTextureSubdivisions => ref _terrainWeakSignalRestoreUseTextureSubdivisions;
    ref WdlPreviewCacheService? IViewerAppHost.WdlPreviewCacheService => ref _wdlPreviewCacheService;
    (int tileX, int tileY) IViewerAppHost.GetCameraTile() => GetCameraTile();
    string? IViewerAppHost.GetCurrentSessionMapName() => _dataSourceSession.GetCurrentSessionMapName();
    IReadOnlyList<(int tileX, int tileY)> IViewerAppHost.GetTileScopeList(TerrainTileScope scope) => _terrainTileIo.GetTileScopeList(scope);
    ref TerrainTileScope IViewerAppHost.MapGlbScope => ref _mapGlbScope;
    ref Md5TranslateIndex? IViewerAppHost.Md5Index => ref _md5Index;
    ref bool IViewerAppHost.ShowAlphaFolderImportScope => ref _showAlphaFolderImportScope;
    ref bool IViewerAppHost.ShowHeightmapFolderImportScope => ref _showHeightmapFolderImportScope;
    ref bool IViewerAppHost.ShowMccvFolderImportScope => ref _showMccvFolderImportScope;
    ref string IViewerAppHost.StatusMessage => ref _statusMessage;
    ref TerrainExportKind IViewerAppHost.TerrainExportKind => ref _terrainExportKind;
    ref TerrainImportKind IViewerAppHost.TerrainImportKind => ref _terrainImportKind;
    ref int IViewerAppHost.TerrainTileRangeEndX => ref _terrainTileRangeEndX;
    ref int IViewerAppHost.TerrainTileRangeEndY => ref _terrainTileRangeEndY;
    ref int IViewerAppHost.TerrainTileRangeStartX => ref _terrainTileRangeStartX;
    ref int IViewerAppHost.TerrainTileRangeStartY => ref _terrainTileRangeStartY;
    TerrainWeakSignalRestoreService IViewerAppHost.TerrainWeakSignalRestore => _terrainWeakSignalRestore;
    ref Camera IViewerAppHost.Camera => ref _camera;
    ref ChunkClipboard? IViewerAppHost.ChunkClipboard => ref _chunkClipboard;
    ref (int tileX, int tileY, int chunkX, int chunkY)? IViewerAppHost.ChunkClipboardCopiedKey => ref _chunkClipboardCopiedKey;
    ref (int tileX, int tileY, int chunkX, int chunkY)? IViewerAppHost.ChunkClipboardLockedTargetKey => ref _chunkClipboardLockedTargetKey;
    ref ChunkClipboardSet? IViewerAppHost.ChunkClipboardSet => ref _chunkClipboardSet;
    ref bool IViewerAppHost.ChunkClipboardShowOverlay => ref _chunkClipboardShowOverlay;
    ref string IViewerAppHost.ChunkClipboardStatus => ref _chunkClipboardStatus;
    ref bool IViewerAppHost.ChunkToolEnabled => ref _chunkToolEnabled;
    TerrainTileIoService IViewerAppHost.TerrainTileIo => _terrainTileIo;
    string IViewerAppHost.EnsureEditorProjectOutputDirectory(bool forceNew) => _projectOutput.EnsureEditorProjectOutputDirectory(forceNew);
    string IViewerAppHost.GetEditorProjectName(string? fallbackName) => _projectOutput.GetEditorProjectName(fallbackName);
    string? IViewerAppHost.GetEditorProjectSourceKey() => _projectOutput.GetEditorProjectSourceKey();
    bool IViewerAppHost.TryPickTerrainChunkUnderMouse(TerrainRenderer renderer, out TerrainRenderer.TerrainChunkInfo info) => _terrainQuery.TryPickTerrainChunkUnderMouse(renderer, out info);
    ref string IViewerAppHost.EditorProjectOutputDir => ref _editorProjectOutputDir;
    ref string IViewerAppHost.SelectedPlacementSaveStatus => ref _selectedPlacementSaveStatus;
    ref string? IViewerAppHost.SelectedPlacementSaveTargetPath => ref _selectedPlacementSaveTargetPath;
    ref WorldScene? IViewerAppHost.WorldScene => ref _worldScene;
    void IViewerAppHost.RefreshSelectedWorldObjectInfo() => RefreshSelectedWorldObjectInfo();
    ref int IViewerAppHost.CurrentMapId => ref _currentMapId;
    ref bool IViewerAppHost.SqlForceStreamRefresh => ref _sqlForceStreamRefresh;
    ref SqlWorldPopulationService? IViewerAppHost.SqlPopulationService => ref _sqlPopulationService;
    void IViewerAppHost.DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent) => _viewerChrome.DrawToolbarPopupButton(label, summary, popupId, drawContent);
    void IViewerAppHost.ExportAnimationStateJson(IAnimationController animator, int currentSeq, string currentSeqName, float seqStart, float seqEnd) => _modelInspector.ExportAnimationStateJson(animator, currentSeq, currentSeqName, seqStart, seqEnd);
    ref string? IViewerAppHost.LastVirtualPath => ref _lastVirtualPath;
    Dictionary<string, Dictionary<int, string>> IViewerAppHost.SavedTaxiActorModelOverridesByMap => _savedTaxiActorModelOverridesByMap;
    ref int IViewerAppHost.SelectedAreaPoiId => ref _selectedAreaPoiId;
    ref int IViewerAppHost.SelectedObjectIndex => ref _selectedObjectIndex;
    ref string IViewerAppHost.SelectedObjectInfo => ref _selectedObjectInfo;
    ref string IViewerAppHost.SelectedObjectType => ref _selectedObjectType;
    ref string IViewerAppHost.TaxiActorModelOverrideInput => ref _taxiActorModelOverrideInput;
    ref int IViewerAppHost.TaxiActorModelOverrideInputRouteId => ref _taxiActorModelOverrideInputRouteId;
    ref int IViewerAppHost.TaxiActorModelOverrideTargetRouteId => ref _taxiActorModelOverrideTargetRouteId;
    void IViewerAppHost.SaveViewerSettings() => _settings.SaveViewerSettings();
    bool IViewerAppHost.TryGetSelectedBrowserModelPath(out string assetPath) => _navigatorPanel.TryGetSelectedBrowserModelPath(out assetPath);
    ref EditorWorkspaceTask IViewerAppHost.EditorWorkspaceTask => ref _editorWorkspaceTask;
    ref float IViewerAppHost.FovDegrees => ref _fovDegrees;
    ref GL IViewerAppHost.Gl => ref _gl;
    ref Pm4ObjectMatchObject? IViewerAppHost.HoveredPm4ObjectMatch => ref _hoveredPm4ObjectMatch;
    ref int IViewerAppHost.HoveredPm4ObjectMatchCacheMaxMatches => ref _hoveredPm4ObjectMatchCacheMaxMatches;
    ref (int tileX, int tileY, uint ck24, int objectPart)? IViewerAppHost.HoveredPm4ObjectMatchKey => ref _hoveredPm4ObjectMatchKey;
    ref float IViewerAppHost.LastMouseX => ref _lastMouseX;
    ref float IViewerAppHost.LastMouseY => ref _lastMouseY;
    ref int IViewerAppHost.Pm4ObjectMatchMaxMatchesPerObject => ref _pm4ObjectMatchMaxMatchesPerObject;
    ref SceneClusterSelector3D? IViewerAppHost.SceneClusterSelector3D => ref _sceneClusterSelector3D;
    ref SceneCursorRenderer? IViewerAppHost.SceneCursorRenderer => ref _sceneCursorRenderer;
    TaxiAndAreaPoiSelectionService IViewerAppHost.TaxiAndAreaPoi => _taxiAndAreaPoi;
    ref InvestigationService.VisualInvestigationMode IViewerAppHost.VisualInvestigationMode => ref _investigation._visualInvestigationMode;
    ref WorkspaceMode IViewerAppHost.WorkspaceMode => ref _workspaceMode;
    bool IViewerAppHost.CanSceneConsumeMouse(float x, float y) => _shellLayout.CanSceneConsumeMouse(x, y);
    void IViewerAppHost.ClearSelectedWlLiquidBody(bool clearListIsolation) => _investigation.ClearSelectedWlLiquidBody(clearListIsolation);
    float IViewerAppHost.GetSceneFarPlane() => _terrainQuery.GetSceneFarPlane();
    bool IViewerAppHost.IsSceneMouseCaptureBlocked(float x, float y) => _shellLayout.IsSceneMouseCaptureBlocked(x, y);
    void IViewerAppHost.SelectTerrainChunkFromClick(TerrainRenderer.TerrainChunkInfo info) => _terrainInspection.SelectTerrainChunkFromClick(info);
    void IViewerAppHost.SetSelectedWlLiquidBody(WlLiquidBody body, bool isolateInList, bool focusInspectWorkspace, string? statusMessage) => _investigation.SetSelectedWlLiquidBody(body, isolateInList, focusInspectWorkspace, statusMessage);
    bool IViewerAppHost.ShouldShowHoveredAssetInfoForInvestigation(HoveredAssetInfo info) => _investigation.ShouldShowHoveredAssetInfoForInvestigation(info);
    bool IViewerAppHost.TogglePm4ObjectCollectionMembership((int tileX, int tileY, uint ck24, int objectPart) key, bool reportStatus, bool removeIfPresent) => _pm4Workbench.TogglePm4ObjectCollectionMembership(key, reportStatus, removeIfPresent);
    bool IViewerAppHost.TryFindWlLiquidBodyByKey(string bodyKey, out WlLiquidBody? body) => _investigation.TryFindWlLiquidBodyByKey(bodyKey, out body);
    bool IViewerAppHost.TryGetSceneViewportRect(out float x, out float y, out float width, out float height) => _shellLayout.TryGetSceneViewportRect(out x, out y, out width, out height);
    bool IViewerAppHost.TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info) => _terrainQuery.TryRaycastTerrain(renderer, rayOrigin, rayDir, maxDistance, out info);
    bool IViewerAppHost.TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info, out Vector3 hitPoint) => _terrainQuery.TryRaycastTerrain(renderer, rayOrigin, rayDir, maxDistance, out info, out hitPoint);
    bool IViewerAppHost.TryResolveHoveredWlLiquidBody(HoveredAssetInfo hoveredInfo, out WlLiquidBody? body) => _investigation.TryResolveHoveredWlLiquidBody(hoveredInfo, out body);
    ref FixedBottomDrawerTab IViewerAppHost.ActiveBottomDrawerTab => ref _activeBottomDrawerTab;
    ref float IViewerAppHost.BottomDrawerHeight => ref _bottomDrawerHeight;
    ref Vector2 IViewerAppHost.DockspaceHostPosition => ref _dockspaceHostPosition;
    ref Vector2 IViewerAppHost.DockspaceHostSize => ref _dockspaceHostSize;
    ref bool IViewerAppHost.ForceApplyShellPanelLayout => ref _forceApplyShellPanelLayout;
    ref bool IViewerAppHost.FullscreenMinimap => ref _fullscreenMinimap;
    ref bool IViewerAppHost.HideUiChrome => ref _hideUiChrome;
    ref ImGuiController IViewerAppHost.ImGui => ref _imGui;
    ref float IViewerAppHost.LeftSidebarWidth => ref _leftSidebarWidth;
    ref string IViewerAppHost.ModelInfo => ref _modelInfo;
    ref ShellPanelId? IViewerAppHost.PendingFocusedShellPanel => ref _pendingFocusedShellPanel;
    ref FixedBottomDrawerTab? IViewerAppHost.PendingRightSidebarSection => ref _pendingRightSidebarSection;
    HashSet<ShellPanelId> IViewerAppHost.PendingShellPanelLayoutRestore => _pendingShellPanelLayoutRestore;
    ref float IViewerAppHost.RightSidebarWidth => ref _rightSidebarWidth;
    Dictionary<ShellPanelId, SavedShellPanelLayout> IViewerAppHost.SavedShellPanelLayouts => _savedShellPanelLayouts;
    ref bool IViewerAppHost.ShowLeftSidebar => ref _showLeftSidebar;
    ref bool IViewerAppHost.ShowMinimapWindow => ref _showMinimapWindow;
    ref bool IViewerAppHost.ShowModelInfo => ref _showModelInfo;
    ref bool IViewerAppHost.ShowRightSidebar => ref _showRightSidebar;
    ref bool IViewerAppHost.ShowWorkspaceBarsPanel => ref _showWorkspaceBarsPanel;
    ref bool IViewerAppHost.UseDockspaceUi => ref _useDockspaceUi;
    ref bool IViewerAppHost.UseTabUi => ref _useTabUi;
    ref IWindow IViewerAppHost.Window => ref _window;
    float IViewerAppHost.ClampFixedSidebarWidth(float width, bool isLeftSidebar, float displayWidth) => _viewerChrome.ClampFixedSidebarWidth(width, isLeftSidebar, displayWidth);
    float IViewerAppHost.GetTopChromeHeight() => GetTopChromeHeight();
    void IViewerAppHost.SetEditorWorkspaceTask(EditorWorkspaceTask task) => SetEditorWorkspaceTask(task);
    ref List<MapDefinition> IViewerAppHost.DiscoveredMaps => ref _discoveredMaps;
    ref Vector3? IViewerAppHost.PendingWorldSpawnOverride => ref _pendingWorldSpawnOverride;
    ref MapDefinition? IViewerAppHost.SelectedMapForPreview => ref _selectedMapForPreview;
    ref Vector2? IViewerAppHost.SelectedSpawnTile => ref _selectedSpawnTile;
    ref bool IViewerAppHost.ShowWdlPreview => ref _showWdlPreview;
    ref WdlPreviewRenderer? IViewerAppHost.WdlPreviewRenderer => ref _wdlPreviewRenderer;
    void IViewerAppHost.LoadFileFromDataSource(string virtualPath) => _modelLoader.LoadFileFromDataSource(virtualPath);
    void IViewerAppHost.LoadMapAtDefaultSpawn(MapDefinition map) => _worldLoader.LoadMapAtDefaultSpawn(map);
    string? IViewerAppHost.ResolveMapWdtPath(string mapDirectory) => _worldLoader.ResolveMapWdtPath(mapDirectory);
    ref AreaTableService? IViewerAppHost.AreaTableService => ref _areaTableService;
    ref WowViewer.Core.World.AreaLookupResult? IViewerAppHost.CurrentAreaLookup => ref _currentAreaLookup;
    ref string IViewerAppHost.CurrentAreaName => ref _currentAreaName;
    ref ISceneRenderer? IViewerAppHost.Renderer => ref _renderer;
    HashSet<string> IViewerAppHost.ReportedAreaDiagnostics => _reportedAreaDiagnostics;
    ref bool IViewerAppHost.AutoFrameModelOnLoad => ref _autoFrameModelOnLoad;
    ref string? IViewerAppHost.DbcBuild => ref _dbcBuild;
    ref float IViewerAppHost.LastWorldSceneCameraPitch => ref _lastWorldSceneCameraPitch;
    ref Vector3 IViewerAppHost.LastWorldSceneCameraPosition => ref _lastWorldSceneCameraPosition;
    ref float IViewerAppHost.LastWorldSceneCameraYaw => ref _lastWorldSceneCameraYaw;
    ref string? IViewerAppHost.LastWorldSceneWdtPath => ref _lastWorldSceneWdtPath;
    ref string? IViewerAppHost.LoadedFileName => ref _loadedFileName;
    ref M2StaticRenderModel? IViewerAppHost.LoadedM2Runtime => ref _loadedM2Runtime;
    ref MdxFile? IViewerAppHost.LoadedMdx => ref _loadedMdx;
    ref WmoV14ToV17Converter.WmoV14Data? IViewerAppHost.LoadedWmo => ref _loadedWmo;
    ref Rendering.LoadingScreen? IViewerAppHost.LoadingScreen => ref _loadingScreen;
    HashSet<string> IViewerAppHost.LoggedStandaloneMissingSkinPaths => _loggedStandaloneMissingSkinPaths;
    SqlSpawnStreamingService IViewerAppHost.SqlSpawnStreaming => _sqlSpawnStreaming;
    Dictionary<string, string?> IViewerAppHost.StandaloneSkinPathCache => _standaloneSkinPathCache;
    ref ReplaceableTextureResolver? IViewerAppHost.TexResolver => ref _texResolver;
    WdlPreviewService IViewerAppHost.WdlPreview => _wdlPreview;
    void IViewerAppHost.FrameCurrentModel() => _modelInspector.FrameCurrentModel();
    string? IViewerAppHost.TryGetLoadedLocalWdtPath() => _dataSourceSession.TryGetLoadedLocalWdtPath();
    ref DBCD.Providers.IDBCProvider? IViewerAppHost.DbcProvider => ref _dbcProvider;
    ref string? IViewerAppHost.DbdDir => ref _dbdDir;
    ref float IViewerAppHost.DefaultFogEnd => ref _renderQuality._defaultFogEnd;
    ref float IViewerAppHost.DefaultFogStart => ref _renderQuality._defaultFogStart;
    ref MinimapRenderer? IViewerAppHost.MinimapRenderer => ref _minimapRenderer;
    StandaloneModelLoaderService IViewerAppHost.ModelLoader => _modelLoader;
    ref int IViewerAppHost.SavedDetailedAdtTileCountOverride => ref _savedDetailedAdtTileCountOverride;
    Dictionary<string, SavedObjectPathFilterMap> IViewerAppHost.SavedObjectPathFiltersByMap => _savedObjectPathFiltersByMap;
    List<TerrainHiddenTileCandidate> IViewerAppHost.TerrainAnalysisHiddenCandidates => _terrainAnalysisHiddenCandidates;
    ref int IViewerAppHost.TerrainAnalysisHiddenSelectedIndex => ref _terrainAnalysisHiddenSelectedIndex;
    ref string IViewerAppHost.TerrainAnalysisHiddenStatus => ref _terrainAnalysisHiddenStatus;
    ref (int tileX, int tileY)? IViewerAppHost.TerrainAnalysisPreviewCompareTile => ref _terrainAnalysisPreviewCompareTile;
    ref float? IViewerAppHost.TerrainAnalysisPreviewSimilarity => ref _terrainAnalysisPreviewSimilarity;
    void IViewerAppHost.ApplyLayoutObjectPreviewModeToScene() => ApplyLayoutObjectPreviewModeToScene();
    void IViewerAppHost.ApplySavedPm4AlignmentToScene() => _pm4Workbench.ApplySavedPm4AlignmentToScene();
    void IViewerAppHost.InvalidatePm4DerivedReports() => _pm4Workbench.InvalidatePm4DerivedReports();
    bool IViewerAppHost.FullLoadMode { get => FullLoadMode; set => FullLoadMode = value; }
    ref bool IViewerAppHost.AutoOpenWorldMapsPanel => ref _autoOpenWorldMapsPanel;
    ref AssetCatalogView? IViewerAppHost.CatalogView => ref _catalogView;
    ref string IViewerAppHost.ExtensionFilter => ref _extensionFilter;
    ref List<string> IViewerAppHost.FilteredFiles => ref _filteredFiles;
    ref string IViewerAppHost.LastGameFolderPath => ref _lastGameFolderPath;
    ref string IViewerAppHost.LastLooseOverlayPath => ref _lastLooseOverlayPath;
    ref string IViewerAppHost.SearchFilter => ref _searchFilter;
    ref int IViewerAppHost.SelectedFileIndex => ref _selectedFileIndex;
    WorldLoaderService IViewerAppHost.WorldLoader => _worldLoader;
    M2CameraPathDocument IViewerAppHost.CameraPath => _cameraPath;
    ref Terrain.BoundingBoxRenderer? IViewerAppHost.EditorOverlayBb => ref _editorOverlayBb;
    ref int IViewerAppHost.LastMcnkOverlayChunkCount => ref _investigation._lastMcnkOverlayChunkCount;
    ref int IViewerAppHost.LastMcnkWeakCornerCount => ref _investigation._lastMcnkWeakCornerCount;
    ref InvestigationService.McnkOverlayFlags IViewerAppHost.McnkOverlayFlags => ref _investigation._mcnkOverlayFlags;
    ShellLayoutService IViewerAppHost.ShellLayout => _shellLayout;
    ref bool IViewerAppHost.ShowCameraPathOverlay => ref _showCameraPathOverlay;
    ref bool IViewerAppHost.ShowMcnkFlagOverlay => ref _investigation._showMcnkFlagOverlay;
    ref bool IViewerAppHost.ShowMcnkWeakCorners => ref _investigation._showMcnkWeakCorners;
    DataSourceSessionService IViewerAppHost.DataSourceSession => _dataSourceSession;
    Dictionary<(int tileX, int tileY), WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyTileAnalysis> IViewerAppHost.StratigraphyTileAnalyses => _stratigraphyTileAnalyses;
    ConverterDialogsService IViewerAppHost.ConverterDialogs => _converterDialogs;
    PlacementEditService IViewerAppHost.PlacementEditing => _placementEditing;
    ref bool IViewerAppHost.TaxiRideCameraEnabled => ref _taxiRideCameraEnabled;
    ref bool IViewerAppHost.WlLayerListIsolationEnabled => ref _wlLayerListIsolationEnabled;
    ref string IViewerAppHost.WlLayerSelectedBodyKey => ref _wlLayerSelectedBodyKey;
    void IViewerAppHost.DrawTerrainChunkInvestigationPanel(bool defaultOpen) => _investigation.DrawTerrainChunkInvestigationPanel(defaultOpen);
    void IViewerAppHost.DrawVisualInvestigationToolbox(bool showWorldObjectRangeControls) => _investigation.DrawVisualInvestigationToolbox(showWorldObjectRangeControls);
    void IViewerAppHost.OpenPm4Workbench(Pm4WorkbenchTab tab) => _pm4Workbench.OpenPm4Workbench(tab);
    bool IViewerAppHost.ShouldIncludeWlBodyInUiList(WlLiquidBody body) => _investigation.ShouldIncludeWlBodyInUiList(body);
    bool IViewerAppHost.IsWlListIsolationActive => _investigation.IsWlListIsolationActive;
    ref int IViewerAppHost.ActiveBottomTabIndex => ref _activeBottomTabIndex;
    ref string IViewerAppHost.ActiveDatasetVersionRoot => ref _datasetCatalog._activeDatasetVersionRoot;
    ref WorkbenchTab IViewerAppHost.ActiveTopTab => ref _activeTopTab;
    ref int IViewerAppHost.ActiveUtilitiesTabIndex => ref _activeUtilitiesTabIndex;
    ref bool IViewerAppHost.ArcheologyApplyToNextCapture => ref _archeologyApplyToNextCapture;
    ref bool IViewerAppHost.ArcheologyApplyToVideoRecording => ref _archeologyApplyToVideoRecording;
    ref int IViewerAppHost.ArcheologyMaxUniqueId => ref _archeologyMaxUniqueId;
    ref int IViewerAppHost.ArcheologyMinUniqueId => ref _archeologyMinUniqueId;
    ref bool IViewerAppHost.ArcheologyPlaybackLoop => ref _archeologyPlaybackLoop;
    ref float IViewerAppHost.ArcheologyPlaybackSpeed => ref _archeologyPlaybackSpeed;
    ref int IViewerAppHost.ArcheologyScopeIndex => ref _archeologyScopeIndex;
    ref float IViewerAppHost.CameraSpeed => ref _cameraSpeed;
    ref string IViewerAppHost.CaptureOutputDir => ref _captureOutputDir;
    List<WoWViewer.Terrain.ClientBuildOption> IViewerAppHost.ClientBuildOptions => _clientBuildOptions;
    ref string IViewerAppHost.DatasetCatalogRoot => ref _datasetCatalog._datasetCatalogRoot;
    ref bool IViewerAppHost.EnableMultisample => ref _renderQuality._enableMultisample;
    ref bool IViewerAppHost.EnableTerrainBackfaceCulling => ref _renderQuality._enableTerrainBackfaceCulling;
    ref bool IViewerAppHost.HasExplicitWmoMliqRotationOverride => ref _hasExplicitWmoMliqRotationOverride;
    ref List<KnownGoodClientPath> IViewerAppHost.KnownGoodClientPaths => ref _knownGoodClientPaths;
    ref Vector2 IViewerAppHost.MinimapPanOffset => ref _minimapPanOffset;
    ref float IViewerAppHost.MinimapZoom => ref _minimapZoom;
    ref bool IViewerAppHost.OpenForgetKnownGoodClientConfirm => ref _openForgetKnownGoodClientConfirm;
    ref string? IViewerAppHost.PendingForgetKnownGoodClientDisplayName => ref _pendingForgetKnownGoodClientDisplayName;
    ref string? IViewerAppHost.PendingForgetKnownGoodClientPath => ref _pendingForgetKnownGoodClientPath;
    ref Vector3 IViewerAppHost.Pm4SavedOverlayRotationDegrees => ref _pm4SavedOverlayRotationDegrees;
    ref Vector3 IViewerAppHost.Pm4SavedOverlayScale => ref _pm4SavedOverlayScale;
    ref Vector3 IViewerAppHost.Pm4SavedOverlayTranslation => ref _pm4SavedOverlayTranslation;
    ref Dictionary<string, Pm4WmoMatchEntry> IViewerAppHost.Pm4WmoMatchEntries => ref _pm4WmoMatchEntries;
    ref Pm4WmoMatchStore? IViewerAppHost.Pm4WmoMatchStore => ref _pm4WmoMatchStore;
    Dictionary<string, SavedPm4ObjectMatchSelection> IViewerAppHost.SavedPm4ObjectMatches => _savedPm4ObjectMatches;
    ref int IViewerAppHost.SelectedBuildOptionIndex => ref _selectedBuildOptionIndex;
    ref string IViewerAppHost.SelectedDatasetVersionRoot => ref _datasetCatalog._selectedDatasetVersionRoot;
    ref TextureFilteringMode IViewerAppHost.TextureFilteringMode => ref _renderQuality._textureFilteringMode;
    ref float IViewerAppHost.UiFontScale => ref _uiFontScale;
    ref ThemesService.UiThemeKind IViewerAppHost.UiTheme => ref _themes._uiTheme;
    ref int IViewerAppHost.VideoCaptureContainerIndex => ref _videoCaptureContainerIndex;
    ref int IViewerAppHost.VideoCaptureFps => ref _videoCaptureFps;
    ref bool IViewerAppHost.VideoCaptureIncludeUi => ref _videoCaptureIncludeUi;
    ref string IViewerAppHost.VideoEncoderExecutable => ref _videoEncoderExecutable;
    int IViewerAppHost.FindBuildOptionIndex(string? buildVersion) => _clientDialogs.FindBuildOptionIndex(buildVersion);
    void IViewerAppHost.NormalizeWorkbenchStateAfterLoad() => _workbenchPanels.NormalizeWorkbenchStateAfterLoad();
    void IViewerAppHost.RefreshClientBuildOptions() => _clientDialogs.RefreshClientBuildOptions();
    void IViewerAppHost.RefreshDatasetCatalog() => _datasetCatalog.RefreshDatasetCatalog();
    ProjectOutputService IViewerAppHost.ProjectOutput => _projectOutput;
    ref string IViewerAppHost.FolderInputBuf => ref _folderInputBuf;
    ref bool IViewerAppHost.PendingKnownGoodClientAttachLooseFolder => ref _pendingKnownGoodClientAttachLooseFolder;
    ref string? IViewerAppHost.PendingKnownGoodClientBuildVersion => ref _pendingKnownGoodClientBuildVersion;
    ref string? IViewerAppHost.PendingKnownGoodClientPath => ref _pendingKnownGoodClientPath;
    ViewerSettingsService IViewerAppHost.Settings => _settings;
    ref bool IViewerAppHost.ShowBuildSelectionDialog => ref _showBuildSelectionDialog;
    ref bool IViewerAppHost.ShowFolderInput => ref _showFolderInput;
    ref bool IViewerAppHost.ShowListfileInput => ref _showListfileInput;
    ref bool IViewerAppHost.ShowRosettaDatastoreDialog => ref _showRosettaDatastoreDialog;
    CascAhdrSourceService IViewerAppHost.CascAhdrSource => _cascAhdrSource;
    ClientDialogsService IViewerAppHost.ClientDialogs => _clientDialogs;
    ref bool IViewerAppHost.ShowFileBrowser => ref _showFileBrowser;
    ref bool IViewerAppHost.ShowLogViewer => ref _showLogViewer;
    ref bool IViewerAppHost.ShowPerfWindow => ref _showPerfWindow;
    ref bool IViewerAppHost.ShowSynthesizedMinimapExportDialog => ref _showSynthesizedMinimapExportDialog;
    ref bool IViewerAppHost.WantExportGlb => ref _wantExportGlb;
    ref bool IViewerAppHost.WantExportGlbCollision => ref _wantExportGlbCollision;
    ref bool IViewerAppHost.WantExportMapGlbTiles => ref _wantExportMapGlbTiles;
    ref bool IViewerAppHost.WantOpenFile => ref _wantOpenFile;
    ref bool IViewerAppHost.WantTerrainExport => ref _wantTerrainExport;
    ref bool IViewerAppHost.WantTerrainImport => ref _wantTerrainImport;
    ref bool IViewerAppHost.WorkbenchOpen => ref _workbenchOpen;
    void IViewerAppHost.ResetCamera() => ResetCamera();
    ref int IViewerAppHost.ActivePm4TabIndex => ref _activePm4TabIndex;
    void IViewerAppHost.CopyTextToClipboard(string text, string description) => _navigatorPanel.CopyTextToClipboard(text, description);
    ref bool IViewerAppHost.ArcheologyPlaybackActive => ref _archeologyPlaybackActive;
    ref MapListSortMode IViewerAppHost.MapListSortMode => ref _mapListSortMode;
    Pm4WorkbenchService IViewerAppHost.Pm4Workbench => _pm4Workbench;
    ref bool IViewerAppHost.ShowUniqueIdArchaeologyWindow => ref _showUniqueIdArchaeologyWindow;
    HashSet<int> IViewerAppHost.HighlightedStandaloneWmoGroupIndices => _highlightedStandaloneWmoGroupIndices;
    ref int IViewerAppHost.HoveredStandaloneWmoGroupIndex => ref _hoveredStandaloneWmoGroupIndex;
    ref int IViewerAppHost.SelectedStandaloneWmoGroupIndex => ref _selectedStandaloneWmoGroupIndex;
    ref bool IViewerAppHost.StandaloneWmoGroupLabelsAllEnabled => ref _standaloneWmoGroupLabelsAllEnabled;
    ref bool IViewerAppHost.StandaloneWmoGroupOverlayEnabled => ref _standaloneWmoGroupOverlayEnabled;
    ref bool IViewerAppHost.StandaloneWmoOverlayIncludeHiddenGroups => ref _standaloneWmoOverlayIncludeHiddenGroups;
    void IViewerAppHost.DrawAssetPathActions(string label, string assetPath, string idSuffix) => _navigatorPanel.DrawAssetPathActions(label, assetPath, idSuffix);
    void IViewerAppHost.FramePoint(Vector3 target, float radius) => _navigatorPanel.FramePoint(target, radius);
    ChunkEditService IViewerAppHost.ChunkEdit => _chunkEdit;
    ref bool IViewerAppHost.LayoutObjectPreviewMode => ref _layoutObjectPreviewMode;
    ref bool IViewerAppHost.ShowWeakSignalWindow => ref _showWeakSignalWindow;
    StratigraphyService IViewerAppHost.Stratigraphy => _stratigraphy;
    ref int IViewerAppHost.PendingMinimapTeleportClickCount => ref _pendingMinimapTeleportClickCount;
    ref (int tileX, int tileY)? IViewerAppHost.PendingMinimapTeleportTile => ref _pendingMinimapTeleportTile;
    ModelInspectorPanelService IViewerAppHost.ModelInspector => _modelInspector;
    TerrainControlsPanelService IViewerAppHost.TerrainControlsPanel => _terrainControlsPanel;
    TerrainQueryService IViewerAppHost.TerrainQuery => _terrainQuery;
    NavigatorPanelService IViewerAppHost.NavigatorPanel => _navigatorPanel;
    InvestigationService IViewerAppHost.Investigation => _investigation;
    TerrainInspectionPanelService IViewerAppHost.TerrainInspection => _terrainInspection;
    ArchaeologyPanelService IViewerAppHost.ArchaeologyPanel => _archaeologyPanel;
    AudioPanelService IViewerAppHost.AudioPanel => _audioPanel;
    InspectorPayloadsService IViewerAppHost.InspectorPayloads => _inspectorPayloads;
    LightingPanelService IViewerAppHost.LightingPanel => _lightingPanel;
    MainMenuBarService IViewerAppHost.MainMenuBar => _mainMenuBar;
    TaxiPanelService IViewerAppHost.TaxiPanel => _taxiPanel;
    ThemesService IViewerAppHost.Themes => _themes;
    ViewerChromeService IViewerAppHost.ViewerChrome => _viewerChrome;
    WorldObjectsPanelService IViewerAppHost.WorldObjectsPanel => _worldObjectsPanel;
    ref CameraHudRig? IViewerAppHost.CameraHudRig => ref _cameraHudRig;
    DatasetCatalogService IViewerAppHost.DatasetCatalog => _datasetCatalog;
    RenderQualityService IViewerAppHost.RenderQuality => _renderQuality;
    ref TerrainAnalysisPreviewTexture? IViewerAppHost.TerrainAnalysisAlphaTexture => ref _terrainAnalysisAlphaTexture;
    ref TerrainAnalysisPreviewTexture? IViewerAppHost.TerrainAnalysisGlobalTexture => ref _terrainAnalysisGlobalTexture;
    ref TerrainAnalysisPreviewTexture? IViewerAppHost.TerrainAnalysisLocalTexture => ref _terrainAnalysisLocalTexture;
    // HOST-IMPL-END

    public void Run(string[]? initialArgs = null)
    {
        var opts = WindowOptions.Default;
        opts.Size = new Vector2D<int>(1600, 900);
        opts.Title = ViewerProductName;
        opts.API = new GraphicsAPI(ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion(3, 3));
        opts.VSync = false; // Disable VSync — let the GPU run uncapped for profiling

        _window = Window.Create(opts);
        _window.Load += () => OnLoad(initialArgs);
        _window.Render += OnRender;
        _window.Update += OnUpdate;
        _window.Resize += OnWindowResize;
        _window.FramebufferResize += OnResize;
        _window.Closing += OnClose;

        _window.Run();
    }

    private void OnLoad(string[]? initialArgs)
    {
        _gl = _window.CreateOpenGL();
        _input = _window.CreateInput();
        _imGui = new ImGuiController(_gl, _window, _input);
        _shellLayout.SyncImGuiWindowMetrics(_window.Size, _window.FramebufferSize);
        ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.DockingEnable;

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Enable(EnableCap.CullFace);

        _loadingScreen = new Rendering.LoadingScreen(_gl);
        _sceneCursorRenderer = new SceneCursorRenderer(_gl, _dataSource, _texResolver);
        _sceneClusterSelector3D = new SceneClusterSelector3D(_gl);
        _cameraHudRig = new CameraHudRig(_gl);

        _sqlSpawnStreaming.TryAutoPopulateAlphaCoreRoot();
        _settings.LoadViewerSettings();
        _themes.ApplyActiveUiTheme();
        LoadCameraShotPoints();
        _renderQuality.DetectRenderQualityCapabilities();
        _renderQuality.ApplyRenderQualitySettings(refreshTextures: false);

        // Mouse input for viewport (not consumed by ImGui)
        foreach (var mouse in _input.Mice)
        {
            mouse.MouseDown += (_, btn) =>
            {
                _shellLayout.QueueImGuiMouseButtonEvent(btn, down: true);

                if (btn == MouseButton.Right && _shellLayout.CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
                    _mouseDown = true;
                if (btn == MouseButton.Left && _shellLayout.CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
                {
                    bool shift = ImGui.GetIO().KeyShift;
                    var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
                    if (terrainRenderer != null && _chunkToolEnabled)
                    {
                        bool ctrl = ImGui.GetIO().KeyCtrl;

                        if (ctrl && !shift)
                        {
                            if (_chunkEdit.TryLockChunkPasteTarget(terrainRenderer))
                                return;
                        }
                        else if (shift)
                        {
                            if (_chunkEdit.TryHandleChunkSelectionClick(terrainRenderer, shift))
                                return;
                        }
                    }

                    if (_worldScene != null)
                        _sceneHoverPick.PickObjectAtMouse(_lastMouseX, _lastMouseY, addPm4ToCollection: shift);
                    else if (terrainRenderer != null && !shift && !_chunkToolEnabled
                        && _terrainQuery.TryPickTerrainChunkUnderMouse(terrainRenderer, out var terrainChunk))
                        _terrainInspection.SelectTerrainChunkFromClick(terrainChunk);
                }
            };
            mouse.MouseUp += (_, btn) =>
            {
                _shellLayout.QueueImGuiMouseButtonEvent(btn, down: false);

                if (btn == MouseButton.Right) _mouseDown = false;
            };
            mouse.MouseMove += (_, pos) =>
            {
                float dx = pos.X - _lastMouseX;
                float dy = pos.Y - _lastMouseY;
                _lastMouseX = pos.X;
                _lastMouseY = pos.Y;

                if (_mouseDown && !_shellLayout.IsSceneMouseCaptureBlocked(_lastMouseX, _lastMouseY))
                {
                    if (_taxiRideCameraEnabled)
                    {
                        AdjustTaxiRideFreeLook(-dx * 0.5f, -dy * 0.5f);
                    }
                    else
                    {
                        _camera.Yaw -= dx * 0.5f;   // Drag left = look left, Drag right = look right
                        _camera.Pitch -= dy * 0.5f; // Drag up = look up, Drag down = look down
                        _camera.Pitch = Math.Clamp(_camera.Pitch, -89f, 89f);
                    }
                }
            };
            mouse.Scroll += (_, scroll) =>
            {
                _pendingSceneMouseWheelDelta += scroll.Y;
            };
        }

        ApplyStartupAutomation(initialArgs);
    }

    private void OnUpdate(double dt)
    {
        _shellLayout.SyncImGuiWindowMetrics(_window.Size, _window.FramebufferSize);
        _imGui.Update((float)dt);
        _shellLayout.FlushPendingImGuiMouseButtonEvents();
        HandleSceneMouseWheelInput();
        HandleKeyboardInput((float)dt);
        UpdateCameraPathPlayback(dt);
        UpdateCameraPathPreload();
        UpdateTaxiRideCamera();
        _archaeologyPanel.UpdateArcheologyPlayback(dt);
        _minimapRenderer?.ProcessPendingLoads(
            maxLoads: (_fullscreenMinimap || _showMinimapWindow) ? 4 : 1,
            maxBudgetMs: (_fullscreenMinimap || _showMinimapWindow) ? 6.0 : 1.5);
        _sqlSpawnStreaming.UpdateSqlSpawnStreaming();
        _terrainWeakSignalRestore.UpdateTerrainWeakSignalRestoreForCamera();
    }

    private (int tileX, int tileY) GetCameraTile()
    {
        int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
        return (tileX, tileY);
    }

    private bool _mKeyWasPressed = false;
    private bool _pKeyWasPressed = false;
    private bool _iKeyWasPressed = false;
    private bool _tabKeyWasPressed = false;
    private bool _escKeyWasPressed = false;
    private bool _leftArrowWasPressed = false;
    private bool _rightArrowWasPressed = false;
    private bool _spaceWasPressed = false;

    private void HandleSceneMouseWheelInput()
    {
        if (MathF.Abs(_pendingSceneMouseWheelDelta) <= float.Epsilon)
            return;

        float scrollDelta = _pendingSceneMouseWheelDelta;
        _pendingSceneMouseWheelDelta = 0f;

        if (!_shellLayout.CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
            return;

        _camera.Move(5f * scrollDelta, 0f, 0f, 1f);
    }

    private bool CanSceneConsumeKeyboardInput()
    {
        return !IsSceneKeyboardCaptureBlocked();
    }

    private bool IsSceneKeyboardCaptureBlocked()
    {
        ImGuiIOPtr io = ImGui.GetIO();
        return io.WantCaptureKeyboard || io.WantTextInput;
    }

    private void HandleKeyboardInput(float dt)
    {
        if (_input.Keyboards.Count == 0) return;
        var kb = _input.Keyboards[0];
        bool canSceneConsumeKeyboard = CanSceneConsumeKeyboardInput();

        bool ctrlDown = kb.IsKeyPressed(Key.ControlLeft) || kb.IsKeyPressed(Key.ControlRight);
        bool cDown = kb.IsKeyPressed(Key.C);
        bool vDown = kb.IsKeyPressed(Key.V);
        bool ctrlCDown = ctrlDown && cDown;
        bool ctrlVDown = ctrlDown && vDown;
        bool cameraPathKeyboardAction = HandleCameraPathKeyboardInput(kb, ctrlDown, shiftDown: kb.IsKeyPressed(Key.ShiftLeft) || kb.IsKeyPressed(Key.ShiftRight));

        if (_chunkToolEnabled && canSceneConsumeKeyboard)
        {
            var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            if (terrainRenderer != null)
            {
                if (ctrlCDown && !_chunkClipboardCtrlCWasPressed)
                    _chunkEdit.ExecuteChunkClipboardCopy(terrainRenderer);

                if (ctrlVDown && !_chunkClipboardCtrlVWasPressed)
                    _chunkEdit.ExecuteChunkClipboardPaste(terrainRenderer);
            }
        }

        _chunkClipboardCtrlCWasPressed = ctrlCDown;
        _chunkClipboardCtrlVWasPressed = ctrlVDown;

        bool tabPressed = kb.IsKeyPressed(Key.Tab);
        if (canSceneConsumeKeyboard && tabPressed && !_tabKeyWasPressed)
            _hideUiChrome = !_hideUiChrome;
        _tabKeyWasPressed = tabPressed;

        bool escPressed = kb.IsKeyPressed(Key.Escape);
        if (canSceneConsumeKeyboard && escPressed && !_escKeyWasPressed)
        {
            if (_sceneClusterSelector3D != null && _sceneClusterSelector3D.IsActive)
            {
                _sceneClusterSelector3D.Close();
                _sceneHoverPick.ClearPendingClickSelection();
            }
            else if (_worldScene != null)
            {
                _sceneHoverPick.ClearPendingClickSelection();
                _investigation.ClearSelectedWlLiquidBody(clearListIsolation: true);
                _worldScene.ClearSelection();
                _worldScene.ClearTaxiSelection();
                _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
                _taxiAndAreaPoi.ClearSelectedAreaPoiInfo();
                _selectedObjectIndex = -1;
                _selectedObjectType = "";
                _selectedObjectInfo = "";
            }
        }
        _escKeyWasPressed = escPressed;

        bool pPressed = kb.IsKeyPressed(Key.P);
        if (canSceneConsumeKeyboard && pPressed && !_pKeyWasPressed)
        {
            _showRightSidebar = true;
            _activeBottomDrawerTab = FixedBottomDrawerTab.Pm4;
            if (_workspaceMode == WorkspaceMode.Editor)
                SetEditorWorkspaceTask(EditorWorkspaceTask.Pm4Evidence);
        }
        _pKeyWasPressed = pPressed;

        bool iPressed = kb.IsKeyPressed(Key.I);
        if (canSceneConsumeKeyboard && iPressed && !_iKeyWasPressed)
        {
            _showRightSidebar = !_showRightSidebar;
            if (_showRightSidebar)
                _shellLayout.FocusShellPanel(ShellPanelId.Inspector);
        }
        _iKeyWasPressed = iPressed;

        // M key toggles fullscreen minimap (only when terrain is loaded)
        bool mPressed = kb.IsKeyPressed(Key.M);
        if (canSceneConsumeKeyboard && mPressed && !_mKeyWasPressed && (_terrainManager != null || _vlmTerrainManager != null))
            ToggleFullscreenMinimap();
        _mKeyWasPressed = mPressed;

        // Arrow keys and spacebar for MDX animation control
        if (!cameraPathKeyboardAction && _renderer is IModelRenderer modelRenderer && modelRenderer.Animator != null && modelRenderer.Animator.Sequences.Count > 0)
        {
            var animator = modelRenderer.Animator;
            int currentSeq = animator.CurrentSequence;
            
            if (currentSeq >= 0 && currentSeq < animator.Sequences.Count)
            {
                var seq = animator.Sequences[currentSeq];
                float duration = seq.Time.End - seq.Time.Start;
                float currentFrame = animator.CurrentFrame;
                
                // Left arrow: step backward
                bool leftPressed = kb.IsKeyPressed(Key.Left);
                if (canSceneConsumeKeyboard && leftPressed && !_leftArrowWasPressed)
                {
                    animator.IsPlaying = false;
                    animator.StepToPrevKeyframe();
                }
                _leftArrowWasPressed = leftPressed;
                
                // Right arrow: step forward
                bool rightPressed = kb.IsKeyPressed(Key.Right);
                if (canSceneConsumeKeyboard && rightPressed && !_rightArrowWasPressed)
                {
                    animator.IsPlaying = false;
                    animator.StepToNextKeyframe();
                }
                _rightArrowWasPressed = rightPressed;
                
                // Spacebar: toggle play/pause
                bool spacePressed = kb.IsKeyPressed(Key.Space);
                if (canSceneConsumeKeyboard && spacePressed && !_spaceWasPressed)
                {
                    animator.IsPlaying = !animator.IsPlaying;
                }
                _spaceWasPressed = spacePressed;
            }
        }

        if (_taxiRideCameraEnabled || !canSceneConsumeKeyboard)
            return;

        // Free-fly: WASD moves the camera position, Shift = 5x boost
        bool shift = kb.IsKeyPressed(Key.ShiftLeft) || kb.IsKeyPressed(Key.ShiftRight);
        float speed = _cameraSpeed * dt * (shift ? 5f : 1f);

        bool w = kb.IsKeyPressed(Key.W);
        bool a = kb.IsKeyPressed(Key.A);
        bool s = !ctrlDown && kb.IsKeyPressed(Key.S);
        bool d = kb.IsKeyPressed(Key.D);
        bool q = kb.IsKeyPressed(Key.Q);
        bool e = !ctrlDown && kb.IsKeyPressed(Key.E);

        if (w || a || s || d || q || e)
        {
            float forward = (w ? 1 : 0) - (s ? 1 : 0);
            float right = (d ? 1 : 0) - (a ? 1 : 0);
            float up = (q ? 1 : 0) - (e ? 1 : 0);
            _camera.Move(forward, right, up, speed);
        }
    }

    private unsafe void OnRender(double dt)
    {
        _datasetExportDialogs.PromotePendingMlFinalizeAfterExport();
        PromotePendingMkHarvestViewerValidationCapturePlan();
        PromotePendingRoofCaptureBatch();
        PrepareNextCaptureRequest();

        // FPS tracking
        _frameCount++;
        _fpsTimer += dt;
        _frameTimeMs = dt * 1000.0;
        if (_fpsTimer >= 1.0)
        {
            _currentFps = _frameCount / _fpsTimer;
            _frameCount = 0;
            _fpsTimer = 0;
        }

        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        // If loading screen is active, render it instead of the normal scene.
        // Keep it up until the initial AOI tiles have all finished loading (no more
        // background loads or pending GPU uploads). This prevents the map from appearing
        // half-loaded while tiles are still streaming in.
        if (_loadingScreen != null && _loadingScreen.IsActive)
        {
            bool isWmoOnly = _worldScene != null && _terrainManager != null && _terrainManager.Adapter.IsWmoBased;
            bool hasTiles = _terrainManager != null && _terrainManager.LoadedTileCount > 0;
            bool stillStreaming = _terrainManager != null && _terrainManager.IsStreaming;
            // Dismiss when: WMO-only map, OR tiles are loaded AND no more streaming in progress
            if (isWmoOnly || (hasTiles && !stillStreaming))
            {
                _loadingScreen.Disable();
            }
            else
            {
                // Still loading — update AOI so tiles start streaming
                if (_terrainManager != null)
                    _terrainManager.UpdateAOI(_camera.Position, _camera.Forward);
                // Update progress bar based on loaded vs expected tiles
                if (_terrainManager != null && _terrainManager.LoadedTileCount > 0)
                    _loadingScreen.UpdateProgress(_terrainManager.LoadedTileCount, _terrainManager.LoadedTileCount + 10);
                var sz = _window.Size;
                _loadingScreen.Render(sz.X, sz.Y);
                return;
            }
        }

        // Render 3D scene first (always set up viewport and 3D scene cursor even on startup when _renderer is null)
        var size = _window.Size;
        bool hasSceneViewportRect = _shellLayout.TryGetSceneViewportRect(out float sceneViewportX, out float sceneViewportY, out float sceneViewportWidth, out float sceneViewportHeight);
        int sceneFramebufferX = 0;
        int sceneFramebufferY = 0;
        uint sceneFramebufferWidth = 0;
        uint sceneFramebufferHeight = 0;
        bool hasSceneViewport = hasSceneViewportRect
            && _shellLayout.TryGetSceneFramebufferViewport(out sceneFramebufferX, out sceneFramebufferY, out sceneFramebufferWidth, out sceneFramebufferHeight);
        if (hasSceneViewport)
            _gl.Viewport(sceneFramebufferX, sceneFramebufferY, sceneFramebufferWidth, sceneFramebufferHeight);
        else
            _gl.Viewport(_window.FramebufferSize);

        float aspect = hasSceneViewport
            ? sceneViewportWidth / Math.Max(sceneViewportHeight, 1f)
            : (float)size.X / Math.Max(size.Y, 1);
        float farPlane = _terrainQuery.GetSceneFarPlane();
        Matrix4x4 view;
        Matrix4x4 proj;
        if (!TryGetMkHarvestViewerValidationSceneMatrices(aspect, out view, out proj))
        {
            view = _camera.GetViewMatrix();
            proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, farPlane);
        }

        if (_renderer != null)
        {
            // Update terrain AOI before rendering
            if (_terrainManager != null)
                _terrainManager.UpdateAOI(_camera.Position, _camera.Forward);
            else if (_vlmTerrainManager != null)
                _vlmTerrainManager.UpdateAOI(_camera.Position);

            if (_worldScene != null)
                _sceneHoverPick.UpdateWorldSceneWireframeReveal(view, proj);

            if (_worldScene != null)
                _sceneHoverPick.UpdateWorldSceneHoveredAssetInfo(view, proj);

            // Update native-style ZoneText/SubzoneText from the resident chunk metadata under the
            // camera. Batched terrain owns one GPU mesh per tile, so the area lookup must use the
            // resident chunk-info index instead of the legacy per-chunk GPU mesh list.
            var areaChunkRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            _areaContext.UpdateCurrentAreaContext(areaChunkRenderer);
            _worldScene?.SetCurrentAreaLookup(_currentAreaLookup);
            _areaContext.UpdateAreaOverlay(areaChunkRenderer);

            // Render the scene
            if (_renderer is IModelRenderer modelRenderer)
            {
                // Standalone MDX: render with proper lighting matching terrain viewer
                RenderSkyGradient();
                var lightDir = Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f));
                var lightColor = new Vector3(1.0f, 0.95f, 0.9f);
                var ambientColor = new Vector3(0.35f, 0.35f, 0.4f);
                var fogColor = new Vector3(0.5f, 0.6f, 0.7f);
                float fogStart = farPlane * 0.5f;
                float fogEnd = farPlane;
                var scale = Matrix4x4.CreateScale(-1f, 1f, 1f); // MirrorX for standalone
                modelRenderer.UpdateAnimation(); // Advance skeletal animation before rendering
                _gl.Disable(EnableCap.Blend);
                modelRenderer.RenderWithTransform(scale, view, proj, RenderPass.Opaque, 1.0f,
                    fogColor, fogStart, fogEnd, _camera.Position, lightDir, lightColor, ambientColor);
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                modelRenderer.RenderWithTransform(scale, view, proj, RenderPass.Transparent, 1.0f,
                    fogColor, fogStart, fogEnd, _camera.Position, lightDir, lightColor, ambientColor);
            }
            else if (_renderer is WmoRenderer wmoR)
            {
                // Standalone WMO: render with proper lighting
                RenderSkyGradient();
                var lightDir = Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f));
                var lightColor = new Vector3(1.0f, 0.95f, 0.9f);
                var ambientColor = new Vector3(0.35f, 0.35f, 0.4f);
                var fogColor = new Vector3(0.5f, 0.6f, 0.7f);
                float fogStart = farPlane * 0.5f;
                float fogEnd = farPlane;
                wmoR.RenderWithTransform(Matrix4x4.Identity, view, proj,
                    fogColor, fogStart, fogEnd, _camera.Position, lightDir, lightColor, ambientColor);
                _wmoGroupsPanel.DrawStandaloneWmoGroupOverlay(wmoR, view, proj, sceneViewportX, sceneViewportY, sceneViewportWidth, sceneViewportHeight);
            }
            else
            {
                // WorldScene / VLM terrain — handles its own lighting
                _renderer.Render(view, proj);
                _terrainQuery.DrawEditorOverlays(view, proj);
                if (hasSceneViewportRect)
                {
                    _areaContext.DrawAreaOverlayLabels(
                        view,
                        proj,
                        sceneViewportX,
                        sceneViewportY,
                        sceneViewportWidth,
                        sceneViewportHeight);
                }
            }
        }
        else
        {
            // Empty / startup scene: render clean backdrop gradient so the viewport is lively
            RenderSkyGradient();
        }

        if (hasSceneViewportRect)
        {
            if (_cameraHudRig != null && _cameraHudRig.Enabled && !_hideUiChrome)
            {
                float hudAspect = (float)sceneViewportWidth / Math.Max(1, sceneViewportHeight);
                _cameraHudRig.Render(_camera, proj, _fovDegrees, hudAspect);
            }

            // World-space selection rings belong in the 3D pass: they are scene geometry and must
            // occlude and be occluded like scene geometry. The cursor itself does NOT draw here -
            // see the overlay pass after ImGui.
            _sceneHoverPick.RenderSceneClusterSelector3D(proj);
        }

        if (hasSceneViewport)
            _gl.Viewport(_window.FramebufferSize);

        CaptureVideoFrameIfNeeded(includeUi: false, dt);
        CompleteCaptureIfReady(includeUi: false);

        // Render ImGui overlay when the native ImGui context is live. Startup capture and
        // teardown can briefly produce frames where the controller still exists but the
        // underlying context is not available.
        if (ShellLayoutService.HasImGuiContext())
        {
            bool hideHardwareCursor = _sceneCursorRenderer != null
                && _sceneCursorRenderer.Style != CursorStyle.ClassicOSArrow
                && _shellLayout.CanSceneConsumeMouse(_lastMouseX, _lastMouseY);

            if (hideHardwareCursor)
            {
                ImGui.SetMouseCursor(ImGuiMouseCursor.None);
            }

            if (_input != null)
            {
                CursorMode targetMode = hideHardwareCursor ? CursorMode.Hidden : CursorMode.Normal;
                foreach (var mouse in _input.Mice)
                {
                    if (mouse.Cursor.CursorMode != targetMode)
                    {
                        mouse.Cursor.CursorMode = targetMode;
                    }
                }
            }

            DrawUI();
            // The in-app path picker is a global modal: drive it every frame so it works from any
            // surface, independent of which panel opened it.
            ImGuiPathPicker.Instance.Draw();
            _imGui.Render();
        }

        CaptureVideoFrameIfNeeded(includeUi: true, dt);
        CompleteCaptureIfReady(includeUi: true);

        // The scene cursor is drawn LAST, after ImGui and after both capture taps.
        //
        // It used to draw with the rest of the 3D pass, which put it underneath every ImGui window:
        // a menu dropdown, a popup, a docked panel or the hover card that opens next to the pointer
        // would all paint over it. Because the hardware cursor is hidden while the 3D cursor is
        // active, that left no pointer at all exactly where the user was trying to aim. GL depth
        // cannot fix this - ImGui is a separate pass - so the only correct place is after it.
        //
        // Drawing after the capture taps also keeps the cursor out of recorded frames, which is
        // where it belongs.
        if (hasSceneViewportRect)
        {
            if (hasSceneViewport)
                _gl.Viewport(sceneFramebufferX, sceneFramebufferY, sceneFramebufferWidth, sceneFramebufferHeight);

            _sceneHoverPick.RenderSceneCursor(view, proj, sceneViewportX, sceneViewportY, sceneViewportWidth, sceneViewportHeight);

            if (hasSceneViewport)
                _gl.Viewport(_window.FramebufferSize);
        }
    }

    /// <summary>
    /// Render a fullscreen sky gradient background for standalone model viewing.
    /// Top = light blue sky, bottom = darker horizon. Drawn before the model with depth test off.
    /// </summary>
    private unsafe void RenderSkyGradient()
    {
        if (!_skyReady)
        {
            // Fullscreen triangle (covers entire screen with one triangle)
            // xy = NDC position, z = vertical interpolant (0=bottom, 1=top)
            float[] verts = {
                -1f, -1f, 0f,  // bottom-left
                 3f, -1f, 0f,  // bottom-right (oversized)
                -1f,  3f, 1f,  // top-left (oversized)
            };

            string vertSrc = @"#version 330 core
layout(location=0) in vec3 aPos;
out float vHeight;
void main() {
    gl_Position = vec4(aPos.xy, 0.9999, 1.0);
    vHeight = (aPos.y + 1.0) * 0.5;
}";
            string fragSrc = @"#version 330 core
in float vHeight;
out vec4 FragColor;
uniform vec3 uTopColor;
uniform vec3 uBotColor;
void main() {
    vec3 col = mix(uBotColor, uTopColor, vHeight);
    FragColor = vec4(col, 1.0);
}";

            uint vs = _gl.CreateShader(ShaderType.VertexShader);
            _gl.ShaderSource(vs, vertSrc);
            _gl.CompileShader(vs);
            uint fs = _gl.CreateShader(ShaderType.FragmentShader);
            _gl.ShaderSource(fs, fragSrc);
            _gl.CompileShader(fs);
            _skyShader = _gl.CreateProgram();
            _gl.AttachShader(_skyShader, vs);
            _gl.AttachShader(_skyShader, fs);
            _gl.LinkProgram(_skyShader);
            _gl.DeleteShader(vs);
            _gl.DeleteShader(fs);

            _skyVao = _gl.GenVertexArray();
            _skyVbo = _gl.GenBuffer();
            _gl.BindVertexArray(_skyVao);
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _skyVbo);
            fixed (float* p = verts)
                _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(verts.Length * sizeof(float)), p, BufferUsageARB.StaticDraw);
            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), (void*)0);
            _gl.BindVertexArray(0);
            _skyReady = true;
        }

        // Draw sky gradient (depth write off, depth test off)
        _gl.Disable(EnableCap.DepthTest);
        _gl.DepthMask(false);
        _gl.UseProgram(_skyShader);

        // WoW-ish sky colors: light blue top, pale horizon bottom
        int topLoc = _gl.GetUniformLocation(_skyShader, "uTopColor");
        int botLoc = _gl.GetUniformLocation(_skyShader, "uBotColor");
        _gl.Uniform3(topLoc, 0.35f, 0.55f, 0.85f);  // sky blue
        _gl.Uniform3(botLoc, 0.65f, 0.72f, 0.80f);   // pale horizon

        _gl.BindVertexArray(_skyVao);
        _gl.DrawArrays(PrimitiveType.Triangles, 0, 3);
        _gl.BindVertexArray(0);

        // Restore depth state for model rendering
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true);
        _gl.Clear(ClearBufferMask.DepthBufferBit);
    }

    private void DrawUI()
    {
        if (!ShellLayoutService.HasImGuiContext())
            return;

        _shellLayout.UpdateShellLayout(ImGui.GetIO().DisplaySize);

        _shellLayout.ResetDockPanelStates();

        // Clear the host rect whenever DrawDockspaceHost will not run this frame - which includes
        // tab UI mode, not just the chrome/dockspace toggles. A stale non-zero rect makes
        // ShouldBypassDockspaceMouseCapture claim the mouse for the scene across the whole viewport
        // rect, defeating ImGui's own capture for any floating window drawn over it: clicks meant
        // for that window also fire scene picking, and the hardware cursor is hidden over it.
        if (_hideUiChrome || !_useDockspaceUi || _useTabUi)
        {
            _dockspaceHostPosition = Vector2.Zero;
            _dockspaceHostSize = Vector2.Zero;
        }

        if (!_hideUiChrome)
        {
            _mainMenuBar.DrawMenuBar();

            // 069 Phase 1: tab system. Off by default; old sidebars still active.
            // When enabled, replaces DrawDockspaceHost + DrawLeftSidebar + DrawRightSidebar
            // with top tab bar + bottom tab bar + central content area.
            if (_useTabUi)
            {
                // 071: left sidebar + right workbench squeeze the 3D viewport.
                _navigatorPanel.DrawLeftSidebar();
                _workbenchPanels.DrawRightSidebar();
            }
            else if (_useDockspaceUi)
            {
                DrawDockspaceHost();
            }

            if (!_useTabUi)
            {
                if (_shellLayout.HasAnyShellPanelsInLane(ShellPanelLane.Left))
                    _navigatorPanel.DrawLegacyLeftSidebar();
                if (_shellLayout.HasAnyShellPanelsInLane(ShellPanelLane.Right))
                    _workbenchPanels.DrawLegacyRightSidebar();
            }

            _viewerChrome.DrawFixedSidebarSplitters();

            // Toolbar is drawn after sidebars so it stays on top of any edge overlap.
            _viewerChrome.DrawToolbar();

            _viewerChrome.DrawBottomBar();

            DrawStatusBar();

            // Floating windows: when tab system is active, only the tools that
            // aren't yet routed into a sub-tab render as floating windows.
            // Phase 2 routes Scene + Utilities; later phases route the rest.
            if (_useTabUi)
            {
                if (_showWdlPreview)
                    _wdlPreview.DrawWdlPreviewDialog();
                // All other tools are routed into tab sub-tabs. The
                // _show*Window flags still exist for users who want the
                // legacy window, but tab system renders its own sub-tab body.
            }
            else
            {
                // Asset Catalog (floating window)
                _catalogView?.Draw();

                // Log Viewer (floating window) - legacy mode only; tabbed mode uses Utilities > Log
                if (_showLogViewer && !_useTabUi)
                    _logViewer.DrawLogViewer();

                // WDL Preview (floating window)
                if (_showWdlPreview)
                    _wdlPreview.DrawWdlPreviewDialog();

                // Minimap panel
                if (_shellLayout.IsShellPanelActive(ShellPanelId.Minimap) && !_fullscreenMinimap)
                    DrawMinimapWindow();

                // Perf (floating window) - legacy mode only; tabbed mode uses Utilities > Perf
                if (_showPerfWindow && !_useTabUi)
                    _pm4Workbench.DrawPerfWindow();


                if (_showCaptureAutomationWindow)
                    DrawCaptureAutomationWindow();

                if (_showCameraPathWindow)
                    DrawCameraPathWindow();

                // Tool windows extracted from right sidebar
                if (_showUniqueIdArchaeologyWindow && _worldScene != null)
                    _archaeologyPanel.DrawUniqueIdArchaeologyWindow();

            }

            // Settings (global configuration window) - must render in BOTH tabbed and legacy modes
            if (_settingsWindow._showSettingsWindow)
                _settingsWindow.DrawSettingsWindow();

            if (!_useTabUi && _showWeakSignalWindow && (_terrainManager != null || _vlmTerrainManager != null))
                _terrainControlsPanel.DrawWeakSignalWindow();

        }

        if (_activeVideoRecording?.MarketingTourAttempt?.ActivePresentation is FeatureTourPresentation presentation)
            MarketingTourOverlayRenderer.Draw(presentation);

        _forceApplyShellPanelLayout = false;

        // Fullscreen minimap overlay (M key toggle)
        if (_fullscreenMinimap && (_worldScene != null || _vlmTerrainManager != null))
            DrawFullscreenMinimap();

        // Modal dialogs
        if (_showFolderInput)
            _clientDialogs.DrawFolderInputDialog();
        if (_showBuildSelectionDialog)
            _clientDialogs.DrawBuildSelectionDialog();
        if (_showListfileInput)
            _clientDialogs.DrawListfileInputDialog();
        if (_mlTraining._showMlTrainingDialog || _mlTraining.IsMlTrainingProcessActive())
            _mlTraining.UpdateMlTrainingMonitor();
        if (_showVlmExportDialog)
            _datasetExportDialogs.DrawVlmExportDialog();
        if (_mlTraining._showMlTrainingDialog)
            _mlTraining.DrawMlTrainingDialog();
        if (_showTerrainTextureTransferDialog)
            _datasetExportDialogs.DrawTerrainTextureTransferDialog();
        if (_showAlphaFolderImportScope)
            _terrainTileIo.DrawAlphaFolderImportScopeDialog();
        if (_showHeightmapFolderImportScope)
            _terrainTileIo.DrawHeightmapFolderImportScopeDialog();
        if (_showMccvFolderImportScope)
            _terrainTileIo.DrawMccvFolderImportScopeDialog();
        if (_showMapConverterDialog)
            _converterDialogs.DrawMapConverterDialog();
        if (_showWmoConverterDialog)
            _converterDialogs.DrawWmoConverterDialog();
        if (_showSynthesizedMinimapExportDialog)
            _synthesizedMinimapExport.DrawSynthesizedMinimapExportDialog();
        if (_showRosettaDatastoreDialog)
            _clientDialogs.DrawRosettaDatastoreDialog();

        _sceneHoverPick.DrawSceneHoverAssetOverlay();
        _sceneHoverPick.DrawClickSelectionOverlay();

        _autoOpenWorldMapsPanel = false;
    }

    private void DrawDockspaceHost()
    {
        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float dockHeight = io.DisplaySize.Y - topOffset - StatusBarHeight;
        if (dockHeight <= 10f)
            return;

        _dockspaceHostPosition = new Vector2(0f, topOffset);
        _dockspaceHostSize = new Vector2(io.DisplaySize.X, dockHeight);

        ImGui.SetNextWindowPos(_dockspaceHostPosition, ImGuiCond.Always);
        ImGui.SetNextWindowSize(_dockspaceHostSize, ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 0f);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, 0f);

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoTitleBar
            | ImGuiWindowFlags.NoCollapse
            | ImGuiWindowFlags.NoResize
            | ImGuiWindowFlags.NoMove
            | ImGuiWindowFlags.NoBringToFrontOnFocus
            | ImGuiWindowFlags.NoNavFocus
            | ImGuiWindowFlags.NoBackground;

        if (ImGui.Begin("##MainDockspaceHost", flags))
        {
            uint dockspaceId = ImGui.GetID("MainDockspace");
            ImGui.DockSpace(dockspaceId, Vector2.Zero, ImGuiDockNodeFlags.PassthruCentralNode);
        }

        ImGui.End();
        ImGui.PopStyleVar(3);
    }

    private void RefreshSelectedWorldObjectInfo()
    {
        if (_worldScene == null)
            return;

        ObjectInstance? selected = _worldScene.SelectedInstance;
        if (!selected.HasValue)
        {
            _selectedObjectIndex = -1;
            _selectedObjectType = "";
            _selectedObjectInfo = "";
            return;
        }

        _selectedAreaPoiId = -1;

        ObjectInstance inst = selected.Value;
        string type = _worldScene.SelectedObjectType switch
        {
            Terrain.ObjectType.Wmo => "WMO",
            Terrain.ObjectType.WmoDoodad => "WMO Doodad",
            _ => "MDX"
        };
        int idx = _worldScene.SelectedObjectIndex;
        float wowX = WoWConstants.MapOrigin - inst.PlacementPosition.Y;
        float wowY = WoWConstants.MapOrigin - inst.PlacementPosition.X;
        float wowZ = inst.PlacementPosition.Z;

        _selectedObjectType = type;
        _selectedObjectIndex = idx;

        if (_useTabUi && _worldScene.SelectedObjectType is Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo or Terrain.ObjectType.WmoDoodad)
            _workbenchPanels.OpenWorkbenchTab(ModelBottomTab.Info);

        // WMO doodads come from MODD, which has no uniqueId — uniqueId identifies an MDDF/MODF
        // placement in an ADT. Showing the MODD table index under a "UniqueId" label asserts a
        // relationship that does not exist. Label it for what it is.
        string identityLine = _worldScene.SelectedObjectType == Terrain.ObjectType.WmoDoodad
            ? $"Doodad def: {inst.PlacementEntryIndex} (MODD index, WMO-local; MODD has no uniqueId)\n"
            : $"UniqueId: {inst.UniqueId}\n";

        _selectedObjectInfo = $"{type} [{idx}] {inst.ModelName}\n" +
            $"Path: {inst.ModelPath}\n" +
            identityLine +
            $"Local: ({inst.PlacementPosition.X:F1}, {inst.PlacementPosition.Y:F1}, {inst.PlacementPosition.Z:F1})\n" +
            $"WoW:   ({wowX:F1}, {wowY:F1}, {wowZ:F1})\n" +
            $"Rotation: ({inst.PlacementRotation.X:F1}, {inst.PlacementRotation.Y:F1}, {inst.PlacementRotation.Z:F1})\n" +
            $"Scale: {inst.PlacementScale:F3}\n" +
            $"BB: ({inst.BoundsMin.X:F1},{inst.BoundsMin.Y:F1},{inst.BoundsMin.Z:F1}) - ({inst.BoundsMax.X:F1},{inst.BoundsMax.Y:F1},{inst.BoundsMax.Z:F1})"
            // A placeholder box around the placement point is not the object's extent. Say which
            // one this is rather than letting the operator read a guess as a measurement.
            + (inst.BoundsResolved ? "" : "  [placeholder — model not loaded]");
    }

    private void ResetCamera()
    {
        // Reset to default free-fly position facing origin
        _camera.Position = new System.Numerics.Vector3(50f, 0f, 20f);
        _camera.Yaw = 180f;
        _camera.Pitch = -10f;
    }

    private void OnWindowResize(Vector2D<int> size)
    {
        _shellLayout.SyncImGuiWindowMetrics(size, _window.FramebufferSize);
    }

    private void OnResize(Vector2D<int> size)
    {
        _gl.Viewport(size);
        _shellLayout.SyncImGuiWindowMetrics(_window.Size, size);
    }

    private bool _disposed;

    private void ApplyLayoutObjectPreviewModeToScene()
    {
        if (_worldScene == null)
            return;

        if (_layoutObjectPreviewMode)
        {
            if (!_layoutObjectPreviewStateCaptured)
            {
                _layoutObjectPreviewSavedObjectsVisible = _worldScene.ObjectsVisible;
                _layoutObjectPreviewSavedWmosVisible = _worldScene.WmosVisible;
                _layoutObjectPreviewSavedDoodadsVisible = _worldScene.DoodadsVisible;
                _layoutObjectPreviewSavedVisibilityProfile = _worldScene.ObjectVisibilityProfile;
                _layoutObjectPreviewStateCaptured = true;
            }

            _worldScene.ObjectsVisible = true;
            _worldScene.WmosVisible = true;
            _worldScene.DoodadsVisible = false;
            _worldScene.ObjectVisibilityProfile = WorldObjectVisibilityProfile.Performance;
            return;
        }

        if (_layoutObjectPreviewStateCaptured)
        {
            _worldScene.ObjectsVisible = _layoutObjectPreviewSavedObjectsVisible;
            _worldScene.WmosVisible = _layoutObjectPreviewSavedWmosVisible;
            _worldScene.DoodadsVisible = _layoutObjectPreviewSavedDoodadsVisible;
            _worldScene.ObjectVisibilityProfile = _layoutObjectPreviewSavedVisibilityProfile;
            _layoutObjectPreviewStateCaptured = false;
        }
    }

    private void OnClose()
    {
        Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        StopVideoRecording("Stopped video recording during shutdown.");
        StopTaxiRideCamera();
        _mlTraining.ShutdownMlTrainingMonitor();

        ISceneRenderer? renderer = _renderer;
        WorldScene? worldScene = _worldScene;
        TerrainManager? terrainManager = _terrainManager;
        VlmTerrainManager? vlmTerrainManager = _vlmTerrainManager;

        _settings.SaveViewerSettings();

        _loadingScreen?.Dispose();
        _sceneCursorRenderer?.Dispose();
        _sceneCursorRenderer = null;
        _sceneClusterSelector3D?.Dispose();
        _sceneClusterSelector3D = null;
        _cameraHudRig?.Dispose();
        _cameraHudRig = null;
        _wdlPreviewCacheService?.Dispose();
        _wdlPreviewRenderer?.Dispose();
        _editorOverlayBb?.Dispose();
        _terrainAnalysisLocalTexture?.Dispose();
        _terrainAnalysisGlobalTexture?.Dispose();
        _terrainAnalysisAlphaTexture?.Dispose();
        _sqlPopulationService?.Dispose();
        if (!ReferenceEquals(renderer, worldScene)
            && !ReferenceEquals(renderer, terrainManager)
            && !ReferenceEquals(renderer, vlmTerrainManager))
        {
            renderer?.Dispose();
        }

        worldScene?.Dispose();
        if (worldScene == null)
            terrainManager?.Dispose();

        if (!ReferenceEquals(vlmTerrainManager, renderer))
            vlmTerrainManager?.Dispose();
        else if (worldScene == null)
            vlmTerrainManager?.Dispose();

        _minimapRenderer?.Dispose();
        _dataSource?.Dispose();
        if (_skyReady)
        {
            _gl.DeleteVertexArray(_skyVao);
            _gl.DeleteBuffer(_skyVbo);
            _gl.DeleteProgram(_skyShader);
        }
        _imGui?.Dispose();
        _input?.Dispose();
        _gl?.Dispose();
    }

    internal sealed class SavedPm4ObjectMatchSelection
    {
        public string MapName { get; set; } = "";
        public int TileX { get; set; }
        public int TileY { get; set; }
        public uint Ck24 { get; set; }
        public int ObjectPartId { get; set; }
        public string PlacementKind { get; set; } = "";
        public int PlacementUniqueId { get; set; }
        public int PlacementTileX { get; set; }
        public int PlacementTileY { get; set; }
        public string ModelName { get; set; } = "";
        public string ModelPath { get; set; } = "";
        public string EvidenceSource { get; set; } = "";
    }

    internal sealed class SavedObjectPathFilterMap
    {
        public string MapName { get; set; } = "";
        public bool Enabled { get; set; } = true;
        public List<SavedObjectPathFilterEntry> Filters { get; set; } = new();
    }

    internal sealed class SavedObjectPathFilterEntry
    {
        public string PathPrefix { get; set; } = "";
        public bool AppliesToWmo { get; set; }
        public bool AppliesToMdx { get; set; }
    }

    internal sealed class KnownGoodClientPath
    {
        public string Name { get; set; } = "";
        public string Path { get; set; } = "";
        public string? BuildVersion { get; set; }
    }
}
