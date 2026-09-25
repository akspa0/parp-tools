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
    private enum Pm4WorkbenchTab
    {
        Overlay,
        Selection,
        Correlation,
    }

    private enum WorkspaceMode
    {
        Viewer,
        Editor,
        Archaeology,
    }

    [Obsolete("Shell panel system deprecated in 069. Use tab system (View > Tab System). Will be removed in 070.")]
    private enum ShellPanelId
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



    private enum ShellPanelLane
    {
        Left,
        Right,
        Floating,
    }

    private enum EditorWorkspaceTask
    {
        Terrain,
        Objects,
        Pm4Evidence,
        Inspect,
        Publish,
    }

    private enum FixedBottomDrawerTab
    {
        Workspace,
        Terrain,
        Pm4,
        World,
        Diagnostics,
    }

    private readonly record struct ShellPanelDefinition(
        ShellPanelId Id,
        string WindowName,
        ShellPanelLane Lane,
        float DefaultWidth,
        float MinWidth,
        float CompactMinWidth,
        float MaxWidth);

    private sealed class SavedShellPanelLayout
    {
        public int PanelId { get; set; }
        public float NormalizedX { get; set; }
        public float NormalizedY { get; set; }
        public float NormalizedWidth { get; set; }
        public float NormalizedHeight { get; set; }
    }

    private const string ViewerProductTitle = "WoWViewer";
    private static readonly string ViewerDisplayVersion = GetViewerDisplayVersion();
    private static string ViewerProductName => $"{ViewerProductTitle} v{ViewerDisplayVersion}";
    private const string ViewerAboutPopupTitle = "About WoWViewer";
    private static readonly MethodInfo? ImGuiControllerWindowResizedMethod =
        typeof(ImGuiController).GetMethod("WindowResized", BindingFlags.Instance | BindingFlags.NonPublic);

    private enum ModelContainerKind
    {
        Unknown,
        Mdlx,
        Md20,
        Md21,
    }

    private IWindow _window = null!;
    private GL _gl = null!;
    private IInputContext _input = null!;
    private ImGuiController _imGui = null!;
    private readonly Lock _pendingImGuiMouseEventLock = new();
    private readonly Queue<(int ButtonIndex, bool Down)> _pendingImGuiMouseButtonEvents = new();
    private Camera _camera = new();
    private ISceneRenderer? _renderer;
    private Vector2D<int> _lastSyncedImGuiWindowSize;
    private Vector2D<int> _lastSyncedImGuiFramebufferSize;

    // Data source
    private IDataSource? _dataSource;
    private ReplaceableTextureResolver? _texResolver;
    private DBCD.Providers.IDBCProvider? _dbcProvider;
    private string? _dbdDir;
    private string? _dbcBuild;
    private static readonly WoWViewer.Terrain.ClientBuildOption[] FallbackClientBuildOptions =
    {
        new("Alpha (0.x) - 0.5.3.3368", "0.5.3.3368"),
        new("Alpha (0.x) - 0.7.0.3694", "0.7.0.3694"),
        new("Alpha (0.x) - 0.8.0.3734", "0.8.0.3734"),
        new("Alpha (0.x) - 0.9.0.3807", "0.9.0.3807"),
        new("Alpha (0.x) - 0.9.1.3810", "0.9.1.3810"),
        new("Alpha (0.x) - 0.10.3892", "0.10.3892"),
        new("Burning Crusade (2.x) - 2.4.3.8606", "2.4.3.8606"),
        new("Wrath (3.x) - 3.0.1.8303", "3.0.1.8303"),
        new("Wrath (3.x) - 3.3.5.12340", "3.3.5.12340"),
        new("Cataclysm (4.x) - 4.0.0.11927", "4.0.0.11927"),
        new("Cataclysm (4.x) - 4.0.1.12304", "4.0.1.12304")
    };
    private const float MaxTerrainFogDistance = 20000f;
    private const float MinTerrainFarPlane = 1f;
    // Keep the WDL horizon visible well past the LIT/DBC fog endpoint.  FogEnd
    // remains the full-detail/visibility authority; this is projection room for
    // the low-detail WDL replacement terrain, not a second fog range.
    private const float TerrainFarPlanePadding = 2500f;
    private const float MaxTerrainFarPlane = MaxTerrainFogDistance + TerrainFarPlanePadding;

    private readonly List<WoWViewer.Terrain.ClientBuildOption> _clientBuildOptions = new();
    private string? _lastVirtualPath; // Virtual path of last loaded file (for DBC lookup)
    private string _statusMessage = "No data source loaded. Use File > Open Game Folder (MPQ) first, then Open File for standalone assets.";
    private AreaTableService? _areaTableService;
    private string _currentAreaName = "";
    private string _currentZoneName = "";
    private WowViewer.Core.World.AreaLookupResult? _currentAreaLookup;
    private Vector3 _lastAreaLookupCameraPosition = new(float.NaN);
    private int _areaLookupTick;
    private int _lastAreaLookupLoadedTileCount = -1;
    private int _lastAreaLookupMapId = int.MinValue;
    private TerrainRenderer? _areaOverlayRenderer;
    private AreaTableService? _areaOverlayAreaTableService;
    private int _areaOverlayRevision = int.MinValue;
    private int _areaOverlayMapId = int.MinValue;
    private int _currentMapId = -1; // MapID of the currently loaded world
    private string? _lastWorldSceneWdtPath;
    private Vector3 _lastWorldSceneCameraPosition;
    private float _lastWorldSceneCameraYaw = 180f;
    private float _lastWorldSceneCameraPitch = -20f;
    private string? _pendingDataSourceWorldReloadVirtualPath;
    private string? _pendingDataSourceWorldReloadLocalPath;
    private Vector3? _pendingDataSourceWorldReloadCameraPosition;
    private float _pendingDataSourceWorldReloadCameraYaw = 180f;
    private float _pendingDataSourceWorldReloadCameraPitch = -20f;
    private int _activeDataSourceReloadGeneration;
    private int _pendingDataSourceReloadGeneration;
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
    private string _wdlPreviewWarmupStatus = string.Empty;
    private float _minimapZoom = 4f; // Number of tiles visible in each direction from camera
    private bool _fullscreenMinimap = false; // M key toggles fullscreen minimap
    private Vector2 _minimapPanOffset = Vector2.Zero; // Pan offset for click-and-drag
    private bool _minimapDragging = false;
    private (int tileX, int tileY)? _pendingMinimapTeleportTile;
    private int _pendingMinimapTeleportClickCount;
    private DateTime _pendingMinimapTeleportLastClickUtc = DateTime.MinValue;
    private Rendering.LoadingScreen? _loadingScreen;

    // Output directories (next to the executable)
    private static readonly string OutputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");
    private static readonly string CacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache");
    internal static readonly string ExportDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "export");
    private static readonly string ProjectsDir = Path.Combine(OutputDir, "projects");
    private static readonly string SettingsDir = Path.Combine(OutputDir, "settings");
    private static readonly string ViewerSettingsPath = Path.Combine(SettingsDir, "viewer_settings.json");
    private const int CurrentShellPanelLayoutVersion = 4;
    private const int CurrentWorkbenchNavigationVersion = 4;
    private const int MinimapTeleportConfirmClicks = 3;

    // File browser state
    private List<string> _filteredFiles = new();
    private string _searchFilter = "";
    private string _extensionFilter = ".mdx";
    private static readonly string[] EarlyModelBrowserExtensions = { ".mdx", ".mdl" };
    private int _selectedFileIndex = -1;
    private string? _loadedFilePath;
    private string? _loadedFileName;

    // Model info
    private string _modelInfo = "";
    private string? _standaloneCharacterCustomizationModelPath;
    private readonly List<int> _standaloneCharacterHairVariationIds = new();
    private readonly List<int> _standaloneCharacterFacialHairVariationIds = new();
    private int _standaloneCharacterHairVariationOverride = -1;
    private int _standaloneCharacterFacialHairVariationOverride = -1;
    private bool _preserveStandaloneCharacterCustomizationOnNextLoad;
    private readonly Dictionary<string, string?> _standaloneSkinPathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _loggedStandaloneMissingSkinPaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _reportedAreaDiagnostics = new(StringComparer.Ordinal);
    
    // Stored loaded model data for export (avoids re-parsing from disk)
    private WmoV14ToV17Converter.WmoV14Data? _loadedWmo;

    private static string GetViewerDisplayVersion()
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
    private bool _showTerrainControls = false;
    private bool _showWorkspaceBarsPanel = true;
    private bool _hideUiChrome;
    private bool _showLogViewer = false;
    private bool _showMinimapWindow = false;
    private bool _showPerfWindow = false;
    private bool _openAboutPopup;
    private WorkspaceMode _workspaceMode = WorkspaceMode.Viewer;
    private EditorWorkspaceTask _editorWorkspaceTask = EditorWorkspaceTask.Terrain;
    private FixedBottomDrawerTab _activeBottomDrawerTab = FixedBottomDrawerTab.Workspace;
    private FixedBottomDrawerTab? _pendingRightSidebarSection;
    private bool _useDockspaceUi = true;

    // 069 Phase 1: tab system state. On by default; can toggle off via View > Legacy Sidebar UI.
    private bool _useTabUi = true;
    private WorkbenchTab _activeTopTab = WorkbenchTab.Quick;
    private int _activeBottomTabIndex = 0;

    // Experimental pages retain their existing internal selectors while the
    // top-level destination owns the only visible category switch.
    private int _activeArcheologyTabIndex = 0;
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
    private double _archeologyPlaybackAccumulator = 0.0; // for fractional uniqueId advancement
    private int _archeologyPlaybackRestoreMin = -1; // saved on Play, restored on Stop
    private int _archeologyPlaybackRestoreMax = -1;
    private bool _archeologyPlaybackRestoreFilter = false;

    // 069 Phase 7: capture integration flags
    private bool _archeologyApplyToNextCapture = false;
    private bool _archeologyApplyToVideoRecording = false;
    private bool _autoOpenWorldMapsPanel;
    private Vector2 _dockspaceHostPosition;
    private Vector2 _dockspaceHostSize;
    private AssetCatalogView? _catalogView;
    private bool _wantOpenFile = false;
    private bool _wantAttachLooseMapFolder = false;
    private bool _wantOpenWdtFile = false;
    private bool _wantOpenPm4File = false;
    private bool _wantExportGlb = false;
    private bool _wantExportGlbCollision = false;
    private bool _wantExportMapGlbTiles = false;
    private string _projectOutputRootDir = ProjectsDir;
    private string _editorProjectOutputDir = string.Empty;
    private string _editorProjectSourceKey = string.Empty;
    private string? _selectedPlacementSaveTargetPath;
    private string _selectedPlacementSaveStatus = "Select a tile-backed world object to stage a translation-only save.";

    private struct DockPanelState
    {
        public bool Visible;
        public bool IsDocked;
        public Vector2 Position;
        public Vector2 Size;
    }

    private static readonly ShellPanelDefinition[] ShellPanelDefinitions =
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

    private static readonly ShellPanelId[] TopLeftQuadrantPanels = { ShellPanelId.Navigator };
    private static readonly ShellPanelId[] TopRightQuadrantPanels = { ShellPanelId.Inspector, ShellPanelId.WorldObjects, ShellPanelId.ModelInfo, ShellPanelId.RuntimeStats };
    private static readonly ShellPanelId[] BottomRightQuadrantPanels = { ShellPanelId.Pm4Workbench, ShellPanelId.Pm4Info, ShellPanelId.TerrainControls, ShellPanelId.Pm4SceneGraph };
    private static readonly ShellPanelId[] BottomLeftQuadrantPanels = { ShellPanelId.Minimap };

    private DockPanelState _navigatorDockState;
    private DockPanelState _inspectorDockState;
    private DockPanelState _pm4WorkbenchDockState;
    private DockPanelState _terrainControlsDockState;
    private DockPanelState _runtimeStatsDockState;
    private DockPanelState _worldObjectsDockState;
    private DockPanelState _modelInfoDockState;
    private DockPanelState _minimapDockState;
    private DockPanelState _workspaceBarsDockState;
    private DockPanelState _pm4InfoDockState;
    private DockPanelState _pm4SceneGraphDockState;
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
    private (int tileX, int tileY)? _terrainWorkbenchFocusedTile;
    private bool _terrainWorkbenchTileSelectionActive;
    private bool _terrainWorkbenchMapPanActive;
    private Vector2 _terrainWorkbenchMapDragStart;
    private Vector2 _terrainWorkbenchMapPanOrigin;
    private (int tileX, int tileY)? _terrainWorkbenchTileSelectionAnchor;
    private bool _terrainWorkbenchChunkSelectionActive;
    private (int chunkX, int chunkY)? _terrainWorkbenchChunkSelectionAnchor;

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
    private const float TerrainWeakSignalRestoreDefaultMinZ = -10f;
    private const float TerrainWeakSignalRestoreDefaultMaxZ = 10f;
    internal const float TerrainWeakSignalRestoreMaxFactor = 512f;
    private bool _terrainWeakSignalRestoreEnabled;
    private bool _terrainWeakSignalRestoreAllLoadedTiles = true;
    private bool _terrainWeakSignalRestoreUseTextureSubdivisions = true;
    private bool _terrainWeakSignalRestoreUseAutoFactor = true;
    private float _terrainWeakSignalRestoreManualFactor = 16f;
    private float _terrainWeakSignalRestoreCandidateMinHeight = TerrainWeakSignalRestoreDefaultMinZ;
    private float _terrainWeakSignalRestoreCandidateMaxHeight = TerrainWeakSignalRestoreDefaultMaxZ;
    private string _terrainWeakSignalRestoreStatus = string.Empty;
    private readonly Dictionary<(int tileX, int tileY), WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyTileAnalysis> _stratigraphyTileAnalyses = new();
    private bool _stratigraphyUnhideDevMeshes = true;
    private bool _stratigraphyStitchBoundaries = true;
    private bool _stratigraphyPreserveNegativeFloor = true;
    private bool _stratigraphyPolarityInverted = false;
    private WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode _stratigraphyAnchorMode = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode.LowestZ_Floor;
    private bool _stratigraphyUseNeighborAutoFit = false;
    private bool _stratigraphyUseWdlMagnetization = false;
    private float _stratigraphyWdlMagnetizationStrength = 1.0f;
    private string _stratigraphySaveOutputDirectory = string.Empty;
    private (int tileX, int tileY)? _terrainAnalysisPreviewTile;
    private float _terrainAnalysisPreviewTileMin;
    private float _terrainAnalysisPreviewTileMax;
    private float _terrainAnalysisPreviewVisibilityRatio;
    private float _terrainAnalysisPreviewAmplification = 1f;
    private (int tileX, int tileY)? _terrainAnalysisPreviewCompareTile;
    private float? _terrainAnalysisPreviewSimilarity;
    private float _terrainAnalysisGlobalMin;
    private float _terrainAnalysisGlobalMax;
    private int _terrainAnalysisGlobalTileCount;
    private TerrainTileScope _terrainAnalysisGlobalScope = TerrainTileScope.LoadedTiles;
    private bool _terrainAnalysisHasGlobalBounds;
    private bool _terrainAnalysisFollowCameraTile = true;
    private string _terrainAnalysisStatus = string.Empty;
    private int _terrainAnalysisHiddenCompareOffsetX;
    private int _terrainAnalysisHiddenCompareOffsetY = 2;
    private float _terrainAnalysisHiddenMinSimilarity = 0.85f;
    private float _terrainAnalysisHiddenMaxVisibilityRatio = 0.05f;
    private int _terrainAnalysisHiddenMaxResults = 24;
    private TerrainTileScope _terrainAnalysisHiddenScope = TerrainTileScope.LoadedTiles;
    private readonly List<TerrainHiddenTileCandidate> _terrainAnalysisHiddenCandidates = new();
    private int _terrainAnalysisHiddenSelectedIndex = -1;
    private string _terrainAnalysisHiddenStatus = string.Empty;
    private Terrain.BoundingBoxRenderer? _editorOverlayBb;
    private bool _standaloneWmoGroupOverlayEnabled = true;
    private bool _standaloneWmoGroupLabelsAllEnabled = true;
    private bool _standaloneWmoOverlayIncludeHiddenGroups = true;
    private int _hoveredStandaloneWmoGroupIndex = -1;
    private int _selectedStandaloneWmoGroupIndex = -1;
    private int _selectedStandaloneWmoDoodadIndex = -1;
    private int _selectedWorldWmoDoodadIndex = -1;
    private int _standaloneWmoDoodadGroupFilter = -1;
    private int _worldWmoDoodadGroupFilter = -1;
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
    private const float DefaultSidebarWidth = 360f;
    private const float DefaultRightSidebarWidth = 480f;
    private const float SidebarMinWidth = 280f;
    private const float SidebarCompactMinWidth = 240f;
    private const float SidebarMaxWidth = 1080f;
    private const float SidebarSplitterWidth = 8f;
    private const float DefaultBottomDrawerHeight = 280f;
    private const float BottomDrawerMinHeight = 220f;
    private const float BottomDrawerCompactMinHeight = 160f;
    private const float BottomDrawerMaxHeight = 520f;
    private const float BottomDrawerSplitterHeight = 8f;
    private const float SceneViewportPreferredMinWidth = 420f;
    private const float SceneViewportHardMinWidth = 240f;
    private const float SceneViewportPreferredMinHeight = 280f;
    private const float SceneViewportHardMinHeight = 160f;
    private float _leftSidebarWidth = DefaultSidebarWidth;
    private float _rightSidebarWidth = DefaultRightSidebarWidth;
    private float _bottomDrawerHeight = DefaultBottomDrawerHeight;
    private bool _suppressLeftSidebarForLayout;
    private bool _suppressRightSidebarForLayout;
    private bool _suppressMinimapForLayout;
    private const float MenuBarHeight = 22f;
    private const float ToolbarHeight = 32f;
    private const float BottomBarHeight = 36f;
    private const float StatusBarHeight = 24f;

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
    private bool _wantOpenVlmProject = false;
    private bool _wantOpenZarrDataset = false;

    // Object picking state
    private int _selectedObjectIndex = -1; // -1=none, 0..modf-1=WMO, modf..modf+mddf-1=MDX
    private string _selectedObjectType = "";
    private string _selectedObjectInfo = "";
        private int _selectedAreaPoiId = -1;
    private const float TaxiNodePickHalfWidth = 42f;
    private const float TaxiNodePickBottomPadding = 18f;
    private const float TaxiNodePickTopPadding = 96f;
    private const float TaxiRouteHandlePickHalfWidth = 40f;
    private const float TaxiRouteHandlePickBottomPadding = 20f;
    private const float TaxiRouteHandlePickTopPadding = 72f;
    private const float TaxiRouteSegmentPickHalfWidth = 28f;
    private string _taxiActorModelOverrideInput = "";
    private int _taxiActorModelOverrideInputRouteId = -1;
    private int _taxiActorModelOverrideTargetRouteId = -1;
    private string _objectPathFilterInput = "";
    private bool _objectPathFilterInputAppliesToWmo = true;
    private bool _objectPathFilterInputAppliesToMdx = true;
    private string _taxiRouteFilter = "";
    private int _taxiRouteListGroupingMode = 1;
    private bool _layoutObjectPreviewMode;
    private bool _layoutObjectPreviewStateCaptured;
    private bool _layoutObjectPreviewSavedObjectsVisible = true;
    private bool _layoutObjectPreviewSavedWmosVisible = true;
    private bool _layoutObjectPreviewSavedDoodadsVisible = true;
    private WorldObjectVisibilityProfile _layoutObjectPreviewSavedVisibilityProfile = WorldObjectVisibilityProfile.Performance;
    private string _sqlAlphaCoreRoot = "";
    private SqlWorldPopulationService? _sqlPopulationService;
    private bool _sqlIncludeCreatures = true;
    private bool _sqlIncludeGameObjects = true;
    private int _sqlMaxSpawns = 2000;
    private float _sqlGameObjectMdxScaleMultiplier = 1.0f;
    private bool _sqlUseAoiFilter = true;
    private int _sqlAoiTileRadius = 3;
    private bool _sqlStreamWithCamera = true;
    private string _sqlSpawnStatus = "Not loaded";
    private string _sqlServiceRoot = "";
    private List<WorldSpawnRecord>? _sqlMapSpawnsCache;
    private int _sqlMapSpawnsCacheMapId = -1;
    private (int tileX, int tileY)? _sqlLastCameraTile;
    private bool _sqlForceStreamRefresh;
    private string _wlLayerSelectedBodyKey = "";
    private bool _wlLayerListIsolationEnabled;
    private bool _wlPendingScrollToSelectedBody;
    private Vector3 _pm4SavedOverlayTranslation = Vector3.Zero;
    private Vector3 _pm4SavedOverlayRotationDegrees = Vector3.Zero;
    private Vector3 _pm4SavedOverlayScale = Vector3.One;
    private float _pm4TranslationStepUnits = 10f;
    private float _pm4RotationStepDegrees = 90f;
    private float _pm4ScaleStepUnits = 0.1f;
    private ShellPanelId? _pendingFocusedShellPanel;
    private Pm4WorkbenchTab? _pendingPm4WorkbenchTab;
    private Pm4ObjectMatchReport? _pm4ObjectMatchReport;
    private Pm4ObjectMatchObject? _selectedPm4ObjectMatch;
    private (int tileX, int tileY, uint ck24, int objectPart)? _selectedPm4ObjectMatchKey;
    private int _selectedPm4ObjectMatchCacheMaxMatches = -1;
    private Pm4ObjectMatchObject? _hoveredPm4ObjectMatch;
    private (int tileX, int tileY, uint ck24, int objectPart)? _hoveredPm4ObjectMatchKey;
    private int _hoveredPm4ObjectMatchCacheMaxMatches = -1;
    private readonly List<(int tileX, int tileY, uint ck24, int objectPart)> _pm4ObjectCollection = new();
    private int _pm4ObjectMatchMaxMatchesPerObject = 5;
    private int _selectedPm4ObjectMatchObjectIndex = -1;
    private int _selectedPm4ObjectMatchCandidateIndex;
    private readonly Dictionary<string, SavedPm4ObjectMatchSelection> _savedPm4ObjectMatches = new(StringComparer.OrdinalIgnoreCase);
    private Pm4WmoMatchResult? _pm4WmoGroupMatchResult;
    private Pm4WmoMatchStore? _pm4WmoMatchStore;
    private Dictionary<string, Pm4WmoMatchEntry> _pm4WmoMatchEntries = new(StringComparer.OrdinalIgnoreCase);
    private string _pm4WmoMatchStatus = "";
    private Pm4WmoCorrelationReport? _pm4WmoCorrelationReport;
    private int _pm4WmoCorrelationMaxMatchesPerPlacement = 8;
    private int _selectedPm4WmoCorrelationPlacementIndex = -1;
    private int _selectedPm4WmoCorrelationMatchIndex;
    private bool _pm4WmoCorrelationNearOnly = true;
    private string _pm4WmoCorrelationModelFilter = string.Empty;
    private bool _showCaptureAutomationWindow = false;
    private bool _showCameraPathWindow;
    private bool _showUniqueIdArchaeologyWindow;
    private bool _showWeakSignalWindow;
    private bool _showPm4SceneGraph = true;
    private string _pm4SceneFilter = "";

    // Camera speed (adjustable via UI)
    private float _cameraSpeed = 50f;
    // Field of view in degrees (adjustable via UI)
    private float _fovDegrees = 45f;
    private int _savedDetailedAdtTileCountOverride;

    private bool _autoFrameModelOnLoad = true;
    private static readonly string[] WmoLiquidRotationLabels = { "0°", "90°", "180°", "270°" };
    private bool _hasExplicitWmoMliqRotationOverride;

    // Sky gradient for standalone model viewing
    private uint _skyVao, _skyVbo, _skyShader;
    private bool _skyReady;

    // Folder dialog workaround (ImGui doesn't have native dialogs)
    private bool _showFolderInput = false;
    private string _folderInputBuf = "";
    private bool _showBuildSelectionDialog;
    private string? _pendingGameFolderPath;
    private int _selectedBuildOptionIndex;
    private string _buildSelectionFilter = "";
    private string? _buildSelectionHint;
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
    private string _synthesizedMinimapClientRoot = string.Empty;
    private string _synthesizedMinimapMapName = string.Empty;
    private string _synthesizedMinimapOutputDirectory = string.Empty;
    private float _synthesizedMinimapTimeHours = 12f;
    private int _synthesizedMinimapHour = 12;
    private int _synthesizedMinimapMinute;
    private int _synthesizedMinimapResolution = 256;
    private bool _synthesizedMinimapEmitTiles = true;
    private bool _synthesizedMinimapEmitWholeMap = true;
    private bool _synthesizedMinimapIncludeWmos;
    private bool _synthesizedMinimapBakeMcsh;
    private bool _synthesizedMinimapCastShadows = true;
    private bool _synthesizedMinimapRunning;
    private bool _synthesizedMinimapDone;
    private string? _synthesizedMinimapError;
    private readonly List<string> _synthesizedMinimapLog = new();
    private bool _synthesizedMinimapScrollToBottom;

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

    public ViewerApp()
    {
        _converterDialogs = new ConverterDialogsService(this);
        _datasetExportDialogs = new DatasetExportDialogsService(this);
        _terrainWeakSignalRestore = new TerrainWeakSignalRestoreService(this);
        _terrainTileIo = new TerrainTileIoService(this);
        _chunkEdit = new ChunkEditService(this);
        _placementEditing = new PlacementEditService(this);
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
    string IViewerAppHost.GetProjectOutputRootDirectory() => GetProjectOutputRootDirectory();
    void IViewerAppHost.HandleProjectOutputRootChanged() => HandleProjectOutputRootChanged();
    void IViewerAppHost.LoadWdtTerrain(string wdtPath) => LoadWdtTerrain(wdtPath);
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
    void IViewerAppHost.LoadVlmProject(string projectRoot) => LoadVlmProject(projectRoot);
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
    string? IViewerAppHost.GetCurrentSessionMapName() => GetCurrentSessionMapName();
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
    string IViewerAppHost.EnsureEditorProjectOutputDirectory(bool forceNew) => EnsureEditorProjectOutputDirectory(forceNew);
    string IViewerAppHost.GetEditorProjectName(string? fallbackName) => GetEditorProjectName(fallbackName);
    string? IViewerAppHost.GetEditorProjectSourceKey() => GetEditorProjectSourceKey();
    bool IViewerAppHost.TryPickTerrainChunkUnderMouse(TerrainRenderer renderer, out TerrainRenderer.TerrainChunkInfo info) => TryPickTerrainChunkUnderMouse(renderer, out info);
    ref string IViewerAppHost.EditorProjectOutputDir => ref _editorProjectOutputDir;
    ref string IViewerAppHost.SelectedPlacementSaveStatus => ref _selectedPlacementSaveStatus;
    ref string? IViewerAppHost.SelectedPlacementSaveTargetPath => ref _selectedPlacementSaveTargetPath;
    ref WorldScene? IViewerAppHost.WorldScene => ref _worldScene;
    void IViewerAppHost.RefreshSelectedWorldObjectInfo() => RefreshSelectedWorldObjectInfo();
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
        SyncImGuiWindowMetrics(_window.Size, _window.FramebufferSize);
        ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.DockingEnable;

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Enable(EnableCap.CullFace);

        _loadingScreen = new Rendering.LoadingScreen(_gl);
        _sceneCursorRenderer = new SceneCursorRenderer(_gl, _dataSource, _texResolver);
        _sceneClusterSelector3D = new SceneClusterSelector3D(_gl);
        _cameraHudRig = new CameraHudRig(_gl);

        TryAutoPopulateAlphaCoreRoot();
        LoadViewerSettings();
        ApplyActiveUiTheme();
        LoadCameraShotPoints();
        DetectRenderQualityCapabilities();
        ApplyRenderQualitySettings(refreshTextures: false);

        // Mouse input for viewport (not consumed by ImGui)
        foreach (var mouse in _input.Mice)
        {
            mouse.MouseDown += (_, btn) =>
            {
                QueueImGuiMouseButtonEvent(btn, down: true);

                if (btn == MouseButton.Right && CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
                    _mouseDown = true;
                if (btn == MouseButton.Left && CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
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
                        PickObjectAtMouse(_lastMouseX, _lastMouseY, addPm4ToCollection: shift);
                    else if (terrainRenderer != null && !shift && !_chunkToolEnabled
                        && TryPickTerrainChunkUnderMouse(terrainRenderer, out var terrainChunk))
                        SelectTerrainChunkFromClick(terrainChunk);
                }
            };
            mouse.MouseUp += (_, btn) =>
            {
                QueueImGuiMouseButtonEvent(btn, down: false);

                if (btn == MouseButton.Right) _mouseDown = false;
            };
            mouse.MouseMove += (_, pos) =>
            {
                float dx = pos.X - _lastMouseX;
                float dy = pos.Y - _lastMouseY;
                _lastMouseX = pos.X;
                _lastMouseY = pos.Y;

                if (_mouseDown && !IsSceneMouseCaptureBlocked(_lastMouseX, _lastMouseY))
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

    private void TryAutoPopulateAlphaCoreRoot()
    {
        if (!string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
            return;

        string[] candidates =
        {
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "external", "alpha-core")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "external", "alpha-core")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "external", "alpha-core"))
        };

        foreach (var candidate in candidates)
        {
            string worldDir = Path.Combine(candidate, "etc", "databases", "world");
            string dbcDir = Path.Combine(candidate, "etc", "databases", "dbc");
            if (Directory.Exists(worldDir) && Directory.Exists(dbcDir))
            {
                _sqlAlphaCoreRoot = candidate;
                _sqlSpawnStatus = $"Auto-detected alpha-core SQL root: {candidate}";
                return;
            }
        }
    }

    private void DrawSelectedSqlGameObjectAnimationControls()
    {
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return;
        if (_worldScene.SelectedObjectType != Terrain.ObjectType.Mdx)
            return;
        if (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId)
            return;

        var inst = _worldScene.SelectedInstance.Value;
        var spawn = _sqlMapSpawnsCache.FirstOrDefault(s =>
            s.SpawnType == WorldSpawnType.GameObject &&
            s.SpawnId == inst.UniqueId &&
            (string.IsNullOrEmpty(s.ModelPath) || string.Equals(Path.GetFileName(s.ModelPath), inst.ModelName, StringComparison.OrdinalIgnoreCase)));
        if (spawn == null)
            return;

        var mdxRenderer = _worldScene.Assets.GetMdx(inst.ModelKey);
        var animator = mdxRenderer?.Animator;

        ImGui.Separator();
        ImGui.TextColored(new Vector4(0.85f, 1f, 0.85f, 1f), "SQL GameObject Animation");
        ImGui.TextDisabled($"SpawnId: {spawn.SpawnId}  Entry: {spawn.EntryId}  Type: {spawn.GameObjectType}");

        if (animator == null || !animator.HasAnimation || animator.Sequences.Count == 0)
        {
            ImGui.TextDisabled("This gameobject model has no animation sequences.");
            return;
        }

        int currentSeq = animator.CurrentSequence;
        string currentSeqName = currentSeq >= 0 && currentSeq < animator.Sequences.Count
            ? animator.Sequences[currentSeq].Name
            : "None";
        if (string.IsNullOrWhiteSpace(currentSeqName))
            currentSeqName = $"Sequence {currentSeq}";

        if (ImGui.BeginCombo("##sqlgo_anim_seq", currentSeqName))
        {
            for (int s = 0; s < animator.Sequences.Count; s++)
            {
                bool selected = s == currentSeq;
                string seqName = animator.Sequences[s].Name;
                if (string.IsNullOrWhiteSpace(seqName))
                    seqName = $"Sequence {s}";
                if (ImGui.Selectable(seqName, selected))
                    animator.SetSequence(s);
                if (selected) ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

var seq = animator.Sequences[animator.CurrentSequence];
        float seqStart = seq.Time.Start;
        float seqEnd = seq.Time.End;

        bool isPlaying = animator.IsPlaying;
        if (ImGui.Button(isPlaying ? "Pause GO Anim" : "Play GO Anim"))
            animator.IsPlaying = !isPlaying;

        ImGui.SameLine();
        if (ImGui.Button("Stop GO Anim"))
        {
            animator.IsPlaying = false;
            animator.CurrentFrame = seqStart;
        }

        ImGui.SameLine();
        if (ImGui.Button("Prev Key"))
        {
            animator.IsPlaying = false;
            animator.StepToPrevKeyframe();
        }

        ImGui.SameLine();
        if (ImGui.Button("Next Key"))
        {
            animator.IsPlaying = false;
            animator.StepToNextKeyframe();
        }

        float currentFrame = Math.Clamp(animator.CurrentFrame, seqStart, seqEnd);
        if (ImGui.SliderFloat("GO Frame", ref currentFrame, seqStart, seqEnd, "%.0f"))
        {
            animator.IsPlaying = false;
            animator.CurrentFrame = currentFrame;
        }

        ImGui.SameLine();
        if (ImGui.Button("Export JSON##GO"))
            ExportAnimationStateJson(animator, currentSeq, currentSeqName, seqStart, seqEnd);

        ImGui.TextDisabled("Note: this affects all visible instances using the same MDX model renderer.");
    }

    private void OnUpdate(double dt)
    {
        SyncImGuiWindowMetrics(_window.Size, _window.FramebufferSize);
        _imGui.Update((float)dt);
        FlushPendingImGuiMouseButtonEvents();
        HandleSceneMouseWheelInput();
        HandleKeyboardInput((float)dt);
        UpdateCameraPathPlayback(dt);
        UpdateCameraPathPreload();
        UpdateTaxiRideCamera();
        UpdateArcheologyPlayback(dt);
        _minimapRenderer?.ProcessPendingLoads(
            maxLoads: (_fullscreenMinimap || _showMinimapWindow) ? 4 : 1,
            maxBudgetMs: (_fullscreenMinimap || _showMinimapWindow) ? 6.0 : 1.5);
        UpdateSqlSpawnStreaming();
        _terrainWeakSignalRestore.UpdateTerrainWeakSignalRestoreForCamera();
    }

    private void UpdateArcheologyPlayback(double dt)
    {
        if (!_archeologyPlaybackActive)
            return;

        if (_worldScene == null)
        {
            _archeologyPlaybackActive = false;
            _archeologyPlaybackAccumulator = 0;
            _statusMessage = "Archeology playback stopped because the world was unloaded.";
            return;
        }

        if (!_worldScene.TryGetUniqueIdFilterRange(out int minId, out int maxId, out _))
        {
            _archeologyPlaybackActive = false;
            _archeologyPlaybackAccumulator = 0;
            _statusMessage = "Archeology playback stopped because no scoped UniqueId range is available.";
            return;
        }

        _archeologyPlaybackAccumulator += dt * _archeologyPlaybackSpeed;
        int advance = (int)Math.Floor(_archeologyPlaybackAccumulator);
        if (advance <= 0) return;
        _archeologyPlaybackAccumulator -= advance;

        int currentMax = _worldScene.UniqueIdFilterMax;
        int newMax = currentMax + advance;
        if (newMax >= maxId)
        {
            if (_archeologyPlaybackLoop)
            {
                // Loop: snap back to min
                int restoreMin = _archeologyPlaybackRestoreMin >= 0 ? _archeologyPlaybackRestoreMin : minId;
                _worldScene.SetUniqueIdFilterRange(restoreMin, restoreMin);
                _archeologyPlaybackAccumulator = 0;
            }
            else
            {
                _worldScene.UniqueIdFilterMax = maxId;
                _archeologyPlaybackActive = false;
                _archeologyPlaybackAccumulator = 0;
                _statusMessage = "Archeology playback reached end of range.";
            }
        }
        else
        {
            _worldScene.UniqueIdFilterMax = newMax;
        }
    }

    private void UpdateSqlSpawnStreaming()
    {
        if (_worldScene == null || !_sqlStreamWithCamera || !_sqlUseAoiFilter)
            return;

        if (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId)
            return;

        var camTile = GetCameraTile();
        if (_sqlForceStreamRefresh || _sqlLastCameraTile == null || _sqlLastCameraTile.Value != camTile)
        {
            _sqlLastCameraTile = camTile;
            ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: false);
            _sqlForceStreamRefresh = false;
        }
    }

    private (int tileX, int tileY) GetCameraTile()
    {
        int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
        return (tileX, tileY);
    }

    private void ResetSqlSpawnStreamingState(bool clearSceneSpawns)
    {
        _sqlMapSpawnsCache = null;
        _sqlMapSpawnsCacheMapId = -1;
        _sqlLastCameraTile = null;
        _sqlForceStreamRefresh = false;
        if (clearSceneSpawns && _worldScene != null)
            _worldScene.ClearExternalSpawns();
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

        if (!CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
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
                ClearPendingClickSelection();
            }
            else if (_worldScene != null)
            {
                ClearPendingClickSelection();
                ClearSelectedWlLiquidBody(clearListIsolation: true);
                _worldScene.ClearSelection();
                _worldScene.ClearTaxiSelection();
                _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
                ClearSelectedAreaPoiInfo();
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
                FocusShellPanel(ShellPanelId.Inspector);
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
        bool hasSceneViewportRect = TryGetSceneViewportRect(out float sceneViewportX, out float sceneViewportY, out float sceneViewportWidth, out float sceneViewportHeight);
        int sceneFramebufferX = 0;
        int sceneFramebufferY = 0;
        uint sceneFramebufferWidth = 0;
        uint sceneFramebufferHeight = 0;
        bool hasSceneViewport = hasSceneViewportRect
            && TryGetSceneFramebufferViewport(out sceneFramebufferX, out sceneFramebufferY, out sceneFramebufferWidth, out sceneFramebufferHeight);
        if (hasSceneViewport)
            _gl.Viewport(sceneFramebufferX, sceneFramebufferY, sceneFramebufferWidth, sceneFramebufferHeight);
        else
            _gl.Viewport(_window.FramebufferSize);

        float aspect = hasSceneViewport
            ? sceneViewportWidth / Math.Max(sceneViewportHeight, 1f)
            : (float)size.X / Math.Max(size.Y, 1);
        float farPlane = GetSceneFarPlane();
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
                UpdateWorldSceneWireframeReveal(view, proj);

            if (_worldScene != null)
                UpdateWorldSceneHoveredAssetInfo(view, proj);

            // Update native-style ZoneText/SubzoneText from the resident chunk metadata under the
            // camera. Batched terrain owns one GPU mesh per tile, so the area lookup must use the
            // resident chunk-info index instead of the legacy per-chunk GPU mesh list.
            var areaChunkRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            UpdateCurrentAreaContext(areaChunkRenderer);
            _worldScene?.SetCurrentAreaLookup(_currentAreaLookup);
            UpdateAreaOverlay(areaChunkRenderer);

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
                DrawStandaloneWmoGroupOverlay(wmoR, view, proj, sceneViewportX, sceneViewportY, sceneViewportWidth, sceneViewportHeight);
            }
            else
            {
                // WorldScene / VLM terrain — handles its own lighting
                _renderer.Render(view, proj);
                DrawEditorOverlays(view, proj);
                if (hasSceneViewportRect)
                {
                    DrawAreaOverlayLabels(
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
            RenderSceneClusterSelector3D(proj);
        }

        if (hasSceneViewport)
            _gl.Viewport(_window.FramebufferSize);

        CaptureVideoFrameIfNeeded(includeUi: false, dt);
        CompleteCaptureIfReady(includeUi: false);

        // Render ImGui overlay when the native ImGui context is live. Startup capture and
        // teardown can briefly produce frames where the controller still exists but the
        // underlying context is not available.
        if (HasImGuiContext())
        {
            bool hideHardwareCursor = _sceneCursorRenderer != null
                && _sceneCursorRenderer.Style != CursorStyle.ClassicOSArrow
                && CanSceneConsumeMouse(_lastMouseX, _lastMouseY);

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

            RenderSceneCursor(view, proj, sceneViewportX, sceneViewportY, sceneViewportWidth, sceneViewportHeight);

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
        if (!HasImGuiContext())
            return;

        UpdateShellLayout(ImGui.GetIO().DisplaySize);

        ResetDockPanelStates();

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
            DrawMenuBar();

            // 069 Phase 1: tab system. Off by default; old sidebars still active.
            // When enabled, replaces DrawDockspaceHost + DrawLeftSidebar + DrawRightSidebar
            // with top tab bar + bottom tab bar + central content area.
            if (_useTabUi)
            {
                // 071: left sidebar + right workbench squeeze the 3D viewport.
                DrawLeftSidebar();
                DrawRightSidebar();
            }
            else if (_useDockspaceUi)
            {
                DrawDockspaceHost();
            }

            if (!_useTabUi)
            {
                if (HasAnyShellPanelsInLane(ShellPanelLane.Left))
                    DrawLegacyLeftSidebar();
                if (HasAnyShellPanelsInLane(ShellPanelLane.Right))
                    DrawLegacyRightSidebar();
            }

            DrawFixedSidebarSplitters();

            // Toolbar is drawn after sidebars so it stays on top of any edge overlap.
            DrawToolbar();

            DrawBottomBar();

            DrawStatusBar();

            // Floating windows: when tab system is active, only the tools that
            // aren't yet routed into a sub-tab render as floating windows.
            // Phase 2 routes Scene + Utilities; later phases route the rest.
            if (_useTabUi)
            {
                if (_showWdlPreview)
                    DrawWdlPreviewDialog();
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
                    DrawLogViewer();

                // WDL Preview (floating window)
                if (_showWdlPreview)
                    DrawWdlPreviewDialog();

                // Minimap panel
                if (IsShellPanelActive(ShellPanelId.Minimap) && !_fullscreenMinimap)
                    DrawMinimapWindow();

                // Perf (floating window) - legacy mode only; tabbed mode uses Utilities > Perf
                if (_showPerfWindow && !_useTabUi)
                    DrawPerfWindow();


                if (_showCaptureAutomationWindow)
                    DrawCaptureAutomationWindow();

                if (_showCameraPathWindow)
                    DrawCameraPathWindow();

                // Tool windows extracted from right sidebar
                if (_showUniqueIdArchaeologyWindow && _worldScene != null)
                    DrawUniqueIdArchaeologyWindow();

            }

            // Settings (global configuration window) - must render in BOTH tabbed and legacy modes
            if (_showSettingsWindow)
                DrawSettingsWindow();

            if (!_useTabUi && _showWeakSignalWindow && (_terrainManager != null || _vlmTerrainManager != null))
                DrawWeakSignalWindow();

        }

        if (_activeVideoRecording?.MarketingTourAttempt?.ActivePresentation is FeatureTourPresentation presentation)
            MarketingTourOverlayRenderer.Draw(presentation);

        _forceApplyShellPanelLayout = false;

        // Fullscreen minimap overlay (M key toggle)
        if (_fullscreenMinimap && (_worldScene != null || _vlmTerrainManager != null))
            DrawFullscreenMinimap();

        // Modal dialogs
        if (_showFolderInput)
            DrawFolderInputDialog();
        if (_showBuildSelectionDialog)
            DrawBuildSelectionDialog();
        if (_showListfileInput)
            DrawListfileInputDialog();
        if (_showMlTrainingDialog || IsMlTrainingProcessActive())
            UpdateMlTrainingMonitor();
        if (_showVlmExportDialog)
            _datasetExportDialogs.DrawVlmExportDialog();
        if (_showMlTrainingDialog)
            DrawMlTrainingDialog();
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
            DrawSynthesizedMinimapExportDialog();
        if (_showRosettaDatastoreDialog)
            DrawRosettaDatastoreDialog();

        DrawSceneHoverAssetOverlay();
        DrawClickSelectionOverlay();

        _autoOpenWorldMapsPanel = false;
    }

    private void DrawMenuBar()
    {
        if (ImGui.BeginMainMenuBar())
        {
            if (ImGui.BeginMenu("File"))
            {
                if (ImGui.MenuItem("Open File..."))
                    _wantOpenFile = true;

                if (ImGui.MenuItem("Open Alpha WDT (loose map)..."))
                    _wantOpenWdtFile = true;

                if (ImGui.MenuItem("Open Loose PM4 / PD4 File..."))
                    _wantOpenPm4File = true;

                ImGui.Separator();

                if (ImGui.MenuItem("Open Game Folder (MPQ)..."))
                {
                    _showFolderInput = true;
                    _folderInputBuf = string.IsNullOrWhiteSpace(_lastGameFolderPath) ? "" : _lastGameFolderPath;
                }

                if (ImGui.MenuItem("Open CASC Install (local)..."))
                    _wantOpenCascInstall = true;

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Pick a Battle.net install folder, then the game version to load. Reads only what is on disk.");

                if (ImGui.MenuItem("Open CASC Install (local + CDN fill)..."))
                    _wantOpenCascInstallWithCdnFill = true;

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Pick a Battle.net install folder, then the game version to load. Files the build lists but the install does not have on disk are downloaded from Blizzard's CDN for the same build.");

                // Spec 247: every DAT action lives under one submenu instead of four peers in File.
                if (ImGui.BeginMenu("DAT Terrain (v22/23/26)"))
                {
                    if (ImGui.MenuItem("Open DAT Terrain Folder..."))
                        _wantOpenAhdrTerrainFolder = true;

                    ImGui.Separator();
                    ImGui.TextDisabled("Export format");
                    Terrain.MapExportFormats.DrawCheckboxes("filemenu");

                    bool datLoaded = _terrainManager?.Adapter is AhdrTerrainAdapter;
                    if (ImGui.MenuItem("Export Loaded DAT Map...", null, false, datLoaded && Terrain.MapExportFormats.Any))
                        ExportLoadedDatMap();

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.SetTooltip(!datLoaded
                            ? "Open a DAT terrain folder first."
                            : !Terrain.MapExportFormats.Any
                                ? "Tick at least one output format above."
                                : $"Writes the loaded DAT folder as {Terrain.MapExportFormats.Summary}, plus a manifest naming every field carried and dropped.");
                    }

                    ImGui.Separator();
                    if (ImGui.MenuItem("Export Nearby Tiles as DAT v26 (experimental)", null, false, _terrainManager?.Adapter is StandardTerrainAdapter))
                        ExportNearbyTilesAsDatV26(radius: 2);

                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("Writes the ADT tiles within 2 tiles of the camera as DAT v26 files (terrain, texture layers, vertex colours, normals, objects) to output/dat_v26_export. Reopen with Open DAT Terrain Folder to compare.");

                    DrawAhdrHeightScaleMenu();
                    ImGui.EndMenu();
                }

                if (ImGui.BeginMenu("Open Saved Game Folder", _knownGoodClientPaths.Count > 0))
                {
                    foreach (var knownClient in _knownGoodClientPaths)
                    {
                        if (ImGui.MenuItem($"{knownClient.Name}##open_saved_{knownClient.Path}"))
                            QueueKnownGoodClientAction(knownClient.Path, knownClient.BuildVersion, attachLooseFolder: false);

                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip(BuildKnownGoodClientTooltip(knownClient));
                    }

                    ImGui.EndMenu();
                }

                if (ImGui.MenuItem("Attach Loose Map Folder...", "", false, _dataSource is MpqDataSource))
                    _wantAttachLooseMapFolder = true;

                if (ImGui.BeginMenu("Load Loose Map Folder Against Saved Base", _knownGoodClientPaths.Count > 0))
                {
                    foreach (var knownClient in _knownGoodClientPaths)
                    {
                        if (ImGui.MenuItem($"{knownClient.Name}##attach_saved_{knownClient.Path}"))
                            QueueKnownGoodClientAction(knownClient.Path, knownClient.BuildVersion, attachLooseFolder: true);

                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip(BuildKnownGoodClientTooltip(knownClient));
                    }

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.MenuItem("Save Current Game Folder As Known-Good Base", "", false, _dataSource is MpqDataSource))
                    SaveCurrentGameFolderAsKnownGoodBase();

                if (ImGui.BeginMenu("Forget Known-Good Base", _knownGoodClientPaths.Count > 0))
                {
                    foreach (var knownClient in _knownGoodClientPaths)
                    {
                        if (ImGui.MenuItem($"{knownClient.Name}##forget_saved_{knownClient.Path}"))
                            QueueForgetKnownGoodClientPath(knownClient);

                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip(BuildKnownGoodClientTooltip(knownClient));
                    }

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.MenuItem("Settings..."))
                    _showSettingsWindow = true;

                ImGui.Separator();

                if (ImGui.MenuItem("Quit"))
                    _window.Close();

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("View"))
            {
                if (ImGui.MenuItem("Wireframe", "W"))
                    _renderer?.ToggleWireframe();

                if (ImGui.MenuItem("Reset Camera"))
                    ResetCamera();

                if (ImGui.MenuItem("Hide UI Chrome", "Tab", _hideUiChrome))
                    _hideUiChrome = !_hideUiChrome;

                ImGui.Separator();

                if (ImGui.MenuItem("Tab System (069)", "", ref _useTabUi))
                {
                    // Save preference so it sticks across restarts.
                    SaveViewerSettings();
                }

                bool useDockspaceUi = _useDockspaceUi;
                if (_useTabUi) ImGui.BeginDisabled();
                if (ImGui.MenuItem("Dockable Shell Panels", "", ref useDockspaceUi))
                {
                    _useDockspaceUi = useDockspaceUi;
                    _forceApplyShellPanelLayout = _useDockspaceUi;
                    SaveViewerSettings();
                }
                if (_useTabUi) ImGui.EndDisabled();

                ImGui.MenuItem("Left Sidebar", "", ref _showLeftSidebar);
                ImGui.MenuItem("Right Sidebar", "I", ref _showRightSidebar);
                if (ImGui.MenuItem("Log Console..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Log);
                    else _showLogViewer = true;
                }
                if (ImGui.MenuItem("Performance & Profiling..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Perf);
                    else _showPerfWindow = true;
                }
                if (ImGui.MenuItem("Lighting Diagnostics..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Lighting);
                    else OpenLegacyWorkbenchUtility(UtilitiesBottomTab.Lighting);
                }
                if (ImGui.MenuItem("Focus PM4 Tools", "P"))
                    OpenPm4Workbench(Pm4WorkbenchTab.Selection);
                if (ImGui.MenuItem("Reset Shell Layout"))
                    ResetShellLayoutToDefaults();
                ImGui.Separator();
                ImGui.MenuItem("File Browser", "", ref _showFileBrowser);
                ImGui.MenuItem("Model Info", "", ref _showModelInfo);
                ImGui.Separator();
                if (ImGui.MenuItem("Asset Catalog"))
                {
                    if (_useTabUi)
                        OpenWorkbenchTab(UtilitiesBottomTab.AssetCatalog);
                    else
                    {
                        if (_catalogView == null)
                        {
                            _catalogView = new AssetCatalogView(_gl);
                            _catalogView.SetDataSource(_dataSource);
                            _catalogView.OnLoadModelRequested = OnCatalogLoadModel;
                        }
                        _catalogView.IsVisible = !_catalogView.IsVisible;
                    }
                }

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Tools"))
            {
                if (ImGui.MenuItem("Taxi Routes...", "", false, _worldScene != null))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Taxi);
                    else OpenLegacyWorkbenchUtility(UtilitiesBottomTab.Taxi);
                }
                if (ImGui.MenuItem("Audio Settings..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Audio);
                    else OpenLegacyWorkbenchUtility(UtilitiesBottomTab.Audio);
                }
                ImGui.Separator();
                // 071: floating-window toggles removed. Every tool lives in a
                // workbench tab under Tools > Panels or the relevant top tab.

                if (ImGui.BeginMenu("Converters"))
                {
                    if (ImGui.MenuItem("Map Converter..."))
                    {
                        PrepareMapConverterDialogInputs();
                        _showMapConverterDialog = true;
                    }

                    if (ImGui.MenuItem("WMO Converter..."))
                    {
                        PrepareWmoConverterDialogInputs();
                        _showWmoConverterDialog = true;
                    }

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.BeginMenu("Panels"))
                {
                    bool hasTerrain = _terrainManager != null || _vlmTerrainManager != null;
                    bool hasWorld = _worldScene != null;

                    if (ImGui.MenuItem("Model Info"))
                        OpenWorkbenchTab(ModelBottomTab.Info);

                    ImGui.Separator();

                    if (ImGui.MenuItem("Log Viewer"))
                        OpenWorkbenchTab(UtilitiesBottomTab.Log);
                    if (ImGui.MenuItem("Perf"))
                        OpenWorkbenchTab(UtilitiesBottomTab.Perf);
                    if (ImGui.MenuItem("Settings..."))
                        _showSettingsWindow = true;

                    ImGui.Separator();

                    if (ImGui.MenuItem("Asset Catalog"))
                        OpenWorkbenchTab(UtilitiesBottomTab.AssetCatalog);
                    if (ImGui.MenuItem("Capture Automation"))
                        OpenCapturePanelTab(CapturePanelTab.Automation);
                    if (ImGui.MenuItem("Camera Path"))
                        OpenCapturePanelTab(CapturePanelTab.CameraPath);
                    if (ImGui.MenuItem("Taxi", hasWorld))
                        OpenWorkbenchTab(UtilitiesBottomTab.Taxi);

                    ImGui.Separator();

                    if (ImGui.MenuItem("Weak Signal & Stratigraphy", hasTerrain))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 0);
                    if (ImGui.MenuItem("UniqueId Archaeology", hasWorld))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 1);
                    if (ImGui.MenuItem("PM4 Analysis", hasWorld))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 4);
                    if (ImGui.MenuItem("Cartography", hasTerrain))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 5);

                    ImGui.Separator();

                    if (ImGui.MenuItem("Editor Workbench", hasTerrain || hasWorld))
                        OpenWorkbenchTab(WorkbenchTab.Editor, 0);

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.BeginMenu("Export"))
                {
                    if (ImGui.MenuItem("Synthesized Terrain Minimap..."))
                    {
                        PrepareSynthesizedMinimapExportDialogInputs();
                        _showSynthesizedMinimapExportDialog = true;
                    }

                    ImGui.Separator();

                    if (ImGui.BeginMenu("GLB"))
                    {
                        // Terrain (including DAT folders opened with no client data source) can export
                        // GLB too, so the menu is enabled for either a standalone model or a terrain.
                        bool canExportGlb = _renderer != null || _terrainManager != null;
                        if (ImGui.MenuItem("Export GLB...", canExportGlb))
                            _wantExportGlb = true;
                        if (ImGui.MenuItem("Export GLB (Collision Only)...", _renderer != null))
                            _wantExportGlbCollision = true;

                        ImGui.Separator();

                        bool canExportMapGlb = _terrainManager != null;
                        if (ImGui.BeginMenu("Map Tiles", canExportMapGlb))
                        {
                            if (ImGui.MenuItem("Current Tile (Terrain + Objects)", "", false, canExportMapGlb))
                            {
                                _mapGlbScope = TerrainTileScope.CurrentTile;
                                _wantExportMapGlbTiles = true;
                            }
                            if (ImGui.MenuItem("Loaded Tiles Folder", "", false, canExportMapGlb))
                            {
                                _mapGlbScope = TerrainTileScope.LoadedTiles;
                                _wantExportMapGlbTiles = true;
                            }
                            if (ImGui.MenuItem("Whole Map Folder", "", false, canExportMapGlb))
                            {
                                _mapGlbScope = TerrainTileScope.WholeMap;
                                _wantExportMapGlbTiles = true;
                            }
                            ImGui.EndMenu();
                        }

                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Terrain"))
                    {
                        bool hasTerrain = _terrainManager != null || _vlmTerrainManager != null;

                        if (ImGui.BeginMenu("Alpha Masks"))
                        {
                            if (ImGui.MenuItem("Current Tile Atlas (PNG)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaCurrentTileAtlas;
                            }

                            if (ImGui.MenuItem("Current Tile Chunks Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaCurrentTileChunksFolder;
                            }

                            if (ImGui.MenuItem("Loaded Tiles Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaLoadedTilesFolder;
                            }

                            if (ImGui.MenuItem("Whole Map Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaWholeMapFolder;
                            }

                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("Heightmaps"))
                        {
                            if (ImGui.MenuItem("Current Tile (257x257 L16 PNG + JSON)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.Heightmap257CurrentTilePerTile;
                            }

                            if (ImGui.MenuItem("Loaded Tiles Folder (per-tile)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.Heightmap257LoadedTilesFolderPerTile;
                            }

                            if (ImGui.MenuItem("Whole Map Folder (per-map)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.Heightmap257WholeMapFolderPerMap;
                            }

                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("MCCV"))
                        {
                            if (ImGui.MenuItem("Current Tile PNG...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.MccvCurrentTilePng;
                            }

                            if (ImGui.MenuItem("Loaded Tiles Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.MccvLoadedTilesFolder;
                            }

                            if (ImGui.MenuItem("Whole Map Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.MccvWholeMapFolder;
                            }

                            ImGui.EndMenu();
                        }

                        ImGui.EndMenu();
                    }

                    ImGui.EndMenu();
                }

                if (ImGui.BeginMenu("Import"))
                {
                    if (ImGui.BeginMenu("Terrain"))
                    {
                        bool hasTerrain = _terrainManager != null || _vlmTerrainManager != null;

                        if (ImGui.BeginMenu("Alpha Masks"))
                        {
                            if (ImGui.MenuItem("From Folder of Tile Atlases...", hasTerrain))
                            {
                                _wantTerrainImport = true;
                                _terrainImportKind = TerrainImportKind.AlphaFolder;
                            }
                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("Heightmaps"))
                        {
                            if (ImGui.MenuItem("From Folder of Tile Heightmaps...", hasTerrain))
                            {
                                _wantTerrainImport = true;
                                _terrainImportKind = TerrainImportKind.Heightmap257Folder;
                            }
                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("MCCV"))
                        {
                            if (ImGui.MenuItem("From Folder of Tile MCCV PNGs...", hasTerrain))
                            {
                                _wantTerrainImport = true;
                                _terrainImportKind = TerrainImportKind.MccvFolder;
                            }
                            ImGui.EndMenu();
                        }

                        ImGui.EndMenu();
                    }

                    ImGui.EndMenu();
                }

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Help"))
            {
                if (ImGui.MenuItem("Keyboard Shortcuts"))
                    _showKeyboardShortcutsWindow = true;

                if (ImGui.MenuItem("About"))
                {
                    _openAboutPopup = true;
                    _statusMessage = ViewerProductName;
                }
                ImGui.EndMenu();
            }

            // Top-Level Mode & Workspace Profile Switcher (centered on the main menu bar)
            float modeBtnWidthViewer = 90f;
            float modeBtnWidthEditor = 90f;
            float modeBtnWidthArch = 115f;
            float itemSpacing = ImGui.GetStyle().ItemSpacing.X;
            float totalWidth = modeBtnWidthViewer + modeBtnWidthEditor + modeBtnWidthArch + (itemSpacing * 2);
            float windowWidth = ImGui.GetWindowWidth();
            float targetCenterX = (windowWidth - totalWidth) * 0.5f;
            if (targetCenterX > ImGui.GetCursorPosX())
            {
                ImGui.SetCursorPosX(targetCenterX);
            }

            bool isEditor = _workspaceMode == WorkspaceMode.Editor;
            bool isArchaeology = _workspaceMode == WorkspaceMode.Archaeology;
            bool isViewer = _workspaceMode == WorkspaceMode.Viewer;

            if (isViewer)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.18f, 0.42f, 0.75f, 1f));
            if (ImGui.Button("Viewer##top_mode_viewer", new Vector2(modeBtnWidthViewer, 0)))
                SetWorkspaceMode(WorkspaceMode.Viewer);
            if (isViewer)
                ImGui.PopStyleColor();

            ImGui.SameLine();
            if (isEditor)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.75f, 0.42f, 0.15f, 1f));
            if (ImGui.Button("Editor##top_mode_editor", new Vector2(modeBtnWidthEditor, 0)))
                SetWorkspaceMode(WorkspaceMode.Editor);
            if (isEditor)
                ImGui.PopStyleColor();

            ImGui.SameLine();
            if (isArchaeology)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.55f, 0.25f, 0.75f, 1f));
            if (ImGui.Button("Archaeology##top_mode_archaeology", new Vector2(modeBtnWidthArch, 0)))
                SetWorkspaceMode(WorkspaceMode.Archaeology);
            if (isArchaeology)
                ImGui.PopStyleColor();

            ImGui.EndMainMenuBar();
        }

        DrawKeyboardShortcutsWindow();

        if (_openForgetKnownGoodClientConfirm)
        {
            _openForgetKnownGoodClientConfirm = false;
            ImGui.OpenPopup("Confirm Forget Known-Good Base");
        }

        if (_openAboutPopup)
        {
            _openAboutPopup = false;
            ImGui.OpenPopup(ViewerAboutPopupTitle);
        }

        bool keepAboutPopupOpen = true;
        if (ImGui.BeginPopupModal(ViewerAboutPopupTitle, ref keepAboutPopupOpen, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.TextUnformatted(ViewerProductName);
            ImGui.TextDisabled($"Version {GetViewerDisplayVersion()}");
            ImGui.Spacing();
            ImGui.TextWrapped("World/model viewer and debugging surface for WoW Alpha, Wrath, and early Cataclysm data.");
            ImGui.Spacing();
            ImGui.TextWrapped("Author: github.com/akspa0/parp-tools");
            ImGui.TextWrapped("Discord: discord.gg/6YdUksuKuU");
            ImGui.Spacing();
            ImGui.TextUnformatted("In memory of Hayven Games");
            ImGui.TextWrapped("An inspiration for this project and a friend, whose short films explored World of Warcraft's secrets and little-known details through game footage.");
            ImGui.Spacing();
            ImGui.Separator();
            ImGui.TextWrapped("Thanks to...");
            ImGui.TextWrapped("Marlamin, schlumpf, Dovah, Pirate the Explorer, fean, implave, IS4, Mjollna, Adspartan (Noggit), and Skarn (Noggit-Red).");
            ImGui.TextWrapped("Without the WoW Exploration community, this project would not exist. Everyone named here contributed inspiration to this project in some way.");
            ImGui.TextDisabled("This tooling is about restoration, not touching up or polishing what we recover - it is the instrument for restoring what already exists. Noggit and Noggit-Red remain the preferred editors for fine-tuning the results this library and tooling produce.");
            ImGui.Spacing();
            ImGui.TextWrapped("Special thanks to WoWdev.wiki, Exploration Reboot, The Alpha Project, and everyone in the Pre-Alpha Restoration Project discord!");
            ImGui.Spacing();
            if (ImGui.Button("Close", new Vector2(120f, 0f)))
                ImGui.CloseCurrentPopup();

            ImGui.EndPopup();
        }

        bool keepForgetKnownGoodPopupOpen = true;
        if (ImGui.BeginPopupModal("Confirm Forget Known-Good Base", ref keepForgetKnownGoodPopupOpen, ImGuiWindowFlags.AlwaysAutoResize))
        {
            string displayName = string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientDisplayName)
                ? "this saved base"
                : _pendingForgetKnownGoodClientDisplayName!;

            ImGui.TextWrapped($"Remove saved base '{displayName}'?");
            if (!string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientPath))
                ImGui.TextDisabled(_pendingForgetKnownGoodClientPath);

            ImGui.Spacing();
            if (ImGui.Button("Remove", new Vector2(120f, 0f)))
            {
                if (!string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientPath))
                    ForgetKnownGoodClientPath(_pendingForgetKnownGoodClientPath);

                ClearPendingForgetKnownGoodClientPath();
                ImGui.CloseCurrentPopup();
            }

            ImGui.SameLine();
            if (ImGui.Button("Cancel", new Vector2(120f, 0f)))
            {
                ClearPendingForgetKnownGoodClientPath();
                ImGui.CloseCurrentPopup();
            }

            ImGui.EndPopup();
        }

        if (!keepForgetKnownGoodPopupOpen)
            ClearPendingForgetKnownGoodClientPath();

        if (!keepAboutPopupOpen)
            _openAboutPopup = false;

        // Handle deferred actions
        if (_wantOpenFile)
        {
            _wantOpenFile = false;
            _showFolderInput = false;
            // Use ImGui text input as a simple file path dialog
            ImGui.OpenPopup("OpenFilePopup");
        }

        if (ImGui.BeginPopup("OpenFilePopup"))
        {
            ImGui.Text("Enter file path:");
            var buf = _folderInputBuf;
            if (ImGui.InputText("##filepath", ref buf, 512, ImGuiInputTextFlags.EnterReturnsTrue))
            {
                if (File.Exists(buf))
                {
                    LoadFileFromDisk(buf);
                    ImGui.CloseCurrentPopup();
                }
                else
                {
                    _statusMessage = $"File not found: {buf}";
                }
            }
            _folderInputBuf = buf;
            if (ImGui.Button("Cancel"))
                ImGui.CloseCurrentPopup();
            ImGui.EndPopup();
        }

        if (_wantOpenVlmProject)
        {
            _wantOpenVlmProject = false;
            ImGuiPathPicker.Instance.Open(
                "Select ML Dataset folder (containing dataset/ with JSON files)",
                pickFolder: true,
                initialPath: null,
                filterExtension: null,
                vlmPath =>
                {
                    if (!string.IsNullOrEmpty(vlmPath) && Directory.Exists(vlmPath))
                        LoadVlmProject(vlmPath);
                });
        }

        if (_wantOpenZarrDataset)
        {
            _wantOpenZarrDataset = false;
            ImGuiPathPicker.Instance.Open(
                "Select Zarr tile dataset folder (parent of <build>.zarr/ or the store root itself)",
                pickFolder: true,
                initialPath: null,
                filterExtension: null,
                zarrPath =>
                {
                    if (!string.IsNullOrEmpty(zarrPath) && Directory.Exists(zarrPath))
                        LoadZarrDataset(zarrPath);
                });
        }

        if (_wantSelectDatasetCatalogRoot)
        {
            _wantSelectDatasetCatalogRoot = false;
            ImGuiPathPicker.Instance.Open(
                "Select dataset catalog root",
                pickFolder: true,
                initialPath: _datasetCatalogRoot,
                filterExtension: null,
                catalogRoot =>
                {
                    if (!string.IsNullOrWhiteSpace(catalogRoot) && Directory.Exists(catalogRoot))
                    {
                        _datasetCatalogRoot = catalogRoot;
                        RefreshDatasetCatalog();
                        SaveViewerSettings();
                    }
                });
        }

        if (_wantOpenWdtFile)
        {
            _wantOpenWdtFile = false;
            ImGuiPathPicker.Instance.Open(
                "Select Alpha WDT file (loose map)",
                pickFolder: false,
                initialPath: _lastLooseOverlayPath,
                filterExtension: ".wdt;.mpq",
                wdtPath =>
                {
                    if (!string.IsNullOrEmpty(wdtPath) && File.Exists(wdtPath))
                    {
                        LoadFileFromDisk(wdtPath);
                        _statusMessage = $"Loaded alpha WDT: {wdtPath}";
                    }
                });
        }

        if (_wantOpenPm4File)
        {
            _wantOpenPm4File = false;
            ImGuiPathPicker.Instance.Open(
                "Select Loose PM4 / PD4 File",
                pickFolder: false,
                initialPath: _lastLooseOverlayPath,
                filterExtension: ".pm4;.pd4",
                pm4Path =>
                {
                    if (!string.IsNullOrEmpty(pm4Path) && File.Exists(pm4Path))
                    {
                        _lastLooseOverlayPath = Path.GetDirectoryName(pm4Path);
                        if (_worldScene != null)
                        {
                            if (_worldScene.Pm4Overlay.LoadLoosePm4File(pm4Path))
                                _statusMessage = _worldScene.Pm4Overlay.Pm4Status;
                            else
                                _statusMessage = $"Failed to decode loose PM4/PD4 file: {pm4Path}";
                        }
                        else
                        {
                            _statusMessage = $"Load a world scene or map first before displaying loose PM4/PD4 overlays: {pm4Path}";
                        }
                    }
                });
        }

        HandleCascAhdrMenuRequests();

        if (_wantAttachLooseMapFolder)
        {
            _wantAttachLooseMapFolder = false;

            if (_dataSource is MpqDataSource)
            {
                ImGuiPathPicker.Instance.Open(
                    "Select loose map overlay folder (contains World\\Maps or a map directory under World\\Maps)",
                    pickFolder: true,
                    initialPath: string.IsNullOrWhiteSpace(_lastLooseOverlayPath) ? null : _lastLooseOverlayPath,
                    filterExtension: null,
                    overlayPath =>
                    {
                        if (!string.IsNullOrEmpty(overlayPath) && Directory.Exists(overlayPath))
                            AttachLooseMapOverlay(overlayPath);
                    });
            }
        }

        if (!string.IsNullOrWhiteSpace(_pendingKnownGoodClientPath))
        {
            string savedBasePath = _pendingKnownGoodClientPath!;
            string? savedBuildVersion = _pendingKnownGoodClientBuildVersion;
            bool attachLooseFolder = _pendingKnownGoodClientAttachLooseFolder;
            _pendingKnownGoodClientPath = null;
            _pendingKnownGoodClientBuildVersion = null;
            _pendingKnownGoodClientAttachLooseFolder = false;

            if (!Directory.Exists(savedBasePath))
            {
                _statusMessage = $"Saved client path no longer exists: {savedBasePath}";
            }
            else if (attachLooseFolder)
            {
                ImGuiPathPicker.Instance.Open(
                    "Select loose map folder to load against the saved base client",
                    pickFolder: true,
                    initialPath: string.IsNullOrWhiteSpace(_lastLooseOverlayPath) ? null : _lastLooseOverlayPath,
                    filterExtension: null,
                    overlayPath =>
                    {
                        if (!string.IsNullOrWhiteSpace(overlayPath) && Directory.Exists(overlayPath))
                        {
                            LoadMpqDataSource(savedBasePath, null, savedBuildVersion, deferWorldReload: true);
                            AttachLooseMapOverlay(overlayPath);
                            RestoreWorldAfterDataSourceReload();
                        }
                    });
            }
            else
            {
                LoadMpqDataSource(savedBasePath, null, savedBuildVersion);
            }
        }

        if (_wantTerrainExport)
        {
            _wantTerrainExport = false;
            _terrainTileIo.RunTerrainExport();
        }

        if (_wantTerrainImport)
        {
            _wantTerrainImport = false;
            _terrainTileIo.RunTerrainImport();
        }

        if (_wantExportGlbCollision)
        {
            _wantExportGlbCollision = false;
            if (_loadedFilePath != null)
            {
                Directory.CreateDirectory(ExportDir);
                string glbPath = Path.Combine(ExportDir, Path.ChangeExtension(_loadedFileName!, ".collision.glb"));
                try
                {
                    string dir = Path.GetDirectoryName(_loadedFilePath) ?? ".";
                    if (_loadedWmo != null)
                    {
                        GlbExporter.ExportWmoCollision(_loadedWmo, dir, glbPath);
                    }
                    else
                    {
                        var ext = Path.GetExtension(_loadedFilePath).ToLowerInvariant();
                        if (ext == ".wmo")
                        {
                            var converter = new WmoV14ToV17Converter();
                            var wmo = converter.ParseWmoV14(_loadedFilePath);
                            GlbExporter.ExportWmoCollision(wmo, dir, glbPath);
                        }
                        else
                        {
                            throw new InvalidOperationException("Collision-only GLB export is currently supported for WMO and Terrain only.");
                        }
                    }
                    _statusMessage = $"Exported: {glbPath}";
                }
                catch (Exception ex)
                {
                    _statusMessage = $"Export failed: {ex.Message}";
                }
            }
            else if (_terrainManager != null && _dataSource != null)
            {
                Directory.CreateDirectory(ExportDir);
                int curTx = _terrainManager.CameraTileX;
                int curTy = _terrainManager.CameraTileY;
                if (curTx >= 0 && curTy >= 0)
                {
                    string glbPath = Path.Combine(ExportDir, $"{_terrainManager.MapName}_{curTx:D2}_{curTy:D2}.collision.glb");
                    try
                    {
                        MapGlbExporter.ExportTile(_terrainManager, _dataSource, _md5Index, curTx, curTy, glbPath, includePlacements: false);
                        _statusMessage = $"Exported GLB Collision Mesh for Tile ({curTx},{curTy}) to: {glbPath}";
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"GLB Collision export failed: {ex.Message}";
                    }
                }
                else
                {
                    _statusMessage = "Camera tile out of range for collision export.";
                }
            }
            else
            {
                _statusMessage = "No model or terrain loaded for GLB collision export.";
            }
        }

        if (_wantExportGlb)
        {
            _wantExportGlb = false;
            if (_loadedFilePath != null)
            {
                Directory.CreateDirectory(ExportDir);
                string glbPath = Path.Combine(ExportDir, Path.ChangeExtension(_loadedFileName!, ".glb"));
                try
                {
                    string dir = Path.GetDirectoryName(_loadedFilePath) ?? ".";
                    if (_loadedWmo != null)
                    {
                        GlbExporter.ExportWmoWithDoodads(_loadedWmo, dir, glbPath, _dataSource);
                    }
                    else if (_loadedMdx != null)
                    {
                        GlbExporter.ExportMdx(_loadedMdx, dir, glbPath, _dataSource);
                    }
                    else
                    {
                        // Fallback: re-parse from disk (legacy path)
                        var ext = Path.GetExtension(_loadedFilePath).ToLowerInvariant();
                        if (ext == ".mdx")
                        {
                            var mdx = MdxFile.Load(_loadedFilePath);
                            GlbExporter.ExportMdx(mdx, dir, glbPath, _dataSource);
                        }
                        else if (ext == ".m2")
                        {
                            byte[] m2Bytes = File.ReadAllBytes(_loadedFilePath);
                            byte[]? skinBytes = null;
                            foreach (var skinPath in WarcraftNetM2Adapter.BuildSkinCandidates(_loadedFilePath))
                            {
                                if (File.Exists(skinPath)) { skinBytes = File.ReadAllBytes(skinPath); break; }
                            }
                            var converter = new WoWViewer.Transfer.M2ToMdxConverter();
                            byte[]? mdxBytes = converter.ConvertToBytes(m2Bytes, skinBytes, null);
                            if (mdxBytes != null)
                            {
                                using var ms = new MemoryStream(mdxBytes);
                                var mdx = MdxFile.Load(ms);
                                GlbExporter.ExportMdx(mdx, dir, glbPath, _dataSource);
                            }
                        }
                        else if (ext == ".wmo")
                        {
                            var converter = new WmoV14ToV17Converter();
                            var wmo = converter.ParseWmoV14(_loadedFilePath);
                            GlbExporter.ExportWmoWithDoodads(wmo, dir, glbPath, _dataSource);
                        }
                    }
                    _statusMessage = $"Exported: {glbPath}";
                }
                catch (Exception ex)
                {
                    _statusMessage = $"Export failed: {ex.Message}";
                }
            }
            else if (_terrainManager != null)
            {
                Directory.CreateDirectory(ExportDir);
                int curTx = _terrainManager.CameraTileX;
                int curTy = _terrainManager.CameraTileY;
                if (curTx >= 0 && curTy >= 0)
                {
                    string glbPath = Path.Combine(ExportDir, $"{_terrainManager.MapName}_{curTx:D2}_{curTy:D2}.glb");
                    try
                    {
                        MapGlbExporter.ExportTile(_terrainManager, _dataSource, _md5Index, curTx, curTy, glbPath, includePlacements: true);
                        _statusMessage = $"Exported GLB Scene for Tile ({curTx},{curTy}) to: {glbPath}";
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"GLB Scene export failed: {ex.Message}";
                    }
                }
                else
                {
                    _statusMessage = "Camera tile out of range for GLB export.";
                }
            }
            else
            {
                _statusMessage = "No model or terrain loaded for GLB export.";
            }
        }

        if (_wantExportMapGlbTiles)
        {
            _wantExportMapGlbTiles = false;
            try
            {
                _terrainTileIo.RunMapGlbTilesExport();
            }
            catch (Exception ex)
            {
                _statusMessage = $"Map GLB export failed: {ex.Message}";
            }
        }
    }

    private void QueueForgetKnownGoodClientPath(KnownGoodClientPath knownClient)
    {
        _pendingForgetKnownGoodClientPath = knownClient.Path;
        _pendingForgetKnownGoodClientDisplayName = knownClient.Name;
        _openForgetKnownGoodClientConfirm = true;
    }

    private void ClearPendingForgetKnownGoodClientPath()
    {
        _pendingForgetKnownGoodClientPath = null;
        _pendingForgetKnownGoodClientDisplayName = null;
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

    private void AnalyzeActiveCameraTileStratigraphy()
    {
        var cameraTile = GetCameraTile();
        int tileX = cameraTile.tileX;
        int tileY = cameraTile.tileY;

        IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
        if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
            chunks = result.Chunks;
        else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var vlmResult))
            chunks = vlmResult.Chunks;

        if (chunks == null || chunks.Count == 0)
        {
            _terrainWeakSignalRestoreStatus = $"Tile ({tileY}, {tileX}) is not currently loaded.";
            return;
        }

        var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
        float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);

        var holeMasks = new ushort[256];
        for (int i = 0; i < Math.Min(chunks.Count, 256); i++)
            holeMasks[i] = (ushort)chunks[i].HoleMask;

        string tileName = $"tile_{tileX}_{tileY}";
        var analysis = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyLevelAnalyzer.AnalyzeTile(lattice257, holeMasks, tileX, tileY, tileName);
        _stratigraphyTileAnalyses[(tileX, tileY)] = analysis;

        _terrainWeakSignalRestoreStatus = $"Tile ({tileY}, {tileX}) analyzed: {analysis.DominantStratum}, {analysis.TotalSurvivingLevels:N0} levels, {analysis.SqueezedChunkCount} squeezed chunks, {analysis.HoledChunkCount} dev mesh chunks.";
    }

    private void AnalyzeAllLoadedTilesStratigraphy()
    {
        var loadedTiles = new HashSet<(int tileX, int tileY)>();
        if (_terrainManager != null)
        {
            foreach (var key in _terrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }
        if (_vlmTerrainManager != null)
        {
            foreach (var key in _vlmTerrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }

        int count = 0;
        int squeezedTotal = 0;
        int holedTotal = 0;

        foreach (var (tileX, tileY) in loadedTiles)
        {
            IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
            if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
                chunks = result.Chunks;
            else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var vlmResult))
                chunks = vlmResult.Chunks;

            if (chunks == null || chunks.Count == 0) continue;

            var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);

            var holeMasks = new ushort[256];
            for (int i = 0; i < Math.Min(chunks.Count, 256); i++)
                holeMasks[i] = (ushort)chunks[i].HoleMask;

            var analysis = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyLevelAnalyzer.AnalyzeTile(lattice257, holeMasks, tileX, tileY, $"tile_{tileX}_{tileY}");
            _stratigraphyTileAnalyses[(tileX, tileY)] = analysis;

            count++;
            squeezedTotal += analysis.SqueezedChunkCount;
            holedTotal += analysis.HoledChunkCount;
        }

        _terrainWeakSignalRestoreStatus = $"Analyzed {count} loaded tile(s): {squeezedTotal} squeezed chunks, {holedTotal} dev mesh chunks across scene.";
    }

    private void OpenStratigraphySaveDialog()
    {
        string initial = string.IsNullOrEmpty(_stratigraphySaveOutputDirectory)
            ? Directory.GetCurrentDirectory()
            : _stratigraphySaveOutputDirectory;

        ImGuiPathPicker.Instance.Open(
            "Select Output Directory to Save Restored ADT / WDT Tiles",
            pickFolder: true,
            initialPath: initial,
            filterExtension: null,
            selectedPath =>
            {
                if (!string.IsNullOrEmpty(selectedPath))
                {
                    _stratigraphySaveOutputDirectory = selectedPath;
                    ExportLoadedStratigraphyTiles(selectedPath);
                }
            });
    }

    private void ExportLoadedStratigraphyTiles(string outputDir)
    {
        if (string.IsNullOrWhiteSpace(outputDir)) return;
        Directory.CreateDirectory(outputDir);

        var loadedTiles = new HashSet<(int tileX, int tileY)>();
        if (_terrainManager != null)
        {
            foreach (var key in _terrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }
        if (_vlmTerrainManager != null)
        {
            foreach (var key in _vlmTerrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }

        string mapName = _terrainManager?.MapName ?? GetCurrentSessionMapName() ?? "CustomMap";
        string outputMapDir = Path.Combine(outputDir, "World", "Maps", mapName);
        Directory.CreateDirectory(outputMapDir);

        int exported = 0;
        foreach (var (tx, ty) in loadedTiles)
        {
            IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
            if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tx, ty, out var result))
                chunks = result.Chunks;
            else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tx, ty, out var vlmResult))
                chunks = vlmResult.Chunks;

            if (chunks == null || chunks.Count == 0) continue;

            var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);

            string outAdtPath = Path.Combine(outputMapDir, $"{mapName}_{tx}_{ty}.adt");
            float[] flat = WowViewer.Core.IO.Maps.StratigraphyTileExporter.FlattenHeights257(lattice257);

            var blankAdt = WowViewer.Core.IO.Maps.BlankAdtFactory.CreateBlank(mapName, tx, ty);
            WowViewer.Core.IO.Maps.LkAdtWriter.Write(outAdtPath, blankAdt);
            WowViewer.Core.IO.Maps.AdtTerrainWriter.Write(outAdtPath, outAdtPath, flat);
            exported++;
        }

        // Also write companion modified WDL file
        try
        {
            var wdlDict = new Dictionary<(int tileX, int tileY), WowViewer.Core.Runtime.World.Terrain.Stratigraphy.WdlTileData>();
            foreach (var (tx, ty) in loadedTiles)
            {
                IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
                if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tx, ty, out var result))
                    chunks = result.Chunks;
                else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tx, ty, out var vlmResult))
                    chunks = vlmResult.Chunks;

                if (chunks == null || chunks.Count == 0) continue;
                var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
                float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);
                wdlDict[(tx, ty)] = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.WdlFileWriter.FromLattice257(lattice257);
            }

            if (wdlDict.Count > 0)
            {
                byte[] wdlBytes = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.WdlFileWriter.Write(wdlDict);
                string outWdlPath = Path.Combine(outputMapDir, $"{mapName}.wdl");
                File.WriteAllBytes(outWdlPath, wdlBytes);
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"Failed to write companion WDL: {ex.Message}");
        }

        _terrainWeakSignalRestoreStatus = $"Successfully exported {exported} restored tile(s) and companion WDL to '{outputMapDir}'.";
        _statusMessage = $"Exported {exported} restored stratigraphy tiles + WDL.";
    }

    private void DrawEditorOverlays(Matrix4x4 view, Matrix4x4 proj)
    {
        var renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        bool drawCameraPathOverlay = _showCameraPathOverlay && _cameraPath.Keyframes.Count > 0;
        if (renderer == null && !drawCameraPathOverlay)
            return;

        bool drawChunkClipboardOverlay = renderer != null && _chunkClipboardShowOverlay
            && (_selectedChunks.Count > 0 || _chunkClipboardLockedTargetKey != null || _chunkClipboardCopiedKey != null);
        bool drawMcnkOverlay = renderer != null && ShouldDrawMcnkFlagOverlay(renderer);
        if (!drawChunkClipboardOverlay && !drawMcnkOverlay && !drawCameraPathOverlay)
            return;

        _editorOverlayBb ??= new Terrain.BoundingBoxRenderer(_gl);

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);

        float overlayTime = (float)(System.Diagnostics.Stopwatch.GetTimestamp() / (double)System.Diagnostics.Stopwatch.Frequency);

        if (drawMcnkOverlay)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _editorOverlayBb.BeginSolidBatch();
            _editorOverlayBb.BeginBatch();
            BatchMcnkFlagOverlayGeometry(_editorOverlayBb);
            _editorOverlayBb.FlushSolidBatch(view, proj);
            _gl.Disable(EnableCap.Blend);
        }

        if (drawChunkClipboardOverlay)
        {
            if (!drawMcnkOverlay)
                _editorOverlayBb.BeginBatch();

            if (_selectedChunks.Count > 0)
            {
                foreach (var (tx, ty, cx, cy) in _selectedChunks)
                {
                    if (renderer.TryGetChunkInfo(tx, ty, cx, cy, out var sel))
                        _editorOverlayBb.BatchBoxMinMax(sel.BoundsMin, sel.BoundsMax, new Vector3(0f, 1f, 1f));
                }
            }

            if (_chunkClipboardLockedTargetKey is { } locked && renderer.TryGetChunkInfo(locked.tileX, locked.tileY, locked.chunkX, locked.chunkY, out var lockedInfo))
                _editorOverlayBb.BatchHighlightedBoxMinMax(
                    lockedInfo.BoundsMin,
                    lockedInfo.BoundsMax,
                    overlayTime,
                    new Vector3(1f, 1f, 1f),
                    new Vector3(1f, 0.8f, 0.1f),
                    new Vector3(0.1f, 0.9f, 1f));

            if (_chunkClipboardCopiedKey is (int copiedTx, int copiedTy, int copiedCx, int copiedCy) copied && renderer.TryGetChunkInfo(copiedTx, copiedTy, copiedCx, copiedCy, out var copiedInfo))
                _editorOverlayBb.BatchBoxMinMax(copiedInfo.BoundsMin, copiedInfo.BoundsMax, new Vector3(1f, 1f, 0f));
        }

        if (drawCameraPathOverlay)
        {
            if (!drawMcnkOverlay && !drawChunkClipboardOverlay)
                _editorOverlayBb.BeginBatch();
            DrawCameraPathOverlay(_editorOverlayBb);
        }

        _editorOverlayBb.FlushBatch(view, proj);

        _gl.DepthMask(true);
    }

    private bool TryPickTerrainChunkUnderMouse(TerrainRenderer renderer, out TerrainRenderer.TerrainChunkInfo info)
    {
        info = default;

        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return false;

        var mouse = ImGui.GetMousePos();
        float mouseX = mouse.X;
        float mouseY = mouse.Y;
        if (mouseX < vpX || mouseX > vpX + vpW || mouseY < vpY || mouseY > vpY + vpH)
            return false;

        float aspect = vpW / Math.Max(vpH, 1f);
        var view = _camera.GetViewMatrix();
        float farPlane = GetSceneFarPlane();
        var proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, farPlane);

        float localX = mouseX - vpX;
        float localY = mouseY - vpY;
        float ndcX = (localX / vpW) * 2f - 1f;
        float ndcY = 1f - (localY / vpH) * 2f;

        var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
        return TryRaycastTerrain(renderer, rayOrigin, rayDir, farPlane, out info);
    }

    private bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info)
    {
        return TryRaycastTerrain(renderer, rayOrigin, rayDir, maxDistance, out info, out _);
    }

    private bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info, out Vector3 hitPoint)
    {
        info = default;
        hitPoint = default;

        const float step = 16f;
        int maxSteps = (int)MathF.Ceiling(maxDistance / step);
        maxSteps = Math.Clamp(maxSteps, 16, 1024);

        float prevT = 0f;
        float prevD = float.NaN;

        for (int i = 0; i <= maxSteps; i++)
        {
            float t = i * step;
            var p = rayOrigin + rayDir * t;

            if (!TrySampleTerrainHeightLoaded(renderer, p.X, p.Y, out float height, out var curInfo))
                continue;

            float d = p.Z - height;
            if (!float.IsNaN(prevD))
            {
                if (prevD > 0f && d <= 0f)
                {
                    float a = prevT;
                    float b = t;
                    TerrainRenderer.TerrainChunkInfo best = curInfo;
                    for (int it = 0; it < 10; it++)
                    {
                        float m = (a + b) * 0.5f;
                        var pm = rayOrigin + rayDir * m;
                        if (!TrySampleTerrainHeightLoaded(renderer, pm.X, pm.Y, out float hm, out var mi))
                        {
                            a = m;
                            continue;
                        }

                        best = mi;
                        float dm = pm.Z - hm;
                        if (dm > 0f)
                            a = m;
                        else
                            b = m;
                    }

                    float hitDistance = (a + b) * 0.5f;
                    hitPoint = rayOrigin + rayDir * hitDistance;
                    info = best;
                    return true;
                }
            }

            prevT = t;
            prevD = d;
        }

        return false;
    }

    private float GetSceneFarPlane()
    {
        if (_terrainManager != null)
            return ComputeSceneFarPlane(_terrainManager.Lighting.FogEnd);

        if (_vlmTerrainManager != null)
            return ComputeSceneFarPlane(_vlmTerrainManager.Lighting.FogEnd);

        return 10000f;
    }

    internal static float ComputeSceneFarPlane(float fogEnd)
    {
        float safeFogEnd = float.IsFinite(fogEnd) && fogEnd > 0f ? fogEnd : 1500f;
        return Math.Clamp(safeFogEnd + TerrainFarPlanePadding, MinTerrainFarPlane, MaxTerrainFarPlane);
    }

    private bool TrySampleTerrainHeightLoaded(TerrainRenderer renderer, float worldX, float worldY, out float height, out TerrainRenderer.TerrainChunkInfo info)
    {
        height = 0f;
        info = default;

        var ci = renderer.GetChunkInfoAt(worldX, worldY);
        if (!ci.HasValue)
            return false;

        info = ci.Value;
        if (!TryGetChunkDataLoadedOnly(info.TileX, info.TileY, info.ChunkX, info.ChunkY, out var chunk))
            return false;

        float localX = chunk.WorldPosition.Y - worldY;
        float localY = chunk.WorldPosition.X - worldX;
        localX = Math.Clamp(localX, 0f, WoWConstants.ChunkSize);
        localY = Math.Clamp(localY, 0f, WoWConstants.ChunkSize);

        height = TerrainChunkMath.SampleHeightOuterGrid(chunk, localX, localY);
        return true;
    }

    private bool TryGetChunkDataLoadedOnly(int tileX, int tileY, int chunkX, int chunkY, out Terrain.TerrainChunkData chunk)
    {
        chunk = new Terrain.TerrainChunkData();

        List<Terrain.TerrainChunkData>? chunks = null;
        if (_terrainManager != null)
        {
            if (!_terrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
                return false;
            chunks = tile.Chunks;
        }
        else if (_vlmTerrainManager != null)
        {
            if (!_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
                return false;
            chunks = tile.Chunks;
        }

        if (chunks == null || chunks.Count == 0)
            return false;

        var found = chunks.FirstOrDefault(c => c != null && c.ChunkX == chunkX && c.ChunkY == chunkY);
        if (found == null || found.Heights == null || found.Heights.Length < 145)
            return false;

        chunk = found;
        return true;
    }

    private string GetProjectOutputRootDirectory()
    {
        if (string.IsNullOrWhiteSpace(_projectOutputRootDir))
            _projectOutputRootDir = ProjectsDir;

        return Path.GetFullPath(_projectOutputRootDir);
    }

    private void HandleProjectOutputRootChanged()
    {
        _editorProjectOutputDir = string.Empty;
        _editorProjectSourceKey = string.Empty;
        _mapConvertOutputDir = string.Empty;
        _mapConvertProjectSourceKey = string.Empty;
        _placementEditing.RefreshProjectManagedPlacementTargets();

        if (!string.IsNullOrWhiteSpace(_mapConvertSourcePath))
            _converterDialogs.EnsureMapConverterProjectOutputDirectory(forceNew: false);
    }

    internal static string SanitizeProjectPathSegment(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "project";

        char[] invalid = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Trim().Length);
        foreach (char c in value.Trim())
        {
            builder.Append(Array.IndexOf(invalid, c) >= 0 || char.IsControl(c)
                ? '_'
                : char.IsWhiteSpace(c) ? '_' : c);
        }

        string sanitized = builder.ToString().Trim('.', ' ');
        return string.IsNullOrWhiteSpace(sanitized) ? "project" : sanitized;
    }

    internal static string CreateTimestampedProjectOutputDirectory(string rootDirectory, string projectName)
    {
        string safeProjectName = SanitizeProjectPathSegment(projectName);
        string projectRoot = Path.Combine(rootDirectory, safeProjectName);
        Directory.CreateDirectory(projectRoot);

        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string candidate = Path.Combine(projectRoot, timestamp);
        int suffix = 1;
        while (Directory.Exists(candidate))
        {
            candidate = Path.Combine(projectRoot, $"{timestamp}_{suffix:D2}");
            suffix++;
        }

        return candidate;
    }

    private string GetEditorProjectName(string? fallbackName = null)
    {
        if (!string.IsNullOrWhiteSpace(GetCurrentSessionMapName()))
            return SanitizeProjectPathSegment(GetCurrentSessionMapName()!);

        string? wdtPath = TryGetLoadedLocalWdtPath();
        if (!string.IsNullOrWhiteSpace(wdtPath))
            return SanitizeProjectPathSegment(Path.GetFileNameWithoutExtension(wdtPath));

        if (!string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath) && File.Exists(_lastWorldSceneWdtPath))
            return SanitizeProjectPathSegment(Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath));

        if (!string.IsNullOrWhiteSpace(fallbackName))
            return SanitizeProjectPathSegment(fallbackName);

        return "project";
    }

    private string? GetEditorProjectSourceKey()
    {
        string? wdtPath = TryGetLoadedLocalWdtPath();
        if (!string.IsNullOrWhiteSpace(wdtPath))
            return Path.GetFullPath(wdtPath);

        if (!string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath) && File.Exists(_lastWorldSceneWdtPath))
            return Path.GetFullPath(_lastWorldSceneWdtPath);

        if (!string.IsNullOrWhiteSpace(_loadedFilePath) && File.Exists(_loadedFilePath))
            return Path.GetFullPath(_loadedFilePath);

        string? currentMapName = GetCurrentSessionMapName();
        return string.IsNullOrWhiteSpace(currentMapName) ? null : $"map:{currentMapName}";
    }

    private string EnsureEditorProjectOutputDirectory(bool forceNew = false)
    {
        string sourceKey = GetEditorProjectSourceKey() ?? $"editor:{GetEditorProjectName()}";
        if (!forceNew
            && !string.IsNullOrWhiteSpace(_editorProjectOutputDir)
            && string.Equals(_editorProjectSourceKey, sourceKey, StringComparison.OrdinalIgnoreCase))
        {
            return _editorProjectOutputDir;
        }

        _editorProjectSourceKey = sourceKey;
        _editorProjectOutputDir = CreateTimestampedProjectOutputDirectory(GetProjectOutputRootDirectory(), GetEditorProjectName());
        return _editorProjectOutputDir;
    }

    private string DescribeEditorProjectOutputDirectory()
    {
        if (!string.IsNullOrWhiteSpace(_editorProjectOutputDir))
            return _editorProjectOutputDir;

        return Path.Combine(GetProjectOutputRootDirectory(), GetEditorProjectName(), "<timestamp>");
    }

    private void StartNewEditorProjectOutputDirectory()
    {
        _editorProjectOutputDir = EnsureEditorProjectOutputDirectory(forceNew: true);
        _placementEditing.RefreshProjectManagedPlacementTargets();
        _selectedPlacementSaveStatus = $"Created new project output folder: {_editorProjectOutputDir}";
    }

    /// <summary>
    /// Canonical Scene > Placements body. Keep this list-only so scene
    /// navigation does not also become the owner for diagnostics or tools.
    /// </summary>
    private void DrawPlacementListsContent()
    {
        if (_worldScene == null)
            return;

        if (_worldScene.ModfPlacements.Count > 0 && ImGui.TreeNode($"WMO Placements ({_worldScene.ModfPlacements.Count})"))
        {
            if (ImGui.BeginChild("##CanonicalWmoPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(_worldScene.ModfPlacements.Count, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.ModfPlacements[i];
                    string name = p.NameIndex < _worldScene.WmoModelNames.Count
                        ? Path.GetFileName(_worldScene.WmoModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick)
                        && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    {
                        _camera.Position = p.Position + new Vector3(0, 0, 50);
                        _camera.Pitch = -30f;
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Flags: 0x{p.Flags:X4}");
                        ImGui.Text($"Bounds: ({p.BoundsMin.X:F0},{p.BoundsMin.Y:F0},{p.BoundsMin.Z:F0}) - ({p.BoundsMax.X:F0},{p.BoundsMax.Y:F0},{p.BoundsMax.Z:F0})");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < _worldScene.ModfPlacements.Count)
                    ImGui.Dummy(new Vector2(0, (_worldScene.ModfPlacements.Count - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        int mddfCount = _worldScene.MddfPlacements.Count;
        int mddfShow = Math.Min(mddfCount, 200);
        if (mddfCount > 0 && ImGui.TreeNode($"MDX Placements ({mddfCount}{(mddfCount > mddfShow ? $", showing {mddfShow}" : "")})"))
        {
            if (ImGui.BeginChild("##CanonicalMdxPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(mddfShow, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.MddfPlacements[i];
                    string name = p.NameIndex < _worldScene.MdxModelNames.Count
                        ? Path.GetFileName(_worldScene.MdxModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name} s={p.Scale:F2}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick)
                        && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    {
                        _camera.Position = p.Position + new Vector3(0, 0, 20);
                        _camera.Pitch = -30f;
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Scale: {p.Scale:F3}");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < mddfShow)
                    ImGui.Dummy(new Vector2(0, (mddfShow - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        if (mddfCount == 0 && _worldScene.ModfPlacements.Count == 0)
            ImGui.TextDisabled("No WMO or MDX placements are loaded for the current world.");
    }

    private void DrawWorldObjectsContentCore()
    {
        if (_worldScene == null) return;

        _placementEditing.DrawSelectedPlacementEditControls();
        DrawVisualInvestigationToolbox(showWorldObjectRangeControls: true);
        ImGui.Separator();
        DrawTerrainChunkInvestigationPanel(defaultOpen: _visualInvestigationMode == VisualInvestigationMode.Adt);
        ImGui.Separator();

        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

        ImGui.Separator();
        DrawPopulationSubTabContent();

        bool showPm4Overlay = _worldScene.Pm4Overlay.ShowPm4Overlay;
        if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
            _worldScene.Pm4Overlay.ShowPm4Overlay = showPm4Overlay;
        if (ImGui.IsItemHovered() && _worldScene.Pm4Overlay.ShowPm4Overlay)
            ImGui.SetTooltip(_worldScene.Pm4Overlay.Pm4Status);

        DrawToolbarPopupButton("PM4 Actions", string.Empty, "##Pm4OverlayActionsPopup", () =>
        {
            if (ImGui.Button("PM4 Workbench"))
            {
                OpenPm4Workbench(_worldScene.Pm4Overlay.HasSelectedPm4Object ? Pm4WorkbenchTab.Selection : Pm4WorkbenchTab.Overlay);
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button("Reload PM4"))
            {
                _worldScene.Pm4Overlay.ReloadPm4Overlay();
                ImGui.CloseCurrentPopup();
            }
        });

        if (_worldScene.Pm4Overlay.IsPm4Loading)
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), $"PM4 loading... {_worldScene.Pm4Overlay.Pm4Status}");
        else if (_worldScene.Pm4Overlay.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4Overlay.Pm4VisibleObjectCount}/{_worldScene.Pm4Overlay.Pm4ObjectCount} visible objects, {_worldScene.Pm4Overlay.Pm4VisibleLineCount}/{_worldScene.Pm4Overlay.Pm4LineCount} lines, {_worldScene.Pm4Overlay.Pm4VisibleTriangleCount}/{_worldScene.Pm4Overlay.Pm4TriangleCount} tris");
        else
            ImGui.TextDisabled("PM4 stays lightweight here. Use the inspector workbench for overlay tuning, object matches, and correlation.");
        if (_worldScene.Pm4Overlay.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4 status: {_worldScene.Pm4Overlay.Pm4Status}");
        ImGui.TextDisabled("PM4 settings and deep analysis live in Inspector > PM4 Workbench.");

        ImGui.Separator();

        // POI toggle — lazy-loaded on first request
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0)
        {
            bool showPoi = _worldScene.ShowPoi;
            if (ImGui.Checkbox($"Area POIs ({_worldScene.PoiLoader.Entries.Count})", ref showPoi))
                _worldScene.ShowPoi = showPoi;
        }
        else if (!_worldScene.PoiLoadAttempted)
        {
            DrawToolbarPopupButton("POI Actions", "load", "##PoiActionsPopup", () =>
            {
                if (ImGui.Button("Load Area POIs"))
                {
                    _worldScene.ShowPoi = true;
                    ImGui.CloseCurrentPopup();
                }
            });
        }
        else if (_worldScene.PoiLoadAttempted && (_worldScene.PoiLoader == null || _worldScene.PoiLoader.Entries.Count == 0))
        {
            ImGui.TextDisabled("Area POIs: none found");
        }

        ImGui.Separator();

        string taxiSummary = _worldScene.SelectedTaxiRouteId >= 0
            ? $"route {_worldScene.SelectedTaxiRouteId}"
            : _worldScene.SelectedTaxiNodeId >= 0
                ? $"node {_worldScene.SelectedTaxiNodeId}"
                : _taxiRideCameraEnabled
                    ? "ride active"
                    : _worldScene.ShowTaxi
                        ? "visible"
                        : string.Empty;
        // Taxi panel is accessed via the Utilities workbench tab only.
        // The toolbar popup was removed because ImGui popups have no title bar and
        // dismiss on any outside click, making route selection impossible.

        // WL loose liquid files (WLW/WLQ/WLM) — lazy-loaded on first toggle
        if (_worldScene.WlLoader != null && _worldScene.WlLoader.HasData)
        {
            bool showWl = _worldScene.ShowWlLiquids;
            if (ImGui.Checkbox($"WL Liquids ({_worldScene.WlLoader.Bodies.Count})", ref showWl))
                _worldScene.ShowWlLiquids = showWl;
            if (_worldScene.ShowWlLiquids && ImGui.IsItemHovered())
                ImGui.SetTooltip("Loose WLW/WLQ/WLM liquid project files.\nContains water data for deleted/missing tiles.");

            if (liquidRenderer != null && ImGui.TreeNode("WL Bodies"))
            {
                int visibleCount = 0;
                foreach (var b in _worldScene.WlLoader.Bodies)
                {
                    if (liquidRenderer.IsWlBodyVisible(b.BodyKey))
                        visibleCount++;
                }

                bool hasSelected = !string.IsNullOrWhiteSpace(_wlLayerSelectedBodyKey);
                DrawToolbarPopupButton("WL Body Actions", string.Empty, "##WlBodyActionsPopup", () =>
                {
                    if (ImGui.Button("Show All"))
                    {
                        liquidRenderer.SetAllWlBodiesVisible(true);
                        ImGui.CloseCurrentPopup();
                    }

                    if (ImGui.Button("Hide All"))
                    {
                        liquidRenderer.SetAllWlBodiesVisible(false);
                        ImGui.CloseCurrentPopup();
                    }

                    if (!hasSelected)
                        ImGui.BeginDisabled();
                    if (ImGui.Button("Solo Selected"))
                    {
                        liquidRenderer.SetAllWlBodiesVisible(false);
                        liquidRenderer.SetWlBodyVisible(_wlLayerSelectedBodyKey, true);
                        ImGui.CloseCurrentPopup();
                    }
                    if (!hasSelected)
                        ImGui.EndDisabled();

                    if (IsWlListIsolationActive && ImGui.Button("Clear List Isolation"))
                    {
                        _wlLayerListIsolationEnabled = false;
                        ImGui.CloseCurrentPopup();
                    }
                });

                ImGui.TextDisabled($"Visible: {visibleCount}/{_worldScene.WlLoader.Bodies.Count}");

                if (ImGui.BeginTable("##wl_layers", 4, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp))
                {
                    ImGui.TableSetupColumn("V", ImGuiTableColumnFlags.WidthFixed, 24f);
                    ImGui.TableSetupColumn("Type", ImGuiTableColumnFlags.WidthFixed, 48f);
                    ImGui.TableSetupColumn("Group", ImGuiTableColumnFlags.WidthFixed, 72f);
                    ImGui.TableSetupColumn("Layer", ImGuiTableColumnFlags.WidthStretch);
                    ImGui.TableHeadersRow();

                    for (int i = 0; i < _worldScene.WlLoader.Bodies.Count; i++)
                    {
                        var body = _worldScene.WlLoader.Bodies[i];
                        if (!ShouldIncludeWlBodyInUiList(body))
                            continue;

                        ImGui.TableNextRow();

                        ImGui.TableSetColumnIndex(0);
                        bool visible = liquidRenderer.IsWlBodyVisible(body.BodyKey);
                        if (ImGui.Checkbox($"##wl_vis_{i}", ref visible))
                            liquidRenderer.SetWlBodyVisible(body.BodyKey, visible);

                        ImGui.TableSetColumnIndex(1);
                        ImGui.TextUnformatted(body.FileType.ToString());

                        ImGui.TableSetColumnIndex(2);
                        ImGui.TextUnformatted(body.GroupLabel);

                        ImGui.TableSetColumnIndex(3);
                        bool isSelected = string.Equals(_wlLayerSelectedBodyKey, body.BodyKey, StringComparison.OrdinalIgnoreCase);
                        string label = $"{body.Name}##wl_layer_{i}";
                        if (ImGui.Selectable(label, isSelected, ImGuiSelectableFlags.SpanAllColumns))
                            SetSelectedWlLiquidBody(body, isolateInList: false, focusInspectWorkspace: false);
                        if (ImGui.IsItemHovered())
                        {
                            ImGui.BeginTooltip();
                            ImGui.TextUnformatted(body.SourcePath);
                            ImGui.Text($"Blocks: {body.BlockCount}  Verts: {body.Vertices.Length}");
                            ImGui.Text($"Mode: {body.GroupLabel}  Z: {body.MinHeight:F1}..{body.MaxHeight:F1}");
                            ImGui.EndTooltip();
                        }
                    }

                    ImGui.EndTable();
                }

                ImGui.TreePop();
            }

            if (ImGui.TreeNode("WL Transform Tuning"))
            {
                var ts = WlLiquidLoader.TransformSettings;

                bool enabled = ts.Enabled;
                if (ImGui.Checkbox("Enable Transform", ref enabled))
                    ts.Enabled = enabled;

                bool swapXY = ts.SwapXYBeforeRotation;
                if (ImGui.Checkbox("Swap XY Before Rotation", ref swapXY))
                    ts.SwapXYBeforeRotation = swapXY;

                var rot = ts.RotationDegrees;
                if (ImGui.InputFloat3("Rotation (deg)", ref rot, "%.3f"))
                    ts.RotationDegrees = rot;

                var tr = ts.Translation;
                if (ImGui.InputFloat3("Translation", ref tr, "%.3f"))
                    ts.Translation = tr;

                WlLiquidLoader.WlBodyGroupingMode groupingMode = ts.GroupingMode;
                if (ImGui.BeginCombo("Grouping", GetWlLiquidGroupingModeLabel(groupingMode)))
                {
                    foreach (WlLiquidLoader.WlBodyGroupingMode option in Enum.GetValues<WlLiquidLoader.WlBodyGroupingMode>())
                    {
                        bool isSelected = option == groupingMode;
                        if (ImGui.Selectable(GetWlLiquidGroupingModeLabel(option), isSelected))
                            ts.GroupingMode = option;
                        if (isSelected)
                            ImGui.SetItemDefaultFocus();
                    }

                    ImGui.EndCombo();
                }

                float planeHeightTolerance = ts.PlaneHeightTolerance;
                if (ImGui.SliderFloat("Plane Weld Tolerance", ref planeHeightTolerance, 0.05f, 4.00f, "%.2f"))
                    ts.PlaneHeightTolerance = planeHeightTolerance;

                DrawToolbarPopupButton("WL Transform Actions", string.Empty, "##WlTransformActionsPopup", () =>
                {
                    if (ImGui.Button("Apply + Reload WL"))
                    {
                        _worldScene.ReloadWlLiquids();
                        ImGui.CloseCurrentPopup();
                    }

                    if (ImGui.Button("Print Current WL Transform"))
                    {
                        ViewerLog.Important(ViewerLog.Category.Terrain,
                            $"[WL Transform] Enabled={ts.Enabled} SwapXY={ts.SwapXYBeforeRotation} " +
                            $"Rot=({ts.RotationDegrees.X:F1},{ts.RotationDegrees.Y:F1},{ts.RotationDegrees.Z:F1}) " +
                            $"Trans=({ts.Translation.X:F1},{ts.Translation.Y:F1},{ts.Translation.Z:F1})");
                        ImGui.CloseCurrentPopup();
                    }
                });

                ImGui.TextDisabled("Tune here, then share the printed values to hard-wire final config.");
                ImGui.TreePop();
            }
        }

        if (_worldScene.LitLoader != null && _worldScene.LitLoader.HasData)
        {
            bool showLitLights = _worldScene.ShowLitLights;
            if (ImGui.Checkbox($"LIT Lights ({_worldScene.LitLoader.Lights.Count})", ref showLitLights))
                _worldScene.ShowLitLights = showLitLights;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Alpha-era lights.lit placement overlay. Pins show light origins; boxes show approximate influence radius.");

            bool useLitFogOverride = _worldScene.UseLitFogOverride;
            if (ImGui.Checkbox("Use LIT Lighting Override", ref useLitFogOverride))
                _worldScene.UseLitFogOverride = useLitFogOverride;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Experimental: apply the selected LIT profile over the viewer's always-present global lighting path.");

            if (_worldScene.LastLitSample != null)
                ImGui.TextDisabled($"LIT sample: {_worldScene.LastLitSample.DominantLightName}  fogEnd={_worldScene.LastLitSample.FogEnd:F1}");
            else
                ImGui.TextDisabled(_worldScene.LitStatus);
        }
        else if (!_worldScene.LitLoadAttempted)
        {
            DrawToolbarPopupButton("LIT Actions", "load", "##LitActionsPopup", () =>
            {
                if (ImGui.Button("Load LIT Lights"))
                {
                    _worldScene.ShowLitLights = true;
                    ImGui.CloseCurrentPopup();
                }
            });
        }
        else
        {
            ImGui.TextDisabled(_worldScene.LitStatus);
        }

        bool objectFogEnabled = _worldScene.ObjectFogEnabled;
        if (ImGui.Checkbox("Fog Objects", ref objectFogEnabled))
            _worldScene.ObjectFogEnabled = objectFogEnabled;

        bool showHoverTooltips = _worldScene.ShowHoveredAssetTooltips;
        if (ImGui.Checkbox("Hover Tooltips", ref showHoverTooltips))
            _worldScene.ShowHoveredAssetTooltips = showHoverTooltips;

        bool limitHoverPickRange = _worldScene.LimitHoveredAssetRange;
        if (ImGui.Checkbox("Limit Hover/Pick Range", ref limitHoverPickRange))
            _worldScene.LimitHoveredAssetRange = limitHoverPickRange;

        if (_worldScene.LimitHoveredAssetRange)
        {
            bool useDynamicHoverRange = _worldScene.UseDynamicHoveredAssetRange;
            if (ImGui.Checkbox("Dynamic Hover Range", ref useDynamicHoverRange))
                _worldScene.UseDynamicHoveredAssetRange = useDynamicHoverRange;

            float hoverPickRange = _worldScene.HoveredAssetMaxDistance;
            if (ImGui.SliderFloat("Hover/Pick Range", ref hoverPickRange, 100f, MaxTerrainFogDistance, "%.2f yd"))
                _worldScene.HoveredAssetMaxDistance = hoverPickRange;

            ImGui.TextDisabled($"Effective range: {_worldScene.EffectiveHoveredAssetMaxDistance:F2} yd");
        }

        bool showSelectedObjectBounds = _worldScene.ShowSelectedObjectBounds;
        if (ImGui.Checkbox("Show Selected Object Bounds", ref showSelectedObjectBounds))
            _worldScene.ShowSelectedObjectBounds = showSelectedObjectBounds;

        DrawObjectPathFilterControls();

        ImGui.TextDisabled("UniqueId ranges and playback are in Tools > Archeology.");

        if (!_worldScene.WlLoadAttempted)
        {
            if (ImGui.Button("Load WL Liquids"))
            {
                _worldScene.ShowWlLiquids = true;
            }
        }
        else if (_worldScene.WlLoadAttempted && (_worldScene.WlLoader == null || !_worldScene.WlLoader.HasData))
        {
            ImGui.TextDisabled("WL Liquids: none found");
        }

        // AreaTriggers — lazy-loaded on first request
        if (_worldScene.AreaTriggerLoader != null && _worldScene.AreaTriggerLoader.Count > 0)
        {
            bool showTriggers = _worldScene.ShowAreaTriggers;
            if (ImGui.Checkbox($"AreaTriggers ({_worldScene.AreaTriggerLoader.Count})", ref showTriggers))
                _worldScene.ShowAreaTriggers = showTriggers;
            if (_worldScene.ShowAreaTriggers && ImGui.IsItemHovered())
                ImGui.SetTooltip("Instance portals, event markers, and script triggers.\nGreen spheres/boxes from AreaTrigger.dbc");
        }
        else if (!_worldScene.AreaTriggerLoadAttempted)
        {
            if (ImGui.Button("Load AreaTriggers"))
            {
                _worldScene.ShowAreaTriggers = true;
            }
        }
        else if (_worldScene.AreaTriggerLoadAttempted && (_worldScene.AreaTriggerLoader == null || _worldScene.AreaTriggerLoader.Count == 0))
        {
            ImGui.TextDisabled("AreaTriggers: none found");
        }

        // WMO placements
        if (_worldScene.ModfPlacements.Count > 0 && ImGui.TreeNode($"WMO Placements ({_worldScene.ModfPlacements.Count})"))
        {
            if (ImGui.BeginChild("##WmoPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(_worldScene.ModfPlacements.Count, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.ModfPlacements[i];
                    string name = p.NameIndex < _worldScene.WmoModelNames.Count
                        ? Path.GetFileName(_worldScene.WmoModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = p.Position + new System.Numerics.Vector3(0, 0, 50);
                            _camera.Pitch = -30f;
                        }
                    }
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Flags: 0x{p.Flags:X4}");
                        ImGui.Text($"Bounds: ({p.BoundsMin.X:F0},{p.BoundsMin.Y:F0},{p.BoundsMin.Z:F0}) - ({p.BoundsMax.X:F0},{p.BoundsMax.Y:F0},{p.BoundsMax.Z:F0})");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < _worldScene.ModfPlacements.Count)
                    ImGui.Dummy(new Vector2(0, (_worldScene.ModfPlacements.Count - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        // MDX placements (show first 200 to avoid UI lag)
        int mddfCount = _worldScene.MddfPlacements.Count;
        int mddfShow = Math.Min(mddfCount, 200);
        if (mddfCount > 0 && ImGui.TreeNode($"MDX Placements ({mddfCount}{(mddfCount > mddfShow ? $", showing {mddfShow}" : "")})"))
        {
            if (ImGui.BeginChild("##MdxPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(mddfShow, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.MddfPlacements[i];
                    string name = p.NameIndex < _worldScene.MdxModelNames.Count
                        ? Path.GetFileName(_worldScene.MdxModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name} s={p.Scale:F2}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = p.Position + new System.Numerics.Vector3(0, 0, 20);
                            _camera.Pitch = -30f;
                        }
                    }
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Scale: {p.Scale:F3}");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < mddfShow)
                    ImGui.Dummy(new Vector2(0, (mddfShow - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        // Area POI list
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0 &&
            ImGui.TreeNode($"Area POIs ({_worldScene.PoiLoader.Entries.Count})"))
        {
            if (ImGui.BeginChild("##AreaPoiList", new Vector2(0, 200f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                int poiCount = _worldScene.PoiLoader.Entries.Count;
                GetVisibleListRange(poiCount, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var poi = _worldScene.PoiLoader.Entries[i];
                    string label = $"[{poi.Id}] {poi.Name}";
                    bool isSelected = _selectedAreaPoiId == poi.Id;
                    if (ImGui.Selectable(label, isSelected, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        SelectAreaPoi(poi.Id, toggle: false);
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = poi.Position + new System.Numerics.Vector3(0, 0, 50);
                            _camera.Pitch = -30f;
                        }
                    }
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({poi.Position.X:F1}, {poi.Position.Y:F1}, {poi.Position.Z:F1})");
                        ImGui.Text($"WoW Pos: ({poi.WoWPosition.X:F1}, {poi.WoWPosition.Y:F1}, {poi.WoWPosition.Z:F1})");
                        ImGui.Text($"Icon: {poi.Icon}  Importance: {poi.Importance}  Flags: 0x{poi.Flags:X}");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < poiCount)
                    ImGui.Dummy(new Vector2(0, (poiCount - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

    }

    private void LoadSqlSpawnsForCurrentMap()
    {
        if (_worldScene == null)
        {
            _sqlSpawnStatus = "No world loaded.";
            return;
        }

        if (_currentMapId < 0)
        {
            _sqlSpawnStatus = "Current map ID unavailable.";
            return;
        }

        if (string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
        {
            _sqlSpawnStatus = "Enter alpha-core root path first.";
            return;
        }

        try
        {
            if (_sqlPopulationService == null ||
                !string.Equals(_sqlServiceRoot, _sqlAlphaCoreRoot, StringComparison.OrdinalIgnoreCase))
            {
                _sqlPopulationService?.Dispose();
                _sqlPopulationService = new SqlWorldPopulationService(_sqlAlphaCoreRoot);
                _sqlServiceRoot = _sqlAlphaCoreRoot;
            }

            var (ok, message) = _sqlPopulationService.Validate();
            if (!ok)
            {
                _sqlSpawnStatus = message;
                return;
            }

            _sqlSpawnStatus = "Parsing SQL and building spawn list...";

            int requestedMax = (_sqlUseAoiFilter || _sqlStreamWithCamera) ? 0 : _sqlMaxSpawns;
            var mapSpawns = _sqlPopulationService
                .LoadMapSpawnsAsync(_currentMapId, requestedMax, _sqlIncludeCreatures, _sqlIncludeGameObjects)
                .GetAwaiter()
                .GetResult();

            _sqlMapSpawnsCache = mapSpawns.ToList();
            _sqlMapSpawnsCacheMapId = _currentMapId;
            _sqlLastCameraTile = null;
            _sqlForceStreamRefresh = true;

            ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: true);
        }
        catch (Exception ex)
        {
            _sqlSpawnStatus = $"Error: {ex.Message}";
        }
    }

    private void ApplySqlSpawnsToScene(IReadOnlyList<WorldSpawnRecord> mapSpawns, bool updateStatus)
    {
        if (_worldScene == null)
            return;

        _worldScene.SqlGameObjectMdxScaleMultiplier = _sqlGameObjectMdxScaleMultiplier;

        IReadOnlyList<WorldSpawnRecord> finalSpawns = mapSpawns;
        if (_sqlUseAoiFilter)
            finalSpawns = FilterSpawnsToCameraAoi(mapSpawns, _sqlAoiTileRadius, _sqlMaxSpawns);
        else if (_sqlMaxSpawns > 0 && mapSpawns.Count > _sqlMaxSpawns)
            finalSpawns = mapSpawns.Take(_sqlMaxSpawns).ToList();

        _worldScene.SetExternalSpawns(finalSpawns);

        if (updateStatus)
        {
            _sqlSpawnStatus = _sqlUseAoiFilter
                ? $"Loaded {finalSpawns.Count}/{mapSpawns.Count} SQL spawns for map {_currentMapId} (AOI radius {_sqlAoiTileRadius} tiles{(_sqlStreamWithCamera ? ", streaming" : "")})."
                : $"Loaded {finalSpawns.Count} SQL spawns for map {_currentMapId}.";
        }
    }

    private List<WorldSpawnRecord> FilterSpawnsToCameraAoi(IReadOnlyList<WorldSpawnRecord> spawns, int tileRadius, int maxCount)
    {
        if (spawns.Count == 0) return new List<WorldSpawnRecord>();

        float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize;
        float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize;

        var inRange = new List<(WorldSpawnRecord spawn, float distSq)>();
        foreach (var spawn in spawns)
        {
            var pos = SqlSpawnCoordinateConverter.ToRendererPosition(spawn.PositionWow);
            float spawnTileX = (WoWConstants.MapOrigin - pos.X) / WoWConstants.ChunkSize;
            float spawnTileY = (WoWConstants.MapOrigin - pos.Y) / WoWConstants.ChunkSize;

            if (MathF.Abs(spawnTileX - camTileX) > tileRadius || MathF.Abs(spawnTileY - camTileY) > tileRadius)
                continue;

            float dx = pos.X - _camera.Position.X;
            float dy = pos.Y - _camera.Position.Y;
            float dz = pos.Z - _camera.Position.Z;
            inRange.Add((spawn, dx * dx + dy * dy + dz * dz));
        }

        inRange.Sort((a, b) => a.distSq.CompareTo(b.distSq));

        int take = maxCount > 0 ? Math.Min(maxCount, inRange.Count) : inRange.Count;
        var result = new List<WorldSpawnRecord>(take);
        for (int i = 0; i < take; i++)
            result.Add(inRange[i].spawn);

        return result;
    }


    private void RefreshFileList()
    {
        if (_dataSource == null) return;

        var allFiles = GetFilesForBrowserFilter();
        IEnumerable<string> candidates = allFiles;
        if (!string.IsNullOrEmpty(_searchFilter))
            candidates = candidates.Where(f => f.Contains(_searchFilter, StringComparison.OrdinalIgnoreCase));

        var filtered = new List<string>(capacity: 5000);
        foreach (string file in candidates)
        {
            if (!_dataSource.FileExists(file))
                continue;

            filtered.Add(file);
            if (filtered.Count >= 5000)
                break;
        }

        _filteredFiles = filtered;

        _selectedFileIndex = -1;
    }

    private IReadOnlyList<string> GetFilesForBrowserFilter()
    {
        if (_dataSource == null)
            return Array.Empty<string>();

        if (!_extensionFilter.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
            return _dataSource.GetFileList(_extensionFilter);

        var combined = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string extension in EarlyModelBrowserExtensions)
        {
            foreach (string file in _dataSource.GetFileList(extension))
            {
                if (seen.Add(file))
                    combined.Add(file);
            }
        }

        return combined;
    }

    private void RefreshDiscoveredMaps()
    {
        if (_dataSource == null)
        {
            _discoveredMaps.Clear();
            _autoOpenWorldMapsPanel = false;
            return;
        }

        int previousDiscoveredMapCount = _discoveredMaps.Count;

        if (_dbcProvider != null && !string.IsNullOrWhiteSpace(_dbdDir) && !string.IsNullOrWhiteSpace(_dbcBuild))
        {
            var mapDiscovery = new MapDiscoveryService(_dbcProvider, _dbdDir!, _dbcBuild!, _dataSource);
            _discoveredMaps = mapDiscovery.DiscoverMaps();
            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"Discovered {_discoveredMaps.Count} maps via Map.dbc/data source ({_discoveredMaps.Count(m => m.HasWdt)} with WDTs, {_discoveredMaps.Count(m => !m.HasDbcEntry)} custom loose maps)");
        }
        else
        {
            _discoveredMaps = MapDiscoveryService.DiscoverLooseMapsOnly(_dataSource);
            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"Discovered {_discoveredMaps.Count} loose maps without Map.dbc metadata.");
        }

        _autoOpenWorldMapsPanel = _discoveredMaps.Count > 0 && previousDiscoveredMapCount == 0;
        WarmDiscoveredWdlPreviews();
    }

    private void LoadMpqDataSource(string gamePath, string? listfilePath, string? explicitBuildVersion = null, bool deferWorldReload = false)
    {
        _pendingDataSourceReloadGeneration = ++_activeDataSourceReloadGeneration;
        try
        {
            string? resolvedListfilePath = ResolveListfilePath(listfilePath);
            _statusMessage = $"Loading MPQ archives from {gamePath}...";
            StageCurrentWorldForDataSourceReload();
            ClearActiveSceneForDataSourceReload();
            _lastGameFolderPath = Path.GetFullPath(gamePath);
            _standaloneSkinPathCache.Clear();
            _loggedStandaloneMissingSkinPaths.Clear();
            _discoveredMaps.Clear();
            _areaTableService = null;
            ResetWdlPreviewSupport();
            _dataSource?.Dispose();
            _dataSource = new MpqDataSource(gamePath, resolvedListfilePath);
            _statusMessage = $"Loaded: {_dataSource.Name}";
            InitializeWdlPreviewSupport();

            // Load DBC tables directly from MPQ for replaceable texture resolution
            _texResolver = new ReplaceableTextureResolver();
            _texResolver.SetDataSource(_dataSource);
            _catalogView?.SetDataSource(_dataSource, _texResolver);
            var mpqDs = _dataSource as MpqDataSource;
            _dbcProvider = mpqDs != null
                ? new MpqDBCProvider(mpqDs.ArchiveReader, _dataSource)
                : new MpqDBCProvider(_dataSource);
            var dbcProvider = _dbcProvider;

            InitializeMinimapSupport();

            string? dbdDir = ResolveDbdDefinitionsDir();
            if (dbdDir != null)
            {
                _dbdDir = dbdDir;

                string buildAlias = explicitBuildVersion ?? InferBuildFromPath(gamePath, dbdDir);
                ViewerLog.Trace(explicitBuildVersion == null
                    ? $"[WoWViewer] Inferred build: '{buildAlias}' from path: {gamePath}"
                    : $"[WoWViewer] Using explicitly selected build: '{buildAlias}' for path: {gamePath}");
                
                if (!string.IsNullOrEmpty(buildAlias))
                {
                    _dbcBuild = buildAlias;
                    ViewerLog.Trace($"[WoWViewer] Loading DBCs via DBCD (build: {buildAlias}, DBDs: {dbdDir})");
                    _texResolver.LoadFromDBC(dbcProvider, dbdDir, buildAlias);

                    // Load AreaTable for area name display
                    _areaTableService = new AreaTableService();
                    _areaTableService.Load(dbcProvider, dbdDir, buildAlias);
                }
                else
                {
                    _dbcBuild = null;
                    ViewerLog.Trace("[WoWViewer] Could not determine build version. DBC texture resolution unavailable.");
                }
            }
            else
            {
                _dbcBuild = null;
                ViewerLog.Trace("[WoWViewer] WoWDBDefs definitions not found. DBC texture resolution unavailable.");
            }

            RefreshDiscoveredMaps();

            RefreshFileList();

            if (!deferWorldReload)
                RestoreWorldAfterDataSourceReload();
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to load MPQs: {ex.Message}";
        }
    }

    private void PrepareVlmExportDialogInputs()
    {
        string? activeGamePath = GetActiveGamePath();
        if (!string.IsNullOrWhiteSpace(activeGamePath))
            _vlmClientPath = activeGamePath;

        string? currentMapName = GetCurrentSessionMapName();
        if (!string.IsNullOrWhiteSpace(currentMapName))
            _vlmMapName = currentMapName;

        if (!string.IsNullOrWhiteSpace(_vlmClientPath) && !string.IsNullOrWhiteSpace(_vlmMapName) && string.IsNullOrWhiteSpace(_vlmOutputDir))
            _vlmOutputDir = DatasetExportDialogsService.GenerateVlmOutputPath(_vlmClientPath, _vlmMapName);
    }

    private void PrepareTerrainTextureTransferDialogInputs()
    {
        string? overlayMapDir = TryResolveCurrentMapDirectory(preferLooseOverlay: true);
        string? baseMapDir = TryResolveCurrentMapDirectory(preferLooseOverlay: false);

        if (!string.IsNullOrWhiteSpace(overlayMapDir))
            _terrainTransferSourceDir = overlayMapDir;

        if (!string.IsNullOrWhiteSpace(baseMapDir))
            _terrainTransferTargetDir = baseMapDir;
        else if (!string.IsNullOrWhiteSpace(overlayMapDir))
            _terrainTransferTargetDir = overlayMapDir;

        string? currentMapName = GetCurrentSessionMapName();
        bool usingDefaultOutput = string.IsNullOrWhiteSpace(_terrainTransferOutputDir)
            || string.Equals(_terrainTransferOutputDir, Path.Combine("output", "terrain-texture-transfer-ui"), StringComparison.OrdinalIgnoreCase);
        if (usingDefaultOutput && !string.IsNullOrWhiteSpace(currentMapName))
            _terrainTransferOutputDir = Path.Combine("output", "terrain-texture-transfer-ui", currentMapName);
    }

    private void PrepareMapConverterDialogInputs()
    {
        string? preferredWdt = TryGetLoadedLocalWdtPath();
        preferredWdt ??= TryResolveCurrentMapWdtPath(preferLooseOverlay: true);
        preferredWdt ??= TryResolveCurrentMapWdtPath(preferLooseOverlay: false);

        if (!string.IsNullOrWhiteSpace(preferredWdt))
            _mapConvertSourcePath = preferredWdt;

        string? preferredMapDir = TryResolveCurrentMapDirectory(preferLooseOverlay: true);
        preferredMapDir ??= TryResolveCurrentMapDirectory(preferLooseOverlay: false);
        if (!string.IsNullOrWhiteSpace(preferredMapDir))
            _mapConvertLkMapDir = preferredMapDir;

        if (!string.IsNullOrWhiteSpace(_mapConvertSourcePath))
            _converterDialogs.EnsureMapConverterProjectOutputDirectory(forceNew: false);
    }

    private void PrepareWmoConverterDialogInputs()
    {
        if (!string.IsNullOrEmpty(_loadedFilePath)
            && string.Equals(Path.GetExtension(_loadedFilePath), ".wmo", StringComparison.OrdinalIgnoreCase))
        {
            _wmoConvertSourcePath = _loadedFilePath;
        }
    }

    private string? GetActiveGamePath()
    {
        if (_dataSource is MpqDataSource mpqDataSource && !string.IsNullOrWhiteSpace(mpqDataSource.GamePath))
            return Path.GetFullPath(mpqDataSource.GamePath);

        if (!string.IsNullOrWhiteSpace(_lastGameFolderPath))
            return Path.GetFullPath(_lastGameFolderPath);

        return null;
    }

    private string? GetCurrentSessionMapName()
    {
        if (_terrainManager != null && !string.IsNullOrWhiteSpace(_terrainManager.MapName))
            return _terrainManager.MapName;

        if (_vlmTerrainManager != null && !string.IsNullOrWhiteSpace(_vlmTerrainManager.MapName))
            return _vlmTerrainManager.MapName;

        return null;
    }

    private string? TryResolveCurrentMapDirectory(bool preferLooseOverlay)
    {
        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return null;

        foreach (string root in EnumerateCurrentSessionRoots(preferLooseOverlay))
        {
            string? mapDirectory = TryResolveMapDirectoryUnderRoot(root, currentMapName);
            if (!string.IsNullOrWhiteSpace(mapDirectory))
                return mapDirectory;
        }

        return null;
    }

    private string? TryResolveCurrentMapWdtPath(bool preferLooseOverlay)
    {
        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return null;

        foreach (string root in EnumerateCurrentSessionRoots(preferLooseOverlay))
        {
            string? wdtPath = TryResolveMapWdtUnderRoot(root, currentMapName);
            if (!string.IsNullOrWhiteSpace(wdtPath))
                return wdtPath;
        }

        return null;
    }

    private IEnumerable<string> EnumerateCurrentSessionRoots(bool preferLooseOverlay)
    {
        var yielded = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            IEnumerable<string> overlayRoots = preferLooseOverlay
                ? mpqDataSource.OverlayRoots.Reverse()
                : mpqDataSource.OverlayRoots;

            foreach (string overlayRoot in overlayRoots)
            {
                string normalizedRoot = Path.GetFullPath(overlayRoot);
                if (yielded.Add(normalizedRoot))
                    yield return normalizedRoot;
            }

            string gamePath = Path.GetFullPath(mpqDataSource.GamePath);
            if (yielded.Add(gamePath))
                yield return gamePath;

            yield break;
        }

        if (!string.IsNullOrWhiteSpace(_lastLooseOverlayPath))
        {
            string looseRoot = Path.GetFullPath(_lastLooseOverlayPath);
            if (yielded.Add(looseRoot))
                yield return looseRoot;
        }

        if (!string.IsNullOrWhiteSpace(_lastGameFolderPath))
        {
            string gameRoot = Path.GetFullPath(_lastGameFolderPath);
            if (yielded.Add(gameRoot))
                yield return gameRoot;
        }
    }

    private static string? TryResolveMapDirectoryUnderRoot(string rootPath, string mapName)
    {
        if (string.IsNullOrWhiteSpace(rootPath) || string.IsNullOrWhiteSpace(mapName))
            return null;

        string[] candidates =
        {
            Path.Combine(rootPath, "World", "Maps", mapName),
            Path.Combine(rootPath, "Data", "World", "Maps", mapName),
            Path.Combine(rootPath, mapName),
        };

        foreach (string candidate in candidates)
        {
            if (Directory.Exists(candidate))
                return candidate;
        }

        return null;
    }

    private static string? TryResolveMapWdtUnderRoot(string rootPath, string mapName)
    {
        string? mapDirectory = TryResolveMapDirectoryUnderRoot(rootPath, mapName);
        if (string.IsNullOrWhiteSpace(mapDirectory))
            return null;

        string wdtPath = Path.Combine(mapDirectory, mapName + ".wdt");
        return File.Exists(wdtPath) ? wdtPath : null;
    }

    private void StageCurrentWorldForDataSourceReload()
    {
        _pendingDataSourceWorldReloadVirtualPath = null;
        _pendingDataSourceWorldReloadLocalPath = null;
        _pendingDataSourceWorldReloadCameraPosition = null;

        if (_worldScene == null || _terrainManager == null)
            return;

        string? virtualWdtPath = !string.IsNullOrWhiteSpace(_lastVirtualPath)
            && string.Equals(Path.GetExtension(_lastVirtualPath), ".wdt", StringComparison.OrdinalIgnoreCase)
            ? _lastVirtualPath
            : null;
        string? localWdtPath = TryGetLoadedLocalWdtPath();

        if (string.IsNullOrWhiteSpace(virtualWdtPath) && string.IsNullOrWhiteSpace(localWdtPath))
            return;

        _pendingDataSourceWorldReloadVirtualPath = virtualWdtPath;
        _pendingDataSourceWorldReloadLocalPath = localWdtPath;
        _pendingDataSourceWorldReloadCameraPosition = _camera.Position;
        _pendingDataSourceWorldReloadCameraYaw = _camera.Yaw;
        _pendingDataSourceWorldReloadCameraPitch = _camera.Pitch;
    }

    private void ClearActiveSceneForDataSourceReload()
    {
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        ResetSqlSpawnStreamingState(clearSceneSpawns: false);
        _renderer = null;
        _loadedWmo = null;
        _loadedMdx = null;
        _loadedM2Runtime = null;
    }

    private void RestoreWorldAfterDataSourceReload()
    {
        if (_pendingDataSourceReloadGeneration != _activeDataSourceReloadGeneration)
            return;

        string? virtualPath = _pendingDataSourceWorldReloadVirtualPath;
        string? localPath = _pendingDataSourceWorldReloadLocalPath;
        Vector3? cameraPosition = _pendingDataSourceWorldReloadCameraPosition;
        float cameraYaw = _pendingDataSourceWorldReloadCameraYaw;
        float cameraPitch = _pendingDataSourceWorldReloadCameraPitch;

        _pendingDataSourceWorldReloadVirtualPath = null;
        _pendingDataSourceWorldReloadLocalPath = null;
        _pendingDataSourceWorldReloadCameraPosition = null;

        if (cameraPosition == null)
            return;

        // Probe the *new* data source for the WDT before any fallback. The previous-client
        // local cache (if any) was written by the prior data source; loading it through the
        // new data source can hang the viewer when the StandardTerrainAdapter then queries
        // ADTs that the new source does not have. Capture the result up front so the
        // status message at the end can tell the user which case they hit.
        bool newSourceHasWdt = !string.IsNullOrWhiteSpace(virtualPath)
            && _dataSource is MpqDataSource probe
            && probe.FileExists(virtualPath);

        if (!string.IsNullOrWhiteSpace(virtualPath) && _dataSource != null)
            LoadFileFromDataSource(virtualPath);

        if (_worldScene == null
            && !string.IsNullOrWhiteSpace(localPath)
            && File.Exists(localPath)
            && newSourceHasWdt)
        {
            LoadWdtTerrain(localPath);
        }

        if (_worldScene == null)
        {
            string missingMapName = Path.GetFileNameWithoutExtension(virtualPath ?? localPath ?? string.Empty);
            _statusMessage = !newSourceHasWdt && !string.IsNullOrWhiteSpace(virtualPath)
                ? $"Map \"{missingMapName}\" not present in the new client; previous world cleared."
                : $"Previous world could not be restored after client switch (data source: {_dataSource?.Name ?? "unknown"}).";
            return;
        }

        _camera.Position = cameraPosition.Value;
        _camera.Yaw = cameraYaw;
        _camera.Pitch = cameraPitch;
        _statusMessage = $"Reloaded world for client: {_terrainManager?.MapName ?? Path.GetFileNameWithoutExtension(virtualPath ?? localPath ?? string.Empty)}";
    }

    private string? TryGetLoadedLocalWdtPath()
    {
        if (string.IsNullOrWhiteSpace(_loadedFilePath))
            return null;

        if (!string.Equals(Path.GetExtension(_loadedFilePath), ".wdt", StringComparison.OrdinalIgnoreCase))
            return null;

        return File.Exists(_loadedFilePath) ? _loadedFilePath : null;
    }

    private bool HasWorldReturnTarget()
        => !string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath) && File.Exists(_lastWorldSceneWdtPath);

    private void CaptureWorldReturnState()
    {
        if (_worldScene == null || _terrainManager == null)
            return;

        string? wdtPath = TryGetLoadedLocalWdtPath();
        if (string.IsNullOrWhiteSpace(wdtPath))
            return;

        _lastWorldSceneWdtPath = wdtPath;
        _lastWorldSceneCameraPosition = _camera.Position;
        _lastWorldSceneCameraYaw = _camera.Yaw;
        _lastWorldSceneCameraPitch = _camera.Pitch;
    }

    private void ReturnToLastWorldScene()
    {
        if (!HasWorldReturnTarget())
        {
            _statusMessage = "No saved world scene is available to restore.";
            return;
        }

        _pendingWorldSpawnOverride = _lastWorldSceneCameraPosition;
        LoadWdtTerrain(_lastWorldSceneWdtPath!);
        _camera.Yaw = _lastWorldSceneCameraYaw;
        _camera.Pitch = _lastWorldSceneCameraPitch;
        _statusMessage = $"Returned to world: {_terrainManager?.MapName ?? Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath!)}";
    }

    private bool TryGetSelectedBrowserAssetPath(out string assetPath)
    {
        assetPath = string.Empty;
        if (_selectedFileIndex < 0 || _selectedFileIndex >= _filteredFiles.Count)
            return false;

        assetPath = _filteredFiles[_selectedFileIndex];
        return !string.IsNullOrWhiteSpace(assetPath);
    }

    private bool TryGetSelectedBrowserModelPath(out string assetPath)
    {
        if (TryGetSelectedBrowserAssetPath(out assetPath) && IsTaxiActorModelPath(assetPath))
            return true;

        assetPath = string.Empty;
        return false;
    }

    private void CopyTextToClipboard(string text, string description)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;

        ImGui.SetClipboardText(text);
        _statusMessage = $"Copied {description} to clipboard.";
    }

    private static string NormalizeAssetPathForUi(string assetPath)
        => string.IsNullOrWhiteSpace(assetPath)
            ? string.Empty
            : assetPath.Trim().Replace('/', '\\');

    private bool CanLoadAssetFromDataSource(string assetPath)
        => _dataSource != null
            && !string.IsNullOrWhiteSpace(assetPath)
            && !Path.IsPathRooted(assetPath);

    private void FramePoint(Vector3 target, float radius = 2f)
    {
        float effectiveRadius = MathF.Max(radius, 1f);
        float distance = MathF.Max(effectiveRadius * 4f, 12f);
        Vector3 cameraPosition = target + new Vector3(-distance, 0f, effectiveRadius * 1.2f);
        Vector3 lookDirection = Vector3.Normalize(target - cameraPosition);

        _camera.Position = cameraPosition;
        _camera.Yaw = MathF.Atan2(lookDirection.Y, lookDirection.X) * (180f / MathF.PI);
        _camera.Pitch = MathF.Asin(Math.Clamp(lookDirection.Z, -1f, 1f)) * (180f / MathF.PI);
    }

    private bool TryFrameStandaloneWmoDoodad(WmoRenderer wmoRenderer, WmoDoodadInfo doodad)
    {
        if (wmoRenderer.TryGetDoodadBounds(doodad.Index, Matrix4x4.Identity, out Vector3 boundsMin, out Vector3 boundsMax))
        {
            FrameBounds(boundsMin, boundsMax, mdxMirrorX: false);
            _statusMessage = $"Framed standalone WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
            return true;
        }

        FramePoint(doodad.LocalPosition, radius: 2f);
        _statusMessage = $"Framed standalone WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
        return true;
    }

    private bool TryFrameSelectedWorldWmoDoodad(WmoRenderer wmoRenderer, WmoDoodadInfo doodad)
    {
        if (_worldScene?.SelectedInstance is not ObjectInstance selectedInstance)
            return false;

        if (wmoRenderer.TryGetDoodadBounds(doodad.Index, selectedInstance.Transform, out Vector3 boundsMin, out Vector3 boundsMax))
        {
            FrameBounds(boundsMin, boundsMax, mdxMirrorX: false);
            _statusMessage = $"Framed world WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
            return true;
        }

        Vector3 worldPosition = Vector3.Transform(doodad.LocalPosition, selectedInstance.Transform);
        FramePoint(worldPosition, radius: 2f);
        _statusMessage = $"Framed world WMO doodad [{doodad.Index}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}.";
        return true;
    }

    private void DrawAssetPathActions(string label, string assetPath, string idSuffix)
    {
        string normalizedPath = NormalizeAssetPathForUi(assetPath);
        if (string.IsNullOrWhiteSpace(normalizedPath))
        {
            ImGui.TextDisabled($"{label}: unavailable");
            return;
        }

        ImGui.Text(label);
        if (ImGui.SmallButton($"Copy Path##{idSuffix}"))
            CopyTextToClipboard(normalizedPath, "asset path");

        ImGui.SameLine();
        bool canLoad = CanLoadAssetFromDataSource(normalizedPath);
        if (!canLoad)
            ImGui.BeginDisabled();
        if (ImGui.SmallButton($"Load Asset##{idSuffix}"))
            LoadFileFromDataSource(normalizedPath);
        if (!canLoad)
            ImGui.EndDisabled();

        ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 520f);
        ImGui.TextDisabled(normalizedPath);
        ImGui.PopTextWrapPos();
    }

    private bool TryGetStandaloneWmoAssetPath(out string assetPath)
    {
        assetPath = string.Empty;
        if (_renderer is not WmoRenderer || string.IsNullOrWhiteSpace(_lastVirtualPath))
            return false;

        assetPath = NormalizeAssetPathForUi(_lastVirtualPath);
        return !string.IsNullOrWhiteSpace(assetPath);
    }

    private bool TryInspectHoveredSceneAssetInSelection()
    {
        if (_worldScene?.HoveredAssetInfo is not HoveredAssetInfo info || !info.HasSceneObject)
            return false;

        if (!_worldScene.SelectSceneObject(info.SceneObjectType, info.SceneObjectIndex, info.ParentWmoIndex))
            return false;

        ClearSelectedWlLiquidBody(clearListIsolation: true);
        _worldScene.ClearTaxiSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
        ClearSelectedAreaPoiInfo();
        RefreshSelectedWorldObjectInfo();
        return true;
    }

    private void ApplyTaxiActorModelOverride(int routeId, string? modelPath)
    {
        if (_worldScene == null || routeId < 0)
            return;

        string? currentMapName = GetCurrentSessionMapName();
        if (!string.IsNullOrWhiteSpace(currentMapName))
        {
            if (!_savedTaxiActorModelOverridesByMap.TryGetValue(currentMapName, out Dictionary<int, string>? overridesByRoute))
            {
                overridesByRoute = new Dictionary<int, string>();
                _savedTaxiActorModelOverridesByMap[currentMapName] = overridesByRoute;
            }

            if (string.IsNullOrWhiteSpace(modelPath))
            {
                overridesByRoute.Remove(routeId);
                if (overridesByRoute.Count == 0)
                    _savedTaxiActorModelOverridesByMap.Remove(currentMapName);
            }
            else
            {
                overridesByRoute[routeId] = modelPath.Trim().Replace('/', '\\');
            }
        }

        _worldScene.SetTaxiActorModelOverride(routeId, modelPath);
        SaveViewerSettings();
    }

    private void ApplySavedTaxiActorModelOverridesForCurrentMap()
    {
        if (_worldScene == null)
            return;

        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return;

        if (!_savedTaxiActorModelOverridesByMap.TryGetValue(currentMapName, out Dictionary<int, string>? overridesByRoute))
            return;

        foreach ((int routeId, string modelPath) in overridesByRoute)
            _worldScene.SetTaxiActorModelOverride(routeId, modelPath);
    }

    private void PersistObjectPathFiltersForCurrentMap()
    {
        if (_worldScene == null)
            return;

        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return;

        List<SavedObjectPathFilterEntry> savedEntries = _worldScene.ObjectPathFilters
            .Where(entry => !string.IsNullOrWhiteSpace(entry.PathPrefix) && (entry.AppliesToWmo || entry.AppliesToMdx))
            .OrderBy(entry => entry.PathPrefix, StringComparer.OrdinalIgnoreCase)
            .Select(entry => new SavedObjectPathFilterEntry
            {
                PathPrefix = entry.PathPrefix,
                AppliesToWmo = entry.AppliesToWmo,
                AppliesToMdx = entry.AppliesToMdx,
            })
            .ToList();

        if (savedEntries.Count == 0 && _worldScene.ObjectPathFiltersEnabled)
        {
            _savedObjectPathFiltersByMap.Remove(currentMapName);
            SaveViewerSettings();
            return;
        }

        _savedObjectPathFiltersByMap[currentMapName] = new SavedObjectPathFilterMap
        {
            MapName = currentMapName,
            Enabled = _worldScene.ObjectPathFiltersEnabled,
            Filters = savedEntries,
        };

        SaveViewerSettings();
    }

    private void ApplySavedObjectPathFiltersForCurrentMap()
    {
        if (_worldScene == null)
            return;

        _worldScene.ClearObjectPathFilters();
        _worldScene.ObjectPathFiltersEnabled = true;

        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return;

        if (!_savedObjectPathFiltersByMap.TryGetValue(currentMapName, out SavedObjectPathFilterMap? savedMap))
            return;

        _worldScene.ObjectPathFiltersEnabled = savedMap.Enabled;
        foreach (SavedObjectPathFilterEntry filter in savedMap.Filters)
            _worldScene.AddObjectPathFilter(filter.PathPrefix, filter.AppliesToWmo, filter.AppliesToMdx);
    }

    private bool TryGetSelectedWorldObjectModelPath(out string modelPath, out bool isWmo)
    {
        modelPath = string.Empty;
        isWmo = false;

        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return false;

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        if (string.IsNullOrWhiteSpace(selected.ModelPath))
            return false;

        modelPath = selected.ModelPath.Trim().Replace('/', '\\').Trim('\\');
        if (string.IsNullOrWhiteSpace(modelPath))
            return false;

        isWmo = modelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);
        return true;
    }

    private static List<string> BuildObjectPathFilterPrefixCandidates(string modelPath)
    {
        var prefixes = new List<string>();
        if (string.IsNullOrWhiteSpace(modelPath))
            return prefixes;

        string normalizedPath = modelPath.Trim().Replace('/', '\\').Trim('\\');
        if (string.IsNullOrWhiteSpace(normalizedPath))
            return prefixes;

        string[] segments = normalizedPath.Split('\\', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length == 0)
            return prefixes;

        string currentPrefix = string.Empty;
        for (int i = 0; i < segments.Length; i++)
        {
            currentPrefix = string.IsNullOrEmpty(currentPrefix)
                ? segments[i]
                : $"{currentPrefix}\\{segments[i]}";

            if (i < segments.Length - 1 || !Path.HasExtension(segments[i]) || segments.Length == 1)
                prefixes.Add(currentPrefix);
        }

        if (!prefixes.Contains(normalizedPath, StringComparer.OrdinalIgnoreCase))
            prefixes.Add(normalizedPath);

        return prefixes;
    }

    private bool TryApplySelectedBrowserAssetToTaxiOverride()
    {
        if (!TryGetTaxiActorOverrideRouteId(out int routeId))
        {
            _statusMessage = "Select a taxi node or route first.";
            return false;
        }

        if (!TryGetSelectedBrowserModelPath(out string assetPath))
        {
            _statusMessage = "Select an .mdx, .mdl, or .m2 asset in the file browser first.";
            return false;
        }

        _taxiActorModelOverrideTargetRouteId = routeId;
        _taxiActorModelOverrideInput = assetPath.Replace('/', '\\');
        _taxiActorModelOverrideInputRouteId = routeId;
        ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
        RefreshSelectedTaxiInfo();
        _statusMessage = $"Applied taxi actor override from browser asset to route {routeId}.";
        return true;
    }

    private void AttachLooseMapOverlay(string selectedPath)
    {
        if (_dataSource is not MpqDataSource mpqDataSource)
        {
            _statusMessage = "Load a base MPQ game path first, then attach a loose map overlay.";
            return;
        }

        string selectedFullPath = Path.GetFullPath(selectedPath);
        string? overlayRoot = ResolveLooseMapOverlayRoot(selectedFullPath);
        if (string.IsNullOrWhiteSpace(overlayRoot))
        {
            _statusMessage = $"Selected folder must contain World\\Maps or be a map directory under World\\Maps. Selected: {selectedFullPath}";
            return;
        }

        if (!mpqDataSource.AddOverlayRoot(overlayRoot, out string normalizedRoot, out string message))
        {
            _statusMessage = $"{message} (selected: {selectedFullPath}; resolved root: {overlayRoot})";
            ViewerLog.Important(ViewerLog.Category.MpqData,
                $"Loose overlay attach failed. selected='{selectedFullPath}', resolvedRoot='{overlayRoot}', reason='{message}'");
            return;
        }

        _lastLooseOverlayPath = selectedFullPath;
        _standaloneSkinPathCache.Clear();
    _loggedStandaloneMissingSkinPaths.Clear();
        ResetWdlPreviewSupport();
        InitializeWdlPreviewSupport();
        InitializeMinimapSupport();
        RefreshDiscoveredMaps();
        RefreshFileList();
        if (_worldScene != null && (_worldScene.Pm4Overlay.ShowPm4Overlay || _worldScene.Pm4Overlay.Pm4LoadAttempted))
            _worldScene.Pm4Overlay.ReloadPm4Overlay();

        string? overlayBuildHint = TryDetectLooseOverlayBuildHint(normalizedRoot);
        if (!string.IsNullOrWhiteSpace(overlayBuildHint) && !string.Equals(_dbcBuild, overlayBuildHint, StringComparison.OrdinalIgnoreCase))
        {
            ViewerLog.Important(ViewerLog.Category.MpqData,
                $"Loose overlay at '{normalizedRoot}' carries PM4 of a format version associated with the {overlayBuildHint} era (the PM4 MVER word is a format version, NOT a build read from the file), but the active base client build is {_dbcBuild ?? "unknown"}. If PM4-linked objects do not match, try a {overlayBuildHint} base client.");
            _statusMessage = $"Attached loose map overlay: {normalizedRoot} (PM4 hint {overlayBuildHint}; current base {_dbcBuild ?? "unknown"})";
        }
        else
        {
            _statusMessage = $"Attached loose map overlay: {normalizedRoot}";
        }
    }

    private static string? TryDetectLooseOverlayBuildHint(string overlayRoot)
    {
        try
        {
            string worldMapsRoot = Path.Combine(overlayRoot, "World", "Maps");
            if (!Directory.Exists(worldMapsRoot))
                return null;

            string? pm4Path = Directory.EnumerateFiles(worldMapsRoot, "*.pm4", SearchOption.AllDirectories)
                .FirstOrDefault();
            if (string.IsNullOrWhiteSpace(pm4Path))
                return null;

            var pm4 = CorePm4DocumentReader.ReadFile(pm4Path);

            // PM4 MVER is a FORMAT VERSION WORD, not a client build number. Measured 2026-08-23:
            // the value is a constant 12304 (0x3010) across corpus files spanning a 20x size range
            // (63,628 to 1,267,923 bytes) with identical 32-byte MSHD, so it is neither a size nor
            // any content-derived quantity. Pm4VersionFormatter reads it as version 16 in the low
            // byte with an undecoded 0x30 high byte; PD4 by comparison stores 0x0030 (version 48).
            // That 12304 also happens to read like the real client build 4.0.1.12304 is a
            // coincidence of digits, and treating it as one is what put a false build in the status
            // bar. The mapping below is retained only as an ERA heuristic for picking a base client
            // - it says "files of this format version belong to this era", never "this file came
            // from that build". Do not present it as read from the file.
            return pm4.Version switch
            {
                11927 => "4.0.0.11927",
                12304 => "4.0.1.12304",
                _ => null,
            };
        }
        catch
        {
            return null;
        }
    }

    private static string? ResolveLooseMapOverlayRoot(string selectedPath)
    {
        string fullPath = Path.GetFullPath(selectedPath);
        if (!Directory.Exists(fullPath))
            return null;

        if (Directory.Exists(Path.Combine(fullPath, "World", "Maps")))
            return fullPath;

        var directoryInfo = new DirectoryInfo(fullPath);

        if (directoryInfo.Name.Equals("World", StringComparison.OrdinalIgnoreCase) &&
            Directory.Exists(Path.Combine(directoryInfo.FullName, "Maps")))
        {
            return directoryInfo.Parent?.FullName;
        }

        if (directoryInfo.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) &&
            directoryInfo.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
        {
            return directoryInfo.Parent.Parent?.FullName;
        }

        if (directoryInfo.Parent?.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) == true &&
            directoryInfo.Parent.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
        {
            return directoryInfo.Parent.Parent.Parent?.FullName;
        }

        // Only resolve ancestors that are part of the selected World\Maps tree.
        // Avoid broad drive-root fallback if an unrelated World\Maps exists elsewhere under the same root.
        for (DirectoryInfo? current = directoryInfo; current != null; current = current.Parent)
        {
            if (current.Name.Equals("World", StringComparison.OrdinalIgnoreCase) &&
                Directory.Exists(Path.Combine(current.FullName, "Maps")))
            {
                return current.Parent?.FullName;
            }

            if (current.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) &&
                current.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
            {
                return current.Parent.Parent?.FullName;
            }

            if (current.Parent?.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) == true &&
                current.Parent.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
            {
                return current.Parent.Parent.Parent?.FullName;
            }
        }

        return null;
    }

    private static string? ResolveListfilePath(string? explicitListfilePath)
    {
        if (!string.IsNullOrWhiteSpace(explicitListfilePath) && File.Exists(explicitListfilePath))
            return explicitListfilePath;

        string[] bundledCandidates =
        {
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data", "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "test_data", "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "gillijimproject_refactor", "test_data", "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "test_data", "community-listfile-withcapitals.csv")),
        };

        // New builds get listfile coverage within hours, so prefer the most recently written copy
        // (bundled or downloaded) over the first bundled candidate that happens to exist.
        string? downloadedPath = ListfileDownloader.GetListfilePath();
        string? newest = bundledCandidates
            .Append(downloadedPath ?? string.Empty)
            .Where(static candidate => candidate.Length > 0 && File.Exists(candidate))
            .OrderByDescending(static candidate => File.GetLastWriteTimeUtc(candidate))
            .FirstOrDefault();
        if (newest is not null)
        {
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Using listfile: {newest} ({File.GetLastWriteTime(newest):yyyy-MM-dd HH:mm})");
            return newest;
        }

        ViewerLog.Important(ViewerLog.Category.MpqData, "No external listfile available. MPQ file discovery will rely on archive-internal names only.");
        return null;
    }

    private void InitializeWdlPreviewSupport()
    {
        if (_dataSource == null)
            return;

        string cacheIdentity = BuildWdlPreviewCacheIdentity();
        string cacheSegment = BuildCacheSegment(cacheIdentity);

        _wdlPreviewCacheService?.Dispose();
        _wdlPreviewCacheService = new WdlPreviewCacheService(_dataSource, Path.Combine(CacheDir, "wdl-preview", cacheSegment));
        _wdlPreviewWarmupStatus = string.Empty;
    }

    private void InitializeMinimapSupport()
    {
        _md5Index = null;

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            var searchPaths = new List<string> { mpqDataSource.GamePath };
            searchPaths.AddRange(mpqDataSource.OverlayRoots);
            searchPaths.AddRange(mpqDataSource.LooseRoots);

            if (Md5TranslateResolver.TryLoad(
                searchPaths,
                mpqDataSource.ArchiveReader.FileExists,
                mpqDataSource.ArchiveReader.ReadFile,
                out var md5Idx))
            {
                _md5Index = md5Idx;
                ViewerLog.Important(
                    ViewerLog.Category.Dbc,
                    $"Loaded MD5 Translate Index: {md5Idx?.HashToPlain.Count} entries");
            }
            else
            {
                ViewerLog.Trace(
                    $"[WoWViewer] No MD5 translate index found for minimaps under '{mpqDataSource.GamePath}'. Minimap loading will fall back to direct tile path variants.");
            }
        }

        _minimapRenderer?.Dispose();
        _minimapRenderer = null;
        if (_dataSource != null)
        {
            string minimapCacheSegment = BuildCacheSegment(BuildWdlPreviewCacheIdentity());
            _minimapRenderer = new MinimapRenderer(_gl, _dataSource, _md5Index, Path.Combine(CacheDir, "minimap", minimapCacheSegment));
        }
    }

    private static string BuildCacheSegment(string cacheIdentity)
    {
        string cacheSegment = string.IsNullOrWhiteSpace(cacheIdentity)
            ? "default"
            : Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(cacheIdentity))).ToLowerInvariant();
        return string.IsNullOrWhiteSpace(cacheSegment) ? "default" : cacheSegment;
    }

    private string BuildWdlPreviewCacheIdentity()
    {
        if (_dataSource is MpqDataSource mpqDataSource)
        {
            var parts = new List<string> { mpqDataSource.GamePath };
            parts.AddRange(mpqDataSource.OverlayRoots.OrderBy(path => path, StringComparer.OrdinalIgnoreCase));
            return string.Join("||", parts);
        }

        return _dataSource?.Name ?? "default";
    }

    private void ResetWdlPreviewSupport()
    {
        _wdlPreviewCacheService?.Dispose();
        _wdlPreviewCacheService = null;
        _wdlPreviewWarmupStatus = string.Empty;
        _wdlPreviewRenderer?.ClearPreview();
    }

    private void WarmDiscoveredWdlPreviews()
    {
        if (_wdlPreviewCacheService == null || _discoveredMaps.Count == 0)
            return;

        var mapsWithWdl = _discoveredMaps.Where(map => map.HasWdl).ToList();
        if (mapsWithWdl.Count == 0)
            return;

        _wdlPreviewCacheService.WarmMaps(mapsWithWdl);
        _wdlPreviewWarmupStatus = $"Warming {mapsWithWdl.Count} WDL previews in the background.";
    }

    private bool CanUseWdlPreviewFeature()
    {
        return _dataSource != null;
    }

    private static IEnumerable<string> EnumerateMapWdtCandidates(string mapDirectory)
    {
        string basePath = $"World\\Maps\\{mapDirectory}\\{mapDirectory}.wdt";
        yield return basePath;
    }

    private string? ResolveMapWdtPath(string mapDirectory)
    {
        if (_dataSource == null)
            return null;

        foreach (string candidate in EnumerateMapWdtCandidates(mapDirectory))
        {
            byte[]? data = _dataSource.ReadFile(candidate);
            if (data != null && data.Length > 0)
                return candidate;

            if (_dataSource is not MpqDataSource mpqDataSource)
                continue;

            string? found = mpqDataSource.FindInFileSet(candidate);
            if (string.IsNullOrWhiteSpace(found))
                continue;

            data = _dataSource.ReadFile(found);
            if (data != null && data.Length > 0)
                return found;
        }

        return null;
    }

    private void LoadMapAtDefaultSpawn(MapDefinition map)
    {
        if (!map.HasWdt)
            return;

        string? resolvedWdtPath = ResolveMapWdtPath(map.Directory);
        if (string.IsNullOrWhiteSpace(resolvedWdtPath))
        {
            _statusMessage = $"Failed to resolve WDT for {map.Directory}.";
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[WorldLoad] Failed to resolve map WDT for {map.Directory} from discovery actions.");
            return;
        }

        _selectedMapForPreview = null;
        _selectedSpawnTile = null;
        _pendingWorldSpawnOverride = null;
        _showWdlPreview = false;

        LoadFileFromDataSource(resolvedWdtPath);
    }

    private void LoadSelectedPreviewMapAtSpawn()
    {
        if (_selectedMapForPreview == null || !_selectedMapForPreview.HasWdt)
            return;

        string? resolvedWdtPath = ResolveMapWdtPath(_selectedMapForPreview.Directory);
        if (string.IsNullOrWhiteSpace(resolvedWdtPath))
        {
            _statusMessage = $"Failed to resolve WDT for {_selectedMapForPreview.Directory}.";
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[WorldLoad] Failed to resolve map WDT for {_selectedMapForPreview.Directory} from spawn preview.");
            return;
        }

        _pendingWorldSpawnOverride = _selectedSpawnTile.HasValue && _wdlPreviewRenderer?.HasPreview == true
            ? _wdlPreviewRenderer.TileToWorldPosition(
                (int)_selectedSpawnTile.Value.X,
                (int)_selectedSpawnTile.Value.Y)
            : null;

        LoadFileFromDataSource(resolvedWdtPath);

        _showWdlPreview = false;
    }

    private void OpenWdlPreview(MapDefinition map)
    {
        if (!map.HasWdt)
            return;

        if (!map.HasWdl || !CanUseWdlPreviewFeature())
        {
            LoadMapAtDefaultSpawn(map);
            return;
        }

        _selectedMapForPreview = map;
        _selectedSpawnTile = null;
        _showWdlPreview = true;

        if (_wdlPreviewRenderer == null)
            _wdlPreviewRenderer = new WdlPreviewRenderer(_gl);

        TryLoadSelectedWdlPreviewFromCache(map.Directory);

        if (!_wdlPreviewRenderer.HasPreview && _wdlPreviewCacheService != null)
        {
            if (_wdlPreviewCacheService.TryBuildPreviewNow(map.Directory, out var previewData, out var error) && previewData != null)
            {
                _wdlPreviewRenderer.LoadPreview(previewData);
                _wdlPreviewWarmupStatus = string.Empty;
            }
            else if (!string.IsNullOrWhiteSpace(error))
            {
                _wdlPreviewWarmupStatus = error;
            }
        }

        if (_wdlPreviewRenderer.HasPreview)
        {
            _showWdlPreview = true;
            return;
        }

        if (GetSelectedWdlPreviewState() == WdlPreviewWarmState.Failed)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain,
                $"[WDL] Preview unavailable for {map.Directory}; using default map spawn.");
            LoadMapAtDefaultSpawn(map);
            return;
        }
    }

    private void TryLoadSelectedWdlPreviewFromCache(string mapDirectory)
    {
        if (_wdlPreviewRenderer == null)
            return;

        if (_wdlPreviewCacheService != null && _wdlPreviewCacheService.TryGetPreview(mapDirectory, out var previewData) && previewData != null)
        {
            _wdlPreviewRenderer.LoadPreview(previewData);
            _wdlPreviewWarmupStatus = string.Empty;
            return;
        }

        _wdlPreviewRenderer.ClearPreview();

        if (_wdlPreviewCacheService != null)
        {
            _wdlPreviewCacheService.EnsurePrefetch(mapDirectory);
            var state = _wdlPreviewCacheService.GetState(mapDirectory);
            _wdlPreviewWarmupStatus = state switch
            {
                WdlPreviewWarmState.Ready => string.Empty,
                WdlPreviewWarmState.Failed => _wdlPreviewCacheService.GetError(mapDirectory) ?? $"Failed to prepare preview for {mapDirectory}.",
                _ => $"Preparing WDL preview for {mapDirectory}...",
            };
            return;
        }

        if (_dataSource != null)
        {
            bool loaded = _wdlPreviewRenderer.LoadWdl(_dataSource, mapDirectory);
            _wdlPreviewWarmupStatus = loaded ? string.Empty : _wdlPreviewRenderer.LastError ?? string.Empty;
        }
    }

    private WdlPreviewWarmState GetSelectedWdlPreviewState()
    {
        if (_wdlPreviewRenderer?.HasPreview == true)
            return WdlPreviewWarmState.Ready;

        if (_selectedMapForPreview == null)
            return WdlPreviewWarmState.NotQueued;

        if (_wdlPreviewCacheService != null)
            return _wdlPreviewCacheService.GetState(_selectedMapForPreview.Directory);

        return string.IsNullOrWhiteSpace(_wdlPreviewRenderer?.LastError)
            ? WdlPreviewWarmState.Loading
            : WdlPreviewWarmState.Failed;
    }

    private string? GetSelectedWdlPreviewError()
    {
        if (_selectedMapForPreview == null)
            return null;

        if (_wdlPreviewCacheService != null)
            return _wdlPreviewCacheService.GetError(_selectedMapForPreview.Directory);

        return _wdlPreviewRenderer?.LastError;
    }

    private (int total, int ready, int loading, int failed) GetWdlPreviewWarmupStats()
    {
        if (_wdlPreviewCacheService == null || _discoveredMaps.Count == 0)
            return (0, 0, 0, 0);

        int total = 0;
        int ready = 0;
        int loading = 0;
        int failed = 0;

        foreach (var map in _discoveredMaps)
        {
            if (!map.HasWdl)
                continue;

            total++;
            switch (_wdlPreviewCacheService.GetState(map.Directory))
            {
                case WdlPreviewWarmState.Ready:
                    ready++;
                    break;
                case WdlPreviewWarmState.Loading:
                    loading++;
                    break;
                case WdlPreviewWarmState.Failed:
                    failed++;
                    break;
            }
        }

        return (total, ready, loading, failed);
    }

    /// <summary>
    /// Infer the full build string (e.g. "0.10.0.3892") from the game path.
    /// Strategy:
    ///   1. Regex-extract all X.Y.Z.NNNN candidates from the path
    ///   2. Validate each against WoWDBDefs BUILD lines
    ///   3. If no 4-part match, try X.Y.Z short versions and resolve to full build via DBD
    ///   4. Fallback: MPQ heuristics for 3.3.5
    /// </summary>
    private static string InferBuildFromPath(string path, string? dbdDir)
    {
        // Collect all known builds from WoWDBDefs (cached per call)
        HashSet<string> dbdBuilds = new(StringComparer.OrdinalIgnoreCase);
        if (!string.IsNullOrEmpty(dbdDir) && Directory.Exists(dbdDir))
        {
            // Parse Map.dbd — it covers all versions and is always present
            var mapDbd = Path.Combine(dbdDir, "Map.dbd");
            if (File.Exists(mapDbd))
            {
                foreach (var line in File.ReadLines(mapDbd))
                {
                    var trimmed = line.Trim();
                    if (!trimmed.StartsWith("BUILD ")) continue;
                    // Parse "BUILD X.Y.Z.NNNN" or "BUILD X.Y.Z.NNNN-X.Y.Z.NNNN" or comma-separated
                    var parts = trimmed[6..].Split(',', StringSplitOptions.TrimEntries);
                    foreach (var part in parts)
                    {
                        // Handle ranges: "0.9.0.3807-0.12.0.3988"
                        var rangeParts = part.Split('-', StringSplitOptions.TrimEntries);
                        foreach (var rp in rangeParts)
                            if (Regex.IsMatch(rp, @"^\d+\.\d+\.\d+\.\d+$"))
                                dbdBuilds.Add(rp);
                    }
                }
            }
        }
        ViewerLog.Trace($"[BuildDetect] Loaded {dbdBuilds.Count} known builds from WoWDBDefs");

        // 1. Extract all X.Y.Z.NNNN candidates from the path
        var fullMatches = Regex.Matches(path, @"(\d+\.\d+\.\d+\.\d+)");
        foreach (Match m in fullMatches)
        {
            string candidate = m.Groups[1].Value;
            if (dbdBuilds.Contains(candidate))
            {
                ViewerLog.Trace($"[BuildDetect] Exact match from path: {candidate}");
                return candidate;
            }
        }

        // 2. Extract X.Y.Z short versions and find matching full build in DBD
        var shortMatches = Regex.Matches(path, @"(\d+\.\d+\.\d+)");
        foreach (Match m in shortMatches)
        {
            string shortVer = m.Groups[1].Value;
            // Find any DBD build that starts with this short version
            var match = dbdBuilds.FirstOrDefault(b => b.StartsWith(shortVer + "."));
            if (!string.IsNullOrEmpty(match))
            {
                ViewerLog.Trace($"[BuildDetect] Short version '{shortVer}' resolved to: {match}");
                return match;
            }
        }

        // 3. Check for full build in path that might be in a BUILD range (not exact endpoint)
        foreach (Match m in fullMatches)
        {
            string candidate = m.Groups[1].Value;
            // Try to find it in DBD range lines
            string? rangeMatch = FindBuildInDbdRanges(dbdDir, candidate);
            if (!string.IsNullOrEmpty(rangeMatch))
            {
                ViewerLog.Trace($"[BuildDetect] Range match from path: {candidate}");
                return candidate;
            }
        }

        // 4. Fallback: MPQ heuristics
        if (Directory.Exists(path))
        {
            try
            {
                var mpqs = Directory.GetFiles(path, "*.mpq", SearchOption.AllDirectories)
                    .Select(f => Path.GetFileName(f).ToLowerInvariant()).ToArray();

                // LK 3.3.5: has patch MPQs with "3" in name
                if (mpqs.Any(m => m.Contains("patch") && m.Contains("3")))
                {
                    var lkBuild = dbdBuilds.FirstOrDefault(b => b.StartsWith("3.3.5."));
                    return lkBuild ?? "3.3.5.12340";
                }

                // Alpha 0.5.3: dbc.mpq + model.mpq + texture.mpq, no common.mpq or patch-*.mpq
                bool hasAlphaSignature = mpqs.Contains("dbc.mpq")
                    && mpqs.Contains("model.mpq")
                    && mpqs.Contains("texture.mpq")
                    && !mpqs.Any(m => m.StartsWith("common"))
                    && !mpqs.Any(m => m.StartsWith("patch-"));
                if (hasAlphaSignature)
                {
                    // Check for patch.mpq → 0.7.0+, otherwise 0.5.3
                    bool hasPatch = mpqs.Contains("patch.mpq");
                    if (hasPatch)
                    {
                        // 0.6.0–0.8.0 range: try each in order
                        foreach (var prefix in new[] { "0.8.0.", "0.7.0.", "0.6.0." })
                        {
                            var match = dbdBuilds.FirstOrDefault(b => b.StartsWith(prefix));
                            if (!string.IsNullOrEmpty(match))
                            {
                                ViewerLog.Trace($"[BuildDetect] MPQ heuristic (alpha+patch): {match}");
                                return match;
                            }
                        }
                    }
                    else
                    {
                        var alphaBuild = dbdBuilds.FirstOrDefault(b => b.StartsWith("0.5.3."));
                        if (!string.IsNullOrEmpty(alphaBuild))
                        {
                            ViewerLog.Trace($"[BuildDetect] MPQ heuristic (alpha): {alphaBuild}");
                            return alphaBuild;
                        }
                        return "0.5.3.3368";
                    }
                }
            }
            catch { }
        }

        return "";
    }

    /// <summary>
    /// Check if a build number falls within any BUILD range in the DBD files.
    /// Parses ranges like "BUILD 0.9.0.3807-0.12.0.3988" and checks if the candidate
    /// build falls within [start, end] using numeric tuple comparison.
    /// </summary>
    private static string? FindBuildInDbdRanges(string? dbdDir, string build)
    {
        if (string.IsNullOrEmpty(dbdDir)) return null;
        var mapDbd = Path.Combine(dbdDir, "Map.dbd");
        if (!File.Exists(mapDbd)) return null;

        var buildTuple = ParseBuildTuple(build);
        if (buildTuple == null) return null;

        foreach (var line in File.ReadLines(mapDbd))
        {
            var trimmed = line.Trim();
            if (!trimmed.StartsWith("BUILD ")) continue;

            // Check explicit listing first
            if (trimmed.Contains(build)) return build;

            // Check ranges: "BUILD 0.9.0.3807-0.12.0.3988"
            var entries = trimmed[6..].Split(',', StringSplitOptions.TrimEntries);
            foreach (var entry in entries)
            {
                var rangeParts = entry.Split('-', StringSplitOptions.TrimEntries);
                if (rangeParts.Length == 2)
                {
                    var lo = ParseBuildTuple(rangeParts[0]);
                    var hi = ParseBuildTuple(rangeParts[1]);
                    if (lo != null && hi != null &&
                        CompareBuild(buildTuple, lo) >= 0 &&
                        CompareBuild(buildTuple, hi) <= 0)
                    {
                        ViewerLog.Trace($"[BuildDetect] '{build}' falls within range {rangeParts[0]}-{rangeParts[1]}");
                        return build;
                    }
                }
            }
        }
        return null;
    }

    private static int[]? ParseBuildTuple(string build)
    {
        var parts = build.Split('.');
        if (parts.Length != 4) return null;
        var nums = new int[4];
        for (int i = 0; i < 4; i++)
            if (!int.TryParse(parts[i], out nums[i])) return null;
        return nums;
    }

    private static int CompareBuild(int[] a, int[] b)
    {
        for (int i = 0; i < 4; i++)
        {
            if (a[i] < b[i]) return -1;
            if (a[i] > b[i]) return 1;
        }
        return 0;
    }

    private void LoadFileFromDisk(string filePath)
    {
        _loadedFilePath = filePath;
        _loadedFileName = Path.GetFileName(filePath);
        _window.Title = $"{ViewerProductName} - {_loadedFileName}";

        var ext = Path.GetExtension(filePath).ToLowerInvariant();
        string dir = Path.GetDirectoryName(filePath) ?? ".";

        if (ext != ".wdt")
            CaptureWorldReturnState();

        try
        {
            _renderer?.Dispose();
            _renderer = null;

            switch (ext)
            {
                case ".mdx":
                case ".mdl":
                case ".m2":
                    var modelBytes = File.ReadAllBytes(filePath);
                    LoadModelFromBytesWithContainerProbe(modelBytes, filePath, dir, "Disk");
                    break;

                case ".wmo":
                    LoadWmoFromDisk(filePath, dir);
                    break;

                case ".wdt":
                    LoadWdtTerrain(filePath);
                    break;

                default:
                    _statusMessage = $"Unsupported format: {ext}";
                    break;
            }
        }
        catch (Exception ex)
        {
            LogLoadFailure("DiskLoad", filePath, ex);
            _statusMessage = $"Failed to load: {BuildStatusExceptionSummary(ex)}";
            _modelInfo = "";
        }
    }

    /// <summary>
    /// Load an M2 model from disk using Warcraft.NET parser + companion .skin geometry.
    /// </summary>
    private void LoadM2FromDisk(string filePath, string dir)
    {
        var m2Bytes = File.ReadAllBytes(filePath);
        LoadM2FromBytes(m2Bytes, filePath, dir);
    }

    /// <summary>
    /// Load an M2 model from raw bytes using Warcraft.NET model/skin support.
    /// </summary>
    private void LoadM2FromBytes(byte[] m2Bytes, string originalPath, string dir)
    {
        string resolvedModelPath = ResolveStandaloneCanonicalModelPath(originalPath);

        // Detect era FIRST — 1.0.0 and 1.12.1 models have embedded geometry and don't
        // need a format profile or external .skin files. Only WotLK+ (264+) needs the
        // profile registry + external .skin companion path.
        M2Era1121EraTag detectedEra = M2ModelReaderDispatcher.DetectEra(m2Bytes.AsSpan(), resolvedModelPath);

        if (TryLoadStandaloneCameraPathM2(m2Bytes, resolvedModelPath))
        {
            CaptureWorldReturnState();
            return;
        }

        if (detectedEra is M2Era1121EraTag.Md20_1X_V100_Era100)
        {
            try
            {
                M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildEra100StaticRenderModel(m2Bytes, resolvedModelPath);
                LoadM2RuntimeModel(runtimeModel, modelDir: dir, virtualPath: resolvedModelPath);
                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Loaded native 1.0.0 M2 geometry for {Path.GetFileName(originalPath)} (era={detectedEra.ToDisplayString()})");
                _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                return;
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] Embedded 1.0.0 fallback failed for {Path.GetFileName(originalPath)}: {ex.Message}");
                throw new InvalidDataException(
                    $"Failed to load embedded 1.0.0 geometry for {Path.GetFileName(originalPath)}: {ex.Message}", ex);
            }
        }

        if (detectedEra is M2Era1121EraTag.Md20_1X_V100 or M2Era1121EraTag.Md20_1X_V101)
        {
            try
            {
                var embeddedMdx = WarcraftNetM2Adapter.BuildRuntimeModel(m2Bytes, null, resolvedModelPath, _dbcBuild);
                LoadMdxModel(embeddedMdx, dir, resolvedModelPath, isM2AdapterModel: true);
                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Loaded embedded 1.12.1 geometry for {Path.GetFileName(originalPath)} (era={detectedEra.ToDisplayString()})");
                _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                return;
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] Embedded 1.12.1 fallback failed for {Path.GetFileName(originalPath)}: {ex.Message}");
                throw new InvalidDataException(
                    $"Failed to load embedded 1.12.1 geometry for {Path.GetFileName(originalPath)}: {ex.Message}", ex);
            }
        }

        // WotLK+ (264+) path: requires a format profile + external .skin companion.
        var profile = FormatProfileRegistry.ResolveModelProfile(_dbcBuild);
        if (profile == null)
        {
            string buildLabel = string.IsNullOrWhiteSpace(_dbcBuild) ? "unknown" : _dbcBuild;
            throw new InvalidDataException(
                $"Standalone M2-family loading is not yet implemented for build {buildLabel}. " +
                "This asset is an M2-family model; .mdx/.mdl is not a substitute for 1.x M2 data. " +
                "Use the version-specific M2 reader path or load a supported client build.");
        }

        WarcraftNetM2Adapter.ValidateModelProfile(m2Bytes, resolvedModelPath, profile, _dbcBuild);

        var candidatePaths = new List<string>(WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath));

        Exception? lastError = null;
        bool anySkinFound = false;
        bool triedBestSkinPath = false;

        while (true)
        {
            foreach (var skinPath in candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                byte[]? skinBytes = ReadStandaloneFileData(skinPath);
                if (skinBytes == null || skinBytes.Length == 0)
                    continue;

                anySkinFound = true;

                try
                {
                    ViewerLog.Trace($"[M2] Trying skin: {skinPath} ({skinBytes.Length} bytes)");
                    M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(m2Bytes, skinBytes, resolvedModelPath, skinPath);
                    MdxFile? adaptedMdx = null;
                    try
                    {
                        adaptedMdx = WarcraftNetM2Adapter.BuildRuntimeModel(m2Bytes, skinBytes, resolvedModelPath, _dbcBuild);
                    }
                    catch (Exception adapterEx)
                    {
                        ViewerLog.Debug(ViewerLog.Category.Mdx,
                            $"[M2] M2->MDX adapter fallback failed for {Path.GetFileName(resolvedModelPath)}: {adapterEx.Message} (native renderer will be used)");
                    }
                    LoadM2RuntimeModel(runtimeModel, adaptedMdx, dir, resolvedModelPath);
                    CaptureWorldReturnState();
                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Selected skin for {Path.GetFileName(originalPath)}: {skinPath} ({skinBytes.Length} bytes)");
                    _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                    return;
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Skin candidate failed for {Path.GetFileName(originalPath)}: {skinPath} ({ex.Message})");
                }
            }

            if (triedBestSkinPath)
                break;

            triedBestSkinPath = true;
            string? bestSkinPath = ResolveBestStandaloneSkinPath(resolvedModelPath);
            if (string.IsNullOrWhiteSpace(bestSkinPath))
                break;

            candidatePaths.Add(bestSkinPath);
        }

        if (!anySkinFound && string.Equals(FormatProfileRegistry.ResolveModelProfile(_dbcBuild)?.ProfileId, FormatProfileRegistry.M2Profile3018303.ProfileId, StringComparison.Ordinal))
        {
            try
            {
                var embeddedMdx = WarcraftNetM2Adapter.BuildRuntimeModel(m2Bytes, null, resolvedModelPath, _dbcBuild);
                LoadMdxModel(embeddedMdx, dir, resolvedModelPath, isM2AdapterModel: true);
                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Loaded embedded root-profile geometry for {Path.GetFileName(originalPath)} after no external .skin resolved");
                _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                return;
            }
            catch (Exception ex)
            {
                lastError = ex;
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] Embedded root-profile fallback failed for {Path.GetFileName(originalPath)}: {ex.Message}");
            }
        }

        if (WarcraftNetM2Adapter.IsMd20(m2Bytes))
        {
            byte[]? convertedBytes = ConvertStandaloneM2ToMdx(m2Bytes, resolvedModelPath);
            if (convertedBytes != null && convertedBytes.Length > 0)
            {
                try
                {
                    using var convertedStream = new MemoryStream(convertedBytes);
                    var convertedMdx = MdxFile.Load(convertedStream);
                    if (WarcraftNetM2Adapter.HasRenderableGeometry(convertedMdx))
                    {
                        LoadMdxModel(convertedMdx, dir, resolvedModelPath, isM2AdapterModel: true);
                        ViewerLog.Info(ViewerLog.Category.Mdx,
                            $"[M2] Falling back to M2->MDX conversion for {Path.GetFileName(originalPath)} after adapter failure");
                        _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                        return;
                    }

                    lastError = new InvalidDataException(
                        $"M2->MDX fallback produced no renderable geometry for {Path.GetFileName(originalPath)} ({WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)})");
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Rejecting converted fallback for {Path.GetFileName(originalPath)}: {WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)}");
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Converted fallback load failed for {Path.GetFileName(originalPath)}: {ex.Message}");
                }
            }
        }

        if (!anySkinFound)
        {
            bool isTracedPreRelease301 = string.Equals(
                FormatProfileRegistry.ResolveModelProfile(_dbcBuild)?.ProfileId,
                FormatProfileRegistry.M2Profile3018303.ProfileId,
                StringComparison.Ordinal);

            InvalidDataException missingSkinError = isTracedPreRelease301
                ? new InvalidDataException(
                    $"No external .skin resolved for pre-release M2: {Path.GetFileName(originalPath)}. wow.exe 3.0.1.8303 traces root-contained profile tables for CM2Shared; WoWViewer root-profile geometry parsing is still incomplete.")
                : new InvalidDataException($"Missing companion .skin for M2: {Path.GetFileName(originalPath)}");

            if (_loggedStandaloneMissingSkinPaths.Add(resolvedModelPath))
            {
                ViewerLog.Error(ViewerLog.Category.Mdx,
                    $"[M2] {missingSkinError.Message} (build={_dbcBuild ?? "unknown"}, resolved='{resolvedModelPath}', candidateCount={candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase).Count()})");
            }

            throw missingSkinError;
        }

        var adaptFailure = new InvalidDataException(
            $"Failed to adapt M2 with available .skin candidates: {Path.GetFileName(originalPath)}",
            lastError);
        ViewerLog.Error(ViewerLog.Category.Mdx,
            $"[M2] {adaptFailure.Message} for '{resolvedModelPath}' (build={_dbcBuild ?? "unknown"}): {DescribeExceptionChain(lastError ?? adaptFailure)}");
        throw adaptFailure;
    }

    private bool TryLoadStandaloneCameraPathM2(byte[] m2Bytes, string resolvedModelPath)
    {
        if (!WarcraftNetM2Adapter.IsMd20(m2Bytes))
            return false;

        try
        {
            using MemoryStream stream = new(m2Bytes, writable: false);
            M2ModelDocument model = M2ModelReader.Read(stream, resolvedModelPath);
            if (!M2CameraPathOverlayBuilder.CanBuild(model))
                return false;

            M2CameraPathVisualization visualization = M2CameraPathOverlayBuilder.Build(model);
            LoadStandaloneCameraPathModel(model, visualization, resolvedModelPath);
            ViewerLog.Info(ViewerLog.Category.Mdx,
                $"[M2] Loaded camera-path visualization for {Path.GetFileName(resolvedModelPath)}: cameras={model.CameraCount}, sequences={model.SequenceCount}");
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] Camera-path probe skipped for {Path.GetFileName(resolvedModelPath)}: {ex.Message}");
            return false;
        }
    }

    private static string DescribeExceptionChain(Exception ex, int maxDepth = 6)
    {
        var parts = new List<string>();
        Exception? current = ex;
        while (current != null && parts.Count < maxDepth)
        {
            parts.Add($"{current.GetType().Name}: {current.Message}");
            current = current.InnerException;
        }

        return string.Join(" -> ", parts);
    }

    private static string BuildStatusExceptionSummary(Exception ex)
    {
        string summary = DescribeExceptionChain(ex, 3);
        return summary.Length <= 240 ? summary : summary[..237] + "...";
    }

    private void LogLoadFailure(string operation, string sourcePath, Exception ex, byte[]? modelBytes = null)
    {
        string byteSummary = modelBytes == null
            ? string.Empty
            : $" magic={GetModelMagicLabel(modelBytes)} md20Version={GetMd20VersionLabel(modelBytes)} bytes={modelBytes.Length}";
        ViewerLog.Error(ViewerLog.Category.General,
            $"[{operation}] Failed for '{sourcePath}': {DescribeExceptionChain(ex)}{byteSummary}");
    }

    private void LogDataSourceReadFailure(string requestedPath, string resolvedPath, string ext)
    {
        bool requestedExists = false;
        bool resolvedExists = false;
        try { requestedExists = _dataSource?.FileExists(requestedPath) ?? false; } catch { }
        try { resolvedExists = _dataSource?.FileExists(resolvedPath) ?? false; } catch { }

        string indexedRequested = "-";
        string indexedResolved = "-";
        if (_dataSource is MpqDataSource mpqDataSource)
        {
            try
            {
                indexedRequested = mpqDataSource.FindInFileSet(requestedPath.Replace('/', '\\')) ?? "-";
                indexedResolved = mpqDataSource.FindInFileSet(resolvedPath.Replace('/', '\\')) ?? "-";
            }
            catch { }
        }

        ViewerLog.Error(ViewerLog.Category.General,
            $"[DataSourceRead] Failed to read requested='{requestedPath}' resolved='{resolvedPath}' ext={ext} source={_dataSource?.GetType().Name ?? "<null>"} exists(requested)={requestedExists} exists(resolved)={resolvedExists} indexedRequested='{indexedRequested}' indexedResolved='{indexedResolved}'");
    }

    private void ReportAreaLookupDiagnostic(int areaId)
    {
        if (_areaTableService == null)
            return;

        string diagnostic = _areaTableService.DescribeLookup(areaId, _currentMapId);
        if (_reportedAreaDiagnostics.Add(diagnostic))
            ViewerLog.Important(ViewerLog.Category.General, diagnostic);
    }

    private void UpdateCurrentAreaContext(TerrainRenderer? renderer)
    {
        if (_areaTableService == null)
        {
            _currentAreaLookup = null;
            _currentAreaName = string.Empty;
            _currentZoneName = string.Empty;
            return;
        }

        int loadedTileCount = _terrainManager?.LoadedTileCount ?? _vlmTerrainManager?.LoadedTileCount ?? 0;
        bool cameraMoved = float.IsNaN(_lastAreaLookupCameraPosition.X)
            || Vector3.DistanceSquared(_camera.Position, _lastAreaLookupCameraPosition) >= 16f;
        bool mapChanged = _currentMapId != _lastAreaLookupMapId;
        bool residencyChanged = loadedTileCount != _lastAreaLookupLoadedTileCount;

        if (++_areaLookupTick < 10 && !cameraMoved && !mapChanged && !residencyChanged)
            return;

        _areaLookupTick = 0;
        _lastAreaLookupCameraPosition = _camera.Position;
        _lastAreaLookupLoadedTileCount = loadedTileCount;
        _lastAreaLookupMapId = _currentMapId;

        // 1. Check if camera is inside a placed WMO group in the world scene
        if (_worldScene != null && _worldScene.TryGetWmoGroupAt(_camera.Position, out var wmoInst, out var wmoR, out int renderGroupIndex))
        {
            uint wmoGroupId = wmoR.GetRenderGroupAreaId(renderGroupIndex);
            string? rawGroupName = wmoR.GetRenderGroupRawName(renderGroupIndex);
            var wmoArea = _areaTableService.ResolveWmoArea(wmoR.WmoId, renderGroupIndex, wmoGroupId, _currentMapId, rawGroupName);
            if (wmoArea.Reason == WowViewer.Core.World.AreaResolutionReason.Resolved)
            {
                _currentAreaLookup = wmoArea;
                _currentZoneName = _currentAreaLookup.ZoneText ?? string.Empty;
                _currentAreaName = _currentAreaLookup.SubzoneText ?? _currentAreaLookup.ZoneText ?? string.Empty;
                return;
            }
        }
        else if (_renderer is WmoRenderer standaloneWmo)
        {
            int standaloneGroupIndex = standaloneWmo.FindGroupContainingPoint(_camera.Position);
            if (standaloneGroupIndex >= 0)
            {
                uint wmoGroupId = standaloneWmo.GetRenderGroupAreaId(standaloneGroupIndex);
                string? rawGroupName = standaloneWmo.GetRenderGroupRawName(standaloneGroupIndex);
                var wmoArea = _areaTableService.ResolveWmoArea(standaloneWmo.WmoId, standaloneGroupIndex, wmoGroupId, _currentMapId, rawGroupName);
                if (wmoArea.Reason == WowViewer.Core.World.AreaResolutionReason.Resolved)
                {
                    _currentAreaLookup = wmoArea;
                    _currentZoneName = _currentAreaLookup.ZoneText ?? string.Empty;
                    _currentAreaName = _currentAreaLookup.SubzoneText ?? _currentAreaLookup.ZoneText ?? string.Empty;
                    return;
                }
            }
        }

        // 2. Fall back to terrain chunk under camera
        if (renderer == null)
        {
            _currentAreaLookup = WowViewer.Core.World.AreaLookupResult.Unresolved(0, _currentMapId, WowViewer.Core.World.AreaResolutionReason.NoTerrainChunk);
            _currentAreaName = string.Empty;
            _currentZoneName = string.Empty;
            return;
        }

        var chunk = renderer.GetChunkInfoAt(_camera.Position.X, _camera.Position.Y);
        _currentAreaLookup = chunk is null
            ? WowViewer.Core.World.AreaLookupResult.Unresolved(0, _currentMapId, WowViewer.Core.World.AreaResolutionReason.NoTerrainChunk)
            : _areaTableService.ResolveArea(chunk.Value.AreaId, _currentMapId);

        _currentZoneName = _currentAreaLookup.ZoneText ?? string.Empty;
        _currentAreaName = _currentAreaLookup.SubzoneText ?? _currentAreaLookup.ZoneText ?? string.Empty;

        if (_currentAreaLookup.Reason != WowViewer.Core.World.AreaResolutionReason.Resolved)
            ReportAreaLookupDiagnostic(_currentAreaLookup.RawAreaId);
    }

    private void UpdateAreaOverlay(TerrainRenderer? renderer)
    {
        if (_worldScene == null || !_worldScene.ShowAreaRegionOverlay)
            return;

        if (_areaTableService == null || renderer == null)
        {
            _worldScene.SetAreaOverlay(new AreaOverlayBuildResult(
                Array.Empty<AreaOverlayRegion>(),
                0,
                0));
            _areaOverlayRenderer = renderer;
            _areaOverlayAreaTableService = _areaTableService;
            _areaOverlayRevision = int.MinValue;
            _areaOverlayMapId = _currentMapId;
            return;
        }

        if (ReferenceEquals(_areaOverlayRenderer, renderer)
            && ReferenceEquals(_areaOverlayAreaTableService, _areaTableService)
            && _areaOverlayRevision == renderer.ResidentChunkRevision
            && _areaOverlayMapId == _currentMapId)
        {
            return;
        }

        AreaOverlayBuildResult result = AreaOverlayRegionBuilder.Build(
            renderer.EnumerateResidentChunkInfos(),
            _areaTableService,
            _currentMapId);
        _worldScene.SetAreaOverlay(result);
        _areaOverlayRenderer = renderer;
        _areaOverlayAreaTableService = _areaTableService;
        _areaOverlayRevision = renderer.ResidentChunkRevision;
        _areaOverlayMapId = _currentMapId;
    }

    private void DrawAreaOverlayLabels(
        Matrix4x4 view,
        Matrix4x4 proj,
        float viewportX,
        float viewportY,
        float viewportWidth,
        float viewportHeight)
    {
        if (_worldScene is not { ShowAreaRegionOverlay: true } scene || scene.AreaOverlayRegions.Count == 0)
            return;

        var drawList = ImGui.GetForegroundDrawList();
        foreach (AreaOverlayRegion region in scene.AreaOverlayRegions)
        {
            if (!TryProjectWorldToViewport(
                    region.LabelPosition,
                    view,
                    proj,
                    viewportWidth,
                    viewportHeight,
                    out Vector2 projected))
            {
                continue;
            }

            if (projected.X < -80f || projected.X > viewportWidth + 80f
                || projected.Y < -40f || projected.Y > viewportHeight + 40f)
            {
                continue;
            }

            string label = $"{(region.Kind == AreaOverlayRegionKind.Zone ? "Zone" : "Subzone")}: {region.Name}";
            Vector2 textSize = ImGui.CalcTextSize(label);
            Vector2 textPos = new(
                viewportX + projected.X - textSize.X * 0.5f,
                viewportY + projected.Y - textSize.Y - 16f);
            Vector2 rectMin = textPos - new Vector2(8f, 5f);
            Vector2 rectMax = textPos + textSize + new Vector2(8f, 5f);
            Vector4 color = new(region.Color, 1f);
            Vector4 background = new(region.Color * 0.32f + new Vector3(0.05f), 0.92f);

            drawList.AddCircleFilled(
                new(viewportX + projected.X, viewportY + projected.Y),
                4f,
                ImGui.ColorConvertFloat4ToU32(color));
            drawList.AddRectFilled(rectMin, rectMax, ImGui.ColorConvertFloat4ToU32(background), 4f);
            drawList.AddRect(rectMin, rectMax, ImGui.ColorConvertFloat4ToU32(color), 4f, ImDrawFlags.None, 1.5f);
            drawList.AddText(textPos, ImGui.ColorConvertFloat4ToU32(new Vector4(0.98f, 0.99f, 1f, 1f)), label);
        }
    }

    private static ModelContainerKind DetectModelContainer(byte[] modelBytes)
    {
        if (modelBytes.Length < 4) return ModelContainerKind.Unknown;

        uint magic = BitConverter.ToUInt32(modelBytes, 0);
        if (magic == MdxHeaders.MAGIC) return ModelContainerKind.Mdlx;
        if (magic == 0x3032444D) return ModelContainerKind.Md20; // "MD20"
        if (magic == 0x3132444D) return ModelContainerKind.Md21; // "MD21"

        return ModelContainerKind.Unknown;
    }

    private static string GetModelMagicLabel(byte[] modelBytes)
    {
        if (modelBytes.Length < 4) return "<short>";

        uint magic = BitConverter.ToUInt32(modelBytes, 0);
        return magic switch
        {
            MdxHeaders.MAGIC => "MDLX",
            0x3032444D => "MD20",
            0x3132444D => "MD21",
            _ => $"0x{magic:X8}"
        };
    }

    private static string GetMd20VersionLabel(byte[] modelBytes)
    {
        if (modelBytes.Length < 8 || BitConverter.ToUInt32(modelBytes, 0) != 0x3032444D)
            return "n/a";

        uint version = BitConverter.ToUInt32(modelBytes, 4);
        return $"0x{version:X}";
    }

    private void LogModelRouteProbe(string entrypoint, string sourcePath, string ext, byte[] modelBytes, ModelContainerKind container)
    {
        ViewerLog.Trace(
            $"[ModelRouting] probe build={_dbcBuild ?? "unknown"} entrypoint={entrypoint} file={sourcePath} ext={ext} magic={GetModelMagicLabel(modelBytes)} md20Version={GetMd20VersionLabel(modelBytes)} container={container}");
    }

    private void LoadModelFromBytesWithContainerProbe(byte[] modelBytes, string sourcePath, string dir, string entrypoint,
        IReadOnlyList<string>? explicitTextureVariations = null)
    {
        var container = DetectModelContainer(modelBytes);
        string ext = Path.GetExtension(sourcePath).ToLowerInvariant();
        LogModelRouteProbe(entrypoint, sourcePath, ext, modelBytes, container);

        switch (container)
        {
            case ModelContainerKind.Mdlx:
                if (ext != ".mdx")
                    ViewerLog.Important(ViewerLog.Category.Mdx,
                        $"[ModelRouting] Extension/container mismatch: '{ext}' with MDLX root. Routing as MDX: {Path.GetFileName(sourcePath)}");

                // Use the legacy MDX renderer for .mdx files. The chunked MDX-to-M2
                // runtime conversion path produces incorrect animation for converted
                // MDX data (M2 CPU skinning doesn't properly handle Alpha-era models).
                MdxRuntimeSharedInfo? sharedRuntimeInfo = TryReadSharedMdxRuntimeInfo(sourcePath, modelBytes);

                using (var ms = new MemoryStream(modelBytes))
                using (var br = new BinaryReader(ms))
                {
                    var mdx = MdxFile.Load(br);
                    LoadMdxModel(mdx, dir, sourcePath, sharedRuntimeInfo: sharedRuntimeInfo,
                        explicitTextureVariations: explicitTextureVariations);
                }
                return;

            case ModelContainerKind.Md20:
            case ModelContainerKind.Md21:
                if (ext == ".mdx" || ext == ".mdl")
                    ViewerLog.Important(ViewerLog.Category.Mdx,
                        $"[ModelRouting] Extension/container mismatch: '{ext}' with {GetModelMagicLabel(modelBytes)} root. Routing as M2-family: {Path.GetFileName(sourcePath)}");

                LoadM2FromBytes(modelBytes, sourcePath, dir);
                return;

            default:
                throw new InvalidDataException(
                    $"Unsupported model root magic ({GetModelMagicLabel(modelBytes)}) for '{Path.GetFileName(sourcePath)}'. Expected MDLX or MD20.");
        }
    }

    private void LoadChunkedMdxFromBytes(byte[] modelBytes, string sourcePath, string dir)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        string resolvedModelPath = ResolveStandaloneCanonicalModelPath(sourcePath);
        using MemoryStream stream = new(modelBytes, writable: false);
        M2ChunkedReadResult chunked = M2ChunkedModelReader.ReadDetailed(stream, resolvedModelPath, ReadStandaloneFileData);

        if (TryLoadStandaloneCameraPathM2(chunked.Conversion.ModelBytes, chunked.Conversion.ModelPath))
        {
            CaptureWorldReturnState();
            return;
        }

        M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(
            chunked.Conversion.ModelBytes,
            chunked.Conversion.SkinBytes,
            chunked.Conversion.ModelPath,
            chunked.Conversion.SkinPath);

        LoadM2RuntimeModel(runtimeModel, modelDir: dir, virtualPath: resolvedModelPath);
        ViewerLog.Info(ViewerLog.Category.Mdx,
            $"[ModelRouting] Loaded chunked MDX through M2 runtime: file={Path.GetFileName(sourcePath)} chunks={chunked.Chunks.Count} geosets={chunked.Geometry.GeosetCount} vertices={chunked.VertexCount} triangles={chunked.TriangleCount}");
    }

    /// <summary>
    /// Load a WMO from disk, auto-detecting v14 (Alpha) vs v17+ (standard) format.
    /// v17 files are converted to v14 in-memory before rendering.
    /// </summary>
    private void LoadWmoFromDisk(string filePath, string dir)
    {
        int version = DetectWmoVersion(filePath);
        ViewerLog.Trace($"[WMO] Detected version {version} for {Path.GetFileName(filePath)}");

        if (version >= 17)
        {
            // v17+: parse directly into WmoV14Data — no lossy binary roundtrip
            var v17RootBytes = File.ReadAllBytes(filePath);

            var groupBytesList = new List<byte[]>();
            string baseName = Path.GetFileNameWithoutExtension(filePath);
            for (int gi = 0; gi < 512; gi++)
            {
                string groupPath = Path.Combine(dir, $"{baseName}_{gi:D3}.wmo");
                if (!File.Exists(groupPath)) break;
                groupBytesList.Add(File.ReadAllBytes(groupPath));
                ViewerLog.Trace($"[WMO] Loaded group file: {Path.GetFileName(groupPath)}");
            }

            var v17Parser = new WmoV17ToV14Converter();
            var wmo = v17Parser.ParseV17ToModel(v17RootBytes, groupBytesList);
            ViewerLog.Trace($"[WMO] Parsed v{version} direct ({wmo.Groups.Count} groups)");
            LoadWmoModel(wmo, dir);
            _statusMessage = $"Loaded WMO v{version}: {Path.GetFileName(filePath)}";
        }
        else
        {
            // v14 (Alpha): use existing pipeline directly
            var converter = new WmoV14ToV17Converter();
            var wmo = converter.ParseWmoV14(filePath);
            LoadWmoModel(wmo, dir);
        }
    }

    /// <summary>
    /// Load a WMO from data source bytes, auto-detecting v14 vs v17+ format.
    /// </summary>
    private void LoadWmoFromDataSource(byte[] rootBytes, string virtualPath, string cachePath)
    {
        // Detect version from bytes
        int version;
        using (var ms = new MemoryStream(rootBytes))
        using (var br = new BinaryReader(ms))
            version = DetectWmoVersionFromBytes(br);

        ViewerLog.Trace($"[WMO] Detected version {version} for {Path.GetFileName(virtualPath)}");

        if (version >= 17)
        {
            // v17+: parse directly into WmoV14Data — no lossy binary roundtrip
            var wmoDir = Path.GetDirectoryName(virtualPath)?.Replace('/', '\\') ?? "";
            var wmoBase = Path.GetFileNameWithoutExtension(virtualPath);

            var groupBytesList = new List<byte[]>();
            for (int gi = 0; gi < 512; gi++)
            {
                var groupName = $"{wmoBase}_{gi:D3}.wmo";
                var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                var groupBytes = _dataSource?.ReadFile(groupPath);
                if (groupBytes == null || groupBytes.Length == 0) break;
                groupBytesList.Add(groupBytes);
                ViewerLog.Trace($"[WMO] Group {gi}: loaded {groupBytes.Length} bytes");
            }

            var v17Parser = new WmoV17ToV14Converter();
            var wmo = v17Parser.ParseV17ToModel(rootBytes, groupBytesList);
            ViewerLog.Trace($"[WMO] Parsed v{version} direct ({wmo.Groups.Count} groups)");
            LoadWmoModel(wmo, CacheDir);
            _statusMessage = $"Loaded WMO v{version}: {Path.GetFileName(virtualPath)}";
        }
        else
        {
            // v14 (Alpha): use existing pipeline
            var converter = new WmoV14ToV17Converter();
            var wmo = converter.ParseWmoV14(cachePath);

            // v16 split format: root has GroupCount but no embedded MOGP chunks
            if (wmo.Groups.Count == 0 && wmo.GroupCount > 0 && _dataSource != null)
            {
                var wmoDir = Path.GetDirectoryName(virtualPath)?.Replace('/', '\\') ?? "";
                var wmoBase = Path.GetFileNameWithoutExtension(virtualPath);
                ViewerLog.Trace($"[WMO] v14/v16 split: loading {wmo.GroupCount} group files from data source");

                for (int gi = 0; gi < wmo.GroupCount; gi++)
                {
                    var groupName = $"{wmoBase}_{gi:D3}.wmo";
                    var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                    var groupBytes = _dataSource.ReadFile(groupPath);
                    if (groupBytes != null && groupBytes.Length > 0)
                    {
                        ViewerLog.Trace($"[WMO] Group {gi}: loaded {groupBytes.Length} bytes from '{groupPath}'");
                        converter.ParseGroupFile(groupBytes, wmo, gi);
                    }
                    else
                    {
                        ViewerLog.Trace($"[WMO] Group {gi}: NOT FOUND '{groupPath}'");
                    }
                }

                for (int gi = 0; gi < wmo.Groups.Count && gi < wmo.GroupInfos.Count; gi++)
                {
                    if (wmo.Groups[gi].Name == null)
                        wmo.Groups[gi].Name = $"group_{gi}";
                }

                var bMin = new Vector3(float.MaxValue);
                var bMax = new Vector3(float.MinValue);
                foreach (var g in wmo.Groups)
                {
                    foreach (var v in g.Vertices)
                    {
                        bMin = Vector3.Min(bMin, v);
                        bMax = Vector3.Max(bMax, v);
                    }
                }
                if (bMin.X < float.MaxValue)
                {
                    wmo.BoundsMin = bMin;
                    wmo.BoundsMax = bMax;
                    ViewerLog.Trace($"[WMO] Recalculated bounds: ({bMin.X:F1},{bMin.Y:F1},{bMin.Z:F1}) - ({bMax.X:F1},{bMax.Y:F1},{bMax.Z:F1})");
                }
            }

            LoadWmoModel(wmo, CacheDir);
        }
    }

    /// <summary>
    /// Detect Alpha WDT format by examining MPHD data.
    /// Alpha MPHD stores absolute file offsets to MDNM (byte 4) and MONM (byte 12).
    /// Standard MPHD stores flags at byte 0 and has no MDNM/MONM offsets.
    /// If MPHD byte 4 contains a large value (absolute offset to MDNM), it's Alpha.
    /// </summary>
    private static bool DetectAlphaWdt(byte[] wdtBytes)
    {
        // Find MPHD chunk (reversed on disk: "DHPM")
        for (int i = 0; i + 8 <= wdtBytes.Length;)
        {
            string fcc = System.Text.Encoding.ASCII.GetString(wdtBytes, i, 4);
            int sz = BitConverter.ToInt32(wdtBytes, i + 4);
            if (sz < 0 || i + 8 + sz > wdtBytes.Length) break;

            string reversed = new string(fcc.Reverse().ToArray());
            if (fcc == "DHPM" || reversed == "DHPM") // MPHD
            {
                int dataStart = i + 8;
                if (sz >= 16)
                {
                    // Alpha MPHD: [0..3]=nTextures, [4..7]=MDNM abs offset, [8..11]=nMapObjNames, [12..15]=MONM abs offset
                    // Standard MPHD: [0..3]=flags (small: 0,1,4,8), rest is different
                    int mdnmOffset = BitConverter.ToInt32(wdtBytes, dataStart + 4);
                    // MDNM offset in Alpha is always after MVER+MPHD+MAIN, so > ~32KB
                    // Standard MPHD byte 4 is 0 or a small relative offset
                    if (mdnmOffset > 1000 && mdnmOffset < wdtBytes.Length)
                        return true;
                }
                break;
            }

            int next = i + 8 + sz;
            if (next <= i) break;
            i = next;
        }

        return false;
    }

    private string ResolveStandaloneCanonicalModelPath(string sourcePath)
    {
        string normalizedPath = sourcePath.Replace('/', '\\');
        if (_dataSource == null)
            return normalizedPath;

        if (_dataSource is not MpqDataSource mpqDataSource)
            return normalizedPath;

        foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
        {
            string? found = mpqDataSource.FindInFileSet(candidate);
            if (!string.IsNullOrWhiteSpace(found))
                return found.Replace('/', '\\');
        }

        string baseName = Path.GetFileNameWithoutExtension(normalizedPath);
        if (!string.IsNullOrWhiteSpace(baseName))
        {
            string? indexed = mpqDataSource.FindByBaseName(baseName, GetLikelyStandaloneModelExtensions(normalizedPath));
            if (!string.IsNullOrWhiteSpace(indexed))
                return indexed.Replace('/', '\\');
        }

        foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
        {
            if (_dataSource.FileExists(candidate))
                return candidate.Replace('/', '\\');
        }

        return normalizedPath;
    }

    private string? ResolveBestStandaloneSkinPath(string resolvedModelPath)
    {
        if (_dataSource == null)
            return null;

        if (_standaloneSkinPathCache.TryGetValue(resolvedModelPath, out string? cachedPath))
            return cachedPath;

        string? bestSkinPath = WarcraftNetM2Adapter.FindSkinInFileList(resolvedModelPath, _dataSource.GetFileList(".skin"));
        _standaloneSkinPathCache[resolvedModelPath] = bestSkinPath;
        return bestSkinPath;
    }

    private byte[]? ReadStandaloneFileData(string path)
    {
        if (File.Exists(path))
            return File.ReadAllBytes(path);

        if (_dataSource == null)
            return null;

        byte[]? data = _dataSource.ReadFile(path);
        if (data != null && data.Length > 0)
            return data;

        string normalizedPath = path.Replace('/', '\\');
        if (!normalizedPath.Equals(path, StringComparison.OrdinalIgnoreCase))
        {
            data = _dataSource.ReadFile(normalizedPath);
            if (data != null && data.Length > 0)
                return data;
        }

        if (IsStandaloneModelPath(normalizedPath))
        {
            foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
            {
                if (candidate.Equals(normalizedPath, StringComparison.OrdinalIgnoreCase))
                    continue;

                data = _dataSource.ReadFile(candidate);
                if (data != null && data.Length > 0)
                    return data;
            }
        }

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
            {
                string? found = mpqDataSource.FindInFileSet(candidate);
                if (string.IsNullOrWhiteSpace(found))
                    continue;

                data = _dataSource.ReadFile(found);
                if (data != null && data.Length > 0)
                    return data;
            }

            string baseName = Path.GetFileNameWithoutExtension(normalizedPath);
            if (!string.IsNullOrWhiteSpace(baseName))
            {
                string? indexed = mpqDataSource.FindByBaseName(baseName, GetLikelyStandaloneModelExtensions(normalizedPath));
                if (!string.IsNullOrWhiteSpace(indexed))
                {
                    data = _dataSource.ReadFile(indexed);
                    if (data != null && data.Length > 0)
                        return data;
                }
            }
        }

        return null;
    }

    private static bool IsStandaloneModelPath(string path)
    {
        string ext = Path.GetExtension(path);
        return ext.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".mdl", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".m2", StringComparison.OrdinalIgnoreCase);
    }

    private static IEnumerable<string> BuildStandaloneFileSetCandidates(string path)
    {
        yield return path;

        foreach (string alternatePath in EnumerateStandaloneAlternateModelPaths(path))
            yield return alternatePath;

        string fileName = Path.GetFileName(path);
        if (!string.IsNullOrWhiteSpace(fileName) && !fileName.Equals(path, StringComparison.OrdinalIgnoreCase))
        {
            yield return fileName;

            foreach (string alternatePath in EnumerateStandaloneAlternateModelPaths(fileName))
                yield return alternatePath;
        }

        string baseName = Path.GetFileNameWithoutExtension(path);
        if (!string.IsNullOrWhiteSpace(baseName))
        {
            yield return $"Creature\\{baseName}\\{baseName}.mdx";
            yield return $"Creature\\{baseName}\\{baseName}.m2";
            yield return $"Creature\\{baseName}\\{baseName}.mdl";
        }
    }

    private byte[]? ConvertStandaloneM2ToMdx(byte[] m2Bytes, string resolvedModelPath)
    {
        try
        {
            byte[]? skinBytes = null;
            foreach (string skinPath in WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath).Distinct(StringComparer.OrdinalIgnoreCase))
            {
                skinBytes = ReadStandaloneFileData(skinPath);
                if (skinBytes != null && skinBytes.Length > 0)
                    break;
            }

            if ((skinBytes == null || skinBytes.Length == 0) && _dataSource != null)
            {
                string? bestSkinPath = ResolveBestStandaloneSkinPath(resolvedModelPath);
                if (!string.IsNullOrWhiteSpace(bestSkinPath))
                    skinBytes = ReadStandaloneFileData(bestSkinPath);
            }

            throw new NotSupportedException("M2 to MDX conversion is not supported in the standalone viewer."); return new byte[0];
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] Standalone M2->MDX converter fallback failed for {Path.GetFileName(resolvedModelPath)}: {ex.Message}");
            return null;
        }
    }

    private static IEnumerable<string> EnumerateStandaloneAlternateModelPaths(string path)
    {
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".m2";
            yield return path[..^4] + ".mdl";
            yield break;
        }

        if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".mdx";
            yield return path[..^4] + ".m2";
            yield break;
        }

        if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^3] + ".mdx";
            yield return path[..^3] + ".mdl";
        }
    }

    private static IEnumerable<string> GetLikelyStandaloneModelExtensions(string path)
    {
        string ext = Path.GetExtension(path);
        if (ext.Equals(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".m2";
            yield return ".mdx";
            yield return ".mdl";
            yield break;
        }

        if (ext.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".mdl";
            yield return ".mdx";
            yield return ".m2";
            yield break;
        }

        yield return ".mdx";
        yield return ".m2";
        yield return ".mdl";
    }

    /// <summary>
    /// Detect WMO version by reading the MVER chunk from the file.
    /// Returns 14 for Alpha, 17 for standard WotLK+, or 0 if detection fails.
    /// </summary>
    private static int DetectWmoVersion(string filePath)
    {
        try
        {
            using var fs = File.OpenRead(filePath);
            using var br = new BinaryReader(fs);
            return DetectWmoVersionFromBytes(br);
        }
        catch { return 0; }
    }

    /// <summary>
    /// Detect WMO version from a BinaryReader by scanning for MVER chunk.
    /// Handles both forward and reversed FourCC ordering.
    /// </summary>
    private static int DetectWmoVersionFromBytes(BinaryReader br)
    {
        long startPos = br.BaseStream.Position;
        try
        {
            // Read first 8 bytes to check for MOMO container (v14) or MVER (v17)
            if (br.BaseStream.Length < 12) return 0;

            var magic = System.Text.Encoding.ASCII.GetString(br.ReadBytes(4));
            var reversed = new string(magic.Reverse().ToArray());

            // v14 Alpha: starts with MOMO container
            if (magic == "MOMO" || reversed == "MOMO")
                return 14;

            // v17+: starts with MVER chunk directly
            if (magic == "MVER" || reversed == "MVER")
            {
                uint size = br.ReadUInt32();
                if (size >= 4)
                {
                    uint version = br.ReadUInt32();
                    return (int)version;
                }
            }

            // Fallback: scan first 64 bytes for MVER
            br.BaseStream.Position = startPos;
            byte[] header = br.ReadBytes((int)Math.Min(64, br.BaseStream.Length));
            string headerStr = System.Text.Encoding.ASCII.GetString(header);
            int mverIdx = headerStr.IndexOf("MVER");
            if (mverIdx < 0) mverIdx = headerStr.IndexOf("REVM"); // reversed
            if (mverIdx >= 0 && mverIdx + 12 <= header.Length)
            {
                uint ver = BitConverter.ToUInt32(header, mverIdx + 8);
                return (int)ver;
            }

            return 0;
        }
        finally
        {
            br.BaseStream.Position = startPos;
        }
    }

    /// <summary>
    /// Called when the user double-clicks an entry in the Asset Catalog.
    /// Loads the model into the viewer using the same pipeline as the file browser.
    /// </summary>
    private void OnCatalogLoadModel(string modelPath, bool isWmo, AssetCatalogEntry entry)
    {
        if (_dataSource == null)
        {
            _statusMessage = "No data source loaded";
            return;
        }

        // Try exact path first, then fuzzy resolve via the data source file list
        byte[]? data = _dataSource.ReadFile(modelPath);
        string resolvedPath = modelPath;

        if (data == null)
        {
            // Fuzzy: try Creature\Name\Name.mdx pattern and case variations
            string baseName = Path.GetFileNameWithoutExtension(modelPath);
            string[] candidates = {
                modelPath,
                $"Creature\\{baseName}\\{baseName}.mdx",
                modelPath.Replace('/', '\\'),
                modelPath.Replace('\\', '/'),
            };
            foreach (var c in candidates)
            {
                data = _dataSource.ReadFile(c);
                if (data != null) { resolvedPath = c; break; }
            }

            // Last resort: search file list
            if (data == null)
            {
                string ext = isWmo ? ".wmo" : ".mdx";
                var files = _dataSource.GetFileList(ext);
                string target = baseName.ToLowerInvariant();
                var match = files.FirstOrDefault(f =>
                    Path.GetFileNameWithoutExtension(f).Equals(target, StringComparison.OrdinalIgnoreCase));
                if (match != null)
                {
                    data = _dataSource.ReadFile(match);
                    if (data != null) resolvedPath = match;
                }
            }
        }

        if (data == null || data.Length == 0)
        {
            _statusMessage = $"Model not found: {modelPath}";
            return;
        }

        try
        {
            _renderer?.Dispose();
            _renderer = null;
            _loadedFileName = Path.GetFileName(resolvedPath);
            _lastVirtualPath = resolvedPath;

            string dir = Path.GetDirectoryName(resolvedPath)?.Replace('/', '\\') ?? "";

            if (isWmo)
            {
                // WMO: write to temp, parse, load
                string tempFile = Path.Combine(Path.GetTempPath(), $"catalog_wmo_{entry.EntryId}.wmo");
                File.WriteAllBytes(tempFile, data);
                var converter = new WmoV14ToV17Converter();
                var wmo = converter.ParseWmoV14(tempFile);

                // Handle split WMO groups
                if (wmo.Groups.Count == 0 && wmo.GroupCount > 0)
                {
                    string wmoBase = Path.GetFileNameWithoutExtension(resolvedPath);
                    for (int gi = 0; gi < wmo.GroupCount; gi++)
                    {
                        var groupName = $"{wmoBase}_{gi:D3}.wmo";
                        var groupPath = string.IsNullOrEmpty(dir) ? groupName : $"{dir}\\{groupName}";
                        var groupBytes = _dataSource.ReadFile(groupPath);
                        if (groupBytes != null)
                            converter.ParseGroupFile(groupBytes, wmo, gi);
                    }
                }

                try { File.Delete(tempFile); } catch { }
                LoadWmoModel(wmo, dir);
            }
            else
            {
                LoadModelFromBytesWithContainerProbe(data, resolvedPath, dir, "Catalog", entry.TextureVariations);
            }

            _window.Title = $"{ViewerProductName} - {entry.Name} ({_loadedFileName})";
            _statusMessage = $"Loaded from catalog: {entry.Name} [{entry.EntryId}]";
        }
        catch (Exception ex)
        {
            LogLoadFailure("CatalogLoad", resolvedPath, ex, isWmo ? null : data);
            _statusMessage = $"Failed to load {entry.Name}: {BuildStatusExceptionSummary(ex)}";
            _modelInfo = "";
        }
    }

    private void LoadFileFromDataSource(string virtualPath)
    {
        if (_dataSource == null) return;

        _statusMessage = $"Loading {Path.GetFileName(virtualPath)}...";
        _loadedFileName = Path.GetFileName(virtualPath);
        _lastVirtualPath = virtualPath;

        string resolvedVirtualPath = virtualPath;
        string ext = Path.GetExtension(virtualPath).ToLowerInvariant();
        byte[]? data = null;

        if (ext != ".wdt")
            CaptureWorldReturnState();

        try
        {
            if (ext is ".mdx" or ".mdl" or ".m2")
            {
                resolvedVirtualPath = ResolveStandaloneCanonicalModelPath(virtualPath);
                data = ReadStandaloneFileData(resolvedVirtualPath);
                if ((data == null || data.Length == 0) && !resolvedVirtualPath.Equals(virtualPath, StringComparison.OrdinalIgnoreCase))
                    data = ReadStandaloneFileData(virtualPath);
            }
            else
            {
                data = _dataSource.ReadFile(virtualPath);
            }

            if (data == null || data.Length == 0)
            {
                LogDataSourceReadFailure(virtualPath, resolvedVirtualPath, ext);
                _statusMessage = resolvedVirtualPath.Equals(virtualPath, StringComparison.OrdinalIgnoreCase)
                    ? $"Failed to read: {virtualPath}"
                    : $"Failed to read: {virtualPath} (resolved: {resolvedVirtualPath})";
                return;
            }

            _renderer?.Dispose();
            _renderer = null;

            _lastVirtualPath = resolvedVirtualPath;
            _loadedFileName = Path.GetFileName(resolvedVirtualPath);

            // Write to cache folder for parsers that expect file paths. The cache is VERSIONED by
            // client root: a flat cache let a stale Shadowfang.wdt extracted from one client
            // version shadow the 0.5.3 alphaWDT from another, silently feeding the terrain
            // pipeline a WDT that was never from the active client (Spec 222, 2026-09-04).
            Directory.CreateDirectory(CacheDir);
            string clientCacheSegment = BuildCacheSegment(BuildWdlPreviewCacheIdentity());
            var cachePath = Path.Combine(CacheDir, clientCacheSegment, _loadedFileName!);
            Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
            File.WriteAllBytes(cachePath, data);
            _loadedFilePath = cachePath;

            switch (ext)
            {
                case ".mdx":
                case ".m2":
                case ".mdl":
                    LoadModelFromBytesWithContainerProbe(data, resolvedVirtualPath, CacheDir, "DataSource");
                    break;

                case ".wmo":
                    LoadWmoFromDataSource(data, virtualPath, cachePath);
                    break;

                case ".wdt":
                    LoadWdtTerrain(cachePath);
                    break;

                default:
                    _statusMessage = $"Viewing {ext} not yet supported.";
                    break;
            }

            _window.Title = $"{ViewerProductName} - {_loadedFileName}";
        }
        catch (Exception ex)
        {
            LogLoadFailure("DataSourceLoad", resolvedVirtualPath, ex,
                ext is ".mdx" or ".mdl" or ".m2" ? data : null);
            _statusMessage = $"Load failed: {BuildStatusExceptionSummary(ex)}";
            _modelInfo = "";
        }
    }

    private void LoadMdxModel(MdxFile mdx, string dir, string? virtualPath = null, bool isM2AdapterModel = false,
        MdxRuntimeSharedInfo? sharedRuntimeInfo = null, IReadOnlyList<string>? explicitTextureVariations = null)
    {
        CaptureWorldReturnState();
        ExitToStandaloneView();

        _loadedWmo = null;
        _loadedMdx = mdx;
        _loadedM2Runtime = null;

        CoreMdxSummary? sharedSummary = sharedRuntimeInfo?.Summary;
        CoreMdxGeometryFile? sharedGeometry = sharedRuntimeInfo?.Geometry;

        int geosetCount = sharedGeometry?.GeosetCount ?? mdx.Geosets.Count;
        int validGeosets = sharedGeometry != null
            ? sharedGeometry.Geosets.Count(g => g.VertexCount > 0 && g.IndexCount > 0)
            : mdx.Geosets.Count(g => g.Vertices.Count > 0 && g.Indices.Count > 0);
        int totalVerts = sharedGeometry != null
            ? sharedGeometry.Geosets.Sum(g => g.VertexCount)
            : mdx.Geosets.Sum(g => g.Vertices.Count);
        int totalTris = sharedGeometry != null
            ? sharedGeometry.Geosets.Sum(g => g.TriangleCount)
            : mdx.Geosets.Sum(g => g.Indices.Count / 3);
        string versionLabel = sharedSummary?.Version?.ToString()
            ?? sharedGeometry?.Version?.ToString()
            ?? mdx.Version.ToString();
        string modelName = sharedSummary?.ModelName
            ?? sharedGeometry?.ModelName
            ?? mdx.Model.Name;
        int textureCount = sharedSummary?.TextureCount ?? mdx.Textures.Count;
        int materialCount = sharedSummary?.MaterialCount ?? mdx.Materials.Count;
        int boneCount = sharedSummary?.BoneCount ?? mdx.Bones.Count;
        int sequenceCount = sharedSummary?.SequenceCount ?? mdx.Sequences.Count;
        int pivotPointCount = sharedSummary?.PivotPointCount ?? mdx.PivotPoints.Count;
        CoreMdxCollisionSummary? collision = sharedSummary?.Collision;

        _renderer = new MdxRenderer(_gl, mdx, dir, _dataSource, _texResolver, virtualPath, isM2AdapterModel, _dbcBuild,
            explicitTextureVariations: explicitTextureVariations);
        RefreshStandaloneCharacterCustomizationState(virtualPath, isM2AdapterModel);

        if (sharedRuntimeInfo != null)
        {
            ViewerLog.Trace(
                $"[SharedMDX] Runtime metadata consumer: summary={(sharedSummary != null ? "yes" : "no")} geometry={(sharedGeometry != null ? "yes" : "no")} file={Path.GetFileName(virtualPath ?? _loadedFileName ?? "<memory>")}");
        }

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        string typeLabel = isM2AdapterModel
            ? "M2 (compatibility runtime via MDX renderer)"
            : "MDX (Alpha 0.5.3)";
        string statusTypeLabel = isM2AdapterModel ? "M2" : "MDX";

        _modelInfo = $"Path: {virtualPath ?? _loadedFileName ?? "<unknown>"}\n" +
                     $"Type: {typeLabel}\n" +
                     $"Version: {versionLabel}\n" +
                     $"Name: {modelName}\n\n" +
                     $"Geosets: {geosetCount} ({validGeosets} valid)\n" +
                     $"Vertices: {totalVerts:N0}\n" +
                     $"Triangles: {totalTris:N0}\n" +
                     $"Pivot Points: {pivotPointCount}\n" +
                     (collision != null
                        ? $"Collision: {collision.VertexCount} verts, {collision.TriangleCount} tris\n"
                        : string.Empty) +
                     "\n" +
                     $"Materials: {materialCount}\n" +
                     $"Textures: {textureCount}\n" +
                     $"Bones: {boneCount}\n" +
                     $"Sequences: {sequenceCount}\n";

        if (mdx.Sequences.Count > 0)
        {
            _modelInfo += "\nAnimations:\n";
            foreach (var seq in mdx.Sequences)
                _modelInfo += $"  {seq.Name} ({seq.Time.Start}-{seq.Time.End})\n";
        }

        if (mdx.Textures.Count > 0)
        {
            _modelInfo += "\nTextures:\n";
            foreach (var tex in mdx.Textures)
            {
                string name = string.IsNullOrEmpty(tex.Path) ? $"Replaceable #{tex.ReplaceableId}" : tex.Path;
                _modelInfo += $"  {name}\n";
            }
        }

        if (isM2AdapterModel)
        {
            _modelInfo += "\nCompatibility Notes:\n" +
                          "  Source asset is M2, but the current viewer path still adapts it into MdxFile/MdxRenderer state.\n" +
                          "  Animated M2 compatibility is currently disabled by default because that path is not reliable.\n";
        }

        _statusMessage = $"Loaded {statusTypeLabel}: {_loadedFileName} ({validGeosets} geosets, {totalVerts:N0} verts)";
    }

    private void LoadM2RuntimeModel(M2StaticRenderModel runtimeModel, MdxFile? adaptedMdx = null, string? modelDir = null, string? virtualPath = null)
    {
        ArgumentNullException.ThrowIfNull(runtimeModel);

        CaptureWorldReturnState();
        ExitToStandaloneView();

        _loadedWmo = null;
        _loadedMdx = null;
        _loadedM2Runtime = runtimeModel;
        string sourceModelPath = virtualPath ?? runtimeModel.Model.Identity.CanonicalModelPath;
        _renderer = WowViewerM2RuntimeBridge.CreateRenderer(
            _gl,
            runtimeModel,
            adaptedMdx,
            modelDir,
            _dataSource,
            _texResolver,
            _dbcBuild,
            sourceModelPath);
        RefreshStandaloneCharacterCustomizationState(sourceModelPath, isM2AdapterModel: adaptedMdx != null);
        ApplyStandaloneCharacterCustomizationOverrides();

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        int sectionCount = runtimeModel.Sections.Count;
        int vertexCount = runtimeModel.Sections.Sum(static section => section.Vertices.Count);
        int triangleCount = runtimeModel.Sections.Sum(static section => section.Indices.Count / 3);
        int transparentSectionCount = runtimeModel.Sections.Count(static section => section.Material.IsTransparent);
        List<string> textureNames = runtimeModel.Sections
            .Select(static section => section.Material.TexturePath)
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList()!;

        bool usesNativeStaticRenderer = WowViewerM2RuntimeBridge.ShouldUseNativeStaticRenderer(adaptedMdx);
        string runtimeTypeLabel = usesNativeStaticRenderer
            ? "M2 (wow-viewer static renderer in WoWViewer)"
            : "M2 (wow-viewer runtime + legacy draw backend)";

        _modelInfo = $"Path: {virtualPath ?? runtimeModel.Model.Identity.CanonicalModelPath}\n" +
                     $"Type: {runtimeTypeLabel}\n" +
                     $"Version: {runtimeModel.Model.Version}\n" +
                     $"Name: {runtimeModel.Model.ModelName ?? Path.GetFileNameWithoutExtension(runtimeModel.Model.Identity.CanonicalModelPath)}\n\n" +
                     $"Sections: {sectionCount}\n" +
                     $"Transparent Sections: {transparentSectionCount}\n" +
                     $"Vertices: {vertexCount:N0}\n" +
                     $"Triangles: {triangleCount:N0}\n" +
                     $"Bounds Radius: {runtimeModel.Model.BoundsRadius:F3}\n";

        if (textureNames.Count > 0)
        {
            _modelInfo += "\nTextures:\n";
            foreach (string textureName in textureNames)
                _modelInfo += $"  {textureName}\n";
        }

        _modelInfo += "\nRuntime Notes:\n" +
                      "  Geometry is submitted from wow-viewer active skin sections.\n" +
                      (usesNativeStaticRenderer
                          ? "  Draw path: Native wow-viewer runtime renderer in WoWViewer.\n  Skeletal sequence playback advances through wow-viewer pose evaluation.\n  Shading: Native runtime material pipeline (textured diffuse, directional + ambient lighting, alpha cutout & blending).\n"
                          : "  Draw path: Legacy MDX backend compatibility pipeline.\n");

        _statusMessage = $"Loaded M2: {_loadedFileName} ({sectionCount} sections, {vertexCount:N0} verts, {triangleCount:N0} tris)";
    }

    private void LoadStandaloneCameraPathModel(M2ModelDocument cameraModel, M2CameraPathVisualization visualization, string virtualPath)
    {
        ArgumentNullException.ThrowIfNull(cameraModel);
        ArgumentNullException.ThrowIfNull(visualization);

        CaptureWorldReturnState();
        ExitToStandaloneView();

        _loadedWmo = null;
        _loadedMdx = null;
        _loadedM2Runtime = null;
        _renderer = new M2CameraPathRenderer(_gl, visualization, virtualPath);
        ClearStandaloneCharacterCustomizationState(resetOverrides: true);

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        var info = new StringBuilder();
        info.AppendLine($"Path: {virtualPath}");
        info.AppendLine("Type: M2 camera path");
        info.AppendLine($"Version: {cameraModel.Version}");
        info.AppendLine($"Name: {cameraModel.ModelName ?? Path.GetFileNameWithoutExtension(cameraModel.Identity.CanonicalModelPath)}");
        info.AppendLine();
        info.AppendLine($"Cameras: {cameraModel.CameraCount}");
        info.AppendLine($"Sequences: {cameraModel.SequenceCount}");
        info.AppendLine($"Bounds Radius: {cameraModel.BoundsRadius:F3}");
        info.AppendLine();
        info.AppendLine("Camera Definitions:");

        foreach (M2CameraDefinition camera in cameraModel.Cameras)
        {
            string typeLabel = DescribeStandaloneCameraType(camera.Type);
            string fovLabel = camera.HasAnimatedFieldOfView
                ? "animated FoV"
                : $"FoV {camera.StaticFieldOfView.GetValueOrDefault():F3} rad";
            info.AppendLine($"  [{camera.Index}] {typeLabel}: near {camera.NearClip:F2}, far {camera.FarClip:F2}, {fovLabel}");
        }

        info.AppendLine();
        info.AppendLine("Runtime Notes:");
        info.AppendLine("  Geometry-less camera-only M2 assets are visualized as sampled camera and target paths.");
        info.AppendLine("  This path intentionally bypasses .skin resolution because flyby cameras can be valid MD20 assets without mesh data.");

        _modelInfo = info.ToString();
        _statusMessage = $"Loaded M2 camera path: {_loadedFileName} ({cameraModel.CameraCount} cameras)";
    }

    private static string DescribeStandaloneCameraType(int cameraType)
    {
        return cameraType switch
        {
            0 => "portrait",
            1 => "character info",
            -1 => "flyby",
            _ => $"type {cameraType}",
        };
    }

    private MdxRuntimeSharedInfo? TryReadSharedMdxRuntimeInfo(string sourcePath, byte[] modelBytes)
    {
        CoreMdxSummary? summary = null;
        CoreMdxGeometryFile? geometry = null;

        try
        {
            using var summaryStream = new MemoryStream(modelBytes, writable: false);
            summary = MdxSummaryReader.Read(summaryStream, sourcePath);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[SharedMDX] Summary metadata unavailable for runtime consumer {Path.GetFileName(sourcePath)}: {ex.Message}");
        }

        try
        {
            using var geometryStream = new MemoryStream(modelBytes, writable: false);
            geometry = MdxGeometryReader.Read(geometryStream, sourcePath);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[SharedMDX] GEOS metadata unavailable for runtime consumer {Path.GetFileName(sourcePath)}: {ex.Message}");
        }

        if (summary == null && geometry == null)
            return null;

        return new MdxRuntimeSharedInfo(summary, geometry);
    }

    private readonly record struct MdxRuntimeSharedInfo(
        CoreMdxSummary? Summary,
        CoreMdxGeometryFile? Geometry);

    /// <summary>
    /// Tears down the world/terrain scene so the viewer switches to standalone
    /// object-view mode (WMO or M2 model rendering without the world scene).
    /// </summary>
    private void ExitToStandaloneView()
    {
        _loadingScreen?.Disable();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        ResetSqlSpawnStreamingState(clearSceneSpawns: false);
    }

    private void LoadWmoModel(WmoV14ToV17Converter.WmoV14Data wmo, string dir)
    {
        // Loading a standalone WMO fully switches the viewer to object-view mode — the render
        // path draws this WMO and no longer draws the world scene. Tear down any lingering
        // world/terrain scene so the object-view UI drives THIS renderer. Otherwise the sidebar
        // keys off the still-alive _worldScene/_terrainManager and shows world-scene controls
        // (e.g. the "M2/WMO WF" wireframe checkbox drives the dormant world scene, so toggling it
        // has no visible effect on the loaded WMO — the object-view wireframe checkbox is skipped
        // because a stale terrain renderer is still present).
        ExitToStandaloneView();

        _loadedMdx = null;
        _loadedM2Runtime = null;
        _loadedWmo = wmo;
        
        int totalVerts = wmo.Groups.Sum(g => g.Vertices.Count);
        int totalTris = wmo.Groups.Sum(g => g.Indices.Count / 3);

        _renderer = new WmoRenderer(_gl, wmo, dir, _dataSource, _texResolver, _dbcBuild,
            enableRuntimeGroupVisibility: false);

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        var wmoCenter = (wmo.BoundsMin + wmo.BoundsMax) * 0.5f;
        var wmoExtent = wmo.BoundsMax - wmo.BoundsMin;

        // Position camera offset from WMO center
        float dist = Math.Max(wmoExtent.Length() * 1.5f, 100f);
        _camera.Position = wmoCenter + new System.Numerics.Vector3(dist, 0, wmoExtent.Z * 0.3f);
        _camera.Yaw = 180f;
        _camera.Pitch = -10f;

        _modelInfo = $"Path: {_loadedFileName ?? "<unknown>"}\n" +
                     $"Type: WMO v{wmo.Version}\n\n" +
                     $"Groups: {wmo.Groups.Count}\n" +
                     $"Vertices: {totalVerts:N0}\n" +
                     $"Triangles: {totalTris:N0}\n\n" +
                     $"Materials: {wmo.Materials.Count}\n" +
                     $"Textures: {wmo.Textures.Count}\n" +
                     $"Doodad Sets: {wmo.DoodadSets.Count}\n" +
                     $"Doodad Defs: {wmo.DoodadDefs.Count}\n" +
                     $"Portals: {wmo.Portals.Count}\n" +
                     $"Lights: {wmo.Lights.Count}\n";

        if (wmo.DoodadSets.Count > 0)
        {
            _modelInfo += "\nDoodad Sets:\n";
            for (int i = 0; i < wmo.DoodadSets.Count; i++)
            {
                var ds = wmo.DoodadSets[i];
                _modelInfo += $"  [{i}] {ds.Name ?? "unnamed"} ({ds.Count} doodads)\n";
            }
        }

        if (wmo.Textures.Count > 0)
        {
            _modelInfo += "\nTextures:\n";
            foreach (var tex in wmo.Textures)
                _modelInfo += $"  {tex}\n";
        }

        if (wmo.Groups.Count > 0)
        {
            _modelInfo += "\nGroups:\n";
            for (int i = 0; i < wmo.Groups.Count; i++)
            {
                var g = wmo.Groups[i];
                string name = g.Name ?? $"group_{i}";
                _modelInfo += $"  [{i}] {name} ({g.Vertices.Count}v, {g.Indices.Count / 3}t)\n";
            }
        }

        _statusMessage = $"Loaded WMO: {_loadedFileName} ({wmo.Groups.Count} groups, {totalVerts:N0} verts, {wmo.DoodadDefs.Count} doodads)";
    }

    private void LoadWdtTerrain(string wdtPath)
    {
        _statusMessage = $"Loading world from {Path.GetFileName(wdtPath)}...";

        _terrainWeakSignalRestore.ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        ResetSqlSpawnStreamingState(clearSceneSpawns: false);

        // Show loading screen (replicates Alpha client's EnableLoadingScreen)
        _loadingScreen?.Enable(_dataSource);
        PresentLoadingFrame();

        try
        {
            // Detect Alpha WDT vs Standard WDT by checking for MDNM chunk.
            // Alpha WDTs are monolithic: MVER+MPHD+MAIN+MDNM+MONM+embedded ADTs.
            // Standard WDTs have: MVER+MPHD+MAIN only, referencing external .adt files.
            var wdtRawBytes = File.ReadAllBytes(wdtPath);
            bool isAlpha = DetectAlphaWdt(wdtRawBytes);
            string wdtType;
            int loadStep = 0;

            // onStatus callback: update loading screen progress and force-present a frame.
            // This replicates the Alpha client's UpdateProgressBar → GxScenePresent pattern.
            void OnLoadStatus(string status)
            {
                _statusMessage = status;
                loadStep++;
                _loadingScreen?.UpdateProgress(loadStep, 20); // Estimate ~20 status updates per load
                PresentLoadingFrame();
            }

            if (isAlpha)
            {
                // Alpha WDT: monolithic file with embedded ADTs
                _worldScene = new WorldScene(_gl, wdtPath, _dataSource, _texResolver, _dbcBuild, _minimapRenderer,
                    onStatus: OnLoadStatus);
                wdtType = "Alpha WDT";
            }
            else
            {
                // Standard WDT: small file referencing separate ADT files via IDataSource (MPQ)
                if (_dataSource == null)
                {
                    _loadingScreen?.Disable();
                    _statusMessage = "Standard WDT requires an MPQ data source. Open a game folder first.";
                    _modelInfo = "Standard WDT detected but no data source loaded.\n\nUse File > Open Game Folder to load MPQ archives first,\nthen open the WDT from the file browser.";
                    return;
                }

                string mapName = Path.GetFileNameWithoutExtension(wdtPath);
                var adapter = new Terrain.StandardTerrainAdapter(wdtRawBytes, mapName, _dataSource, _dbcBuild, _dbcProvider, _dbdDir);
                var tm = new Terrain.TerrainManager(_gl, adapter, mapName, _dataSource);
                _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver, _dbcBuild, _minimapRenderer,
                    onStatus: OnLoadStatus);
                wdtType = "Standard WDT";
            }

            _terrainManager = _worldScene.Terrain;
            _terrainManager.DetailedTileCountOverride = _savedDetailedAdtTileCountOverride;
            ApplyGlobalFogDefaults(_terrainManager.Lighting);
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreHooks();
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _terrainWeakSignalRestore.WireBaseWdlEdgeBlendLookup();
            _renderer = _worldScene;
            ApplyLayoutObjectPreviewModeToScene();
            ApplySavedPm4AlignmentToScene();
            ApplySavedObjectPathFiltersForCurrentMap();
            // Full-load mode: load all tiles synchronously during loading screen
            if (FullLoadMode && !_terrainManager.Adapter.IsWmoBased)
            {
                int total = _terrainManager.Adapter.ExistingTiles.Count;
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"Full-load mode: loading all {total} tiles...");
                _terrainManager.LoadAllTiles((loaded, tot, tileName) =>
                {
                    _statusMessage = $"Loading tiles... {loaded}/{tot} ({tileName})";
                    _loadingScreen?.UpdateProgress(loaded, tot);
                    PresentLoadingFrame();
                });
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"Full-load complete: {_terrainManager.LoadedTileCount} tiles, {_terrainManager.LoadedChunkCount} chunks");
            }

            // Find mapId for this world
            string curMapName = _terrainManager.MapName;
            var curMapDef = _discoveredMaps.FirstOrDefault(m =>
                string.Equals(m.Directory, curMapName, StringComparison.OrdinalIgnoreCase));
            _currentMapId = curMapDef?.HasDbcEntry == true ? curMapDef.Id : -1;
            _reportedAreaDiagnostics.Clear();
            ViewerLog.Important(ViewerLog.Category.General,
                $"[WorldLoad] Map='{curMapName}' resolvedMapId={_currentMapId} build={_dbcBuild ?? "unknown"} areaTable={_areaTableService?.DescribeLoadContext() ?? "not loaded"}");
            _sqlForceStreamRefresh = true;

            // Store DBC credentials for lazy loading (POI + Taxi deferred until user toggles them on)
            // Only Lighting is loaded eagerly since it affects rendering immediately.
            if (_dbcProvider != null && _dbdDir != null && _dbcBuild != null)
            {
                int mapId = curMapDef?.HasDbcEntry == true ? curMapDef.Id : -1;
                _worldScene.SetDbcCredentials(_dbcProvider, _dbdDir, _dbcBuild, mapId);

                _worldScene.LoadLighting(_dbcProvider, _dbdDir, _dbcBuild, mapId);
            }
            else
            {
                _worldScene.EnableLitFallback(
                    "No Light DBC provider is available for this client; LIT is enabled automatically.");
            }

            // Position camera — WMO-only maps use the WMO position, terrain maps use tile center
            var startPos = _pendingWorldSpawnOverride ?? _worldScene.WmoCameraOverride ?? _terrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _pendingWorldSpawnOverride = null;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;
            if (!_terrainManager.Adapter.IsWmoBased)
                _terrainManager.UpdateAOI(startPos, _camera.Forward);

            int poiCount = _worldScene.PoiLoader?.Entries.Count ?? 0;
            int taxiNodeCount = _worldScene.TaxiLoader?.Nodes.Count ?? 0;
            int taxiRouteCount = _worldScene.TaxiLoader?.Routes.Count ?? 0;
            _modelInfo = $"Type: {wdtType} World\n" +
                         $"Map: {_terrainManager.MapName}\n\n" +
                         $"Tiles: {_terrainManager.LoadedTileCount}\n" +
                         $"Chunks: {_terrainManager.LoadedChunkCount}\n\n" +
                         $"WMO instances: {_worldScene.WmoInstanceCount} ({_worldScene.UniqueWmoModels} unique)\n" +
                         $"MDX instances: {_worldScene.MdxInstanceCount} ({_worldScene.UniqueMdxModels} unique)\n" +
                         (poiCount > 0 ? $"Area POIs: {poiCount}\n" : "") +
                         (taxiNodeCount > 0 ? $"Taxi Nodes: {taxiNodeCount}, Routes: {taxiRouteCount}\n" : "") +
                         $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";

            _statusMessage = $"Loaded world: {_terrainManager.MapName} ({_terrainManager.LoadedTileCount} tiles, {_worldScene.WmoInstanceCount} WMOs, {_worldScene.MdxInstanceCount} doodads)";

            // Signal world loaded (progress → 75%). Loading screen stays active
            // until the first terrain tiles are actually rendered (checked in OnRender).
            _loadingScreen?.SetWorldLoaded();
            PresentLoadingFrame();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] WDT load failed: {ex}");
            _statusMessage = $"Load failed: {ex.Message}";
            _modelInfo = $"WDT load error:\n{ex.Message}\n\nFile: {wdtPath}\nSize: {(File.Exists(wdtPath) ? new FileInfo(wdtPath).Length : 0)} bytes";
            _worldScene?.Dispose();
            _worldScene = null;
            _terrainManager = null;
            InvalidatePm4DerivedReports();
            _loadingScreen?.Disable();
        }
    }

    private void LoadRosettaDatastoreTerrain(WowViewer.Core.IO.Maps.RosettaObjectLibrary library, string buildId, string mapName)
    {
        _statusMessage = $"Loading Rosetta world '{mapName}' [{buildId}] from Zarr datastore...";

        _terrainWeakSignalRestore.ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        ResetSqlSpawnStreamingState(clearSceneSpawns: false);

        // Show loading screen
        _loadingScreen?.Enable(_dataSource);
        PresentLoadingFrame();

        try
        {
            int loadStep = 0;
            void OnLoadStatus(string status)
            {
                _statusMessage = status;
                loadStep++;
                _loadingScreen?.UpdateProgress(loadStep, 20);
                PresentLoadingFrame();
            }

            var adapter = new Terrain.RosettaDatastoreTerrainAdapter(library, buildId, mapName, _dataSource);
            var tm = new Terrain.TerrainManager(_gl, adapter, mapName, _dataSource);
            _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver, _dbcBuild, _minimapRenderer,
                onStatus: OnLoadStatus);

            _terrainManager = _worldScene.Terrain;
            _terrainManager.DetailedTileCountOverride = _savedDetailedAdtTileCountOverride;
            _terrainWeakSignalRestore.WireBaseWdlEdgeBlendLookup();
            ApplyGlobalFogDefaults(_terrainManager.Lighting);
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreHooks();
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _renderer = _worldScene;
            ApplyLayoutObjectPreviewModeToScene();
            ApplySavedPm4AlignmentToScene();
            ApplySavedObjectPathFiltersForCurrentMap();

            _worldScene.EnableLitFallback(
                "Rosetta Zarr map loaded directly; LIT/analytical lighting enabled.");

            // Full-load mode: load all tiles synchronously during loading screen
            if (FullLoadMode && !_terrainManager.Adapter.IsWmoBased)
            {
                int total = _terrainManager.Adapter.ExistingTiles.Count;
                _terrainManager.LoadAllTiles((loaded, tot, tileName) =>
                {
                    _statusMessage = $"Loading tiles... {loaded}/{tot} ({tileName})";
                    _loadingScreen?.UpdateProgress(loaded, tot);
                    PresentLoadingFrame();
                });
            }

            // Position camera at initial tile center
            var startPos = _pendingWorldSpawnOverride ?? _worldScene.WmoCameraOverride ?? _terrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _pendingWorldSpawnOverride = null;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;
            if (!_terrainManager.Adapter.IsWmoBased)
                _terrainManager.UpdateAOI(startPos, _camera.Forward);

            _modelInfo = $"Type: Rosetta Zarr Datastore World\n" +
                         $"Build: {buildId}\n" +
                         $"Map: {mapName}\n\n" +
                         $"Tiles: {adapter.ExistingTiles.Count}\n" +
                         $"WMO instances: {_worldScene.WmoInstanceCount} ({_worldScene.UniqueWmoModels} unique)\n" +
                         $"MDX instances: {_worldScene.MdxInstanceCount} ({_worldScene.UniqueMdxModels} unique)\n" +
                         $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";

            _statusMessage = $"Loaded Rosetta Zarr World: {mapName} [{buildId}] ({adapter.ExistingTiles.Count} tiles, {_worldScene.WmoInstanceCount} WMOs, {_worldScene.MdxInstanceCount} doodads)";

            _loadingScreen?.SetWorldLoaded();
            PresentLoadingFrame();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] Rosetta Datastore load failed: {ex}");
            _statusMessage = $"Load failed: {ex.Message}";
            _modelInfo = $"Rosetta Datastore load error:\n{ex.Message}\n\nBuild: {buildId}\nMap: {mapName}";
            _worldScene?.Dispose();
            _worldScene = null;
            _terrainManager = null;
            InvalidatePm4DerivedReports();
            _loadingScreen?.Disable();
        }
    }

    private void RefreshStandaloneCharacterCustomizationState(string? modelPath, bool isM2AdapterModel)
    {
        if (_texResolver == null || string.IsNullOrWhiteSpace(modelPath))
        {
            ClearStandaloneCharacterCustomizationState(resetOverrides: true);
            return;
        }

        string normalizedPath = modelPath.Replace('/', '\\');
        if (_texResolver.GetDefaultCharacterSelectionGroups(normalizedPath) == null)
        {
            ClearStandaloneCharacterCustomizationState(resetOverrides: true);
            return;
        }

        bool preserveExistingSelection = _preserveStandaloneCharacterCustomizationOnNextLoad
            || string.Equals(_standaloneCharacterCustomizationModelPath, normalizedPath, StringComparison.OrdinalIgnoreCase);

        _standaloneCharacterCustomizationModelPath = normalizedPath;
        _standaloneCharacterHairVariationIds.Clear();
        _standaloneCharacterHairVariationIds.AddRange(_texResolver.GetCharacterHairVariationIds(normalizedPath));
        _standaloneCharacterFacialHairVariationIds.Clear();
        _standaloneCharacterFacialHairVariationIds.AddRange(_texResolver.GetCharacterFacialHairVariationIds(normalizedPath));

        if (!preserveExistingSelection)
        {
            _standaloneCharacterHairVariationOverride = -1;
            _standaloneCharacterFacialHairVariationOverride = -1;
        }

        NormalizeStandaloneCharacterCustomizationSelection();
        _preserveStandaloneCharacterCustomizationOnNextLoad = false;

        ApplyStandaloneCharacterCustomizationOverrides();
    }

    private void ClearStandaloneCharacterCustomizationState(bool resetOverrides)
    {
        _standaloneCharacterCustomizationModelPath = null;
        _standaloneCharacterHairVariationIds.Clear();
        _standaloneCharacterFacialHairVariationIds.Clear();
        _preserveStandaloneCharacterCustomizationOnNextLoad = false;

        if (!resetOverrides)
            return;

        _standaloneCharacterHairVariationOverride = -1;
        _standaloneCharacterFacialHairVariationOverride = -1;
    }

    private void PrepareStandaloneCharacterCustomizationForNextLoad(int? hairVariationId, int? facialHairVariationId)
    {
        _standaloneCharacterHairVariationOverride = hairVariationId is >= 0 ? hairVariationId.Value : -1;
        _standaloneCharacterFacialHairVariationOverride = facialHairVariationId is >= 0 ? facialHairVariationId.Value : -1;
        _preserveStandaloneCharacterCustomizationOnNextLoad = hairVariationId.HasValue || facialHairVariationId.HasValue;
    }

    private void NormalizeStandaloneCharacterCustomizationSelection()
    {
        if (_standaloneCharacterHairVariationOverride >= 0
            && !_standaloneCharacterHairVariationIds.Contains(_standaloneCharacterHairVariationOverride))
        {
            _standaloneCharacterHairVariationOverride = -1;
        }

        if (_standaloneCharacterFacialHairVariationOverride >= 0
            && !_standaloneCharacterFacialHairVariationIds.Contains(_standaloneCharacterFacialHairVariationOverride))
        {
            _standaloneCharacterFacialHairVariationOverride = -1;
        }
    }

    private void ApplyStandaloneCharacterCustomizationOverrides()
    {
        if (_texResolver == null || string.IsNullOrWhiteSpace(_standaloneCharacterCustomizationModelPath))
            return;

        IReadOnlyCollection<uint>? selectedGroups = _texResolver.GetCharacterSelectionGroups(
            _standaloneCharacterCustomizationModelPath,
            _standaloneCharacterHairVariationOverride >= 0 ? _standaloneCharacterHairVariationOverride : null,
            _standaloneCharacterFacialHairVariationOverride >= 0 ? _standaloneCharacterFacialHairVariationOverride : null);
        if (selectedGroups == null)
            return;

        string reasonLabel = _standaloneCharacterHairVariationOverride >= 0 || _standaloneCharacterFacialHairVariationOverride >= 0
            ? $"character geosets (hair={FormatStandaloneCharacterVariationLabel(_standaloneCharacterHairVariationOverride)}, facial={FormatStandaloneCharacterVariationLabel(_standaloneCharacterFacialHairVariationOverride)})"
            : "default character geosets";

        int? hairVariationId = _standaloneCharacterHairVariationOverride >= 0 ? _standaloneCharacterHairVariationOverride : null;
        int? facialHairVariationId = _standaloneCharacterFacialHairVariationOverride >= 0 ? _standaloneCharacterFacialHairVariationOverride : null;

        switch (_renderer)
        {
            case MdxRenderer mdxRenderer:
                mdxRenderer.TryApplyCharacterCustomization(selectedGroups, hairVariationId, facialHairVariationId, reasonLabel);
                break;

            case M2Renderer m2Renderer:
                m2Renderer.TryApplyCharacterCustomization(selectedGroups, hairVariationId, facialHairVariationId, reasonLabel);
                break;
        }
    }

    private static string FormatStandaloneCharacterVariationLabel(int variationId)
        => variationId >= 0 ? variationId.ToString() : "default";

    /// <summary>
    /// Force-present a loading screen frame. Replicates the Alpha client's
    /// UpdateProgressBar → GxScenePresent pattern: clear, draw loading screen, swap.
    /// Called from the blocking WorldScene constructor via onStatus callback.
    /// </summary>
    private void PresentLoadingFrame()
    {
        if (_loadingScreen == null || !_loadingScreen.IsActive) return;
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
        var sz = _window.Size;
        _loadingScreen.Render(sz.X, sz.Y);
        _window.GLContext?.SwapBuffers();
    }

    private void LoadZarrDataset(string datasetRoot)
    {
        try
        {
            var loader = new ZarrTileDatasetLoader(datasetRoot);
            ZarrStoreSummary summary = loader.Open();
            _statusMessage =
                $"Zarr store '{summary.MapName}' discovered: {summary.Arrays.Count} arrays; " +
                $"liquid mask={(summary.HasLiquidMask ? "yes" : "no")}, " +
                $"height={(summary.HasLiquidHeight ? "yes" : "no")}, " +
                $"type={(summary.HasLiquidType256 || summary.HasLiquidBasicType || summary.HasMh2oTypeMask ? "yes" : "no")}; " +
                $"objects={(summary.HasObjectSignals ? "yes" : "no")}, " +
                $"tilesets={(summary.HasTilesetSignals ? "yes" : "no")}, " +
                $"textures={(summary.HasTextureSignals ? "yes" : "no")}, " +
                $"placements={(summary.HasPlacementSignals ? "yes" : "no")}. " +
                "Summary only: per-tile loading still requires the Zarr decoder/rehydration slice.";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Zarr dataset load failed: {ex.Message}";
        }
    }

    private void LoadVlmProject(string projectRoot)
    {
        _statusMessage = $"Loading MK dataset from {projectRoot}...";

        _terrainAnalysisHiddenCandidates.Clear();
        _terrainAnalysisHiddenSelectedIndex = -1;
        _terrainAnalysisHiddenStatus = string.Empty;
        _terrainAnalysisPreviewCompareTile = null;
        _terrainAnalysisPreviewSimilarity = null;
        _terrainWeakSignalRestore.ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);

        // Clean up any existing scene
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        _renderer = null;

        try
        {
            _vlmTerrainManager = new VlmTerrainManager(_gl, projectRoot);
            ApplyGlobalFogDefaults(_vlmTerrainManager.Lighting);
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreHooks();
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _renderer = _vlmTerrainManager;

            // Position camera at center of loaded tiles
            var startPos = _vlmTerrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;

            var loader = _vlmTerrainManager.Loader;
            _modelInfo = $"Type: MK Dataset\n" +
                         $"Map: {loader.MapName}\n" +
                         $"Path: {projectRoot}\n\n" +
                         $"Tiles: {loader.TileCoords.Count}\n" +
                         $"MDX names: {loader.MdxModelNames.Count}\n" +
                         $"WMO names: {loader.WmoModelNames.Count}\n" +
                         $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";

            // Set MapID for AreaTable lookups
            var vlmMapDef = _discoveredMaps.FirstOrDefault(m =>
                string.Equals(m.Directory, loader.MapName, StringComparison.OrdinalIgnoreCase));
            _currentMapId = vlmMapDef?.Id ?? -1;
            _statusMessage = $"Loaded MK dataset: {loader.MapName} ({loader.TileCoords.Count} tiles)";
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] VLM project load failed: {ex}");
            _statusMessage = $"MK dataset load failed: {ex.Message}";
            _modelInfo = $"MK dataset load error:\n{ex.Message}\n\nPath: {projectRoot}";
            _vlmTerrainManager?.Dispose();
            _vlmTerrainManager = null;
        }
    }

    private void SelectTaxiNode(int nodeId, bool toggle)
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        int nextNodeId = toggle && _worldScene.SelectedTaxiNodeId == nodeId ? -1 : nodeId;
        _worldScene.SelectedTaxiNodeId = nextNodeId;
        _worldScene.ClearSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
        ClearSelectedAreaPoiInfo();

        if (nextNodeId < 0)
        {
            ClearSelectedTaxiInfo();
            return;
        }

        RefreshSelectedTaxiInfo();
    }

    private void SelectTaxiRoute(int pathId, bool toggle)
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        int nextRouteId = toggle && _worldScene.SelectedTaxiRouteId == pathId ? -1 : pathId;
        _worldScene.SelectedTaxiRouteId = nextRouteId;
        _worldScene.ClearSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
        ClearSelectedAreaPoiInfo();

        if (nextRouteId < 0)
        {
            ClearSelectedTaxiInfo();
            return;
        }

        RefreshSelectedTaxiInfo();
    }

    private void RefreshSelectedTaxiInfo()
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        _selectedObjectIndex = -1;

        if (_worldScene.SelectedTaxiNodeId >= 0)
        {
            var node = _worldScene.GetTaxiNode(_worldScene.SelectedTaxiNodeId);
            if (node == null)
            {
                ClearSelectedTaxiInfo();
                return;
            }

            int routeCount = _worldScene.TaxiLoader.Routes.Count(route => route.FromNodeId == node.Id || route.ToNodeId == node.Id);
            string mountCreatureIds = node.MountCreatureIds.Length > 0
                ? string.Join(", ", node.MountCreatureIds.Where(id => id > 0))
                : "none";

            _selectedObjectType = "Taxi Node";
            _selectedObjectInfo =
                $"Taxi Node [{node.Id}] {node.Name}\n" +
                $"Position: ({node.Position.X:F1}, {node.Position.Y:F1}, {node.Position.Z:F1})\n" +
                $"Routes: {routeCount}\n" +
                $"Mount Creature IDs: {mountCreatureIds}\n" +
                $"Resolved Mount Creature: {node.MountCreatureId}\n" +
                $"Resolved Display ID: {node.MountDisplayId}\n" +
                $"Resolved Model: {node.MountModelPath ?? "not found"}";
            return;
        }

        if (_worldScene.SelectedTaxiRouteId >= 0)
        {
            var route = _worldScene.GetTaxiRoute(_worldScene.SelectedTaxiRouteId);
            if (route == null)
            {
                ClearSelectedTaxiInfo();
                return;
            }

            var fromNode = _worldScene.GetTaxiNode(route.FromNodeId);
            var toNode = _worldScene.GetTaxiNode(route.ToNodeId);
            TaxiPathLoader.TaxiNode? mountNode = fromNode;
            if (mountNode == null || string.IsNullOrWhiteSpace(mountNode.MountModelPath))
                mountNode = toNode;

            string fromName = fromNode?.Name ?? $"#{route.FromNodeId}";
            string toName = toNode?.Name ?? $"#{route.ToNodeId}";
            string? actorOverridePath = _worldScene.GetTaxiActorModelOverride(route.PathId);
            string resolvedActorModelPath = _worldScene.GetResolvedTaxiActorModelPath(route.PathId) ?? "not found";

            _selectedObjectType = "Taxi Route";
            _selectedObjectInfo =
                $"Taxi Route [{route.PathId}]\n" +
                $"From: {fromName}\n" +
                $"To: {toName}\n" +
                $"Cost: {route.Cost}\n" +
                $"Waypoints: {route.Waypoints.Count}\n" +
                $"Actor Override: {actorOverridePath ?? "auto"}\n" +
                $"Resolved Actor Model: {resolvedActorModelPath}";
            return;
        }

        ClearSelectedTaxiInfo();
    }

    private void SelectAreaPoi(int poiId, bool toggle)
    {
        if (_worldScene?.PoiLoader == null)
            return;

        int nextPoiId = toggle && _selectedAreaPoiId == poiId ? -1 : poiId;
        _selectedAreaPoiId = nextPoiId;
        _worldScene.ClearSelection();
        _worldScene.ClearTaxiSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();

        if (nextPoiId < 0)
        {
            ClearSelectedAreaPoiInfo();
            return;
        }

        RefreshSelectedAreaPoiInfo();
    }

    private void RefreshSelectedAreaPoiInfo()
    {
        if (_worldScene?.PoiLoader == null || _selectedAreaPoiId < 0)
        {
            ClearSelectedAreaPoiInfo();
            return;
        }

        AreaPoiLoader.AreaPoiEntry? poi = _worldScene.PoiLoader.Entries
            .FirstOrDefault(entry => entry.Id == _selectedAreaPoiId);
        if (poi == null)
        {
            ClearSelectedAreaPoiInfo();
            return;
        }

        _selectedObjectIndex = -1;
        _selectedObjectType = "Area POI";
        _selectedObjectInfo =
            $"Area POI [{poi.Id}] {poi.Name}\n" +
            $"Position: ({poi.Position.X:F1}, {poi.Position.Y:F1}, {poi.Position.Z:F1})\n" +
            $"WoW Position: ({poi.WoWPosition.X:F1}, {poi.WoWPosition.Y:F1}, {poi.WoWPosition.Z:F1})\n" +
            $"Icon: {poi.Icon}\n" +
            $"Importance: {poi.Importance}\n" +
            $"Flags: 0x{poi.Flags:X}\n" +
            $"Continent ID: {poi.ContinentId}";
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
            OpenWorkbenchTab(ModelBottomTab.Info);

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

    private void ClearSelectedTaxiInfo()
    {
        if (!_selectedObjectType.StartsWith("Taxi", StringComparison.OrdinalIgnoreCase))
            return;

        _selectedObjectIndex = -1;
        _selectedObjectType = "";
        _selectedObjectInfo = "";
        _taxiActorModelOverrideInput = "";
        _taxiActorModelOverrideInputRouteId = -1;
        _taxiActorModelOverrideTargetRouteId = -1;
    }

    private void ClearSelectedAreaPoiInfo()
    {
        _selectedAreaPoiId = -1;
        if (!string.Equals(_selectedObjectType, "Area POI", StringComparison.OrdinalIgnoreCase))
            return;

        _selectedObjectIndex = -1;
        _selectedObjectType = "";
        _selectedObjectInfo = "";
    }

    private bool TryPickTaxiNodeAtMouse(float localX, float localY, float viewportWidth, float viewportHeight, Matrix4x4 view, Matrix4x4 proj, out int nodeId)
    {
        nodeId = -1;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        const float pickRadiusPixels = 48f;
        float bestDistanceSq = pickRadiusPixels * pickRadiusPixels;

        foreach (var node in _worldScene.TaxiLoader.Nodes)
        {
            if (!_worldScene.IsTaxiNodeVisible(node))
                continue;

            if (!TryProjectWorldToViewport(node.Position + new Vector3(0f, 0f, 50f), view, proj, viewportWidth, viewportHeight, out Vector2 projected))
                continue;

            float dx = projected.X - localX;
            float dy = projected.Y - localY;
            float distSq = dx * dx + dy * dy;
            if (distSq > bestDistanceSq)
                continue;

            bestDistanceSq = distSq;
            nodeId = node.Id;
        }

        return nodeId >= 0;
    }

    private bool TryPickTaxiRouteAtMouse(float localX, float localY, float viewportWidth, float viewportHeight, Matrix4x4 view, Matrix4x4 proj, out int pathId)
    {
        pathId = -1;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        Vector2 pointer = new(localX, localY);

        const float handlePickRadiusPixels = 72f;
        float bestHandleDistSq = handlePickRadiusPixels * handlePickRadiusPixels;

        foreach (var route in _worldScene.TaxiLoader.Routes)
        {
            if (!_worldScene.IsTaxiRouteVisible(route))
                continue;

            if (!_worldScene.TryGetTaxiRouteSelectionPoint(route.PathId, out Vector3 selectionPoint))
                continue;

            if (!TryProjectWorldToViewport(selectionPoint + new Vector3(0f, 0f, 30f), view, proj, viewportWidth, viewportHeight, out Vector2 projected))
                continue;

            float distSq = Vector2.DistanceSquared(projected, pointer);
            if (distSq > bestHandleDistSq)
                continue;

            bestHandleDistSq = distSq;
            pathId = route.PathId;
        }

        if (pathId >= 0)
            return true;

        const float linePickRadiusPixels = 56f;
        float bestLineDistSq = linePickRadiusPixels * linePickRadiusPixels;

        foreach (var route in _worldScene.TaxiLoader.Routes)
        {
            if (!_worldScene.IsTaxiRouteVisible(route) || route.Waypoints.Count < 2)
                continue;

            for (int i = 0; i < route.Waypoints.Count - 1; i++)
            {
                if (!TryProjectWorldToViewport(route.Waypoints[i], view, proj, viewportWidth, viewportHeight, out Vector2 a)
                    || !TryProjectWorldToViewport(route.Waypoints[i + 1], view, proj, viewportWidth, viewportHeight, out Vector2 b))
                {
                    continue;
                }

                float distSq = DistanceSquaredPointToSegment(pointer, a, b);
                if (distSq > bestLineDistSq)
                    continue;

                bestLineDistSq = distSq;
                pathId = route.PathId;
            }
        }

        return pathId >= 0;
    }

    private bool TryPickTaxiNodeByRay(Vector3 rayOrigin, Vector3 rayDir, out int nodeId, out float hitDistance)
    {
        nodeId = -1;
        hitDistance = float.MaxValue;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        foreach (TaxiPathLoader.TaxiNode node in _worldScene.TaxiLoader.Nodes)
        {
            if (!_worldScene.IsTaxiNodeVisible(node))
                continue;

            float localDistance = RayAabbIntersect(
                rayOrigin,
                rayDir,
                node.Position - new Vector3(TaxiNodePickHalfWidth, TaxiNodePickHalfWidth, TaxiNodePickBottomPadding),
                node.Position + new Vector3(TaxiNodePickHalfWidth, TaxiNodePickHalfWidth, TaxiNodePickTopPadding));
            if (localDistance < 0f || localDistance >= hitDistance)
                continue;

            hitDistance = localDistance;
            nodeId = node.Id;
        }

        return nodeId >= 0;
    }

    private bool TryPickTaxiRouteByRay(Vector3 rayOrigin, Vector3 rayDir, out int pathId, out float hitDistance)
    {
        pathId = -1;
        hitDistance = float.MaxValue;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        foreach (TaxiPathLoader.TaxiRoute route in _worldScene.TaxiLoader.Routes)
        {
            if (!_worldScene.IsTaxiRouteVisible(route))
                continue;

            if (_worldScene.TryGetTaxiRouteSelectionPoint(route.PathId, out Vector3 selectionPoint))
            {
                float handleDistance = RayAabbIntersect(
                    rayOrigin,
                    rayDir,
                    selectionPoint - new Vector3(TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickBottomPadding),
                    selectionPoint + new Vector3(TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickTopPadding));
                if (handleDistance >= 0f && handleDistance < hitDistance)
                {
                    hitDistance = handleDistance;
                    pathId = route.PathId;
                }
            }

            if (route.Waypoints.Count < 2)
                continue;

            for (int index = 0; index < route.Waypoints.Count - 1; index++)
            {
                Vector3 segmentMin = Vector3.Min(route.Waypoints[index], route.Waypoints[index + 1])
                    - new Vector3(TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth);
                Vector3 segmentMax = Vector3.Max(route.Waypoints[index], route.Waypoints[index + 1])
                    + new Vector3(TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth);
                float segmentDistance = RayAabbIntersect(rayOrigin, rayDir, segmentMin, segmentMax);
                if (segmentDistance < 0f || segmentDistance >= hitDistance)
                    continue;

                hitDistance = segmentDistance;
                pathId = route.PathId;
            }
        }

        return pathId >= 0;
    }

    private bool TryPickAreaPoiAtMouse(float localX, float localY, float viewportWidth, float viewportHeight, Matrix4x4 view, Matrix4x4 proj, out int poiId)
    {
        poiId = -1;
        if (_worldScene?.PoiLoader == null || !_worldScene.ShowPoi)
            return false;

        const float pickRadiusPixels = 36f;
        float bestDistanceSq = pickRadiusPixels * pickRadiusPixels;
        Vector2 pointer = new(localX, localY);

        foreach (AreaPoiLoader.AreaPoiEntry poi in _worldScene.PoiLoader.Entries)
        {
            if (!TryProjectWorldToViewport(poi.Position + new Vector3(0f, 0f, 56f), view, proj, viewportWidth, viewportHeight, out Vector2 projected))
                continue;

            float distSq = Vector2.DistanceSquared(projected, pointer);
            if (distSq > bestDistanceSq)
                continue;

            bestDistanceSq = distSq;
            poiId = poi.Id;
        }

        return poiId >= 0;
    }

    private static float DistanceSquaredPointToSegment(Vector2 point, Vector2 start, Vector2 end)
    {
        Vector2 segment = end - start;
        float segmentLengthSq = segment.LengthSquared();
        if (segmentLengthSq <= 0.0001f)
            return Vector2.DistanceSquared(point, start);

        float t = Vector2.Dot(point - start, segment) / segmentLengthSq;
        t = Math.Clamp(t, 0f, 1f);
        Vector2 closest = start + segment * t;
        return Vector2.DistanceSquared(point, closest);
    }

    private static float RayAabbIntersect(Vector3 origin, Vector3 dir, Vector3 boundsMin, Vector3 boundsMax)
    {
        float tmin = 0f;
        float tmax = float.MaxValue;

        if (!UpdateRayAabbInterval(origin.X, dir.X, boundsMin.X, boundsMax.X, ref tmin, ref tmax)
            || !UpdateRayAabbInterval(origin.Y, dir.Y, boundsMin.Y, boundsMax.Y, ref tmin, ref tmax)
            || !UpdateRayAabbInterval(origin.Z, dir.Z, boundsMin.Z, boundsMax.Z, ref tmin, ref tmax))
        {
            return -1f;
        }

        return tmin >= 0f ? tmin : tmax >= 0f ? tmax : -1f;
    }

    private static bool UpdateRayAabbInterval(float origin, float direction, float min, float max, ref float tmin, ref float tmax)
    {
        if (MathF.Abs(direction) < 0.0001f)
            return origin >= min && origin <= max;

        float invDir = 1f / direction;
        float t1 = (min - origin) * invDir;
        float t2 = (max - origin) * invDir;
        if (t1 > t2)
            (t1, t2) = (t2, t1);

        tmin = MathF.Max(tmin, t1);
        tmax = MathF.Min(tmax, t2);
        return tmax >= tmin;
    }

    private void FocusSelectedTaxi()
    {
        if (_worldScene == null)
            return;

        if (_worldScene.SelectedTaxiRouteId >= 0)
        {
            int routeId = _worldScene.SelectedTaxiRouteId;
            if (_worldScene.TryGetTaxiRouteSelectionPoint(routeId, out Vector3 routePoint))
            {
                _camera.Position = routePoint + new Vector3(0f, 0f, 100f);
                _camera.Pitch = -30f;
                _statusMessage = $"Focused taxi route {routeId}.";
            }
            return;
        }

        if (_worldScene.SelectedTaxiNodeId >= 0)
        {
            TaxiPathLoader.TaxiNode? node = _worldScene.GetTaxiNode(_worldScene.SelectedTaxiNodeId);
            if (node != null)
            {
                _camera.Position = node.Position + new Vector3(0f, 0f, 50f);
                _camera.Pitch = -30f;
                _statusMessage = $"Focused taxi node {node.Id}.";
            }
        }
    }

    private IReadOnlyList<TaxiPathLoader.TaxiRoute> GetTaxiActorOverrideCandidateRoutes()
    {
        if (_worldScene?.TaxiLoader == null)
            return Array.Empty<TaxiPathLoader.TaxiRoute>();

        if (_worldScene.SelectedTaxiRouteId >= 0)
        {
            TaxiPathLoader.TaxiRoute? selectedRoute = _worldScene.GetTaxiRoute(_worldScene.SelectedTaxiRouteId);
            return selectedRoute != null
                ? new[] { selectedRoute }
                : Array.Empty<TaxiPathLoader.TaxiRoute>();
        }

        if (_worldScene.SelectedTaxiNodeId >= 0)
        {
            int nodeId = _worldScene.SelectedTaxiNodeId;
            return _worldScene.TaxiLoader.Routes
                .Where(route => route.FromNodeId == nodeId || route.ToNodeId == nodeId)
                .OrderBy(route => route.PathId)
                .ToList();
        }

        return Array.Empty<TaxiPathLoader.TaxiRoute>();
    }

    private bool TryGetTaxiActorOverrideRouteId(out int routeId)
    {
        routeId = -1;
        IReadOnlyList<TaxiPathLoader.TaxiRoute> candidateRoutes = GetTaxiActorOverrideCandidateRoutes();
        if (candidateRoutes.Count == 0)
        {
            _taxiActorModelOverrideTargetRouteId = -1;
            return false;
        }

        int preferredRouteId = _worldScene?.SelectedTaxiRouteId >= 0
            ? _worldScene.SelectedTaxiRouteId
            : _taxiActorModelOverrideTargetRouteId;

        TaxiPathLoader.TaxiRoute? activeRoute = candidateRoutes.FirstOrDefault(route => route.PathId == preferredRouteId)
            ?? candidateRoutes[0];

        _taxiActorModelOverrideTargetRouteId = activeRoute.PathId;
        routeId = activeRoute.PathId;
        return true;
    }

    private string GetTaxiRouteDisplayLabel(int pathId)
    {
        if (_worldScene == null)
            return $"Route #{pathId}";

        TaxiPathLoader.TaxiRoute? route = _worldScene.GetTaxiRoute(pathId);
        if (route == null)
            return $"Route #{pathId}";

        string fromName = _worldScene.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
        string toName = _worldScene.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
        return $"[{route.PathId}] {fromName} -> {toName}";
    }

    private void SyncTaxiActorModelOverrideInput(int routeId)
    {
        if (_worldScene == null || routeId < 0)
        {
            _taxiActorModelOverrideInputRouteId = -1;
            _taxiActorModelOverrideInput = "";
            return;
        }

        if (_taxiActorModelOverrideInputRouteId == routeId)
            return;

        _taxiActorModelOverrideInputRouteId = routeId;
        _taxiActorModelOverrideInput = _worldScene.GetTaxiActorModelOverride(routeId)
            ?? _worldScene.GetResolvedTaxiActorModelPath(routeId)
            ?? "";
    }

    private bool TryGetLoadedTaxiActorModelPath(out string modelPath)
    {
        modelPath = string.Empty;

        string? candidatePath = _lastVirtualPath;
        if (string.IsNullOrWhiteSpace(candidatePath) || !IsTaxiActorModelPath(candidatePath))
            return false;

        modelPath = candidatePath.Replace('/', '\\');
        return true;
    }

    private static bool IsTaxiActorModelPath(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".m2", StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryProjectWorldToViewport(Vector3 worldPosition, Matrix4x4 view, Matrix4x4 proj, float viewportWidth, float viewportHeight, out Vector2 projected)
    {
        Vector4 clip = Vector4.Transform(Vector4.Transform(new Vector4(worldPosition, 1f), view), proj);
        if (clip.W <= 0.0001f)
        {
            projected = Vector2.Zero;
            return false;
        }

        Vector3 ndc = new Vector3(clip.X, clip.Y, clip.Z) / clip.W;
        if (ndc.Z < -1f || ndc.Z > 1f)
        {
            projected = Vector2.Zero;
            return false;
        }

        projected = new Vector2(
            (ndc.X * 0.5f + 0.5f) * viewportWidth,
            (1f - (ndc.Y * 0.5f + 0.5f)) * viewportHeight);
        return true;
    }

    private void PickObjectAtMouse(float mouseX, float mouseY, bool addPm4ToCollection = false)
    {
        if (_worldScene == null) return;

        System.Diagnostics.Stopwatch clickSw = WoWViewer.Logging.Pm4Profiling.Enabled
            ? System.Diagnostics.Stopwatch.StartNew() : null;
        long clickPickStartTicks = 0;
        long clickSelectionStartTicks = 0;
        double clickPickMs = 0;
        double clickSelectionMs = 0;

        try
        {
            if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
                return;

            if (mouseX < vpX || mouseX > vpX + vpW || mouseY < vpY || mouseY > vpY + vpH)
                return;

            float aspect = vpW / Math.Max(vpH, 1f);
            var view = _camera.GetViewMatrix();
            float farPlane = GetSceneFarPlane();
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, farPlane);

            // Convert viewport-local mouse coords to NDC (-1..1)
            float localX = mouseX - vpX;
            float localY = mouseY - vpY;
            float ndcX = (localX / vpW) * 2f - 1f;
            float ndcY = 1f - (localY / vpH) * 2f; // flip Y

            var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
            var hoveredPm4Key = _worldScene.Pm4Overlay.ShowPm4Overlay ? _worldScene.HoveredAssetInfo?.Pm4ObjectKey : null;

            if (addPm4ToCollection)
            {
                // Only the Shift+LMB collection branch needs the ray PM4 pick;
                // the normal-click path picks PM4 inside TryHandleSceneClickSelection
                // (a duplicate outer pick here doubled the per-click cost on dense maps).
                if (clickSw != null) clickPickStartTicks = clickSw.ElapsedTicks;
                _worldScene.Pm4Overlay.TryPickPm4ObjectByRay(rayOrigin, rayDir, out var pm4HitKey, out var _, out _);
                if (clickSw != null) clickPickMs = (clickSw.ElapsedTicks - clickPickStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

                ClearPendingClickSelection();
                _worldScene.ClearTaxiSelection();
                _worldScene.ClearSelection();
                ClearSelectedAreaPoiInfo();

                var collectionPm4Key = hoveredPm4Key ?? pm4HitKey;
                if (collectionPm4Key.HasValue && _worldScene.Pm4Overlay.SelectPm4Object(collectionPm4Key.Value))
                {
                    TogglePm4ObjectCollectionMembership(collectionPm4Key.Value, reportStatus: true);
                    UpdateSelectedPm4ObjectInfo(collectionPm4Key);
                }
                else
                {
                    _statusMessage = "Shift+LMB PM4 add failed: no PM4 object was hit under the cursor. Use the PM4 graph Collect buttons when overlaps are dense.";
                }

                return;
            }

            if (clickSw != null) clickSelectionStartTicks = clickSw.ElapsedTicks;
            bool handledBySelection = TryHandleSceneClickSelection(mouseX, mouseY, localX, localY, vpW, vpH, view, proj, rayOrigin, rayDir);
            if (clickSw != null) clickSelectionMs = (clickSw.ElapsedTicks - clickSelectionStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            if (handledBySelection)
                return;

            ClearPendingClickSelection();
            ClearSelectedWlLiquidBody(clearListIsolation: true);
            _worldScene.ClearSelection();
            _worldScene.ClearTaxiSelection();
            _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
            ClearSelectedAreaPoiInfo();
            _selectedObjectIndex = -1;
            _selectedObjectType = "";
            _selectedObjectInfo = "";
        }
        finally
        {
            if (clickSw != null)
            {
                clickSw.Stop();
                double totalMs = clickSw.ElapsedMilliseconds;
                if (totalMs >= 50.0)
                {
                    ViewerLog.Info(ViewerLog.Category.Terrain,
                        $"[PM4-PROFILE] PickObjectAtMouse: total={totalMs:0.0}ms pick={clickPickMs:0.0}ms selection={clickSelectionMs:0.0}ms shift={addPm4ToCollection}");
                }
            }
        }
    }

    private void UpdateSelectedPm4ObjectInfo((int tileX, int tileY, uint ck24, int objectPart)? pm4ObjectKey)
    {
        if (_worldScene == null)
            return;

        _selectedObjectType = "PM4";

        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            string nearestRef = float.IsNaN(debugInfo.NearestPositionRefDistance)
                ? "n/a"
                : $"{debugInfo.NearestPositionRefDistance:F2}";

            // Identity first, raw slices last. MSUR._0x1C is the producing placement's Z read as a
            // float (see docs/wowdev-wiki/pm4-pd4-draft.md), so that height plus the tile is what
            // actually names this object. "CK24" is the top three bytes of that float and its
            // "type" byte is the float's exponent band; both stay below for cross-referencing older
            // reports, not as identity.
            float selectedPlacementZ = BitConverter.UInt32BitsToSingle(debugInfo.Ck24 << 8);

            _selectedObjectInfo =
                $"PM4 Object\n" +
                $"Identity: {(debugInfo.Ck24 == 0 ? "NO PLACEMENT HEIGHT - unattributed; population is mostly M2 doodad collision" : $"placement Z {selectedPlacementZ:F3}")} on tile ({debugInfo.TileX}, {debugInfo.TileY})\n" +
                $"Region: {debugInfo.MshdRegionId}\n" +
                $"Raw MSUR._0x1C slice: 0x{debugInfo.Ck24:X6}  (exponent band 0x{debugInfo.Ck24Type:X2}, NOT a type)\n" +
                $"Viewer Part: {debugInfo.ObjectPartId} - assigned during the current overlay build after viewer-side splitting; not a raw PM4 field\n" +
                $"MSLK Group: 0x{debugInfo.LinkGroupObjectId:X8}\n" +
                $"Linked MPRL refs: {debugInfo.LinkedPositionRefCount}\n" +
                $"Surfaces: {debugInfo.SurfaceCount}\n" +
                $"GroupKey: 0x{debugInfo.DominantGroupKey:X2}  AttrMask: 0x{debugInfo.DominantAttributeMask:X2}  MscnRef: {debugInfo.DominantMscnRefIndex}\n" +
                $"Planar: swap={debugInfo.SwapPlanarAxes} invertU={debugInfo.InvertU} invertV={debugInfo.InvertV} windingFlip={debugInfo.InvertsWinding}\n" +
                $"Center: ({debugInfo.Center.X:F1}, {debugInfo.Center.Y:F1}, {debugInfo.Center.Z:F1})\n" +
                $"Nearest MPRL: {nearestRef}\n" +
                $"Offset: ({_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Z:F2})";
            return;
        }

        if (!pm4ObjectKey.HasValue)
            return;

        var selectedPm4 = pm4ObjectKey.Value;
        _selectedObjectInfo =
            $"PM4 Object\n" +
            $"Tile: ({selectedPm4.tileX}, {selectedPm4.tileY})\n" +
            $"CK24: 0x{selectedPm4.ck24:X6} (viewerPart={selectedPm4.objectPart})\n" +
            $"Viewer Part: assigned during the current overlay build; not a raw PM4 field\n" +
            $"Offset: ({_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Z:F2})";
    }

    private void UpdateWorldSceneWireframeReveal(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_worldScene == null || !_worldScene.WireframeRevealEnabled)
            return;

        if (IsSceneMouseCaptureBlocked(_lastMouseX, _lastMouseY) || !TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
        {
            _worldScene.ClearWireframeReveal();
            return;
        }

        if (_lastMouseX < vpX || _lastMouseX > vpX + vpW || _lastMouseY < vpY || _lastMouseY > vpY + vpH)
        {
            _worldScene.ClearWireframeReveal();
            return;
        }

        float localX = _lastMouseX - vpX;
        float localY = _lastMouseY - vpY;
        _worldScene.UpdateWireframeReveal(view, proj, localX, localY, vpW, vpH);
    }

    private void UpdateWorldSceneHoveredAssetInfo(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_worldScene == null)
            return;

        if (IsSceneMouseCaptureBlocked(_lastMouseX, _lastMouseY) || !TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
        {
            _worldScene.ClearHoveredAssetInfo();
            return;
        }

        if (_lastMouseX < vpX || _lastMouseX > vpX + vpW || _lastMouseY < vpY || _lastMouseY > vpY + vpH)
        {
            _worldScene.ClearHoveredAssetInfo();
            return;
        }

        float localX = _lastMouseX - vpX;
        float localY = _lastMouseY - vpY;
        _worldScene.UpdateHoveredAssetInfo(view, proj, localX, localY, vpW, vpH);

        // Operator feedback 2026-09-06: the mouse picked objects many tiles away
        // THROUGH the ground. The hover picker tests object distance but never
        // terrain occlusion, so an object behind a hill was still hovered. If
        // terrain is hit first along the same ray, the hover is invalid.
        // WL bodies are source-data inspection targets, not scene objects. A composed layer can
        // legitimately put terrain in front of their original bounds, but that must not erase the
        // WL hover identity that the click inspector consumes. Keep terrain occlusion for actual
        // placed scene objects (WMO/M2/PM4) only.
        if (_worldScene.HoveredAssetInfo is { IsPreciseRayHit: true } hovered
            && !string.Equals(hovered.AssetKind, "WL liquid", StringComparison.OrdinalIgnoreCase))
        {
            float ndcX = (localX / MathF.Max(vpW, 1f)) * 2f - 1f;
            float ndcY = 1f - (localY / MathF.Max(vpH, 1f)) * 2f;
            var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
            TerrainRenderer? occlusionRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            if (occlusionRenderer != null
                && TryRaycastTerrain(occlusionRenderer, rayOrigin, rayDir, GetSceneFarPlane(), out _, out Vector3 terrainHit))
            {
                float terrainDistance = Vector3.Distance(rayOrigin, terrainHit);
                float objectDistance = Vector3.Distance(rayOrigin, hovered.WorldPosition);
                if (objectDistance > terrainDistance + 1f)
                    _worldScene.ClearHoveredAssetInfo();
            }
        }
    }

    /// <summary>
    /// World-space pass for the in-scene cluster selection rings. Runs with the rest of the 3D
    /// scene so the rings occlude correctly against terrain and objects.
    /// </summary>
    private void RenderSceneClusterSelector3D(Matrix4x4 proj)
    {
        if (_sceneClusterSelector3D != null && _sceneClusterSelector3D.IsActive)
        {
            _sceneClusterSelector3D.RenderWorld3D(_camera, proj);
        }
    }

    /// <summary>
    /// Overlay pass for the 3D scene cursor.
    /// </summary>
    /// <remarks>
    /// Called after <c>_imGui.Render()</c>, never with the 3D scene. ImGui draws in its own pass on
    /// top of whatever the 3D pass produced, so a cursor drawn during the 3D pass is painted over by
    /// every panel, menu, popup and hover card - and since the hardware cursor is hidden while this
    /// cursor is active, the pointer simply vanishes under the UI. Depth state cannot help; only
    /// draw order can. The caller is responsible for setting the scene viewport before this and
    /// restoring the full framebuffer viewport after.
    ///
    /// ImGui's Silk.NET controller restores the GL state it found, but this pass runs on whatever
    /// it left, so the state the cursor depends on is set explicitly below rather than assumed.
    /// </remarks>
    private void RenderSceneCursor(
        Matrix4x4 view,
        Matrix4x4 proj,
        float vpX,
        float vpY,
        float vpW,
        float vpH)
    {
        if (_sceneCursorRenderer == null || _sceneCursorRenderer.Style == CursorStyle.ClassicOSArrow)
            return;

        if (!CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
            return;

        float localX = _lastMouseX - vpX;
        float localY = _lastMouseY - vpY;
        float ndcX = (localX / vpW) * 2f - 1f;
        float ndcY = 1f - (localY / vpH) * 2f;

        var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);

        float? hitDistance = null;
        if (_worldScene?.HoveredAssetInfo is HoveredAssetInfo hoverInfo && hoverInfo.IsPreciseRayHit)
        {
            hitDistance = (hoverInfo.WorldPosition - rayOrigin).Length();
            _sceneCursorRenderer.State = (hoverInfo.AssetKind.Contains("NPC", StringComparison.OrdinalIgnoreCase)
                || hoverInfo.DisplayName.Contains("Creature", StringComparison.OrdinalIgnoreCase))
                ? SceneCursorState.Speak
                : SceneCursorState.Interact;
        }
        else
        {
            TerrainRenderer? terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            if (terrainRenderer != null && TryRaycastTerrain(terrainRenderer, rayOrigin, rayDir, GetSceneFarPlane(), out _, out Vector3 hitPoint))
            {
                hitDistance = (hitPoint - rayOrigin).Length();
            }

            _sceneCursorRenderer.State = _workspaceMode == WorkspaceMode.Editor
                && (_editorWorkspaceTask == EditorWorkspaceTask.Terrain || _editorWorkspaceTask == EditorWorkspaceTask.Objects)
                ? SceneCursorState.CastGlow
                : SceneCursorState.Pointer;
        }

        // ImGui leaves scissor test enabled and clipped to its last draw command; anything left
        // clipped here would silently discard the cursor. Blending must be on for the cursor's
        // alpha, and face culling off because the billboard can present either winding.
        _gl.Disable(EnableCap.ScissorTest);
        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        _gl.DepthMask(true);

        _sceneCursorRenderer.Render(_camera, proj, rayOrigin, rayDir, hitDistance, _fovDegrees, 0.1f);
    }

    /// <summary>
    /// Coloured text that is NOT run through printf formatting.
    /// </summary>
    /// <remarks>
    /// <c>ImGui.Text</c> and <c>ImGui.TextColored</c> treat their argument as a format string, so a
    /// '%' arriving from data - an asset path, a percentage in a detail line - is read as a
    /// conversion specifier and prints garbage pulled off the stack. A tooltip reading
    /// "95.135345743157f" where "95.1%" was written is exactly that. Any string that comes from data
    /// rather than from a literal must go through here.
    /// </remarks>
    private static void TextColoredUnformatted(Vector4 color, string text)
    {
        ImGui.PushStyleColor(ImGuiCol.Text, color);
        ImGui.TextUnformatted(text ?? string.Empty);
        ImGui.PopStyleColor();
    }

    private void DrawSceneHoverAssetOverlay()
    {
        if (_visualInvestigationMode == VisualInvestigationMode.Adt)
        {
            TryDrawTerrainChunkHoverOverlay();
            return;
        }

        if (_sceneCursorRenderer != null && _sceneCursorRenderer.Style != CursorStyle.ClassicOSArrow)
            return;

        if (_sceneClusterSelector3D != null && _sceneClusterSelector3D.IsActive)
            return;

        if (_worldScene != null && !_worldScene.ShowHoveredAssetTooltips)
            return;

        if (_worldScene?.HoveredAssetInfo is not HoveredAssetInfo info)
            return;

        if (!info.IsPreciseRayHit || !ShouldShowHoveredAssetInfoForInvestigation(info))
            return;

        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return;

        if (_lastMouseX < vpX || _lastMouseX > vpX + vpW || _lastMouseY < vpY || _lastMouseY > vpY + vpH)
            return;

        Vector2 displaySize = ImGui.GetIO().DisplaySize;
        Vector2 overlayPos = new(
            MathF.Min(_lastMouseX + 18f, MathF.Max(8f, displaySize.X - 390f)),
            MathF.Min(_lastMouseY + 18f, MathF.Max(8f, displaySize.Y - 290f)));

        ImGui.SetNextWindowPos(overlayPos, ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(16f, 13f));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 2f);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, 4f);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0.04f, 0.05f, 0.09f, 0.985f));
        ImGui.PushStyleColor(ImGuiCol.Border, new Vector4(0.95f, 0.79f, 0.28f, 0.98f));
        ImGui.PushStyleColor(ImGuiCol.Separator, new Vector4(0.88f, 0.73f, 0.22f, 0.82f));

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoDecoration
            | ImGuiWindowFlags.AlwaysAutoResize
            | ImGuiWindowFlags.NoDocking
            | ImGuiWindowFlags.NoSavedSettings
            | ImGuiWindowFlags.NoFocusOnAppearing
            | ImGuiWindowFlags.NoNav
            | ImGuiWindowFlags.NoMove
            | ImGuiWindowFlags.NoInputs;

        if (!ImGui.Begin("##SceneHoverAssetOverlay", flags))
        {
            ImGui.End();
            ImGui.PopStyleColor(3);
            ImGui.PopStyleVar(3);
            return;
        }

        ImGui.SetWindowFontScale(1.22f);
        TextColoredUnformatted(GetHoveredAssetTitleColor(info), info.DisplayName);
        ImGui.SetWindowFontScale(1.0f);
        TextColoredUnformatted(new Vector4(1.0f, 0.91f, 0.56f, 1.0f), info.AssetKind);

        if (!string.IsNullOrWhiteSpace(info.SourcePath))
        {
            ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 340f);
            TextColoredUnformatted(new Vector4(0.54f, 0.84f, 0.52f, 1.0f), info.SourcePath);
            ImGui.PopTextWrapPos();
        }

        if (!string.IsNullOrWhiteSpace(info.ParentSourcePath))
        {
            ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 340f);
            TextColoredUnformatted(new Vector4(0.62f, 0.72f, 0.86f, 1.0f), $"Parent WMO: {info.ParentSourcePath}");
            ImGui.PopTextWrapPos();
        }

        if (!string.IsNullOrWhiteSpace(info.DetailLine))
            TextColoredUnformatted(new Vector4(0.86f, 0.88f, 0.94f, 1.0f), info.DetailLine);

        ImGui.Separator();

        ImGui.TextColored(new Vector4(0.72f, 0.78f, 0.90f, 1.0f), $"World: ({info.WorldPosition.X:F1}, {info.WorldPosition.Y:F1}, {info.WorldPosition.Z:F1})");

        if (info.Pm4ObjectKey.HasValue && ShouldShowHoveredPm4MatchCandidates())
            DrawHoveredPm4MatchCandidates(info.Pm4ObjectKey.Value);

        ImGui.Separator();
        ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.38f, 1.0f), "Left-click to view in Inspector");

        if (info.AdditionalHitCount > 0)
        {
            string suffix = info.AdditionalHitCount == 1 ? string.Empty : "s";
            ImGui.TextColored(new Vector4(0.78f, 0.73f, 0.57f, 1.0f), $"+{info.AdditionalHitCount} more nearby asset hit{suffix}");
        }

        ImGui.SetWindowFontScale(1.0f);
        ImGui.End();
        ImGui.PopStyleColor(3);
        ImGui.PopStyleVar(3);
    }

    private static Vector4 GetHoveredAssetTitleColor(HoveredAssetInfo info)
    {
        return info.AssetKind switch
        {
            "PM4" => new Vector4(1.0f, 0.82f, 0.32f, 1.0f),
            "WMO" => new Vector4(0.78f, 0.92f, 1.0f, 1.0f),
            "WMO Doodad" => new Vector4(0.96f, 0.84f, 0.58f, 1.0f),
            "WL liquid" => new Vector4(0.60f, 0.88f, 1.0f, 1.0f),
            _ => new Vector4(0.92f, 0.96f, 1.0f, 1.0f)
        };
    }

    private bool TryGetHoveredPm4ObjectMatch((int tileX, int tileY, uint ck24, int objectPart) objectKey, out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        int maxMatches = Math.Max(3, Math.Min(5, _pm4ObjectMatchMaxMatchesPerObject));
        if (_hoveredPm4ObjectMatch != null
            && _hoveredPm4ObjectMatchKey.HasValue
            && _hoveredPm4ObjectMatchKey.Value == objectKey
            && _hoveredPm4ObjectMatchCacheMaxMatches == maxMatches)
        {
            objectMatch = _hoveredPm4ObjectMatch;
            return true;
        }

        if (_worldScene == null || !_worldScene.Pm4Overlay.TryBuildPm4ObjectMatch(objectKey, maxMatches, out Pm4ObjectMatchObject hoveredMatch))
            return false;

        _hoveredPm4ObjectMatch = hoveredMatch;
        _hoveredPm4ObjectMatchKey = objectKey;
        _hoveredPm4ObjectMatchCacheMaxMatches = maxMatches;
        objectMatch = hoveredMatch;
        return true;
    }

    private void DrawHoveredPm4MatchCandidates((int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        ImGui.Separator();
        ImGui.TextColored(new Vector4(1.0f, 0.90f, 0.52f, 1.0f), "Likely matches");

        if (!TryGetHoveredPm4ObjectMatch(objectKey, out Pm4ObjectMatchObject objectMatch))
        {
            ImGui.TextColored(new Vector4(0.74f, 0.78f, 0.86f, 1.0f), "No PM4 match preview available for this hovered part.");
            return;
        }

        if (objectMatch.Candidates.Count == 0)
        {
            ImGui.TextColored(new Vector4(0.74f, 0.78f, 0.86f, 1.0f), "No nearby WMO or M2 placement candidates were found.");
            return;
        }

        int shownCount = Math.Min(3, objectMatch.Candidates.Count);
        for (int i = 0; i < shownCount; i++)
        {
            Pm4ObjectMatchCandidate candidate = objectMatch.Candidates[i];
            ImGui.PushID($"HoverPm4Candidate_{i}");
            ImGui.TextColored(new Vector4(0.80f, 0.96f, 0.82f, 1.0f), $"{i + 1}. {candidate.Kind}  gap={candidate.PlanarGap:F1}");

            if (!string.IsNullOrWhiteSpace(candidate.ModelName))
                ImGui.TextColored(new Vector4(0.92f, 0.94f, 0.98f, 1.0f), candidate.ModelName);

            ImGui.TextColored(
                new Vector4(0.72f, 0.78f, 0.90f, 1.0f),
                $"{candidate.EvidenceSource}  vertical={candidate.VerticalGap:F1}  overlap={candidate.PlanarOverlapRatio:P0}");
            ImGui.PopID();
        }

        if (objectMatch.Candidates.Count > shownCount)
            ImGui.TextColored(new Vector4(0.78f, 0.73f, 0.57f, 1.0f), $"+{objectMatch.Candidates.Count - shownCount} more in the inspector");
    }

    private bool IsPointInSceneViewport(float x, float y)
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            if (!IsShellPanelActive(panel.Id))
                continue;

            if (IsPointInVisibleShellPanel(GetDockPanelStateRef(panel.Id), x, y))
                return false;
        }

        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return false;
        return x >= vpX && x <= vpX + vpW && y >= vpY && y <= vpY + vpH;
    }

    private bool CanSceneConsumeMouse(float x, float y)
    {
        return IsPointInSceneViewport(x, y) && !IsSceneMouseCaptureBlocked(x, y);
    }

    private bool IsSceneMouseCaptureBlocked(float x, float y)
    {
        if (!ImGui.GetIO().WantCaptureMouse)
            return false;

        return !ShouldBypassDockspaceMouseCapture(x, y);
    }

    private bool ShouldBypassDockspaceMouseCapture(float x, float y)
    {
        return _useDockspaceUi
            && _dockspaceHostSize.X > 10f
            && _dockspaceHostSize.Y > 10f
            && IsPointInSceneViewport(x, y);
    }

    private static bool IsPointInVisibleShellPanel(in DockPanelState state, float x, float y)
    {
        if (!state.Visible || state.Size.X <= 1f || state.Size.Y <= 1f)
            return false;

        return x >= state.Position.X
            && x <= state.Position.X + state.Size.X
            && y >= state.Position.Y
            && y <= state.Position.Y + state.Size.Y;
    }

    private void QueueImGuiMouseButtonEvent(MouseButton button, bool down)
    {
        int? buttonIndex = button switch
        {
            MouseButton.Left => 0,
            MouseButton.Right => 1,
            MouseButton.Middle => 2,
            _ => null,
        };

        if (!buttonIndex.HasValue)
            return;

        lock (_pendingImGuiMouseEventLock)
        {
            _pendingImGuiMouseButtonEvents.Enqueue((buttonIndex.Value, down));
        }
    }

    private void FlushPendingImGuiMouseButtonEvents()
    {
        lock (_pendingImGuiMouseEventLock)
        {
            if (_pendingImGuiMouseButtonEvents.Count == 0)
                return;

            var io = ImGui.GetIO();
            while (_pendingImGuiMouseButtonEvents.Count > 0)
            {
                var (buttonIndex, down) = _pendingImGuiMouseButtonEvents.Dequeue();
                io.AddMouseButtonEvent(buttonIndex, down);
            }
        }
    }

    private static ShellPanelDefinition GetShellPanelDefinition(ShellPanelId panelId)
    {
        return ShellPanelDefinitions[(int)panelId];
    }

    private ref DockPanelState GetDockPanelStateRef(ShellPanelId panelId)
    {
        switch (panelId)
        {
            case ShellPanelId.Navigator:
                return ref _navigatorDockState;
            case ShellPanelId.Inspector:
                return ref _inspectorDockState;
            case ShellPanelId.Pm4Workbench:
                return ref _pm4WorkbenchDockState;
            case ShellPanelId.TerrainControls:
                return ref _terrainControlsDockState;
            case ShellPanelId.RuntimeStats:
                return ref _runtimeStatsDockState;
            case ShellPanelId.WorldObjects:
                return ref _worldObjectsDockState;
            case ShellPanelId.ModelInfo:
                return ref _modelInfoDockState;
            case ShellPanelId.Minimap:
                return ref _minimapDockState;
            case ShellPanelId.WorkspaceBars:
                return ref _workspaceBarsDockState;
            case ShellPanelId.Pm4Info:
                return ref _pm4InfoDockState;
            case ShellPanelId.Pm4SceneGraph:
                return ref _pm4SceneGraphDockState;
            default:
                throw new ArgumentOutOfRangeException(nameof(panelId), panelId, null);
        }
    }

    private bool IsShellPanelRequested(ShellPanelId panelId)
    {
        return panelId switch
        {
            ShellPanelId.Navigator => _showLeftSidebar,
            ShellPanelId.Inspector => _showRightSidebar,
            ShellPanelId.Pm4Workbench => _showRightSidebar && _worldScene != null,
            ShellPanelId.TerrainControls => _showRightSidebar && _showTerrainControls && (_terrainManager != null || _vlmTerrainManager != null),
            ShellPanelId.RuntimeStats => _showRightSidebar && (_terrainManager != null || _vlmTerrainManager != null || _worldScene != null),
            ShellPanelId.WorldObjects => _showRightSidebar && _worldScene != null,
            ShellPanelId.ModelInfo => _showRightSidebar && _showModelInfo && !string.IsNullOrWhiteSpace(_modelInfo),
            ShellPanelId.Minimap => _showMinimapWindow,
            ShellPanelId.WorkspaceBars => false,
            ShellPanelId.Pm4Info => _showRightSidebar && _worldScene != null,
            ShellPanelId.Pm4SceneGraph => _showPm4SceneGraph && _worldScene != null,
            _ => false,
        };
    }

    private bool IsShellPanelSuppressedForLayout(ShellPanelId panelId)
    {
        return panelId switch
        {
            ShellPanelId.Navigator => _suppressLeftSidebarForLayout,
            ShellPanelId.Inspector => _suppressRightSidebarForLayout,
            ShellPanelId.Minimap => _suppressMinimapForLayout,
            ShellPanelId.WorkspaceBars => _suppressLeftSidebarForLayout,
            ShellPanelId.Pm4Info => _suppressRightSidebarForLayout,
            ShellPanelId.Pm4SceneGraph => _suppressRightSidebarForLayout,
            _ => false,
        };
    }

    private bool IsShellPanelActive(ShellPanelId panelId)
    {
        return IsShellPanelRequested(panelId) && !IsShellPanelSuppressedForLayout(panelId);
    }

    private bool HasAnyShellPanelsInLane(ShellPanelLane lane)
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            if (panel.Lane == lane && IsShellPanelActive(panel.Id))
                return true;
        }

        return false;
    }

    private void FocusShellPanel(ShellPanelId panelId)
    {
        if (!_useDockspaceUi)
        {
            switch (panelId)
            {
                case ShellPanelId.Navigator:
                    _showLeftSidebar = true;
                    return;
                case ShellPanelId.Inspector:
                    _showRightSidebar = true;
                    return;
                case ShellPanelId.WorkspaceBars:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Workspace;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Workspace;
                    return;
                case ShellPanelId.Pm4Workbench:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Pm4;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Pm4;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Pm4Evidence);
                    return;
                case ShellPanelId.TerrainControls:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Terrain;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Terrain;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Terrain);
                    return;
                case ShellPanelId.WorldObjects:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.World;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.World;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Objects);
                    return;
                case ShellPanelId.RuntimeStats:
                case ShellPanelId.ModelInfo:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Diagnostics;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Diagnostics;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Inspect);
                    return;
                case ShellPanelId.Minimap:
                    _showMinimapWindow = true;
                    return;
                case ShellPanelId.Pm4Info:
                    _showRightSidebar = true;
                    return;
                case ShellPanelId.Pm4SceneGraph:
                    _showPm4SceneGraph = true;
                    return;
            }
        }

        if (panelId == ShellPanelId.WorkspaceBars)
        {
            _showWorkspaceBarsPanel = true;
            _pendingFocusedShellPanel = panelId;
            return;
        }

        switch (GetShellPanelDefinition(panelId).Lane)
        {
            case ShellPanelLane.Left:
                _showLeftSidebar = true;
                break;
            case ShellPanelLane.Right:
                _showRightSidebar = true;
                break;
            case ShellPanelLane.Floating:
                if (panelId == ShellPanelId.Minimap)
                    _showMinimapWindow = true;
                break;
        }

        _pendingFocusedShellPanel = panelId;
    }

    private void ResetDockPanelStates()
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            ref DockPanelState state = ref GetDockPanelStateRef(panel.Id);
            state = default;
        }
    }

    private void ResetShellLayoutToDefaults()
    {
        _savedShellPanelLayouts.Clear();
        _pendingShellPanelLayoutRestore.Clear();
        _showLeftSidebar = true;
        _showRightSidebar = true;
        _showTerrainControls = false;
        _leftSidebarWidth = DefaultSidebarWidth;
        _rightSidebarWidth = DefaultRightSidebarWidth;
        _bottomDrawerHeight = DefaultBottomDrawerHeight;
        _activeBottomDrawerTab = FixedBottomDrawerTab.Workspace;
        _useDockspaceUi = true;
        _showPm4SceneGraph = true;
        _forceApplyShellPanelLayout = true;
        SaveViewerSettings();
    }

    private void CaptureDockPanelState(ShellPanelId panelId)
    {
        ref DockPanelState state = ref GetDockPanelStateRef(panelId);
        state.Visible = true;
        state.IsDocked = ImGui.IsWindowDocked();
        state.Position = ImGui.GetWindowPos();
        state.Size = ImGui.GetWindowSize();

        CaptureSavedShellPanelLayout(panelId, state);
    }

    private void CaptureSavedShellPanelLayout(ShellPanelId panelId, in DockPanelState state)
    {
        if (!_useDockspaceUi || !state.Visible || state.Size.X <= 1f || state.Size.Y <= 1f)
            return;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 size))
            return;

        float normalizedWidth = Math.Clamp(state.Size.X / Math.Max(size.X, 1f), 0.12f, 1f);
        float normalizedHeight = Math.Clamp(state.Size.Y / Math.Max(size.Y, 1f), 0.12f, 1f);
        float normalizedX = Math.Clamp((state.Position.X - origin.X) / Math.Max(size.X, 1f), 0f, 1f - normalizedWidth);
        float normalizedY = Math.Clamp((state.Position.Y - origin.Y) / Math.Max(size.Y, 1f), 0f, 1f - normalizedHeight);

        _savedShellPanelLayouts[panelId] = new SavedShellPanelLayout
        {
            PanelId = (int)panelId,
            NormalizedX = normalizedX,
            NormalizedY = normalizedY,
            NormalizedWidth = normalizedWidth,
            NormalizedHeight = normalizedHeight,
        };
    }

    private void PrepareDockableShellPanelWindow(ShellPanelId panelId, Vector2 defaultSize, Vector2 minSize, Vector2 maxSize)
    {
        if (!_useDockspaceUi)
        {
            ImGui.SetNextWindowSize(defaultSize, ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowSizeConstraints(minSize, maxSize);
            return;
        }

        bool shouldForceLayout = _forceApplyShellPanelLayout || _pendingShellPanelLayoutRestore.Contains(panelId);
        if (TryResolveShellPanelRect(panelId, minSize, maxSize, out Vector2 position, out Vector2 size))
        {
            ImGuiCond cond = shouldForceLayout ? ImGuiCond.Always : ImGuiCond.Appearing;
            ImGui.SetNextWindowPos(position, cond);
            ImGui.SetNextWindowSize(size, cond);

            if (shouldForceLayout)
                _pendingShellPanelLayoutRestore.Remove(panelId);
        }
        else
        {
            ImGui.SetNextWindowSize(defaultSize, ImGuiCond.FirstUseEver);
        }

        ImGui.SetNextWindowSizeConstraints(minSize, maxSize);
    }

    private bool TryResolveShellPanelRect(ShellPanelId panelId, Vector2 minSize, Vector2 maxSize, out Vector2 position, out Vector2 size)
    {
        if (TryGetSavedShellPanelRect(panelId, minSize, maxSize, out position, out size))
            return true;

        return TryGetDefaultShellPanelRect(panelId, minSize, maxSize, out position, out size);
    }

    private bool TryGetSavedShellPanelRect(ShellPanelId panelId, Vector2 minSize, Vector2 maxSize, out Vector2 position, out Vector2 size)
    {
        position = Vector2.Zero;
        size = Vector2.Zero;

        if (!_savedShellPanelLayouts.TryGetValue(panelId, out SavedShellPanelLayout? savedLayout))
            return false;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 hostSize))
            return false;

        size = new Vector2(
            hostSize.X * savedLayout.NormalizedWidth,
            hostSize.Y * savedLayout.NormalizedHeight);
        position = new Vector2(
            origin.X + hostSize.X * savedLayout.NormalizedX,
            origin.Y + hostSize.Y * savedLayout.NormalizedY);

        ClampShellPanelRect(origin, hostSize, minSize, maxSize, ref position, ref size);
        return true;
    }

    private bool TryGetDefaultShellPanelRect(ShellPanelId panelId, Vector2 minSize, Vector2 maxSize, out Vector2 position, out Vector2 size)
    {
        position = Vector2.Zero;
        size = Vector2.Zero;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 hostSize))
            return false;

        ShellPanelId[] group = GetDefaultShellPanelGroup(panelId);
        int activeCount = 0;
        int panelIndex = -1;
        for (int i = 0; i < group.Length; i++)
        {
            if (!IsShellPanelActive(group[i]))
                continue;

            if (group[i] == panelId)
                panelIndex = activeCount;

            activeCount++;
        }

        if (activeCount == 0 || panelIndex < 0)
            return false;

        const float padding = 12f;
        const float gap = 10f;
        float columnWidth = Math.Clamp(hostSize.X * 0.26f, 280f, 420f);
        float quadrantHeight = Math.Max(220f, (hostSize.Y - padding * 2f - gap) * 0.5f);
        float leftX = origin.X + padding;
        float rightX = Math.Max(leftX + gap, origin.X + hostSize.X - columnWidth - padding);
        float topY = origin.Y + padding;
        float bottomY = origin.Y + hostSize.Y - quadrantHeight - padding;

        bool isLeftQuadrant = panelId == ShellPanelId.Navigator
            || panelId == ShellPanelId.Inspector
            || panelId == ShellPanelId.Pm4Workbench
            || panelId == ShellPanelId.Minimap;
        bool isTopQuadrant = panelId == ShellPanelId.Navigator
            || panelId == ShellPanelId.Inspector
            || panelId == ShellPanelId.RuntimeStats
            || panelId == ShellPanelId.ModelInfo;

        float groupX = isLeftQuadrant ? leftX : rightX;
        float groupY = isTopQuadrant ? topY : bottomY;
        float slotHeight = (quadrantHeight - gap * Math.Max(0, activeCount - 1)) / activeCount;
        position = new Vector2(groupX, groupY + panelIndex * (slotHeight + gap));
        size = new Vector2(columnWidth, slotHeight);

        if (panelId == ShellPanelId.Minimap)
        {
            float squareSize = MathF.Min(size.X, size.Y);
            size = new Vector2(squareSize, squareSize);
        }

        ClampShellPanelRect(origin, hostSize, minSize, maxSize, ref position, ref size);
        return true;
    }

    private static ShellPanelId[] GetDefaultShellPanelGroup(ShellPanelId panelId)
    {
        return panelId switch
        {
            ShellPanelId.Navigator => TopLeftQuadrantPanels,
            ShellPanelId.Inspector or ShellPanelId.WorldObjects or ShellPanelId.ModelInfo or ShellPanelId.RuntimeStats => TopRightQuadrantPanels,
            ShellPanelId.Pm4Workbench or ShellPanelId.Pm4Info or ShellPanelId.TerrainControls => BottomRightQuadrantPanels,
            ShellPanelId.Minimap => BottomLeftQuadrantPanels,
            _ => TopRightQuadrantPanels,
        };
    }

    private bool TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 size)
    {
        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float height = io.DisplaySize.Y - topOffset - StatusBarHeight;

        if (_useDockspaceUi && _dockspaceHostSize.X > 10f && _dockspaceHostSize.Y > 10f)
        {
            origin = _dockspaceHostPosition;
            size = _dockspaceHostSize;
            return true;
        }

        origin = new Vector2(0f, topOffset);
        size = new Vector2(io.DisplaySize.X, MathF.Max(0f, height));
        return size.X > 10f && size.Y > 10f;
    }

    private static void ClampShellPanelRect(Vector2 origin, Vector2 hostSize, Vector2 minSize, Vector2 maxSize, ref Vector2 position, ref Vector2 size)
    {
        float clampedWidth = Math.Clamp(size.X, minSize.X, Math.Min(maxSize.X, hostSize.X));
        float clampedHeight = Math.Clamp(size.Y, minSize.Y, Math.Min(maxSize.Y, hostSize.Y));
        size = new Vector2(clampedWidth, clampedHeight);

        float maxX = Math.Max(origin.X, origin.X + hostSize.X - size.X);
        float maxY = Math.Max(origin.Y, origin.Y + hostSize.Y - size.Y);
        position = new Vector2(
            Math.Clamp(position.X, origin.X, maxX),
            Math.Clamp(position.Y, origin.Y, maxY));
    }

    private bool TryGetDockedShellPanelState(ShellPanelLane lane, out DockPanelState state)
    {
        bool found = false;
        state = default;

        foreach (var panel in ShellPanelDefinitions)
        {
            if (panel.Lane != lane || !IsShellPanelActive(panel.Id))
                continue;

            ref DockPanelState panelState = ref GetDockPanelStateRef(panel.Id);
            if (!panelState.Visible || !panelState.IsDocked)
                continue;

            if (!found)
            {
                state = panelState;
                found = true;
                continue;
            }

            float left = MathF.Min(state.Position.X, panelState.Position.X);
            float top = MathF.Min(state.Position.Y, panelState.Position.Y);
            float right = MathF.Max(state.Position.X + state.Size.X, panelState.Position.X + panelState.Size.X);
            float bottom = MathF.Max(state.Position.Y + state.Size.Y, panelState.Position.Y + panelState.Size.Y);

            state.Visible = true;
            state.IsDocked = true;
            state.Position = new Vector2(left, top);
            state.Size = new Vector2(right - left, bottom - top);
        }

        if (found)
            return true;

        return false;
    }

    private bool TryGetVisibleShellPanelInsetState(bool isLeftPanel, out DockPanelState state)
    {
        state = default;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 hostSize))
            return false;

        bool found = false;
        float hostLeft = origin.X;
        float hostRight = origin.X + hostSize.X;
        const float edgeTolerance = 24f;

        foreach (var panel in ShellPanelDefinitions)
        {
            if (!IsShellPanelActive(panel.Id))
                continue;

            ref DockPanelState panelState = ref GetDockPanelStateRef(panel.Id);
            if (!panelState.Visible || panelState.Size.X <= 1f || panelState.Size.Y <= 1f)
                continue;

            bool touchesEdge = isLeftPanel
                ? panelState.Position.X <= hostLeft + edgeTolerance
                : panelState.Position.X + panelState.Size.X >= hostRight - edgeTolerance;
            if (!touchesEdge)
                continue;

            if (!found)
            {
                state = panelState;
                found = true;
                continue;
            }

            float left = MathF.Min(state.Position.X, panelState.Position.X);
            float top = MathF.Min(state.Position.Y, panelState.Position.Y);
            float right = MathF.Max(state.Position.X + state.Size.X, panelState.Position.X + panelState.Size.X);
            float bottom = MathF.Max(state.Position.Y + state.Size.Y, panelState.Position.Y + panelState.Size.Y);

            state.Visible = true;
            state.IsDocked = state.IsDocked || panelState.IsDocked;
            state.Position = new Vector2(left, top);
            state.Size = new Vector2(right - left, bottom - top);
        }

        return found;
    }

    private void UpdateShellLayout(Vector2 displaySize)
    {
        _suppressLeftSidebarForLayout = false;
        _suppressRightSidebarForLayout = false;
        _suppressMinimapForLayout = false;
        if (_hideUiChrome || displaySize.X <= 0f)
            return;

        float maxSidebarWidthBudget = MathF.Max(0f, displaySize.X - SceneViewportHardMinWidth);
        float requiredCompactWidth = (_showLeftSidebar ? SidebarCompactMinWidth : 0f)
            + (_showRightSidebar ? SidebarCompactMinWidth : 0f);

        if (requiredCompactWidth > maxSidebarWidthBudget && _showRightSidebar)
            _suppressRightSidebarForLayout = true;

        requiredCompactWidth = (_showLeftSidebar ? SidebarCompactMinWidth : 0f)
            + (IsShellPanelActive(ShellPanelId.Inspector) ? SidebarCompactMinWidth : 0f);

        if (requiredCompactWidth > maxSidebarWidthBudget && _showLeftSidebar)
            _suppressLeftSidebarForLayout = true;

        ClampFixedSidebarLayout(displaySize.X);

        if (_showMinimapWindow && !_fullscreenMinimap && _useDockspaceUi)
        {
            float requiredMinimapWidth = GetShellPanelDefinition(ShellPanelId.Minimap).CompactMinWidth;
            _suppressMinimapForLayout = displaySize.X < SceneViewportHardMinWidth + requiredMinimapWidth;
        }
    }

    private float ClampFixedBottomDrawerHeight(float height, float displayHeight)
    {
        GetFixedBottomDrawerHeightRange(displayHeight, out float minHeight, out float maxHeight);
        return Math.Clamp(height, minHeight, maxHeight);
    }

    private void GetFixedBottomDrawerHeightRange(float displayHeight, out float minHeight, out float maxHeight)
    {
        float availableHeight = MathF.Max(0f, displayHeight - GetTopChromeHeight() - StatusBarHeight);
        float preferredMaxHeight = availableHeight - SceneViewportPreferredMinHeight;
        float hardMaxHeight = availableHeight - SceneViewportHardMinHeight;
        maxHeight = MathF.Min(BottomDrawerMaxHeight, MathF.Max(BottomDrawerCompactMinHeight, MathF.Max(preferredMaxHeight, hardMaxHeight)));
        minHeight = MathF.Min(BottomDrawerMinHeight, maxHeight);
    }

    private void ClampFixedSidebarLayout(float displayWidth)
    {
        if (displayWidth <= 0f)
            return;

        if (IsShellPanelActive(ShellPanelId.Navigator))
            _leftSidebarWidth = Math.Clamp(_leftSidebarWidth, SidebarCompactMinWidth, SidebarMaxWidth);

        if (IsShellPanelActive(ShellPanelId.Inspector))
            _rightSidebarWidth = Math.Clamp(_rightSidebarWidth, SidebarCompactMinWidth, SidebarMaxWidth);

        if (IsShellPanelActive(ShellPanelId.Navigator))
            _leftSidebarWidth = ClampFixedSidebarWidth(_leftSidebarWidth, isLeftSidebar: true, displayWidth);

        if (IsShellPanelActive(ShellPanelId.Inspector))
            _rightSidebarWidth = ClampFixedSidebarWidth(_rightSidebarWidth, isLeftSidebar: false, displayWidth);
    }

    private static void ApplyDockedSidePanelInset(in DockPanelState state, bool isLeftPanel, float viewportY, float viewportHeight, ref float x, ref float width)
    {
        if (!state.Visible || !state.IsDocked || state.Size.X <= 1f || state.Size.Y <= 1f)
            return;

        float panelTop = state.Position.Y;
        float panelBottom = state.Position.Y + state.Size.Y;
        float viewportBottom = viewportY + viewportHeight;
        if (panelBottom <= viewportY || panelTop >= viewportBottom)
            return;

        const float edgeTolerance = 4f;
        if (isLeftPanel)
        {
            if (state.Position.X > x + edgeTolerance)
                return;

            x += state.Size.X;
            width -= state.Size.X;
            return;
        }

        float viewportRight = x + width;
        if (state.Position.X + state.Size.X < viewportRight - edgeTolerance)
            return;

        width -= state.Size.X;
    }

    private bool TryGetSceneViewportRect(out float x, out float y, out float width, out float height)
    {
        var io = ImGui.GetIO();

        if (_hideUiChrome)
        {
            x = 0f;
            y = 0f;
            width = io.DisplaySize.X;
            height = io.DisplaySize.Y;
            return width > 10f && height > 10f;
        }

        float topOffset = GetTopChromeHeight();
        x = 0f;
        y = topOffset;
        width = io.DisplaySize.X;
        height = io.DisplaySize.Y - topOffset - BottomBarHeight - StatusBarHeight;

        // 071: tab system uses fixed left/right sidebars; viewport is the
        // middle area between them. Sidebars auto-hide when the window is
        // too small (see UpdateShellLayout suppression logic).
        if (_useTabUi)
        {
            if (_showLeftSidebar)
            {
                x += _leftSidebarWidth;
                width -= _leftSidebarWidth;
            }

            if (_showRightSidebar)
                width -= _rightSidebarWidth;

            width = MathF.Max(width, 0f);
            height = MathF.Max(height, 0f);
            return width > 10f && height > 10f;
        }

        if (_useDockspaceUi && _dockspaceHostSize.X > 10f && _dockspaceHostSize.Y > 10f)
        {
            x = _dockspaceHostPosition.X;
            y = _dockspaceHostPosition.Y;
            width = _dockspaceHostSize.X;
            height = _dockspaceHostSize.Y;

            if (TryGetVisibleShellPanelInsetState(isLeftPanel: true, out DockPanelState leftDockPanel))
                ApplyDockedSidePanelInset(leftDockPanel, isLeftPanel: true, y, height, ref x, ref width);

            if (TryGetVisibleShellPanelInsetState(isLeftPanel: false, out DockPanelState rightDockPanel))
                ApplyDockedSidePanelInset(rightDockPanel, isLeftPanel: false, y, height, ref x, ref width);
        }
        else
        {
            if (IsShellPanelActive(ShellPanelId.Navigator))
            {
                x += _leftSidebarWidth;
                width -= _leftSidebarWidth;
            }

            if (IsShellPanelActive(ShellPanelId.Inspector))
                width -= _rightSidebarWidth;

        }

        width = MathF.Max(width, 0f);
        height = MathF.Max(height, 0f);
        return width > 10f && height > 10f;
    }

    private bool TryGetSceneFramebufferViewport(out int x, out int y, out uint width, out uint height)
    {
        x = y = 0;
        width = height = 0;

        if (!TryGetSceneViewportRect(out float viewportX, out float viewportY, out float viewportWidth, out float viewportHeight))
            return false;

        Vector2D<int> windowSize = _window.Size;
        Vector2D<int> framebufferSize = _window.FramebufferSize;
        if (windowSize.X <= 0 || windowSize.Y <= 0 || framebufferSize.X <= 0 || framebufferSize.Y <= 0)
            return false;

        float scaleX = (float)framebufferSize.X / windowSize.X;
        float scaleY = (float)framebufferSize.Y / windowSize.Y;

        int viewportLeft = (int)MathF.Round(viewportX * scaleX);
        int viewportTop = (int)MathF.Round(viewportY * scaleY);
        int viewportRight = (int)MathF.Round((viewportX + viewportWidth) * scaleX);
        int viewportBottom = (int)MathF.Round((viewportY + viewportHeight) * scaleY);

        viewportLeft = Math.Clamp(viewportLeft, 0, framebufferSize.X);
        viewportRight = Math.Clamp(viewportRight, viewportLeft, framebufferSize.X);
        viewportTop = Math.Clamp(viewportTop, 0, framebufferSize.Y);
        viewportBottom = Math.Clamp(viewportBottom, viewportTop, framebufferSize.Y);

        x = viewportLeft;
        y = framebufferSize.Y - viewportBottom;
        width = (uint)Math.Max(1, viewportRight - viewportLeft);
        height = (uint)Math.Max(1, viewportBottom - viewportTop);
        return true;
    }

    private static bool TryProjectToScreen(Vector3 worldPos, Matrix4x4 viewProj, int screenW, int screenH, out float sx, out float sy)
    {
        var clip = Vector4.Transform(new Vector4(worldPos, 1f), viewProj);
        if (clip.W <= 0) { sx = sy = 0; return false; }
        float ndcX = clip.X / clip.W;
        float ndcY = clip.Y / clip.W;
        sx = (ndcX * 0.5f + 0.5f) * screenW;
        sy = (1f - (ndcY * 0.5f + 0.5f)) * screenH;
        return true;
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
        SyncImGuiWindowMetrics(size, _window.FramebufferSize);
    }

    private void OnResize(Vector2D<int> size)
    {
        _gl.Viewport(size);
        SyncImGuiWindowMetrics(_window.Size, size);
    }

    private void SyncImGuiWindowMetrics(Vector2D<int> windowSize, Vector2D<int> framebufferSize)
    {
        if (_imGui == null || !HasImGuiContext())
            return;

        if (windowSize.X <= 0 || windowSize.Y <= 0 || framebufferSize.X <= 0 || framebufferSize.Y <= 0)
            return;

        bool windowSizeChanged = !windowSize.Equals(_lastSyncedImGuiWindowSize);
        bool framebufferSizeChanged = !framebufferSize.Equals(_lastSyncedImGuiFramebufferSize);
        if (!windowSizeChanged && !framebufferSizeChanged)
            return;

        if (windowSizeChanged)
            ImGuiControllerWindowResizedMethod?.Invoke(_imGui, new object[] { windowSize });

        ImGuiIOPtr io = ImGui.GetIO();
        io.DisplaySize = new Vector2(windowSize.X, windowSize.Y);
        io.DisplayFramebufferScale = new Vector2(
            windowSize.X > 0 ? (float)framebufferSize.X / windowSize.X : 1f,
            windowSize.Y > 0 ? (float)framebufferSize.Y / windowSize.Y : 1f);

        _lastSyncedImGuiWindowSize = windowSize;
        _lastSyncedImGuiFramebufferSize = framebufferSize;
    }

    private static bool HasImGuiContext()
        => ImGui.GetCurrentContext() != IntPtr.Zero;

    private void LoadViewerSettings()
    {
        try
        {
            RefreshClientBuildOptions();

            if (!File.Exists(ViewerSettingsPath))
            {
                _hasExplicitWmoMliqRotationOverride = false;
                WmoRenderer.MliqRotationQuarterTurns = 0;
                RefreshDatasetCatalog();
                return;
            }

            string json = File.ReadAllText(ViewerSettingsPath);
            var settings = JsonSerializer.Deserialize<ViewerSettings>(json);
            if (settings == null)
                return;

            _uiTheme = Enum.IsDefined(typeof(UiThemeKind), settings.UiTheme)
                ? (UiThemeKind)settings.UiTheme
                : UiThemeKind.ModernSlate;

            int savedWmoMliqRotation = ((settings.WmoMliqRotationQuarterTurns % 4) + 4) % 4;
            if (settings.HasExplicitWmoMliqRotationOverride)
            {
                _hasExplicitWmoMliqRotationOverride = true;
                WmoRenderer.MliqRotationQuarterTurns = savedWmoMliqRotation;
            }
            else if (savedWmoMliqRotation == 3)
            {
                _hasExplicitWmoMliqRotationOverride = false;
                WmoRenderer.MliqRotationQuarterTurns = 0;
                ViewerLog.Important(ViewerLog.Category.Wmo,
                    "[ViewerSettings] Migrated legacy WMO MLIQ 270° default to neutral override; WMO liquid rotation is now resolved from the asset version path.");
            }
            else
            {
                _hasExplicitWmoMliqRotationOverride = savedWmoMliqRotation != 0;
                WmoRenderer.MliqRotationQuarterTurns = savedWmoMliqRotation;
            }

            _lastGameFolderPath = settings.LastGameFolderPath ?? "";
            _lastLooseOverlayPath = settings.LastLooseOverlayPath ?? "";
            _datasetCatalogRoot = string.IsNullOrWhiteSpace(settings.LastDatasetCatalogRoot)
                ? _datasetCatalogRoot
                : settings.LastDatasetCatalogRoot;
            _selectedDatasetVersionRoot = settings.LastDatasetVersionRoot ?? string.Empty;
            _activeDatasetVersionRoot = settings.LastActiveDatasetVersionRoot ?? string.Empty;
            RefreshDatasetCatalog();
            _knownGoodClientPaths = NormalizeKnownGoodClientPaths(settings.KnownGoodClientPaths);
            _selectedBuildOptionIndex = FindBuildOptionIndex(settings.LastSelectedBuildVersion);
            _textureFilteringMode = Enum.IsDefined(typeof(TextureFilteringMode), settings.TextureFilteringMode)
                ? (TextureFilteringMode)settings.TextureFilteringMode
                : TextureFilteringMode.Trilinear;
            _enableMultisample = settings.EnableMultisample;
            _enableTerrainBackfaceCulling = settings.EnableTerrainBackfaceCulling;
            RenderQualitySettings.EnableTerrainBackfaceCulling = _enableTerrainBackfaceCulling;
            _defaultFogStart = float.IsFinite(settings.DefaultFogStart)
                ? Math.Clamp(settings.DefaultFogStart, 0f, 5000f)
                : 200f;
            _defaultFogEnd = float.IsFinite(settings.DefaultFogEnd)
                ? Math.Clamp(settings.DefaultFogEnd, 100f, 6000f)
                : 1500f;
            _uiFontScale = float.IsFinite(settings.UiFontScale) && settings.UiFontScale > 0.5f
                ? Math.Clamp(settings.UiFontScale, 0.75f, 2.5f)
                : 1.0f;
            if (HasImGuiContext())
            {
                ImGui.GetIO().FontGlobalScale = _uiFontScale;
            }
            _cameraSpeed = float.IsFinite(settings.CameraSpeed)
                ? Math.Clamp(settings.CameraSpeed, 1f, 500f)
                : 50f;
            _fovDegrees = float.IsFinite(settings.FovDegrees)
                ? Math.Clamp(settings.FovDegrees, 20f, 90f)
                : 45f;
            _showMinimapWindow = settings.ShowMinimapWindow;
            _useDockspaceUi = settings.ShellPanelLayoutVersion < CurrentShellPanelLayoutVersion
                ? true
                : settings.UseDockspaceUi;

            // 069 Phase 6: sticky archeology + tab system persistence
            _archeologyMinUniqueId = settings.ArcheologyMinUniqueId;
            _archeologyMaxUniqueId = settings.ArcheologyMaxUniqueId;
            _archeologyScopeIndex = settings.ArcheologyScopeIndex;
            _archeologyPlaybackSpeed = float.IsFinite(settings.ArcheologyPlaybackSpeed)
                ? Math.Clamp(settings.ArcheologyPlaybackSpeed, 1f, 5000f)
                : 50f;
            _archeologyPlaybackLoop = settings.ArcheologyPlaybackLoop;
            _archeologyApplyToNextCapture = settings.ArcheologyApplyToNextCapture;
            _archeologyApplyToVideoRecording = settings.ArcheologyApplyToVideoRecording;
            _useTabUi = settings.UseTabUi;
            if (Enum.IsDefined(typeof(WorkbenchTab), settings.ActiveTopTab))
                _activeTopTab = (WorkbenchTab)settings.ActiveTopTab;
            else
                _activeTopTab = WorkbenchTab.Quick;
            _activeBottomTabIndex = Math.Max(0, settings.ActiveBottomTab);
            if (_activeTopTab == WorkbenchTab.Editor
                && settings.WorkbenchNavigationVersion < CurrentWorkbenchNavigationVersion)
            {
                // Spec 231: pre-231 Editor page indices remap onto the 4-page IA.
                _activeBottomTabIndex = Workbench.Pages.EditorWorkbenchPages.MigrateLegacyEditorPageIndex(_activeBottomTabIndex);
            }
            _activeUtilitiesTabIndex = _activeTopTab == WorkbenchTab.Utilities
                ? _activeBottomTabIndex
                : 0;
            if (_useTabUi)
                NormalizeWorkbenchStateAfterLoad();
            _showLeftSidebar = settings.ShowLeftSidebar;
            _showRightSidebar = settings.ShowRightSidebar;
            _showWorkspaceBarsPanel = settings.ShowWorkspaceBarsPanel;
            _terrainWeakSignalRestoreEnabled = false;
            _terrainWeakSignalRestoreAllLoadedTiles = false;
            _terrainWeakSignalRestoreUseTextureSubdivisions = true;
            _terrainWeakSignalRestoreUseAutoFactor = settings.EnableWeakSignalTerrainRestoreAutoFactor;
            _terrainWeakSignalRestoreManualFactor = float.IsFinite(settings.WeakSignalTerrainRestoreManualFactor)
                ? Math.Clamp(settings.WeakSignalTerrainRestoreManualFactor, 1f, TerrainWeakSignalRestoreMaxFactor)
                : 16f;
            _terrainWeakSignalRestoreCandidateMinHeight = float.IsFinite(settings.WeakSignalTerrainRestoreCandidateMinHeight)
                ? TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(settings.WeakSignalTerrainRestoreCandidateMinHeight)
                : TerrainWeakSignalRestoreDefaultMinZ;
            _terrainWeakSignalRestoreCandidateMaxHeight = float.IsFinite(settings.WeakSignalTerrainRestoreCandidateMaxHeight)
                ? TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(settings.WeakSignalTerrainRestoreCandidateMaxHeight)
                : TerrainWeakSignalRestoreDefaultMaxZ;
            _terrainWeakSignalRestore.GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
            _leftSidebarWidth = float.IsFinite(settings.LeftSidebarWidth)
                ? settings.LeftSidebarWidth
                : DefaultSidebarWidth;
            _rightSidebarWidth = float.IsFinite(settings.RightSidebarWidth)
                ? settings.RightSidebarWidth
                : DefaultRightSidebarWidth;
            _bottomDrawerHeight = float.IsFinite(settings.BottomDrawerHeight)
                ? settings.BottomDrawerHeight
                : DefaultBottomDrawerHeight;
            _minimapZoom = float.IsFinite(settings.MinimapZoom)
                ? Math.Clamp(settings.MinimapZoom, 1f, 32f)
                : 4f;
            _minimapPanOffset = new Vector2(
                float.IsFinite(settings.MinimapPanOffsetX) ? settings.MinimapPanOffsetX : 0f,
                float.IsFinite(settings.MinimapPanOffsetY) ? settings.MinimapPanOffsetY : 0f);
            _captureOutputDir = string.IsNullOrWhiteSpace(settings.CaptureOutputDir)
                ? Path.Combine(OutputDir, "captures")
                : settings.CaptureOutputDir;
            _videoEncoderExecutable = string.IsNullOrWhiteSpace(settings.VideoEncoderExecutable)
                ? "ffmpeg"
                : settings.VideoEncoderExecutable;
            _videoCaptureFps = Math.Clamp(settings.VideoCaptureFps, 12, 60);
            _videoCaptureIncludeUi = settings.VideoCaptureIncludeUi;
            _videoCaptureContainerIndex = Math.Clamp(settings.VideoCaptureContainerIndex, 0, 1);
            _savedDetailedAdtTileCountOverride = Math.Clamp(settings.DetailedAdtTileCountOverride, 0, Terrain.TerrainManager.MaxManualDetailedTileCount);
            _pm4SavedOverlayTranslation = new Vector3(settings.Pm4TranslationX, settings.Pm4TranslationY, settings.Pm4TranslationZ);
            _pm4SavedOverlayRotationDegrees = new Vector3(settings.Pm4RotationX, settings.Pm4RotationY, settings.Pm4RotationZ);
            _pm4SavedOverlayScale = new Vector3(settings.Pm4ScaleX, settings.Pm4ScaleY, settings.Pm4ScaleZ);
            if (MathF.Abs(_pm4SavedOverlayScale.X) < 0.0001f ||
                MathF.Abs(_pm4SavedOverlayScale.Y) < 0.0001f ||
                MathF.Abs(_pm4SavedOverlayScale.Z) < 0.0001f)
            {
                _pm4SavedOverlayScale = Vector3.One;
            }

            // Migrate the short-lived MirrorX default workaround back to neutral scale
            // now that PM4 tile-local coordinates are remapped at conversion time.
            bool isLegacyMirrorX = MathF.Abs(_pm4SavedOverlayScale.X + 1f) < 0.0001f
                && MathF.Abs(_pm4SavedOverlayScale.Y - 1f) < 0.0001f
                && MathF.Abs(_pm4SavedOverlayScale.Z - 1f) < 0.0001f;
            if (isLegacyMirrorX
                && _pm4SavedOverlayTranslation.LengthSquared() < 0.0001f
                && _pm4SavedOverlayRotationDegrees.LengthSquared() < 0.0001f)
            {
                _pm4SavedOverlayScale = Vector3.One;
            }
            if (_pm4SavedOverlayRotationDegrees == Vector3.Zero && MathF.Abs(settings.Pm4YawDegrees) > 0.001f)
                _pm4SavedOverlayRotationDegrees = new Vector3(0f, 0f, settings.Pm4YawDegrees);

            // Load PM4 WMO match store
            _pm4WmoMatchStore = new Pm4WmoMatchStore(AppContext.BaseDirectory);
            _pm4WmoMatchEntries = _pm4WmoMatchStore.Load();

                        _savedTaxiActorModelOverridesByMap.Clear();
                        if (settings.TaxiActorModelOverrides != null)
                        {
                            foreach (SavedTaxiActorOverride savedOverride in settings.TaxiActorModelOverrides)
                            {
                                if (savedOverride == null
                                    || string.IsNullOrWhiteSpace(savedOverride.MapName)
                                    || savedOverride.RouteId < 0
                                    || string.IsNullOrWhiteSpace(savedOverride.ModelPath))
                                {
                                    continue;
                                }

                                if (!_savedTaxiActorModelOverridesByMap.TryGetValue(savedOverride.MapName, out Dictionary<int, string>? overridesByRoute))
                                {
                                    overridesByRoute = new Dictionary<int, string>();
                                    _savedTaxiActorModelOverridesByMap[savedOverride.MapName] = overridesByRoute;
                                }

                                overridesByRoute[savedOverride.RouteId] = savedOverride.ModelPath.Trim().Replace('/', '\\');
                            }
                        }

                        _savedPm4ObjectMatches.Clear();
                        if (settings.Pm4ObjectMatchSelections != null)
                        {
                            foreach (SavedPm4ObjectMatchSelection selection in settings.Pm4ObjectMatchSelections)
                            {
                                if (selection == null
                                    || string.IsNullOrWhiteSpace(selection.MapName)
                                    || string.IsNullOrWhiteSpace(selection.PlacementKind)
                                    || string.IsNullOrWhiteSpace(selection.ModelPath)
                                    || selection.ObjectPartId < 0)
                                {
                                    continue;
                                }

                                string key = BuildSavedPm4ObjectMatchKey(selection.MapName, selection.TileX, selection.TileY, selection.Ck24, selection.ObjectPartId);
                                _savedPm4ObjectMatches[key] = selection;
                            }
                        }

                        _savedObjectPathFiltersByMap.Clear();
                        if (settings.ObjectPathFilters != null)
                        {
                            foreach (SavedObjectPathFilterMap savedMap in settings.ObjectPathFilters)
                            {
                                if (string.IsNullOrWhiteSpace(savedMap.MapName))
                                    continue;

                                List<SavedObjectPathFilterEntry> savedEntries = savedMap.Filters
                                    .Where(entry => !string.IsNullOrWhiteSpace(entry.PathPrefix) && (entry.AppliesToWmo || entry.AppliesToMdx))
                                    .Select(entry => new SavedObjectPathFilterEntry
                                    {
                                        PathPrefix = entry.PathPrefix.Trim().Replace('/', '\\').Trim('\\'),
                                        AppliesToWmo = entry.AppliesToWmo,
                                        AppliesToMdx = entry.AppliesToMdx,
                                    })
                                    .Where(entry => !string.IsNullOrWhiteSpace(entry.PathPrefix))
                                    .OrderBy(entry => entry.PathPrefix, StringComparer.OrdinalIgnoreCase)
                                    .ToList();

                                if (savedEntries.Count == 0 && savedMap.Enabled)
                                    continue;

                                _savedObjectPathFiltersByMap[savedMap.MapName] = new SavedObjectPathFilterMap
                                {
                                    MapName = savedMap.MapName,
                                    Enabled = savedMap.Enabled,
                                    Filters = savedEntries,
                                };
                            }
                        }

                        _savedShellPanelLayouts.Clear();
                        _pendingShellPanelLayoutRestore.Clear();
                        _forceApplyShellPanelLayout = settings.ShellPanelLayoutVersion != CurrentShellPanelLayoutVersion;
                        if (!_forceApplyShellPanelLayout && settings.ShellPanelLayouts != null)
                        {
                            foreach (SavedShellPanelLayout savedLayout in settings.ShellPanelLayouts)
                            {
                                if (!Enum.IsDefined(typeof(ShellPanelId), savedLayout.PanelId))
                                    continue;

                                if (!float.IsFinite(savedLayout.NormalizedX)
                                    || !float.IsFinite(savedLayout.NormalizedY)
                                    || !float.IsFinite(savedLayout.NormalizedWidth)
                                    || !float.IsFinite(savedLayout.NormalizedHeight))
                                {
                                    continue;
                                }

                                var panelId = (ShellPanelId)savedLayout.PanelId;
                                _savedShellPanelLayouts[panelId] = new SavedShellPanelLayout
                                {
                                    PanelId = savedLayout.PanelId,
                                    NormalizedX = Math.Clamp(savedLayout.NormalizedX, 0f, 0.95f),
                                    NormalizedY = Math.Clamp(savedLayout.NormalizedY, 0f, 0.95f),
                                    NormalizedWidth = Math.Clamp(savedLayout.NormalizedWidth, 0.12f, 1f),
                                    NormalizedHeight = Math.Clamp(savedLayout.NormalizedHeight, 0.12f, 1f),
                                };
                                _pendingShellPanelLayoutRestore.Add(panelId);
                            }
                        }

            ApplySavedPm4AlignmentToScene();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerSettings] Failed to load settings: {ex.Message}");
        }
    }

    private void SaveViewerSettings()
    {
        try
        {
            Directory.CreateDirectory(SettingsDir);

            var settings = new ViewerSettings
            {
                UiTheme = (int)_uiTheme,
                WmoMliqRotationQuarterTurns = WmoRenderer.MliqRotationQuarterTurns,
                HasExplicitWmoMliqRotationOverride = _hasExplicitWmoMliqRotationOverride,
                LastGameFolderPath = _lastGameFolderPath,
                LastLooseOverlayPath = _lastLooseOverlayPath,
                LastDatasetCatalogRoot = _datasetCatalogRoot,
                LastDatasetVersionRoot = string.IsNullOrWhiteSpace(_selectedDatasetVersionRoot)
                    ? null
                    : _selectedDatasetVersionRoot,
                LastActiveDatasetVersionRoot = string.IsNullOrWhiteSpace(_activeDatasetVersionRoot)
                    ? null
                    : _activeDatasetVersionRoot,
                LastSelectedBuildVersion = _clientBuildOptions.Count > 0
                    ? _clientBuildOptions[Math.Clamp(_selectedBuildOptionIndex, 0, _clientBuildOptions.Count - 1)].BuildVersion
                    : null,
                TextureFilteringMode = (int)_textureFilteringMode,
                EnableMultisample = _enableMultisample,
                EnableTerrainBackfaceCulling = _enableTerrainBackfaceCulling,
                DefaultFogStart = _defaultFogStart,
                DefaultFogEnd = _defaultFogEnd,
                CameraSpeed = _cameraSpeed,
                FovDegrees = _fovDegrees,
                KnownGoodClientPaths = _knownGoodClientPaths,
                UiFontScale = _uiFontScale,
                ShowMinimapWindow = _showMinimapWindow,
                UseDockspaceUi = _useDockspaceUi,
                ShowLeftSidebar = _showLeftSidebar,
                ShowRightSidebar = _showRightSidebar,
                ShowWorkspaceBarsPanel = _showWorkspaceBarsPanel,
                ShowBottomDrawer = false,
                EnableWeakSignalTerrainRestore = false,
                EnableWeakSignalTerrainRestoreAllLoadedTiles = false,
                EnableWeakSignalTerrainRestoreUseChunkMode = false,
                EnableWeakSignalTerrainRestoreUseTextureSubdivisions = true,
                EnableWeakSignalTerrainRestoreAutoFactor = _terrainWeakSignalRestoreUseAutoFactor,
                EnableWeakSignalTerrainRestoreUseShadowHeuristic = false,
                WeakSignalTerrainRestoreManualFactor = _terrainWeakSignalRestoreManualFactor,
                WeakSignalTerrainRestoreCandidateMinHeight = _terrainWeakSignalRestoreCandidateMinHeight,
                WeakSignalTerrainRestoreCandidateMaxHeight = _terrainWeakSignalRestoreCandidateMaxHeight,
                LeftSidebarWidth = _leftSidebarWidth,
                RightSidebarWidth = _rightSidebarWidth,
                BottomDrawerHeight = _bottomDrawerHeight,
                MinimapZoom = _minimapZoom,
                MinimapPanOffsetX = _minimapPanOffset.X,
                MinimapPanOffsetY = _minimapPanOffset.Y,
                CaptureOutputDir = _captureOutputDir,
                VideoEncoderExecutable = _videoEncoderExecutable,
                VideoCaptureFps = _videoCaptureFps,
                VideoCaptureIncludeUi = _videoCaptureIncludeUi,
                VideoCaptureContainerIndex = _videoCaptureContainerIndex,
                DetailedAdtTileCountOverride = _savedDetailedAdtTileCountOverride,
                Pm4TranslationX = _pm4SavedOverlayTranslation.X,
                Pm4TranslationY = _pm4SavedOverlayTranslation.Y,
                Pm4TranslationZ = _pm4SavedOverlayTranslation.Z,
                Pm4RotationX = _pm4SavedOverlayRotationDegrees.X,
                Pm4RotationY = _pm4SavedOverlayRotationDegrees.Y,
                Pm4RotationZ = _pm4SavedOverlayRotationDegrees.Z,
                Pm4ScaleX = _pm4SavedOverlayScale.X,
                Pm4ScaleY = _pm4SavedOverlayScale.Y,
                Pm4ScaleZ = _pm4SavedOverlayScale.Z,
                Pm4YawDegrees = _pm4SavedOverlayRotationDegrees.Z,
                TaxiActorModelOverrides = _savedTaxiActorModelOverridesByMap
                    .OrderBy(entry => entry.Key, StringComparer.OrdinalIgnoreCase)
                    .SelectMany(entry => entry.Value
                        .OrderBy(routeEntry => routeEntry.Key)
                        .Select(routeEntry => new SavedTaxiActorOverride
                        {
                            MapName = entry.Key,
                            RouteId = routeEntry.Key,
                            ModelPath = routeEntry.Value
                        }))
                    .ToList(),
                Pm4ObjectMatchSelections = _savedPm4ObjectMatches.Values
                    .OrderBy(selection => selection.MapName, StringComparer.OrdinalIgnoreCase)
                    .ThenBy(selection => selection.TileX)
                    .ThenBy(selection => selection.TileY)
                    .ThenBy(selection => selection.Ck24)
                    .ThenBy(selection => selection.ObjectPartId)
                    .ToList(),
                ObjectPathFilters = _savedObjectPathFiltersByMap.Values
                    .OrderBy(entry => entry.MapName, StringComparer.OrdinalIgnoreCase)
                    .Select(entry => new SavedObjectPathFilterMap
                    {
                        MapName = entry.MapName,
                        Enabled = entry.Enabled,
                        Filters = entry.Filters
                            .OrderBy(filter => filter.PathPrefix, StringComparer.OrdinalIgnoreCase)
                            .Select(filter => new SavedObjectPathFilterEntry
                            {
                                PathPrefix = filter.PathPrefix,
                                AppliesToWmo = filter.AppliesToWmo,
                                AppliesToMdx = filter.AppliesToMdx,
                            })
                            .ToList(),
                    })
                    .ToList(),
                ShellPanelLayouts = _savedShellPanelLayouts.Values
                    .OrderBy(layout => layout.PanelId)
                    .Select(layout => new SavedShellPanelLayout
                    {
                        PanelId = layout.PanelId,
                        NormalizedX = layout.NormalizedX,
                        NormalizedY = layout.NormalizedY,
                        NormalizedWidth = layout.NormalizedWidth,
                        NormalizedHeight = layout.NormalizedHeight,
                    })
                    .ToList(),
                ArcheologyMinUniqueId = _archeologyMinUniqueId,
                ArcheologyMaxUniqueId = _archeologyMaxUniqueId,
                ArcheologyScopeIndex = _archeologyScopeIndex,
                ArcheologyPlaybackSpeed = _archeologyPlaybackSpeed,
                ArcheologyPlaybackLoop = _archeologyPlaybackLoop,
                ArcheologyApplyToNextCapture = _archeologyApplyToNextCapture,
                ArcheologyApplyToVideoRecording = _archeologyApplyToVideoRecording,
                UseTabUi = _useTabUi,
                WorkbenchNavigationVersion = CurrentWorkbenchNavigationVersion,
                ActiveTopTab = (int)_activeTopTab,
                ActiveBottomTab = _activeBottomTabIndex
            };

            string json = JsonSerializer.Serialize(settings, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(ViewerSettingsPath, json);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerSettings] Failed to save settings: {ex.Message}");
        }
    }

    private static List<KnownGoodClientPath> NormalizeKnownGoodClientPaths(List<KnownGoodClientPath>? knownGoodClientPaths)
    {
        if (knownGoodClientPaths == null || knownGoodClientPaths.Count == 0)
            return new List<KnownGoodClientPath>();

        var normalizedEntries = new List<KnownGoodClientPath>();
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var entry in knownGoodClientPaths)
        {
            if (entry == null || string.IsNullOrWhiteSpace(entry.Path))
                continue;

            string normalizedPath;
            try
            {
                normalizedPath = Path.GetFullPath(entry.Path);
            }
            catch
            {
                continue;
            }

            if (!seenPaths.Add(normalizedPath))
                continue;

            string name = string.IsNullOrWhiteSpace(entry.Name)
                ? Path.GetFileName(Path.TrimEndingDirectorySeparator(normalizedPath))
                : entry.Name.Trim();

            normalizedEntries.Add(new KnownGoodClientPath
            {
                Name = name,
                Path = normalizedPath,
                BuildVersion = string.IsNullOrWhiteSpace(entry.BuildVersion) ? null : entry.BuildVersion.Trim()
            });
        }

        return normalizedEntries
            .OrderBy(entry => entry.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private bool _disposed;

    private void SetLayoutObjectPreviewMode(bool enabled)
    {
        if (_layoutObjectPreviewMode == enabled)
            return;

        _layoutObjectPreviewMode = enabled;
        ApplyLayoutObjectPreviewModeToScene();
    }

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
        ShutdownMlTrainingMonitor();

        ISceneRenderer? renderer = _renderer;
        WorldScene? worldScene = _worldScene;
        TerrainManager? terrainManager = _terrainManager;
        VlmTerrainManager? vlmTerrainManager = _vlmTerrainManager;

        SaveViewerSettings();

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

    private sealed class ViewerSettings
    {
        public int UiTheme { get; set; } = (int)UiThemeKind.ModernSlate;
        public float UiFontScale { get; set; } = 1.0f;
        public int WmoMliqRotationQuarterTurns { get; set; }
        public bool HasExplicitWmoMliqRotationOverride { get; set; }
        public string? LastGameFolderPath { get; set; }
        public string? LastLooseOverlayPath { get; set; }
        public string? LastDatasetCatalogRoot { get; set; }
        public string? LastDatasetVersionRoot { get; set; }
        public string? LastActiveDatasetVersionRoot { get; set; }
        public string? LastSelectedBuildVersion { get; set; }
        public int TextureFilteringMode { get; set; } = (int)Rendering.TextureFilteringMode.Trilinear;
        public bool EnableMultisample { get; set; } = true;
        public bool EnableTerrainBackfaceCulling { get; set; } = true;
        public List<KnownGoodClientPath> KnownGoodClientPaths { get; set; } = new();
        public bool ShowMinimapWindow { get; set; } = true;
        public bool UseDockspaceUi { get; set; }
        public bool ShowLeftSidebar { get; set; } = true;
        public bool ShowRightSidebar { get; set; } = true;
        public bool ShowWorkspaceBarsPanel { get; set; } = true;
        public bool ShowBottomDrawer { get; set; } = true;
        public bool EnableWeakSignalTerrainRestore { get; set; }
        public bool EnableWeakSignalTerrainRestoreAllLoadedTiles { get; set; } = true;
        public bool EnableWeakSignalTerrainRestoreUseChunkMode { get; set; }
        public bool EnableWeakSignalTerrainRestoreUseTextureSubdivisions { get; set; } = true;
        public bool EnableWeakSignalTerrainRestoreAutoFactor { get; set; } = true;
        public bool EnableWeakSignalTerrainRestoreUseShadowHeuristic { get; set; }
        public float WeakSignalTerrainRestoreManualFactor { get; set; } = 16f;
        public float WeakSignalTerrainRestoreCandidateMinHeight { get; set; } = TerrainWeakSignalRestoreDefaultMinZ;
        public float WeakSignalTerrainRestoreCandidateMaxHeight { get; set; } = TerrainWeakSignalRestoreDefaultMaxZ;
        public int ShellPanelLayoutVersion { get; set; } = CurrentShellPanelLayoutVersion;
        public float LeftSidebarWidth { get; set; } = DefaultSidebarWidth;
        public float RightSidebarWidth { get; set; } = DefaultRightSidebarWidth;
        public float BottomDrawerHeight { get; set; } = DefaultBottomDrawerHeight;
        public float MinimapZoom { get; set; } = 4f;
        public float MinimapPanOffsetX { get; set; }
        public float MinimapPanOffsetY { get; set; }
        public string CaptureOutputDir { get; set; } = Path.Combine(OutputDir, "captures");
        public string VideoEncoderExecutable { get; set; } = "ffmpeg";
        public int VideoCaptureFps { get; set; } = 30;
        public bool VideoCaptureIncludeUi { get; set; }
        public int VideoCaptureContainerIndex { get; set; }
        public int DetailedAdtTileCountOverride { get; set; }
        public float Pm4TranslationX { get; set; }
        public float Pm4TranslationY { get; set; }
        public float Pm4TranslationZ { get; set; }
        public float Pm4RotationX { get; set; }
        public float Pm4RotationY { get; set; }
        public float Pm4RotationZ { get; set; }
        public float Pm4ScaleX { get; set; } = 1f;
        public float Pm4ScaleY { get; set; } = 1f;
        public float Pm4ScaleZ { get; set; } = 1f;
        public float Pm4YawDegrees { get; set; }
        public List<SavedTaxiActorOverride> TaxiActorModelOverrides { get; set; } = new();
        public List<SavedPm4ObjectMatchSelection> Pm4ObjectMatchSelections { get; set; } = new();
        public List<SavedObjectPathFilterMap> ObjectPathFilters { get; set; } = new();
        public List<SavedShellPanelLayout> ShellPanelLayouts { get; set; } = new();

        // 069 Phase 6: sticky archeology settings
        public int ArcheologyMinUniqueId { get; set; } = -1;
        public int ArcheologyMaxUniqueId { get; set; } = -1;
        public int ArcheologyScopeIndex { get; set; }

        // 069 Phase 7: archeology playback + capture integration
        public float ArcheologyPlaybackSpeed { get; set; } = 50f;
        public bool ArcheologyPlaybackLoop { get; set; }
        public bool ArcheologyApplyToNextCapture { get; set; }
        public bool ArcheologyApplyToVideoRecording { get; set; }

        // 069 tab system persistence
        public bool UseTabUi { get; set; } = true;
        public int WorkbenchNavigationVersion { get; set; }
        public int ActiveTopTab { get; set; }
        public int ActiveBottomTab { get; set; }

        // Global fog defaults
        public float DefaultFogStart { get; set; } = 200f;
        public float DefaultFogEnd { get; set; } = 1500f;

        // Camera defaults
        public float CameraSpeed { get; set; } = 50f;
        public float FovDegrees { get; set; } = 45f;
    }

    private sealed class SavedTaxiActorOverride
    {
        public string MapName { get; set; } = "";
        public int RouteId { get; set; }
        public string ModelPath { get; set; } = "";
    }

    private sealed class SavedPm4ObjectMatchSelection
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

    private sealed class SavedObjectPathFilterMap
    {
        public string MapName { get; set; } = "";
        public bool Enabled { get; set; } = true;
        public List<SavedObjectPathFilterEntry> Filters { get; set; } = new();
    }

    private sealed class SavedObjectPathFilterEntry
    {
        public string PathPrefix { get; set; } = "";
        public bool AppliesToWmo { get; set; }
        public bool AppliesToMdx { get; set; }
    }

    private sealed class KnownGoodClientPath
    {
        public string Name { get; set; } = "";
        public string Path { get; set; } = "";
        public string? BuildVersion { get; set; }
    }
}
