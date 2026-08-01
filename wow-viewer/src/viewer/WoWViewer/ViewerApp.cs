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
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
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
public partial class ViewerApp : IDisposable
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

    private readonly record struct PlacementEditKey(Terrain.ObjectType ObjectType, int TileX, int TileY, int EntryIndex, int UniqueId);

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

    private sealed class StagedPlacementEdit
    {
        public PlacementEditKey Key { get; init; }
        public string SourcePath { get; init; } = string.Empty;
        public Vector3 OriginalPosition { get; set; }
        public Vector3 EditedPosition { get; set; }
    }

    private const string ViewerProductName = "WoWViewer v0.5.2";
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
    private Vector2 _minimapDragStart = Vector2.Zero;
    private Vector2 _minimapDragOrigin = Vector2.Zero;
    private (int tileX, int tileY)? _pendingMinimapTeleportTile;
    private int _pendingMinimapTeleportClickCount;
    private DateTime _pendingMinimapTeleportLastClickUtc = DateTime.MinValue;
    private Rendering.LoadingScreen? _loadingScreen;

    // Output directories (next to the executable)
    private static readonly string OutputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");
    private static readonly string CacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache");
    private static readonly string ExportDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "export");
    private static readonly string ProjectsDir = Path.Combine(OutputDir, "projects");
    private static readonly string SettingsDir = Path.Combine(OutputDir, "settings");
    private static readonly string ViewerSettingsPath = Path.Combine(SettingsDir, "viewer_settings.json");
    private const int CurrentShellPanelLayoutVersion = 4;
    private const int MinimapTeleportConfirmClicks = 3;
    private const float MinimapClickMovementThresholdPixels = 3f;
    private static readonly TimeSpan MinimapTeleportConfirmWindow = TimeSpan.FromSeconds(3);

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
        return typeof(ViewerApp).Assembly
                   .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
                   ?.InformationalVersion
               ?? typeof(ViewerApp).Assembly.GetName().Version?.ToString(3)
               ?? "unknown";
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
    private bool _showRenderQualityWindow = false;
    private bool _openAboutPopup;
    private WorkspaceMode _workspaceMode = WorkspaceMode.Viewer;
    private EditorWorkspaceTask _editorWorkspaceTask = EditorWorkspaceTask.Terrain;
    private FixedBottomDrawerTab _activeBottomDrawerTab = FixedBottomDrawerTab.Workspace;
    private FixedBottomDrawerTab? _pendingRightSidebarSection;
    private bool _useDockspaceUi = true;

    // 069 Phase 1: tab system state. On by default; can toggle off via View > Legacy Sidebar UI.
    private bool _useTabUi = true;
    private WorkbenchTab _activeTopTab = WorkbenchTab.Tools;
    private int _activeBottomTabIndex = 0;

    // Nested sub-tab selection. These must be separate from _activeBottomTabIndex: Tools >
    // Archeology, Terrain, and Utilities are a second nesting level, so reusing the parent's index
    // pins the child to whatever slot the parent occupies and makes its other tabs unreachable.
    private int _activeArcheologyTabIndex = 0;
    private int _activeTerrainTabIndex = 0;
    private int _activeUtilitiesTabIndex = 0;

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
    private bool _wantExportGlb = false;
    private bool _wantExportGlbCollision = false;
    private bool _wantExportMapGlbTiles = false;
    private string _projectOutputRootDir = ProjectsDir;
    private string _editorProjectOutputDir = string.Empty;
    private string _editorProjectSourceKey = string.Empty;
    private Terrain.ObjectType _selectedPlacementEditType = Terrain.ObjectType.None;
    private int _selectedPlacementEditUniqueId = -1;
    private int _selectedPlacementEditTileX = -1;
    private int _selectedPlacementEditTileY = -1;
    private int _selectedPlacementEditEntryIndex = -1;
    private Vector3 _selectedPlacementOriginalPosition;
    private Vector3 _selectedPlacementEditedPosition;
    private bool _selectedPlacementDirty;
    private string? _selectedPlacementSourcePath;
    private string? _selectedPlacementSaveTargetPath;
    private readonly Dictionary<PlacementEditKey, StagedPlacementEdit> _stagedPlacementEdits = new();
    private readonly Dictionary<string, string> _placementSaveTargetsBySourcePath = new(StringComparer.OrdinalIgnoreCase);
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

    private enum TerrainTileScope
    {
        CurrentTile = 0,
        LoadedTiles = 1,
        WholeMap = 2,
        CustomList = 3,
        RectRange = 4,
    }

    private enum TerrainExportKind
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

    private enum TerrainImportKind
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
    private string _terrainImportFolder = "";
    private string _terrainCustomTilesText = "";
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
    private bool _chunkClipboardUseMouse;
    private bool _chunkClipboardPasteRelativeHeights = true;
    private bool _chunkClipboardIncludeAlphaShadow;
    private bool _chunkClipboardIncludeTextures;
    private (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardCopiedKey;
    private (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardLockedTargetKey;
    private int _chunkClipboardSelectionRotation;
    private bool _chunkClipboardCtrlCWasPressed;
    private bool _chunkClipboardCtrlVWasPressed;
    private readonly HashSet<(int tileX, int tileY, int chunkX, int chunkY)> _selectedChunks = new();
    private ChunkClipboardSet? _chunkClipboardSet;
    private bool _chunkClipboardShowOverlay = true;
    private readonly Dictionary<(int tileX, int tileY), HashSet<(int chunkX, int chunkY)>> _chunkClipboardDirtyTileChunks = new();
    private string _chunkClipboardLastSaveFolder = string.Empty;
    private TerrainAnalysisPreviewTexture? _terrainAnalysisLocalTexture;
    private TerrainAnalysisPreviewTexture? _terrainAnalysisGlobalTexture;
    private TerrainAnalysisPreviewTexture? _terrainAnalysisAlphaTexture;
    private const float TerrainWeakSignalRestoreMinZLimit = -8192f;
    private const float TerrainWeakSignalRestoreMaxZLimit = 512f;
    private const float TerrainWeakSignalRestoreDefaultMinZ = -10f;
    private const float TerrainWeakSignalRestoreDefaultMaxZ = 10f;
    private const float TerrainWeakSignalRestoreMaxFactor = 512f;
    private const float TerrainWeakSignalShadowEdgeMinCoverage = 0.55f;
    private const float TerrainWeakSignalShadowLitMaxCoverage = 0.45f;
    private const float TerrainWeakSignalShadowEdgeMinHeightDelta = 0.5f;
    private bool _terrainWeakSignalRestoreEnabled;
    private bool _terrainWeakSignalRestoreAllLoadedTiles = true;
    private bool _terrainWeakSignalRestoreUseTextureSubdivisions = true;
    private bool _terrainWeakSignalRestoreUseAutoFactor = true;
    private float _terrainWeakSignalRestoreManualFactor = 16f;
    private float _terrainWeakSignalRestoreCandidateMinHeight = TerrainWeakSignalRestoreDefaultMinZ;
    private float _terrainWeakSignalRestoreCandidateMaxHeight = TerrainWeakSignalRestoreDefaultMaxZ;
    private string _terrainWeakSignalRestoreStatus = string.Empty;
    private readonly Dictionary<(int tileX, int tileY), List<Terrain.TerrainChunkData>> _terrainWeakSignalOriginalTiles = new();
    private readonly Dictionary<(int tileX, int tileY), int> _terrainWeakSignalAppliedPlans = new();
    private readonly HashSet<(int tileX, int tileY)> _terrainWeakSignalApplyingTiles = new();
    private Terrain.TerrainManager? _terrainWeakSignalHookedTerrainManager;
    private Terrain.VlmTerrainManager? _terrainWeakSignalHookedVlmTerrainManager;
    private string? _terrainWeakSignalWdlMapName;
    private WdlParser.WdlData? _terrainWeakSignalWdlData;
    private (int tileX, int tileY)? _terrainWeakSignalRestoreLastCameraTile;
    private bool _terrainWeakSignalRestoreNeedsRefresh = true;
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

    private sealed class HeightmapMetadata
    {
        public int Version { get; set; } = 1;
        public int Resolution { get; set; } = TerrainHeightmapIo.TileHeightmapSize;
        public float MinHeight { get; set; }
        public float MaxHeight { get; set; }
        public string Normalization { get; set; } = "per_tile";
    }

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

    private readonly struct TerrainWeakSignalSubChunkCell
    {
        public int CellX { get; init; }
        public int CellY { get; init; }
        public int DominantLayerIndex { get; init; }
        public float MinHeight { get; init; }
        public float MaxHeight { get; init; }
        public float AverageHeight { get; init; }
        public bool IsWeakSignalCandidate { get; init; }
        public bool TouchesBorder { get; init; }
    }

    private sealed class TerrainWeakSignalTextureGuidance
    {
        public TerrainWeakSignalSubChunkCell[] Cells { get; init; } = Array.Empty<TerrainWeakSignalSubChunkCell>();
        public bool[] SelectedMask { get; init; } = Array.Empty<bool>();
        public int DominantLayerIndex { get; init; }
        public int SelectedCellCount { get; init; }
        public int BorderSelectedCellCount { get; init; }
        public float ObservedMinHeight { get; init; }
        public float ObservedMaxHeight { get; init; }
        public float ObservedAverageHeight { get; init; }
    }

    private sealed class ChunkToolHeightmapSaveManifest
    {
        public int Version { get; set; } = 1;
        public string Format { get; set; } = "heightmap_257_per_tile";
        public string ProjectName { get; set; } = string.Empty;
        public string SourceKey { get; set; } = string.Empty;
        public string GeneratedUtc { get; set; } = string.Empty;
        public List<ChunkToolHeightmapSaveTile> Tiles { get; set; } = new();
    }

    private sealed class ChunkToolHeightmapSaveTile
    {
        public int TileX { get; set; }
        public int TileY { get; set; }
        public List<string> EditedChunks { get; set; } = new();
        public string HeightmapPng { get; set; } = string.Empty;
        public string HeightmapMetadataJson { get; set; } = string.Empty;
        public float MinHeight { get; set; }
        public float MaxHeight { get; set; }
    }

    private sealed class ChunkClipboard
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
            AlphaMaps = CloneAlphaMaps(chunk.AlphaMaps);
            ShadowMap = chunk.ShadowMap != null ? (byte[])chunk.ShadowMap.Clone() : null;
            MccvColors = chunk.MccvColors != null ? (byte[])chunk.MccvColors.Clone() : null;
        }
    }

    private sealed class ChunkClipboardSet
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
    private bool _showPm4AlignmentWindow;
    private bool _showPm4ObjectMatchWindow;
    private bool _showPm4WmoCorrelationWindow;
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
    private bool _showChunkClipboardWindow = false;
    private bool _showTerrainAnalysisWindow;
    private bool _showTerrainToolsWindow;
    private bool _showMcnkExplorerWindow;
    private bool _showCaptureAutomationWindow = false;
    private bool _showUniqueIdArchaeologyWindow;
    private bool _showWeakSignalWindow;
    private bool _showPm4SceneGraph = true;

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
    private int _mapConvertDirection = 0; // 0 = Alpha→LK, 1 = LK→Alpha
    private string _mapConvertSourcePath = "";
    private string _mapConvertOutputDir = "";
    private string _mapConvertProjectSourceKey = string.Empty;
    private string _mapConvertLkMapDir = ""; // LK→Alpha: directory containing LK ADT files
    private bool _mapConvertCopyAlphaSourceWdt = true;
    private bool _mapConvertEmitLkSplitOutputs = true;
    private bool _mapConvertVerbose = true;
    private string _areaCrosswalkPath = ""; // Optional area crosswalk CSV for Alpha→LK conversion
    private bool _mapConverting = false;
    private readonly List<string> _mapConvertLog = new();
    private bool _mapConvertScrollToBottom = false;
    private string? _mapConvertError = null;
    private bool _mapConvertDone = false;
    private string? _mapConvertLastLoadPath;

    // WMO Converter state
    private bool _showWmoConverterDialog = false;
    private int _wmoConvertDirection = 0; // 0 = Alpha(v14/v16)→LK(v17), 1 = LK(v17)→Alpha(v14)
    private string _wmoConvertSourcePath = "";
    private string _wmoConvertOutputPath = "";
    private bool _wmoConvertCopyTextures = true;
    private bool _wmoConverting = false;
    private readonly List<string> _wmoConvertLog = new();
    private bool _wmoConvertScrollToBottom = false;
    private string? _wmoConvertError = null;
    private bool _wmoConvertDone = false;

    // Terrain-derived minimap export state. This is intentionally separate from the retired
    // VLM/MK dataset workflow: it invokes the direct client terrain synthesis command.
    private bool _showSynthesizedMinimapExportDialog;
    private string _synthesizedMinimapClientRoot = string.Empty;
    private string _synthesizedMinimapMapName = string.Empty;
    private string _synthesizedMinimapOutputDirectory = string.Empty;
    private float _synthesizedMinimapTimeHours = 12f;
    private int _synthesizedMinimapResolution = 256;
    private bool _synthesizedMinimapEmitTiles = true;
    private bool _synthesizedMinimapEmitWholeMap = true;
    private bool _synthesizedMinimapIncludeWmos;
    private bool _synthesizedMinimapBakeMcsh;
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
    private int _vlmTileLimit = 0; // 0 = unlimited
    private bool _vlmExporting = false;
    private readonly List<string> _vlmExportLog = new();
    private bool _vlmExportScrollToBottom = false;
    private VlmExportResult? _vlmExportResult = null;

    // ML Dataset manifest and validation state
    private string _mkHarvestDatasetRoot = "";
    private string _mkHarvestManifestOutputPath = "";
    private string _mkHarvestReferenceOutputDir = "";
    private string _mkHarvestViewerValidationOutputDir = "";
    private bool _mlFinalizeAfterExport = true;
    private bool _pendingMlFinalizeAfterExport = false;
    private bool _mkHarvestGenerateViewerValidationMinimaps = true;
    private bool _mkHarvestForceViewerValidationRegeneration = false;
    private int _mkHarvestViewerValidationResolution = 512;
    private bool _mkHarvestRunning = false;
    private readonly List<string> _mkHarvestLog = new();
    private bool _mkHarvestScrollToBottom = false;
    private MkDatasetHarvestResult? _mkHarvestResult = null;
    private int _mkHarvestViewerValidationQueued = 0;
    private int _mkHarvestViewerValidationCompleted = 0;
    private int _mkHarvestViewerValidationFailed = 0;

    // Terrain texture transfer state
    private bool _showTerrainTextureTransferDialog = false;
    private string _terrainTransferSourceDir = Pm4CoordinateService.DefaultDevelopmentMapDirectory;
    private string _terrainTransferTargetDir = Pm4CoordinateService.DefaultDevelopmentMapDirectory;
    private string _terrainTransferOutputDir = Path.Combine("output", "terrain-texture-transfer-ui");
    private bool _terrainTransferApplyMode = false;
    private bool _terrainTransferUseGlobalDelta = false;
    private int _terrainTransferSourceTileX = 0;
    private int _terrainTransferSourceTileY = 0;
    private int _terrainTransferTargetTileX = 0;
    private int _terrainTransferTargetTileY = 0;
    private int _terrainTransferDeltaX = 0;
    private int _terrainTransferDeltaY = 0;
    private int _terrainTransferTileLimit = 1;
    private int _terrainTransferChunkOffsetX = 0;
    private int _terrainTransferChunkOffsetY = 0;
    private bool _terrainTransferCopyMtex = true;
    private bool _terrainTransferCopyMcly = true;
    private bool _terrainTransferCopyMcal = true;
    private bool _terrainTransferCopyMcsh = true;
    private bool _terrainTransferCopyHoles = true;
    private string _terrainTransferManifestPath = "";
    private bool _terrainTransferRunning = false;
    private readonly List<string> _terrainTransferLog = new();
    private bool _terrainTransferScrollToBottom = false;
    private string? _terrainTransferError = null;
    private WoWViewer.Transfer.TerrainTextureTransferExecutionReport? _terrainTransferReport = null;

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
                            if (TryLockChunkPasteTarget(terrainRenderer))
                                return;
                        }
                        else if (shift)
                        {
                            if (TryHandleChunkSelectionClick(terrainRenderer, shift))
                                return;
                        }
                    }

                    if (_worldScene != null)
                        PickObjectAtMouse(_lastMouseX, _lastMouseY, addPm4ToCollection: shift);
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
        UpdateTaxiRideCamera();
        UpdateArcheologyPlayback(dt);
        _minimapRenderer?.ProcessPendingLoads(
            maxLoads: (_fullscreenMinimap || _showMinimapWindow) ? 4 : 1,
            maxBudgetMs: (_fullscreenMinimap || _showMinimapWindow) ? 6.0 : 1.5);
        UpdateSqlSpawnStreaming();
        UpdateTerrainWeakSignalRestoreForCamera();
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

        if (_chunkToolEnabled && canSceneConsumeKeyboard)
        {
            var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            if (terrainRenderer != null)
            {
                if (ctrlCDown && !_chunkClipboardCtrlCWasPressed)
                    ExecuteChunkClipboardCopy(terrainRenderer);

                if (ctrlVDown && !_chunkClipboardCtrlVWasPressed)
                    ExecuteChunkClipboardPaste(terrainRenderer);
            }
        }

        _chunkClipboardCtrlCWasPressed = ctrlCDown;
        _chunkClipboardCtrlVWasPressed = ctrlVDown;

        bool tabPressed = kb.IsKeyPressed(Key.Tab);
        if (canSceneConsumeKeyboard && tabPressed && !_tabKeyWasPressed)
            _hideUiChrome = !_hideUiChrome;
        _tabKeyWasPressed = tabPressed;

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
        if (_renderer is IModelRenderer modelRenderer && modelRenderer.Animator != null && modelRenderer.Animator.Sequences.Count > 0)
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
        bool s = kb.IsKeyPressed(Key.S);
        bool d = kb.IsKeyPressed(Key.D);
        bool q = kb.IsKeyPressed(Key.Q);
        bool e = kb.IsKeyPressed(Key.E);

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
        PromotePendingMlFinalizeAfterExport();
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

        // Render 3D scene first
        if (_renderer != null)
        {
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

            // Update terrain AOI before rendering
            if (_terrainManager != null)
                _terrainManager.UpdateAOI(_camera.Position, _camera.Forward);
            else if (_vlmTerrainManager != null)
                _vlmTerrainManager.UpdateAOI(_camera.Position);

            if (_worldScene != null)
                UpdateWorldSceneWireframeReveal(view, proj);

            if (_worldScene != null)
                UpdateWorldSceneHoveredAssetInfo(view, proj);

            // Update current area name from chunk under camera (throttled to avoid per-frame overhead).
            // Covers BOTH terrain paths: the status bar shows coordinates for _terrainManager and
            // _vlmTerrainManager alike, so the area lookup must too (it previously only ran for
            // _terrainManager, leaving the Area segment permanently blank in VLM sessions).
            var areaChunkRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            if (_areaTableService != null && areaChunkRenderer != null && _frameCount == 0)
            {
                var chunk = areaChunkRenderer.GetChunkAt(_camera.Position.X, _camera.Position.Y);
                if (chunk != null && chunk.AreaId != 0)
                {
                    // Filter by MapID to avoid showing areas from other continents
                    var name = _areaTableService.GetAreaDisplayNameForMap(chunk.AreaId, _currentMapId);
                    if (name == null)
                    {
                        ReportAreaLookupDiagnostic(chunk.AreaId);
                        // Fallback if MapID filtering fails
                        name = _areaTableService.GetAreaDisplayName(chunk.AreaId);
                    }
                    _currentAreaName = name ?? "";
                }
                else
                    _currentAreaName = "";
            }

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
            }

            if (hasSceneViewport)
                _gl.Viewport(_window.FramebufferSize);
        }

        CaptureVideoFrameIfNeeded(includeUi: false, dt);
        CompleteCaptureIfReady(includeUi: false);

        // Render ImGui overlay when the native ImGui context is live. Startup capture and
        // teardown can briefly produce frames where the controller still exists but the
        // underlying context is not available.
        if (HasImGuiContext())
        {
            DrawUI();
            _imGui.Render();
        }

        CaptureVideoFrameIfNeeded(includeUi: true, dt);
        CompleteCaptureIfReady(includeUi: true);
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
        if (_hideUiChrome || !_useDockspaceUi)
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
                if (_fullscreenMinimap)
                    DrawFullscreenMinimap();
                // All other tools are routed into tab sub-tabs. The
                // _show*Window flags still exist for users who want the
                // legacy window, but tab system renders its own sub-tab body.
            }
            else
            {
                // Asset Catalog (floating window)
                _catalogView?.Draw();

                // Log Viewer (floating window)
                if (_showLogViewer)
                    DrawLogViewer();

                // WDL Preview (floating window)
                if (_showWdlPreview)
                    DrawWdlPreviewDialog();

                // Minimap panel
                if (IsShellPanelActive(ShellPanelId.Minimap) && !_fullscreenMinimap)
                    DrawMinimapWindow();

                // Perf (floating window)
                if (_showPerfWindow)
                    DrawPerfWindow();

                // Render quality (floating window)
                if (_showRenderQualityWindow)
                    DrawRenderQualityWindow();

                // Terrain Tools (floating window) - available in both modes; tabbed mode also has Terrain > Tools sub-tab
                if (_showTerrainToolsWindow && (_terrainManager != null || _vlmTerrainManager != null))
                    DrawTerrainToolsWindow();

                // Chunk Clipboard (floating window)
                if (_showChunkClipboardWindow && (_terrainManager?.Renderer != null || _vlmTerrainManager?.Renderer != null))
                    DrawChunkClipboardWindow();

                if (_showTerrainAnalysisWindow && (_terrainManager != null || _vlmTerrainManager != null))
                    DrawTerrainAnalysisWindow();

                if (_showMcnkExplorerWindow && (_terrainManager != null || _vlmTerrainManager != null))
                    DrawMcnkExplorerWindow();

                if (_showCaptureAutomationWindow)
                    DrawCaptureAutomationWindow();

                // PM4 alignment (advanced fallback) - only in legacy mode; tabbed mode uses PM4 > Alignment sub-tab
                if (_showPm4AlignmentWindow && !_useTabUi)
                    DrawPm4AlignmentWindow();

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
            DrawVlmExportDialog();
        if (_showMlTrainingDialog)
            DrawMlTrainingDialog();
        if (_showTerrainTextureTransferDialog)
            DrawTerrainTextureTransferDialog();
        if (_showAlphaFolderImportScope)
            DrawAlphaFolderImportScopeDialog();
        if (_showHeightmapFolderImportScope)
            DrawHeightmapFolderImportScopeDialog();
        if (_showMccvFolderImportScope)
            DrawMccvFolderImportScopeDialog();
        if (_showMapConverterDialog)
            DrawMapConverterDialog();
        if (_showWmoConverterDialog)
            DrawWmoConverterDialog();
        if (_showSynthesizedMinimapExportDialog)
            DrawSynthesizedMinimapExportDialog();

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

                if (ImGui.MenuItem("Open Game Folder (MPQ)..."))
                {
                    _showFolderInput = true;
                    _folderInputBuf = string.IsNullOrWhiteSpace(_lastGameFolderPath) ? "" : _lastGameFolderPath;
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
                    if (_catalogView == null)
                    {
                        _catalogView = new AssetCatalogView(_gl);
                        _catalogView.SetDataSource(_dataSource);
                        _catalogView.OnLoadModelRequested = OnCatalogLoadModel;
                    }
                    _catalogView.IsVisible = !_catalogView.IsVisible;
                }

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Tools"))
            {
                // 071: floating-window toggles removed. Every tool lives in a
                // workbench tab under Tools > Panels or the relevant top tab.

                if (ImGui.MenuItem("Settings..."))
                    _showSettingsWindow = true;

                ImGui.Separator();

                if (ImGui.BeginMenu("Offline Data / Conversion"))
                {
                    if (ImGui.MenuItem("Open Zarr Dataset..."))
                        _wantOpenZarrDataset = true;

                    ImGui.Separator();

                    if (ImGui.MenuItem("Build ML Dataset..."))
                    {
                        PrepareVlmExportDialogInputs();
                        PrepareMkHarvestDialogInputs();
                        _showVlmExportDialog = true;
                    }

                    if (ImGui.MenuItem("Train V7 Terrain Model..."))
                    {
                        PrepareMlTrainingDialogInputs();
                        _showMlTrainingDialog = true;
                    }

                    if (ImGui.MenuItem("Terrain Texture Transfer..."))
                    {
                        PrepareTerrainTextureTransferDialogInputs();
                        _showTerrainTextureTransferDialog = true;
                    }

                    ImGui.Separator();

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
                        OpenWorkbenchTab(ToolsBottomTab.Utilities);
                    if (ImGui.MenuItem("Perf"))
                        OpenWorkbenchTab(ToolsBottomTab.Utilities);
                    if (ImGui.MenuItem("Settings..."))
                        _showSettingsWindow = true;

                    ImGui.Separator();

                    if (ImGui.MenuItem("Asset Catalog"))
                        OpenWorkbenchTab(ToolsBottomTab.Utilities);
                    if (ImGui.MenuItem("Capture Automation"))
                        OpenWorkbenchTab(ToolsBottomTab.Utilities);
                    if (ImGui.MenuItem("Taxi", hasWorld))
                        OpenWorkbenchTab(ToolsBottomTab.Utilities);

                    ImGui.Separator();

                    if (ImGui.MenuItem("UniqueId Archeology", hasWorld))
                    {
                        if (_useTabUi)
                            OpenWorkbenchTab(ToolsBottomTab.Archeology);
                        else
                            _showUniqueIdArchaeologyWindow = true;
                    }

                    ImGui.Separator();

                    if (ImGui.MenuItem("Chunk Clipboard", hasTerrain))
                        OpenWorkbenchTab(ToolsBottomTab.Terrain);
                    if (ImGui.MenuItem("Terrain Analysis", hasTerrain))
                        OpenWorkbenchTab(ToolsBottomTab.Terrain);
                    if (ImGui.MenuItem("MCNK Explorer", hasTerrain))
                        OpenWorkbenchTab(ToolsBottomTab.Terrain);
                    if (ImGui.MenuItem("Weak Signal", hasTerrain))
                        OpenWorkbenchTab(ToolsBottomTab.Terrain);
                    if (ImGui.MenuItem("Terrain Tools", hasTerrain))
                        OpenWorkbenchTab(ToolsBottomTab.Terrain);

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
                        if (ImGui.MenuItem("Export GLB...", _renderer != null))
                            _wantExportGlb = true;
                        if (ImGui.MenuItem("Export GLB (Collision Only)...", _renderer != null))
                            _wantExportGlbCollision = true;

                        ImGui.Separator();

                        bool canExportMapGlb = _terrainManager != null && _dataSource != null;
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
                if (ImGui.MenuItem("About"))
                {
                    _openAboutPopup = true;
                    _statusMessage = $"{ViewerProductName} {GetViewerDisplayVersion()}";
                }
                ImGui.EndMenu();
            }

            ImGui.EndMainMenuBar();
        }

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

            string? vlmPath = ShowFolderDialogSTA(
                "Select ML Dataset folder (containing dataset/ with JSON files)",
                initialDir: null,
                showNewFolderButton: false);

            if (!string.IsNullOrEmpty(vlmPath) && Directory.Exists(vlmPath))
                LoadVlmProject(vlmPath);
        }

        if (_wantOpenZarrDataset)
        {
            _wantOpenZarrDataset = false;

            string? zarrPath = ShowFolderDialogSTA(
                "Select Zarr tile dataset folder (parent of <build>.zarr/ or the store root itself)",
                initialDir: null,
                showNewFolderButton: false);

            if (!string.IsNullOrEmpty(zarrPath) && Directory.Exists(zarrPath))
                LoadZarrDataset(zarrPath);
        }

        if (_wantOpenWdtFile)
        {
            _wantOpenWdtFile = false;
            string? wdtPath = ShowFileDialogSTA(
                "Select Alpha WDT file (loose map)",
                "WoW map files (*.wdt;*.wdt.MPQ)|*.wdt;*.wdt.MPQ|All files (*.*)|*.*",
                _lastLooseOverlayPath);
            if (!string.IsNullOrEmpty(wdtPath) && File.Exists(wdtPath))
            {
                LoadFileFromDisk(wdtPath);
                _statusMessage = $"Loaded alpha WDT: {wdtPath}";
            }
        }

        if (_wantAttachLooseMapFolder)
        {
            _wantAttachLooseMapFolder = false;

            if (_dataSource is MpqDataSource)
            {
                string? overlayPath = ShowFolderDialogSTA(
                    "Select loose map overlay folder (contains World\\Maps or a map directory under World\\Maps)",
                    initialDir: string.IsNullOrWhiteSpace(_lastLooseOverlayPath) ? null : _lastLooseOverlayPath,
                    showNewFolderButton: false);

                if (!string.IsNullOrEmpty(overlayPath) && Directory.Exists(overlayPath))
                    AttachLooseMapOverlay(overlayPath);
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
                string? overlayPath = ShowFolderDialogSTA(
                    "Select loose map folder to load against the saved base client",
                    initialDir: string.IsNullOrWhiteSpace(_lastLooseOverlayPath) ? null : _lastLooseOverlayPath,
                    showNewFolderButton: false);

                if (!string.IsNullOrWhiteSpace(overlayPath) && Directory.Exists(overlayPath))
                {
                    LoadMpqDataSource(savedBasePath, null, savedBuildVersion, deferWorldReload: true);
                    AttachLooseMapOverlay(overlayPath);
                    RestoreWorldAfterDataSourceReload();
                }
            }
            else
            {
                LoadMpqDataSource(savedBasePath, null, savedBuildVersion);
            }
        }

        if (_wantTerrainExport)
        {
            _wantTerrainExport = false;
            RunTerrainExport();
        }

        if (_wantTerrainImport)
        {
            _wantTerrainImport = false;
            RunTerrainImport();
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
                            throw new InvalidOperationException("Collision-only GLB export is currently supported for WMO only.");
                        }
                    }
                    _statusMessage = $"Exported: {glbPath}";
                }
                catch (Exception ex)
                {
                    _statusMessage = $"Export failed: {ex.Message}";
                }
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
        }

        if (_wantExportMapGlbTiles)
        {
            _wantExportMapGlbTiles = false;
            try
            {
                RunMapGlbTilesExport();
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

    private void RunMapGlbTilesExport()
    {
        if (_terrainManager == null)
        {
            _statusMessage = "No terrain loaded.";
            return;
        }

        if (_dataSource == null)
        {
            _statusMessage = "No data source loaded (required to export textures/models).";
            return;
        }

        var tiles = GetTileScopeList(_mapGlbScope);
        if (tiles.Count == 0)
        {
            _statusMessage = "No tiles in scope.";
            return;
        }

        string outDir = Path.Combine(ExportDir, "map_glb", _terrainManager.MapName);
        Directory.CreateDirectory(outDir);

        int exported = 0;
        foreach (var (tileX, tileY) in tiles)
        {
            string outPath = Path.Combine(outDir, $"{_terrainManager.MapName}_{tileX:D2}_{tileY:D2}.glb");
            MapGlbExporter.ExportTile(_terrainManager, _dataSource, _md5Index, tileX, tileY, outPath, includePlacements: true);
            exported++;
        }

        _statusMessage = $"Exported {exported} tile GLB(s) to: {outDir}";
    }

    private void RunTerrainExport()
    {
        try
        {
            switch (_terrainExportKind)
            {
                case TerrainExportKind.AlphaCurrentTileAtlas:
                    ExportAlphaCurrentTileAtlas();
                    break;
                case TerrainExportKind.AlphaCurrentTileChunksFolder:
                    ExportAlphaCurrentTileChunksFolder();
                    break;
                case TerrainExportKind.AlphaLoadedTilesFolder:
                    ExportAlphaTilesFolder(TerrainTileScope.LoadedTiles);
                    break;
                case TerrainExportKind.AlphaWholeMapFolder:
                    ExportAlphaTilesFolder(TerrainTileScope.WholeMap);
                    break;
                case TerrainExportKind.Heightmap257CurrentTilePerTile:
                    ExportHeightmap257CurrentTilePerTile();
                    break;
                case TerrainExportKind.Heightmap257LoadedTilesFolderPerTile:
                    ExportHeightmap257TilesFolderPerTile(TerrainTileScope.LoadedTiles);
                    break;
                case TerrainExportKind.Heightmap257WholeMapFolderPerMap:
                    ExportHeightmap257TilesFolderPerMap();
                    break;
                case TerrainExportKind.MccvCurrentTilePng:
                    ExportMccvCurrentTilePng();
                    break;
                case TerrainExportKind.MccvLoadedTilesFolder:
                    ExportMccvTilesFolder(TerrainTileScope.LoadedTiles);
                    break;
                case TerrainExportKind.MccvWholeMapFolder:
                    ExportMccvTilesFolder(TerrainTileScope.WholeMap);
                    break;
            }
        }
        catch (Exception ex)
        {
            _statusMessage = $"Terrain export failed: {ex.Message}";
        }
        finally
        {
            _terrainExportKind = TerrainExportKind.None;
        }
    }

    private void RunTerrainImport()
    {
        try
        {
            switch (_terrainImportKind)
            {
                case TerrainImportKind.AlphaFolder:
                    BeginAlphaFolderImport();
                    break;
                case TerrainImportKind.Heightmap257Folder:
                    BeginHeightmapFolderImport();
                    break;
                case TerrainImportKind.MccvFolder:
                    BeginMccvFolderImport();
                    break;
            }
        }
        catch (Exception ex)
        {
            _statusMessage = $"Terrain import failed: {ex.Message}";
        }
        finally
        {
            _terrainImportKind = TerrainImportKind.None;
        }
    }

    private static bool TryParseTileCoordsFromFileName(string filePath, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;
        string name = Path.GetFileNameWithoutExtension(filePath);

        var matches = Regex.Matches(name, @"\d+");
        if (matches.Count < 2)
            return false;

        var candidates = new List<int>(matches.Count);
        foreach (Match m in matches)
        {
            if (int.TryParse(m.Value, out int v) && v >= 0 && v < 64)
                candidates.Add(v);
        }

        if (candidates.Count < 2)
            return false;

        tileX = candidates[^2];
        tileY = candidates[^1];
        return true;
    }

    private static IEnumerable<(int tileX, int tileY)> ParseCustomTileList(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            yield break;

        var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var line in lines)
        {
            var parts = line.Split(new[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;
            if (!int.TryParse(parts[0], out int x)) continue;
            if (!int.TryParse(parts[1], out int y)) continue;
            if ((uint)x >= 64u || (uint)y >= 64u) continue;
            yield return (x, y);
        }
    }

    private void GetTerrainTileRange(out int startX, out int startY, out int endX, out int endY)
    {
        startX = Math.Clamp(Math.Min(_terrainTileRangeStartX, _terrainTileRangeEndX), 0, 63);
        startY = Math.Clamp(Math.Min(_terrainTileRangeStartY, _terrainTileRangeEndY), 0, 63);
        endX = Math.Clamp(Math.Max(_terrainTileRangeStartX, _terrainTileRangeEndX), 0, 63);
        endY = Math.Clamp(Math.Max(_terrainTileRangeStartY, _terrainTileRangeEndY), 0, 63);
    }

    private IEnumerable<(int tileX, int tileY)> EnumerateTerrainTileRange()
    {
        GetTerrainTileRange(out int startX, out int startY, out int endX, out int endY);
        for (int tileY = startY; tileY <= endY; tileY++)
        {
            for (int tileX = startX; tileX <= endX; tileX++)
                yield return (tileX, tileY);
        }
    }

    private void DrawTerrainTileScopeSelector(string idSuffix, bool includeCurrentTile)
    {
        int scope = (int)_terrainTileScope;
        TerrainTileScope previousScope = _terrainTileScope;
        if (includeCurrentTile)
            ImGui.RadioButton($"Current tile##{idSuffix}", ref scope, (int)TerrainTileScope.CurrentTile);
        ImGui.RadioButton($"Loaded tiles##{idSuffix}", ref scope, (int)TerrainTileScope.LoadedTiles);
        ImGui.RadioButton($"Whole map##{idSuffix}", ref scope, (int)TerrainTileScope.WholeMap);
        ImGui.RadioButton($"Custom list##{idSuffix}", ref scope, (int)TerrainTileScope.CustomList);
        ImGui.RadioButton($"Row/Column range##{idSuffix}", ref scope, (int)TerrainTileScope.RectRange);
        _terrainTileScope = (TerrainTileScope)scope;
        bool restoreScopeChanged = previousScope != _terrainTileScope;

        if (_terrainTileScope == TerrainTileScope.CustomList)
        {
            ImGui.TextDisabled("One tile per line: x y (or x,y)");
            if (ImGui.InputTextMultiline($"##customTiles_{idSuffix}", ref _terrainCustomTilesText, 8192, new Vector2(480, 160)))
                restoreScopeChanged = true;
        }
        else if (_terrainTileScope == TerrainTileScope.RectRange)
        {
            int startX = _terrainTileRangeStartX;
            int startY = _terrainTileRangeStartY;
            int endX = _terrainTileRangeEndX;
            int endY = _terrainTileRangeEndY;
            if (ImGui.InputInt($"Column Start##{idSuffix}", ref startX))
            {
                _terrainTileRangeStartX = Math.Clamp(startX, 0, 63);
                restoreScopeChanged = true;
            }
            if (ImGui.InputInt($"Row Start##{idSuffix}", ref startY))
            {
                _terrainTileRangeStartY = Math.Clamp(startY, 0, 63);
                restoreScopeChanged = true;
            }
            if (ImGui.InputInt($"Column End##{idSuffix}", ref endX))
            {
                _terrainTileRangeEndX = Math.Clamp(endX, 0, 63);
                restoreScopeChanged = true;
            }
            if (ImGui.InputInt($"Row End##{idSuffix}", ref endY))
            {
                _terrainTileRangeEndY = Math.Clamp(endY, 0, 63);
                restoreScopeChanged = true;
            }

            GetTerrainTileRange(out int normalizedStartX, out int normalizedStartY, out int normalizedEndX, out int normalizedEndY);
            int width = normalizedEndX - normalizedStartX + 1;
            int height = normalizedEndY - normalizedStartY + 1;
            ImGui.TextDisabled($"Range: columns {normalizedStartX}..{normalizedEndX}, rows {normalizedStartY}..{normalizedEndY} ({width * height} tile(s)).");
        }

        if (restoreScopeChanged)
            MarkTerrainWeakSignalRestoreDirty();
    }

    private IReadOnlyList<(int tileX, int tileY)> GetTileScopeList(TerrainTileScope scope)
    {
        if (scope == TerrainTileScope.CurrentTile)
        {
            var cam = GetCameraTile();
            return new List<(int, int)> { cam };
        }

        if (scope == TerrainTileScope.CustomList)
            return ParseCustomTileList(_terrainCustomTilesText).Distinct().ToList();

        if (scope == TerrainTileScope.RectRange)
            return EnumerateTerrainTileRange().ToList();

        if (_terrainManager != null)
        {
            if (scope == TerrainTileScope.LoadedTiles)
                return _terrainManager.LoadedTiles.ToList();

            if (scope == TerrainTileScope.WholeMap)
                return _terrainManager.Adapter.ExistingTiles.Select(idx => (idx / 64, idx % 64)).ToList();
        }

        if (_vlmTerrainManager != null)
        {
            if (scope == TerrainTileScope.LoadedTiles)
                return _vlmTerrainManager.Loader.TileCoords
                    .Where(t => _vlmTerrainManager.IsTileLoaded(t.tileX, t.tileY))
                    .ToList();

            if (scope == TerrainTileScope.WholeMap)
                return _vlmTerrainManager.Loader.TileCoords.ToList();
        }

        return new List<(int, int)>();
    }

    private IReadOnlyList<WoWViewer.Terrain.TerrainChunkData>? LoadTileChunksForExport(int tileX, int tileY)
    {
        if (_terrainManager != null)
        {
            return _terrainManager.GetOrLoadTileLoadResult(tileX, tileY).Chunks;
        }

        if (_vlmTerrainManager != null)
        {
            if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
                return tile.Chunks;

            if (_vlmTerrainManager.Loader.TileCoords.Contains((tileX, tileY)))
                return _vlmTerrainManager.Loader.LoadTile(tileX, tileY).Chunks;
        }

        return null;
    }

    private void ExportAlphaCurrentTileAtlas()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        string defaultName = $"tile_{tx}_{ty}_alpha.png";
        var picked = ShowSaveFileDialogSTA(
            "Save Alpha Mask Atlas",
            "PNG Files (*.png)|*.png|All Files (*.*)|*.*",
            ExportDir,
            defaultName);
        if (string.IsNullOrEmpty(picked))
            return;

        using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
        using (var fs = File.Create(picked))
            atlas.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
        _statusMessage = $"Exported: {picked}";
    }

    private void ExportAlphaCurrentTileChunksFolder()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        string? folder = ShowFolderDialogSTA(
            "Select output folder for chunk alpha masks",
            ExportDir,
            showNewFolderButton: true);
        if (string.IsNullOrEmpty(folder))
            return;

        using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
        var chunkImages = TerrainImageIo.BuildAlphaChunkImagesFromAtlas(atlas);
        foreach (var kvp in chunkImages)
        {
            var (cx, cy) = kvp.Key;
            string path = Path.Combine(folder, $"tile_{tx}_{ty}_chunk_{cx}_{cy}_alpha.png");
            using (var fs = File.Create(path))
                kvp.Value.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            kvp.Value.Dispose();
        }

        _statusMessage = $"Exported chunks: {folder}";
    }

    private void ExportAlphaTilesFolder(TerrainTileScope scope)
    {
        string? folder = ShowFolderDialogSTA(
            "Select output folder for tile alpha atlases",
            ExportDir,
            showNewFolderButton: true);
        if (string.IsNullOrEmpty(folder))
            return;

        var tiles = GetTileScopeList(scope);

        int written = 0;
        foreach (var (tx, ty) in tiles)
        {
            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null) continue;

            using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
            string path = Path.Combine(folder, $"tile_{tx}_{ty}_alpha.png");
            using (var fs = File.Create(path))
                atlas.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            written++;
        }

        _statusMessage = $"Exported {written} tiles: {folder}";
    }

    private void BeginAlphaFolderImport()
    {
        string? folder = ShowFolderDialogSTA(
            "Select folder containing tile alpha atlases",
            initialDir: null,
            showNewFolderButton: false);
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        _terrainImportFolder = folder;
        _showAlphaFolderImportScope = true;
    }

    private void DrawAlphaFolderImportScopeDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(520, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Import Alpha Masks", ref _showAlphaFolderImportScope, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.Text("Apply imported alpha masks to:");
        ImGui.Separator();
        DrawTerrainTileScopeSelector("AlphaImport", includeCurrentTile: true);

        ImGui.Separator();
        if (ImGui.Button("Import"))
        {
            ApplyAlphaFolderImport(_terrainImportFolder, _terrainTileScope);
            _terrainImportFolder = "";
            _showAlphaFolderImportScope = false;
        }
        ImGui.SameLine();
        if (ImGui.Button("Cancel"))
        {
            _terrainImportFolder = "";
            _showAlphaFolderImportScope = false;
        }

        ImGui.End();
    }

    private void ApplyAlphaFolderImport(string folder, TerrainTileScope scope)
    {
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        var targets = new HashSet<(int tileX, int tileY)>(GetTileScopeList(scope));
        if (targets.Count == 0)
        {
            _statusMessage = "No target tiles selected.";
            return;
        }

        if (scope == TerrainTileScope.WholeMap && _terrainManager != null)
            _terrainManager.LoadAllTiles();

        var renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
        {
            _statusMessage = "No terrain renderer.";
            return;
        }

        int applied = 0;
        foreach (var file in Directory.EnumerateFiles(folder, "*.png"))
        {
            if (!TryParseTileCoordsFromFileName(file, out int tx, out int ty))
                continue;

            if (!targets.Contains((tx, ty)))
                continue;

            if (_terrainManager != null && !_terrainManager.IsTileLoaded(tx, ty))
                continue;
            if (_vlmTerrainManager != null && !_vlmTerrainManager.IsTileLoaded(tx, ty))
                continue;

            using var atlas = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(file);
            var alphaShadow = TerrainImageIo.DecodeAlphaShadowArrayFromAtlas(atlas);
            renderer.ReplaceTileAlphaShadowArray(tx, ty, alphaShadow);
            applied++;
        }

        _statusMessage = $"Imported alpha masks for {applied} tiles.";
    }

    private void ExportMccvCurrentTilePng()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        string defaultName = $"tile_{tx}_{ty}_mccv.png";
        var picked = ShowSaveFileDialogSTA(
            "Save MCCV Tile PNG",
            "PNG Files (*.png)|*.png|All Files (*.*)|*.*",
            ExportDir,
            defaultName);
        if (string.IsNullOrEmpty(picked))
            return;

        using var image = TerrainMccvIo.BuildTileImage(chunks);
        using (var fs = File.Create(picked))
            image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

        _statusMessage = $"Exported: {picked}";
    }

    private void ExportMccvTilesFolder(TerrainTileScope scope)
    {
        string? folder = ShowFolderDialogSTA(
            "Select output folder for tile MCCV PNGs",
            ExportDir,
            showNewFolderButton: true);
        if (string.IsNullOrEmpty(folder))
            return;

        var tiles = GetTileScopeList(scope);

        int written = 0;
        foreach (var (tx, ty) in tiles)
        {
            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null)
                continue;

            using var image = TerrainMccvIo.BuildTileImage(chunks);
            string path = Path.Combine(folder, $"tile_{tx}_{ty}_mccv.png");
            using (var fs = File.Create(path))
                image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            written++;
        }

        _statusMessage = $"Exported {written} MCCV tiles: {folder}";
    }

    private void BeginMccvFolderImport()
    {
        string? folder = ShowFolderDialogSTA(
            "Select folder containing tile MCCV PNGs",
            initialDir: null,
            showNewFolderButton: false);
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        _terrainImportFolder = folder;
        _showMccvFolderImportScope = true;
    }

    private void DrawMccvFolderImportScopeDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(520, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Import MCCV", ref _showMccvFolderImportScope, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.Text("Apply imported MCCV to:");
        ImGui.Separator();
        DrawTerrainTileScopeSelector("MccvImport", includeCurrentTile: true);

        ImGui.Separator();
        ImGui.TextDisabled("PNG channels preserve raw MCCV bytes in file order for VLM/tooling compatibility.");
        if (ImGui.Button("Import"))
        {
            ApplyMccvFolderImport(_terrainImportFolder, _terrainTileScope);
            _terrainImportFolder = "";
            _showMccvFolderImportScope = false;
        }
        ImGui.SameLine();
        if (ImGui.Button("Cancel"))
        {
            _terrainImportFolder = "";
            _showMccvFolderImportScope = false;
        }

        ImGui.End();
    }

    private void ApplyMccvFolderImport(string folder, TerrainTileScope scope)
    {
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        var targets = new HashSet<(int tileX, int tileY)>(GetTileScopeList(scope));
        if (targets.Count == 0)
        {
            _statusMessage = "No target tiles selected.";
            return;
        }

        if (scope == TerrainTileScope.WholeMap && _terrainManager != null)
            _terrainManager.LoadAllTiles();

        int applied = 0;
        foreach (var file in Directory.EnumerateFiles(folder, "*.png"))
        {
            if (!TryParseTileCoordsFromFileName(file, out int tx, out int ty))
                continue;
            if (!targets.Contains((tx, ty)))
                continue;

            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null)
                continue;

            using var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(file);
            var newChunks = TerrainMccvIo.ApplyTileImageToChunks(chunks, image);
            if (_terrainManager != null)
                _terrainManager.ReplaceTileChunksAndRebuild(tx, ty, newChunks);
            else
                _vlmTerrainManager?.ReplaceTileChunksAndRebuild(tx, ty, newChunks);

            applied++;
        }

        _statusMessage = $"Imported MCCV for {applied} tiles.";
    }

    private void ExportHeightmap257CurrentTilePerTile()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        string defaultName = $"tile_{tx}_{ty}_height_257.png";
        var picked = ShowSaveFileDialogSTA(
            "Save Heightmap (257x257 L16)",
            "PNG Files (*.png)|*.png|All Files (*.*)|*.*",
            ExportDir,
            defaultName);
        if (string.IsNullOrEmpty(picked))
            return;

        var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
        using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, tile.MinHeight, tile.MaxHeight);
        using (var fs = File.Create(picked))
            img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

        var meta = new HeightmapMetadata
        {
            MinHeight = tile.MinHeight,
            MaxHeight = tile.MaxHeight,
            Normalization = "per_tile",
        };
        string jsonPath = Path.ChangeExtension(picked, ".json");
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));

        _statusMessage = $"Exported: {picked}";
    }

    private void ExportHeightmap257TilesFolderPerTile(TerrainTileScope scope)
    {
        string? folder = ShowFolderDialogSTA(
            "Select output folder for tile heightmaps",
            ExportDir,
            showNewFolderButton: true);
        if (string.IsNullOrEmpty(folder))
            return;

        var tiles = GetTileScopeList(scope);

        int written = 0;
        foreach (var (tx, ty) in tiles)
        {
            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null) continue;

            var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, tile.MinHeight, tile.MaxHeight);
            string pngPath = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.png");
            using (var fs = File.Create(pngPath))
                img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

            var meta = new HeightmapMetadata
            {
                MinHeight = tile.MinHeight,
                MaxHeight = tile.MaxHeight,
                Normalization = "per_tile",
            };
            string jsonPath = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.json");
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));
            written++;
        }

        _statusMessage = $"Exported {written} tiles: {folder}";
    }

    private void ExportHeightmap257TilesFolderPerMap()
    {
        string? folder = ShowFolderDialogSTA(
            "Select output folder for map-normalized tile heightmaps",
            ExportDir,
            showNewFolderButton: true);
        if (string.IsNullOrEmpty(folder))
            return;

        var tiles = GetTileScopeList(TerrainTileScope.WholeMap);
        if (tiles.Count == 0)
        {
            _statusMessage = "No tiles available.";
            return;
        }

        float gMin = float.MaxValue;
        float gMax = float.MinValue;
        foreach (var (tx, ty) in tiles)
        {
            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null) continue;
            var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            if (tile.MinHeight < gMin) gMin = tile.MinHeight;
            if (tile.MaxHeight > gMax) gMax = tile.MaxHeight;
        }
        if (gMin == float.MaxValue || gMax == float.MinValue)
        {
            gMin = 0f;
            gMax = 0f;
        }

        var mapMeta = new HeightmapMetadata
        {
            MinHeight = gMin,
            MaxHeight = gMax,
            Normalization = "per_map",
        };
        string mapJson = Path.Combine(folder, "heightmap_257_map.json");
        File.WriteAllText(mapJson, JsonSerializer.Serialize(mapMeta, new JsonSerializerOptions { WriteIndented = true }));

        int written = 0;
        foreach (var (tx, ty) in tiles)
        {
            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null) continue;

            var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, gMin, gMax);
            string pngPath = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.png");
            using (var fs = File.Create(pngPath))
                img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            written++;
        }

        _statusMessage = $"Exported {written} tiles (per-map): {folder}";
    }

    private void BeginHeightmapFolderImport()
    {
        string? folder = ShowFolderDialogSTA(
            "Select folder containing tile heightmaps",
            initialDir: null,
            showNewFolderButton: false);
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        _terrainImportFolder = folder;
        _showHeightmapFolderImportScope = true;
    }

    private void DrawHeightmapFolderImportScopeDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(520, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Import Heightmaps", ref _showHeightmapFolderImportScope, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.Text("Apply imported heightmaps to:");
        ImGui.Separator();
        DrawTerrainTileScopeSelector("HeightImport", includeCurrentTile: true);

        ImGui.Separator();
        if (ImGui.Button("Import"))
        {
            ApplyHeightmapFolderImport(_terrainImportFolder, _terrainTileScope);
            _terrainImportFolder = "";
            _showHeightmapFolderImportScope = false;
        }
        ImGui.SameLine();
        if (ImGui.Button("Cancel"))
        {
            _terrainImportFolder = "";
            _showHeightmapFolderImportScope = false;
        }

        ImGui.End();
    }

    private void ApplyHeightmapFolderImport(string folder, TerrainTileScope scope)
    {
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        var targets = new HashSet<(int tileX, int tileY)>(GetTileScopeList(scope));
        if (targets.Count == 0)
        {
            _statusMessage = "No target tiles selected.";
            return;
        }

        HeightmapMetadata? mapMeta = null;
        string mapMetaPath = Path.Combine(folder, "heightmap_257_map.json");
        if (File.Exists(mapMetaPath))
        {
            try
            {
                mapMeta = JsonSerializer.Deserialize<HeightmapMetadata>(File.ReadAllText(mapMetaPath));
            }
            catch
            {
                mapMeta = null;
            }
        }

        int applied = 0;
        foreach (var file in Directory.EnumerateFiles(folder, "*.png"))
        {
            if (!TryParseTileCoordsFromFileName(file, out int tx, out int ty))
                continue;
            if (!targets.Contains((tx, ty)))
                continue;

            if (_terrainManager != null && !_terrainManager.IsTileLoaded(tx, ty))
                continue;
            if (_vlmTerrainManager != null && !_vlmTerrainManager.IsTileLoaded(tx, ty))
                continue;

            HeightmapMetadata? meta = null;
            string perTileJson = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.json");
            if (File.Exists(perTileJson))
            {
                try
                {
                    meta = JsonSerializer.Deserialize<HeightmapMetadata>(File.ReadAllText(perTileJson));
                }
                catch
                {
                    meta = null;
                }
            }

            meta ??= mapMeta;
            if (meta == null)
                continue;

            using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.L16>(file);
            var tileHeights = TerrainHeightmapIo.DecodeL16(img, meta.MinHeight, meta.MaxHeight);

            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null)
                continue;

            var newChunks = TerrainHeightmapIo.ApplyHeightmap257ToChunks(chunks, tileHeights);
            if (_terrainManager != null)
                _terrainManager.ReplaceTileChunksAndRebuild(tx, ty, newChunks);
            else
                _vlmTerrainManager?.ReplaceTileChunksAndRebuild(tx, ty, newChunks);

            applied++;
        }

        _statusMessage = $"Imported heightmaps for {applied} tiles.";
    }

    private TerrainRenderer.TerrainChunkInfo? GetChunkClipboardTarget(TerrainRenderer renderer)
    {
        if (_chunkClipboardUseMouse && TryPickTerrainChunkUnderMouse(renderer, out var mouseChunk))
            return mouseChunk;
        return renderer.GetChunkInfoAt(_camera.Position.X, _camera.Position.Y);
    }

    private bool TryLockChunkPasteTarget(TerrainRenderer renderer)
    {
        if (!TryPickTerrainChunkUnderMouse(renderer, out var info))
            return false;

        _chunkClipboardLockedTargetKey = (info.TileX, info.TileY, info.ChunkX, info.ChunkY);
        _chunkClipboardStatus = $"Locked paste target: tile({info.TileX},{info.TileY}) chunk({info.ChunkX},{info.ChunkY})";
        return true;
    }

    private void ExecuteChunkClipboardCopy(TerrainRenderer renderer)
    {
        if (_selectedChunks.Count > 0)
        {
            CopySelectedChunks(renderer);
            return;
        }

        CopyChunkAtTarget(renderer);
    }

    private void ExecuteChunkClipboardPaste(TerrainRenderer renderer)
    {
        if (_chunkClipboardLockedTargetKey == null)
        {
            _chunkClipboardStatus = "Paste blocked: lock a paste target with Ctrl+LMB.";
            return;
        }

        if (_chunkClipboardSet != null)
            PasteClipboardSetAtTarget(renderer);
        else
            PasteChunkAtTarget(renderer);
    }

    private void InvertSelectedChunkHeights(TerrainRenderer renderer)
    {
        List<(int tileX, int tileY, int chunkX, int chunkY)> targets = GetChunkEditTargets(renderer);
        if (targets.Count == 0)
        {
            _chunkClipboardStatus = "Invert Z failed: select chunk(s) or point at a loaded chunk.";
            return;
        }

        var groupedTargets = targets
            .GroupBy(target => (target.tileX, target.tileY))
            .OrderBy(group => group.Key.tileX)
            .ThenBy(group => group.Key.tileY);

        int inverted = 0;
        int skipped = 0;

        foreach (var group in groupedTargets)
        {
            if (!TryGetTileChunksForEdit(group.Key.tileX, group.Key.tileY, out var chunks))
            {
                skipped += group.Count();
                continue;
            }

            var newChunks = chunks.ToList();
            var editedChunks = new List<(int chunkX, int chunkY)>();

            foreach (var target in group)
            {
                int idx = newChunks.FindIndex(chunk => chunk.ChunkX == target.chunkX && chunk.ChunkY == target.chunkY);
                if (idx < 0)
                {
                    skipped++;
                    continue;
                }

                var sourceChunk = newChunks[idx];
                var invertedHeights = new float[sourceChunk.Heights.Length];
                for (int i = 0; i < sourceChunk.Heights.Length; i++)
                    invertedHeights[i] = -sourceChunk.Heights[i];

                var invertedNormals = GenerateNormalsForChunk(sourceChunk, invertedHeights, sourceChunk.HoleMask);
                newChunks[idx] = CloneTerrainChunk(
                    sourceChunk,
                    heights: invertedHeights,
                    normals: invertedNormals);
                editedChunks.Add((target.chunkX, target.chunkY));
                inverted++;
            }

            if (editedChunks.Count > 0)
                ApplyEditedTileChunks(group.Key.tileX, group.Key.tileY, newChunks, editedChunks);
        }

        _chunkClipboardStatus = $"Inverted Z for {inverted} chunk(s)"
            + (skipped > 0 ? $" (skipped {skipped})" : string.Empty)
            + ". Save edited heightmaps from the chunk tool when you want reusable outputs.";
    }

    private void CopyChunkAtTarget(TerrainRenderer renderer)
    {
        var targetChunk = GetChunkClipboardTarget(renderer);
        if (!targetChunk.HasValue)
        {
            _chunkClipboardStatus = "Copy failed: no loaded chunk at target.";
            return;
        }

        var key = targetChunk.Value;

        if (!TryGetChunkData(key.TileX, key.TileY, key.ChunkX, key.ChunkY, out var chunk))
        {
            _chunkClipboardStatus = $"Copy failed: chunk data not available for tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY}).";
            return;
        }

        _chunkClipboard = new ChunkClipboard(chunk);
        _chunkClipboardSet = null;
        _chunkClipboardCopiedKey = (key.TileX, key.TileY, key.ChunkX, key.ChunkY);
        _chunkClipboardStatus = $"Copied: tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY})";
    }

    private List<(int tileX, int tileY, int chunkX, int chunkY)> GetChunkEditTargets(TerrainRenderer renderer)
    {
        if (_selectedChunks.Count > 0)
        {
            return _selectedChunks
                .OrderBy(chunk => chunk.tileX)
                .ThenBy(chunk => chunk.tileY)
                .ThenBy(chunk => chunk.chunkX)
                .ThenBy(chunk => chunk.chunkY)
                .ToList();
        }

        var targetChunk = GetChunkClipboardTarget(renderer);
        if (!targetChunk.HasValue)
            return new List<(int tileX, int tileY, int chunkX, int chunkY)>();

        var chunk = targetChunk.Value;
        return new List<(int tileX, int tileY, int chunkX, int chunkY)>
        {
            (chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY)
        };
    }

    private bool TryHandleChunkSelectionClick(TerrainRenderer renderer, bool shift)
    {
        if (!TryPickTerrainChunkUnderMouse(renderer, out var info))
            return false;

        var key = (info.TileX, info.TileY, info.ChunkX, info.ChunkY);
        if (shift)
        {
            if (!_selectedChunks.Add(key))
                _selectedChunks.Remove(key);
        }
        else
        {
            _selectedChunks.Clear();
            _selectedChunks.Add(key);
        }

        _chunkClipboardStatus = $"Selected {_selectedChunks.Count} chunk(s)";
        MarkTerrainWeakSignalRestoreDirty();
        return true;
    }

    private void CopySelectedChunks(TerrainRenderer renderer)
    {
        if (_selectedChunks.Count == 0)
            return;

        int minGlobalX = int.MaxValue;
        int minGlobalY = int.MaxValue;
        foreach (var (tx, ty, cx, cy) in _selectedChunks)
        {
            int gx = tx * 16 + cx;
            int gy = ty * 16 + cy;
            minGlobalX = Math.Min(minGlobalX, gx);
            minGlobalY = Math.Min(minGlobalY, gy);
        }

        var set = new ChunkClipboardSet(minGlobalX, minGlobalY);
        int copied = 0;

        foreach (var (stx, sty, scx, scy) in _selectedChunks)
        {
            if (!TryGetChunkData(stx, sty, scx, scy, out var chunk))
                continue;

            int gx = stx * 16 + scx;
            int gy = sty * 16 + scy;
            set.Chunks[(gx - minGlobalX, gy - minGlobalY)] = new ChunkClipboard(chunk);
            copied++;
        }

        if (copied == 0)
        {
            _chunkClipboardStatus = "Copy failed: selection chunks not available.";
            return;
        }

        _chunkClipboardSet = set;
        _chunkClipboard = null;
        _chunkClipboardCopiedKey = _selectedChunks.First();
        _chunkClipboardStatus = $"Copied selection: {copied} chunk(s).";
    }

    private void PasteClipboardSetAtTarget(TerrainRenderer renderer)
    {
        if (_chunkClipboardSet == null)
            return;

        if (_chunkClipboardLockedTargetKey == null)
        {
            _chunkClipboardStatus = "Paste blocked: lock a paste target with Ctrl+LMB.";
            return;
        }

        int targetGlobalX = _chunkClipboardLockedTargetKey.Value.tileX * 16 + _chunkClipboardLockedTargetKey.Value.chunkX;
        int targetGlobalY = _chunkClipboardLockedTargetKey.Value.tileY * 16 + _chunkClipboardLockedTargetKey.Value.chunkY;

        int maxDx = 0;
        int maxDy = 0;
        foreach (var key in _chunkClipboardSet.Chunks.Keys)
        {
            maxDx = Math.Max(maxDx, key.dx);
            maxDy = Math.Max(maxDy, key.dy);
        }
        int width = maxDx + 1;
        int height = maxDy + 1;

        int srcGridW = width * 16 + 1;
        int srcGridH = height * 16 + 1;
        var sum = new float[srcGridW * srcGridH];
        var count = new ushort[srcGridW * srcGridH];

        foreach (var kvp in _chunkClipboardSet.Chunks)
        {
            int baseX = kvp.Key.dx * 16;
            int baseY = kvp.Key.dy * 16;
            var clip = kvp.Value;
            if (clip.Heights == null || clip.Heights.Length < 145)
                continue;

            for (int i = 0; i < 145; i++)
            {
                GetChunkVertexPosition(i, out int row, out int col, out bool isInner);

                int hx;
                int hy;
                if (!isInner)
                {
                    hx = col * 2;
                    hy = (row / 2) * 2;
                }
                else
                {
                    hx = col * 2 + 1;
                    hy = (row / 2) * 2 + 1;
                }

                int px = baseX + hx;
                int py = baseY + hy;
                if ((uint)px >= (uint)srcGridW || (uint)py >= (uint)srcGridH)
                    continue;

                int idx = py * srcGridW + px;
                sum[idx] += clip.Heights[i];
                if (count[idx] != ushort.MaxValue)
                    count[idx]++;
            }
        }

        var srcGrid = new float[srcGridW * srcGridH];
        for (int i = 0; i < srcGrid.Length; i++)
            srcGrid[i] = count[i] > 0 ? (sum[i] / count[i]) : float.NaN;

        var rotatedGrid = RotateFloatGrid(srcGrid, srcGridW, srcGridH, _chunkClipboardSelectionRotation, out int rotGridW, out int rotGridH);

        float heightDelta = 0f;
        if (_chunkClipboardPasteRelativeHeights)
        {
            var sourceClip = _chunkClipboardSet.Chunks.TryGetValue((0, 0), out var origin) ? origin : _chunkClipboardSet.Chunks.Values.First();
            float sourceRef = ComputeAverageHeight(sourceClip.Heights);
            if (TryGetChunkData(_chunkClipboardLockedTargetKey.Value.tileX, _chunkClipboardLockedTargetKey.Value.tileY,
                    _chunkClipboardLockedTargetKey.Value.chunkX, _chunkClipboardLockedTargetKey.Value.chunkY, out var targetChunkData))
            {
                float targetRef = ComputeAverageHeight(targetChunkData.Heights);
                heightDelta = targetRef - sourceRef;
            }
        }

        static (int dx, int dy) RotateInBox(int dx, int dy, int width, int height, int rot)
        {
            rot = ((rot % 4) + 4) % 4;
            return rot switch
            {
                0 => (dx, dy),
                1 => (height - 1 - dy, dx),
                2 => (width - 1 - dx, height - 1 - dy),
                3 => (dy, width - 1 - dx),
                _ => (dx, dy)
            };
        }

        var perTile = new Dictionary<(int tileX, int tileY), List<(int chunkX, int chunkY, int rdx, int rdy, ChunkClipboard clip)>>();
        foreach (var kvp in _chunkClipboardSet.Chunks)
        {
            var (rdx, rdy) = RotateInBox(kvp.Key.dx, kvp.Key.dy, width, height, _chunkClipboardSelectionRotation);
            int destGlobalX = targetGlobalX + rdx;
            int destGlobalY = targetGlobalY + rdy;
            if (destGlobalX < 0 || destGlobalX >= 64 * 16 || destGlobalY < 0 || destGlobalY >= 64 * 16)
                continue;

            int tileX = destGlobalX / 16;
            int tileY = destGlobalY / 16;
            int chunkX = destGlobalX % 16;
            int chunkY = destGlobalY % 16;

            var tkey = (tileX, tileY);
            if (!perTile.TryGetValue(tkey, out var list))
            {
                list = new List<(int, int, int, int, ChunkClipboard)>();
                perTile[tkey] = list;
            }

            list.Add((chunkX, chunkY, rdx, rdy, kvp.Value));
        }

        int pasted = 0;
        int skipped = 0;

        foreach (var entry in perTile)
        {
            var (tileX, tileY) = entry.Key;
            if (!TryGetTileChunksForEdit(tileX, tileY, out var chunks))
            {
                skipped += entry.Value.Count;
                continue;
            }

            var newChunks = chunks.ToList();

            foreach (var (chunkX, chunkY, rdx, rdy, clip) in entry.Value)
            {
                int idx = newChunks.FindIndex(c => c.ChunkX == chunkX && c.ChunkY == chunkY);
                if (idx < 0)
                {
                    skipped++;
                    continue;
                }

                var target = newChunks[idx];
                bool layersMatch = AreLayersCompatible(target.Layers, clip.Layers);

                int baseX = rdx * 16;
                int baseY = rdy * 16;
                var heights = new float[145];
                for (int i = 0; i < 145; i++)
                {
                    GetChunkVertexPosition(i, out int row, out int col, out bool isInner);

                    int hx;
                    int hy;
                    if (!isInner)
                    {
                        hx = col * 2;
                        hy = (row / 2) * 2;
                    }
                    else
                    {
                        hx = col * 2 + 1;
                        hy = (row / 2) * 2 + 1;
                    }

                    int px = baseX + hx;
                    int py = baseY + hy;
                    if ((uint)px >= (uint)rotGridW || (uint)py >= (uint)rotGridH)
                    {
                        heights[i] = target.Heights[i];
                        continue;
                    }

                    float v = rotatedGrid[py * rotGridW + px];
                    heights[i] = float.IsNaN(v) ? target.Heights[i] : (v + heightDelta);
                }

                int holeMask = RotateHoleMask(clip.HoleMask, _chunkClipboardSelectionRotation);
                var normals = GenerateNormalsForChunk(target, heights, holeMask);

                var layersToUse = target.Layers;
                var alphaToUse = target.AlphaMaps;
                byte[]? shadowToUse = target.ShadowMap;

                if (_chunkClipboardIncludeTextures)
                {
                    layersToUse = clip.Layers;
                    if (_chunkClipboardIncludeAlphaShadow)
                    {
                        alphaToUse = CloneAlphaMaps(clip.AlphaMaps);
                        shadowToUse = clip.ShadowMap != null ? (byte[])clip.ShadowMap.Clone() : null;
                    }
                }
                else if (_chunkClipboardIncludeAlphaShadow && layersMatch)
                {
                    alphaToUse = CloneAlphaMaps(clip.AlphaMaps);
                    shadowToUse = clip.ShadowMap != null ? (byte[])clip.ShadowMap.Clone() : null;
                }

                var pastedChunk = CloneTerrainChunk(
                    target,
                    heights: heights,
                    normals: normals,
                    holeMask: holeMask,
                    layers: layersToUse,
                    alphaMaps: alphaToUse,
                    shadowMap: shadowToUse,
                    mccvColors: target.MccvColors);

                newChunks[idx] = pastedChunk;
                pasted++;
            }

            ApplyEditedTileChunks(tileX, tileY, newChunks, entry.Value.Select(value => (value.chunkX, value.chunkY)));
        }

        _chunkClipboardStatus = $"Pasted {pasted} chunk(s)" + (skipped > 0 ? $" (skipped {skipped})" : "") + $". Rotation={_chunkClipboardSelectionRotation * 90}°";
    }

    private static float[] RotateFloatGrid(float[] src, int w, int h, int rot, out int outW, out int outH)
    {
        rot = ((rot % 4) + 4) % 4;
        if (rot == 0)
        {
            outW = w;
            outH = h;
            return src;
        }

        if (rot == 2)
        {
            outW = w;
            outH = h;
            var dst = new float[w * h];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int sx = x;
                    int sy = y;
                    int dx = (w - 1 - sx);
                    int dy = (h - 1 - sy);
                    dst[dy * w + dx] = src[sy * w + sx];
                }
            }
            return dst;
        }

        outW = h;
        outH = w;
        var outGrid = new float[outW * outH];

        if (rot == 1)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int dx = (h - 1 - y);
                    int dy = x;
                    outGrid[dy * outW + dx] = src[y * w + x];
                }
            }
        }
        else
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int dx = y;
                    int dy = (w - 1 - x);
                    outGrid[dy * outW + dx] = src[y * w + x];
                }
            }
        }

        return outGrid;
    }

    private static int RotateHoleMask(int holeMask, int rot)
    {
        rot = ((rot % 4) + 4) % 4;
        if (rot == 0 || holeMask == 0)
            return holeMask;

        int GetBit(int x, int y) => (holeMask >> (y * 4 + x)) & 1;
        int SetBit(int x, int y) => 1 << (y * 4 + x);

        int outMask = 0;
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                if (GetBit(x, y) == 0)
                    continue;

                int rx;
                int ry;
                switch (rot)
                {
                    case 1:
                        rx = 3 - y;
                        ry = x;
                        break;
                    case 2:
                        rx = 3 - x;
                        ry = 3 - y;
                        break;
                    case 3:
                        rx = y;
                        ry = 3 - x;
                        break;
                    default:
                        rx = x;
                        ry = y;
                        break;
                }

                outMask |= SetBit(rx, ry);
            }
        }

        return outMask;
    }

    private static Vector3[] GenerateNormalsForChunk(Terrain.TerrainChunkData chunk, float[] heights, int holeMask)
    {
        var positions = new Vector3[145];
        for (int i = 0; i < 145; i++)
            positions[i] = GetChunkVertexWorldPosition(chunk, heights, i);

        var indices = BuildChunkIndices(holeMask);
        var accum = new Vector3[145];

        for (int t = 0; t + 2 < indices.Length; t += 3)
        {
            int i0 = indices[t + 0];
            int i1 = indices[t + 1];
            int i2 = indices[t + 2];

            var p0 = positions[i0];
            var p1 = positions[i1];
            var p2 = positions[i2];

            var e1 = p1 - p0;
            var e2 = p2 - p0;
            var n = Vector3.Cross(e1, e2);
            float lenSq = n.LengthSquared();
            if (lenSq < 1e-10f)
                continue;

            n = Vector3.Normalize(n);
            accum[i0] += n;
            accum[i1] += n;
            accum[i2] += n;
        }

        var normals = new Vector3[145];
        for (int i = 0; i < 145; i++)
        {
            var n = accum[i];
            float lenSq = n.LengthSquared();
            normals[i] = lenSq > 1e-10f ? Vector3.Normalize(n) : Vector3.UnitZ;
        }

        return normals;
    }

    private static Vector3 GetChunkVertexWorldPosition(Terrain.TerrainChunkData chunk, float[] heights, int index)
    {
        GetChunkVertexLocalPosition(index, out float x, out float y);

        float z = (index < heights.Length) ? heights[index] : 0f;
        float wx = chunk.WorldPosition.X - y;
        float wy = chunk.WorldPosition.Y - x;
        return new Vector3(wx, wy, z);
    }

    private static void GetChunkVertexLocalPosition(int index, out float x, out float y)
    {
        GetChunkVertexPosition(index, out int row, out int col, out bool isInner);

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        if (!isInner)
        {
            x = col * subCellSize;
            y = (row / 2) * subCellSize;
            return;
        }

        x = (col + 0.5f) * subCellSize;
        y = (row / 2 + 0.5f) * subCellSize;
    }

    private static int OuterIndex(int outerRow, int outerCol) => outerRow * 17 + outerCol;
    private static int InnerIndex(int innerRow, int innerCol) => innerRow * 17 + 9 + innerCol;

    private static int[] BuildChunkIndices(int holeMask)
    {
        var indices = new List<int>(256 * 3);

        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                if (holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << (holeY * 4 + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                int tl = OuterIndex(cellY, cellX);
                int tr = OuterIndex(cellY, cellX + 1);
                int bl = OuterIndex(cellY + 1, cellX);
                int br = OuterIndex(cellY + 1, cellX + 1);
                int center = InnerIndex(cellY, cellX);

                indices.Add(center);
                indices.Add(tr);
                indices.Add(tl);

                indices.Add(center);
                indices.Add(br);
                indices.Add(tr);

                indices.Add(center);
                indices.Add(bl);
                indices.Add(br);

                indices.Add(center);
                indices.Add(tl);
                indices.Add(bl);
            }
        }

        return indices.ToArray();
    }

    private static Terrain.TerrainChunkData CloneTerrainChunk(
        Terrain.TerrainChunkData source,
        float[]? heights = null,
        Vector3[]? normals = null,
        int? holeMask = null,
        WoWViewer.Terrain.TerrainLayer[]? layers = null,
        Dictionary<int, byte[]>? alphaMaps = null,
        byte[]? shadowMap = null,
        byte[]? mccvColors = null)
        => new()
        {
            McinIndex = source.McinIndex,
            TileX = source.TileX,
            TileY = source.TileY,
            ChunkX = source.ChunkX,
            ChunkY = source.ChunkY,
            Heights = heights ?? source.Heights,
            Normals = normals ?? source.Normals,
            HoleMask = holeMask ?? source.HoleMask,
            Layers = layers ?? source.Layers,
            AlphaMaps = alphaMaps ?? source.AlphaMaps,
            ShadowMap = shadowMap ?? source.ShadowMap,
            MccvColors = mccvColors ?? source.MccvColors,
            Liquid = source.Liquid,
            WorldPosition = source.WorldPosition,
            AreaId = source.AreaId,
            McnkFlags = source.McnkFlags,
            AlphaSourceFlags = source.AlphaSourceFlags,
        };

    private static List<Terrain.TerrainChunkData> CloneTerrainChunkList(IReadOnlyList<Terrain.TerrainChunkData> chunks)
    {
        var cloned = new List<Terrain.TerrainChunkData>(chunks.Count);
        foreach (var chunk in chunks)
        {
            cloned.Add(CloneTerrainChunk(
                chunk,
                heights: chunk.Heights?.ToArray(),
                normals: chunk.Normals?.ToArray(),
                layers: chunk.Layers?.ToArray(),
                alphaMaps: CloneChunkAlphaMaps(chunk.AlphaMaps),
                shadowMap: chunk.ShadowMap?.ToArray(),
                mccvColors: chunk.MccvColors?.ToArray()));
        }

        return cloned;
    }

    private static Dictionary<int, byte[]> CloneChunkAlphaMaps(Dictionary<int, byte[]>? alphaMaps)
    {
        if (alphaMaps == null || alphaMaps.Count == 0)
            return new Dictionary<int, byte[]>();

        var cloned = new Dictionary<int, byte[]>(alphaMaps.Count);
        foreach (var entry in alphaMaps)
            cloned[entry.Key] = entry.Value?.ToArray() ?? Array.Empty<byte>();

        return cloned;
    }

    private void ResetTerrainWeakSignalRestoreSessionState(bool preserveToggle)
    {
        DetachTerrainWeakSignalRestoreHooks();
        _terrainWeakSignalOriginalTiles.Clear();
        _terrainWeakSignalAppliedPlans.Clear();
        _terrainWeakSignalApplyingTiles.Clear();
        _terrainWeakSignalWdlMapName = null;
        _terrainWeakSignalWdlData = null;
        _terrainWeakSignalRestoreLastCameraTile = null;
        _terrainWeakSignalRestoreNeedsRefresh = true;

        if (!preserveToggle)
            _terrainWeakSignalRestoreEnabled = false;

        _terrainWeakSignalRestoreStatus = string.Empty;
    }

    private void RefreshTerrainWeakSignalRestoreHooks()
    {
        if (!ReferenceEquals(_terrainWeakSignalHookedTerrainManager, _terrainManager))
        {
            if (_terrainWeakSignalHookedTerrainManager != null)
                _terrainWeakSignalHookedTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

            _terrainWeakSignalHookedTerrainManager = _terrainManager;
            if (_terrainWeakSignalHookedTerrainManager != null)
                _terrainWeakSignalHookedTerrainManager.OnTileLoaded += OnTerrainWeakSignalTileLoaded;
        }

        if (!ReferenceEquals(_terrainWeakSignalHookedVlmTerrainManager, _vlmTerrainManager))
        {
            if (_terrainWeakSignalHookedVlmTerrainManager != null)
                _terrainWeakSignalHookedVlmTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

            _terrainWeakSignalHookedVlmTerrainManager = _vlmTerrainManager;
            if (_terrainWeakSignalHookedVlmTerrainManager != null)
                _terrainWeakSignalHookedVlmTerrainManager.OnTileLoaded += OnTerrainWeakSignalTileLoaded;
        }
    }

    private void DetachTerrainWeakSignalRestoreHooks()
    {
        if (_terrainWeakSignalHookedTerrainManager != null)
            _terrainWeakSignalHookedTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

        if (_terrainWeakSignalHookedVlmTerrainManager != null)
            _terrainWeakSignalHookedVlmTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

        _terrainWeakSignalHookedTerrainManager = null;
        _terrainWeakSignalHookedVlmTerrainManager = null;
    }

    private bool SetTerrainWeakSignalRestoreEnabled(bool enabled)
    {
        if (_terrainWeakSignalRestoreEnabled == enabled)
            return false;

        _terrainWeakSignalRestoreEnabled = enabled;
        _terrainWeakSignalRestoreNeedsRefresh = true;
        _terrainWeakSignalRestoreLastCameraTile = null;
        if (enabled)
        {
            RefreshTerrainWeakSignalRestoreHooks();
            RefreshTerrainWeakSignalRestoreForLoadedTiles();
        }
        else
        {
            RestoreAllTerrainWeakSignalTiles();
            _terrainWeakSignalRestoreStatus = "Weak-signal terrain restore disabled.";
        }

        return true;
    }

    private void MarkTerrainWeakSignalRestoreDirty()
    {
        _terrainWeakSignalRestoreNeedsRefresh = true;
    }

    private bool TerrainWeakSignalRestoreUsesWorkbenchScope()
        => _terrainTileScope == TerrainTileScope.CurrentTile
            || _terrainTileScope == TerrainTileScope.CustomList
            || _terrainTileScope == TerrainTileScope.RectRange;

    private IReadOnlyList<(int tileX, int tileY)> GetTerrainWeakSignalRestoreScopedTiles()
    {
        if (_selectedChunks.Count > 0)
        {
            return _selectedChunks
                .Select(chunk => (chunk.tileX, chunk.tileY))
                .Distinct()
                .OrderBy(tile => tile.tileX)
                .ThenBy(tile => tile.tileY)
                .ToList();
        }

        if (TerrainWeakSignalRestoreUsesWorkbenchScope())
            return GetTileScopeList(_terrainTileScope);

        if (_terrainWeakSignalRestoreAllLoadedTiles)
            return GetTileScopeList(TerrainTileScope.LoadedTiles);

        return new List<(int tileX, int tileY)> { GetCameraTile() };
    }

    private bool HasTerrainWeakSignalScopedChunkSelectionForTile(int tileX, int tileY)
        => _selectedChunks.Count > 0 && _selectedChunks.Any(chunk => chunk.tileX == tileX && chunk.tileY == tileY);

    private bool IsTerrainWeakSignalRestoreTileInScope(int tileX, int tileY)
    {
        if (_selectedChunks.Count > 0)
            return HasTerrainWeakSignalScopedChunkSelectionForTile(tileX, tileY);

        if (TerrainWeakSignalRestoreUsesWorkbenchScope())
            return GetTerrainWeakSignalRestoreScopedTiles().Contains((tileX, tileY));

        if (_terrainWeakSignalRestoreAllLoadedTiles)
            return true;

        return GetCameraTile() == (tileX, tileY);
    }

    private bool IsTerrainWeakSignalRestoreChunkInScope(int tileX, int tileY, int chunkX, int chunkY)
    {
        if (_selectedChunks.Count > 0)
            return _selectedChunks.Contains((tileX, tileY, chunkX, chunkY));

        return IsTerrainWeakSignalRestoreTileInScope(tileX, tileY);
    }

    private string GetTerrainWeakSignalRestoreScopeSummary()
    {
        return "camera tile + 4 neighbors";
    }

    private void UpdateTerrainWeakSignalRestoreForCamera()
    {
        if (!_terrainWeakSignalRestoreEnabled)
            return;

        var cameraTile = GetCameraTile();
        if (_terrainWeakSignalRestoreNeedsRefresh
            || _terrainWeakSignalRestoreLastCameraTile == null
            || _terrainWeakSignalRestoreLastCameraTile.Value != cameraTile)
        {
            _terrainWeakSignalRestoreLastCameraTile = cameraTile;
            RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _terrainWeakSignalRestoreNeedsRefresh = false;
        }
    }

    private void RefreshTerrainWeakSignalRestoreForLoadedTiles()
    {
        if (!_terrainWeakSignalRestoreEnabled)
            return;

        var loadedKeys = new HashSet<(int tileX, int tileY)>();

        if (_terrainManager != null)
        {
            foreach (var (tileX, tileY) in _terrainManager.LoadedTiles.ToList())
            {
                loadedKeys.Add((tileX, tileY));
                if (_terrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
                {
                    if (ShouldApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks))
                        ApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks);
                    else
                        RestoreTerrainWeakSignalTile((tileX, tileY), clearCache: true);
                }
            }
        }

        if (_vlmTerrainManager != null)
        {
            foreach (var (tileX, tileY) in _vlmTerrainManager.LoadedTiles.ToList())
            {
                loadedKeys.Add((tileX, tileY));
                if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
                {
                    if (ShouldApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks))
                        ApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks);
                    else
                        RestoreTerrainWeakSignalTile((tileX, tileY), clearCache: true);
                }
            }
        }

        foreach (var key in _terrainWeakSignalOriginalTiles.Keys.Where(key => !loadedKeys.Contains(key)).ToList())
        {
            _terrainWeakSignalOriginalTiles.Remove(key);
            _terrainWeakSignalAppliedPlans.Remove(key);
        }
    }

    private void RestoreAllTerrainWeakSignalTiles()
    {
        foreach (var key in _terrainWeakSignalOriginalTiles.Keys.ToList())
            RestoreTerrainWeakSignalTile(key, clearCache: true);

        _terrainWeakSignalOriginalTiles.Clear();
        _terrainWeakSignalAppliedPlans.Clear();
        _terrainWeakSignalApplyingTiles.Clear();
    }

    private void OnTerrainWeakSignalTileLoaded(int tileX, int tileY, WoWViewer.Terrain.TileLoadResult result)
    {
        if (!_terrainWeakSignalRestoreEnabled || result.Chunks.Count == 0)
            return;

        if (ShouldApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks))
            ApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks);
    }

    private bool ShouldApplyTerrainWeakSignalRestoreToTile(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        if (sourceChunks.Count == 0)
            return false;

        var cameraTile = GetCameraTile();
        int deltaX = Math.Abs(tileX - cameraTile.tileX);
        int deltaY = Math.Abs(tileY - cameraTile.tileY);
        if (deltaX + deltaY > 1)
            return false;

        var key = (tileX, tileY);
        IReadOnlyList<Terrain.TerrainChunkData> baseChunks = _terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks)
            ? originalChunks
            : sourceChunks;

        return HasTerrainWeakSignalRestoreWholeTileEvidence(tileX, tileY, baseChunks);
    }

    private bool IsTerrainWeakSignalRestoreCandidateHeightmap(TerrainHeightmapIo.TileHeightmap257 tileHeightmap)
    {
        GetTerrainWeakSignalRestoreCandidateRange(out float minHeight, out float maxHeight);
        return tileHeightmap.MinHeight >= minHeight && tileHeightmap.MaxHeight <= maxHeight;
    }

    private bool IsTerrainWeakSignalCandidateRange(float minHeight, float maxHeight)
    {
        GetTerrainWeakSignalRestoreCandidateRange(out float candidateMinHeight, out float candidateMaxHeight);
        return minHeight >= candidateMinHeight && maxHeight <= candidateMaxHeight;
    }

    private bool HasTerrainWeakSignalRestoreCandidateChunks(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        if (sourceChunks.Count == 0)
            return false;

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(sourceChunks);
        bool tileWeakSignalCandidate = IsTerrainWeakSignalRestoreCandidateHeightmap(tileHeightmap);

        for (int index = 0; index < sourceChunks.Count; index++)
        {
            Terrain.TerrainChunkData chunk = sourceChunks[index];
            if (!IsTerrainWeakSignalRestoreChunkInScope(tileX, tileY, chunk.ChunkX, chunk.ChunkY))
                continue;

            if (!TryBuildTerrainWeakSignalTextureGuidance(chunk, out TerrainWeakSignalTextureGuidance? textureGuidance) || textureGuidance == null)
                continue;

            if (!_terrainWeakSignalRestoreUseAutoFactor)
                return true;

            if (TryEstimateTerrainWeakSignalRestoreFactorForChunk(tileX, tileY, chunk, tileHeightmap, tileWeakSignalCandidate, textureGuidance, out _, out _))
                return true;
        }

        return false;
    }

    private bool HasTerrainWeakSignalRestoreWholeTileEvidence(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        if (sourceChunks.Count == 0)
            return false;

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(sourceChunks);
        if (IsTerrainWeakSignalRestoreCandidateHeightmap(tileHeightmap))
            return true;

        return TryGetTerrainWeakSignalTileObservedRange(sourceChunks, out _, out _, out _, out _);
    }

    private void GetTerrainWeakSignalRestoreCandidateRange(out float minHeight, out float maxHeight)
    {
        minHeight = ClampTerrainWeakSignalRestoreZ(_terrainWeakSignalRestoreCandidateMinHeight);
        maxHeight = ClampTerrainWeakSignalRestoreZ(_terrainWeakSignalRestoreCandidateMaxHeight);
        if (minHeight > maxHeight)
            (minHeight, maxHeight) = (maxHeight, minHeight);
    }

    private static float ClampTerrainWeakSignalRestoreZ(float value)
        => Math.Clamp(value, TerrainWeakSignalRestoreMinZLimit, TerrainWeakSignalRestoreMaxZLimit);

    private void ApplyTerrainWeakSignalRestoreToTile(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        var key = (tileX, tileY);
        if (_terrainWeakSignalApplyingTiles.Contains(key) || sourceChunks.Count == 0)
            return;

        bool hasOriginal = _terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks);
        IReadOnlyList<Terrain.TerrainChunkData> baseChunks = hasOriginal
            ? originalChunks!
            : sourceChunks;

        if (!TryBuildTerrainWeakSignalRestoredChunks(tileX, tileY, baseChunks, out var restoredChunks, out int planSignature, out string reason))
            return;

        if (hasOriginal
            && _terrainWeakSignalAppliedPlans.TryGetValue(key, out int appliedPlanSignature)
            && appliedPlanSignature == planSignature)
        {
            return;
        }

        if (!hasOriginal)
            _terrainWeakSignalOriginalTiles[key] = CloneTerrainChunkList(sourceChunks);

        _terrainWeakSignalApplyingTiles.Add(key);
        try
        {
            if (_terrainManager != null)
                _terrainManager.ReplaceTileChunksAndRebuild(tileX, tileY, restoredChunks);
            else
                _vlmTerrainManager?.ReplaceTileChunksAndRebuild(tileX, tileY, restoredChunks);

            _terrainWeakSignalAppliedPlans[key] = planSignature;
            _terrainWeakSignalRestoreStatus = $"Weak-signal restore applied to tile ({tileX}, {tileY}) using {reason}.";
        }
        finally
        {
            _terrainWeakSignalApplyingTiles.Remove(key);
        }
    }

    private void RestoreTerrainWeakSignalTile((int tileX, int tileY) key, bool clearCache)
    {
        if (!_terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks) || _terrainWeakSignalApplyingTiles.Contains(key))
            return;

        _terrainWeakSignalApplyingTiles.Add(key);
        try
        {
            if (_terrainManager != null)
                _terrainManager.ReplaceTileChunksAndRebuild(key.tileX, key.tileY, CloneTerrainChunkList(originalChunks));
            else
                _vlmTerrainManager?.ReplaceTileChunksAndRebuild(key.tileX, key.tileY, CloneTerrainChunkList(originalChunks));
        }
        finally
        {
            _terrainWeakSignalApplyingTiles.Remove(key);
        }

        if (clearCache)
        {
            _terrainWeakSignalOriginalTiles.Remove(key);
            _terrainWeakSignalAppliedPlans.Remove(key);
        }
    }

    private bool TryBuildTerrainWeakSignalRestoredChunks(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks,
        out List<Terrain.TerrainChunkData> restoredChunks,
        out int planSignature,
        out string reason)
    {
        return TryBuildTerrainWeakSignalRestoredWholeTile(tileX, tileY, sourceChunks, out restoredChunks, out planSignature, out reason);
    }

    private bool TryBuildTerrainWeakSignalRestoredWholeTile(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks,
        out List<Terrain.TerrainChunkData> restoredChunks,
        out int planSignature,
        out string reason)
    {
        restoredChunks = new List<Terrain.TerrainChunkData>();
        planSignature = 0;
        reason = string.Empty;

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(sourceChunks);
        bool tileWeakSignalCandidate = IsTerrainWeakSignalRestoreCandidateHeightmap(tileHeightmap);
        bool hasPartialSignal = TryGetTerrainWeakSignalTileObservedRange(sourceChunks, out float observedMinHeight, out float observedMaxHeight, out int observedSignalCount, out bool usedTextureGuidance);
        if (!tileWeakSignalCandidate && !hasPartialSignal)
            return false;

        float factor;
        if (_terrainWeakSignalRestoreUseAutoFactor)
        {
            if (tileWeakSignalCandidate)
            {
                if (!TryEstimateTerrainWeakSignalRestoreFactor(tileX, tileY, tileHeightmap, out factor, out reason))
                    return false;
            }
            else if (!TryEstimateTerrainWeakSignalRestoreFactorForObservedRange(tileX, tileY, observedMinHeight, observedMaxHeight, out factor, out reason))
            {
                return false;
            }
        }
        else
        {
            factor = Math.Clamp(_terrainWeakSignalRestoreManualFactor, 1f, TerrainWeakSignalRestoreMaxFactor);
            if (factor <= 1.001f)
                return false;

            reason = "manual scale";
        }

        float? globalMaxHeight = TryGetTerrainWeakSignalGlobalMaxHeight(tileX, tileY, out float resolvedGlobalMaxHeight)
            ? resolvedGlobalMaxHeight
            : null;

        float anchorHeight = tileHeightmap.MinHeight < 0f ? tileHeightmap.MinHeight : 0f;
        bool preserveNegativeFloor = anchorHeight < 0f;
        float[] restoredHeightmap = new float[tileHeightmap.Heights.Length];
        for (int index = 0; index < tileHeightmap.Heights.Length; index++)
        {
            float sourceHeight = tileHeightmap.Heights[index];
            float restoredHeight = anchorHeight + ((sourceHeight - anchorHeight) * factor);
            if (!preserveNegativeFloor && restoredHeight < 0f)
                restoredHeight = 0f;
            if (globalMaxHeight.HasValue && restoredHeight > globalMaxHeight.Value)
                restoredHeight = globalMaxHeight.Value;

            restoredHeightmap[index] = restoredHeight;
        }

        List<Terrain.TerrainChunkData> wholeTileRestoredChunks = TerrainHeightmapIo.ApplyHeightmap257ToChunks(sourceChunks, restoredHeightmap);

        if (_terrainWeakSignalRestoreUseTextureSubdivisions)
        {
            List<Terrain.TerrainChunkData> maskedChunks = CloneTerrainChunkList(sourceChunks);
            var maskedPlanHash = new HashCode();
            int maskedChunkCount = 0;
            int guidedCellCount = 0;

            for (int index = 0; index < sourceChunks.Count; index++)
            {
                Terrain.TerrainChunkData chunk = sourceChunks[index];
                if (!TryBuildTerrainWeakSignalTextureGuidance(chunk, out TerrainWeakSignalTextureGuidance? textureGuidance) || textureGuidance == null)
                    continue;

                float[]? vertexWeights = BuildTerrainWeakSignalTextureGuidanceVertexWeights(textureGuidance);
                Terrain.TerrainChunkData restoredChunk = wholeTileRestoredChunks[index];

                float[] restoredHeights = BlendTerrainWeakSignalMaskedChunkHeights(chunk.Heights, restoredChunk.Heights, vertexWeights);
                Vector3[] restoredNormals = GenerateNormalsForChunk(chunk, restoredHeights, chunk.HoleMask);
                maskedChunks[index] = CloneTerrainChunk(chunk, heights: restoredHeights, normals: restoredNormals);

                maskedPlanHash.Add(chunk.ChunkX);
                maskedPlanHash.Add(chunk.ChunkY);
                maskedPlanHash.Add(textureGuidance.SelectedCellCount);
                maskedPlanHash.Add(GetTerrainWeakSignalSelectedMaskHash(textureGuidance.SelectedMask));
                maskedPlanHash.Add((int)MathF.Round(factor * 1000f));

                maskedChunkCount++;
                guidedCellCount += textureGuidance.SelectedCellCount;
            }

            if (maskedChunkCount > 0)
            {
                restoredChunks = maskedChunks;
                planSignature = maskedPlanHash.ToHashCode();
                string maskedSignalSummary = tileWeakSignalCandidate
                    ? "whole-tile weak range"
                    : $"{observedSignalCount} partial weak signal source(s){(usedTextureGuidance ? ", cell-guided" : string.Empty)}";
                reason += $", whole-tile factor clamped to {maskedChunkCount} cell-guided chunk(s) / {guidedCellCount} weak sub-cell(s) via {maskedSignalSummary}";
                return true;
            }
        }

        restoredChunks = wholeTileRestoredChunks;
        planSignature = HashCode.Combine(false, (int)MathF.Round(factor * 1000f));
        string signalSummary = tileWeakSignalCandidate
            ? "whole-tile weak range"
            : $"{observedSignalCount} partial weak signal source(s){(usedTextureGuidance ? ", cell-guided" : string.Empty)}";
        reason += preserveNegativeFloor
            ? $", whole tile from source floor via {signalSummary}"
            : $", whole tile from z=0 via {signalSummary}";
        return true;
    }

    private bool TryEstimateTerrainWeakSignalRestoreFactorForObservedRange(
        int tileX,
        int tileY,
        float observedMin,
        float observedMax,
        out float factor,
        out string reason)
    {
        factor = 1f;
        reason = string.Empty;

        float observedRange = Math.Max(observedMax - observedMin, 0f);
        if (observedRange < 0.25f)
            return false;

        if (TryGetTerrainWeakSignalLoadedBounds(out float loadedMin, out float loadedMax, out int loadedTileCount))
        {
            float loadedRange = Math.Max(loadedMax - loadedMin, 0f);
            float visibilityRatio = loadedRange > 0.001f
                ? observedRange / loadedRange
                : 1f;
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, loadedMin, loadedMax);
            if (visibilityRatio <= 0.25f && rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"partial-signal relief {observedMin:F1}..{observedMax:F1} vs loaded {loadedMin:F1}..{loadedMax:F1} across {loadedTileCount} tile(s)";
                return true;
            }
        }

        if (TryGetTerrainWeakSignalWdlBounds(tileX, tileY, out float coarseMin, out float coarseMax))
        {
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, coarseMin, coarseMax);
            if (rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"partial-signal relief {observedMin:F1}..{observedMax:F1} vs WDL {coarseMin:F1}..{coarseMax:F1}";
                return true;
            }
        }

        float fallbackFactor = EstimateTerrainWeakSignalFallbackFactor(observedMin, observedMax);
        if (fallbackFactor >= 1.25f)
        {
            factor = fallbackFactor;
            reason = $"partial-signal fallback {observedMin:F1}..{observedMax:F1}";
            return true;
        }

        return false;
    }

    private bool TryGetTerrainWeakSignalTileObservedRange(
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks,
        out float minHeight,
        out float maxHeight,
        out int signalCount,
        out bool usedTextureGuidance)
    {
        minHeight = float.MaxValue;
        maxHeight = float.MinValue;
        signalCount = 0;
        usedTextureGuidance = false;

        for (int index = 0; index < sourceChunks.Count; index++)
        {
            Terrain.TerrainChunkData chunk = sourceChunks[index];
            if (!_terrainWeakSignalRestoreUseTextureSubdivisions)
                continue;

            if (!TryBuildTerrainWeakSignalTextureGuidance(chunk, out TerrainWeakSignalTextureGuidance? textureGuidance) || textureGuidance == null)
                continue;

            if (textureGuidance.ObservedMinHeight < minHeight)
                minHeight = textureGuidance.ObservedMinHeight;
            if (textureGuidance.ObservedMaxHeight > maxHeight)
                maxHeight = textureGuidance.ObservedMaxHeight;
            signalCount++;
            usedTextureGuidance = true;
        }

        return signalCount > 0 && minHeight != float.MaxValue && maxHeight != float.MinValue && maxHeight > minHeight;
    }

    private bool TryEstimateTerrainWeakSignalRestoreFactorForChunk(
        int tileX,
        int tileY,
        Terrain.TerrainChunkData chunk,
        TerrainHeightmapIo.TileHeightmap257 tileHeightmap,
        bool tileWeakSignalCandidate,
        TerrainWeakSignalTextureGuidance? textureGuidance,
        out float factor,
        out string reason)
    {
        factor = 1f;
        reason = string.Empty;

        float observedMin;
        float observedMax;
        if (textureGuidance != null)
        {
            observedMin = textureGuidance.ObservedMinHeight;
            observedMax = textureGuidance.ObservedMaxHeight;
        }
        else if (!TryGetTerrainChunkHeightRange(chunk, out observedMin, out observedMax))
        {
            return false;
        }

        if (TryGetTerrainWeakSignalWdlChunkBounds(tileX, tileY, chunk.ChunkX, chunk.ChunkY, out float chunkGuideMin, out float chunkGuideMax, out _))
        {
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, chunkGuideMin, chunkGuideMax);
            if (rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = textureGuidance != null
                    ? $"WDL chunk relief {chunkGuideMin:F1}..{chunkGuideMax:F1} from {DescribeTerrainWeakSignalGuidance(textureGuidance)}"
                    : $"WDL chunk relief {chunkGuideMin:F1}..{chunkGuideMax:F1}";
                return true;
            }
        }

        if (!tileWeakSignalCandidate)
        {
            float mixedTileFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, tileHeightmap.MinHeight, tileHeightmap.MaxHeight);
            if (mixedTileFactor >= 1.25f)
            {
                factor = mixedTileFactor;
                reason = textureGuidance != null
                    ? $"mixed-tile relief {tileHeightmap.MinHeight:F1}..{tileHeightmap.MaxHeight:F1} from {DescribeTerrainWeakSignalGuidance(textureGuidance)}"
                    : $"mixed-tile relief {tileHeightmap.MinHeight:F1}..{tileHeightmap.MaxHeight:F1}";
                return true;
            }
        }

        if (tileWeakSignalCandidate && TryEstimateTerrainWeakSignalRestoreFactor(tileX, tileY, tileHeightmap, out float tileFactor, out string tileReason))
        {
            factor = tileFactor;
            reason = tileReason;
            return true;
        }

        float fallbackFactor = EstimateTerrainWeakSignalFallbackFactor(observedMin, observedMax);
        if (fallbackFactor >= 1.25f)
        {
            factor = fallbackFactor;
            reason = "chunk sea-level fallback";
            return true;
        }

        return false;
    }

    private bool TryEstimateTerrainWeakSignalRestoreFactor(
        int tileX,
        int tileY,
        TerrainHeightmapIo.TileHeightmap257 tileHeightmap,
        out float factor,
        out string reason)
    {
        factor = 1f;
        reason = string.Empty;

        float observedMin = tileHeightmap.MinHeight;
        float observedMax = tileHeightmap.MaxHeight;
        float observedRange = Math.Max(observedMax - observedMin, 0f);
        if (observedRange < 0.25f)
            return false;

        if (TryGetTerrainWeakSignalLoadedBounds(out float loadedMin, out float loadedMax, out int loadedTileCount))
        {
            float loadedRange = Math.Max(loadedMax - loadedMin, 0f);
            float visibilityRatio = loadedRange > 0.001f
                ? observedRange / loadedRange
                : 1f;
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, loadedMin, loadedMax);
            if (visibilityRatio <= 0.25f && rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"loaded-tile relief {loadedMin:F1}..{loadedMax:F1} across {loadedTileCount} tile(s)";
                return true;
            }
        }

        if (TryGetTerrainWeakSignalWdlBounds(tileX, tileY, out float coarseMin, out float coarseMax))
        {
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, coarseMin, coarseMax);
            if (rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"WDL coarse relief {coarseMin:F1}..{coarseMax:F1}";
                return true;
            }
        }

        float fallbackFactor = EstimateTerrainWeakSignalFallbackFactor(observedMin, observedMax);
        if (fallbackFactor >= 1.25f)
        {
            factor = fallbackFactor;
            reason = "sea-level fallback";
            return true;
        }

        return false;
    }

    private bool TryGetTerrainWeakSignalLoadedBounds(out float minHeight, out float maxHeight, out int tileCount)
    {
        float localMinHeight = float.MaxValue;
        float localMaxHeight = float.MinValue;
        int localTileCount = 0;

        void AccumulateTile(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> chunks)
        {
            var key = (tileX, tileY);
            IReadOnlyList<Terrain.TerrainChunkData> baseChunks = _terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks)
                ? originalChunks
                : chunks;

            TerrainHeightmapIo.TileHeightmap257 heightmap = TerrainHeightmapIo.BuildTileHeightmap257(baseChunks);
            if (float.IsNaN(heightmap.MinHeight) || float.IsNaN(heightmap.MaxHeight))
                return;

            if (heightmap.MinHeight < localMinHeight)
                localMinHeight = heightmap.MinHeight;

            if (heightmap.MaxHeight > localMaxHeight)
                localMaxHeight = heightmap.MaxHeight;

            localTileCount++;
        }

        if (_terrainManager != null)
        {
            foreach (var (tileX, tileY) in _terrainManager.LoadedTiles)
            {
                if (_terrainManager.TryGetTileLoadResult(tileX, tileY, out var result) && result.Chunks.Count > 0)
                    AccumulateTile(tileX, tileY, result.Chunks);
            }
        }

        if (_vlmTerrainManager != null)
        {
            foreach (var (tileX, tileY) in _vlmTerrainManager.LoadedTiles)
            {
                if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var result) && result.Chunks.Count > 0)
                    AccumulateTile(tileX, tileY, result.Chunks);
            }
        }

        minHeight = localMinHeight;
        maxHeight = localMaxHeight;
        tileCount = localTileCount;
        return tileCount > 0 && minHeight != float.MaxValue && maxHeight != float.MinValue && maxHeight > minHeight;
    }

    private bool TryGetTerrainWeakSignalGlobalMaxHeight(int tileX, int tileY, out float maxHeight)
    {
        maxHeight = float.MinValue;

        if (TryGetTerrainWeakSignalWdlTile(tileX, tileY, out _) && _terrainWeakSignalWdlData != null)
        {
            for (int index = 0; index < _terrainWeakSignalWdlData.Tiles.Length; index++)
            {
                WdlParser.WdlTile? tile = _terrainWeakSignalWdlData.Tiles[index];
                if (tile?.HasData != true)
                    continue;

                if (float.IsNaN(tile.MaxZ) || float.IsInfinity(tile.MaxZ))
                    continue;

                if (tile.MaxZ > maxHeight)
                    maxHeight = tile.MaxZ;
            }

            if (maxHeight != float.MinValue)
                return true;
        }

        if (TryGetTerrainWeakSignalLoadedBounds(out _, out float loadedMaxHeight, out _))
        {
            maxHeight = loadedMaxHeight;
            return true;
        }

        return false;
    }

    private bool TryGetTerrainWeakSignalWdlTile(int tileX, int tileY, out WdlParser.WdlTile? tile)
    {
        tile = null;

        if (_dataSource == null)
            return false;

        string? mapName = _terrainManager?.MapName ?? GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(mapName))
            return false;

        if (!string.Equals(_terrainWeakSignalWdlMapName, mapName, StringComparison.OrdinalIgnoreCase))
        {
            _terrainWeakSignalWdlMapName = mapName;
            _terrainWeakSignalWdlData = null;
        }

        if (_terrainWeakSignalWdlData == null)
        {
            if (_wdlPreviewCacheService != null)
                _wdlPreviewCacheService.EnsurePrefetch(mapName);

            if (!WdlDataSourceResolver.TryReadWdlBytes(_dataSource, mapName, out byte[]? wdlBytes, out _)
                || wdlBytes == null
                || wdlBytes.Length == 0)
            {
                return false;
            }

            _terrainWeakSignalWdlData = WdlParser.Parse(wdlBytes);
        }

        if (_terrainWeakSignalWdlData == null)
            return false;

        int tileIndex = tileY * 64 + tileX;
        if ((uint)tileIndex >= _terrainWeakSignalWdlData.Tiles.Length)
            return false;

        tile = _terrainWeakSignalWdlData.Tiles[tileIndex];
        return tile?.HasData == true;
    }

    private bool TryGetTerrainWeakSignalWdlBounds(int tileX, int tileY, out float minHeight, out float maxHeight)
    {
        minHeight = 0f;
        maxHeight = 0f;
        if (!TryGetTerrainWeakSignalWdlTile(tileX, tileY, out WdlParser.WdlTile? tile) || tile == null)
            return false;

        minHeight = tile.MinZ;
        maxHeight = tile.MaxZ;
        return maxHeight > minHeight;
    }

    private bool TryGetTerrainWeakSignalWdlChunkBounds(
        int tileX,
        int tileY,
        int chunkX,
        int chunkY,
        out float minHeight,
        out float maxHeight,
        out float centerHeight)
    {
        minHeight = 0f;
        maxHeight = 0f;
        centerHeight = 0f;

        if ((uint)chunkX >= 16u || (uint)chunkY >= 16u)
            return false;

        if (!TryGetTerrainWeakSignalWdlTile(tileX, tileY, out WdlParser.WdlTile? tile) || tile == null)
            return false;

        float h00 = tile.Height17[chunkY, chunkX];
        float h10 = tile.Height17[chunkY, chunkX + 1];
        float h01 = tile.Height17[chunkY + 1, chunkX];
        float h11 = tile.Height17[chunkY + 1, chunkX + 1];
        centerHeight = tile.Height16[chunkY, chunkX];

        minHeight = MathF.Min(MathF.Min(h00, h10), MathF.Min(MathF.Min(h01, h11), centerHeight));
        maxHeight = MathF.Max(MathF.Max(h00, h10), MathF.Max(MathF.Max(h01, h11), centerHeight));
        return maxHeight > minHeight;
    }

    private static float EstimateTerrainWeakSignalRestoreFactorFromRanges(float observedMin, float observedMax, float coarseMin, float coarseMax)
    {
        const float epsilon = 0.001f;
        float observedRange = Math.Max(observedMax - observedMin, 0f);
        float coarseRange = Math.Max(coarseMax - coarseMin, 0f);
        if (observedRange <= epsilon || coarseRange <= epsilon)
            return 1f;

        float rawFactor = 1f;
        if (coarseRange > observedRange * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseRange / observedRange);

        float observedBelow = Math.Max(0f, -observedMin);
        float coarseBelow = Math.Max(0f, -coarseMin);
        if (observedBelow > epsilon && coarseBelow > observedBelow * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseBelow / observedBelow);

        float observedAbove = Math.Max(0f, observedMax);
        float coarseAbove = Math.Max(0f, coarseMax);
        if (observedAbove > epsilon && coarseAbove > observedAbove * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseAbove / observedAbove);

        return Math.Clamp(rawFactor, 1f, TerrainWeakSignalRestoreMaxFactor);
    }

    private static float EstimateTerrainWeakSignalFallbackFactor(float observedMin, float observedMax)
    {
        float observedBelow = Math.Max(0f, -observedMin);
        float observedRange = Math.Max(observedMax - observedMin, 0f);
        if (observedBelow < 0.5f || observedRange > 12f || observedMax > 4f)
            return 1f;

        float rawFactor = 16f / observedBelow;
        return Math.Clamp(rawFactor, 1f, TerrainWeakSignalRestoreMaxFactor);
    }

    private static float SnapTerrainWeakSignalRestoreFactor(float rawFactor)
    {
        if (rawFactor <= 1f)
            return 1f;

        float[] supported = { 1f, 2f, 4f, 8f, 16f, 32f, 64f, 128f, 256f, 512f };
        foreach (float value in supported)
        {
            if (rawFactor <= value)
                return value;
        }

        return supported[^1];
    }

    private static bool TryGetTerrainChunkHeightRange(Terrain.TerrainChunkData chunk, out float minHeight, out float maxHeight)
    {
        minHeight = float.MaxValue;
        maxHeight = float.MinValue;

        if (chunk.Heights == null || chunk.Heights.Length == 0)
            return false;

        for (int index = 0; index < chunk.Heights.Length; index++)
        {
            float height = chunk.Heights[index];
            if (float.IsNaN(height) || float.IsInfinity(height))
                continue;

            if (height < minHeight)
                minHeight = height;

            if (height > maxHeight)
                maxHeight = height;
        }

        return minHeight != float.MaxValue && maxHeight != float.MinValue && maxHeight > minHeight;
    }

    private static bool HasTerrainWeakSignalShadowSignal(Terrain.TerrainChunkData chunk)
    {
        if (chunk.ShadowMap == null || chunk.ShadowMap.Length == 0)
            return false;

        for (int index = 0; index < chunk.ShadowMap.Length; index++)
        {
            if (chunk.ShadowMap[index] != 0)
                return true;
        }

        return false;
    }

    private bool TryBuildTerrainWeakSignalShadowEdgeVertexWeights(
        Terrain.TerrainChunkData chunk,
        TerrainWeakSignalTextureGuidance? textureGuidance,
        out float[]? vertexWeights)
    {
        vertexWeights = null;
        if (!HasTerrainWeakSignalShadowSignal(chunk) || chunk.ShadowMap == null || chunk.ShadowMap.Length < 64 * 64)
            return false;

        const int subDivisions = 8;
        bool[] selectedMask = new bool[subDivisions * subDivisions];
        float[] shadowCoverage = new float[subDivisions * subDivisions];
        float[] averageHeights = new float[subDivisions * subDivisions];
        int[] dominantLayers = new int[subDivisions * subDivisions];

        for (int cellY = 0; cellY < subDivisions; cellY++)
        {
            for (int cellX = 0; cellX < subDivisions; cellX++)
            {
                int cellIndex = cellY * subDivisions + cellX;
                shadowCoverage[cellIndex] = ComputeTerrainWeakSignalShadowCoverage(chunk.ShadowMap, cellX, cellY, subDivisions);
                averageHeights[cellIndex] = ComputeTerrainWeakSignalAverageHeightForSubCell(chunk, cellX, cellY, subDivisions);
                dominantLayers[cellIndex] = GetTerrainWeakSignalDominantLayerForSubChunkCell(chunk, cellX, cellY, subDivisions);
                if (textureGuidance != null && textureGuidance.SelectedMask.Length == selectedMask.Length && textureGuidance.SelectedMask[cellIndex])
                    selectedMask[cellIndex] = true;
            }
        }

        int preferredLayer = textureGuidance?.DominantLayerIndex ?? -1;
        bool hasDirectionalShadowAnchor = TryInferTerrainWeakSignalShadowDirection(
            textureGuidance?.SelectedMask,
            shadowCoverage,
            averageHeights,
            dominantLayers,
            subDivisions,
            preferredLayer,
            out var litToShadowOffset);
        int selectedCellCount = selectedMask.Count(static value => value);
        for (int cellY = 0; cellY < subDivisions; cellY++)
        {
            for (int cellX = 0; cellX < subDivisions; cellX++)
            {
                int cellIndex = cellY * subDivisions + cellX;
                float coverage = shadowCoverage[cellIndex];
                if (coverage > TerrainWeakSignalShadowLitMaxCoverage)
                    continue;

                if (preferredLayer >= 0 && dominantLayers[cellIndex] != preferredLayer)
                    continue;

                bool touchesSeed = textureGuidance == null
                    || CellTouchesSelectedMask(textureGuidance.SelectedMask, subDivisions, cellX, cellY);
                if (!touchesSeed)
                    continue;

                bool foundShadowNeighbor = hasDirectionalShadowAnchor
                    ? TryGetTerrainWeakSignalDirectionalShadowNeighborAverageHeight(
                        shadowCoverage,
                        averageHeights,
                        subDivisions,
                        cellX,
                        cellY,
                        litToShadowOffset.offsetX,
                        litToShadowOffset.offsetY,
                        out float shadowNeighborAverageHeight)
                    : TryGetTerrainWeakSignalShadowNeighborHeightRange(shadowCoverage, averageHeights, subDivisions, cellX, cellY, out shadowNeighborAverageHeight);
                if (!foundShadowNeighbor)
                    continue;

                if (averageHeights[cellIndex] + TerrainWeakSignalShadowEdgeMinHeightDelta < shadowNeighborAverageHeight)
                    continue;

                if (!selectedMask[cellIndex])
                {
                    selectedMask[cellIndex] = true;
                    selectedCellCount++;
                }
            }
        }

        for (int pass = 0; pass < 2; pass++)
        {
            bool changed = false;
            bool[] nextMask = (bool[])selectedMask.Clone();
            for (int cellY = 0; cellY < subDivisions; cellY++)
            {
                for (int cellX = 0; cellX < subDivisions; cellX++)
                {
                    int cellIndex = cellY * subDivisions + cellX;
                    if (nextMask[cellIndex] || shadowCoverage[cellIndex] > TerrainWeakSignalShadowLitMaxCoverage)
                        continue;

                    if (preferredLayer >= 0 && dominantLayers[cellIndex] != preferredLayer)
                        continue;

                    bool foundSelectedNeighbor = hasDirectionalShadowAnchor
                        ? TryGetTerrainWeakSignalDirectionalSelectedNeighborAverageHeight(
                            selectedMask,
                            averageHeights,
                            subDivisions,
                            cellX,
                            cellY,
                            litToShadowOffset.offsetX,
                            litToShadowOffset.offsetY,
                            out float selectedNeighborAverageHeight)
                        : TryGetTerrainWeakSignalSelectedNeighborAverageHeight(selectedMask, averageHeights, subDivisions, cellX, cellY, out selectedNeighborAverageHeight);
                    if (!foundSelectedNeighbor)
                        continue;

                    if (averageHeights[cellIndex] + 0.25f < selectedNeighborAverageHeight)
                        continue;

                    nextMask[cellIndex] = true;
                    selectedCellCount++;
                    changed = true;
                }
            }

            selectedMask = nextMask;
            if (!changed)
                break;
        }

        if (selectedCellCount == 0)
            return false;

        vertexWeights = BuildTerrainWeakSignalSubCellVertexWeights(selectedMask, subDivisions);
        return true;
    }

    private bool TryBuildTerrainWeakSignalTextureGuidance(Terrain.TerrainChunkData chunk, out TerrainWeakSignalTextureGuidance? guidance)
    {
        guidance = null;

        const int subDivisions = 8;
        float cellSize = WoWConstants.ChunkSize / subDivisions;
        var cells = new TerrainWeakSignalSubChunkCell[subDivisions * subDivisions];
        float selectedMinHeight = float.MaxValue;
        float selectedMaxHeight = float.MinValue;
        float selectedAverageHeightSum = 0f;

        for (int cellY = 0; cellY < subDivisions; cellY++)
        {
            for (int cellX = 0; cellX < subDivisions; cellX++)
            {
                float minHeight = float.MaxValue;
                float maxHeight = float.MinValue;
                float averageHeight = 0f;
                int sampleCount = 0;

                for (int sampleY = 0; sampleY < 3; sampleY++)
                {
                    for (int sampleX = 0; sampleX < 3; sampleX++)
                    {
                        float localX = cellX * cellSize + ((sampleX + 0.5f) / 3f) * cellSize;
                        float localY = cellY * cellSize + ((sampleY + 0.5f) / 3f) * cellSize;
                        float height = SampleHeightOuterGrid(chunk, localX, localY);
                        if (height < minHeight)
                            minHeight = height;
                        if (height > maxHeight)
                            maxHeight = height;
                        averageHeight += height;
                        sampleCount++;
                    }
                }

                averageHeight = sampleCount > 0 ? averageHeight / sampleCount : 0f;
                int dominantLayerIndex = GetTerrainWeakSignalDominantLayerForSubChunkCell(chunk, cellX, cellY, subDivisions);
                bool isWeakSignalCandidate = minHeight != float.MaxValue
                    && maxHeight != float.MinValue
                    && IsTerrainWeakSignalCandidateRange(minHeight, maxHeight);
                bool touchesBorder = cellX == 0 || cellY == 0 || cellX == subDivisions - 1 || cellY == subDivisions - 1;

                var cell = new TerrainWeakSignalSubChunkCell
                {
                    CellX = cellX,
                    CellY = cellY,
                    DominantLayerIndex = dominantLayerIndex,
                    MinHeight = minHeight,
                    MaxHeight = maxHeight,
                    AverageHeight = averageHeight,
                    IsWeakSignalCandidate = isWeakSignalCandidate,
                    TouchesBorder = touchesBorder,
                };
                cells[cellY * subDivisions + cellX] = cell;
            }
        }

        bool[] selectedMask = new bool[subDivisions * subDivisions];
        int selectedCellCount = 0;
        int borderSelectedCellCount = 0;
        for (int index = 0; index < cells.Length; index++)
        {
            TerrainWeakSignalSubChunkCell cell = cells[index];
            if (!cell.IsWeakSignalCandidate)
                continue;

            selectedMask[index] = true;
            selectedCellCount++;
            if (cell.TouchesBorder)
                borderSelectedCellCount++;
            if (cell.MinHeight < selectedMinHeight)
                selectedMinHeight = cell.MinHeight;
            if (cell.MaxHeight > selectedMaxHeight)
                selectedMaxHeight = cell.MaxHeight;
            selectedAverageHeightSum += cell.AverageHeight;
        }

        if (selectedCellCount == 0)
            return false;

        guidance = new TerrainWeakSignalTextureGuidance
        {
            Cells = cells,
            SelectedMask = selectedMask,
            DominantLayerIndex = -1,
            SelectedCellCount = selectedCellCount,
            BorderSelectedCellCount = borderSelectedCellCount,
            ObservedMinHeight = selectedMinHeight,
            ObservedMaxHeight = selectedMaxHeight,
            ObservedAverageHeight = selectedAverageHeightSum / selectedCellCount,
        };
        return true;
    }

    private static int GetTerrainWeakSignalDominantLayerForSubChunkCell(Terrain.TerrainChunkData chunk, int cellX, int cellY, int subDivisions)
    {
        const int alphaSize = 64;
        int pixelStartX = cellX * (alphaSize / subDivisions);
        int pixelStartY = cellY * (alphaSize / subDivisions);
        int pixelEndX = pixelStartX + (alphaSize / subDivisions);
        int pixelEndY = pixelStartY + (alphaSize / subDivisions);

        float[] layerSums = new float[Math.Max(chunk.Layers.Length, 1)];
        for (int y = pixelStartY; y < pixelEndY; y++)
        {
            for (int x = pixelStartX; x < pixelEndX; x++)
            {
                int pixelIndex = y * alphaSize + x;
                float maxOverlay = 0f;
                for (int layerIndex = 1; layerIndex < chunk.Layers.Length; layerIndex++)
                {
                    if (!chunk.AlphaMaps.TryGetValue(layerIndex, out byte[]? alphaMap) || alphaMap.Length <= pixelIndex)
                        continue;

                    float weight = alphaMap[pixelIndex];
                    layerSums[layerIndex] += weight;
                    if (weight > maxOverlay)
                        maxOverlay = weight;
                }

                layerSums[0] += Math.Max(0f, 255f - maxOverlay);
            }
        }

        int dominantLayerIndex = 0;
        float dominantWeight = layerSums[0];
        for (int layerIndex = 1; layerIndex < layerSums.Length; layerIndex++)
        {
            if (layerSums[layerIndex] > dominantWeight)
            {
                dominantWeight = layerSums[layerIndex];
                dominantLayerIndex = layerIndex;
            }
        }

        return dominantLayerIndex;
    }

    private static float[] BuildTerrainWeakSignalTextureGuidanceVertexWeights(TerrainWeakSignalTextureGuidance guidance)
    {
        const int subDivisions = 8;
        return BuildTerrainWeakSignalSubCellVertexWeights(guidance.SelectedMask, subDivisions);
    }

    private static string DescribeTerrainWeakSignalGuidance(TerrainWeakSignalTextureGuidance guidance)
    {
        return $"cell-guided weak cells across {guidance.SelectedCellCount} cell(s)";
    }

    private static float[] BlendTerrainWeakSignalMaskedChunkHeights(float[] sourceHeights, float[] restoredHeights, float[]? vertexWeights)
    {
        float[] blendedHeights = new float[sourceHeights.Length];
        for (int index = 0; index < sourceHeights.Length; index++)
        {
            float sourceHeight = sourceHeights[index];
            float targetHeight = index < restoredHeights.Length ? restoredHeights[index] : sourceHeight;
            float weight = vertexWeights != null && index < vertexWeights.Length ? Math.Clamp(vertexWeights[index], 0f, 1f) : 1f;
            blendedHeights[index] = sourceHeight + ((targetHeight - sourceHeight) * weight);
        }

        return blendedHeights;
    }

    private static int GetTerrainWeakSignalSelectedMaskHash(bool[] selectedMask)
    {
        var hash = new HashCode();
        for (int index = 0; index < selectedMask.Length; index++)
            hash.Add(selectedMask[index]);

        return hash.ToHashCode();
    }

    private static float[] BuildTerrainWeakSignalSubCellVertexWeights(bool[] selectedMask, int subDivisions)
    {
        float[] weights = new float[145];
        for (int vertexIndex = 0; vertexIndex < weights.Length; vertexIndex++)
        {
            GetChunkVertexLocalPosition(vertexIndex, out float localX, out float localY);
            int cellX = Math.Clamp((int)MathF.Floor(Math.Clamp(localX / WoWConstants.ChunkSize, 0f, 0.999f) * subDivisions), 0, subDivisions - 1);
            int cellY = Math.Clamp((int)MathF.Floor(Math.Clamp(localY / WoWConstants.ChunkSize, 0f, 0.999f) * subDivisions), 0, subDivisions - 1);
            weights[vertexIndex] = selectedMask[cellY * subDivisions + cellX] ? 1f : 0f;
        }

        return weights;
    }

    private static bool CellTouchesSelectedMask(bool[] mask, int size, int cellX, int cellY)
    {
        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                if ((uint)neighborX >= size || (uint)neighborY >= size)
                    continue;

                if (mask[neighborY * size + neighborX])
                    return true;
            }
        }

        return false;
    }

    private static bool TryGetTerrainWeakSignalShadowNeighborHeightRange(float[] shadowCoverage, float[] averageHeights, int size, int cellX, int cellY, out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                if (offsetX == 0 && offsetY == 0)
                    continue;

                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                if ((uint)neighborX >= size || (uint)neighborY >= size)
                    continue;

                int neighborIndex = neighborY * size + neighborX;
                if (shadowCoverage[neighborIndex] < TerrainWeakSignalShadowEdgeMinCoverage)
                    continue;

                averageHeight += averageHeights[neighborIndex];
                count++;
            }
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryGetTerrainWeakSignalDirectionalShadowNeighborAverageHeight(
        float[] shadowCoverage,
        float[] averageHeights,
        int size,
        int cellX,
        int cellY,
        int offsetX,
        int offsetY,
        out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        foreach (var (sampleOffsetX, sampleOffsetY) in EnumerateDirectionalOffsets(offsetX, offsetY))
        {
            int neighborX = cellX + sampleOffsetX;
            int neighborY = cellY + sampleOffsetY;
            if ((uint)neighborX >= size || (uint)neighborY >= size)
                continue;

            int neighborIndex = neighborY * size + neighborX;
            if (shadowCoverage[neighborIndex] < TerrainWeakSignalShadowEdgeMinCoverage)
                continue;

            averageHeight += averageHeights[neighborIndex];
            count++;
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryGetTerrainWeakSignalSelectedNeighborAverageHeight(bool[] selectedMask, float[] averageHeights, int size, int cellX, int cellY, out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                if (offsetX == 0 && offsetY == 0)
                    continue;

                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                if ((uint)neighborX >= size || (uint)neighborY >= size)
                    continue;

                int neighborIndex = neighborY * size + neighborX;
                if (!selectedMask[neighborIndex])
                    continue;

                averageHeight += averageHeights[neighborIndex];
                count++;
            }
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryGetTerrainWeakSignalDirectionalSelectedNeighborAverageHeight(
        bool[] selectedMask,
        float[] averageHeights,
        int size,
        int cellX,
        int cellY,
        int offsetX,
        int offsetY,
        out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        foreach (var (sampleOffsetX, sampleOffsetY) in EnumerateDirectionalOffsets(offsetX, offsetY))
        {
            int neighborX = cellX + sampleOffsetX;
            int neighborY = cellY + sampleOffsetY;
            if ((uint)neighborX >= size || (uint)neighborY >= size)
                continue;

            int neighborIndex = neighborY * size + neighborX;
            if (!selectedMask[neighborIndex])
                continue;

            averageHeight += averageHeights[neighborIndex];
            count++;
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryInferTerrainWeakSignalShadowDirection(
        bool[]? seedMask,
        float[] shadowCoverage,
        float[] averageHeights,
        int[] dominantLayers,
        int size,
        int preferredLayer,
        out (int offsetX, int offsetY) litToShadowOffset)
    {
        litToShadowOffset = default;
        float bestScore = float.MinValue;
        int bestMatchCount = 0;

        foreach ((int offsetX, int offsetY) in EnumerateNeighborDirections())
        {
            float score = 0f;
            int matchCount = 0;

            for (int cellY = 0; cellY < size; cellY++)
            {
                for (int cellX = 0; cellX < size; cellX++)
                {
                    int cellIndex = cellY * size + cellX;
                    if (shadowCoverage[cellIndex] > TerrainWeakSignalShadowLitMaxCoverage)
                        continue;

                    if (preferredLayer >= 0 && dominantLayers[cellIndex] != preferredLayer)
                        continue;

                    bool relevantToSeed = seedMask == null
                        || seedMask[cellIndex]
                        || CellTouchesSelectedMask(seedMask, size, cellX, cellY);
                    if (!relevantToSeed)
                        continue;

                    if (!TryGetTerrainWeakSignalDirectionalShadowNeighborAverageHeight(
                        shadowCoverage,
                        averageHeights,
                        size,
                        cellX,
                        cellY,
                        offsetX,
                        offsetY,
                        out float shadowNeighborAverageHeight))
                    {
                        continue;
                    }

                    float heightDelta = averageHeights[cellIndex] - shadowNeighborAverageHeight;
                    if (heightDelta + TerrainWeakSignalShadowEdgeMinHeightDelta < 0f)
                        continue;

                    float localScore = 1f + Math.Max(heightDelta, 0f);
                    if (seedMask != null && seedMask[cellIndex])
                        localScore += 0.5f;

                    score += localScore;
                    matchCount++;
                }
            }

            if (matchCount == 0)
                continue;

            if (score > bestScore || (Math.Abs(score - bestScore) < 0.001f && matchCount > bestMatchCount))
            {
                bestScore = score;
                bestMatchCount = matchCount;
                litToShadowOffset = (offsetX, offsetY);
            }
        }

        return bestMatchCount > 0;
    }

    private static IEnumerable<(int offsetX, int offsetY)> EnumerateNeighborDirections()
    {
        yield return (-1, -1);
        yield return (0, -1);
        yield return (1, -1);
        yield return (-1, 0);
        yield return (1, 0);
        yield return (-1, 1);
        yield return (0, 1);
        yield return (1, 1);
    }

    private static IEnumerable<(int offsetX, int offsetY)> EnumerateDirectionalOffsets(int offsetX, int offsetY)
    {
        yield return (offsetX, offsetY);

        if (offsetX == 0)
        {
            yield return (-1, offsetY);
            yield return (1, offsetY);
            yield break;
        }

        if (offsetY == 0)
        {
            yield return (offsetX, -1);
            yield return (offsetX, 1);
            yield break;
        }

        yield return (offsetX, 0);
        yield return (0, offsetY);
    }

    private static float ComputeTerrainWeakSignalShadowCoverage(byte[] shadowMap, int cellX, int cellY, int subDivisions)
    {
        const int shadowSize = 64;
        int pixelsPerCell = shadowSize / subDivisions;
        int pixelStartX = cellX * pixelsPerCell;
        int pixelStartY = cellY * pixelsPerCell;
        float sum = 0f;
        int count = 0;

        for (int y = pixelStartY; y < pixelStartY + pixelsPerCell; y++)
        {
            for (int x = pixelStartX; x < pixelStartX + pixelsPerCell; x++)
            {
                int pixelIndex = y * shadowSize + x;
                if ((uint)pixelIndex >= shadowMap.Length)
                    continue;

                sum += shadowMap[pixelIndex] / 255f;
                count++;
            }
        }

        return count > 0 ? sum / count : 0f;
    }

    private static float ComputeTerrainWeakSignalAverageHeightForSubCell(Terrain.TerrainChunkData chunk, int cellX, int cellY, int subDivisions)
    {
        float cellSize = WoWConstants.ChunkSize / subDivisions;
        float averageHeight = 0f;
        int sampleCount = 0;
        for (int sampleY = 0; sampleY < 3; sampleY++)
        {
            for (int sampleX = 0; sampleX < 3; sampleX++)
            {
                float localX = cellX * cellSize + ((sampleX + 0.5f) / 3f) * cellSize;
                float localY = cellY * cellSize + ((sampleY + 0.5f) / 3f) * cellSize;
                averageHeight += SampleHeightOuterGrid(chunk, localX, localY);
                sampleCount++;
            }
        }

        return sampleCount > 0 ? averageHeight / sampleCount : 0f;
    }

    private static float[] BuildTerrainWeakSignalRestoredChunkHeights(
        Terrain.TerrainChunkData chunk,
        float factor,
        float[]? vertexWeights = null,
        float? globalMaxHeight = null)
    {
        float[] restoredHeights = new float[chunk.Heights.Length];
        float anchorHeight = 0f;
        bool preserveNegativeFloor = false;
        if (TryGetTerrainChunkHeightRange(chunk, out float chunkMinHeight, out _))
        {
            anchorHeight = chunkMinHeight < 0f ? chunkMinHeight : 0f;
            preserveNegativeFloor = anchorHeight < 0f;
        }

        for (int index = 0; index < chunk.Heights.Length; index++)
        {
            float sourceHeight = chunk.Heights[index];
            float amplifiedHeight = anchorHeight + ((sourceHeight - anchorHeight) * factor);
            float weight = vertexWeights != null && index < vertexWeights.Length ? Math.Clamp(vertexWeights[index], 0f, 1f) : 1f;
            float restoredHeight = sourceHeight + ((amplifiedHeight - sourceHeight) * weight);
            if (!preserveNegativeFloor && restoredHeight < 0f)
                restoredHeight = 0f;
            if (globalMaxHeight.HasValue && restoredHeight > globalMaxHeight.Value)
                restoredHeight = globalMaxHeight.Value;

            restoredHeights[index] = restoredHeight;
        }

        return restoredHeights;
    }

    private void ApplyEditedTileChunks(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> newChunks,
        IEnumerable<(int chunkX, int chunkY)> editedChunks)
    {
        if (_terrainManager != null)
            _terrainManager.ReplaceTileChunksAndRebuild(tileX, tileY, newChunks);
        else
            _vlmTerrainManager?.ReplaceTileChunksAndRebuild(tileX, tileY, newChunks);

        MarkChunkToolTileDirty(tileX, tileY, editedChunks);
    }

    private void MarkChunkToolTileDirty(int tileX, int tileY, IEnumerable<(int chunkX, int chunkY)> editedChunks)
    {
        if (!_chunkClipboardDirtyTileChunks.TryGetValue((tileX, tileY), out var chunkSet))
        {
            chunkSet = new HashSet<(int chunkX, int chunkY)>();
            _chunkClipboardDirtyTileChunks[(tileX, tileY)] = chunkSet;
        }

        foreach (var editedChunk in editedChunks)
            chunkSet.Add(editedChunk);
    }

    private int GetChunkToolDirtyTileCount() => _chunkClipboardDirtyTileChunks.Count;

    private int GetChunkToolDirtyChunkCount()
        => _chunkClipboardDirtyTileChunks.Values.Sum(chunks => chunks.Count);

    private string CreateChunkToolHeightmapOutputDirectory()
    {
        string root = Path.Combine(EnsureEditorProjectOutputDirectory(), "chunk-tool-heightmaps");
        Directory.CreateDirectory(root);

        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string candidate = Path.Combine(root, timestamp);
        int suffix = 1;
        while (Directory.Exists(candidate))
        {
            candidate = Path.Combine(root, $"{timestamp}_{suffix:D2}");
            suffix++;
        }

        Directory.CreateDirectory(candidate);
        return candidate;
    }

    private void SaveChunkToolHeightmapOutputs()
    {
        if (_chunkClipboardDirtyTileChunks.Count == 0)
        {
            _chunkClipboardStatus = "No edited chunk tiles are tracked yet.";
            return;
        }

        string outputDir = CreateChunkToolHeightmapOutputDirectory();
        var manifest = new ChunkToolHeightmapSaveManifest
        {
            ProjectName = GetEditorProjectName(),
            SourceKey = GetEditorProjectSourceKey() ?? string.Empty,
            GeneratedUtc = DateTime.UtcNow.ToString("O"),
        };

        int written = 0;
        int skipped = 0;

        foreach (var entry in _chunkClipboardDirtyTileChunks
            .OrderBy(item => item.Key.tileX)
            .ThenBy(item => item.Key.tileY))
        {
            var (tileX, tileY) = entry.Key;
            var chunks = LoadTileChunksForExport(tileX, tileY);
            if (chunks == null || chunks.Count == 0)
            {
                skipped++;
                continue;
            }

            var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, tile.MinHeight, tile.MaxHeight);

            string pngName = $"tile_{tileX}_{tileY}_height_257.png";
            string jsonName = $"tile_{tileX}_{tileY}_height_257.json";
            string pngPath = Path.Combine(outputDir, pngName);
            string jsonPath = Path.Combine(outputDir, jsonName);

            using (var fs = File.Create(pngPath))
                img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

            var meta = new HeightmapMetadata
            {
                MinHeight = tile.MinHeight,
                MaxHeight = tile.MaxHeight,
                Normalization = "per_tile",
            };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));

            manifest.Tiles.Add(new ChunkToolHeightmapSaveTile
            {
                TileX = tileX,
                TileY = tileY,
                EditedChunks = entry.Value
                    .OrderBy(chunk => chunk.chunkX)
                    .ThenBy(chunk => chunk.chunkY)
                    .Select(chunk => $"{chunk.chunkX},{chunk.chunkY}")
                    .ToList(),
                HeightmapPng = pngName,
                HeightmapMetadataJson = jsonName,
                MinHeight = tile.MinHeight,
                MaxHeight = tile.MaxHeight,
            });
            written++;
        }

        if (written == 0)
        {
            _chunkClipboardStatus = "Chunk-tool save failed: no edited tiles could be exported.";
            return;
        }

        string manifestPath = Path.Combine(outputDir, "chunk_tool_heightmap_manifest.json");
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));

        _chunkClipboardLastSaveFolder = outputDir;
        _chunkClipboardStatus = $"Saved {written} edited tile heightmap output(s) to {outputDir}"
            + (skipped > 0 ? $" (skipped {skipped})" : string.Empty)
            + ". Source terrain files were left untouched.";
    }

    private void ClearChunkToolDirtyTracking()
    {
        _chunkClipboardDirtyTileChunks.Clear();
        _chunkClipboardLastSaveFolder = string.Empty;
        _chunkClipboardStatus = "Cleared chunk-tool dirty tracking.";
    }

    private void PasteChunkAtTarget(TerrainRenderer renderer)
    {
        if (_chunkClipboard == null)
        {
            _chunkClipboardStatus = "Paste failed: clipboard is empty.";
            return;
        }

        if (_chunkClipboardLockedTargetKey == null)
        {
            _chunkClipboardStatus = "Paste blocked: lock a paste target with Ctrl+LMB.";
            return;
        }

        var key = (TileX: _chunkClipboardLockedTargetKey.Value.tileX,
            TileY: _chunkClipboardLockedTargetKey.Value.tileY,
            ChunkX: _chunkClipboardLockedTargetKey.Value.chunkX,
            ChunkY: _chunkClipboardLockedTargetKey.Value.chunkY);

        if (!TryGetTileChunksForEdit(key.TileX, key.TileY, out var chunks))
        {
            _chunkClipboardStatus = $"Paste failed: tile data not available for tile({key.TileX},{key.TileY}).";
            return;
        }

        int idx = chunks.FindIndex(c => c.ChunkX == key.ChunkX && c.ChunkY == key.ChunkY);
        if (idx < 0)
        {
            _chunkClipboardStatus = $"Paste failed: chunk not found in tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY}).";
            return;
        }

        var target = chunks[idx];
        bool layersMatch = AreLayersCompatible(target.Layers, _chunkClipboard.Layers);

        float[] heights = (float[])_chunkClipboard.Heights.Clone();
        Vector3[] normals = (Vector3[])_chunkClipboard.Normals.Clone();
        if (_chunkClipboardPasteRelativeHeights)
        {
            float sourceRef = ComputeAverageHeight(_chunkClipboard.Heights);
            float targetRef = ComputeAverageHeight(target.Heights);
            float delta = targetRef - sourceRef;
            for (int i = 0; i < heights.Length; i++)
                heights[i] += delta;
        }

        var layersToUse = target.Layers;
        var alphaToUse = target.AlphaMaps;
        byte[]? shadowToUse = target.ShadowMap;

        if (_chunkClipboardIncludeTextures)
        {
            layersToUse = _chunkClipboard.Layers;
            if (_chunkClipboardIncludeAlphaShadow)
            {
                alphaToUse = CloneAlphaMaps(_chunkClipboard.AlphaMaps);
                shadowToUse = _chunkClipboard.ShadowMap != null ? (byte[])_chunkClipboard.ShadowMap.Clone() : null;
            }
        }
        else if (_chunkClipboardIncludeAlphaShadow && layersMatch)
        {
            alphaToUse = CloneAlphaMaps(_chunkClipboard.AlphaMaps);
            shadowToUse = _chunkClipboard.ShadowMap != null ? (byte[])_chunkClipboard.ShadowMap.Clone() : null;
        }

        var pasted = CloneTerrainChunk(
            target,
            heights: heights,
            normals: normals,
            holeMask: _chunkClipboard.HoleMask,
            layers: layersToUse,
            alphaMaps: alphaToUse,
            shadowMap: shadowToUse,
            mccvColors: (target.MccvColors != null && _chunkClipboard.MccvColors != null)
                ? (byte[])_chunkClipboard.MccvColors.Clone()
                : target.MccvColors);

        var newChunks = chunks.ToList();
        newChunks[idx] = pasted;

        ApplyEditedTileChunks(key.TileX, key.TileY, newChunks, new[] { (key.ChunkX, key.ChunkY) });

        bool didTextures = _chunkClipboardIncludeTextures;
        bool didAlpha = _chunkClipboardIncludeAlphaShadow && (didTextures || layersMatch);

        _chunkClipboardStatus = $"Pasted heights" +
                       (didTextures ? " + textures" : "") +
                       (didAlpha ? " + alpha/shadow" : "") +
                       $" into tile({key.TileX},{key.TileY}) chunk({key.ChunkX},{key.ChunkY})" +
                       (!didTextures && _chunkClipboardIncludeAlphaShadow && !layersMatch ? " (alpha skipped: layer mismatch)" : "");
    }

    private void DrawEditorOverlays(Matrix4x4 view, Matrix4x4 proj)
    {
        var renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
            return;

        bool drawChunkClipboardOverlay = _chunkClipboardShowOverlay
            && (_selectedChunks.Count > 0 || _chunkClipboardLockedTargetKey != null || _chunkClipboardCopiedKey != null);
        bool drawMcnkOverlay = ShouldDrawMcnkFlagOverlay(renderer);
        if (!drawChunkClipboardOverlay && !drawMcnkOverlay)
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

        _editorOverlayBb.FlushBatch(view, proj);

        _gl.DepthMask(true);
    }

    private static float ComputeAverageHeight(float[] heights)
    {
        if (heights == null || heights.Length == 0)
            return 0f;
        double sum = 0;
        for (int i = 0; i < heights.Length; i++)
            sum += heights[i];
        return (float)(sum / heights.Length);
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

        height = SampleHeightOuterGrid(chunk, localX, localY);
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

    private static float SampleHeightOuterGrid(Terrain.TerrainChunkData chunk, float localX, float localY)
    {
        if (chunk.Heights == null || chunk.Heights.Length < 145) return chunk.WorldPosition.Z;

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        Span<float> grid = stackalloc float[9 * 9];
        grid.Clear();

        for (int i = 0; i < 145; i++)
        {
            GetChunkVertexPosition(i, out int row, out int col, out bool isInner);
            if (isInner) continue;
            int gy = row / 2;
            if ((uint)gy >= 9u || (uint)col >= 9u) continue;
            grid[gy * 9 + col] = chunk.Heights[i];
        }

        float gx = localX / subCellSize;
        float gyf = localY / subCellSize;
        int ix = Math.Clamp((int)MathF.Floor(gx), 0, 7);
        int iy = Math.Clamp((int)MathF.Floor(gyf), 0, 7);
        float fx = Math.Clamp(gx - ix, 0f, 1f);
        float fy = Math.Clamp(gyf - iy, 0f, 1f);

        float h00 = grid[iy * 9 + ix];
        float h10 = grid[iy * 9 + (ix + 1)];
        float h01 = grid[(iy + 1) * 9 + ix];
        float h11 = grid[(iy + 1) * 9 + (ix + 1)];

        float h0 = h00 + (h10 - h00) * fx;
        float h1 = h01 + (h11 - h01) * fx;
        return h0 + (h1 - h0) * fy;
    }

    private static void GetChunkVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int r = 0; r < 17; r++)
        {
            int rowSize = (r % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = r;
                col = remaining;
                isInner = (r % 2 == 1);
                return;
            }
            remaining -= rowSize;
        }
    }

    private bool TryGetChunkData(int tileX, int tileY, int chunkX, int chunkY, out Terrain.TerrainChunkData chunk)
    {
        chunk = new Terrain.TerrainChunkData();

        if (!TryGetTileChunksForEdit(tileX, tileY, out var chunks))
            return false;

        var found = chunks.FirstOrDefault(c => c != null && c.ChunkX == chunkX && c.ChunkY == chunkY);
        if (found == null || found.Heights == null || found.Heights.Length == 0)
            return false;

        chunk = found;
        return true;
    }

    private bool TryGetTileChunksForEdit(int tileX, int tileY, out List<Terrain.TerrainChunkData> chunks)
    {
        chunks = new List<Terrain.TerrainChunkData>();

        if (_terrainManager != null)
        {
            var tile = _terrainManager.GetOrLoadTileLoadResult(tileX, tileY);
            chunks = tile.Chunks;
            return chunks.Count > 0;
        }

        if (_vlmTerrainManager != null)
        {
            if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
            {
                chunks = tile.Chunks;
                return chunks.Count > 0;
            }

            tile = _vlmTerrainManager.Loader.LoadTile(tileX, tileY);
            chunks = tile.Chunks;
            return chunks.Count > 0;
        }

        return false;
    }

    private static bool AreLayersCompatible(Terrain.TerrainLayer[] a, Terrain.TerrainLayer[] b)
    {
        if (a.Length != b.Length)
            return false;

        for (int i = 0; i < a.Length; i++)
        {
            if (a[i].TextureIndex != b[i].TextureIndex)
                return false;
        }

        return true;
    }

    private static Dictionary<int, byte[]> CloneAlphaMaps(Dictionary<int, byte[]> maps)
    {
        var clone = new Dictionary<int, byte[]>(maps.Count);
        foreach (var (k, v) in maps)
            clone[k] = (byte[])v.Clone();
        return clone;
    }

    private void DrawMapConverterDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(580, 520), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 290,
            ImGui.GetIO().DisplaySize.Y / 2 - 260), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Map Converter", ref _showMapConverterDialog))
        {
            ImGui.TextWrapped("Convert maps between Alpha 0.5.3 monolithic WDT and split ADT formats, including LK 3.3.5 and no-MCIN later-era roots where supported.");
            ImGui.TextDisabled("Conversions write into timestamped project folders under the configured project root. Original source files are not overwritten.");
            ImGui.Spacing();

            // Direction selector
            ImGui.Text("Direction:");
            ImGui.RadioButton("Alpha WDT \u2192 LK ADTs", ref _mapConvertDirection, 0);
            ImGui.SameLine();
            ImGui.RadioButton("Split ADTs \u2192 Alpha WDT", ref _mapConvertDirection, 1);
            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Spacing();

            if (!string.IsNullOrWhiteSpace(_mapConvertSourcePath))
                EnsureMapConverterProjectOutputDirectory(forceNew: false);

            if (_mapConvertDirection == 0)
            {
                // Alpha → LK
                ImGui.Text("Source Alpha WDT:");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##a2l_src", ref _mapConvertSourcePath, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##a2l_src"))
                {
                    string? initDir = !string.IsNullOrEmpty(_mapConvertSourcePath) ? Path.GetDirectoryName(_mapConvertSourcePath) : null;
                    var picked = ShowFileDialogSTA("Select Alpha WDT file", "WDT Files (*.wdt)|*.wdt|All Files (*.*)|*.*", initDir);
                    if (picked != null)
                    {
                        _mapConvertSourcePath = picked;
                        EnsureMapConverterProjectOutputDirectory(forceNew: false);
                    }
                }

                ImGui.Checkbox("Copy source Alpha WDT into project", ref _mapConvertCopyAlphaSourceWdt);
                ImGui.Checkbox("Emit converted LK split outputs", ref _mapConvertEmitLkSplitOutputs);
            }
            else
            {
                // Split ADTs → Alpha
                ImGui.Text("Source Split-ADT WDT:");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##l2a_src", ref _mapConvertSourcePath, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##l2a_src"))
                {
                    string? initDir = !string.IsNullOrEmpty(_mapConvertSourcePath) ? Path.GetDirectoryName(_mapConvertSourcePath) : null;
                    var picked = ShowFileDialogSTA("Select split-ADT WDT file", "WDT Files (*.wdt)|*.wdt|All Files (*.*)|*.*", initDir);
                    if (picked != null)
                    {
                        _mapConvertSourcePath = picked;
                        EnsureMapConverterProjectOutputDirectory(forceNew: false);
                    }
                }

                ImGui.Text("Split ADT Directory (containing MapName_X_Y.adt roots):");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##l2a_mapdir", ref _mapConvertLkMapDir, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##l2a_dir"))
                {
                    var picked = ShowFolderDialogSTA("Select directory containing split ADT files");
                    if (picked != null) _mapConvertLkMapDir = picked;
                }
            }

            ImGui.Spacing();
            ImGui.Text("Project Output Root:");
            ImGui.SetNextItemWidth(-80);
            if (ImGui.InputText("##mapconv_project_root", ref _projectOutputRootDir, 512))
                HandleProjectOutputRootChanged();
            ImGui.SameLine();
            if (ImGui.Button("Browse##mapconv_project_root"))
            {
                string? picked = ShowFolderDialogSTA("Select project output root", GetProjectOutputRootDirectory(), showNewFolderButton: true);
                if (!string.IsNullOrWhiteSpace(picked))
                {
                    _projectOutputRootDir = picked;
                    HandleProjectOutputRootChanged();
                }
            }

            ImGui.TextWrapped($"Project Folder: {DescribeMapConverterProjectOutputDirectory()}");
            if (ImGui.Button("New Project Folder##mapconv"))
                EnsureMapConverterProjectOutputDirectory(forceNew: true);

            ImGui.Spacing();
            ImGui.Checkbox("Verbose logging", ref _mapConvertVerbose);
            ImGui.Spacing();

            if (_mapConvertDirection == 0)
            {
                ImGui.Text("Area Crosswalk CSV (optional):");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##area_crosswalk", ref _areaCrosswalkPath, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##area_crosswalk"))
                {
                    var picked = ShowFileDialogSTA("Select area crosswalk CSV", "CSV Files (*.csv)|*.csv|All Files (*.*)|*.*", null);
                    if (picked != null) _areaCrosswalkPath = picked;
                }
                ImGui.SameLine();
                ImGui.TextDisabled("Maps area IDs for Alpha→LK conversion");
            }

            if (_mapConvertDirection == 1 && !string.IsNullOrEmpty(_mapConvertSourcePath) && string.IsNullOrEmpty(_mapConvertLkMapDir))
            {
                _mapConvertLkMapDir = Path.GetDirectoryName(_mapConvertSourcePath) ?? "";
            }

            // Convert button
            bool canConvert = !_mapConverting
                && !string.IsNullOrWhiteSpace(_mapConvertSourcePath)
                && !string.IsNullOrWhiteSpace(_mapConvertOutputDir)
                && (_mapConvertDirection != 0 || _mapConvertCopyAlphaSourceWdt || _mapConvertEmitLkSplitOutputs)
                && (_mapConvertDirection == 0 || !string.IsNullOrWhiteSpace(_mapConvertLkMapDir));

            if (!canConvert) ImGui.BeginDisabled();
            if (ImGui.Button(_mapConverting ? "Converting..." : "Convert", new Vector2(120, 0)))
            {
                _mapConvertLog.Clear();
                _mapConvertError = null;
                _mapConvertDone = false;
                _mapConverting = true;

                string srcPath = _mapConvertSourcePath;
                string outPath = _mapConvertOutputDir;
                string lkMapDir = _mapConvertLkMapDir;
                int direction = _mapConvertDirection;
                bool verbose = _mapConvertVerbose;
                bool copyAlphaSourceWdt = _mapConvertCopyAlphaSourceWdt;
                bool emitLkSplitOutputs = _mapConvertEmitLkSplitOutputs;
                _mapConvertLastLoadPath = null;

                Task.Run(async () =>
                {
                    try
                    {
                        if (direction == 0)
                        {
                            string alphaCopyPath = BuildMapConverterAlphaSourceCopyPath(outPath, srcPath);
                            string lkOutputDir = BuildMapConverterLkOutputDirectory(outPath, srcPath);
                            string lkLoadPath = Path.Combine(lkOutputDir, Path.GetFileNameWithoutExtension(srcPath) + ".wdt");

                            if (copyAlphaSourceWdt)
                            {
                                Directory.CreateDirectory(Path.GetDirectoryName(alphaCopyPath)!);
                                File.Copy(srcPath, alphaCopyPath, overwrite: true);
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"Copied Alpha source WDT: {alphaCopyPath}");
                                _mapConvertScrollToBottom = true;
                                _mapConvertLastLoadPath = alphaCopyPath;
                            }

                            if (emitLkSplitOutputs)
                            {
                                // Execute the converter CLI and capture output
                                var converterExe = FindConverterExecutable();
                                if (string.IsNullOrEmpty(converterExe))
                                {
                                    _mapConvertError = "Converter executable not found. Build the project first.";
                                    lock (_mapConvertLog)
                                        _mapConvertLog.Add($"\n=== ERROR: {_mapConvertError} ===");
                                }
                                else
                                {
                                    var args = new List<string>
                                    {
                                        "convert-alpha-to-lk",
                                        "--input", srcPath,
                                        "--output", lkOutputDir
                                    };
                                    if (verbose) args.Add("--verbose");
                                    if (!string.IsNullOrWhiteSpace(_areaCrosswalkPath))
                                    {
                                        args.Add("--area-crosswalk");
                                        args.Add(_areaCrosswalkPath);
                                    }

                                    var result = await RunConverterAsync(converterExe, args, _mapConvertLog, _mapConvertScrollToBottom);
                                    if (!result.Success)
                                    {
                                        _mapConvertError = result.Error ?? "Conversion failed";
                                    }
                                    else
                                    {
                                        _mapConvertLastLoadPath = lkLoadPath;
                                        lock (_mapConvertLog)
                                            _mapConvertLog.Add($"\n=== SUCCESS: {result.TilesConverted}/{result.TotalTiles} tiles converted in {result.ElapsedMs}ms ===");
                                        lock (_mapConvertLog)
                                            _mapConvertLog.Add($"Project outputs: alpha-source={(copyAlphaSourceWdt ? alphaCopyPath : "skipped")}, lk-split={lkOutputDir}");
                                    }
                                }
                            }
                            else
                            {
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== SUCCESS: source Alpha WDT copied to project only ({alphaCopyPath}) ===");
                            }
                        }
                        else
                        {
                            string alphaOutputPath = BuildMapConverterAlphaOutputPath(outPath, srcPath);

                            // Execute the converter CLI and capture output
                            var converterExe = FindConverterExecutable();
                            if (string.IsNullOrEmpty(converterExe))
                            {
                                _mapConvertError = "Converter executable not found. Build the project first.";
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== ERROR: {_mapConvertError} ===");
                            }
                            else
                            {
                                var args = new List<string>
                                {
                                    "convert-split-adt-to-lk",
                                    "--input-root", srcPath,
                                    "--output", alphaOutputPath
                                };
                                if (verbose) args.Add("--verbose");

                                var result = await RunConverterAsync(converterExe, args, _mapConvertLog, _mapConvertScrollToBottom);
                                if (!result.Success)
                                {
                                    _mapConvertError = result.Error ?? "Conversion failed";
                                }
                                else
                                {
                                    _mapConvertLastLoadPath = alphaOutputPath;
                                    lock (_mapConvertLog)
                                        _mapConvertLog.Add($"\n=== SUCCESS: {result.TilesConverted}/{result.TotalTiles} tiles converted in {result.ElapsedMs}ms ===");
                                    lock (_mapConvertLog)
                                        _mapConvertLog.Add($"Project alpha output: {alphaOutputPath}");
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _mapConvertError = ex.Message;
                        lock (_mapConvertLog)
                            _mapConvertLog.Add($"\n=== EXCEPTION: {ex.Message} ===");
                    }
                    finally
                    {
                        _mapConvertDone = true;
                        _mapConverting = false;
                        _mapConvertScrollToBottom = true;
                    }
                });
            }
            if (!canConvert) ImGui.EndDisabled();

            ImGui.SameLine();
            if (ImGui.Button("Close", new Vector2(80, 0)))
                _showMapConverterDialog = false;

            // Error display
            if (_mapConvertError != null)
            {
                ImGui.Spacing();
                ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(1, 0.3f, 0.3f, 1));
                ImGui.TextWrapped($"Error: {_mapConvertError}");
                ImGui.PopStyleColor();
            }

            // Log output
            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Text("Log:");
            float logHeight = ImGui.GetContentRegionAvail().Y - 4;
            if (ImGui.BeginChild("##mapconv_log", new Vector2(-1, logHeight), true))
            {
                lock (_mapConvertLog)
                {
                    foreach (var line in _mapConvertLog)
                        ImGui.TextUnformatted(line);
                }
                if (_mapConvertScrollToBottom)
                {
                    ImGui.SetScrollHereY(1.0f);
                    _mapConvertScrollToBottom = false;
                }
            }
            ImGui.EndChild();

            // Load result button
            if (_mapConvertDone && _mapConvertError == null && _mapConvertDirection == 0)
            {
                if (ImGui.Button("Load Converted Map in Viewer"))
                {
                    if (!string.IsNullOrWhiteSpace(_mapConvertLastLoadPath) && File.Exists(_mapConvertLastLoadPath))
                    {
                        LoadWdtTerrain(_mapConvertLastLoadPath);
                        _showMapConverterDialog = false;
                    }
                }
            }
            else if (_mapConvertDone && _mapConvertError == null && _mapConvertDirection == 1)
            {
                if (ImGui.Button("Load Converted Alpha WDT in Viewer"))
                {
                    if (!string.IsNullOrWhiteSpace(_mapConvertLastLoadPath) && File.Exists(_mapConvertLastLoadPath))
                    {
                        LoadWdtTerrain(_mapConvertLastLoadPath);
                        _showMapConverterDialog = false;
                    }
                }
            }
        }
        ImGui.End();
    }

    private void DrawWmoConverterDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(580, 520), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 290,
            ImGui.GetIO().DisplaySize.Y / 2 - 260), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("WMO Converter", ref _showWmoConverterDialog))
        {
            ImGui.TextWrapped("Convert WMO objects between Alpha 0.5.3 (v14/v16) and LK 3.3.5 (v17) formats.");
            ImGui.Spacing();

            ImGui.Text("Direction:");
            ImGui.RadioButton("Alpha WMO → LK WMO", ref _wmoConvertDirection, 0);
            ImGui.SameLine();
            ImGui.RadioButton("LK WMO → Alpha WMO", ref _wmoConvertDirection, 1);
            ImGui.Spacing();
            ImGui.TextWrapped("The maintained converter path is now the only active path in this dialog.");

            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Spacing();

            // Auto-select currently loaded WMO
            if (!string.IsNullOrEmpty(_loadedFilePath)
                && string.Equals(Path.GetExtension(_loadedFilePath), ".wmo", StringComparison.OrdinalIgnoreCase)
                && string.IsNullOrEmpty(_wmoConvertSourcePath))
            {
                _wmoConvertSourcePath = _loadedFilePath;
                if (string.IsNullOrWhiteSpace(_wmoConvertOutputPath))
                    _wmoConvertOutputPath = GetDefaultWmoConverterOutputDirectory();
            }

            ImGui.Text("Source WMO:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##wmo_src", ref _wmoConvertSourcePath, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##wmo_src"))
            {
                string? initDir = !string.IsNullOrEmpty(_wmoConvertSourcePath) ? Path.GetDirectoryName(_wmoConvertSourcePath) : null;
                var picked = ShowFileDialogSTA("Select WMO file", "WMO Files (*.wmo)|*.wmo|All Files (*.*)|*.*", initDir);
                if (picked != null)
                {
                    _wmoConvertSourcePath = picked;
                    if (string.IsNullOrWhiteSpace(_wmoConvertOutputPath))
                        _wmoConvertOutputPath = GetDefaultWmoConverterOutputDirectory();
                }
            }

            if (string.IsNullOrWhiteSpace(_wmoConvertOutputPath))
                _wmoConvertOutputPath = GetDefaultWmoConverterOutputDirectory();

            ImGui.Text("Output Folder:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##wmo_out_dir", ref _wmoConvertOutputPath, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##wmo_out_dir"))
            {
                string? picked = ShowFolderDialogSTA(
                    "Select output directory for converted WMO files",
                    GetDefaultWmoConverterOutputDirectory(),
                    showNewFolderButton: true);
                if (picked != null)
                    _wmoConvertOutputPath = picked;
            }

            string outputRootPath = "";
            if (!string.IsNullOrWhiteSpace(_wmoConvertSourcePath)
                && !string.IsNullOrWhiteSpace(_wmoConvertOutputPath))
            {
                string baseName = Path.GetFileNameWithoutExtension(_wmoConvertSourcePath);
                string suffix = (_wmoConvertDirection == 0) ? ".v17.wmo" : ".v14.wmo";
                outputRootPath = Path.Combine(Path.GetFullPath(_wmoConvertOutputPath), baseName + suffix);
            }

            ImGui.Text("Resolved Output File:");
            ImGui.SetNextItemWidth(-1);
            ImGui.BeginDisabled();
            ImGui.InputText("##wmo_out", ref outputRootPath, 512);
            ImGui.EndDisabled();

            ImGui.Spacing();
            ImGui.Checkbox("Copy referenced textures (best-effort)", ref _wmoConvertCopyTextures);
            ImGui.Spacing();

            bool canConvert = !_wmoConverting
                && !string.IsNullOrWhiteSpace(_wmoConvertSourcePath)
                && !string.IsNullOrWhiteSpace(outputRootPath);

            if (!canConvert) ImGui.BeginDisabled();
            if (ImGui.Button(_wmoConverting ? "Converting..." : "Convert", new Vector2(120, 0)))
            {
                _wmoConvertLog.Clear();
                _wmoConvertError = null;
                _wmoConvertDone = false;
                _wmoConverting = true;

                string srcPath = _wmoConvertSourcePath;
                string outPath = outputRootPath;
                int direction = _wmoConvertDirection;
                bool copyTextures = _wmoConvertCopyTextures;
                var dataSource = _dataSource;

                Task.Run(async () =>
                {
                    try
                    {
                        var converterExe = FindConverterExecutable();
                        if (string.IsNullOrEmpty(converterExe))
                        {
                            _wmoConvertError = "Converter executable not found. Build the project first.";
                            lock (_wmoConvertLog)
                                _wmoConvertLog.Add($"\n=== ERROR: {_wmoConvertError} ===");
                            _wmoConvertScrollToBottom = true;
                            return;
                        }

                        var args = new List<string>();
                        if (direction == 0)
                        {
                            args.Add("convert-wmo-v14-to-v17");
                        }
                        else
                        {
                            args.Add("convert-wmo-v17-to-v14");
                        }
                        args.Add("--input-root");
                        args.Add(srcPath);
                        args.Add("--output");
                        args.Add(outPath);
                        if (copyTextures) args.Add("--copy-textures");

                        var result = await RunConverterAsync(converterExe, args, _wmoConvertLog, _wmoConvertScrollToBottom);
                        if (!result.Success)
                        {
                            _wmoConvertError = result.Error ?? "Conversion failed";
                        }
                        else
                        {
                            lock (_wmoConvertLog)
                            {
                                _wmoConvertLog.Add("\n=== SUCCESS ===");
                                _wmoConvertLog.Add($"Wrote: {outPath}");
                            }
                        }

                        _wmoConvertScrollToBottom = true;
                    }
                    catch (Exception ex)
                    {
                        _wmoConvertError = ex.Message;
                        lock (_wmoConvertLog)
                            _wmoConvertLog.Add($"\n=== EXCEPTION: {ex.Message} ===");
                        _wmoConvertScrollToBottom = true;
                    }
                    finally
                    {
                        _wmoConvertDone = true;
                        _wmoConverting = false;
                        _wmoConvertScrollToBottom = true;
                    }
                });
            }
            if (!canConvert) ImGui.EndDisabled();

            ImGui.SameLine();
            if (ImGui.Button("Close", new Vector2(120, 0)))
                _showWmoConverterDialog = false;

            ImGui.Spacing();
            if (_wmoConvertDone)
            {
                if (_wmoConvertError != null)
                    ImGui.TextColored(new Vector4(1, 0.3f, 0.3f, 1), $"Error: {_wmoConvertError}");
                else
                    ImGui.TextColored(new Vector4(0.3f, 1, 0.3f, 1), "Done.");
            }

            ImGui.Separator();

            float logHeight = ImGui.GetContentRegionAvail().Y - 4;
            if (ImGui.BeginChild("##wmoconv_log", new Vector2(-1, logHeight), true))
            {
                lock (_wmoConvertLog)
                {
                    foreach (var line in _wmoConvertLog)
                        ImGui.TextUnformatted(line);
                }
                if (_wmoConvertScrollToBottom)
                {
                    ImGui.SetScrollHereY(1.0f);
                    _wmoConvertScrollToBottom = false;
                }
                ImGui.EndChild();
            }
        }
        ImGui.End();
    }

    private static void CopyWmoTexturesPreservePaths(string inputWmoPath, string outputWmoPath, List<string> textures, IDataSource? dataSource)
    {
        if (textures.Count == 0) return;
        string outputDir = Path.GetDirectoryName(Path.GetFullPath(outputWmoPath)) ?? ".";
        
        foreach (var tex in textures)
        {
            var cleanTex = tex.Replace('/', '\\');
            byte[]? blpData = null;

            // Try to read from data source (MPQ) first for version-correct assets
            if (dataSource != null)
            {
                blpData = dataSource.ReadFile(tex);
                if (blpData == null)
                {
                    // Try normalized path
                    blpData = dataSource.ReadFile(cleanTex);
                }
            }

            if (blpData != null && blpData.Length > 0)
            {
                // Write preserving original folder structure
                var destPath = Path.Combine(outputDir, cleanTex);
                Directory.CreateDirectory(Path.GetDirectoryName(destPath) ?? outputDir);
                File.WriteAllBytes(destPath, blpData);
            }
            else
            {
                // Fallback to best-effort filesystem copy
                CopyWmoTexturesBestEffort(inputWmoPath, outputWmoPath, new List<string> { tex });
            }
        }
    }

    private static void CopyWmoTexturesBestEffort(string inputWmoPath, string outputWmoPath, List<string> textures)
    {
        if (textures.Count == 0) return;
        string inputDir = Path.GetDirectoryName(Path.GetFullPath(inputWmoPath)) ?? ".";
        string outputDir = Path.GetDirectoryName(Path.GetFullPath(outputWmoPath)) ?? ".";
        foreach (var tex in textures)
        {
            var cleanTex = tex.Replace('/', '\\');
            string? srcPath = null;

            var p1 = Path.Combine(inputDir, cleanTex);
            if (File.Exists(p1)) srcPath = p1;
            else
            {
                var curr = new DirectoryInfo(inputDir);
                DirectoryInfo? rootDir = null;
                for (int i = 0; i < 5 && curr != null; i++)
                {
                    var p2 = Path.Combine(curr.FullName, cleanTex);
                    if (File.Exists(p2))
                    {
                        srcPath = p2;
                        break;
                    }
                    if (Directory.Exists(Path.Combine(curr.FullName, "DUNGEONS"))
                        || Directory.Exists(Path.Combine(curr.FullName, "World"))
                        || Directory.Exists(Path.Combine(curr.FullName, "Textures")))
                    {
                        rootDir = curr;
                    }
                    curr = curr.Parent;
                }

                if (srcPath == null)
                {
                    var searchRoot = rootDir ?? new DirectoryInfo(inputDir).Parent?.Parent;
                    if (searchRoot != null && searchRoot.Exists)
                    {
                        var filename = Path.GetFileName(cleanTex);
                        srcPath = Directory.EnumerateFiles(searchRoot.FullName, filename, SearchOption.AllDirectories)
                            .FirstOrDefault();
                    }
                }
            }

            if (srcPath == null) continue;
            string targetRelPath = cleanTex;
            var destPath = Path.Combine(outputDir, targetRelPath);
            Directory.CreateDirectory(Path.GetDirectoryName(destPath) ?? outputDir);
            File.Copy(srcPath, destPath, true);
        }
    }

    private void DrawVlmExportDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(550, 500), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 275,
            ImGui.GetIO().DisplaySize.Y / 2 - 250), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Build ML Dataset", ref _showVlmExportDialog))
        {
            PrepareMkHarvestDialogInputs();

            ImGui.TextWrapped("Export terrain data from a WoW client folder into an ML dataset (JSON + PNG), then optionally build the manifest, baked references, and live WoWViewer validation captures in the same flow. " +
                "Supports Alpha 0.5.3 through Cataclysm 4.0.0.11927 (with additional later-era paths still under validation).");
            ImGui.Spacing();

            // Client Path
            ImGui.Text("Client Data Path:");
            ImGui.SetNextItemWidth(-80);
            string prevClient = _vlmClientPath;
            ImGui.InputText("##vlmClient", ref _vlmClientPath, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##client"))
            {
                string? result = ShowFolderDialogSTA("Select WoW Client Data Folder");
                if (result != null) _vlmClientPath = result;
            }

            // Map Name
            ImGui.Text("Map Name:");
            ImGui.SetNextItemWidth(-1);
            string prevMap = _vlmMapName;
            ImGui.InputText("##vlmMap", ref _vlmMapName, 128);
            ImGui.TextColored(new Vector4(0.6f, 0.6f, 0.6f, 1f),
                "e.g. development, Azeroth, Kalimdor, PVPZone01");

            // Auto-generate output directory when client path or map name changes
            if ((_vlmClientPath != prevClient || _vlmMapName != prevMap) &&
                !string.IsNullOrWhiteSpace(_vlmClientPath) && !string.IsNullOrWhiteSpace(_vlmMapName))
            {
                _vlmOutputDir = GenerateVlmOutputPath(_vlmClientPath, _vlmMapName);
            }

            // Output Directory
            ImGui.Text("Output Directory:");
            ImGui.SetNextItemWidth(-80);
            string prevOutputDir = _vlmOutputDir;
            ImGui.InputText("##vlmOutput", ref _vlmOutputDir, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##output"))
            {
                string? result = ShowFolderDialogSTA("Select Output Directory");
                if (result != null) _vlmOutputDir = result;
            }

            if (!string.Equals(prevOutputDir, _vlmOutputDir, StringComparison.OrdinalIgnoreCase)
                && (string.IsNullOrWhiteSpace(_mkHarvestDatasetRoot)
                    || string.Equals(_mkHarvestDatasetRoot, prevOutputDir, StringComparison.OrdinalIgnoreCase)))
            {
                SyncMkHarvestDerivedPaths(prevOutputDir, _vlmOutputDir);
            }

            // Tile Limit
            ImGui.Text("Tile Limit (0 = all):");
            ImGui.SetNextItemWidth(120);
            ImGui.InputInt("##vlmLimit", ref _vlmTileLimit);
            if (_vlmTileLimit < 0) _vlmTileLimit = 0;

            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Spacing();

            // Export button
            bool canExport = !_vlmExporting &&
                !string.IsNullOrWhiteSpace(_vlmClientPath) &&
                !string.IsNullOrWhiteSpace(_vlmMapName) &&
                !string.IsNullOrWhiteSpace(_vlmOutputDir);

            if (!canExport) ImGui.BeginDisabled();
            if (ImGui.Button("Build Dataset", new Vector2(140, 30)))
            {
                StartVlmExport();
            }
            if (!canExport) ImGui.EndDisabled();

            if (_vlmExporting)
            {
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(1f, 1f, 0f, 1f), "Exporting...");
            }
            else if (_vlmExportResult != null)
            {
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0f, 1f, 0f, 1f),
                    $"Done: {_vlmExportResult.TilesExported} tiles, {_vlmExportResult.UniqueTextures} textures");

                ImGui.SameLine();
                if (ImGui.Button("Open in Viewer"))
                {
                    var datasetDir = Path.Combine(_vlmExportResult.OutputDirectory, "dataset");
                    if (Directory.Exists(datasetDir))
                        LoadVlmProject(_vlmExportResult.OutputDirectory);
                    else
                        LoadVlmProject(_vlmExportResult.OutputDirectory);
                    _showVlmExportDialog = false;
                }

            }

            DrawMlFinalizeSection(showLoadDatasetButton: _vlmExportResult != null);

            // Progress log
            ImGui.Spacing();
            ImGui.Text("Export Log:");
            float logHeight = MathF.Max(120f, ImGui.GetContentRegionAvail().Y - 4);
            if (ImGui.BeginChild("VlmExportLog", new Vector2(-1, logHeight), true))
            {
                lock (_vlmExportLog)
                {
                    foreach (var line in _vlmExportLog)
                        ImGui.TextWrapped(line);
                }
                if (_vlmExportScrollToBottom)
                {
                    ImGui.SetScrollHereY(1.0f);
                    _vlmExportScrollToBottom = false;
                }
            }
            ImGui.EndChild();
        }
        ImGui.End();
    }

    private void DrawMlFinalizeSection(bool showLoadDatasetButton)
    {
        ImGui.Spacing();
        ImGui.Separator();
        ImGui.Spacing();
        ImGui.Text("ML Dataset Manifest + Validation");
        ImGui.TextWrapped("Auto-finalization runs after export. Status is shown below.");
        ImGui.Spacing();

        ImGui.Checkbox("Run manifest + validation automatically after export", ref _mlFinalizeAfterExport);
    }

    private void PromotePendingMlFinalizeAfterExport()
    {
        if (!_pendingMlFinalizeAfterExport || _vlmExporting || _mkHarvestRunning)
            return;

        _pendingMlFinalizeAfterExport = false;

        if (_vlmExportResult == null || string.IsNullOrWhiteSpace(_vlmExportResult.OutputDirectory))
            return;

        SyncMkHarvestDerivedPaths(_mkHarvestDatasetRoot, _vlmExportResult.OutputDirectory);
        StartMkHarvest();
        AppendMkHarvestLogLine("Started manifest + validation automatically after dataset export.");
    }

    private void SyncMkHarvestDerivedPaths(string? previousDatasetRoot, string? nextDatasetRoot)
    {
        string normalizedNextDatasetRoot = nextDatasetRoot ?? string.Empty;
        string oldManifestDefault = GenerateMkHarvestManifestPath(previousDatasetRoot);
        string oldReferenceDefault = GenerateMkReferenceMinimapDirectory(previousDatasetRoot);
        string oldViewerValidationDefault = GenerateMkViewerValidationMinimapDirectory(previousDatasetRoot);

        _mkHarvestDatasetRoot = normalizedNextDatasetRoot;

        if (string.IsNullOrWhiteSpace(_mkHarvestManifestOutputPath)
            || string.Equals(_mkHarvestManifestOutputPath, oldManifestDefault, StringComparison.OrdinalIgnoreCase))
        {
            _mkHarvestManifestOutputPath = GenerateMkHarvestManifestPath(normalizedNextDatasetRoot);
        }

        if (string.IsNullOrWhiteSpace(_mkHarvestReferenceOutputDir)
            || string.Equals(_mkHarvestReferenceOutputDir, oldReferenceDefault, StringComparison.OrdinalIgnoreCase))
        {
            _mkHarvestReferenceOutputDir = GenerateMkReferenceMinimapDirectory(normalizedNextDatasetRoot);
        }

        if (string.IsNullOrWhiteSpace(_mkHarvestViewerValidationOutputDir)
            || string.Equals(_mkHarvestViewerValidationOutputDir, oldViewerValidationDefault, StringComparison.OrdinalIgnoreCase))
        {
            _mkHarvestViewerValidationOutputDir = GenerateMkViewerValidationMinimapDirectory(normalizedNextDatasetRoot);
        }
    }

    private void DrawTerrainTextureTransferDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(650, 620), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 325,
            ImGui.GetIO().DisplaySize.Y / 2 - 310), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Terrain Texture Transfer", ref _showTerrainTextureTransferDialog))
        {
            ImGui.TextWrapped("Run mapped terrain texture transfer using the backend service (MTEX/MCLY/MCAL/MCSH/holes). " +
                "Use explicit tile pair mode for surgical edits or global delta mode for batched remap runs.");
            ImGui.Spacing();

            ImGui.Text("Source Map Directory:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##ttt_source", ref _terrainTransferSourceDir, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##ttt_source"))
            {
                string? picked = ShowFolderDialogSTA("Select source map directory", _terrainTransferSourceDir);
                if (!string.IsNullOrWhiteSpace(picked))
                    _terrainTransferSourceDir = picked;
            }

            ImGui.Text("Target Map Directory:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##ttt_target", ref _terrainTransferTargetDir, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##ttt_target"))
            {
                string? picked = ShowFolderDialogSTA("Select target map directory", _terrainTransferTargetDir);
                if (!string.IsNullOrWhiteSpace(picked))
                    _terrainTransferTargetDir = picked;
            }

            ImGui.Text("Output Directory:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##ttt_output", ref _terrainTransferOutputDir, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##ttt_output"))
            {
                string? picked = ShowFolderDialogSTA("Select output directory", _terrainTransferOutputDir);
                if (!string.IsNullOrWhiteSpace(picked))
                    _terrainTransferOutputDir = picked;
            }

            ImGui.Text("Mode:");
            if (ImGui.RadioButton("Dry Run", !_terrainTransferApplyMode))
                _terrainTransferApplyMode = false;
            ImGui.SameLine();
            if (ImGui.RadioButton("Apply", _terrainTransferApplyMode))
                _terrainTransferApplyMode = true;

            ImGui.Text("Mapping:");
            if (ImGui.RadioButton("Explicit Pair", !_terrainTransferUseGlobalDelta))
                _terrainTransferUseGlobalDelta = false;
            ImGui.SameLine();
            if (ImGui.RadioButton("Global Delta", _terrainTransferUseGlobalDelta))
                _terrainTransferUseGlobalDelta = true;

            if (_terrainTransferUseGlobalDelta)
            {
                ImGui.InputInt("Delta X", ref _terrainTransferDeltaX);
                ImGui.InputInt("Delta Y", ref _terrainTransferDeltaY);
                ImGui.InputInt("Tile Limit (0=all)", ref _terrainTransferTileLimit);
                if (_terrainTransferTileLimit < 0)
                    _terrainTransferTileLimit = 0;
            }
            else
            {
                ImGui.InputInt("Source Tile X", ref _terrainTransferSourceTileX);
                ImGui.InputInt("Source Tile Y", ref _terrainTransferSourceTileY);
                ImGui.InputInt("Target Tile X", ref _terrainTransferTargetTileX);
                ImGui.InputInt("Target Tile Y", ref _terrainTransferTargetTileY);
            }

            ImGui.InputInt("Chunk Offset X", ref _terrainTransferChunkOffsetX);
            ImGui.InputInt("Chunk Offset Y", ref _terrainTransferChunkOffsetY);

            ImGui.Text("Payload:");
            ImGui.Checkbox("MTEX", ref _terrainTransferCopyMtex);
            ImGui.SameLine();
            ImGui.Checkbox("MCLY", ref _terrainTransferCopyMcly);
            ImGui.SameLine();
            ImGui.Checkbox("MCAL", ref _terrainTransferCopyMcal);
            ImGui.SameLine();
            ImGui.Checkbox("MCSH", ref _terrainTransferCopyMcsh);
            ImGui.SameLine();
            ImGui.Checkbox("Holes", ref _terrainTransferCopyHoles);

            ImGui.Text("Summary Manifest Path (optional):");
            ImGui.SetNextItemWidth(-1);
            ImGui.InputText("##ttt_manifest", ref _terrainTransferManifestPath, 512);

            ImGui.Spacing();
            bool canRun = !_terrainTransferRunning
                && !string.IsNullOrWhiteSpace(_terrainTransferSourceDir)
                && !string.IsNullOrWhiteSpace(_terrainTransferTargetDir)
                && !string.IsNullOrWhiteSpace(_terrainTransferOutputDir);

            if (!canRun)
                ImGui.BeginDisabled();

            if (ImGui.Button(_terrainTransferRunning ? "Running..." : "Run Transfer", new Vector2(140, 30)))
            {
                StartTerrainTextureTransfer();
            }

            if (!canRun)
                ImGui.EndDisabled();

            ImGui.SameLine();
            if (ImGui.Button("Close", new Vector2(80, 30)))
                _showTerrainTextureTransferDialog = false;

            if (_terrainTransferRunning)
            {
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(1f, 1f, 0f, 1f), "Running...");
            }
            else if (_terrainTransferReport != null && _terrainTransferError == null)
            {
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0f, 1f, 0f, 1f),
                    $"Done: {_terrainTransferReport.TilesProcessed} processed, {_terrainTransferReport.TilesWritten} written, {_terrainTransferReport.TilesNeedingManualReview} review");
            }

            if (_terrainTransferError != null)
            {
                ImGui.TextColored(new Vector4(1f, 0.3f, 0.3f, 1f), $"Error: {_terrainTransferError}");
            }

            ImGui.Spacing();
            ImGui.Text("Log:");
            float logHeight = ImGui.GetContentRegionAvail().Y - 4;
            if (ImGui.BeginChild("TerrainTextureTransferLog", new Vector2(-1, logHeight), true))
            {
                lock (_terrainTransferLog)
                {
                    foreach (string line in _terrainTransferLog)
                        ImGui.TextWrapped(line);
                }

                if (_terrainTransferScrollToBottom)
                {
                    ImGui.SetScrollHereY(1.0f);
                    _terrainTransferScrollToBottom = false;
                }
            }
            ImGui.EndChild();
        }
        ImGui.End();
    }

    /// <summary>
    /// Generate a versioned output folder path for VLM dataset export.
    /// Format: {clientParent}/vlm_datasets/{mapName}_v{N}
    /// </summary>
    private static string GenerateVlmOutputPath(string clientPath, string mapName)
    {
        string baseDir = Path.Combine(Path.GetDirectoryName(clientPath) ?? clientPath, "vlm_datasets");
        string prefix = $"{mapName}_v";
        int version = 1;
        if (Directory.Exists(baseDir))
        {
            foreach (var dir in Directory.GetDirectories(baseDir, $"{mapName}_v*"))
            {
                string name = Path.GetFileName(dir);
                if (name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) &&
                    int.TryParse(name.Substring(prefix.Length), out int v) && v >= version)
                    version = v + 1;
            }
        }
        return Path.Combine(baseDir, $"{prefix}{version}");
    }

    private static string GenerateMkHarvestManifestPath(string? datasetRoot)
    {
        if (string.IsNullOrWhiteSpace(datasetRoot))
            return string.Empty;

        return Path.Combine(datasetRoot, "ml_dataset_manifest.json");
    }

    private static string GenerateMkReferenceMinimapDirectory(string? datasetRoot)
    {
        if (string.IsNullOrWhiteSpace(datasetRoot))
            return string.Empty;

        return Path.Combine(datasetRoot, "reference_minimaps");
    }

    private static string GenerateMkViewerValidationMinimapDirectory(string? datasetRoot)
    {
        if (string.IsNullOrWhiteSpace(datasetRoot))
            return string.Empty;

        return Path.Combine(datasetRoot, "viewer_validation_minimaps");
    }

    private void AppendMkHarvestLogLine(string message)
    {
        lock (_mkHarvestLog)
        {
            _mkHarvestLog.Add(message);
            if (_mkHarvestLog.Count > 2000)
                _mkHarvestLog.RemoveRange(0, _mkHarvestLog.Count - 1500);
        }

        _mkHarvestScrollToBottom = true;
    }

    private MkHarvestViewerValidationCapturePlan? BuildMkHarvestViewerValidationCapturePlan(
        string datasetRoot,
        string? outputDirectory,
        bool forceRegenerate,
        int requestedResolution,
        out string? statusMessage,
        int requiredSettledFrames = DefaultRequiredSettledFrames,
        int maxFramesBeforeCapture = DefaultMaxFramesBeforeCapture,
        int batchSettledFrames = DefaultBatchSettledFrames)
    {
        statusMessage = null;

        if (string.IsNullOrWhiteSpace(datasetRoot))
        {
            statusMessage = "Skipping WoWViewer validation captures because no dataset root was provided.";
            return null;
        }

        string normalizedDatasetRoot = Path.GetFullPath(datasetRoot);
        string datasetDirectory = Path.Combine(normalizedDatasetRoot, "dataset");
        if (!Directory.Exists(datasetDirectory))
        {
            statusMessage = $"Skipping WoWViewer validation captures because {datasetDirectory} does not exist.";
            return null;
        }

        string validationOutputDirectory = Path.GetFullPath(string.IsNullOrWhiteSpace(outputDirectory)
            ? GenerateMkViewerValidationMinimapDirectory(normalizedDatasetRoot)
            : outputDirectory);
        Directory.CreateDirectory(validationOutputDirectory);
        string validationNoLiquidsOutputDirectory = Path.Combine(validationOutputDirectory, "noliquids");
        Directory.CreateDirectory(validationNoLiquidsOutputDirectory);
        string validationNoObjectsOutputDirectory = Path.Combine(validationOutputDirectory, "noobjects");
        Directory.CreateDirectory(validationNoObjectsOutputDirectory);
        string validationObjectsOnlyOutputDirectory = Path.Combine(validationOutputDirectory, "objectsonly");
        Directory.CreateDirectory(validationObjectsOnlyOutputDirectory);

        var plan = new MkHarvestViewerValidationCapturePlan
        {
            DatasetRoot = normalizedDatasetRoot,
            OutputDirectory = validationOutputDirectory,
            NoLiquidsOutputDirectory = validationNoLiquidsOutputDirectory,
            NoObjectsOutputDirectory = validationNoObjectsOutputDirectory,
            ObjectsOnlyOutputDirectory = validationObjectsOnlyOutputDirectory,
            RequestedResolution = Math.Clamp(requestedResolution, 512, 4096),
            RequiredSettledFrames = requiredSettledFrames,
            MaxFramesBeforeCapture = maxFramesBeforeCapture,
            BatchSettledFrames = batchSettledFrames,
        };

        int skippedFiles = 0;
        foreach (string datasetFile in Directory.GetFiles(datasetDirectory, "*.json"))
        {
            string fileName = Path.GetFileName(datasetFile);
            if (string.Equals(fileName, "texture_database.json", StringComparison.OrdinalIgnoreCase))
                continue;

            string tileName = Path.GetFileNameWithoutExtension(datasetFile);
            if (!TryParseMkDatasetTileCoordinates(tileName, out string mapName, out int tileX, out int tileY))
            {
                skippedFiles++;
                continue;
            }

            if (string.IsNullOrWhiteSpace(plan.MapName))
                plan.MapName = mapName;

            string outputPath = Path.Combine(validationOutputDirectory, $"{tileName}_viewer_validation.png");
            if (forceRegenerate || !File.Exists(outputPath))
            {
                plan.Tiles.Add(new MkHarvestViewerValidationCaptureTile
                {
                    TileName = tileName,
                    TileX = tileX,
                    TileY = tileY,
                    OutputPath = outputPath,
                    HideTerrainLiquids = false,
                });
            }

            string noLiquidsOutputPath = Path.Combine(validationNoLiquidsOutputDirectory, $"{tileName}_viewer_validation.png");
            if (forceRegenerate || !File.Exists(noLiquidsOutputPath))
            {
                plan.Tiles.Add(new MkHarvestViewerValidationCaptureTile
                {
                    TileName = tileName,
                    TileX = tileX,
                    TileY = tileY,
                    OutputPath = noLiquidsOutputPath,
                    HideTerrainLiquids = true,
                });
            }

            string noObjectsOutputPath = Path.Combine(validationNoObjectsOutputDirectory, $"{tileName}_viewer_validation.png");
            if (forceRegenerate || !File.Exists(noObjectsOutputPath))
            {
                plan.Tiles.Add(new MkHarvestViewerValidationCaptureTile
                {
                    TileName = tileName,
                    TileX = tileX,
                    TileY = tileY,
                    OutputPath = noObjectsOutputPath,
                    HideTerrainLiquids = false,
                    HideObjects = true,
                });
            }

            string objectsOnlyOutputPath = Path.Combine(validationObjectsOnlyOutputDirectory, $"{tileName}_viewer_validation.png");
            if (forceRegenerate || !File.Exists(objectsOnlyOutputPath))
            {
                plan.Tiles.Add(new MkHarvestViewerValidationCaptureTile
                {
                    TileName = tileName,
                    TileX = tileX,
                    TileY = tileY,
                    OutputPath = objectsOnlyOutputPath,
                    HideTerrainLiquids = true,
                    HideTerrain = true,
                });
            }
        }

        plan.Tiles.Sort(static (left, right) =>
        {
            int mapCompare = string.Compare(left.TileName, right.TileName, StringComparison.OrdinalIgnoreCase);
            if (mapCompare != 0)
                return mapCompare;

            int tileXCompare = left.TileX.CompareTo(right.TileX);
            if (tileXCompare != 0)
                return tileXCompare;

            int tileYCompare = left.TileY.CompareTo(right.TileY);
            if (tileYCompare != 0)
                return tileYCompare;

            int terrainCompare = left.HideTerrain.CompareTo(right.HideTerrain);
            if (terrainCompare != 0)
                return terrainCompare;

            int objectCompare = left.HideObjects.CompareTo(right.HideObjects);
            if (objectCompare != 0)
                return objectCompare;

            return left.HideTerrainLiquids.CompareTo(right.HideTerrainLiquids);
        });

        if (string.IsNullOrWhiteSpace(plan.MapName))
        {
            statusMessage = "Skipping WoWViewer validation captures because no dataset tile names could be parsed.";
            return null;
        }

        if (plan.Tiles.Count == 0)
        {
            statusMessage = skippedFiles > 0
                ? $"No new WoWViewer validation captures were queued; {skippedFiles} dataset tile file(s) could not be parsed and the rest already had primary, noliquids, noobjects, and objectsonly outputs. Refreshing stitched composites from existing files."
                : "No new WoWViewer validation captures were queued because primary, noliquids, noobjects, and objectsonly outputs already exist for every dataset tile. Refreshing stitched composites from existing files.";
            return plan;
        }

        if (skippedFiles > 0)
            statusMessage = $"Queued {plan.Tiles.Count} WoWViewer validation capture(s) across the primary, noliquids, noobjects, and objectsonly output families; skipped {skippedFiles} dataset tile file(s) with unparseable names.";

        return plan;
    }

    private static bool TryParseMkDatasetTileCoordinates(string tileName, out string mapName, out int tileX, out int tileY)
    {
        mapName = string.Empty;
        tileX = 0;
        tileY = 0;

        string[] parts = tileName.Split('_');
        if (parts.Length < 3
            || !int.TryParse(parts[^2], out int fileX)
            || !int.TryParse(parts[^1], out int fileY))
        {
            return false;
        }

        mapName = string.Join("_", parts[..^2]);
        tileX = fileY;
        tileY = fileX;
        return !string.IsNullOrWhiteSpace(mapName);
    }

    private string GetProjectOutputRootDirectory()
    {
        if (string.IsNullOrWhiteSpace(_projectOutputRootDir))
            _projectOutputRootDir = ProjectsDir;

        return Path.GetFullPath(_projectOutputRootDir);
    }

    private string GetDefaultWmoConverterOutputDirectory()
    {
        if (!string.IsNullOrWhiteSpace(_wmoConvertOutputPath))
            return Path.GetFullPath(_wmoConvertOutputPath);

        if (!string.IsNullOrWhiteSpace(_wmoConvertSourcePath))
        {
            string? sourceDir = Path.GetDirectoryName(Path.GetFullPath(_wmoConvertSourcePath));
            if (!string.IsNullOrWhiteSpace(sourceDir))
                return sourceDir;
        }

        return GetProjectOutputRootDirectory();
    }

    private void HandleProjectOutputRootChanged()
    {
        _editorProjectOutputDir = string.Empty;
        _editorProjectSourceKey = string.Empty;
        _mapConvertOutputDir = string.Empty;
        _mapConvertProjectSourceKey = string.Empty;
        RefreshProjectManagedPlacementTargets();

        if (!string.IsNullOrWhiteSpace(_mapConvertSourcePath))
            EnsureMapConverterProjectOutputDirectory(forceNew: false);
    }

    private static string SanitizeProjectPathSegment(string value)
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

    private static string CreateTimestampedProjectOutputDirectory(string rootDirectory, string projectName)
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
        RefreshProjectManagedPlacementTargets();
        _selectedPlacementSaveStatus = $"Created new project output folder: {_editorProjectOutputDir}";
    }

    private bool IsProjectManagedOutputPath(string? outputPath)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
            return false;

        string fullPath = Path.GetFullPath(outputPath);
        string rootPath = GetProjectOutputRootDirectory();
        return fullPath.StartsWith(rootPath, StringComparison.OrdinalIgnoreCase);
    }

    private string BuildProjectManagedPlacementOutputPath(string sourcePath)
    {
        string normalizedSourcePath = sourcePath.Replace('/', '\\').TrimStart('\\');
        return Path.Combine(EnsureEditorProjectOutputDirectory(), "lk-split", normalizedSourcePath);
    }

    private void RefreshProjectManagedPlacementTargets()
    {
        foreach (string sourcePath in _stagedPlacementEdits.Values
            .Select(edit => edit.SourcePath)
            .Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!_placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? targetPath)
                || string.IsNullOrWhiteSpace(targetPath)
                || IsProjectManagedOutputPath(targetPath))
            {
                _placementSaveTargetsBySourcePath[sourcePath] = BuildProjectManagedPlacementOutputPath(sourcePath);
            }
        }

        if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath)
            && (string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                || IsProjectManagedOutputPath(_selectedPlacementSaveTargetPath)))
        {
            _selectedPlacementSaveTargetPath = BuildProjectManagedPlacementOutputPath(_selectedPlacementSourcePath);
        }
    }

    private string BuildMapConverterProjectSourceKey()
    {
        string fullSourcePath = Path.GetFullPath(_mapConvertSourcePath);
        return $"map-convert:{_mapConvertDirection}:{fullSourcePath}";
    }

    private string EnsureMapConverterProjectOutputDirectory(bool forceNew)
    {
        if (string.IsNullOrWhiteSpace(_mapConvertSourcePath))
            return _mapConvertOutputDir;

        string sourceKey = BuildMapConverterProjectSourceKey();
        if (!forceNew
            && !string.IsNullOrWhiteSpace(_mapConvertOutputDir)
            && string.Equals(_mapConvertProjectSourceKey, sourceKey, StringComparison.OrdinalIgnoreCase))
        {
            return _mapConvertOutputDir;
        }

        _mapConvertProjectSourceKey = sourceKey;
        _mapConvertOutputDir = CreateTimestampedProjectOutputDirectory(
            GetProjectOutputRootDirectory(),
            Path.GetFileNameWithoutExtension(_mapConvertSourcePath));
        return _mapConvertOutputDir;
    }

    private string DescribeMapConverterProjectOutputDirectory()
    {
        if (!string.IsNullOrWhiteSpace(_mapConvertOutputDir))
            return _mapConvertOutputDir;

        string projectName = string.IsNullOrWhiteSpace(_mapConvertSourcePath)
            ? "map-conversion"
            : Path.GetFileNameWithoutExtension(_mapConvertSourcePath);
        return Path.Combine(GetProjectOutputRootDirectory(), SanitizeProjectPathSegment(projectName), "<timestamp>");
    }

    private static string BuildMapConverterAlphaSourceCopyPath(string projectOutputDir, string sourceWdtPath)
    {
        string mapName = Path.GetFileNameWithoutExtension(sourceWdtPath);
        return Path.Combine(projectOutputDir, "alpha-source", "World", "Maps", mapName, Path.GetFileName(sourceWdtPath));
    }

    private static string BuildMapConverterLkOutputDirectory(string projectOutputDir, string sourceWdtPath)
    {
        string mapName = Path.GetFileNameWithoutExtension(sourceWdtPath);
        return Path.Combine(projectOutputDir, "lk-split", "World", "Maps", mapName);
    }

    private static string BuildMapConverterAlphaOutputPath(string projectOutputDir, string sourceWdtPath)
    {
        string mapName = Path.GetFileNameWithoutExtension(sourceWdtPath);
        return Path.Combine(projectOutputDir, "alpha-output", "World", "Maps", mapName, $"{mapName}.wdt");
    }

    /// <summary>
    /// Show a native folder picker on an STA thread to avoid deadlocking the GLFW render thread.
    /// </summary>
    private static string? ShowFolderDialogSTA(string description, string? initialDir = null, bool showNewFolderButton = false)
    {
#if WINDOWS
        string? result = null;
        var thread = new Thread(() =>
        {
            using var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = description,
                UseDescriptionForTitle = true,
                ShowNewFolderButton = showNewFolderButton
            };
            if (!string.IsNullOrEmpty(initialDir) && Directory.Exists(initialDir))
                dialog.InitialDirectory = initialDir;
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                result = dialog.SelectedPath;
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        return result;
#else
        return null;
#endif
    }

    /// <summary>
    /// Show a native file-open picker on an STA thread to avoid deadlocking the GLFW render thread.
    /// </summary>
    private static string? ShowFileDialogSTA(string title, string filter, string? initialDir = null)
    {
#if WINDOWS
        string? result = null;
        var thread = new Thread(() =>
        {
            using var dialog = new System.Windows.Forms.OpenFileDialog
            {
                Title = title,
                Filter = filter,
                RestoreDirectory = true
            };
            if (initialDir != null && Directory.Exists(initialDir))
                dialog.InitialDirectory = initialDir;
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                result = dialog.FileName;
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        return result;
#else
        return null;
#endif
    }

    /// <summary>
    /// Show a native save-file picker on an STA thread.
    /// </summary>
    private static string? ShowSaveFileDialogSTA(string title, string filter, string? initialDir = null, string? defaultFileName = null)
    {
#if WINDOWS
        string? result = null;
        var thread = new Thread(() =>
        {
            using var dialog = new System.Windows.Forms.SaveFileDialog
            {
                Title = title,
                Filter = filter,
                RestoreDirectory = true
            };
            if (initialDir != null && Directory.Exists(initialDir))
                dialog.InitialDirectory = initialDir;
            if (defaultFileName != null)
                dialog.FileName = defaultFileName;
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                result = dialog.FileName;
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        return result;
#else
        return null;
#endif
    }

    private void StartVlmExport()
    {
        _vlmExporting = true;
        _vlmExportResult = null;
        _pendingMlFinalizeAfterExport = false;
        lock (_vlmExportLog) { _vlmExportLog.Clear(); }

        var clientPath = _vlmClientPath;
        var mapName = _vlmMapName;
        var outputDir = _vlmOutputDir;
        var limit = _vlmTileLimit <= 0 ? int.MaxValue : _vlmTileLimit;

        ThreadPool.QueueUserWorkItem(_ =>
        {
            try
            {
                var exporter = new VlmDatasetExporter();
                var progress = new Progress<string>(msg =>
                {
                    lock (_vlmExportLog)
                    {
                        _vlmExportLog.Add(msg);
                        // Keep log from growing unbounded
                        if (_vlmExportLog.Count > 2000)
                            _vlmExportLog.RemoveRange(0, _vlmExportLog.Count - 1500);
                    }
                    _vlmExportScrollToBottom = true;
                });

                var result = exporter.ExportMapAsync(clientPath, mapName, outputDir, progress, limit)
                    .GetAwaiter().GetResult();

                _vlmExportResult = result;
                if (_mlFinalizeAfterExport)
                    _pendingMlFinalizeAfterExport = true;
                lock (_vlmExportLog)
                {
                    _vlmExportLog.Add($"=== Export complete: {result.TilesExported} tiles, {result.TilesSkipped} skipped, {result.UniqueTextures} textures ===");
                    if (_mlFinalizeAfterExport)
                        _vlmExportLog.Add("=== Starting manifest + validation automatically in the same ML dataset build flow ===");
                }
                _vlmExportScrollToBottom = true;
            }
            catch (Exception ex)
            {
                lock (_vlmExportLog)
                {
                    _vlmExportLog.Add($"ERROR: {ex.Message}");
                    _vlmExportLog.Add(ex.StackTrace ?? "");
                }
                _vlmExportScrollToBottom = true;
            }
            finally
            {
                _vlmExporting = false;
            }
        });
    }

    private void StartMkHarvest()
    {
        _mkHarvestRunning = true;
        _mkHarvestResult = null;
        lock (_mkHarvestLog) { _mkHarvestLog.Clear(); }
        _mkHarvestViewerValidationQueued = 0;
        _mkHarvestViewerValidationCompleted = 0;
        _mkHarvestViewerValidationFailed = 0;

        string datasetRoot = _mkHarvestDatasetRoot;
        string? manifestOutputPath = string.IsNullOrWhiteSpace(_mkHarvestManifestOutputPath) ? null : _mkHarvestManifestOutputPath;
        string? viewerValidationOutputDir = string.IsNullOrWhiteSpace(_mkHarvestViewerValidationOutputDir) ? null : _mkHarvestViewerValidationOutputDir;
        bool generateViewerValidationMinimaps = _mkHarvestGenerateViewerValidationMinimaps;
        bool forceViewerValidationRegeneration = _mkHarvestForceViewerValidationRegeneration;
        int viewerValidationResolution = _mkHarvestViewerValidationResolution;

        ThreadPool.QueueUserWorkItem(_ =>
        {
            try
            {
                var harvester = new MkDatasetHarvester();
                var options = new MkDatasetHarvestOptions(
                    DatasetRoot: datasetRoot,
                    ManifestOutputPath: manifestOutputPath,
                    GenerateReferenceMinimaps: false,
                    ForceRegenerateReferenceMinimaps: false,
                    ApplyShadows: true,
                    ShadowIntensity: 0.5f,
                    InvertAlpha: true,
                    ReferenceMinimapDirectory: null);

                var progress = new Progress<string>(msg =>
                {
                    lock (_mkHarvestLog)
                    {
                        _mkHarvestLog.Add(msg);
                        if (_mkHarvestLog.Count > 2000)
                            _mkHarvestLog.RemoveRange(0, _mkHarvestLog.Count - 1500);
                    }

                    _mkHarvestScrollToBottom = true;
                });

                MkDatasetHarvestResult result = harvester.HarvestAsync(options, progress)
                    .GetAwaiter().GetResult();
                _mkHarvestResult = result;

                if (generateViewerValidationMinimaps)
                {
                    MkHarvestViewerValidationCapturePlan? validationPlan = BuildMkHarvestViewerValidationCapturePlan(
                        datasetRoot,
                        viewerValidationOutputDir,
                        forceViewerValidationRegeneration,
                        viewerValidationResolution,
                        out string? validationMessage,
                        DefaultRequiredSettledFrames,
                        DefaultMaxFramesBeforeCapture,
                        DefaultBatchSettledFrames);

                    if (!string.IsNullOrWhiteSpace(validationMessage))
                        AppendMkHarvestLogLine(validationMessage);

                    if (validationPlan != null)
                    {
                        if (validationPlan.Tiles.Count > 0)
                        {
                            _pendingMkHarvestViewerValidationCapturePlan = validationPlan;
                            _mkHarvestViewerValidationQueued = validationPlan.Tiles.Count;
                            AppendMkHarvestLogLine(
                                $"Queued {validationPlan.Tiles.Count} WoWViewer validation capture(s) at {validationPlan.RequestedResolution}px into {validationPlan.OutputDirectory} with matching noliquids captures under {validationPlan.NoLiquidsOutputDirectory}, noobjects captures under {validationPlan.NoObjectsOutputDirectory}, and objectsonly captures under {validationPlan.ObjectsOnlyOutputDirectory}.");
                        }
                        else
                        {
                            StitchMkHarvestViewerValidationOutputs(
                                validationPlan.MapName,
                                validationPlan.OutputDirectory,
                                validationPlan.NoLiquidsOutputDirectory,
                                validationPlan.NoObjectsOutputDirectory,
                                validationPlan.ObjectsOnlyOutputDirectory,
                                validationPlan.RequestedResolution);
                        }
                    }
                }

                lock (_mkHarvestLog)
                {
                    _mkHarvestLog.Add($"=== Harvest complete: {result.TilesProcessed} tiles, {result.TilesWithAlphaMasks} with alpha masks, no baked reference minimaps generated ===");
                    _mkHarvestLog.Add($"Manifest: {result.ManifestPath}");
                }

                _mkHarvestScrollToBottom = true;
            }
            catch (Exception ex)
            {
                lock (_mkHarvestLog)
                {
                    _mkHarvestLog.Add($"ERROR: {ex.Message}");
                    _mkHarvestLog.Add(ex.StackTrace ?? string.Empty);
                }

                _mkHarvestScrollToBottom = true;
            }
            finally
            {
                _mkHarvestRunning = false;
            }
        });
    }

    private void StartTerrainTextureTransfer()
    {
        _terrainTransferRunning = true;
        _terrainTransferError = null;
        _terrainTransferReport = null;
        lock (_terrainTransferLog)
        {
            _terrainTransferLog.Clear();
        }

        string sourceDir = _terrainTransferSourceDir;
        string targetDir = _terrainTransferTargetDir;
        string outputDir = _terrainTransferOutputDir;
        bool applyMode = _terrainTransferApplyMode;
        bool useGlobalDelta = _terrainTransferUseGlobalDelta;
        int srcX = _terrainTransferSourceTileX;
        int srcY = _terrainTransferSourceTileY;
        int dstX = _terrainTransferTargetTileX;
        int dstY = _terrainTransferTargetTileY;
        int deltaX = _terrainTransferDeltaX;
        int deltaY = _terrainTransferDeltaY;
        int tileLimit = _terrainTransferTileLimit;
        int chunkOffsetX = _terrainTransferChunkOffsetX;
        int chunkOffsetY = _terrainTransferChunkOffsetY;
        bool copyMtex = _terrainTransferCopyMtex;
        bool copyMcly = _terrainTransferCopyMcly;
        bool copyMcal = _terrainTransferCopyMcal;
        bool copyMcsh = _terrainTransferCopyMcsh;
        bool copyHoles = _terrainTransferCopyHoles;
        string manifestPath = _terrainTransferManifestPath;

        ThreadPool.QueueUserWorkItem(_ =>
        {
            try
            {
                var pairs = new List<WoWViewer.Transfer.TerrainTilePair>();
                int? globalDeltaX = null;
                int? globalDeltaY = null;

                if (useGlobalDelta)
                {
                    globalDeltaX = deltaX;
                    globalDeltaY = deltaY;
                }
                else
                {
                    pairs.Add(new WoWViewer.Transfer.TerrainTilePair(srcX, srcY, dstX, dstY));
                }

                var options = new WoWViewer.Transfer.TerrainTextureTransferOptions(
                    SourceDirectory: sourceDir,
                    TargetDirectory: targetDir,
                    OutputDirectory: outputDir,
                    Mode: applyMode ? "apply" : "dry-run",
                    Pairs: pairs,
                    TileLimit: tileLimit > 0 ? tileLimit : null,
                    GlobalDeltaX: globalDeltaX,
                    GlobalDeltaY: globalDeltaY,
                    ChunkOffsetX: chunkOffsetX,
                    ChunkOffsetY: chunkOffsetY,
                    CopyMtex: copyMtex,
                    CopyMcly: copyMcly,
                    CopyMcal: copyMcal,
                    CopyMcsh: copyMcsh,
                    CopyHoles: copyHoles,
                    ManifestPath: string.IsNullOrWhiteSpace(manifestPath) ? null : manifestPath);

                WoWViewer.Transfer.TerrainTextureTransferExecutionReport report =
                    WoWViewer.Transfer.TerrainTextureTransferService.Execute(options);

                _terrainTransferReport = report;
                lock (_terrainTransferLog)
                {
                    _terrainTransferLog.Add($"Source map: {report.SourceMapName}");
                    _terrainTransferLog.Add($"Target map: {report.TargetMapName}");
                    _terrainTransferLog.Add($"Tiles planned: {report.TilesPlanned}");
                    _terrainTransferLog.Add($"Tiles processed: {report.TilesProcessed}");
                    _terrainTransferLog.Add($"Tiles written: {report.TilesWritten}");
                    _terrainTransferLog.Add($"Manual review: {report.TilesNeedingManualReview}");
                    _terrainTransferLog.Add($"Chunk pairs: {report.ChunkPairsApplied}");
                    _terrainTransferLog.Add($"Summary manifest: {report.SummaryManifestPath}");

                    foreach (var tile in report.Tiles.Where(tile => tile.NeedsManualReview || tile.Warnings.Count > 0).Take(20))
                    {
                        _terrainTransferLog.Add($"Pair {tile.SourceTileName} -> {tile.TargetTileName}: touched={tile.TargetChunksTouched}, missingSource={tile.MissingSourceChunkCount}, outOfRange={tile.OutOfRangeChunkRemapCount}");
                        if (tile.Warnings.Count > 0)
                            _terrainTransferLog.Add($"  warning: {tile.Warnings[0]}");
                    }
                }
            }
            catch (Exception ex)
            {
                _terrainTransferError = ex.Message;
                lock (_terrainTransferLog)
                {
                    _terrainTransferLog.Add($"ERROR: {ex.Message}");
                    _terrainTransferLog.Add(ex.StackTrace ?? "");
                }
            }
            finally
            {
                _terrainTransferRunning = false;
                _terrainTransferScrollToBottom = true;
            }
        });
    }

    private void DrawWorldObjectsContentCore()
    {
        if (_worldScene == null) return;

        DrawSelectedPlacementEditControls();
        DrawVisualInvestigationToolbox(showWorldObjectRangeControls: true);
        ImGui.Separator();
        DrawTerrainChunkInvestigationPanel(defaultOpen: _visualInvestigationMode == VisualInvestigationMode.Adt);
        ImGui.Separator();

        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

        ImGui.Separator();
        ImGui.Text("SQL World Population");
        ImGui.InputTextWithHint("##sqlroot", "Path to alpha-core root (example: external/alpha-core)", ref _sqlAlphaCoreRoot, 1024);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("WoWViewer reads NPC/GameObject spawns from alpha-core SQL dumps (etc/databases/world + dbc).");

        DrawToolbarPopupButton("SQL Actions", string.Empty, "##SqlWorldActionsPopup", () =>
        {
            if (ImGui.Button("Use Submodule Path"))
            {
                string candidate = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "external", "alpha-core"));
                _sqlAlphaCoreRoot = candidate;
                ImGui.CloseCurrentPopup();
            }

            bool canLoadSqlFromPopup = _currentMapId >= 0 && !string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot);
            if (!canLoadSqlFromPopup)
                ImGui.BeginDisabled();
            if (ImGui.Button("Load SQL Spawns (Current Map)"))
            {
                LoadSqlSpawnsForCurrentMap();
                ImGui.CloseCurrentPopup();
            }
            if (!canLoadSqlFromPopup)
                ImGui.EndDisabled();

            if (ImGui.Button("Clear SQL Spawns"))
            {
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                _sqlSpawnStatus = "Cleared SQL spawns.";
                ImGui.CloseCurrentPopup();
            }
        });
        bool sqlSettingsChanged = false;
        sqlSettingsChanged |= ImGui.Checkbox("NPC Spawns", ref _sqlIncludeCreatures);
        ImGui.SameLine();
        sqlSettingsChanged |= ImGui.Checkbox("GameObject Spawns", ref _sqlIncludeGameObjects);
        sqlSettingsChanged |= ImGui.Checkbox("AOI Tile Filter", ref _sqlUseAoiFilter);
        if (_sqlUseAoiFilter)
            sqlSettingsChanged |= ImGui.SliderInt("AOI Tile Radius", ref _sqlAoiTileRadius, 1, 16);
        sqlSettingsChanged |= ImGui.Checkbox("Stream With Camera", ref _sqlStreamWithCamera);
        sqlSettingsChanged |= ImGui.SliderInt("Max SQL Spawns", ref _sqlMaxSpawns, 100, 20000);
        sqlSettingsChanged |= ImGui.SliderFloat("GO MDX Scale", ref _sqlGameObjectMdxScaleMultiplier, 0.10f, 3.00f, "%.2fx");
        _worldScene.SqlGameObjectMdxScaleMultiplier = _sqlGameObjectMdxScaleMultiplier;
        if (sqlSettingsChanged && _sqlMapSpawnsCache != null)
        {
            _sqlForceStreamRefresh = true;
            if (!_sqlStreamWithCamera || !_sqlUseAoiFilter)
                ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: true);
        }

        ImGui.TextDisabled($"Status: {_sqlSpawnStatus}");
        ImGui.TextDisabled($"Injected: {_worldScene.ExternalSpawnInstanceCount} total ({_worldScene.ExternalSpawnMdxCount} MDX, {_worldScene.ExternalSpawnWmoCount} WMO)");

        bool showPm4Overlay = _worldScene.ShowPm4Overlay;
        if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
            _worldScene.ShowPm4Overlay = showPm4Overlay;
        if (ImGui.IsItemHovered() && _worldScene.ShowPm4Overlay)
            ImGui.SetTooltip(_worldScene.Pm4Status);

        DrawToolbarPopupButton("PM4 Actions", string.Empty, "##Pm4OverlayActionsPopup", () =>
        {
            if (ImGui.Button("PM4 Workbench"))
            {
                OpenPm4Workbench(_worldScene.HasSelectedPm4Object ? Pm4WorkbenchTab.Selection : Pm4WorkbenchTab.Overlay);
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button("Reload PM4"))
            {
                _worldScene.ReloadPm4Overlay();
                ImGui.CloseCurrentPopup();
            }
        });

        if (_worldScene.IsPm4Loading)
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), $"PM4 loading... {_worldScene.Pm4Status}");
        else if (_worldScene.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4VisibleObjectCount}/{_worldScene.Pm4ObjectCount} visible objects, {_worldScene.Pm4VisibleLineCount}/{_worldScene.Pm4LineCount} lines, {_worldScene.Pm4VisibleTriangleCount}/{_worldScene.Pm4TriangleCount} tris");
        else
            ImGui.TextDisabled("PM4 stays lightweight here. Use the inspector workbench for overlay tuning, object matches, and correlation.");
        if (_worldScene.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4 status: {_worldScene.Pm4Status}");
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
            DrawToolbarPopupButton("WL Actions", "load", "##WlActionsPopup", () =>
            {
                if (ImGui.Button("Load WL Liquids"))
                {
                    _worldScene.ShowWlLiquids = true;
                    ImGui.CloseCurrentPopup();
                }
            });
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
            DrawToolbarPopupButton("AreaTrigger Actions", "load", "##AreaTriggerActionsPopup", () =>
            {
                if (ImGui.Button("Load AreaTriggers"))
                {
                    _worldScene.ShowAreaTriggers = true;
                    ImGui.CloseCurrentPopup();
                }
            });
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

    private void DrawMinimap_OLD()
    {
        // Gather tile data from whichever terrain manager is active
        List<(int tx, int ty)>? existingTiles = null;
        Func<int, int, bool>? isTileLoaded = null;
        int loadedTileCount = 0;
        string? mapName = null;

        if (_terrainManager != null)
        {
            var adapter = _terrainManager.Adapter;
            existingTiles = adapter.ExistingTiles.Select(idx => (idx / 64, idx % 64)).ToList();
            isTileLoaded = _terrainManager.IsTileLoaded;
            loadedTileCount = _terrainManager.LoadedTileCount;
            mapName = _terrainManager.MapName;
        }
        else if (_vlmTerrainManager != null)
        {
            existingTiles = _vlmTerrainManager.Loader.TileCoords.ToList();
            isTileLoaded = _vlmTerrainManager.IsTileLoaded;
            loadedTileCount = _vlmTerrainManager.LoadedTileCount;
            mapName = _vlmTerrainManager.MapName;
        }
        else return;

        var io = ImGui.GetIO();
        float mapSize = 200f;
        float padding = 10f;

        ImGui.SetNextWindowPos(new Vector2(padding, io.DisplaySize.Y - mapSize - 34), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowSize(new Vector2(mapSize + 16, mapSize + 36), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Minimap", ImGuiWindowFlags.NoScrollbar))
        {
            var drawList = ImGui.GetWindowDrawList();
            var cursorPos = ImGui.GetCursorScreenPos();
            var contentSize = ImGui.GetContentRegionAvail();
            mapSize = MathF.Min(contentSize.X, contentSize.Y);
            if (mapSize < 50f) mapSize = 50f;

            // Scroll-wheel zoom (when minimap window is hovered)
            if (ImGui.IsWindowHovered())
            {
                float wheel = io.MouseWheel;
                if (wheel != 0)
                {
                    _minimapZoom = Math.Clamp(_minimapZoom - wheel * 0.5f, 1f, 32f);
                }
            }

            // Camera tile position (center of view)
            float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize;
            float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize;

            // View window: _minimapZoom tiles in each direction from camera
            float viewRadius = _minimapZoom;
            float viewMinTx = camTileX - viewRadius;
            float viewMaxTx = camTileX + viewRadius;
            float viewMinTy = camTileY - viewRadius;
            float viewMaxTy = camTileY + viewRadius;
            float viewSpan = viewRadius * 2f;
            float cellSize = mapSize / viewSpan;

            // Background
            drawList.AddRectFilled(cursorPos, cursorPos + new Vector2(mapSize, mapSize), 0xFF1A1A1A);

            // Clip to minimap area
            drawList.PushClipRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize), true);

            // Draw existing tiles
            // Screen: X = tileY (east-west), Y = tileX (north-south)
            foreach (var (tx, ty) in existingTiles)
            {
                // Skip tiles outside view
                if (tx + 1 < viewMinTx || tx > viewMaxTx || ty + 1 < viewMinTy || ty > viewMaxTy)
                    continue;

                float x = cursorPos.X + (ty - viewMinTy) * cellSize;
                float y = cursorPos.Y + (tx - viewMinTx) * cellSize;

                // Try to render BLP minimap tile texture
                bool drewTexture = false;
                if (_minimapRenderer != null && !string.IsNullOrEmpty(mapName))
                {
                    // Ghidra-verified: files are map{x}_{y}.blp. tx=row(y), ty=col(x).
                    // So pass (ty, tx) = (col, row) = (x, y).
                    uint tileTex = _minimapRenderer.GetTileTexture(mapName, ty, tx);
                    if (tileTex != 0)
                    {
                        var texId = (IntPtr)tileTex;
                        var p1 = new Vector2(x, y);
                        var p2 = new Vector2(x + cellSize, y);
                        var p3 = new Vector2(x + cellSize, y + cellSize);
                        var p4 = new Vector2(x, y + cellSize);
                        drawList.AddImageQuad(texId, p1, p2, p3, p4,
                            new Vector2(0, 0), new Vector2(1, 0),
                            new Vector2(1, 1), new Vector2(0, 1),
                            0xFFFFFFFF);
                        drewTexture = true;
                    }
                }

                // Fallback: colored rectangle
                if (!drewTexture)
                {
                    bool loaded = isTileLoaded(tx, ty);
                    uint color = loaded ? 0xFF00AA00 : 0xFF004400;
                    drawList.AddRectFilled(new Vector2(x, y), new Vector2(x + cellSize, y + cellSize), color);
                }
            }

            // Camera position (always centered)
            float camScreenX = cursorPos.X + mapSize * 0.5f;
            float camScreenY = cursorPos.Y + mapSize * 0.5f;

            // Camera direction indicator
            float yawRad = _camera.Yaw * MathF.PI / 180f;
            float dirLen = mapSize * 0.08f;
            float dotRadius = mapSize * 0.02f;
            float dirX = camScreenX - MathF.Sin(yawRad) * dirLen;
            float dirY = camScreenY - MathF.Cos(yawRad) * dirLen;
            drawList.AddLine(new Vector2(camScreenX, camScreenY), new Vector2(dirX, dirY), 0xFFFFFF00, MathF.Max(2f, mapSize * 0.012f));
            drawList.AddCircleFilled(new Vector2(camScreenX, camScreenY), MathF.Max(3f, dotRadius), 0xFFFFFFFF);

            // POI markers (WorldScene only)
            if (_worldScene?.PoiLoader != null && _worldScene.ShowPoi)
            {
                foreach (var poi in _worldScene.PoiLoader.Entries)
                {
                    float poiTileX = (WoWConstants.MapOrigin - poi.Position.X) / WoWConstants.ChunkSize;
                    float poiTileY = (WoWConstants.MapOrigin - poi.Position.Y) / WoWConstants.ChunkSize;
                    float px = cursorPos.X + (poiTileY - viewMinTy) * cellSize;
                    float py = cursorPos.Y + (poiTileX - viewMinTx) * cellSize;
                    if (px >= cursorPos.X && px <= cursorPos.X + mapSize && py >= cursorPos.Y && py <= cursorPos.Y + mapSize)
                        drawList.AddCircleFilled(new Vector2(px, py), MathF.Max(2.5f, cellSize * 0.15f), 0xFFFF00FF);
                }
            }

            // Taxi path lines on minimap (cyan lines, yellow node dots) — filtered by selection
            if (_worldScene?.TaxiLoader != null && _worldScene.ShowTaxi)
            {
                // Draw visible route lines
                foreach (var route in _worldScene.TaxiLoader.Routes)
                {
                    if (!_worldScene.IsTaxiRouteVisible(route)) continue;
                    for (int i = 0; i < route.Waypoints.Count - 1; i++)
                    {
                        var a = route.Waypoints[i];
                        var b = route.Waypoints[i + 1];
                        float ax = cursorPos.X + ((WoWConstants.MapOrigin - a.Y) / WoWConstants.ChunkSize - viewMinTy) * cellSize;
                        float ay = cursorPos.Y + ((WoWConstants.MapOrigin - a.X) / WoWConstants.ChunkSize - viewMinTx) * cellSize;
                        float bx = cursorPos.X + ((WoWConstants.MapOrigin - b.Y) / WoWConstants.ChunkSize - viewMinTy) * cellSize;
                        float by = cursorPos.Y + ((WoWConstants.MapOrigin - b.X) / WoWConstants.ChunkSize - viewMinTx) * cellSize;
                        drawList.AddLine(new Vector2(ax, ay), new Vector2(bx, by), 0xFFFFFF00, 1.5f);
                    }
                }
                // Draw visible taxi nodes
                foreach (var node in _worldScene.TaxiLoader.Nodes)
                {
                    if (!_worldScene.IsTaxiNodeVisible(node)) continue;
                    float nx = cursorPos.X + ((WoWConstants.MapOrigin - node.Position.Y) / WoWConstants.ChunkSize - viewMinTy) * cellSize;
                    float ny = cursorPos.Y + ((WoWConstants.MapOrigin - node.Position.X) / WoWConstants.ChunkSize - viewMinTx) * cellSize;
                    if (nx >= cursorPos.X && nx <= cursorPos.X + mapSize && ny >= cursorPos.Y && ny <= cursorPos.Y + mapSize)
                        drawList.AddCircleFilled(new Vector2(nx, ny), MathF.Max(3f, cellSize * 0.2f), 0xFF00FFFF);
                }
            }

            drawList.PopClipRect();

            // Border
            drawList.AddRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize), 0xFF666666);

            // Double-click to teleport
            if (ImGui.IsWindowHovered() && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
            {
                var mousePos = ImGui.GetMousePos();
                float clickTileY = (mousePos.X - cursorPos.X) / cellSize + viewMinTy;
                float clickTileX = (mousePos.Y - cursorPos.Y) / cellSize + viewMinTx;
                if (clickTileX >= 0 && clickTileX < 64 && clickTileY >= 0 && clickTileY < 64)
                {
                    float worldX = WoWConstants.MapOrigin - clickTileX * WoWConstants.ChunkSize;
                    float worldY = WoWConstants.MapOrigin - clickTileY * WoWConstants.ChunkSize;
                    _camera.Position = new System.Numerics.Vector3(worldX, worldY, _camera.Position.Z);
                }
            }

            // Tile coordinate label + zoom info
            ImGui.SetCursorPosY(ImGui.GetCursorPosY() + mapSize + 2);
            int ctX = (int)MathF.Floor(camTileX);
            int ctY = (int)MathF.Floor(camTileY);
            ImGui.Text($"Tile: ({ctX},{ctY})  Loaded: {loadedTileCount}");
        }
        ImGui.End();
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
            if (mpqDs != null)
            {
                _dbcProvider = new MpqDBCProvider(mpqDs.ArchiveReader);
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
            _vlmOutputDir = GenerateVlmOutputPath(_vlmClientPath, _vlmMapName);
    }

    private void PrepareMkHarvestDialogInputs()
    {
        string? datasetRoot = null;
        if (!string.IsNullOrWhiteSpace(_mkHarvestDatasetRoot))
            datasetRoot = _mkHarvestDatasetRoot;
        else if (_vlmExportResult != null && !string.IsNullOrWhiteSpace(_vlmExportResult.OutputDirectory))
            datasetRoot = _vlmExportResult.OutputDirectory;
        else if (!string.IsNullOrWhiteSpace(_vlmOutputDir))
            datasetRoot = _vlmOutputDir;
        else if (_vlmTerrainManager != null)
            datasetRoot = _vlmTerrainManager.Loader.ProjectRoot;

        if (!string.IsNullOrWhiteSpace(datasetRoot))
            SyncMkHarvestDerivedPaths(_mkHarvestDatasetRoot, datasetRoot);
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
            EnsureMapConverterProjectOutputDirectory(forceNew: false);
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

        if (!_worldScene.SelectSceneObject(info.SceneObjectType, info.SceneObjectIndex))
            return false;

        ClearSelectedWlLiquidBody(clearListIsolation: true);
        _worldScene.ClearTaxiSelection();
        _worldScene.ClearPm4ObjectSelection();
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
        RefreshDiscoveredMaps();
        RefreshFileList();
        if (_worldScene != null && (_worldScene.ShowPm4Overlay || _worldScene.Pm4LoadAttempted))
            _worldScene.ReloadPm4Overlay();

        string? overlayBuildHint = TryDetectLooseOverlayBuildHint(normalizedRoot);
        if (!string.IsNullOrWhiteSpace(overlayBuildHint) && !string.Equals(_dbcBuild, overlayBuildHint, StringComparison.OrdinalIgnoreCase))
        {
            ViewerLog.Important(ViewerLog.Category.MpqData,
                $"Loose overlay at '{normalizedRoot}' hints build {overlayBuildHint} from PM4 files, but the active base client build is {_dbcBuild ?? "unknown"}. If PM4-linked objects do not match, load a {overlayBuildHint} base client instead.");
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

        foreach (string candidate in bundledCandidates)
        {
            if (File.Exists(candidate))
            {
                ViewerLog.Info(ViewerLog.Category.MpqData, $"Using bundled listfile: {candidate}");
                return candidate;
            }
        }

        string? downloadedPath = ListfileDownloader.GetListfilePath();
        if (!string.IsNullOrWhiteSpace(downloadedPath) && File.Exists(downloadedPath))
        {
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Using cached/downloaded listfile: {downloadedPath}");
            return downloadedPath;
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
            if (Md5TranslateResolver.TryLoad(
                new[] { mpqDataSource.GamePath },
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

            // Write to cache folder for parsers that expect file paths
            Directory.CreateDirectory(CacheDir);
            var cachePath = Path.Combine(CacheDir, _loadedFileName!);
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
                          ? "  Draw path is the wow-viewer runtime renderer in WoWViewer.\n  This is now the default route for successful runtime-backed M2 loads; set PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=0 to force the legacy compatibility draw backend.\n  Skeletal sequence playback now advances through wow-viewer pose evaluation inside the runtime renderer. Current shading still uses primary-stage runtime textures and simple lighting, not full native material parity.\n"
                          : "  Draw path still uses the legacy MDX backend for textured compatibility while the wow-viewer runtime supplies geometry/state.\n") +
                      "  Full material/effect parity is still pending.\n";

        _statusMessage = $"Loaded M2: {_loadedFileName} ({sectionCount} sections, {vertexCount:N0} verts, {triangleCount:N0} tris)";
    }

    private void LoadStandaloneCameraPathModel(M2ModelDocument cameraModel, M2CameraPathVisualization visualization, string virtualPath)
    {
        ArgumentNullException.ThrowIfNull(cameraModel);
        ArgumentNullException.ThrowIfNull(visualization);

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

        ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);
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
            RefreshTerrainWeakSignalRestoreHooks();
            RefreshTerrainWeakSignalRestoreForLoadedTiles();
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

                if (curMapDef?.HasDbcEntry == true)
                    _worldScene.LoadLighting(_dbcProvider, _dbdDir, _dbcBuild, curMapDef.Id);
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

        if (_standaloneCharacterHairVariationOverride >= 0 || _standaloneCharacterFacialHairVariationOverride >= 0)
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
                $"Zarr store '{summary.MapName}' discovered: {summary.Arrays.Count} arrays " +
                $"(liquid_basic_type_257: {(summary.HasLiquidBasicType ? "yes" : "MISSING — rebuild tiles")}). " +
                "Per-tile loading is the spec 041 T-10 implementation slice.";
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
        ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);

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
            RefreshTerrainWeakSignalRestoreHooks();
            RefreshTerrainWeakSignalRestoreForLoadedTiles();
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
        _worldScene.ClearPm4ObjectSelection();
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
        _worldScene.ClearPm4ObjectSelection();
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
        _worldScene.ClearPm4ObjectSelection();

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
        string type = _worldScene.SelectedObjectType == Terrain.ObjectType.Wmo ? "WMO" : "MDX";
        int idx = _worldScene.SelectedObjectIndex;
        float wowX = WoWConstants.MapOrigin - inst.PlacementPosition.Y;
        float wowY = WoWConstants.MapOrigin - inst.PlacementPosition.X;
        float wowZ = inst.PlacementPosition.Z;

        _selectedObjectType = type;
        _selectedObjectIndex = idx;

        if (_useTabUi && _worldScene.SelectedObjectType is Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo)
            OpenWorkbenchTab(ModelBottomTab.Info);

        _selectedObjectInfo = $"{type} [{idx}] {inst.ModelName}\n" +
            $"Path: {inst.ModelPath}\n" +
            $"UniqueId: {inst.UniqueId}\n" +
            $"Local: ({inst.PlacementPosition.X:F1}, {inst.PlacementPosition.Y:F1}, {inst.PlacementPosition.Z:F1})\n" +
            $"WoW:   ({wowX:F1}, {wowY:F1}, {wowZ:F1})\n" +
            $"Rotation: ({inst.PlacementRotation.X:F1}, {inst.PlacementRotation.Y:F1}, {inst.PlacementRotation.Z:F1})\n" +
            $"Scale: {inst.PlacementScale:F3}\n" +
            $"BB: ({inst.BoundsMin.X:F1},{inst.BoundsMin.Y:F1},{inst.BoundsMin.Z:F1}) - ({inst.BoundsMax.X:F1},{inst.BoundsMax.Y:F1},{inst.BoundsMax.Z:F1})";
    }

    private void DrawSelectedPlacementEditControls()
    {
        SyncSelectedPlacementEditState();

        ImGui.Separator();
        ImGui.Text("Selected Placement Move");
        ImGui.TextDisabled("Translation-only save for existing ADT MDDF/MODF placements, grouped by source ADT when multiple moves are staged.");

        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
        {
            DrawPlacementSaveQueueActions(includeCurrentSourceSave: false);
            ImGui.TextDisabled(_selectedPlacementSaveStatus);
            return;
        }

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        bool editable = selected.HasTileCoordinate && selected.PlacementEntryIndex >= 0
            && _worldScene.SelectedObjectType is Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo;

        if (!editable)
        {
            DrawPlacementSaveQueueActions(includeCurrentSourceSave: false);
            ImGui.TextDisabled(_selectedPlacementSaveStatus);
            return;
        }

        ImGui.TextDisabled($"Tile ({selected.TileX}, {selected.TileY})  Entry {selected.PlacementEntryIndex}  UniqueId {selected.UniqueId}");

        Vector3 editedPosition = _selectedPlacementEditedPosition;
        if (ImGui.InputFloat3("Placement Position", ref editedPosition, "%.3f"))
        {
            if (!EnsureSelectedPlacementSourcePath(out _, out string sourceError))
            {
                _selectedPlacementSaveStatus = sourceError;
            }
            else if (_worldScene.TryUpdateSelectedPlacementPosition(editedPosition, out string error))
            {
                _selectedPlacementEditedPosition = editedPosition;
                _selectedPlacementDirty = !PositionsNearlyEqual(_selectedPlacementEditedPosition, _selectedPlacementOriginalPosition);
                if (_selectedPlacementDirty)
                {
                    UpsertSelectedPlacementEdit();
                    _selectedPlacementSaveStatus = BuildSelectedPlacementSaveStatus();
                }
                else
                {
                    RemoveSelectedPlacementEdit();
                    _selectedPlacementSaveStatus = HasPendingPlacementEdits()
                        ? "Preview matches the source tile placement. Other staged placement moves remain pending."
                        : "Preview matches the source tile placement position.";
                }

                RefreshSelectedWorldObjectInfo();
            }
            else
            {
                _selectedPlacementSaveStatus = error;
            }
        }

        if (_selectedPlacementDirty && ImGui.Button("Reset Preview"))
        {
            if (_worldScene.TryUpdateSelectedPlacementPosition(_selectedPlacementOriginalPosition, out string error))
            {
                _selectedPlacementEditedPosition = _selectedPlacementOriginalPosition;
                _selectedPlacementDirty = false;
                RemoveSelectedPlacementEdit();
                _selectedPlacementSaveStatus = HasPendingPlacementEdits()
                    ? "Preview reset to the source tile placement position. Other staged placement moves remain pending."
                    : "Preview reset to the source tile placement position.";
                RefreshSelectedWorldObjectInfo();
            }
            else
            {
                _selectedPlacementSaveStatus = error;
            }
        }

        string targetLabel = string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
            ? "No save path selected."
            : _selectedPlacementSaveTargetPath!;
        ImGui.TextWrapped($"Save target: {targetLabel}");
        ImGui.TextDisabled("Writes an ADT copy to disk. The loaded source files are not overwritten in place.");

        if (ImGui.Button("Choose Save Path"))
            ChooseSelectedPlacementSavePath();

        DrawPlacementSaveQueueActions(includeCurrentSourceSave: true);

        ImGui.TextDisabled(_selectedPlacementSaveStatus);
    }

    private void SyncSelectedPlacementEditState()
    {
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
        {
            ResetSelectedPlacementEditState("Select a tile-backed world object to stage a translation-only save.");
            return;
        }

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        if (!selected.HasTileCoordinate || selected.PlacementEntryIndex < 0)
        {
            ResetSelectedPlacementEditState("The selected object is not backed by a writable ADT tile placement.");
            return;
        }

        Terrain.ObjectType selectedType = _worldScene.SelectedObjectType;
        if (selectedType is not (Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo))
        {
            ResetSelectedPlacementEditState("Only MDDF and MODF tile placements are supported by the current save seam.");
            return;
        }

        bool sameSelection = _selectedPlacementEditType == selectedType
            && _selectedPlacementEditUniqueId == selected.UniqueId
            && _selectedPlacementEditTileX == selected.TileX
            && _selectedPlacementEditTileY == selected.TileY
            && _selectedPlacementEditEntryIndex == selected.PlacementEntryIndex;

        if (sameSelection)
            return;

        _selectedPlacementEditType = selectedType;
        _selectedPlacementEditUniqueId = selected.UniqueId;
        _selectedPlacementEditTileX = selected.TileX;
        _selectedPlacementEditTileY = selected.TileY;
        _selectedPlacementEditEntryIndex = selected.PlacementEntryIndex;
        PlacementEditKey key = CreatePlacementEditKey(selectedType, selected);
        if (_stagedPlacementEdits.TryGetValue(key, out StagedPlacementEdit? stagedEdit))
        {
            _selectedPlacementOriginalPosition = stagedEdit.OriginalPosition;
            _selectedPlacementEditedPosition = stagedEdit.EditedPosition;
            _selectedPlacementDirty = !PositionsNearlyEqual(stagedEdit.EditedPosition, stagedEdit.OriginalPosition);
            _selectedPlacementSourcePath = stagedEdit.SourcePath;
        }
        else
        {
            _selectedPlacementOriginalPosition = selected.PlacementPosition;
            _selectedPlacementEditedPosition = selected.PlacementPosition;
            _selectedPlacementDirty = false;
            _selectedPlacementSourcePath = null;
        }

        _selectedPlacementSaveTargetPath = null;
        if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath)
            && _placementSaveTargetsBySourcePath.TryGetValue(_selectedPlacementSourcePath, out string? stagedTarget)
            && !string.IsNullOrWhiteSpace(stagedTarget))
        {
            _selectedPlacementSaveTargetPath = stagedTarget;
        }
        else if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath))
        {
            _selectedPlacementSaveTargetPath = BuildProjectManagedPlacementOutputPath(_selectedPlacementSourcePath);
        }

        _selectedPlacementSaveStatus = _selectedPlacementDirty
            ? BuildSelectedPlacementSaveStatus()
            : string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                ? "Adjust the selected placement to stage a dirty ADT source. A timestamped project output folder will be created for the save target."
                : "Ready to stage placement moves for this ADT source.";
    }

    private void ResetSelectedPlacementEditState(string status)
    {
        _selectedPlacementEditType = Terrain.ObjectType.None;
        _selectedPlacementEditUniqueId = -1;
        _selectedPlacementEditTileX = -1;
        _selectedPlacementEditTileY = -1;
        _selectedPlacementEditEntryIndex = -1;
        _selectedPlacementOriginalPosition = Vector3.Zero;
        _selectedPlacementEditedPosition = Vector3.Zero;
        _selectedPlacementDirty = false;
        _selectedPlacementSourcePath = null;
        _selectedPlacementSaveTargetPath = null;
        _selectedPlacementSaveStatus = HasPendingPlacementEdits()
            ? $"{status} {GetPendingPlacementEditCount()} staged move(s) across {GetPendingPlacementSourceCount()} ADT source(s) remain pending."
            : status;
    }

    private void ChooseSelectedPlacementSavePath()
    {
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return;

        if (!EnsureSelectedPlacementSourcePath(out string sourcePath, out string error))
        {
            _selectedPlacementSaveStatus = error;
            return;
        }

        string initialDir = Environment.CurrentDirectory;
        string defaultFileName = $"placement_{DateTime.Now:yyyyMMdd_HHmmss}.adt";

        defaultFileName = Path.GetFileName(sourcePath);

        string projectManagedTargetPath = BuildProjectManagedPlacementOutputPath(sourcePath);
        if (string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath) || IsProjectManagedOutputPath(_selectedPlacementSaveTargetPath))
            _selectedPlacementSaveTargetPath = projectManagedTargetPath;

        if (!string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath))
        {
            string? existingDir = Path.GetDirectoryName(_selectedPlacementSaveTargetPath);
            if (!string.IsNullOrWhiteSpace(existingDir) && Directory.Exists(existingDir))
                initialDir = existingDir;
            defaultFileName = Path.GetFileName(_selectedPlacementSaveTargetPath);
        }
        else if (_worldScene.TryGetSelectedPlacementWritablePath(out string? writablePath) && !string.IsNullOrWhiteSpace(writablePath))
        {
            string? writableDir = Path.GetDirectoryName(writablePath);
            if (!string.IsNullOrWhiteSpace(writableDir) && Directory.Exists(writableDir))
                initialDir = writableDir;
            defaultFileName = Path.GetFileName(writablePath);
        }

        string? picked = ShowSaveFileDialogSTA(
            "Save moved ADT placement as",
            "ADT Files (*.adt)|*.adt|All Files (*.*)|*.*",
            initialDir,
            defaultFileName);
        if (string.IsNullOrWhiteSpace(picked))
            return;

        _selectedPlacementSaveTargetPath = picked;
        _placementSaveTargetsBySourcePath[sourcePath] = picked;
        int pendingForSource = GetPendingPlacementCountForSource(sourcePath);
        _selectedPlacementSaveStatus = pendingForSource > 0
            ? $"Ready to save {pendingForSource} staged placement move(s) from {Path.GetFileName(sourcePath)} to {picked}."
            : $"Default save target for {Path.GetFileName(sourcePath)} set to {picked}.";
    }

    private void SaveSelectedPlacementEdit()
    {
        if (!EnsureSelectedPlacementSourcePath(out string sourcePath, out string error))
        {
            _selectedPlacementSaveStatus = error;
            return;
        }

        SaveStagedPlacementEdits(sourcePath);
    }

    private void DrawPlacementSaveQueueActions(bool includeCurrentSourceSave)
    {
        int pendingEditCount = GetPendingPlacementEditCount();
        int pendingSourceCount = GetPendingPlacementSourceCount();

        if (pendingEditCount <= 0)
            return;

        ImGui.Separator();
        ImGui.Text($"{pendingEditCount} staged placement move(s) across {pendingSourceCount} ADT source(s).");

        string? currentSourcePath = null;
        int currentSourcePendingCount = 0;
        if (includeCurrentSourceSave && TryGetSelectedPlacementSourcePathForQueue(out string sourcePath))
        {
            currentSourcePath = sourcePath;
            currentSourcePendingCount = GetPendingPlacementCountForSource(sourcePath);
        }

        if (includeCurrentSourceSave)
        {
            if (currentSourcePendingCount <= 0)
                ImGui.BeginDisabled();
            if (ImGui.Button("Save Current Source") && currentSourcePath != null)
                SaveStagedPlacementEdits(currentSourcePath);
            if (currentSourcePendingCount <= 0)
                ImGui.EndDisabled();

            ImGui.SameLine();
        }

        if (ImGui.Button("Save All Pending"))
            SaveStagedPlacementEdits();

        if (ImGui.CollapsingHeader("Pending Dirty Sources", ImGuiTreeNodeFlags.DefaultOpen))
        {
            foreach ((string pendingSourcePath, int editCount, string? targetPath) in EnumeratePendingPlacementSourceSummaries())
            {
                ImGui.TextWrapped($"{editCount} move(s): {pendingSourcePath}");
                ImGui.TextDisabled(string.IsNullOrWhiteSpace(targetPath)
                    ? "Output: choose an .adt path before save."
                    : $"Output: {targetPath}");
            }
        }
    }

    private void SaveStagedPlacementEdits(string? sourcePathFilter = null)
    {
        if (_dataSource == null)
        {
            _selectedPlacementSaveStatus = "Placement save failed: no data source is loaded.";
            return;
        }

        List<(string SourcePath, List<StagedPlacementEdit> Edits)> groups = BuildPendingPlacementSaveGroups(sourcePathFilter);
        if (groups.Count == 0)
        {
            _selectedPlacementSaveStatus = string.IsNullOrWhiteSpace(sourcePathFilter)
                ? "No staged placement moves to save."
                : "No staged placement moves are pending for the selected ADT source.";
            return;
        }

        List<string> missingTargets = new();
        foreach ((string sourcePath, _) in groups)
        {
            if (!_placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? outputPath) || string.IsNullOrWhiteSpace(outputPath))
                missingTargets.Add(sourcePath);
        }

        if (missingTargets.Count > 0)
        {
            string missingSummary = missingTargets.Count == 1
                ? missingTargets[0]
                : $"{missingTargets.Count} ADT sources";
            _selectedPlacementSaveStatus = $"Choose an output .adt path before saving pending placement moves for {missingSummary}.";
            return;
        }

        var savedKeys = new List<PlacementEditKey>();
        int savedSourceCount = 0;
        int savedEditCount = 0;

        try
        {
            foreach ((string sourcePath, List<StagedPlacementEdit> edits) in groups)
            {
                string outputPath = _placementSaveTargetsBySourcePath[sourcePath];
                byte[]? sourceBytes = File.Exists(outputPath)
                    ? File.ReadAllBytes(outputPath)
                    : _dataSource.ReadFile(sourcePath);
                if (sourceBytes == null)
                    throw new InvalidOperationException($"The source ADT could not be read from the current data source: {sourcePath}");

                throw new NotSupportedException("AdtPlacement editing requires legacy WoWRollback module."); byte[] updatedBytes = new byte[0];

                string? outputDirectory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrWhiteSpace(outputDirectory))
                    Directory.CreateDirectory(outputDirectory);

                File.WriteAllBytes(outputPath, updatedBytes);

                savedSourceCount++;
                savedEditCount += edits.Count;
                foreach (StagedPlacementEdit edit in edits)
                    savedKeys.Add(edit.Key);
            }

            foreach (PlacementEditKey key in savedKeys)
                _stagedPlacementEdits.Remove(key);

            if (TryGetSelectedPlacementKey(out PlacementEditKey selectedKey) && savedKeys.Contains(selectedKey))
            {
                _selectedPlacementOriginalPosition = _selectedPlacementEditedPosition;
                _selectedPlacementDirty = false;
            }

            _selectedPlacementSaveStatus = BuildPlacementSaveCompletionStatus(groups, savedEditCount, savedSourceCount);
        }
        catch (Exception ex)
        {
            _selectedPlacementSaveStatus = $"Placement save failed: {ex.Message}";
            return;
        }

        SyncSelectedPlacementEditState();
        RefreshSelectedWorldObjectInfo();
    }

    private string BuildPlacementSaveCompletionStatus(
        IReadOnlyList<(string SourcePath, List<StagedPlacementEdit> Edits)> groups,
        int savedEditCount,
        int savedSourceCount)
    {
        List<string> outputPaths = groups
            .Select(group => _placementSaveTargetsBySourcePath[group.SourcePath])
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (outputPaths.Count == 1)
        {
            return $"Saved {savedEditCount} staged placement move(s) across {savedSourceCount} ADT source(s) to {outputPaths[0]}. Source ADTs were left untouched.";
        }

        if (outputPaths.All(IsProjectManagedOutputPath) && !string.IsNullOrWhiteSpace(_editorProjectOutputDir))
        {
            string projectOutputDir = Path.Combine(_editorProjectOutputDir, "lk-split");
            return $"Saved {savedEditCount} staged placement move(s) across {savedSourceCount} ADT source(s) into {projectOutputDir}. Source ADTs were left untouched.";
        }

        string previewPaths = string.Join("; ", outputPaths.Take(2));
        if (outputPaths.Count > 2)
            previewPaths += $"; +{outputPaths.Count - 2} more";

        return $"Saved {savedEditCount} staged placement move(s) across {savedSourceCount} ADT source(s). Output ADT copies: {previewPaths}. Source ADTs were left untouched.";
    }

    private List<(string SourcePath, List<StagedPlacementEdit> Edits)> BuildPendingPlacementSaveGroups(string? sourcePathFilter)
    {
        var grouped = new Dictionary<string, List<StagedPlacementEdit>>(StringComparer.OrdinalIgnoreCase);

        foreach (StagedPlacementEdit edit in _stagedPlacementEdits.Values)
        {
            if (!string.IsNullOrWhiteSpace(sourcePathFilter)
                && !string.Equals(edit.SourcePath, sourcePathFilter, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!grouped.TryGetValue(edit.SourcePath, out List<StagedPlacementEdit>? edits))
            {
                edits = new List<StagedPlacementEdit>();
                grouped.Add(edit.SourcePath, edits);
            }

            edits.Add(edit);
        }

        return grouped
            .OrderBy(entry => entry.Key, StringComparer.OrdinalIgnoreCase)
            .Select(entry => (entry.Key, entry.Value))
            .ToList();
    }

    private IEnumerable<(string SourcePath, int EditCount, string? TargetPath)> EnumeratePendingPlacementSourceSummaries()
    {
        foreach ((string sourcePath, List<StagedPlacementEdit> edits) in BuildPendingPlacementSaveGroups(sourcePathFilter: null))
        {
            _placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? targetPath);
            yield return (sourcePath, edits.Count, targetPath);
        }
    }

    private bool EnsureSelectedPlacementSourcePath(out string sourcePath, out string error)
    {
        sourcePath = _selectedPlacementSourcePath ?? string.Empty;
        error = string.Empty;

        if (!string.IsNullOrWhiteSpace(sourcePath))
            return true;

        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
        {
            error = "No tile-backed world object is selected.";
            return false;
        }

        if (!_worldScene.TryGetSelectedPlacementSourceData(out sourcePath, out _))
        {
            error = "The selected placement source ADT could not be read from the current data source.";
            return false;
        }

        _selectedPlacementSourcePath = sourcePath;
        if (!string.IsNullOrWhiteSpace(sourcePath))
        {
            if (!_placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? savedTarget)
                || string.IsNullOrWhiteSpace(savedTarget)
                || IsProjectManagedOutputPath(savedTarget))
            {
                savedTarget = BuildProjectManagedPlacementOutputPath(sourcePath);
                _placementSaveTargetsBySourcePath[sourcePath] = savedTarget;
            }

            if (string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath) || IsProjectManagedOutputPath(_selectedPlacementSaveTargetPath))
                _selectedPlacementSaveTargetPath = savedTarget;
        }

        return true;
    }

    private void UpsertSelectedPlacementEdit()
    {
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return;

        if (!EnsureSelectedPlacementSourcePath(out string sourcePath, out _))
            return;

        PlacementEditKey key = CreatePlacementEditKey(_worldScene.SelectedObjectType, _worldScene.SelectedInstance.Value);
        _stagedPlacementEdits[key] = new StagedPlacementEdit
        {
            Key = key,
            SourcePath = sourcePath,
            OriginalPosition = _selectedPlacementOriginalPosition,
            EditedPosition = _selectedPlacementEditedPosition,
        };

        if (!string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath))
            _placementSaveTargetsBySourcePath[sourcePath] = _selectedPlacementSaveTargetPath!;
    }

    private void RemoveSelectedPlacementEdit()
    {
        if (!TryGetSelectedPlacementKey(out PlacementEditKey key))
            return;

        _stagedPlacementEdits.Remove(key);
    }

    private bool TryGetSelectedPlacementKey(out PlacementEditKey key)
    {
        key = default;
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return false;

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        Terrain.ObjectType selectedType = _worldScene.SelectedObjectType;
        if (!selected.HasTileCoordinate || selected.PlacementEntryIndex < 0 || selectedType is not (Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo))
            return false;

        key = CreatePlacementEditKey(selectedType, selected);
        return true;
    }

    private bool TryGetSelectedPlacementSourcePathForQueue(out string sourcePath)
    {
        sourcePath = string.Empty;
        if (!_selectedPlacementDirty && !HasPendingPlacementEdits())
            return false;

        return EnsureSelectedPlacementSourcePath(out sourcePath, out _);
    }

    private PlacementEditKey CreatePlacementEditKey(Terrain.ObjectType objectType, ObjectInstance selected)
    {
        return new PlacementEditKey(objectType, selected.TileX, selected.TileY, selected.PlacementEntryIndex, selected.UniqueId);
    }

    private bool HasPendingPlacementEdits()
    {
        return _stagedPlacementEdits.Count > 0;
    }

    private int GetPendingPlacementEditCount()
    {
        return _stagedPlacementEdits.Count;
    }

    private int GetPendingPlacementSourceCount()
    {
        return _stagedPlacementEdits.Values
            .Select(edit => edit.SourcePath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Count();
    }

    private int GetPendingPlacementCountForSource(string sourcePath)
    {
        int count = 0;
        foreach (StagedPlacementEdit edit in _stagedPlacementEdits.Values)
        {
            if (string.Equals(edit.SourcePath, sourcePath, StringComparison.OrdinalIgnoreCase))
                count++;
        }

        return count;
    }

    private int GetPendingPlacementSourceCountMissingTargets()
    {
        int count = 0;
        foreach ((string sourcePath, _, string? targetPath) in EnumeratePendingPlacementSourceSummaries())
        {
            if (string.IsNullOrWhiteSpace(targetPath))
                count++;
        }

        return count;
    }

    private string BuildSelectedPlacementSaveStatus()
    {
        int pendingForSource = 0;
        if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath))
            pendingForSource = GetPendingPlacementCountForSource(_selectedPlacementSourcePath);

        if (pendingForSource > 0)
        {
            return string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                ? $"Preview updated. {pendingForSource} staged placement move(s) are pending for this ADT source. Choose an output .adt path before saving."
                : $"Preview updated. {pendingForSource} staged placement move(s) are pending for this ADT source.";
        }

        return string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
            ? "Preview updated. A timestamped project output folder will be used unless you choose a different .adt path."
            : "Preview updated. Save writes a translated copy into the active project output folder unless you override the target.";
    }

    private static bool PositionsNearlyEqual(Vector3 left, Vector3 right)
    {
        return Vector3.DistanceSquared(left, right) < 0.0001f;
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
            var hoveredPm4Key = _worldScene.ShowPm4Overlay ? _worldScene.HoveredAssetInfo?.Pm4ObjectKey : null;

            if (addPm4ToCollection)
            {
                // Only the Shift+LMB collection branch needs the ray PM4 pick;
                // the normal-click path picks PM4 inside TryHandleSceneClickSelection
                // (a duplicate outer pick here doubled the per-click cost on dense maps).
                if (clickSw != null) clickPickStartTicks = clickSw.ElapsedTicks;
                _worldScene.TryPickPm4ObjectByRay(rayOrigin, rayDir, out var pm4HitKey, out var _, out _);
                if (clickSw != null) clickPickMs = (clickSw.ElapsedTicks - clickPickStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

                ClearPendingClickSelection();
                _worldScene.ClearTaxiSelection();
                _worldScene.ClearSelection();
                ClearSelectedAreaPoiInfo();

                var collectionPm4Key = hoveredPm4Key ?? pm4HitKey;
                if (collectionPm4Key.HasValue && _worldScene.SelectPm4Object(collectionPm4Key.Value))
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
            _worldScene.ClearPm4ObjectSelection();
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

        if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            string nearestRef = float.IsNaN(debugInfo.NearestPositionRefDistance)
                ? "n/a"
                : $"{debugInfo.NearestPositionRefDistance:F2}";

            _selectedObjectInfo =
                $"PM4 Object\n" +
                $"Tile: ({debugInfo.TileX}, {debugInfo.TileY})\n" +
                $"CK24: 0x{debugInfo.Ck24:X6} (type=0x{debugInfo.Ck24Type:X2}, obj={debugInfo.Ck24ObjectId}, viewerPart={debugInfo.ObjectPartId})\n" +
                $"Viewer Part: assigned during the current overlay build after viewer-side splitting; not a raw PM4 field\n" +
                $"MSLK Group: 0x{debugInfo.LinkGroupObjectId:X8}\n" +
                $"Linked MPRL refs: {debugInfo.LinkedPositionRefCount}\n" +
                $"Surfaces: {debugInfo.SurfaceCount}\n" +
                $"GroupKey: 0x{debugInfo.DominantGroupKey:X2}  AttrMask: 0x{debugInfo.DominantAttributeMask:X2}  MscnRef: {debugInfo.DominantMscnRefIndex}\n" +
                $"Planar: swap={debugInfo.SwapPlanarAxes} invertU={debugInfo.InvertU} invertV={debugInfo.InvertV} windingFlip={debugInfo.InvertsWinding}\n" +
                $"Center: ({debugInfo.Center.X:F1}, {debugInfo.Center.Y:F1}, {debugInfo.Center.Z:F1})\n" +
                $"Nearest MPRL: {nearestRef}\n" +
                $"Offset: ({_worldScene.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.SelectedPm4ObjectTranslation.Z:F2})";
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
            $"Offset: ({_worldScene.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.SelectedPm4ObjectTranslation.Z:F2})";
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
    }

    private void DrawSceneHoverAssetOverlay()
    {
        if (_visualInvestigationMode == VisualInvestigationMode.Adt)
        {
            TryDrawTerrainChunkHoverOverlay();
            return;
        }

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
        ImGui.TextColored(GetHoveredAssetTitleColor(info), info.DisplayName);
        ImGui.SetWindowFontScale(1.0f);
        ImGui.TextColored(new Vector4(1.0f, 0.91f, 0.56f, 1.0f), info.AssetKind);

        if (!string.IsNullOrWhiteSpace(info.SourcePath))
        {
            ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 340f);
            ImGui.TextColored(new Vector4(0.54f, 0.84f, 0.52f, 1.0f), info.SourcePath);
            ImGui.PopTextWrapPos();
        }

        if (!string.IsNullOrWhiteSpace(info.DetailLine))
            ImGui.TextColored(new Vector4(0.86f, 0.88f, 0.94f, 1.0f), info.DetailLine);

        ImGui.Separator();

        ImGui.TextColored(new Vector4(0.72f, 0.78f, 0.90f, 1.0f), $"World: ({info.WorldPosition.X:F1}, {info.WorldPosition.Y:F1}, {info.WorldPosition.Z:F1})");

        if (info.Pm4ObjectKey.HasValue)
            DrawHoveredPm4MatchCandidates(info.Pm4ObjectKey.Value);

        ImGui.Separator();
        ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.38f, 1.0f), "Left-click to inspect");

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

        if (_worldScene == null || !_worldScene.TryBuildPm4ObjectMatch(objectKey, maxMatches, out Pm4ObjectMatchObject hoveredMatch))
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
            _activeBottomTabIndex = Math.Max(0, settings.ActiveBottomTab);
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
                ? ClampTerrainWeakSignalRestoreZ(settings.WeakSignalTerrainRestoreCandidateMinHeight)
                : TerrainWeakSignalRestoreDefaultMinZ;
            _terrainWeakSignalRestoreCandidateMaxHeight = float.IsFinite(settings.WeakSignalTerrainRestoreCandidateMaxHeight)
                ? ClampTerrainWeakSignalRestoreZ(settings.WeakSignalTerrainRestoreCandidateMaxHeight)
                : TerrainWeakSignalRestoreDefaultMaxZ;
            GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
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
        public int WmoMliqRotationQuarterTurns { get; set; }
        public bool HasExplicitWmoMliqRotationOverride { get; set; }
        public string? LastGameFolderPath { get; set; }
        public string? LastLooseOverlayPath { get; set; }
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

    // Helper methods for converter CLI execution
    private string? FindConverterExecutable()
    {
        // Try to find the converter executable in the build output
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var candidates = new[]
        {
            Path.Combine(baseDir, "WowViewer.Tool.Converter.exe"),
            Path.Combine(baseDir, "tools", "converter", "WowViewer.Tool.Converter", "bin", "Debug", "net9.0", "WowViewer.Tool.Converter.exe"),
            Path.Combine(baseDir, "..", "tools", "converter", "WowViewer.Tool.Converter", "bin", "Debug", "net9.0", "WowViewer.Tool.Converter.exe"),
            Path.Combine(baseDir, "..", "..", "tools", "converter", "WowViewer.Tool.Converter", "bin", "Debug", "net9.0", "WowViewer.Tool.Converter.exe"),
        };

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
                return Path.GetFullPath(candidate);
        }

        return null;
    }

    private async Task<ConverterResult> RunConverterAsync(string exePath, List<string> args, List<string> log, bool scrollToBottom)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = exePath,
            Arguments = string.Join(" ", args.Select(a => a.Contains(' ') ? $"\"{a}\"" : a)),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = AppDomain.CurrentDomain.BaseDirectory,
        };

        var result = new ConverterResult { Success = false };

        try
        {
            using var process = Process.Start(startInfo);
            if (process == null)
            {
                result.Error = "Failed to start converter process";
                return result;
            }

            var outputLines = new List<string>();
            var errorLines = new List<string>();

            process.OutputDataReceived += (_, e) =>
            {
                if (e.Data != null)
                {
                    outputLines.Add(e.Data);
                    lock (log)
                    {
                        log.Add(e.Data);
                    }
                    scrollToBottom = true;
                }
            };

            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data != null)
                {
                    errorLines.Add(e.Data);
                    lock (log)
                    {
                        log.Add($"[ERR] {e.Data}");
                    }
                    scrollToBottom = true;
                }
            };

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await process.WaitForExitAsync();

            result.Success = process.ExitCode == 0;
            if (!result.Success)
            {
                result.Error = string.Join("\n", errorLines);
            }

            // Parse output for structured results
            foreach (var line in outputLines)
            {
                if (line.StartsWith("Tiles converted:"))
                {
                    var parts = line.Split(':');
                    if (parts.Length > 1 && int.TryParse(parts[1].Trim(), out int tiles))
                        result.TilesConverted = tiles;
                }
                else if (line.StartsWith("Total tiles:"))
                {
                    var parts = line.Split(':');
                    if (parts.Length > 1 && int.TryParse(parts[1].Trim(), out int total))
                        result.TotalTiles = total;
                }
                else if (line.StartsWith("Elapsed:"))
                {
                    var parts = line.Split(':');
                    if (parts.Length > 1 && int.TryParse(parts[1].Trim().Replace("ms", ""), out int elapsed))
                        result.ElapsedMs = elapsed;
                }
            }
        }
        catch (Exception ex)
        {
            result.Error = ex.Message;
        }

        return result;
    }

    private sealed class ConverterResult
    {
        public bool Success { get; set; }
        public string? Error { get; set; }
        public int TilesConverted { get; set; }
        public int TotalTiles { get; set; }
        public int ElapsedMs { get; set; }
    }
}
