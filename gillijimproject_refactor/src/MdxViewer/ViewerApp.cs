using System.Numerics;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Export;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using MdxViewer.Catalog;
using MdxViewer.Population;
using MdxViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.Converters;
using WoWMapConverter.Core.VLM;

namespace MdxViewer;

/// <summary>
/// Main viewer application. Owns window, GL context, ImGui, camera, renderer.
/// Provides menu bar, file browser, model info panel, and 3D viewport.
/// </summary>
public partial class ViewerApp : IDisposable
{
    private enum ModelContainerKind
    {
        Unknown,
        Mdlx,
        Md20
    }

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
    private string? _lastVirtualPath; // Virtual path of last loaded file (for DBC lookup)
    private string _statusMessage = "No data source loaded. Use File > Open Game Folder or Open File.";
    private AreaTableService? _areaTableService;
    private string _currentAreaName = "";
    private int _currentMapId = -1; // MapID of the currently loaded world

    // Map discovery
    private List<MapDefinition> _discoveredMaps = new();
    private WoWMapConverter.Core.Services.Md5TranslateIndex? _md5Index;
    private MinimapRenderer? _minimapRenderer;
    private WdlPreviewRenderer? _wdlPreviewRenderer;
    private bool _showWdlPreview = false;
    private MapDefinition? _selectedMapForPreview;
    private Vector2? _selectedSpawnTile; // WDL tile coordinates (0-63)
    private float _minimapZoom = 4f; // Number of tiles visible in each direction from camera
    private bool _fullscreenMinimap = false; // M key toggles fullscreen minimap
    private Vector2 _minimapPanOffset = Vector2.Zero; // Pan offset for click-and-drag
    private bool _minimapDragging = false;
    private Vector2 _minimapDragStart = Vector2.Zero;
    private Rendering.LoadingScreen? _loadingScreen;

    // Output directories (next to the executable)
    private static readonly string OutputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");
    private static readonly string CacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache");
    private static readonly string ExportDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "export");
    private static readonly string SettingsDir = Path.Combine(OutputDir, "settings");
    private static readonly string ViewerSettingsPath = Path.Combine(SettingsDir, "viewer_settings.json");
    private static readonly string WmoV14ToV17OutputDir = Path.Combine(ExportDir, "WMOv14_to_v17_output");
    private static readonly string WmoV17ToV14OutputDir = Path.Combine(ExportDir, "WMOv17_to_v14_output");

    // File browser state
    private List<string> _filteredFiles = new();
    private string _searchFilter = "";
    private string _extensionFilter = ".mdx";
    private int _selectedFileIndex = -1;
    private string? _loadedFilePath;
    private string? _loadedFileName;

    // Model info
    private string _modelInfo = "";
    
    // Stored loaded model data for export (avoids re-parsing from disk)
    private WmoV14ToV17Converter.WmoV14Data? _loadedWmo;
    private MdxFile? _loadedMdx;

    // Mouse state
    private float _lastMouseX, _lastMouseY;
    private bool _mouseDown;
    private bool _mouseOverViewport;

    // UI state
    private bool _showFileBrowser = true;
    private bool _showModelInfo = true;
    private bool _showTerrainControls = true;
    private bool _showDemoWindow = false;
    private bool _showLogViewer = false;
    private bool _showMinimapWindow = false;
    private bool _showPerfWindow = false;
    private AssetCatalogView? _catalogView;
    private bool _wantOpenFile = false;
    private bool _wantOpenFolder = false;
    private bool _wantExportGlb = false;
    private bool _wantExportGlbCollision = false;

    private enum TerrainTileScope
    {
        CurrentTile = 0,
        LoadedTiles = 1,
        WholeMap = 2,
        CustomList = 3,
    }

    private enum TerrainExportKind
    {
        None = 0,
        AlphaCurrentTileAtlas = 1,
        AlphaCurrentTileChunksFolder = 2,
        AlphaLoadedTilesFolder = 3,
        AlphaWholeMapFolder = 4,
    }

    private enum TerrainImportKind
    {
        None = 0,
        AlphaFolder = 1,
    }

    private bool _wantTerrainExport;
    private TerrainExportKind _terrainExportKind = TerrainExportKind.None;

    private bool _wantTerrainImport;
    private TerrainImportKind _terrainImportKind = TerrainImportKind.None;
    private bool _showAlphaFolderImportScope;
    private TerrainTileScope _terrainTileScope = TerrainTileScope.LoadedTiles;
    private string _terrainImportFolder = "";
    private string _terrainCustomTilesText = "";

    // Sidebar layout
    private bool _showLeftSidebar = true;
    private bool _showRightSidebar = true;
    private const float SidebarWidth = 320f;
    private const float MenuBarHeight = 22f;
    private const float ToolbarHeight = 32f;
    private const float StatusBarHeight = 24f;

    /// <summary>When true, load all tiles at startup instead of AOI streaming. Default: false (stream tiles as camera moves).</summary>
    public bool FullLoadMode { get; set; } = false;

    // Terrain/World state
    private TerrainManager? _terrainManager;
    private VlmTerrainManager? _vlmTerrainManager;
    private WorldScene? _worldScene;
    private bool _wantOpenVlmProject = false;

    // Object picking state
    private int _selectedObjectIndex = -1; // -1=none, 0..modf-1=WMO, modf..modf+mddf-1=MDX
    private string _selectedObjectType = "";
    private string _selectedObjectInfo = "";
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
    private string _wlLayerSelectedSourcePath = "";

    // Camera speed (adjustable via UI)
    private float _cameraSpeed = 50f;
    // Field of view in degrees (adjustable via UI)
    private float _fovDegrees = 45f;

    private bool _autoFrameModelOnLoad = true;
    private static readonly string[] WmoLiquidRotationLabels = { "0°", "90°", "180°", "270°" };

    // Sky gradient for standalone model viewing
    private uint _skyVao, _skyVbo, _skyShader;
    private bool _skyReady;

    // Folder dialog workaround (ImGui doesn't have native dialogs)
    private bool _showFolderInput = false;
    private string _folderInputBuf = "";
    private bool _showListfileInput = false;
    private string _listfileInputBuf = "";

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
    private string _mapConvertLkMapDir = ""; // LK→Alpha: directory containing LK ADT files
    private bool _mapConvertVerbose = true;
    private bool _mapConverting = false;
    private readonly List<string> _mapConvertLog = new();
    private bool _mapConvertScrollToBottom = false;
    private string? _mapConvertError = null;
    private bool _mapConvertDone = false;

    // WMO Converter state
    private bool _showWmoConverterDialog = false;
    private int _wmoConvertDirection = 0; // 0 = Alpha(v14/v16)→LK(v17), 1 = LK(v17)→Alpha(v14)
    private bool _wmoConvertExtended = false;
    private string _wmoConvertSourcePath = "";
    private string _wmoConvertOutputPath = "";
    private bool _wmoConvertCopyTextures = true;
    private bool _wmoConverting = false;
    private readonly List<string> _wmoConvertLog = new();
    private bool _wmoConvertScrollToBottom = false;
    private string? _wmoConvertError = null;
    private bool _wmoConvertDone = false;

    // VLM Dataset Generator state
    private bool _showVlmExportDialog = false;
    private string _vlmClientPath = "";
    private string _vlmMapName = "development";
    private string _vlmOutputDir = "";
    private int _vlmTileLimit = 0; // 0 = unlimited
    private bool _vlmExporting = false;
    private readonly List<string> _vlmExportLog = new();
    private bool _vlmExportScrollToBottom = false;
    private VlmExportResult? _vlmExportResult = null;

    public void Run(string[]? initialArgs = null)
    {
        var opts = WindowOptions.Default;
        opts.Size = new Vector2D<int>(1600, 900);
        opts.Title = "WoW Model Viewer";
        opts.API = new GraphicsAPI(ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion(3, 3));
        opts.VSync = false; // Disable VSync — let the GPU run uncapped for profiling

        _window = Window.Create(opts);
        _window.Load += () => OnLoad(initialArgs);
        _window.Render += OnRender;
        _window.Update += OnUpdate;
        _window.FramebufferResize += OnResize;
        _window.Closing += OnClose;

        _window.Run();
    }

    private void OnLoad(string[]? initialArgs)
    {
        _gl = _window.CreateOpenGL();
        _input = _window.CreateInput();
        _imGui = new ImGuiController(_gl, _window, _input);

        _gl.ClearColor(0.05f, 0.05f, 0.1f, 1.0f); // Dark blue-black default
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Enable(EnableCap.CullFace);

        _loadingScreen = new Rendering.LoadingScreen(_gl);

        // Style ImGui
        var style = ImGui.GetStyle();
        style.WindowRounding = 4f;
        style.FrameRounding = 2f;
        style.Colors[(int)ImGuiCol.WindowBg] = new Vector4(0.12f, 0.12f, 0.14f, 0.95f);
        style.Colors[(int)ImGuiCol.MenuBarBg] = new Vector4(0.15f, 0.15f, 0.18f, 1.0f);

        TryAutoPopulateAlphaCoreRoot();
        LoadViewerSettings();

        // Mouse input for viewport (not consumed by ImGui)
        foreach (var mouse in _input.Mice)
        {
            mouse.MouseDown += (_, btn) =>
            {
                if (btn == MouseButton.Right && !ImGui.GetIO().WantCaptureMouse)
                    _mouseDown = true;
                if (btn == MouseButton.Left && !ImGui.GetIO().WantCaptureMouse && _worldScene != null
                    && IsPointInSceneViewport(_lastMouseX, _lastMouseY))
                    PickObjectAtMouse(_lastMouseX, _lastMouseY);
            };
            mouse.MouseUp += (_, btn) =>
            {
                if (btn == MouseButton.Right) _mouseDown = false;
            };
            mouse.MouseMove += (_, pos) =>
            {
                float dx = pos.X - _lastMouseX;
                float dy = pos.Y - _lastMouseY;
                _lastMouseX = pos.X;
                _lastMouseY = pos.Y;

                if (_mouseDown && !ImGui.GetIO().WantCaptureMouse)
                {
                    _camera.Yaw -= dx * 0.5f;   // Drag left = look left, Drag right = look right
                    _camera.Pitch -= dy * 0.5f; // Drag up = look up, Drag down = look down
                    _camera.Pitch = Math.Clamp(_camera.Pitch, -89f, 89f);
                }
            };
            mouse.Scroll += (_, scroll) =>
            {
                if (!ImGui.GetIO().WantCaptureMouse)
                {
                    // Free-fly: scroll moves camera forward/back
                    float speed = 5f * scroll.Y;
                    _camera.Move(speed, 0, 0, 1f);
                }
            };
        }

        // If launched with a file argument, load it
        if (initialArgs != null && initialArgs.Length > 0 && File.Exists(initialArgs[0]))
        {
            LoadFileFromDisk(initialArgs[0]);
        }
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

        bool isPlaying = animator.IsPlaying;
        if (ImGui.Button(isPlaying ? "Pause GO Anim" : "Play GO Anim"))
            animator.IsPlaying = !isPlaying;

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

        var seq = animator.Sequences[animator.CurrentSequence];
        float seqStart = seq.Time.Start;
        float seqEnd = seq.Time.End;
        float currentFrame = Math.Clamp(animator.CurrentFrame, seqStart, seqEnd);
        if (ImGui.SliderFloat("GO Frame", ref currentFrame, seqStart, seqEnd, "%.0f"))
        {
            animator.IsPlaying = false;
            animator.CurrentFrame = currentFrame;
        }

        ImGui.TextDisabled("Note: this affects all visible instances using the same MDX model renderer.");
    }

    private void OnUpdate(double dt)
    {
        _imGui.Update((float)dt);
        HandleKeyboardInput((float)dt);
        UpdateSqlSpawnStreaming();
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
        int tileX = (int)((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
        int tileY = (int)((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
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
    private bool _leftArrowWasPressed = false;
    private bool _rightArrowWasPressed = false;
    private bool _spaceWasPressed = false;

    private void HandleKeyboardInput(float dt)
    {
        if (_input.Keyboards.Count == 0) return;
        var kb = _input.Keyboards[0];

        // M key toggles fullscreen minimap (only when terrain is loaded)
        bool mPressed = kb.IsKeyPressed(Key.M);
        if (mPressed && !_mKeyWasPressed && (_terrainManager != null || _vlmTerrainManager != null))
        {
            _fullscreenMinimap = !_fullscreenMinimap;
        }
        _mKeyWasPressed = mPressed;

        // Arrow keys and spacebar for MDX animation control
        if (_renderer is MdxRenderer mdxR && mdxR.Animator != null && mdxR.Animator.Sequences.Count > 0)
        {
            var animator = mdxR.Animator;
            int currentSeq = animator.CurrentSequence;
            
            if (currentSeq >= 0 && currentSeq < animator.Sequences.Count)
            {
                var seq = animator.Sequences[currentSeq];
                float duration = seq.Time.End - seq.Time.Start;
                float currentFrame = animator.CurrentFrame;
                
                // Left arrow: step backward
                bool leftPressed = kb.IsKeyPressed(Key.Left);
                if (leftPressed && !_leftArrowWasPressed)
                {
                    animator.IsPlaying = false;
                    animator.StepToPrevKeyframe();
                }
                _leftArrowWasPressed = leftPressed;
                
                // Right arrow: step forward
                bool rightPressed = kb.IsKeyPressed(Key.Right);
                if (rightPressed && !_rightArrowWasPressed)
                {
                    animator.IsPlaying = false;
                    animator.StepToNextKeyframe();
                }
                _rightArrowWasPressed = rightPressed;
                
                // Spacebar: toggle play/pause
                bool spacePressed = kb.IsKeyPressed(Key.Space);
                if (spacePressed && !_spaceWasPressed)
                {
                    animator.IsPlaying = !animator.IsPlaying;
                }
                _spaceWasPressed = spacePressed;
            }
        }

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
                    _terrainManager.UpdateAOI(_camera.Position);
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
            float aspect = (float)size.X / Math.Max(size.Y, 1);
            var view = _camera.GetViewMatrix();
            float farPlane = (_terrainManager != null || _vlmTerrainManager != null) ? 5000f : 10000f;
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, farPlane);

            // Update terrain AOI before rendering
            if (_terrainManager != null)
                _terrainManager.UpdateAOI(_camera.Position);
            else if (_vlmTerrainManager != null)
                _vlmTerrainManager.UpdateAOI(_camera.Position);

            // Update current area name from chunk under camera (throttled to avoid per-frame overhead)
            if (_areaTableService != null && _terrainManager != null && _frameCount == 0)
            {
                int areaId = 0;
                var chunkInfo = _terrainManager.Renderer.GetChunkInfoAt(_camera.Position.X, _camera.Position.Y);
                if (chunkInfo.HasValue)
                    areaId = chunkInfo.Value.AreaId;
                else
                {
                    var chunk = _terrainManager.Renderer.GetChunkAt(_camera.Position.X, _camera.Position.Y);
                    if (chunk != null)
                        areaId = chunk.AreaId;
                }

                if (areaId != 0)
                {
                    // Filter by MapID to avoid showing areas from other continents
                    var name = _areaTableService.GetAreaDisplayNameForMap(areaId, _currentMapId);
                    if (name == null)
                    {
                        // Fallback if MapID filtering fails
                        name = _areaTableService.GetAreaDisplayName(areaId);
                        if (name.StartsWith("Unknown"))
                            ViewerLog.Trace($"[AreaTable] Lookup miss: AreaId={areaId}, MapId={_currentMapId} → {name}  (table has {_areaTableService.Count} entries)");
                    }
                    _currentAreaName = name ?? "";
                }
                else
                    _currentAreaName = "";
            }

            // Render the scene
            if (_renderer is MdxRenderer mdxR)
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
                mdxR.UpdateAnimation(); // Advance skeletal animation before rendering
                _gl.Disable(EnableCap.Blend);
                mdxR.RenderWithTransform(scale, view, proj, RenderPass.Opaque, 1.0f,
                    fogColor, fogStart, fogEnd, _camera.Position, lightDir, lightColor, ambientColor);
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                mdxR.RenderWithTransform(scale, view, proj, RenderPass.Transparent, 1.0f,
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
            }
            else
            {
                // WorldScene / VLM terrain — handles its own lighting
                _renderer.Render(view, proj);
            }
        }

        // Render ImGui overlay
        DrawUI();
        _imGui.Render();
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
        DrawMenuBar();

        // Top toolbar with visibility checkboxes (only when terrain is loaded)
        if (_terrainManager != null || _vlmTerrainManager != null)
            DrawToolbar();

        // Fixed sidebar layout
        if (_showLeftSidebar)
            DrawLeftSidebar();
        if (_showRightSidebar)
            DrawRightSidebar();

        DrawStatusBar();
        
        // Fullscreen minimap overlay (M key toggle)
        if (_fullscreenMinimap && (_worldScene != null || _vlmTerrainManager != null))
            DrawFullscreenMinimap();

        // Asset Catalog (floating window)
        _catalogView?.Draw();

        // Log Viewer (floating window)
        if (_showLogViewer)
            DrawLogViewer();

        // WDL Preview (floating window)
        if (_showWdlPreview)
            DrawWdlPreviewDialog();

        // Minimap (floating window)
        if (_showMinimapWindow && (_terrainManager != null || _vlmTerrainManager != null))
            DrawMinimapWindow();

        // Perf (floating window)
        if (_showPerfWindow)
            DrawPerfWindow();

        // Modal dialogs
        if (_showFolderInput)
            DrawFolderInputDialog();
        if (_showListfileInput)
            DrawListfileInputDialog();
        if (_showAlphaFolderImportScope)
            DrawAlphaFolderImportScopeDialog();
        if (_showVlmExportDialog)
            DrawVlmExportDialog();
        if (_showMapConverterDialog)
            DrawMapConverterDialog();
        if (_showWmoConverterDialog)
            DrawWmoConverterDialog();
    }

    private void DrawMenuBar()
    {
        if (ImGui.BeginMainMenuBar())
        {
            if (ImGui.BeginMenu("File"))
            {
                if (ImGui.MenuItem("Open File..."))
                    _wantOpenFile = true;

                if (ImGui.MenuItem("Open Game Folder (MPQ)..."))
                {
                    _showFolderInput = true;
                    _folderInputBuf = "";
                }

                if (ImGui.MenuItem("Open VLM Project..."))
                    _wantOpenVlmProject = true;

                ImGui.Separator();

                if (ImGui.MenuItem("Generate VLM Dataset..."))
                    _showVlmExportDialog = true;

                if (ImGui.MenuItem("Map Converter..."))
                    _showMapConverterDialog = true;

                if (ImGui.MenuItem("WMO Converter..."))
                    _showWmoConverterDialog = true;

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

                ImGui.Separator();

                ImGui.MenuItem("Left Sidebar", "", ref _showLeftSidebar);
                ImGui.MenuItem("Right Sidebar", "", ref _showRightSidebar);
                ImGui.Separator();
                ImGui.MenuItem("File Browser", "", ref _showFileBrowser);
                ImGui.MenuItem("Model Info", "", ref _showModelInfo);
                ImGui.MenuItem("Terrain Controls", "", ref _showTerrainControls);
                ImGui.MenuItem("Minimap", "", ref _showMinimapWindow);
                ImGui.MenuItem("Log Viewer", "", ref _showLogViewer);
                ImGui.MenuItem("Perf", "", ref _showPerfWindow);
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

            if (ImGui.BeginMenu("Help"))
            {
                if (ImGui.MenuItem("About"))
                    _statusMessage = "WoW Model Viewer — MDX/WMO viewer with GLB export. Built with Silk.NET + ImGui.";
                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Export"))
            {
                if (ImGui.BeginMenu("GLB"))
                {
                    if (ImGui.MenuItem("Export GLB...", _renderer != null))
                        _wantExportGlb = true;
                    if (ImGui.MenuItem("Export GLB (Collision Only)...", _renderer != null))
                        _wantExportGlbCollision = true;
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

                    ImGui.EndMenu();
                }

                ImGui.EndMenu();
            }

            ImGui.EndMainMenuBar();
        }

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
                "Select VLM Project folder (containing dataset/ with JSON files)",
                initialDir: null,
                showNewFolderButton: false);

            if (!string.IsNullOrEmpty(vlmPath) && Directory.Exists(vlmPath))
                LoadVlmProject(vlmPath);
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
                    ExportAlphaTilesFolder(wholeMap: false);
                    break;
                case TerrainExportKind.AlphaWholeMapFolder:
                    ExportAlphaTilesFolder(wholeMap: true);
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

    private IReadOnlyList<(int tileX, int tileY)> GetTileScopeList(TerrainTileScope scope)
    {
        if (scope == TerrainTileScope.CurrentTile)
        {
            var cam = GetCameraTile();
            return new List<(int, int)> { cam };
        }

        if (scope == TerrainTileScope.CustomList)
            return ParseCustomTileList(_terrainCustomTilesText).Distinct().ToList();

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
        atlas.SaveAsPng(picked);
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
            kvp.Value.SaveAsPng(path);
            kvp.Value.Dispose();
        }

        _statusMessage = $"Exported chunks: {folder}";
    }

    private void ExportAlphaTilesFolder(bool wholeMap)
    {
        string? folder = ShowFolderDialogSTA(
            "Select output folder for tile alpha atlases",
            ExportDir,
            showNewFolderButton: true);
        if (string.IsNullOrEmpty(folder))
            return;

        var tiles = wholeMap
            ? GetTileScopeList(TerrainTileScope.WholeMap)
            : GetTileScopeList(TerrainTileScope.LoadedTiles);

        int written = 0;
        foreach (var (tx, ty) in tiles)
        {
            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null) continue;

            using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
            string path = Path.Combine(folder, $"tile_{tx}_{ty}_alpha.png");
            atlas.SaveAsPng(path);
            written++;
        }

        _statusMessage = $"Exported {written} tiles: {folder}";
    }

    private IReadOnlyList<TerrainChunkData>? LoadTileChunksForExport(int tileX, int tileY)
    {
        if (_terrainManager != null)
        {
            if (_terrainManager.TryGetTileLoadResult(tileX, tileY, out var cached))
                return cached.Chunks;

            if (_terrainManager.Adapter.TileExists(tileX, tileY))
                return _terrainManager.Adapter.LoadTileWithPlacements(tileX, tileY).Chunks;
        }

        if (_vlmTerrainManager != null)
        {
            if (_vlmTerrainManager.Loader.TileCoords.Contains((tileX, tileY)))
                return _vlmTerrainManager.Loader.LoadTile(tileX, tileY).Chunks;
        }

        return null;
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

        int scope = (int)_terrainTileScope;
        ImGui.RadioButton("Current tile", ref scope, (int)TerrainTileScope.CurrentTile);
        ImGui.RadioButton("Loaded tiles", ref scope, (int)TerrainTileScope.LoadedTiles);
        ImGui.RadioButton("Whole map", ref scope, (int)TerrainTileScope.WholeMap);
        ImGui.RadioButton("Custom list", ref scope, (int)TerrainTileScope.CustomList);
        _terrainTileScope = (TerrainTileScope)scope;

        if (_terrainTileScope == TerrainTileScope.CustomList)
        {
            ImGui.TextDisabled("One tile per line: x y (or x,y)");
            ImGui.InputTextMultiline("##customTiles", ref _terrainCustomTilesText, 8192, new Vector2(480, 160));
        }

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

        // Ensure tiles are resident if user chose whole-map (Alpha/Standard path).
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

            // Only apply to tiles currently resident on GPU.
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

    private void DrawPerfWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(360, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Perf", ref _showPerfWindow, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer == null)
        {
            ImGui.Text("No terrain loaded.");
            ImGui.End();
            return;
        }

        ImGui.Text($"Chunks: {terrainRenderer.ChunksRendered} rendered, {terrainRenderer.ChunksCulled} culled");
        ImGui.Text($"Draw calls: {terrainRenderer.LastFrameDrawCalls}");
        ImGui.Separator();
        ImGui.Text($"ActiveTexture: {terrainRenderer.LastFrameActiveTextureCalls} calls, {terrainRenderer.LastFrameActiveTextureSkips} skipped");
        ImGui.Text($"BindTexture: {terrainRenderer.LastFrameBindTextureCalls} calls, {terrainRenderer.LastFrameBindTextureSkips} skipped");
        ImGui.Text($"Uniform1 (uHas*): {terrainRenderer.LastFrameUniform1Calls}");
        ImGui.TextDisabled("Stats are for the last terrain Render() call.");

        ImGui.End();
    }

    private void DrawFolderInputDialog()
    {
        if (!_showFolderInput) return;

        // Use WinForms folder browser for native experience
        _showFolderInput = false;

        string? selectedPath = ShowFolderDialogSTA(
            "Select WoW game folder (containing Data/ with MPQs)",
            initialDir: string.IsNullOrEmpty(_folderInputBuf) ? null : _folderInputBuf,
            showNewFolderButton: false);

        if (!string.IsNullOrEmpty(selectedPath) && Directory.Exists(selectedPath))
        {
            _folderInputBuf = selectedPath;
            LoadMpqDataSource(selectedPath, null);
        }
    }

    private void DrawListfileInputDialog()
    {
        // No longer needed — listfile is auto-downloaded
        _showListfileInput = false;
    }

    private void DrawMapConverterDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(580, 520), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 290,
            ImGui.GetIO().DisplaySize.Y / 2 - 260), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Map Converter", ref _showMapConverterDialog))
        {
            ImGui.TextWrapped("Convert maps between Alpha 0.5.3 monolithic WDT and LK 3.3.5 split ADT formats.");
            ImGui.Spacing();

            // Direction selector
            ImGui.Text("Direction:");
            ImGui.RadioButton("Alpha WDT \u2192 LK ADTs", ref _mapConvertDirection, 0);
            ImGui.SameLine();
            ImGui.RadioButton("LK ADTs \u2192 Alpha WDT", ref _mapConvertDirection, 1);
            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Spacing();

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
                    if (picked != null) _mapConvertSourcePath = picked;
                }

                ImGui.Text("Output Directory:");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##a2l_out", ref _mapConvertOutputDir, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##a2l_out"))
                {
                    var picked = ShowFolderDialogSTA("Select output directory for LK ADT files");
                    if (picked != null) _mapConvertOutputDir = picked;
                }
            }
            else
            {
                // LK → Alpha
                ImGui.Text("Source LK WDT:");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##l2a_src", ref _mapConvertSourcePath, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##l2a_src"))
                {
                    string? initDir = !string.IsNullOrEmpty(_mapConvertSourcePath) ? Path.GetDirectoryName(_mapConvertSourcePath) : null;
                    var picked = ShowFileDialogSTA("Select LK WDT file", "WDT Files (*.wdt)|*.wdt|All Files (*.*)|*.*", initDir);
                    if (picked != null) _mapConvertSourcePath = picked;
                }

                ImGui.Text("LK ADT Directory (containing MapName_X_Y.adt files):");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##l2a_mapdir", ref _mapConvertLkMapDir, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##l2a_dir"))
                {
                    var picked = ShowFolderDialogSTA("Select directory containing LK ADT files");
                    if (picked != null) _mapConvertLkMapDir = picked;
                }

                ImGui.Text("Output Alpha WDT Path:");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##l2a_out", ref _mapConvertOutputDir, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##l2a_out"))
                {
                    string? initDir = !string.IsNullOrEmpty(_mapConvertOutputDir) ? Path.GetDirectoryName(_mapConvertOutputDir) : null;
                    var picked = ShowSaveFileDialogSTA("Save Alpha WDT as", "WDT Files (*.wdt)|*.wdt|All Files (*.*)|*.*", initDir);
                    if (picked != null) _mapConvertOutputDir = picked;
                }
            }

            ImGui.Spacing();
            ImGui.Checkbox("Verbose logging", ref _mapConvertVerbose);
            ImGui.Spacing();

            // Auto-fill hints
            if (!string.IsNullOrEmpty(_mapConvertSourcePath) && string.IsNullOrEmpty(_mapConvertOutputDir))
            {
                string srcDir = Path.GetDirectoryName(_mapConvertSourcePath) ?? "";
                string mapName = Path.GetFileNameWithoutExtension(_mapConvertSourcePath);
                if (_mapConvertDirection == 0)
                    _mapConvertOutputDir = Path.Combine(srcDir, $"{mapName}_lk");
                else
                    _mapConvertOutputDir = Path.Combine(srcDir, $"{mapName}_alpha.wdt");
            }
            if (_mapConvertDirection == 1 && !string.IsNullOrEmpty(_mapConvertSourcePath) && string.IsNullOrEmpty(_mapConvertLkMapDir))
            {
                _mapConvertLkMapDir = Path.GetDirectoryName(_mapConvertSourcePath) ?? "";
            }

            // Convert button
            bool canConvert = !_mapConverting
                && !string.IsNullOrWhiteSpace(_mapConvertSourcePath)
                && !string.IsNullOrWhiteSpace(_mapConvertOutputDir)
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

                Task.Run(async () =>
                {
                    try
                    {
                        if (direction == 0)
                        {
                            // Alpha → LK
                            var opts = new WoWMapConverter.Core.Converters.ConversionOptions { Verbose = verbose };
                            var converter = new WoWMapConverter.Core.Converters.AlphaToLkConverter(opts);

                            // Redirect console output to log
                            var origOut = Console.Out;
                            var sw = new StringWriter();
                            Console.SetOut(sw);

                            var result = await converter.ConvertWdtAsync(srcPath, outPath);

                            Console.SetOut(origOut);
                            var lines = sw.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
                            lock (_mapConvertLog) _mapConvertLog.AddRange(lines);
                            _mapConvertScrollToBottom = true;

                            if (result.Success)
                            {
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== SUCCESS: {result.TilesConverted}/{result.TotalTiles} tiles converted in {result.ElapsedMs}ms ===");
                            }
                            else
                            {
                                _mapConvertError = result.Error ?? "Unknown error";
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== FAILED: {result.Error} ===");
                            }
                            foreach (var w in result.Warnings)
                            {
                                lock (_mapConvertLog) _mapConvertLog.Add($"  WARN: {w}");
                            }
                        }
                        else
                        {
                            // LK → Alpha
                            var opts = new WoWMapConverter.Core.Converters.LkToAlphaOptions { Verbose = verbose };
                            var converter = new WoWMapConverter.Core.Converters.LkToAlphaConverter(opts);

                            var origOut = Console.Out;
                            var sw = new StringWriter();
                            Console.SetOut(sw);

                            var result = await converter.ConvertAsync(srcPath, lkMapDir, outPath);

                            Console.SetOut(origOut);
                            var lines = sw.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
                            lock (_mapConvertLog) _mapConvertLog.AddRange(lines);
                            _mapConvertScrollToBottom = true;

                            if (result.Success)
                            {
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== SUCCESS: {result.TilesConverted}/{result.TotalTiles} tiles converted in {result.ElapsedMs}ms ===");
                            }
                            else
                            {
                                _mapConvertError = result.Error ?? "Unknown error";
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== FAILED: {result.Error} ===");
                            }
                            foreach (var w in result.Warnings)
                            {
                                lock (_mapConvertLog) _mapConvertLog.Add($"  WARN: {w}");
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
                    // Find the WDT in the output directory
                    var wdtFiles = Directory.GetFiles(_mapConvertOutputDir, "*.wdt");
                    if (wdtFiles.Length > 0)
                    {
                        LoadWdtTerrain(wdtFiles[0]);
                        _showMapConverterDialog = false;
                    }
                }
            }
            else if (_mapConvertDone && _mapConvertError == null && _mapConvertDirection == 1)
            {
                if (ImGui.Button("Load Converted Alpha WDT in Viewer"))
                {
                    if (File.Exists(_mapConvertOutputDir))
                    {
                        LoadWdtTerrain(_mapConvertOutputDir);
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

            ImGui.Text("Mode:");
            bool isBasic = !_wmoConvertExtended;
            if (ImGui.RadioButton("Basic", isBasic)) _wmoConvertExtended = false;
            ImGui.SameLine();
            bool isExtended = _wmoConvertExtended;
            if (ImGui.RadioButton("Extended", isExtended)) _wmoConvertExtended = true;

            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Spacing();

            // Auto-select currently loaded WMO
            if (!string.IsNullOrEmpty(_loadedFilePath)
                && string.Equals(Path.GetExtension(_loadedFilePath), ".wmo", StringComparison.OrdinalIgnoreCase)
                && string.IsNullOrEmpty(_wmoConvertSourcePath))
            {
                _wmoConvertSourcePath = _loadedFilePath;
            }

            ImGui.Text("Source WMO:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##wmo_src", ref _wmoConvertSourcePath, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##wmo_src"))
            {
                string? initDir = !string.IsNullOrEmpty(_wmoConvertSourcePath) ? Path.GetDirectoryName(_wmoConvertSourcePath) : null;
                var picked = ShowFileDialogSTA("Select WMO file", "WMO Files (*.wmo)|*.wmo|All Files (*.*)|*.*", initDir);
                if (picked != null) _wmoConvertSourcePath = picked;
            }

            string outputBaseDir = (_wmoConvertDirection == 0) ? WmoV14ToV17OutputDir : WmoV17ToV14OutputDir;
            string outputRootPath = "";
            if (!string.IsNullOrWhiteSpace(_wmoConvertSourcePath))
            {
                string baseName = Path.GetFileNameWithoutExtension(_wmoConvertSourcePath);
                string suffix = (_wmoConvertDirection == 0) ? ".v17.wmo" : ".v14.wmo";
                outputRootPath = Path.Combine(outputBaseDir, baseName + suffix);
            }

            ImGui.Text("Output Root Path:");
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
                bool extendedMode = _wmoConvertExtended;
                bool copyTextures = _wmoConvertCopyTextures;
                var dataSource = _dataSource;

                Task.Run(() =>
                {
                    try
                    {
                        var origOut = Console.Out;
                        var sw = new StringWriter();
                        Console.SetOut(sw);

                        List<string> textures = new();
                        List<string> writtenFiles = new();
                        if (direction == 0)
                        {
                            if (extendedMode)
                            {
                                var converter = new WmoV14ToV17ExtendedConverter();
                                textures = converter.Convert(srcPath, outPath);
                            }
                            else
                            {
                                var converter = new WmoV14ToV17Converter();
                                textures = converter.Convert(srcPath, outPath);
                            }

                            writtenFiles.Add(outPath);
                            string outDir = Path.GetDirectoryName(Path.GetFullPath(outPath)) ?? ".";
                            string baseName = Path.GetFileNameWithoutExtension(outPath);
                            for (int gi = 0; gi < 2048; gi++)
                            {
                                string gp = Path.Combine(outDir, $"{baseName}_{gi:D3}.wmo");
                                if (!File.Exists(gp)) break;
                                writtenFiles.Add(gp);
                            }
                        }
                        else
                        {
                            var converter = new WmoV17ToV14Converter();
                            converter.Convert(srcPath, outPath);
                            writtenFiles.Add(outPath);
                        }

                        Console.SetOut(origOut);
                        var lines = sw.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
                        lock (_wmoConvertLog) _wmoConvertLog.AddRange(lines);
                        lock (_wmoConvertLog)
                        {
                            _wmoConvertLog.Add("\n=== SUCCESS ===");
                            foreach (var f in writtenFiles)
                                _wmoConvertLog.Add($"Wrote: {f}");
                        }

                        if (copyTextures && textures.Count > 0 && direction == 0)
                        {
                            CopyWmoTexturesPreservePaths(srcPath, outPath, textures, dataSource);
                            lock (_wmoConvertLog) _wmoConvertLog.Add($"Copied textures: {textures.Count}");
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

        if (ImGui.Begin("Generate VLM Dataset", ref _showVlmExportDialog))
        {
            ImGui.TextWrapped("Export terrain data from a WoW client folder into a VLM dataset (JSON + PNG). " +
                "Supports Alpha 0.5.3 through Cataclysm 4.0.1.");
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
            ImGui.InputText("##vlmOutput", ref _vlmOutputDir, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##output"))
            {
                string? result = ShowFolderDialogSTA("Select Output Directory");
                if (result != null) _vlmOutputDir = result;
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
            if (ImGui.Button("Export Dataset", new Vector2(140, 30)))
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

            // Progress log
            ImGui.Spacing();
            ImGui.Text("Log:");
            float logHeight = ImGui.GetContentRegionAvail().Y - 4;
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
                lock (_vlmExportLog)
                {
                    _vlmExportLog.Add($"=== Export complete: {result.TilesExported} tiles, {result.TilesSkipped} skipped, {result.UniqueTextures} textures ===");
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

    private void DrawToolbar()
    {
        var io = ImGui.GetIO();
        // Full-width toolbar (no gaps above sidebars)
        ImGui.SetNextWindowPos(new Vector2(0, MenuBarHeight));
        ImGui.SetNextWindowSize(new Vector2(io.DisplaySize.X, ToolbarHeight));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(8, 6));
        ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(6, 0));
        if (ImGui.Begin("##Toolbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings))
        {
            TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

            if (renderer != null)
            {
                // Terrain layers
                bool l0 = renderer.ShowLayer0;
                if (ImGui.Checkbox("Base", ref l0)) renderer.ShowLayer0 = l0;
                ImGui.SameLine();
                bool l1 = renderer.ShowLayer1;
                if (ImGui.Checkbox("L1", ref l1)) renderer.ShowLayer1 = l1;
                ImGui.SameLine();
                bool l2 = renderer.ShowLayer2;
                if (ImGui.Checkbox("L2", ref l2)) renderer.ShowLayer2 = l2;
                ImGui.SameLine();
                bool l3 = renderer.ShowLayer3;
                if (ImGui.Checkbox("L3", ref l3)) renderer.ShowLayer3 = l3;

                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                ImGui.SameLine();

                // Grid overlays
                bool chunkGrid = renderer.ShowChunkGrid;
                if (ImGui.Checkbox("Chunks", ref chunkGrid)) renderer.ShowChunkGrid = chunkGrid;
                ImGui.SameLine();
                bool tileGrid = renderer.ShowTileGrid;
                if (ImGui.Checkbox("Tiles", ref tileGrid)) renderer.ShowTileGrid = tileGrid;

                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                ImGui.SameLine();

                // Debug overlays
                bool alphaMask = renderer.ShowAlphaMask;
                if (ImGui.Checkbox("Alpha", ref alphaMask)) renderer.ShowAlphaMask = alphaMask;

                if (renderer.ShowAlphaMask)
                {
                    ImGui.SameLine();
                    int chan = renderer.AlphaMaskChannel;
                    chan = Math.Clamp(chan, 0, 3);
                    ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(4, 0));
                    ImGui.RadioButton("A1", ref chan, 1);
                    ImGui.SameLine();
                    ImGui.RadioButton("A2", ref chan, 2);
                    ImGui.SameLine();
                    ImGui.RadioButton("A3", ref chan, 3);
                    ImGui.PopStyleVar();
                    renderer.AlphaMaskChannel = chan;
                }

                ImGui.SameLine();
                bool shadowMap = renderer.ShowShadowMap;
                if (ImGui.Checkbox("Shadows", ref shadowMap)) renderer.ShowShadowMap = shadowMap;
                ImGui.SameLine();
                bool contours = renderer.ShowContours;
                if (ImGui.Checkbox("Contours", ref contours)) renderer.ShowContours = contours;

                // Liquid
                if (liquidRenderer != null)
                {
                    ImGui.SameLine();
                    bool showLiquid = liquidRenderer.ShowLiquid;
                    if (ImGui.Checkbox($"Liquid Terrain ({liquidRenderer.MeshCount})", ref showLiquid))
                        liquidRenderer.ShowLiquid = showLiquid;
                }

                if (_worldScene != null)
                {
                    ImGui.SameLine();
                    int wlCount = liquidRenderer?.WlMeshCount ?? 0;
                    bool showWlTop = _worldScene.ShowWlLiquids;
                    if (ImGui.Checkbox($"WL* ({wlCount})", ref showWlTop))
                        _worldScene.ShowWlLiquids = showWlTop;
                }

                if (_worldScene != null)
                {
                    ImGui.SameLine();
                    bool showWdl = _worldScene.ShowWdlTerrain;
                    if (ImGui.Checkbox("WDL", ref showWdl))
                        _worldScene.ShowWdlTerrain = showWdl;
                }

                // World objects
                if (_worldScene != null)
                {
                    ImGui.SameLine();
                    ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                    ImGui.SameLine();
                    bool showBB = _worldScene.ShowBoundingBoxes;
                    if (ImGui.Checkbox("BBs", ref showBB))
                        _worldScene.ShowBoundingBoxes = showBB;

                    ImGui.SameLine();
                    bool showGroundFx = _worldScene.ShowGroundEffects;
                    int gfxCount = _worldScene.GroundEffectInstanceCount;
                    if (ImGui.Checkbox($"GroundFX ({gfxCount})", ref showGroundFx))
                        _worldScene.ShowGroundEffects = showGroundFx;
                }
            }
        }
        ImGui.End();
        ImGui.PopStyleVar(2);
    }

    private void DrawLeftSidebar()
    {
        var io = ImGui.GetIO();
        float topOffset = (_terrainManager != null || _vlmTerrainManager != null) ? MenuBarHeight + ToolbarHeight : MenuBarHeight;
        float sidebarHeight = io.DisplaySize.Y - topOffset - StatusBarHeight;
        ImGui.SetNextWindowPos(new Vector2(0, topOffset));
        ImGui.SetNextWindowSize(new Vector2(SidebarWidth, sidebarHeight));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        if (ImGui.Begin("##LeftSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
        {
            // ── File Browser section ──
            if (_showFileBrowser && ImGui.CollapsingHeader("File Browser", ImGuiTreeNodeFlags.DefaultOpen))
            {
                DrawFileBrowserContent();
            }

            // ── World Maps section ──
            if (_discoveredMaps.Count > 0 && ImGui.CollapsingHeader("World Maps", ImGuiTreeNodeFlags.DefaultOpen))
            {
                DrawMapDiscoveryContent();
            }
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawMapDiscoveryContent()
    {
        if (_discoveredMaps.Count == 0) return;

        ImGui.Text($"{_discoveredMaps.Count} maps discovered");
        ImGui.Separator();

        // Map list — use remaining height or fixed height
        float listHeight = 300f; // Fixed height for map list
        if (ImGui.BeginChild("MapList", new Vector2(0, listHeight), true))
        {
            foreach (var map in _discoveredMaps)
            {
                bool hasWdt = map.HasWdt;
                bool hasWdl = map.HasWdl;
                string label = $"[{map.Id:D3}] {map.Name}";
                if (!hasWdt) ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(0.5f, 0.5f, 0.5f, 1f));

                if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick))
                {
                    if (hasWdt && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    {
                        string wdtPath = $"World\\Maps\\{map.Directory}\\{map.Directory}.wdt";
                        LoadFileFromDataSource(wdtPath);
                    }
                }

                if (!hasWdt) ImGui.PopStyleColor();

                // Show WDL preview button if map has WDL
                if (hasWdl)
                {
                    ImGui.SameLine();
                    if (ImGui.SmallButton($"Preview##{map.Id}"))
                    {
                        _selectedMapForPreview = map;
                        _showWdlPreview = true;
                        _selectedSpawnTile = null;
                        
                        // Load WDL preview
                        if (_wdlPreviewRenderer == null)
                            _wdlPreviewRenderer = new WdlPreviewRenderer(_gl);
                        
                        ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview] Attempting to load WDL for {map.Directory}");
                        bool loaded = _wdlPreviewRenderer.LoadWdl(_dataSource!, map.Directory);
                        ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview] Load result: {loaded}, HasPreview: {_wdlPreviewRenderer.HasPreview}");
                    }
                }

                if (ImGui.IsItemHovered())
                {
                    ImGui.BeginTooltip();
                    ImGui.Text($"Directory: {map.Directory}");
                    ImGui.Text($"WDT: {(hasWdt ? "Found" : "Missing")}");
                    ImGui.Text($"WDL: {(hasWdl ? "Found" : "Missing")}");
                    if (hasWdl)
                        ImGui.TextColored(new Vector4(0f, 1f, 0f, 1f), "Click 'Preview' to select spawn point");
                    ImGui.EndTooltip();
                }
            }
            ImGui.EndChild();
        }
    }

    private void DrawFileBrowserContent()
    {
        if (_dataSource == null || !_dataSource.IsLoaded)
        {
            ImGui.TextWrapped("No data source loaded.\nUse File > Open Game Folder to load MPQ archives.");
            return;
        }

        ImGui.Text($"Source: {_dataSource.Name}");
        ImGui.Separator();

        // Extension filter
        if (ImGui.BeginCombo("Type", _extensionFilter))
        {
            string[] filters = { ".mdx", ".wmo", ".m2", ".blp", ".wdt" };
            foreach (var f in filters)
            {
                if (ImGui.Selectable(f, _extensionFilter == f))
                {
                    _extensionFilter = f;
                    RefreshFileList();
                }
            }
            ImGui.EndCombo();
        }

        // Search filter
        var search = _searchFilter;
        if (ImGui.InputText("Search", ref search, 256))
        {
            _searchFilter = search;
            RefreshFileList();
        }

        ImGui.Text($"{_filteredFiles.Count} files");
        ImGui.Separator();

        // File list — reserve space for World Maps section if present
        float remainingH = ImGui.GetContentRegionAvail().Y;
        if (_discoveredMaps.Count > 0)
            remainingH = MathF.Max(remainingH - 360f, 100f); // Reserve ~360px for World Maps header + list
        if (ImGui.BeginChild("FileList", new Vector2(0, remainingH), true))
        {
            for (int i = 0; i < _filteredFiles.Count; i++)
            {
                var file = _filteredFiles[i];
                var displayName = Path.GetFileName(file);
                bool selected = i == _selectedFileIndex;

                if (ImGui.Selectable(displayName, selected, ImGuiSelectableFlags.AllowDoubleClick))
                {
                    _selectedFileIndex = i;
                    if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    {
                        LoadFileFromDataSource(file);
                    }
                }

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(file);
            }
            ImGui.EndChild();
        }
    }

    private void DrawRightSidebar()
    {
        var io = ImGui.GetIO();
        float topOffset = (_terrainManager != null || _vlmTerrainManager != null) ? MenuBarHeight + ToolbarHeight : MenuBarHeight;
        float sidebarHeight = io.DisplaySize.Y - topOffset - StatusBarHeight;
        ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - SidebarWidth, topOffset));
        ImGui.SetNextWindowSize(new Vector2(SidebarWidth, sidebarHeight));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        if (ImGui.Begin("##RightSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
        {
            // ── Selected Object section (always visible when something is selected) ──
            if (!string.IsNullOrEmpty(_selectedObjectInfo))
            {
                ImGui.TextColored(new Vector4(1f, 1f, 0f, 1f), "Selected Object");
                ImGui.Separator();
                ImGui.TextWrapped(_selectedObjectInfo);
                DrawSelectedSqlGameObjectAnimationControls();
                ImGui.Spacing();
            }

            // ── Model Info section ──
            if (_showModelInfo && ImGui.CollapsingHeader("Model Info", ImGuiTreeNodeFlags.DefaultOpen))
            {
                DrawModelInfoContent();
            }

            // ── Camera section ──
            if (ImGui.CollapsingHeader("Camera", ImGuiTreeNodeFlags.DefaultOpen))
            {
                ImGui.SliderFloat("Camera Speed", ref _cameraSpeed, 1f, 500f, "%.0f");
                ImGui.Text("Hold Shift for 5x boost");
                ImGui.SliderFloat("FOV", ref _fovDegrees, 20f, 90f, "%.0f°");
            }

            // ── Terrain Controls section ──
            if (_showTerrainControls && (_terrainManager != null || _vlmTerrainManager != null))
            {
                if (ImGui.CollapsingHeader("Terrain Controls", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    DrawTerrainControlsContent();
                }
            }

            // ── World Objects section ──
            if (_worldScene != null)
            {
                if (ImGui.CollapsingHeader("World Objects", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    DrawWorldObjectsContent();
                }
            }
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawModelInfoContent()
    {
        if (string.IsNullOrEmpty(_modelInfo))
        {
            ImGui.TextWrapped("No model loaded.");
            return;
        }

        ImGui.TextWrapped(_modelInfo);

        if (_renderer is MdxRenderer || _renderer is WmoRenderer)
        {
            ImGui.Separator();
            ImGui.Checkbox("Auto-frame on load", ref _autoFrameModelOnLoad);
            if (ImGui.Button("Frame Model"))
                FrameCurrentModel();
        }

        // DoodadSet selection (WMO only)
        if (_renderer is WmoRenderer wmoR && wmoR.DoodadSetCount > 0)
        {
            ImGui.Separator();
            ImGui.Text("Doodad Set:");
            int activeSet = wmoR.ActiveDoodadSet;
            string currentSetName = wmoR.GetDoodadSetName(activeSet);
            if (ImGui.BeginCombo("##DoodadSet", currentSetName))
            {
                for (int s = 0; s < wmoR.DoodadSetCount; s++)
                {
                    bool selected = s == activeSet;
                    if (ImGui.Selectable(wmoR.GetDoodadSetName(s), selected))
                        wmoR.SetActiveDoodadSet(s);
                    if (selected) ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
        }

        if (_renderer is WmoRenderer)
        {
            ImGui.Separator();
            DrawWmoLiquidRotationControls("standalone");
        }

        // Animation sequence selection (MDX only)
        if (_renderer is MdxRenderer mdxR && mdxR.Animator != null && mdxR.Animator.Sequences.Count > 0)
        {
            ImGui.Separator();
            ImGui.Text("Animation:");
            
            var animator = mdxR.Animator;
            int currentSeq = animator.CurrentSequence;
            string currentSeqName = currentSeq >= 0 && currentSeq < animator.Sequences.Count 
                ? animator.Sequences[currentSeq].Name 
                : "None";
            
            if (ImGui.BeginCombo("##AnimSequence", currentSeqName))
            {
                for (int s = 0; s < animator.Sequences.Count; s++)
                {
                    bool selected = s == currentSeq;
                    string seqName = animator.Sequences[s].Name;
                    if (string.IsNullOrEmpty(seqName))
                        seqName = $"Sequence {s}";
                    
                    if (ImGui.Selectable(seqName, selected))
                        animator.SetSequence(s);
                    if (selected) ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
            
            // Timeline controls
            if (currentSeq >= 0 && currentSeq < animator.Sequences.Count)
            {
                var seq = animator.Sequences[currentSeq];
                float seqStart = seq.Time.Start;
                float seqEnd = seq.Time.End;
                float duration = seqEnd - seqStart;
                float currentAbs = animator.CurrentFrame;
                float currentRel = currentAbs - seqStart;
                
                // Play/Pause button
                bool isPlaying = animator.IsPlaying;
                if (ImGui.Button(isPlaying ? "⏸ Pause" : "▶ Play"))
                    animator.IsPlaying = !isPlaying;
                
                ImGui.SameLine();
                
                // Step backward button
                if (ImGui.Button("◀"))
                {
                    animator.IsPlaying = false;
                    animator.StepToPrevKeyframe();
                }
                
                ImGui.SameLine();
                
                // Step forward button
                if (ImGui.Button("▶"))
                {
                    animator.IsPlaying = false;
                    animator.StepToNextKeyframe();
                }
                
                // Frame slider
                ImGui.SetNextItemWidth(-1);
                if (ImGui.SliderFloat("##Timeline", ref currentRel, 0, duration, $"Frame: {currentAbs:F0} / {seqEnd:F0}"))
                {
                    animator.IsPlaying = false;
                    animator.CurrentFrame = seqStart + currentRel;
                }
                
                // Duration info
                ImGui.Text($"Duration: {duration:F0}ms ({duration / 1000.0f:F2}s)");

                if (ImGui.TreeNode("Animation Debug"))
                {
                    ImGui.Text($"Current Seq: {currentSeq}");
                    ImGui.Text($"Current Abs Frame: {currentAbs:F2}");
                    ImGui.Text($"Seq Range: [{seqStart}, {seqEnd}]");

                    var stats = animator.GetTrackDebugStatsForCurrentSequence();
                    ImGui.Text($"T keys total/in-range: {stats.TranslationKeysTotal}/{stats.TranslationKeysInSequence}");
                    ImGui.Text($"R keys total/in-range: {stats.RotationKeysTotal}/{stats.RotationKeysInSequence}");
                    ImGui.Text($"S keys total/in-range: {stats.ScalingKeysTotal}/{stats.ScalingKeysInSequence}");

                    string minKey = stats.MinKeyTime?.ToString() ?? "n/a";
                    string maxKey = stats.MaxKeyTime?.ToString() ?? "n/a";
                    ImGui.Text($"All key range: [{minKey}, {maxKey}]");

                    ImGui.Separator();
                    ImGui.Text("Sequences (first 12):");
                    int previewCount = Math.Min(12, animator.Sequences.Count);
                    for (int i = 0; i < previewCount; i++)
                    {
                        var s = animator.Sequences[i];
                        string name = string.IsNullOrWhiteSpace(s.Name) ? "<empty>" : s.Name;
                        ImGui.Text($"{i}: {name} [{s.Time.Start}-{s.Time.End}]");
                    }

                    ImGui.TreePop();
                }
            }
        }

        // Geoset / Group visibility toggles
        if (_renderer != null && _renderer.SubObjectCount > 0)
        {
            ImGui.Separator();
            ImGui.Text("Visibility:");

            if (ImGui.SmallButton("All On"))
                for (int i = 0; i < _renderer.SubObjectCount; i++)
                    _renderer.SetSubObjectVisible(i, true);
            ImGui.SameLine();
            if (ImGui.SmallButton("All Off"))
                for (int i = 0; i < _renderer.SubObjectCount; i++)
                    _renderer.SetSubObjectVisible(i, false);

            for (int i = 0; i < _renderer.SubObjectCount; i++)
            {
                bool vis = _renderer.GetSubObjectVisible(i);
                if (ImGui.Checkbox(_renderer.GetSubObjectName(i), ref vis))
                    _renderer.SetSubObjectVisible(i, vis);
            }
        }
    }

    private void FrameCurrentModel()
    {
        if (_renderer is MdxRenderer mdxR)
        {
            var bmin = mdxR.BoundsMin;
            var bmax = mdxR.BoundsMax;
            FrameBounds(bmin, bmax, mdxMirrorX: true);
        }
        else if (_renderer is WmoRenderer wmoR)
        {
            FrameBounds(wmoR.BoundsMin, wmoR.BoundsMax, mdxMirrorX: false);
        }
    }

    private void FrameBounds(Vector3 boundsMin, Vector3 boundsMax, bool mdxMirrorX)
    {
        var center = (boundsMin + boundsMax) * 0.5f;
        var extent = boundsMax - boundsMin;
        float radius = MathF.Max(extent.Length() * 0.5f, 1f);

        // MDX standalone rendering applies a MirrorX scale at draw time. Keep the previous convention.
        if (mdxMirrorX)
            center.X = -center.X;

        float dist = MathF.Max(radius * 3.0f, 10f);
        _camera.Position = center + new Vector3(-dist, 0, radius * 0.6f);
        _camera.Yaw = 0f;
        _camera.Pitch = -15f;
    }

    private void DrawTerrainControlsContent()
    {
        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (lighting == null || renderer == null) return;

        // Day/night cycle
        float gameTime = lighting.GameTime;
        if (ImGui.SliderFloat("Time of Day", ref gameTime, 0f, 1f, "%.2f"))
            lighting.GameTime = gameTime;
        string timeLabel = gameTime switch
        {
            < 0.15f => "Night",
            < 0.25f => "Dawn",
            < 0.35f => "Morning",
            < 0.65f => "Day",
            < 0.75f => "Evening",
            < 0.85f => "Dusk",
            _ => "Night"
        };
        ImGui.SameLine();
        ImGui.Text(timeLabel);

        // Fog
        float fogStart = lighting.FogStart;
        float fogEnd = lighting.FogEnd;
        if (ImGui.SliderFloat("Fog Start", ref fogStart, 0f, 2000f))
            lighting.FogStart = fogStart;
        if (ImGui.SliderFloat("Fog End", ref fogEnd, 100f, 5000f))
            lighting.FogEnd = fogEnd;

        if (_worldScene != null)
        {
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("Show WDL Far Terrain", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Toggle low-detail WDL background terrain for testing terrain overlap issues.");
        }

        ImGui.Separator();
        bool crispAlpha = renderer.UseNearestForAlphaSampling;
        if (ImGui.Checkbox("Crisp Alpha Masks", ref crispAlpha))
            renderer.UseNearestForAlphaSampling = crispAlpha;
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Switch alpha/shadow sampling between Linear (default) and Nearest (crisper edges).");

        // Contour interval (only when contours enabled via toolbar)
        if (renderer.ShowContours)
        {
            ImGui.Separator();
            float interval = renderer.ContourInterval;
            if (ImGui.SliderFloat("Contour Interval", ref interval, 0.5f, 20.0f, "%.1f"))
                renderer.ContourInterval = interval;
        }

        ImGui.Separator();
        if (ImGui.Button("Toggle Wireframe"))
            _renderer?.ToggleWireframe();

        // Stats
        ImGui.Separator();
        int tiles = _terrainManager?.LoadedTileCount ?? _vlmTerrainManager?.LoadedTileCount ?? 0;
        int chunks = _terrainManager?.LoadedChunkCount ?? _vlmTerrainManager?.LoadedChunkCount ?? 0;
        var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer != null)
            ImGui.Text($"Tiles: {tiles}  Chunks: {terrainRenderer.ChunksRendered}/{chunks}");
        else
            ImGui.Text($"Tiles: {tiles}  Chunks: {chunks}");

        // Culling stats from WorldScene
        if (_worldScene != null)
        {
            ImGui.Text($"WMO: {_worldScene.WmoRenderedCount}/{_worldScene.WmoInstanceCount}  MDX: {_worldScene.MdxRenderedCount}/{_worldScene.MdxInstanceCount}");
        }
    }

    private void DrawWorldObjectsContent()
    {
        if (_worldScene == null) return;

        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

        ImGui.Separator();
        ImGui.Text("SQL World Population");
        ImGui.InputTextWithHint("##sqlroot", "Path to alpha-core root (example: external/alpha-core)", ref _sqlAlphaCoreRoot, 1024);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("MdxViewer reads NPC/GameObject spawns from alpha-core SQL dumps (etc/databases/world + dbc).");

        ImGui.SameLine();
        if (ImGui.Button("Use Submodule Path"))
        {
            string candidate = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "external", "alpha-core"));
            _sqlAlphaCoreRoot = candidate;
        }
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

        bool canLoadSql = _currentMapId >= 0 && !string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot);
        if (!canLoadSql)
            ImGui.BeginDisabled();
        if (ImGui.Button("Load SQL Spawns (Current Map)"))
            LoadSqlSpawnsForCurrentMap();
        if (!canLoadSql)
            ImGui.EndDisabled();

        ImGui.SameLine();
        if (ImGui.Button("Clear SQL Spawns"))
        {
            ResetSqlSpawnStreamingState(clearSceneSpawns: true);
            _sqlSpawnStatus = "Cleared SQL spawns.";
        }

        if (sqlSettingsChanged && _sqlMapSpawnsCache != null)
        {
            _sqlForceStreamRefresh = true;
            if (!_sqlStreamWithCamera || !_sqlUseAoiFilter)
                ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: true);
        }

        ImGui.TextDisabled($"Status: {_sqlSpawnStatus}");
        ImGui.TextDisabled($"Injected: {_worldScene.ExternalSpawnInstanceCount} total ({_worldScene.ExternalSpawnMdxCount} MDX, {_worldScene.ExternalSpawnWmoCount} WMO)");
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
            if (ImGui.Button("Load Area POIs"))
                _worldScene.ShowPoi = true; // triggers lazy load
        }
        else if (_worldScene.PoiLoadAttempted && (_worldScene.PoiLoader == null || _worldScene.PoiLoader.Entries.Count == 0))
        {
            ImGui.TextDisabled("Area POIs: none found");
        }

        // Taxi paths toggle — lazy-loaded on first request
        if (_worldScene.TaxiLoader != null && _worldScene.TaxiLoader.Routes.Count > 0)
        {
            bool showTaxi = _worldScene.ShowTaxi;
            if (ImGui.Checkbox($"Taxi Paths ({_worldScene.TaxiLoader.Routes.Count})", ref showTaxi))
                _worldScene.ShowTaxi = showTaxi;
            if (_worldScene.ShowTaxi && (_worldScene.SelectedTaxiNodeId >= 0 || _worldScene.SelectedTaxiRouteId >= 0))
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Show All"))
                    _worldScene.ClearTaxiSelection();
            }
        }
        else if (!_worldScene.TaxiLoadAttempted)
        {
            if (ImGui.Button("Load Taxi Paths"))
                _worldScene.ShowTaxi = true; // triggers lazy load
        }
        else if (_worldScene.TaxiLoadAttempted && (_worldScene.TaxiLoader == null || _worldScene.TaxiLoader.Routes.Count == 0))
        {
            ImGui.TextDisabled("Taxi Paths: none found");
        }

        // WL loose liquid files (WLW/WLQ/WLM) — lazy-loaded on first toggle
        if (_worldScene.WlLoader != null && _worldScene.WlLoader.HasData)
        {
            bool showWl = _worldScene.ShowWlLiquids;
            if (ImGui.Checkbox($"WL Liquids ({_worldScene.WlLoader.Bodies.Count})", ref showWl))
                _worldScene.ShowWlLiquids = showWl;
            if (_worldScene.ShowWlLiquids && ImGui.IsItemHovered())
                ImGui.SetTooltip("Loose WLW/WLQ/WLM liquid project files.\nContains water data for deleted/missing tiles.");

            if (liquidRenderer != null && ImGui.TreeNode("WL Layers"))
            {
                int visibleCount = 0;
                foreach (var b in _worldScene.WlLoader.Bodies)
                {
                    if (liquidRenderer.IsWlBodyVisible(b.SourcePath))
                        visibleCount++;
                }

                if (ImGui.SmallButton("Show All"))
                    liquidRenderer.SetAllWlBodiesVisible(true);
                ImGui.SameLine();
                if (ImGui.SmallButton("Hide All"))
                    liquidRenderer.SetAllWlBodiesVisible(false);
                ImGui.SameLine();
                bool hasSelected = !string.IsNullOrWhiteSpace(_wlLayerSelectedSourcePath);
                if (!hasSelected)
                    ImGui.BeginDisabled();
                if (ImGui.SmallButton("Solo Selected"))
                {
                    liquidRenderer.SetAllWlBodiesVisible(false);
                    liquidRenderer.SetWlBodyVisible(_wlLayerSelectedSourcePath, true);
                }
                if (!hasSelected)
                    ImGui.EndDisabled();

                ImGui.TextDisabled($"Visible: {visibleCount}/{_worldScene.WlLoader.Bodies.Count}");

                if (ImGui.BeginTable("##wl_layers", 3, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp))
                {
                    ImGui.TableSetupColumn("V", ImGuiTableColumnFlags.WidthFixed, 24f);
                    ImGui.TableSetupColumn("Type", ImGuiTableColumnFlags.WidthFixed, 48f);
                    ImGui.TableSetupColumn("Layer", ImGuiTableColumnFlags.WidthStretch);
                    ImGui.TableHeadersRow();

                    for (int i = 0; i < _worldScene.WlLoader.Bodies.Count; i++)
                    {
                        var body = _worldScene.WlLoader.Bodies[i];
                        ImGui.TableNextRow();

                        ImGui.TableSetColumnIndex(0);
                        bool visible = liquidRenderer.IsWlBodyVisible(body.SourcePath);
                        if (ImGui.Checkbox($"##wl_vis_{i}", ref visible))
                            liquidRenderer.SetWlBodyVisible(body.SourcePath, visible);

                        ImGui.TableSetColumnIndex(1);
                        ImGui.TextUnformatted(body.FileType.ToString());

                        ImGui.TableSetColumnIndex(2);
                        bool isSelected = string.Equals(_wlLayerSelectedSourcePath, body.SourcePath, StringComparison.OrdinalIgnoreCase);
                        string label = $"{body.Name}##wl_layer_{i}";
                        if (ImGui.Selectable(label, isSelected, ImGuiSelectableFlags.SpanAllColumns))
                            _wlLayerSelectedSourcePath = body.SourcePath;
                        if (ImGui.IsItemHovered())
                        {
                            ImGui.BeginTooltip();
                            ImGui.TextUnformatted(body.SourcePath);
                            ImGui.Text($"Blocks: {body.BlockCount}  Verts: {body.Vertices.Length}");
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

                if (ImGui.Button("Apply + Reload WL"))
                    _worldScene.ReloadWlLiquids();

                ImGui.SameLine();
                if (ImGui.Button("Print Current WL Transform"))
                {
                    ViewerLog.Important(ViewerLog.Category.Terrain,
                        $"[WL Transform] Enabled={ts.Enabled} SwapXY={ts.SwapXYBeforeRotation} " +
                        $"Rot=({ts.RotationDegrees.X:F1},{ts.RotationDegrees.Y:F1},{ts.RotationDegrees.Z:F1}) " +
                        $"Trans=({ts.Translation.X:F1},{ts.Translation.Y:F1},{ts.Translation.Z:F1})");
                }

                ImGui.TextDisabled("Tune here, then share the printed values to hard-wire final config.");
                ImGui.TreePop();
            }
        }
        else if (!_worldScene.WlLoadAttempted)
        {
            if (ImGui.Button("Load WL Liquids"))
                _worldScene.ShowWlLiquids = true; // triggers lazy load
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Load loose WLW/WLQ/WLM liquid project files.\nContains water heightmaps including deleted bodies of water.");
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
                _worldScene.ShowAreaTriggers = true; // triggers lazy load
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Load AreaTrigger.dbc to visualize instance portals,\nevent markers, and script trigger zones.");
        }
        else if (_worldScene.AreaTriggerLoadAttempted && (_worldScene.AreaTriggerLoader == null || _worldScene.AreaTriggerLoader.Count == 0))
        {
            ImGui.TextDisabled("AreaTriggers: none found");
        }

        // WMO placements
        if (_worldScene.ModfPlacements.Count > 0 && ImGui.TreeNode($"WMO Placements ({_worldScene.ModfPlacements.Count})"))
        {
            for (int i = 0; i < _worldScene.ModfPlacements.Count; i++)
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
            ImGui.TreePop();
        }

        // MDX placements (show first 200 to avoid UI lag)
        int mddfCount = _worldScene.MddfPlacements.Count;
        int mddfShow = Math.Min(mddfCount, 200);
        if (mddfCount > 0 && ImGui.TreeNode($"MDX Placements ({mddfCount}{(mddfCount > mddfShow ? $", showing {mddfShow}" : "")})"))
        {
            for (int i = 0; i < mddfShow; i++)
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
            ImGui.TreePop();
        }

        // Area POI list
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0 &&
            ImGui.TreeNode($"Area POIs ({_worldScene.PoiLoader.Entries.Count})"))
        {
            foreach (var poi in _worldScene.PoiLoader.Entries)
            {
                string label = $"[{poi.Id}] {poi.Name}";
                if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick))
                {
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
            ImGui.TreePop();
        }

        // Taxi Nodes list — single-click to select/filter, double-click to teleport
        if (_worldScene.TaxiLoader != null && _worldScene.TaxiLoader.Nodes.Count > 0 &&
            ImGui.TreeNode($"Taxi Nodes ({_worldScene.TaxiLoader.Nodes.Count})"))
        {
            foreach (var node in _worldScene.TaxiLoader.Nodes)
            {
                bool isSelected = _worldScene.SelectedTaxiNodeId == node.Id;
                string label = $"[{node.Id}] {node.Name}";
                if (ImGui.Selectable(label, isSelected, ImGuiSelectableFlags.AllowDoubleClick))
                {
                    if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    {
                        _camera.Position = node.Position + new System.Numerics.Vector3(0, 0, 50);
                        _camera.Pitch = -30f;
                    }
                    else
                    {
                        // Toggle selection: click again to deselect
                        _worldScene.SelectedTaxiNodeId = isSelected ? -1 : node.Id;
                    }
                }
                if (ImGui.IsItemHovered())
                {
                    ImGui.BeginTooltip();
                    ImGui.Text($"Position: ({node.Position.X:F1}, {node.Position.Y:F1}, {node.Position.Z:F1})");
                    int routeCount = _worldScene.TaxiLoader.Routes.Count(r => r.FromNodeId == node.Id || r.ToNodeId == node.Id);
                    ImGui.Text($"Routes: {routeCount}");
                    ImGui.Text("Click to filter, double-click to teleport");
                    ImGui.EndTooltip();
                }
            }
            ImGui.TreePop();
        }

        // Taxi Routes list — single-click to select/filter, double-click to teleport
        if (_worldScene.TaxiLoader != null && _worldScene.TaxiLoader.Routes.Count > 0 &&
            ImGui.TreeNode($"Taxi Routes ({_worldScene.TaxiLoader.Routes.Count})"))
        {
            foreach (var route in _worldScene.TaxiLoader.Routes)
            {
                bool isSelected = _worldScene.SelectedTaxiRouteId == route.PathId;
                string fromName = _worldScene.TaxiLoader.Nodes.FirstOrDefault(n => n.Id == route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
                string toName = _worldScene.TaxiLoader.Nodes.FirstOrDefault(n => n.Id == route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
                string label = $"[{route.PathId}] {fromName} → {toName} ({route.Waypoints.Count} pts)";
                if (ImGui.Selectable(label, isSelected, ImGuiSelectableFlags.AllowDoubleClick))
                {
                    if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left) && route.Waypoints.Count > 0)
                    {
                        var mid = route.Waypoints[route.Waypoints.Count / 2];
                        _camera.Position = mid + new System.Numerics.Vector3(0, 0, 100);
                        _camera.Pitch = -30f;
                    }
                    else
                    {
                        // Toggle selection: click again to deselect
                        _worldScene.SelectedTaxiRouteId = isSelected ? -1 : route.PathId;
                    }
                }
                if (ImGui.IsItemHovered())
                {
                    ImGui.BeginTooltip();
                    ImGui.Text($"Cost: {route.Cost}");
                    ImGui.Text($"Waypoints: {route.Waypoints.Count}");
                    if (route.Waypoints.Count > 0)
                    {
                        var first = route.Waypoints[0];
                        var last = route.Waypoints[^1];
                        ImGui.Text($"Start: ({first.X:F0}, {first.Y:F0}, {first.Z:F0})");
                        ImGui.Text($"End: ({last.X:F0}, {last.Y:F0}, {last.Z:F0})");
                    }
                    ImGui.Text("Click to filter, double-click to teleport");
                    ImGui.EndTooltip();
                }
            }
            ImGui.TreePop();
        }
    }

    private static void DrawWmoLiquidRotationControls(string idSuffix)
    {
        int quarterTurns = WmoRenderer.MliqRotationQuarterTurns;
        string currentLabel = WmoLiquidRotationLabels[Math.Clamp(quarterTurns, 0, WmoLiquidRotationLabels.Length - 1)];

        if (ImGui.BeginCombo($"WMO MLIQ Rotation##{idSuffix}", currentLabel))
        {
            for (int i = 0; i < WmoLiquidRotationLabels.Length; i++)
            {
                bool selected = i == quarterTurns;
                if (ImGui.Selectable(WmoLiquidRotationLabels[i], selected))
                    WmoRenderer.MliqRotationQuarterTurns = i;
                if (selected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        ImGui.TextDisabled("Applies to all WMO MLIQ surfaces. Changes are live.");
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

    private void DrawStatusBar()
    {
        var io = ImGui.GetIO();
        var windowHeight = io.DisplaySize.Y;
        ImGui.SetNextWindowPos(new Vector2(0, windowHeight - 24));
        ImGui.SetNextWindowSize(new Vector2(io.DisplaySize.X, 24));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(8, 4));
        if (ImGui.Begin("##statusbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings))
        {
            ImGui.Text(_statusMessage);
            if (!string.IsNullOrEmpty(_currentAreaName))
            {
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(1f, 0.9f, 0.5f, 1f), $"  Area: {_currentAreaName}");
            }

            // Camera coordinates (local + WoW world) in the center-ish area
            if (_terrainManager != null || _vlmTerrainManager != null)
            {
                var pos = _camera.Position;
                float wowX = WoWConstants.MapOrigin - pos.Y;
                float wowY = WoWConstants.MapOrigin - pos.X;
                float wowZ = pos.Z;
                string coordText = $"Local: ({pos.X:F0}, {pos.Y:F0}, {pos.Z:F0})  WoW: ({wowX:F0}, {wowY:F0}, {wowZ:F0})";
                float coordWidth = ImGui.CalcTextSize(coordText).X;
                float centerX = (io.DisplaySize.X - coordWidth) * 0.5f;
                ImGui.SameLine(centerX);
                ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), coordText);
            }

            // FPS counter on the right side
            string fpsText = $"{_currentFps:F0} FPS  {_frameTimeMs:F1} ms";
            float textWidth = ImGui.CalcTextSize(fpsText).X;
            ImGui.SameLine(io.DisplaySize.X - textWidth - 16);
            var fpsColor = _currentFps >= 30 ? new Vector4(0.4f, 1f, 0.4f, 1f)
                         : _currentFps >= 15 ? new Vector4(1f, 1f, 0.4f, 1f)
                         : new Vector4(1f, 0.4f, 0.4f, 1f);
            ImGui.TextColored(fpsColor, fpsText);
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawMinimapWindow()
    {
        // Gather tile data
        List<(int tx, int ty)>? existingTiles = null;
        Func<int, int, bool>? isTileLoaded = null;
        string? mapName = null;

        if (_terrainManager != null)
        {
            var adapter = _terrainManager.Adapter;
            existingTiles = adapter.ExistingTiles.Select(idx => (idx / 64, idx % 64)).ToList();
            isTileLoaded = _terrainManager.IsTileLoaded;
            mapName = _terrainManager.MapName;
        }
        else if (_vlmTerrainManager != null)
        {
            existingTiles = _vlmTerrainManager.Loader.TileCoords.ToList();
            isTileLoaded = _vlmTerrainManager.IsTileLoaded;
            mapName = _vlmTerrainManager.MapName;
        }
        else return;

        var io = ImGui.GetIO();
        
        // Position in top-right, but accounting for right sidebar if visible
        float rightOffset = _showRightSidebar ? SidebarWidth + 20 : 20;
        ImGui.SetNextWindowSize(new Vector2(360, 360), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowSizeConstraints(new Vector2(300, 300), new Vector2(520, 520));
        ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - 360 - rightOffset, MenuBarHeight + ToolbarHeight + 20), ImGuiCond.FirstUseEver);

        if (!ImGui.Begin("Minimap", ref _showMinimapWindow,
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse))
        {
            ImGui.End();
            return;
        }

        // Compact controls: tile readout + zoom in/out (+ wheel zoom while hovered).
        float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize;
        float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize;
        int ctX = (int)MathF.Floor(camTileX);
        int ctY = (int)MathF.Floor(camTileY);

        ImGui.Text($"Tile: ({ctX},{ctY})");
        ImGui.SameLine();
        if (ImGui.SmallButton("-##minimapZoomOut"))
            _minimapZoom = Math.Clamp(_minimapZoom + 0.5f, 1f, 32f);
        ImGui.SameLine();
        if (ImGui.SmallButton("+##minimapZoomIn"))
            _minimapZoom = Math.Clamp(_minimapZoom - 0.5f, 1f, 32f);
        ImGui.SameLine();
        ImGui.TextDisabled($"Zoom {_minimapZoom:F1}x");

        float controlsHeight = ImGui.GetCursorPosY() + 8f;
        float mapAvailableWidth = ImGui.GetContentRegionAvail().X;
        float mapAvailableHeight = ImGui.GetContentRegionAvail().Y - 4f;
        float mapSize = MathF.Max(64f, MathF.Min(mapAvailableWidth, mapAvailableHeight));
        
        var cursorPos = ImGui.GetCursorScreenPos();

        // Scroll-wheel zoom (map region only)
        if (ImGui.IsWindowHovered() && ImGui.IsMouseHoveringRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize)))
        {
            float wheel = io.MouseWheel;
            if (wheel != 0)
                _minimapZoom = Math.Clamp(_minimapZoom - wheel * 0.5f, 1f, 32f);
        }

        MinimapHelpers.RenderMinimapContent(
            cursorPos, mapSize, existingTiles, isTileLoaded, _minimapRenderer, mapName,
            camTileX, camTileY, _minimapZoom, _minimapPanOffset, _camera, _worldScene,
            out float viewMinTx, out float viewMinTy, out float cellSize);

        // Make minimap area interactive with invisible button overlay
        ImGui.SetCursorScreenPos(cursorPos);
        ImGui.InvisibleButton("##minimapInteraction", new Vector2(mapSize, mapSize));
        bool isHovered = ImGui.IsItemHovered();
        bool isActive = ImGui.IsItemActive();

        // Handle click-and-drag panning or click-to-teleport
        var mousePos = ImGui.GetMousePos();

        if (isHovered || isActive)
        {
            // Start drag on mouse down
            if (ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            {
                _minimapDragging = true;
                _minimapDragStart = mousePos;
            }
            // Continue drag while mouse is down
            else if (ImGui.IsMouseDown(ImGuiMouseButton.Left) && _minimapDragging)
            {
                Vector2 delta = mousePos - _minimapDragStart;
                if (delta.LengthSquared() > 0.01f) // Any movement counts as drag
                {
                    _minimapPanOffset -= new Vector2(delta.Y / cellSize, delta.X / cellSize);
                    _minimapDragStart = mousePos;
                }
            }
            // Mouse released - check if it was a click or drag
            else if (ImGui.IsMouseReleased(ImGuiMouseButton.Left) && _minimapDragging)
            {
                Vector2 delta = mousePos - _minimapDragStart;
                if (delta.Length() <= 3f) // Was a click, not a drag
                {
                    // Teleport on single click
                    float clickTileY = (mousePos.X - cursorPos.X) / cellSize + viewMinTy;
                    float clickTileX = (mousePos.Y - cursorPos.Y) / cellSize + viewMinTx;
                    if (clickTileX >= 0 && clickTileX < 64 && clickTileY >= 0 && clickTileY < 64)
                    {
                        float worldX = WoWConstants.MapOrigin - clickTileX * WoWConstants.ChunkSize;
                        float worldY = WoWConstants.MapOrigin - clickTileY * WoWConstants.ChunkSize;
                        _camera.Position = new System.Numerics.Vector3(worldX, worldY, _camera.Position.Z);
                    }
                }
                _minimapDragging = false;
            }
        }
        else if (_minimapDragging)
        {
            _minimapDragging = false;
        }

        // Keep cursor aligned under map to avoid adding overflow content that creates scrollbars.
        ImGui.SetCursorPosY(controlsHeight + mapSize + 2f);

        ImGui.End();
    }

    private void DrawFullscreenMinimap()
    {
        // Gather tile data
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
        float mapSize = MathF.Min(io.DisplaySize.X * 0.8f, io.DisplaySize.Y * 0.8f);
        float padding = (io.DisplaySize.X - mapSize) * 0.5f;
        float topPadding = (io.DisplaySize.Y - mapSize) * 0.5f;

        ImGui.SetNextWindowPos(Vector2.Zero);
        ImGui.SetNextWindowSize(io.DisplaySize);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0, 0, 0, 0.85f));
        
        if (ImGui.Begin("##FullscreenMinimap", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings |
            ImGuiWindowFlags.NoScrollbar))
        {
            ImGui.SetCursorPos(new Vector2(padding, topPadding));
            var cursorPos = ImGui.GetCursorScreenPos();

            // Scroll-wheel zoom
            if (ImGui.IsWindowHovered())
            {
                float wheel = io.MouseWheel;
                if (wheel != 0)
                    _minimapZoom = Math.Clamp(_minimapZoom - wheel * 0.5f, 1f, 32f);
            }

            float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize;
            float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize;

            MinimapHelpers.RenderMinimapContent(
                cursorPos, mapSize, existingTiles, isTileLoaded, _minimapRenderer, mapName,
                camTileX, camTileY, _minimapZoom, _minimapPanOffset, _camera, _worldScene,
                out float viewMinTx, out float viewMinTy, out float cellSize);

            // Make minimap area interactive with invisible button overlay
            ImGui.SetCursorPos(new Vector2(padding, topPadding));
            ImGui.InvisibleButton("##fullscreenMinimapInteraction", new Vector2(mapSize, mapSize));
            bool isHovered = ImGui.IsItemHovered();
            bool isActive = ImGui.IsItemActive();

            // Handle click-and-drag panning or click-to-teleport
            var mousePos = ImGui.GetMousePos();

            if (isHovered || isActive)
            {
                // Start drag on mouse down
                if (ImGui.IsMouseClicked(ImGuiMouseButton.Left))
                {
                    _minimapDragging = true;
                    _minimapDragStart = mousePos;
                }
                // Continue drag while mouse is down
                else if (ImGui.IsMouseDown(ImGuiMouseButton.Left) && _minimapDragging)
                {
                    Vector2 delta = mousePos - _minimapDragStart;
                    if (delta.LengthSquared() > 0.01f) // Any movement counts as drag
                    {
                        _minimapPanOffset -= new Vector2(delta.Y / cellSize, delta.X / cellSize);
                        _minimapDragStart = mousePos;
                    }
                }
                // Mouse released - check if it was a click or drag
                else if (ImGui.IsMouseReleased(ImGuiMouseButton.Left) && _minimapDragging)
                {
                    Vector2 delta = mousePos - _minimapDragStart;
                    if (delta.Length() <= 3f) // Was a click, not a drag
                    {
                        // Teleport on single click
                        float clickTileY = (mousePos.X - cursorPos.X) / cellSize + viewMinTy;
                        float clickTileX = (mousePos.Y - cursorPos.Y) / cellSize + viewMinTx;
                        if (clickTileX >= 0 && clickTileX < 64 && clickTileY >= 0 && clickTileY < 64)
                        {
                            float worldX = WoWConstants.MapOrigin - clickTileX * WoWConstants.ChunkSize;
                            float worldY = WoWConstants.MapOrigin - clickTileY * WoWConstants.ChunkSize;
                            _camera.Position = new System.Numerics.Vector3(worldX, worldY, _camera.Position.Z);
                        }
                    }
                    _minimapDragging = false;
                }
            }
            else if (_minimapDragging)
            {
                _minimapDragging = false;
            }

            // Info overlay
            ImGui.SetCursorPos(new Vector2(padding, topPadding + mapSize + 10));
            int ctX = (int)MathF.Floor(camTileX);
            int ctY = (int)MathF.Floor(camTileY);
            ImGui.TextColored(new Vector4(1, 1, 1, 1), $"Tile: ({ctX},{ctY})  Zoom: {_minimapZoom:F1}x  Loaded: {loadedTileCount}");
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.7f, 0.7f, 0.7f, 1), "  |  Press M to close  |  Scroll to zoom  |  Drag to pan  |  Click to teleport");
            
            if (_minimapPanOffset != Vector2.Zero)
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Reset Pan"))
                    _minimapPanOffset = Vector2.Zero;
            }
        }
        ImGui.End();
        ImGui.PopStyleColor();
        ImGui.PopStyleVar();
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

        var allFiles = _dataSource.GetFileList(_extensionFilter);

        if (!string.IsNullOrEmpty(_searchFilter))
        {
            _filteredFiles = allFiles
                .Where(f => f.Contains(_searchFilter, StringComparison.OrdinalIgnoreCase))
                .Take(5000)
                .ToList();
        }
        else
        {
            _filteredFiles = allFiles.Take(5000).ToList();
        }

        _selectedFileIndex = -1;
    }

    private void LoadMpqDataSource(string gamePath, string? listfilePath)
    {
        try
        {
            _statusMessage = $"Loading MPQ archives from {gamePath}...";
            _dataSource?.Dispose();
            _dataSource = new MpqDataSource(gamePath, listfilePath);
            _statusMessage = $"Loaded: {_dataSource.Name}";

            // Load DBC tables directly from MPQ for replaceable texture resolution
            _texResolver = new ReplaceableTextureResolver();
            _texResolver.SetDataSource(_dataSource);
            _catalogView?.SetDataSource(_dataSource, _texResolver);
            var mpqDs = _dataSource as MpqDataSource;
            if (mpqDs != null)
            {
                _dbcProvider = new MpqDBCProvider(mpqDs.MpqService);
                var dbcProvider = _dbcProvider;

                // Load MD5 translate index for minimaps
                if (WoWMapConverter.Core.Services.Md5TranslateResolver.TryLoad(new[] { gamePath }, mpqDs.MpqService, out var md5Idx))
                {
                    _md5Index = md5Idx;
                    ViewerLog.Important(ViewerLog.Category.Dbc, $"Loaded MD5 Translate Index: {md5Idx?.HashToPlain.Count} entries");
                }

                _minimapRenderer?.Dispose();
                _minimapRenderer = new MinimapRenderer(_gl, _dataSource, _md5Index);

                // Find WoWDBDefs definitions directory
                string[] dbdSearchPaths = {
                    Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "lib", "WoWDBDefs", "definitions"),
                    Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "definitions"),
                    Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "WoWDBDefs", "definitions"),
                };

                string? dbdDir = null;
                foreach (var path in dbdSearchPaths)
                {
                    var resolved = Path.GetFullPath(path);
                    if (Directory.Exists(resolved) && Directory.GetFiles(resolved, "*.dbd").Length > 0)
                    {
                        dbdDir = resolved;
                        break;
                    }
                }

                if (dbdDir != null)
                {
                    _dbdDir = dbdDir;

                    // Infer build version from game path, validated against WoWDBDefs
                    string buildAlias = InferBuildFromPath(gamePath, dbdDir);
                    ViewerLog.Trace($"[MdxViewer] Inferred build: '{buildAlias}' from path: {gamePath}");
                    
                    if (!string.IsNullOrEmpty(buildAlias))
                    {
                        _dbcBuild = buildAlias;
                        ViewerLog.Trace($"[MdxViewer] Loading DBCs via DBCD (build: {buildAlias}, DBDs: {dbdDir})");
                        _texResolver.LoadFromDBC(dbcProvider, dbdDir, buildAlias);

                        // Discover maps
                        var mapDiscovery = new MapDiscoveryService(dbcProvider, dbdDir, buildAlias, _dataSource);
                        _discoveredMaps = mapDiscovery.DiscoverMaps();
                        ViewerLog.Important(ViewerLog.Category.Dbc, $"Discovered {_discoveredMaps.Count} maps via Map.dbc ({_discoveredMaps.Count(m => m.HasWdt)} with WDTs)");

                        // Load AreaTable for area name display
                        _areaTableService = new AreaTableService();
                        _areaTableService.Load(dbcProvider, dbdDir, buildAlias);
                    }
                    else
                    {
                        ViewerLog.Trace("[MdxViewer] Could not determine build version. DBC texture resolution unavailable.");
                    }
                }
                else
                {
                    ViewerLog.Trace("[MdxViewer] WoWDBDefs definitions not found. DBC texture resolution unavailable.");
                }
            }

            RefreshFileList();
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to load MPQs: {ex.Message}";
        }
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
        _window.Title = $"WoW Viewer - {_loadedFileName}";

        var ext = Path.GetExtension(filePath).ToLowerInvariant();
        string dir = Path.GetDirectoryName(filePath) ?? ".";

        try
        {
            _renderer?.Dispose();
            _renderer = null;

            switch (ext)
            {
                case ".mdx":
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
            _statusMessage = $"Failed to load: {ex.Message}";
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
        var candidatePaths = new List<string>(WarcraftNetM2Adapter.BuildSkinCandidates(originalPath));

        if (_dataSource != null)
        {
            var bestSkinPath = WarcraftNetM2Adapter.FindSkinInFileList(originalPath, _dataSource.GetFileList(".skin"));
            if (!string.IsNullOrWhiteSpace(bestSkinPath))
                candidatePaths.Add(bestSkinPath);
        }

        Exception? lastError = null;
        bool anySkinFound = false;

        foreach (var skinPath in candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            byte[]? skinBytes = null;

            if (File.Exists(skinPath))
                skinBytes = File.ReadAllBytes(skinPath);

            if (skinBytes == null && _dataSource != null)
                skinBytes = _dataSource.ReadFile(skinPath);

            if (skinBytes == null || skinBytes.Length == 0)
                continue;

            anySkinFound = true;

            try
            {
                ViewerLog.Trace($"[M2] Trying skin: {skinPath} ({skinBytes.Length} bytes)");
                var mdx = WarcraftNetM2Adapter.BuildRuntimeModel(m2Bytes, skinBytes, originalPath);
                LoadMdxModel(mdx, dir, originalPath);
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

        if (!anySkinFound)
            throw new InvalidDataException($"Missing companion .skin for M2: {Path.GetFileName(originalPath)}");

        throw new InvalidDataException(
            $"Failed to adapt M2 with available .skin candidates: {Path.GetFileName(originalPath)}",
            lastError);
    }

    private static ModelContainerKind DetectModelContainer(byte[] modelBytes)
    {
        if (modelBytes.Length < 4) return ModelContainerKind.Unknown;

        uint magic = BitConverter.ToUInt32(modelBytes, 0);
        if (magic == MdxHeaders.MAGIC) return ModelContainerKind.Mdlx;
        if (magic == 0x3032444D) return ModelContainerKind.Md20; // "MD20"

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

    private void LoadModelFromBytesWithContainerProbe(byte[] modelBytes, string sourcePath, string dir, string entrypoint)
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

                using (var ms = new MemoryStream(modelBytes))
                using (var br = new BinaryReader(ms))
                {
                    var mdx = MdxFile.Load(br);
                    LoadMdxModel(mdx, dir, sourcePath);
                }
                return;

            case ModelContainerKind.Md20:
                if (ext == ".mdx" || ext == ".mdl")
                    ViewerLog.Important(ViewerLog.Category.Mdx,
                        $"[ModelRouting] Extension/container mismatch: '{ext}' with MD20 root. Routing as M2-family: {Path.GetFileName(sourcePath)}");

                LoadM2FromBytes(modelBytes, sourcePath, dir);
                return;

            default:
                throw new InvalidDataException(
                    $"Unsupported model root magic ({GetModelMagicLabel(modelBytes)}) for '{Path.GetFileName(sourcePath)}'. Expected MDLX or MD20.");
        }
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
                LoadModelFromBytesWithContainerProbe(data, resolvedPath, dir, "Catalog");
            }

            _window.Title = $"WoW Viewer - {entry.Name} ({_loadedFileName})";
            _statusMessage = $"Loaded from catalog: {entry.Name} [{entry.EntryId}]";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to load {entry.Name}: {ex.Message}";
            _modelInfo = "";
        }
    }

    private void LoadFileFromDataSource(string virtualPath)
    {
        if (_dataSource == null) return;

        _statusMessage = $"Loading {Path.GetFileName(virtualPath)}...";
        _loadedFileName = Path.GetFileName(virtualPath);
        _lastVirtualPath = virtualPath;

        try
        {
            var data = _dataSource.ReadFile(virtualPath);
            if (data == null || data.Length == 0)
            {
                _statusMessage = $"Failed to read: {virtualPath}";
                return;
            }

            _renderer?.Dispose();
            _renderer = null;

            var ext = Path.GetExtension(virtualPath).ToLowerInvariant();

            // Write to cache folder for parsers that expect file paths
            Directory.CreateDirectory(CacheDir);
            var cachePath = Path.Combine(CacheDir, _loadedFileName!);
            File.WriteAllBytes(cachePath, data);
            _loadedFilePath = cachePath;

            switch (ext)
            {
                case ".mdx":
                case ".m2":
                    LoadModelFromBytesWithContainerProbe(data, virtualPath, CacheDir, "DataSource");
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

            _window.Title = $"WoW Viewer - {_loadedFileName}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Load failed: {ex.Message}";
            _modelInfo = "";
        }
    }

    private void LoadMdxModel(MdxFile mdx, string dir, string? virtualPath = null)
    {
        _loadedWmo = null;
        _loadedMdx = mdx;
        
        int validGeosets = mdx.Geosets.Count(g => g.Vertices.Count > 0 && g.Indices.Count > 0);
        int totalVerts = mdx.Geosets.Sum(g => g.Vertices.Count);
        int totalTris = mdx.Geosets.Sum(g => g.Indices.Count / 3);

        _renderer = new MdxRenderer(_gl, mdx, dir, _dataSource, _texResolver, virtualPath);

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        _modelInfo = $"Type: MDX (Alpha 0.5.3)\n" +
                     $"Version: {mdx.Version}\n" +
                     $"Name: {mdx.Model.Name}\n\n" +
                     $"Geosets: {mdx.Geosets.Count} ({validGeosets} valid)\n" +
                     $"Vertices: {totalVerts:N0}\n" +
                     $"Triangles: {totalTris:N0}\n\n" +
                     $"Materials: {mdx.Materials.Count}\n" +
                     $"Textures: {mdx.Textures.Count}\n" +
                     $"Bones: {mdx.Bones.Count}\n" +
                     $"Sequences: {mdx.Sequences.Count}\n";

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

        _statusMessage = $"Loaded MDX: {_loadedFileName} ({validGeosets} geosets, {totalVerts:N0} verts)";
    }

    private void LoadWmoModel(WmoV14ToV17Converter.WmoV14Data wmo, string dir)
    {
        _loadedMdx = null;
        _loadedWmo = wmo;
        
        int totalVerts = wmo.Groups.Sum(g => g.Vertices.Count);
        int totalTris = wmo.Groups.Sum(g => g.Indices.Count / 3);

        _renderer = new WmoRenderer(_gl, wmo, dir, _dataSource, _texResolver);

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        var wmoCenter = (wmo.BoundsMin + wmo.BoundsMax) * 0.5f;
        var wmoExtent = wmo.BoundsMax - wmo.BoundsMin;

        // Position camera offset from WMO center
        float dist = Math.Max(wmoExtent.Length() * 1.5f, 100f);
        _camera.Position = wmoCenter + new System.Numerics.Vector3(dist, 0, wmoExtent.Z * 0.3f);
        _camera.Yaw = 180f;
        _camera.Pitch = -10f;

        _modelInfo = $"Type: WMO v{wmo.Version}\n\n" +
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
                _worldScene = new WorldScene(_gl, wdtPath, _dataSource, _texResolver,
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
                var adapter = new Terrain.StandardTerrainAdapter(wdtRawBytes, mapName, _dataSource, _dbcBuild);
                var tm = new Terrain.TerrainManager(_gl, adapter, mapName, _dataSource);
                _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver,
                    onStatus: OnLoadStatus);
                wdtType = "Standard WDT";
            }

            _terrainManager = _worldScene.Terrain;
            _renderer = _worldScene;

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
            _currentMapId = curMapDef?.Id ?? -1;
            _sqlForceStreamRefresh = true;

            // Store DBC credentials for lazy loading (POI + Taxi deferred until user toggles them on)
            // Only Lighting is loaded eagerly since it affects rendering immediately.
            if (_dbcProvider != null && _dbdDir != null && _dbcBuild != null)
            {
                int mapId = curMapDef?.Id ?? -1;
                _worldScene.SetDbcCredentials(_dbcProvider, _dbdDir, _dbcBuild, mapId);

                if (curMapDef != null)
                    _worldScene.LoadLighting(_dbcProvider, _dbdDir, _dbcBuild, curMapDef.Id);
            }

            // Position camera — WMO-only maps use the WMO position, terrain maps use tile center
            var startPos = _worldScene.WmoCameraOverride ?? _terrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;

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
            _loadingScreen?.Disable();
        }
    }

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

    private void LoadVlmProject(string projectRoot)
    {
        _statusMessage = $"Loading VLM project from {projectRoot}...";

        // Clean up any existing scene
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
            _renderer = _vlmTerrainManager;

            // Position camera at center of loaded tiles
            var startPos = _vlmTerrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;

            var loader = _vlmTerrainManager.Loader;
            _modelInfo = $"Type: VLM Project\n" +
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

            _statusMessage = $"Loaded VLM project: {loader.MapName} ({loader.TileCoords.Count} tiles)";
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] VLM project load failed: {ex}");
            _statusMessage = $"VLM load failed: {ex.Message}";
            _modelInfo = $"VLM load error:\n{ex.Message}\n\nPath: {projectRoot}";
            _vlmTerrainManager?.Dispose();
            _vlmTerrainManager = null;
        }
    }

    private void PickObjectAtMouse(float mouseX, float mouseY)
    {
        if (_worldScene == null) return;

        var size = _window.Size;
        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return;

        if (mouseX < vpX || mouseX > vpX + vpW || mouseY < vpY || mouseY > vpY + vpH)
            return;

        float aspect = vpW / Math.Max(vpH, 1f);
        var view = _camera.GetViewMatrix();
        float farPlane = (_terrainManager != null || _vlmTerrainManager != null) ? 5000f : 10000f;
        var proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, farPlane);

        // Convert viewport-local mouse coords to NDC (-1..1)
        float localX = mouseX - vpX;
        float localY = mouseY - vpY;
        float ndcX = (localX / vpW) * 2f - 1f;
        float ndcY = 1f - (localY / vpH) * 2f; // flip Y

        var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
        _worldScene.SelectObjectByRay(rayOrigin, rayDir);

        // Build info string from the selected instance's embedded metadata
        var sel = _worldScene.SelectedInstance;
        if (sel.HasValue)
        {
            var inst = sel.Value;
            string type = _worldScene.SelectedObjectType == Terrain.ObjectType.Wmo ? "WMO" : "MDX";
            int idx = _worldScene.SelectedObjectIndex;

            // Convert renderer coords to WoW world coords
            float wowX = WoWConstants.MapOrigin - inst.PlacementPosition.Y;
            float wowY = WoWConstants.MapOrigin - inst.PlacementPosition.X;
            float wowZ = inst.PlacementPosition.Z;

            _selectedObjectType = type;
            _selectedObjectInfo = $"{type} [{idx}] {inst.ModelName}\n" +
                $"Path: {inst.ModelPath}\n" +
                $"UniqueId: {inst.UniqueId}\n" +
                $"Local: ({inst.PlacementPosition.X:F1}, {inst.PlacementPosition.Y:F1}, {inst.PlacementPosition.Z:F1})\n" +
                $"WoW:   ({wowX:F1}, {wowY:F1}, {wowZ:F1})\n" +
                $"Rotation: ({inst.PlacementRotation.X:F1}, {inst.PlacementRotation.Y:F1}, {inst.PlacementRotation.Z:F1})\n" +
                $"Scale: {inst.PlacementScale:F3}\n" +
                $"BB: ({inst.BoundsMin.X:F1},{inst.BoundsMin.Y:F1},{inst.BoundsMin.Z:F1}) - ({inst.BoundsMax.X:F1},{inst.BoundsMax.Y:F1},{inst.BoundsMax.Z:F1})";
        }
        else
        {
            _worldScene.ClearSelection();
            _selectedObjectIndex = -1;
            _selectedObjectType = "";
            _selectedObjectInfo = "";
        }
    }

    private bool IsPointInSceneViewport(float x, float y)
    {
        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return false;
        return x >= vpX && x <= vpX + vpW && y >= vpY && y <= vpY + vpH;
    }

    private bool TryGetSceneViewportRect(out float x, out float y, out float width, out float height)
    {
        var io = ImGui.GetIO();
        float topOffset = (_terrainManager != null || _vlmTerrainManager != null) ? MenuBarHeight + ToolbarHeight : MenuBarHeight;
        float leftInset = _showLeftSidebar ? SidebarWidth : 0f;
        float rightInset = _showRightSidebar ? SidebarWidth : 0f;

        x = leftInset;
        y = topOffset;
        width = io.DisplaySize.X - leftInset - rightInset;
        height = io.DisplaySize.Y - topOffset - StatusBarHeight;
        return width > 10f && height > 10f;
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

    private void OnResize(Vector2D<int> size)
    {
        _gl.Viewport(size);
    }

    private void LoadViewerSettings()
    {
        try
        {
            if (!File.Exists(ViewerSettingsPath))
            {
                // First run: default WMO liquid rotation for 3.3.5.
                WmoRenderer.MliqRotationQuarterTurns = 3;
                return;
            }

            string json = File.ReadAllText(ViewerSettingsPath);
            var settings = JsonSerializer.Deserialize<ViewerSettings>(json);
            if (settings == null)
                return;

            WmoRenderer.MliqRotationQuarterTurns = settings.WmoMliqRotationQuarterTurns;
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
                WmoMliqRotationQuarterTurns = WmoRenderer.MliqRotationQuarterTurns
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

    private bool _disposed;

    private void OnClose()
    {
        Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        SaveViewerSettings();

        _loadingScreen?.Dispose();
        _sqlPopulationService?.Dispose();
        _renderer?.Dispose();
        _worldScene?.Dispose();
        _terrainManager?.Dispose();
        _vlmTerrainManager?.Dispose();
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
        public int WmoMliqRotationQuarterTurns { get; set; }
    }
}
