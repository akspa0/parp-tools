using System.Numerics;
using ImGuiNET;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Export;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using MdxViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WoWMapConverter.Core.Converters;
using WoWMapConverter.Core.VLM;

namespace MdxViewer;

/// <summary>
/// Main viewer application. Owns window, GL context, ImGui, camera, renderer.
/// Provides menu bar, file browser, model info panel, and 3D viewport.
/// </summary>
public class ViewerApp : IDisposable
{
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
    private float _minimapZoom = 4f; // Number of tiles visible in each direction from camera
    private Rendering.LoadingScreen? _loadingScreen;

    // Output directories (next to the executable)
    private static readonly string OutputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");
    private static readonly string CacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache");
    private static readonly string ExportDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "export");

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
    private bool _wantOpenFile = false;
    private bool _wantOpenFolder = false;
    private bool _wantExportGlb = false;

    // Sidebar layout
    private bool _showLeftSidebar = true;
    private bool _showRightSidebar = true;
    private const float SidebarWidth = 320f;
    private const float MenuBarHeight = 22f;
    private const float ToolbarHeight = 32f;
    private const float StatusBarHeight = 24f;

    // Terrain/World state
    private TerrainManager? _terrainManager;
    private VlmTerrainManager? _vlmTerrainManager;
    private WorldScene? _worldScene;
    private bool _wantOpenVlmProject = false;

    // Object picking state
    private int _selectedObjectIndex = -1; // -1=none, 0..modf-1=WMO, modf..modf+mddf-1=MDX
    private string _selectedObjectType = "";
    private string _selectedObjectInfo = "";

    // Camera speed (adjustable via UI)
    private float _cameraSpeed = 50f;

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

        // Mouse input for viewport (not consumed by ImGui)
        foreach (var mouse in _input.Mice)
        {
            mouse.MouseDown += (_, btn) =>
            {
                if (btn == MouseButton.Right && !ImGui.GetIO().WantCaptureMouse)
                    _mouseDown = true;
                if (btn == MouseButton.Left && !ImGui.GetIO().WantCaptureMouse && _worldScene != null)
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

    private void OnUpdate(double dt)
    {
        _imGui.Update((float)dt);
        HandleKeyboardInput((float)dt);
    }

    private void HandleKeyboardInput(float dt)
    {
        if (_input.Keyboards.Count == 0) return;
        var kb = _input.Keyboards[0];

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
        // Disable once terrain tiles have actually loaded (LoadedTileCount > 0).
        if (_loadingScreen != null && _loadingScreen.IsActive)
        {
            // Check if terrain has loaded enough to dismiss
            bool terrainReady = _terrainManager != null && _terrainManager.LoadedTileCount > 0;
            bool isWmoOnly = _worldScene != null && _terrainManager != null && _terrainManager.Adapter.IsWmoBased;
            if (terrainReady || isWmoOnly)
            {
                _loadingScreen.Disable();
            }
            else
            {
                // Still loading — update AOI so tiles start streaming
                if (_terrainManager != null)
                    _terrainManager.UpdateAOI(_camera.Position);
                _loadingScreen.Render();
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
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 4f, aspect, 0.1f, farPlane);

            // Update terrain AOI before rendering
            if (_terrainManager != null)
                _terrainManager.UpdateAOI(_camera.Position);
            else if (_vlmTerrainManager != null)
                _vlmTerrainManager.UpdateAOI(_camera.Position);

            // Update current area name from chunk under camera (throttled to avoid per-frame overhead)
            if (_areaTableService != null && _terrainManager != null && _frameCount == 0)
            {
                var chunk = _terrainManager.Renderer.GetChunkAt(_camera.Position.X, _camera.Position.Y);
                if (chunk != null && chunk.AreaId != 0)
                {
                    // Simple direct lookup — AreaID from MCNK maps to AreaTable key
                    var name = _areaTableService.GetAreaDisplayName(chunk.AreaId);
                    if (name.StartsWith("Unknown") && _currentAreaName != name)
                        Console.WriteLine($"[AreaLookup] MCNK AreaId={chunk.AreaId} → {name} (mapId={_currentMapId})");
                    _currentAreaName = name;
                }
                else
                    _currentAreaName = "";
            }

            // Render the scene (WorldScene handles terrain + objects + BBs; standalone renderers handle themselves)
            _renderer.Render(view, proj);
        }

        // Render ImGui overlay
        DrawUI();
        _imGui.Render();
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

        // Minimap floats independently (bottom-left or bottom-right depending on sidebars)
        if (_worldScene != null || _vlmTerrainManager != null)
            DrawMinimap();

        DrawStatusBar();

        // Modal dialogs
        if (_showFolderInput)
            DrawFolderInputDialog();
        if (_showListfileInput)
            DrawListfileInputDialog();
        if (_showVlmExportDialog)
            DrawVlmExportDialog();
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

                ImGui.Separator();

                if (ImGui.MenuItem("Export GLB...", _renderer != null))
                    _wantExportGlb = true;

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

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Help"))
            {
                if (ImGui.MenuItem("About"))
                    _statusMessage = "WoW Model Viewer — MDX/WMO viewer with GLB export. Built with Silk.NET + ImGui.";
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

            using var vlmDialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select VLM Project folder (containing dataset/ with JSON files)",
                UseDescriptionForTitle = true,
                ShowNewFolderButton = false
            };

            string? vlmPath = null;
            var vlmThread = new System.Threading.Thread(() =>
            {
                if (vlmDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                    vlmPath = vlmDialog.SelectedPath;
            });
            vlmThread.SetApartmentState(System.Threading.ApartmentState.STA);
            vlmThread.Start();
            vlmThread.Join();

            if (!string.IsNullOrEmpty(vlmPath) && Directory.Exists(vlmPath))
                LoadVlmProject(vlmPath);
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
                            GlbExporter.ExportWmo(wmo, dir, glbPath, _dataSource);
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

    private void DrawFolderInputDialog()
    {
        if (!_showFolderInput) return;

        // Use WinForms folder browser for native experience
        _showFolderInput = false;

        using var dialog = new System.Windows.Forms.FolderBrowserDialog
        {
            Description = "Select WoW game folder (containing Data/ with MPQs)",
            UseDescriptionForTitle = true,
            ShowNewFolderButton = false
        };

        if (!string.IsNullOrEmpty(_folderInputBuf))
            dialog.InitialDirectory = _folderInputBuf;

        // Run dialog on STA thread
        string? selectedPath = null;
        var thread = new System.Threading.Thread(() =>
        {
            var result = dialog.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
                selectedPath = dialog.SelectedPath;
        });
        thread.SetApartmentState(System.Threading.ApartmentState.STA);
        thread.Start();
        thread.Join();

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
    private static string? ShowFolderDialogSTA(string description)
    {
        string? result = null;
        var thread = new Thread(() =>
        {
            using var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = description,
                UseDescriptionForTitle = true
            };
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                result = dialog.SelectedPath;
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        return result;
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
                    if (ImGui.Checkbox($"Liquid ({liquidRenderer.MeshCount})", ref showLiquid))
                        liquidRenderer.ShowLiquid = showLiquid;
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

                if (ImGui.IsItemHovered())
                {
                    ImGui.BeginTooltip();
                    ImGui.Text($"Directory: {map.Directory}");
                    ImGui.Text($"Status: {(hasWdt ? "WDT Found" : "WDT Missing")}");
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
                ImGui.Spacing();
            }

            // ── Model Info section ──
            if (_showModelInfo && ImGui.CollapsingHeader("Model Info", ImGuiTreeNodeFlags.DefaultOpen))
            {
                DrawModelInfoContent();
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

        // Camera speed
        ImGui.SliderFloat("Camera Speed", ref _cameraSpeed, 5f, 500f, "%.0f");
        ImGui.Text("Hold Shift for 5x boost");

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

        // POI toggle
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0)
        {
            bool showPoi = _worldScene.ShowPoi;
            if (ImGui.Checkbox($"Area POIs ({_worldScene.PoiLoader.Entries.Count})", ref showPoi))
                _worldScene.ShowPoi = showPoi;
        }

        // Taxi paths toggle + clear selection button
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

    private void DrawMinimap()
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
                    // Minimap files: map{row}_{col}.blp. tx=col, ty=row.
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

                    // Infer build version from game path first
                    string buildAlias = InferBuildFromPath(gamePath);
                    
                    // If not found, search WoWDBDefs definitions
                    if (string.IsNullOrEmpty(buildAlias))
                    {
                        buildAlias = FindBuildInWoWDBDefs(dbdDir);
                    }
                    
                    // Resolve short alias to full build string (e.g. "0.6.0" -> "0.6.0.3592")
                    buildAlias = ResolveFullBuild(buildAlias);
                    
                    if (!string.IsNullOrEmpty(buildAlias))
                    {
                        _dbcBuild = buildAlias;
                        Console.WriteLine($"[MdxViewer] Loading DBCs via DBCD (build: {buildAlias}, DBDs: {dbdDir})");
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
                        Console.WriteLine("[MdxViewer] Could not determine build version. DBC texture resolution unavailable.");
                    }
                }
                else
                {
                    Console.WriteLine("[MdxViewer] WoWDBDefs definitions not found. DBC texture resolution unavailable.");
                }
            }

            RefreshFileList();
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to load MPQs: {ex.Message}";
        }
    }

    private static string InferBuildFromPath(string path)
    {
        var p = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar).ToLowerInvariant();
        
        // Check for version strings in path (with or without dots)
        if (p.Contains("0.5.3") || p.Contains("053")) return "0.5.3";
        if (p.Contains("0.5.5") || p.Contains("055")) return "0.5.5";
        if (p.Contains("0.6.0") || p.Contains("060")) return "0.6.0";
        if (p.Contains("3.3.5") || p.Contains("335")) return "3.3.5";
        
        // Fallback: check for known MPQ names
        if (Directory.Exists(path))
        {
            var mpqs = Directory.GetFiles(path, "*.mpq", SearchOption.AllDirectories)
                .Select(f => Path.GetFileName(f).ToLowerInvariant()).ToArray();
            if (mpqs.Any(m => m.Contains("patch") && m.Contains("3"))) return "3.3.5";
        }
        
        return "";
    }

    private static readonly Dictionary<string, string> FullBuildMap = new()
    {
        { "0.5.3", "0.5.3.3368" },
        { "0.5.5", "0.5.5.3494" },
        { "0.6.0", "0.6.0.3592" },
        { "3.3.5", "3.3.5.12340" },
    };

    private static string ResolveFullBuild(string alias)
    {
        if (string.IsNullOrEmpty(alias)) return alias;
        return FullBuildMap.TryGetValue(alias, out var full) ? full : alias;
    }

    private static string FindBuildInWoWDBDefs(string dbdDir)
    {
        // Search WoWDBDefs for known Alpha/early builds
        if (!Directory.Exists(dbdDir)) return "";
        
        // Ordered by preference — check newest Alpha first
        var knownBuilds = new (string search, string result)[]
        {
            ("BUILD 0.6.0.3592", "0.6.0"),
            ("BUILD 0.5.5.3494", "0.5.5"),
            ("BUILD 0.5.3.3368", "0.5.3"),
        };
        
        // Only need to check one DBD file — AreaTable.dbd is always present
        var areaTable = Path.Combine(dbdDir, "AreaTable.dbd");
        if (File.Exists(areaTable))
        {
            var content = File.ReadAllText(areaTable);
            foreach (var (search, result) in knownBuilds)
            {
                if (content.Contains(search))
                {
                    Console.WriteLine($"[MdxViewer] Found {result} build definition in AreaTable.dbd");
                    return result;
                }
            }
        }
        
        return "";
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
                    var mdx = MdxFile.Load(filePath);
                    LoadMdxModel(mdx, dir);
                    break;

                case ".wmo":
                    var converter = new WmoV14ToV17Converter();
                    var wmo = converter.ParseWmoV14(filePath);
                    LoadWmoModel(wmo, dir);
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
                    var mdx = MdxFile.Load(cachePath);
                    LoadMdxModel(mdx, CacheDir, virtualPath);
                    break;

                case ".wmo":
                    var converter = new WmoV14ToV17Converter();
                    var wmo = converter.ParseWmoV14(cachePath);
                    
                    // v16 split format: root has GroupCount but no embedded MOGP chunks
                    // Load separate group files from the data source
                    if (wmo.Groups.Count == 0 && wmo.GroupCount > 0 && _dataSource != null)
                    {
                        var wmoDir = Path.GetDirectoryName(virtualPath)?.Replace('/', '\\') ?? "";
                        var wmoBase = Path.GetFileNameWithoutExtension(virtualPath);
                        Console.WriteLine($"[WMO] v16 split: loading {wmo.GroupCount} group files from data source");
                        
                        for (int gi = 0; gi < wmo.GroupCount; gi++)
                        {
                            var groupName = $"{wmoBase}_{gi:D3}.wmo";
                            var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                            var groupBytes = _dataSource.ReadFile(groupPath);
                            if (groupBytes != null && groupBytes.Length > 0)
                            {
                                Console.WriteLine($"[WMO] Group {gi}: loaded {groupBytes.Length} bytes from '{groupPath}'");
                                converter.ParseGroupFile(groupBytes, wmo, gi);
                            }
                            else
                            {
                                Console.WriteLine($"[WMO] Group {gi}: NOT FOUND '{groupPath}'");
                            }
                        }
                        
                        // Populate group names after loading
                        for (int gi = 0; gi < wmo.Groups.Count && gi < wmo.GroupInfos.Count; gi++)
                        {
                            var nameOfs = wmo.GroupInfos[gi].NameOffset;
                            if (wmo.Groups[gi].Name == null)
                                wmo.Groups[gi].Name = $"group_{gi}";
                        }
                        
                        // Recalculate bounds from loaded group geometry
                        var bMin = new System.Numerics.Vector3(float.MaxValue);
                        var bMax = new System.Numerics.Vector3(float.MinValue);
                        foreach (var g in wmo.Groups)
                        {
                            foreach (var v in g.Vertices)
                            {
                                bMin = System.Numerics.Vector3.Min(bMin, v);
                                bMax = System.Numerics.Vector3.Max(bMax, v);
                            }
                        }
                        if (bMin.X < float.MaxValue)
                        {
                            wmo.BoundsMin = bMin;
                            wmo.BoundsMax = bMax;
                            Console.WriteLine($"[WMO] Recalculated bounds from groups: ({bMin.X:F1},{bMin.Y:F1},{bMin.Z:F1}) - ({bMax.X:F1},{bMax.Y:F1},{bMax.Z:F1})");
                        }
                    }
                    
                    LoadWmoModel(wmo, CacheDir);
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

        // Position camera to view model from a good angle
        // MirrorX in Render() negates X, so camera sees mirrored model.
        // Camera at -X with yaw=0 looks toward +X → sees the model's front (which is at -X after mirror).
        var bmin = mdx.Model.Bounds.Extent.Min;
        var bmax = mdx.Model.Bounds.Extent.Max;
        var center = new System.Numerics.Vector3(
            -(bmin.X + bmax.X) * 0.5f,
            (bmin.Y + bmax.Y) * 0.5f,
            (bmin.Z + bmax.Z) * 0.5f);
        var extent = new System.Numerics.Vector3(
            bmax.X - bmin.X, bmax.Y - bmin.Y, bmax.Z - bmin.Z);

        float dist = Math.Max(extent.Length() * 1.5f, 50f);
        _camera.Position = center + new System.Numerics.Vector3(-dist, 0, extent.Z * 0.3f);
        _camera.Yaw = 0f;
        _camera.Pitch = -10f; // Slight downward angle

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

        // Show loading screen (replicates Alpha client's EnableLoadingScreen)
        _loadingScreen?.Enable(_dataSource);
        PresentLoadingFrame();

        try
        {
            // Detect Alpha WDT vs Standard WDT by file size.
            // Alpha WDTs are monolithic files containing all embedded ADTs (typically ≥64KB).
            // Standard WDTs are small files (~33KB MAIN chunk) pointing to separate .adt files.
            long fileSize = new FileInfo(wdtPath).Length;
            bool isAlpha = fileSize >= 65536;
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

                var wdtBytes = File.ReadAllBytes(wdtPath);
                string mapName = Path.GetFileNameWithoutExtension(wdtPath);
                var adapter = new Terrain.StandardTerrainAdapter(wdtBytes, mapName, _dataSource);
                var tm = new Terrain.TerrainManager(_gl, adapter, mapName, _dataSource);
                _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver,
                    onStatus: OnLoadStatus);
                wdtType = "Standard WDT";
            }

            _terrainManager = _worldScene.Terrain;
            _renderer = _worldScene;

            // Find mapId for this world
            string curMapName = _terrainManager.MapName;
            var curMapDef = _discoveredMaps.FirstOrDefault(m =>
                string.Equals(m.Directory, curMapName, StringComparison.OrdinalIgnoreCase));
            _currentMapId = curMapDef?.Id ?? -1;

            // Load AreaPOI, TaxiPaths, and Lighting from DBC if available
            if (_dbcProvider != null && _dbdDir != null && _dbcBuild != null)
            {
                _worldScene.LoadAreaPoi(_dbcProvider, _dbdDir, _dbcBuild);

                if (curMapDef != null)
                {
                    var dbcd = new DBCD.DBCD(_dbcProvider, new DBCD.Providers.FilesystemDBDProvider(_dbdDir));
                    _worldScene.LoadTaxiPaths(dbcd, _dbcBuild, curMapDef.Id);
                    _worldScene.LoadLighting(_dbcProvider, _dbdDir, _dbcBuild, curMapDef.Id);
                }
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
            Console.WriteLine($"[ViewerApp] WDT load failed: {ex}");
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
        _loadingScreen.Render();
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

            _statusMessage = $"Loaded VLM project: {loader.MapName} ({loader.TileCoords.Count} tiles)";
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ViewerApp] VLM project load failed: {ex}");
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
        float aspect = (float)size.X / Math.Max(size.Y, 1);
        var view = _camera.GetViewMatrix();
        float farPlane = (_terrainManager != null || _vlmTerrainManager != null) ? 5000f : 10000f;
        var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 4f, aspect, 0.1f, farPlane);

        // Convert mouse coords to NDC (-1..1)
        float ndcX = (mouseX / size.X) * 2f - 1f;
        float ndcY = 1f - (mouseY / size.Y) * 2f; // flip Y

        var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
        _worldScene.SelectObjectByRay(rayOrigin, rayDir);

        // Build info string from selection
        var sel = _worldScene.SelectedInstance;
        if (sel.HasValue)
        {
            var inst = sel.Value;
            string type = _worldScene.SelectedObjectType == Terrain.ObjectType.Wmo ? "WMO" : "MDX";
            int idx = _worldScene.SelectedObjectIndex;

            // Find placement data for name/rotation info
            if (_worldScene.SelectedObjectType == Terrain.ObjectType.Wmo && idx < _worldScene.ModfPlacements.Count)
            {
                var p = _worldScene.ModfPlacements[idx];
                string name = p.NameIndex < _worldScene.WmoModelNames.Count
                    ? Path.GetFileName(_worldScene.WmoModelNames[p.NameIndex]) : "?";
                // Convert renderer coords to WoW world coords
                float wowX = WoWConstants.MapOrigin - p.Position.Y;
                float wowY = WoWConstants.MapOrigin - p.Position.X;
                float wowZ = p.Position.Z;
                _selectedObjectType = "WMO";
                _selectedObjectInfo = $"WMO [{idx}] {name}\n" +
                    $"Local: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})\n" +
                    $"WoW:   ({wowX:F1}, {wowY:F1}, {wowZ:F1})\n" +
                    $"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})\n" +
                    $"Flags: 0x{p.Flags:X4}";
            }
            else if (_worldScene.SelectedObjectType == Terrain.ObjectType.Mdx && idx < _worldScene.MddfPlacements.Count)
            {
                var p = _worldScene.MddfPlacements[idx];
                string name = p.NameIndex < _worldScene.MdxModelNames.Count
                    ? Path.GetFileName(_worldScene.MdxModelNames[p.NameIndex]) : "?";
                float wowX = WoWConstants.MapOrigin - p.Position.Y;
                float wowY = WoWConstants.MapOrigin - p.Position.X;
                float wowZ = p.Position.Z;
                _selectedObjectType = "MDX";
                _selectedObjectInfo = $"MDX [{idx}] {name}\n" +
                    $"Local: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})\n" +
                    $"WoW:   ({wowX:F1}, {wowY:F1}, {wowZ:F1})\n" +
                    $"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})\n" +
                    $"Scale: {p.Scale:F3}";
            }
        }
        else
        {
            _worldScene.ClearSelection();
            _selectedObjectIndex = -1;
            _selectedObjectType = "";
            _selectedObjectInfo = "";
        }
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

    private void OnClose()
    {
        _loadingScreen?.Dispose();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager = null; // owned by WorldScene
        _renderer?.Dispose();
        _dataSource?.Dispose();
        _imGui?.Dispose();
        _gl?.Dispose();
        _input?.Dispose();
    }

    public void Dispose()
    {
        _renderer?.Dispose();
        _worldScene?.Dispose();
        _terrainManager?.Dispose();
        _vlmTerrainManager?.Dispose();
        _minimapRenderer?.Dispose();
        _imGui.Dispose();
        _gl.Dispose();
    }
}
