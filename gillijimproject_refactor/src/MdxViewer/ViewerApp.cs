using System.Numerics;
using ImGuiNET;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Export;
using MdxViewer.Rendering;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WoWMapConverter.Core.Converters;

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
    private string? _lastVirtualPath; // Virtual path of last loaded file (for DBC lookup)
    private string _statusMessage = "No data source loaded. Use File > Open Game Folder or Open File.";

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

    // Mouse state
    private float _lastMouseX, _lastMouseY;
    private bool _mouseDown;
    private bool _mouseOverViewport;

    // UI state
    private bool _showFileBrowser = true;
    private bool _showModelInfo = true;
    private bool _showDemoWindow = false;
    private bool _wantOpenFile = false;
    private bool _wantOpenFolder = false;
    private bool _wantExportGlb = false;

    // Folder dialog workaround (ImGui doesn't have native dialogs)
    private bool _showFolderInput = false;
    private string _folderInputBuf = "";
    private bool _showListfileInput = false;
    private string _listfileInputBuf = "";

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

        _gl.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Enable(EnableCap.CullFace);

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
                    // Free-fly: mouse controls camera rotation (yaw/pitch)
                    // DEBUG: Log mouse movement to diagnose inversion
                    Console.WriteLine($"[Camera] Mouse move: dx={dx:F2}, dy={dy:F2}, Yaw={_camera.Yaw:F2}, Pitch={_camera.Pitch:F2}");
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
        // Free-fly: WASD moves the camera position
        float speed = 20f * dt;
        bool w = _input.Keyboards.Count > 0 && _input.Keyboards[0].IsKeyPressed(Key.W);
        bool a = _input.Keyboards.Count > 0 && _input.Keyboards[0].IsKeyPressed(Key.A);
        bool s = _input.Keyboards.Count > 0 && _input.Keyboards[0].IsKeyPressed(Key.S);
        bool d = _input.Keyboards.Count > 0 && _input.Keyboards[0].IsKeyPressed(Key.D);
        bool q = _input.Keyboards.Count > 0 && _input.Keyboards[0].IsKeyPressed(Key.Q);
        bool e = _input.Keyboards.Count > 0 && _input.Keyboards[0].IsKeyPressed(Key.E);

        if (w || a || s || d || q || e)
        {
            float forward = (w ? 1 : 0) - (s ? 1 : 0);
            float right = (d ? 1 : 0) - (a ? 1 : 0);
            float up = (e ? 1 : 0) - (q ? 1 : 0);
            _camera.Move(forward, right, up, speed);
        }
    }

    private unsafe void OnRender(double dt)
    {
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        // Render 3D scene first
        if (_renderer != null)
        {
            var size = _window.Size;
            float aspect = (float)size.X / Math.Max(size.Y, 1);
            var view = _camera.GetViewMatrix();
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 4f, aspect, 0.1f, 10000f);
            _renderer.Render(view, proj);
        }

        // Render ImGui overlay
        DrawUI();
        _imGui.Render();
    }

    private void DrawUI()
    {
        DrawMenuBar();

        if (_showFileBrowser)
            DrawFileBrowser();

        if (_showModelInfo)
            DrawModelInfoPanel();

        DrawStatusBar();

        // Modal dialogs
        if (_showFolderInput)
            DrawFolderInputDialog();
        if (_showListfileInput)
            DrawListfileInputDialog();
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

                ImGui.MenuItem("File Browser", "", ref _showFileBrowser);
                ImGui.MenuItem("Model Info", "", ref _showModelInfo);

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

        if (_wantExportGlb)
        {
            _wantExportGlb = false;
            if (_loadedFilePath != null)
            {
                Directory.CreateDirectory(ExportDir);
                string glbPath = Path.Combine(ExportDir, Path.ChangeExtension(_loadedFileName!, ".glb"));
                try
                {
                    var ext = Path.GetExtension(_loadedFilePath).ToLowerInvariant();
                    string dir = Path.GetDirectoryName(_loadedFilePath) ?? ".";
                    if (ext == ".mdx")
                    {
                        var mdx = MdxFile.Load(_loadedFilePath);
                        GlbExporter.ExportMdx(mdx, dir, glbPath);
                    }
                    else if (ext == ".wmo")
                    {
                        var converter = new WmoV14ToV17Converter();
                        var wmo = converter.ParseWmoV14(_loadedFilePath);
                        GlbExporter.ExportWmo(wmo, dir, glbPath);
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

    private void DrawFileBrowser()
    {
        ImGui.SetNextWindowSize(new Vector2(350, 600), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(0, 22), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("File Browser", ref _showFileBrowser))
        {
            if (_dataSource == null || !_dataSource.IsLoaded)
            {
                ImGui.TextWrapped("No data source loaded.\nUse File > Open Game Folder to load MPQ archives.");
                ImGui.End();
                return;
            }

            ImGui.Text($"Source: {_dataSource.Name}");
            ImGui.Separator();

            // Extension filter
            if (ImGui.BeginCombo("Type", _extensionFilter))
            {
                string[] filters = { ".mdx", ".wmo", ".m2", ".blp" };
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

            // File list
            if (ImGui.BeginChild("FileList", new Vector2(0, 0), true))
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
        ImGui.End();
    }

    private void DrawModelInfoPanel()
    {
        ImGui.SetNextWindowSize(new Vector2(300, 500), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(_window.Size.X - 300, 22), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("Model Info", ref _showModelInfo))
        {
            if (string.IsNullOrEmpty(_modelInfo))
            {
                ImGui.TextWrapped("No model loaded.");
            }
            else
            {
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

                    // Show All / Hide All buttons
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
        }
        ImGui.End();
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
        }
        ImGui.End();
        ImGui.PopStyleVar();
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
            var mpqDs = _dataSource as MpqDataSource;
            if (mpqDs != null)
            {
                var dbcProvider = new MpqDBCProvider(mpqDs.MpqService);

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
                    // Infer build version from game path
                    string buildAlias = InferBuildFromPath(gamePath);
                    if (!string.IsNullOrEmpty(buildAlias))
                    {
                        Console.WriteLine($"[MdxViewer] Loading DBCs via DBCD (build: {buildAlias}, DBDs: {dbdDir})");
                        _texResolver.LoadFromDBC(dbcProvider, dbdDir, buildAlias);
                    }
                    else
                    {
                        Console.WriteLine("[MdxViewer] Could not infer build version from game path. DBC texture resolution unavailable.");
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
        if (p.Contains("0.5.3")) return "0.5.3";
        if (p.Contains("0.5.5")) return "0.5.5";
        if (p.Contains("0.6.0")) return "0.6.0";
        if (p.Contains("3.3.5")) return "3.3.5";
        // Fallback: check for known MPQ names
        if (Directory.Exists(path))
        {
            var mpqs = Directory.GetFiles(path, "*.mpq", SearchOption.AllDirectories)
                .Select(f => Path.GetFileName(f).ToLowerInvariant()).ToArray();
            if (mpqs.Any(m => m.Contains("patch") && m.Contains("3"))) return "3.3.5";
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
                    LoadWmoModel(wmo, CacheDir);
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
        int validGeosets = mdx.Geosets.Count(g => g.Vertices.Count > 0 && g.Indices.Count > 0);
        int totalVerts = mdx.Geosets.Sum(g => g.Vertices.Count);
        int totalTris = mdx.Geosets.Sum(g => g.Indices.Count / 3);

        _renderer = new MdxRenderer(_gl, mdx, dir, _dataSource, _texResolver, virtualPath);

        // Position camera to view model from a good angle
        var bmin = mdx.Model.Bounds.Extent.Min;
        var bmax = mdx.Model.Bounds.Extent.Max;
        var center = new System.Numerics.Vector3(
            (bmin.X + bmax.X) * 0.5f,
            (bmin.Y + bmax.Y) * 0.5f,
            (bmin.Z + bmax.Z) * 0.5f);
        var extent = new System.Numerics.Vector3(
            bmax.X - bmin.X, bmax.Y - bmin.Y, bmax.Z - bmin.Z);

        // Position camera offset from model, looking at it
        float dist = Math.Max(extent.Length() * 1.5f, 50f);
        _camera.Position = center + new System.Numerics.Vector3(dist, 0, extent.Z * 0.3f);
        _camera.Yaw = 180f; // Face toward origin (model center)
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
        _renderer?.Dispose();
        _dataSource?.Dispose();
        _imGui?.Dispose();
        _gl?.Dispose();
        _input?.Dispose();
    }

    public void Dispose()
    {
        _renderer?.Dispose();
        _dataSource?.Dispose();
    }
}
