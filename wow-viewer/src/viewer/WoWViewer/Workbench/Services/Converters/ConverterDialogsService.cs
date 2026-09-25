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
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;
using static WoWViewer.ProjectOutputService;

namespace WoWViewer;

/// <summary>
/// Map and WMO converter dialogs, their output-path helpers, and the external converter process runner.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class ConverterDialogsService
{
    private readonly IViewerAppHost _host;

    internal ConverterDialogsService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ref string _mapConvertLkMapDir => ref _host.MapConvertLkMapDir;
    private ref string _mapConvertOutputDir => ref _host.MapConvertOutputDir;
    private ref string _mapConvertProjectSourceKey => ref _host.MapConvertProjectSourceKey;
    private ref string _mapConvertSourcePath => ref _host.MapConvertSourcePath;
    private ref string _projectOutputRootDir => ref _host.ProjectOutputRootDir;
    private ref bool _showMapConverterDialog => ref _host.ShowMapConverterDialog;
    private ref bool _showWmoConverterDialog => ref _host.ShowWmoConverterDialog;
    private ref string _wmoConvertSourcePath => ref _host.WmoConvertSourcePath;
    private string GetProjectOutputRootDirectory() => _host.GetProjectOutputRootDirectory();
    private void HandleProjectOutputRootChanged() => _host.HandleProjectOutputRootChanged();
    private void LoadWdtTerrain(string wdtPath) => _host.LoadWdtTerrain(wdtPath);

    private int _mapConvertDirection = 0; // 0 = Alpha WDT source, 1 = split ADT source
    private MapConversionTargetFormat _mapConvertTargetFormat = MapConversionTargetFormat.LkAdtV18;
    private string _mapConvertSplitClientRoot = "";
    private string _mapConvertSplitMapName = "";
    private bool _mapConvertCopyAlphaSourceWdt = true;
    private bool _mapConvertVerbose = true;
    private string _areaCrosswalkPath = ""; // Optional area crosswalk CSV for Alpha→LK conversion
    private bool _mapConverting = false;
    private readonly List<string> _mapConvertLog = new();
    private bool _mapConvertScrollToBottom = false;
    private string? _mapConvertError = null;
    private bool _mapConvertDone = false;
    private string? _mapConvertLastLoadPath;
    private int _wmoConvertDirection = 0; // 0 = Alpha(v14/v16)→LK(v17), 1 = LK(v17)→Alpha(v14)
    private string _wmoConvertOutputPath = "";
    private bool _wmoConvertCopyTextures = true;
    private bool _wmoConverting = false;
    private readonly List<string> _wmoConvertLog = new();
    private bool _wmoConvertScrollToBottom = false;
    private string? _wmoConvertError = null;
    private bool _wmoConvertDone = false;

    internal void DrawMapConverterDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(620, 640), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 310,
            ImGui.GetIO().DisplaySize.Y / 2 - 320), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Map Converter", ref _showMapConverterDialog))
        {
            ImGui.TextWrapped("Convert maps between Alpha 0.5.3 monolithic WDT and later ADT families. The target format is explicit; a split ADT source is never silently sent to the LK writer as MoP output.");
            ImGui.TextDisabled("Conversions write into timestamped project folders under the configured project root. Original source files are not overwritten.");
            ImGui.Spacing();

            // Source family selector. The destination is selected independently below.
            ImGui.Text("Source family:");
            int previousDirection = _mapConvertDirection;
            ImGui.RadioButton("Alpha 0.5.3 WDT", ref _mapConvertDirection, 0);
            ImGui.SameLine();
            ImGui.RadioButton("Split ADT family", ref _mapConvertDirection, 1);
            if (previousDirection != _mapConvertDirection)
            {
                _mapConvertTargetFormat = _mapConvertDirection == 0
                    ? MapConversionTargetFormat.LkAdtV18
                    : MapConversionTargetFormat.AlphaWdt053;
                _mapConvertDone = false;
                _mapConvertError = null;
                _mapConvertLastLoadPath = null;
                EnsureMapConverterProjectOutputDirectory(forceNew: false);
            }

            ImGui.Text("Target format / era:");
            if (ImGui.BeginCombo("##mapconv_target_format", MapConversionFormats.GetDisplayName(_mapConvertTargetFormat)))
            {
                foreach (MapConversionTargetFormat format in MapConversionFormats.AllTargets)
                {
                    bool isSelected = format == _mapConvertTargetFormat;
                    bool hasWriter = MapConversionFormats.HasWriter(format);
                    string label = hasWriter
                        ? MapConversionFormats.GetDisplayName(format)
                        : $"{MapConversionFormats.GetDisplayName(format)} (writer unavailable)";
                    if (ImGui.Selectable($"{label}##mapconv_target_{format}", isSelected))
                    {
                        _mapConvertTargetFormat = format;
                        _mapConvertDone = false;
                        _mapConvertError = null;
                        _mapConvertLastLoadPath = null;
                        EnsureMapConverterProjectOutputDirectory(forceNew: false);
                    }

                    if (isSelected)
                        ImGui.SetItemDefaultFocus();
                }

                ImGui.EndCombo();
            }

            MapConversionSourceFormat sourceFormat = _mapConvertDirection == 0
                ? MapConversionSourceFormat.AlphaWdt053
                : MapConversionSourceFormat.SplitAdtFamily;
            MapConversionValidationResult formatValidation = MapConversionFormats.Validate(
                sourceFormat,
                _mapConvertTargetFormat);
            if (!formatValidation.IsSupported)
            {
                ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(1f, 0.45f, 0.25f, 1f));
                ImGui.TextWrapped($"Blocked: {formatValidation.Error}");
                ImGui.PopStyleColor();
            }
            else if (formatValidation.IsLossy)
            {
                ImGui.TextColored(new Vector4(1f, 0.8f, 0.35f, 1f), "Lossy conversion: modern or source-era-only state may be reduced.");
            }

            foreach (string warning in formatValidation.Warnings)
                ImGui.TextWrapped($"Warning: {warning}");

            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Spacing();

            if (!string.IsNullOrWhiteSpace(_mapConvertSourcePath))
                EnsureMapConverterProjectOutputDirectory(forceNew: false);

            if (_mapConvertDirection == 0)
            {
                ImGui.Text("Source Alpha 0.5.3 WDT:");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##a2l_src", ref _mapConvertSourcePath, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##a2l_src"))
                {
                    string? initDir = !string.IsNullOrEmpty(_mapConvertSourcePath) ? Path.GetDirectoryName(_mapConvertSourcePath) : null;
                    ImGuiPathPicker.Instance.Open(
                        "Select Alpha WDT file",
                        pickFolder: false,
                        initialPath: initDir,
                        filterExtension: ".wdt;.mpq",
                        picked =>
                        {
                            if (!string.IsNullOrWhiteSpace(picked))
                            {
                                _mapConvertSourcePath = picked;
                                EnsureMapConverterProjectOutputDirectory(forceNew: false);
                            }
                        });
                }

                ImGui.Checkbox("Copy source Alpha WDT into project", ref _mapConvertCopyAlphaSourceWdt);
                if (_mapConvertTargetFormat == MapConversionTargetFormat.LkAdtV18)
                    ImGui.TextDisabled("Target output: one LK v18 monolithic ADT per occupied tile plus an LK WDT.");
                else if (_mapConvertTargetFormat == MapConversionTargetFormat.AlphaWdt053)
                    ImGui.TextDisabled("Target output: an explicit Alpha 0.5.3 identity copy under the project folder.");
            }
            else
            {
                ImGui.Text("Source split-ADT WDT:");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##l2a_src", ref _mapConvertSourcePath, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##l2a_src"))
                {
                    string? initDir = !string.IsNullOrEmpty(_mapConvertSourcePath) ? Path.GetDirectoryName(_mapConvertSourcePath) : null;
                    ImGuiPathPicker.Instance.Open(
                        "Select split-ADT WDT file",
                        pickFolder: false,
                        initialPath: initDir,
                        filterExtension: ".wdt",
                        picked =>
                        {
                            if (!string.IsNullOrWhiteSpace(picked))
                            {
                                _mapConvertSourcePath = picked;
                                EnsureMapConverterProjectOutputDirectory(forceNew: false);
                            }
                         });
                }

                ImGui.Text("Split ADT Directory (containing MapName_X_Y.adt roots):");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##l2a_mapdir", ref _mapConvertLkMapDir, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##l2a_dir"))
                {
                    ImGuiPathPicker.Instance.Open(
                        "Select directory containing split ADT files",
                        pickFolder: true,
                        initialPath: _mapConvertLkMapDir,
                        filterExtension: null,
                        picked =>
                        {
                            if (!string.IsNullOrWhiteSpace(picked))
                                _mapConvertLkMapDir = picked;
                         });
                }

                if (_mapConvertTargetFormat == MapConversionTargetFormat.LkAdtV18)
                {
                    ImGui.Text("Split ADT client root (required for the split-to-LK command):");
                    ImGui.SetNextItemWidth(-80);
                    ImGui.InputText("##split_client_root", ref _mapConvertSplitClientRoot, 512);
                    ImGui.SameLine();
                    if (ImGui.Button("Browse##split_client_root"))
                    {
                        ImGuiPathPicker.Instance.Open(
                            "Select split ADT client root",
                            pickFolder: true,
                            initialPath: _mapConvertSplitClientRoot,
                            filterExtension: null,
                            picked =>
                            {
                                if (!string.IsNullOrWhiteSpace(picked))
                                    _mapConvertSplitClientRoot = picked;
                            });
                    }

                    ImGui.Text("Split ADT map name:");
                    ImGui.SetNextItemWidth(-1);
                    ImGui.InputText("##split_map_name", ref _mapConvertSplitMapName, 256);
                }

                if (_mapConvertTargetFormat == MapConversionTargetFormat.AlphaWdt053)
                    ImGui.TextDisabled("Target output: Alpha 0.5.3 monolithic WDT; split companions are flattened by the Alpha converter.");
            }

            ImGui.Spacing();
            ImGui.Text("Project Output Root:");
            ImGui.SetNextItemWidth(-80);
            if (ImGui.InputText("##mapconv_project_root", ref _projectOutputRootDir, 512))
                HandleProjectOutputRootChanged();
            ImGui.SameLine();
            if (ImGui.Button("Browse##mapconv_project_root"))
            {
                ImGuiPathPicker.Instance.Open(
                    "Select project output root",
                    pickFolder: true,
                    initialPath: GetProjectOutputRootDirectory(),
                    filterExtension: null,
                    picked =>
                    {
                        if (!string.IsNullOrWhiteSpace(picked))
                        {
                            _projectOutputRootDir = picked;
                            HandleProjectOutputRootChanged();
                        }
                    });
            }

            ImGui.TextWrapped($"Project Folder: {DescribeMapConverterProjectOutputDirectory()}");
            if (ImGui.Button("New Project Folder##mapconv"))
                EnsureMapConverterProjectOutputDirectory(forceNew: true);

            ImGui.Spacing();
            ImGui.Checkbox("Verbose logging", ref _mapConvertVerbose);
            ImGui.Spacing();

            if (_mapConvertDirection == 0 && _mapConvertTargetFormat == MapConversionTargetFormat.LkAdtV18)
            {
                ImGui.Text("Area Crosswalk CSV (optional):");
                ImGui.SetNextItemWidth(-80);
                ImGui.InputText("##area_crosswalk", ref _areaCrosswalkPath, 512);
                ImGui.SameLine();
                if (ImGui.Button("Browse##area_crosswalk"))
                {
                    ImGuiPathPicker.Instance.Open(
                        "Select area crosswalk CSV",
                        pickFolder: false,
                        initialPath: null,
                        filterExtension: ".csv",
                        picked =>
                        {
                            if (!string.IsNullOrWhiteSpace(picked))
                                _areaCrosswalkPath = picked;
                        });
                }
                ImGui.SameLine();
                ImGui.TextDisabled("Maps area IDs for Alpha→LK conversion");
            }

            if (_mapConvertDirection == 1 && !string.IsNullOrEmpty(_mapConvertSourcePath))
            {
                if (string.IsNullOrEmpty(_mapConvertLkMapDir))
                    _mapConvertLkMapDir = Path.GetDirectoryName(_mapConvertSourcePath) ?? "";
                if (string.IsNullOrEmpty(_mapConvertSplitMapName))
                    _mapConvertSplitMapName = Path.GetFileNameWithoutExtension(_mapConvertSourcePath);
                if (string.IsNullOrEmpty(_mapConvertSplitClientRoot))
                    _mapConvertSplitClientRoot = TryInferMapConverterClientRoot(_mapConvertSourcePath) ?? "";
            }

            // Convert button
            bool hasSourceInput = !string.IsNullOrWhiteSpace(_mapConvertSourcePath);
            bool hasTargetInputs = _mapConvertDirection == 0
                || (_mapConvertTargetFormat == MapConversionTargetFormat.AlphaWdt053
                    ? !string.IsNullOrWhiteSpace(_mapConvertLkMapDir)
                    : !string.IsNullOrWhiteSpace(_mapConvertSplitClientRoot)
                        && !string.IsNullOrWhiteSpace(_mapConvertSplitMapName));
            bool canConvert = !_mapConverting
                && formatValidation.IsSupported
                && hasSourceInput
                && !string.IsNullOrWhiteSpace(_mapConvertOutputDir)
                && hasTargetInputs;

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
                string splitClientRoot = _mapConvertSplitClientRoot;
                string splitMapName = _mapConvertSplitMapName;
                int direction = _mapConvertDirection;
                MapConversionTargetFormat targetFormat = _mapConvertTargetFormat;
                bool verbose = _mapConvertVerbose;
                bool copyAlphaSourceWdt = _mapConvertCopyAlphaSourceWdt;
                string areaCrosswalkPath = _areaCrosswalkPath;
                _mapConvertLastLoadPath = null;

                Task.Run(async () =>
                {
                    try
                    {
                        if (direction == 0)
                        {
                            string alphaCopyPath = BuildMapConverterAlphaSourceCopyPath(outPath, srcPath);

                            if (copyAlphaSourceWdt)
                            {
                                Directory.CreateDirectory(Path.GetDirectoryName(alphaCopyPath)!);
                                File.Copy(srcPath, alphaCopyPath, overwrite: true);
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"Copied Alpha source WDT: {alphaCopyPath}");
                                _mapConvertScrollToBottom = true;
                                _mapConvertLastLoadPath = alphaCopyPath;
                            }

                            if (targetFormat == MapConversionTargetFormat.AlphaWdt053)
                            {
                                string alphaMapName = Path.GetFileNameWithoutExtension(srcPath);
                                string alphaOutputPath = BuildMapConverterAlphaOutputPath(outPath, alphaMapName);
                                Directory.CreateDirectory(Path.GetDirectoryName(alphaOutputPath)!);
                                File.Copy(srcPath, alphaOutputPath, overwrite: true);
                                _mapConvertLastLoadPath = alphaOutputPath;
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== SUCCESS: Alpha 0.5.3 identity output written to {alphaOutputPath} ===");
                            }
                            else if (targetFormat == MapConversionTargetFormat.LkAdtV18)
                            {
                                string sourceMapName = Path.GetFileNameWithoutExtension(srcPath);
                                string lkOutputDir = BuildMapConverterLkOutputDirectory(outPath, sourceMapName);
                                string lkLoadPath = Path.Combine(lkOutputDir, $"{sourceMapName}.wdt");
                                string? converterExe = FindConverterExecutable();
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
                                    if (!string.IsNullOrWhiteSpace(areaCrosswalkPath))
                                    {
                                        args.Add("--area-crosswalk");
                                        args.Add(areaCrosswalkPath);
                                    }

                                    ConverterResult result = await RunConverterAsync(converterExe, args, _mapConvertLog, _mapConvertScrollToBottom);
                                    if (!result.Success)
                                    {
                                        _mapConvertError = result.Error ?? "Conversion failed";
                                    }
                                    else
                                    {
                                        _mapConvertLastLoadPath = lkLoadPath;
                                        lock (_mapConvertLog)
                                            _mapConvertLog.Add($"\n=== SUCCESS: {result.TilesConverted}/{result.TotalTiles} tiles converted to {MapConversionFormats.GetDisplayName(targetFormat)} in {result.ElapsedMs}ms ===");
                                        lock (_mapConvertLog)
                                            _mapConvertLog.Add($"Project outputs: alpha-source={(copyAlphaSourceWdt ? alphaCopyPath : "skipped")}, lk-v18={lkOutputDir}");
                                    }
                                }
                            }
                            else
                            {
                                _mapConvertError = MapConversionFormats.GetUnavailableReason(targetFormat);
                            }
                        }
                        else
                        {
                            string? converterExe = FindConverterExecutable();
                            if (string.IsNullOrEmpty(converterExe))
                            {
                                _mapConvertError = "Converter executable not found. Build the project first.";
                                lock (_mapConvertLog)
                                    _mapConvertLog.Add($"\n=== ERROR: {_mapConvertError} ===");
                            }
                            else if (targetFormat == MapConversionTargetFormat.AlphaWdt053)
                            {
                                var args = new List<string>
                                {
                                    "convert-lk-to-alpha",
                                    "--input", lkMapDir,
                                    "--output", BuildMapConverterAlphaOutputPath(outPath, splitMapName)
                                };
                                if (verbose) args.Add("--verbose");

                                ConverterResult result = await RunConverterAsync(converterExe, args, _mapConvertLog, _mapConvertScrollToBottom);
                                if (!result.Success)
                                {
                                    _mapConvertError = result.Error ?? "Conversion failed";
                                }
                                else
                                {
                                    _mapConvertLastLoadPath = BuildMapConverterAlphaOutputPath(outPath, splitMapName);
                                    lock (_mapConvertLog)
                                        _mapConvertLog.Add($"\n=== SUCCESS: {result.TilesConverted}/{result.TotalTiles} tiles converted to {MapConversionFormats.GetDisplayName(targetFormat)} in {result.ElapsedMs}ms ===");
                                    lock (_mapConvertLog)
                                        _mapConvertLog.Add($"Project alpha-0.5.3 output: {_mapConvertLastLoadPath}");
                                }
                            }
                            else if (targetFormat == MapConversionTargetFormat.LkAdtV18)
                            {
                                string lkOutputDir = BuildMapConverterLkOutputDirectory(outPath, splitMapName);
                                string lkLoadPath = Path.Combine(lkOutputDir, $"{splitMapName}.wdt");
                                var args = new List<string>
                                {
                                    "convert-split-adt-to-lk",
                                    "--client-root", splitClientRoot,
                                    "--map", splitMapName,
                                    "--output-dir", lkOutputDir
                                };
                                if (!string.IsNullOrWhiteSpace(lkMapDir))
                                {
                                    args.Add("--overlay-root");
                                    args.Add(lkMapDir);
                                }
                                if (verbose) args.Add("--verbose");

                                ConverterResult result = await RunConverterAsync(converterExe, args, _mapConvertLog, _mapConvertScrollToBottom);
                                if (!result.Success)
                                {
                                    _mapConvertError = result.Error ?? "Conversion failed";
                                }
                                else
                                {
                                    _mapConvertLastLoadPath = lkLoadPath;
                                    lock (_mapConvertLog)
                                        _mapConvertLog.Add($"\n=== SUCCESS: {result.TilesConverted}/{result.TotalTiles} tiles converted to {MapConversionFormats.GetDisplayName(targetFormat)} in {result.ElapsedMs}ms ===");
                                    lock (_mapConvertLog)
                                        _mapConvertLog.Add($"Project LK v18 output: {lkOutputDir}");
                                }
                            }
                            else
                            {
                                _mapConvertError = MapConversionFormats.GetUnavailableReason(targetFormat);
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

            // Load result button. The target, not the source direction, determines the label and
            // the format that the viewer will consume.
            if (_mapConvertDone && _mapConvertError == null && !string.IsNullOrWhiteSpace(_mapConvertLastLoadPath))
            {
                string loadLabel = _mapConvertTargetFormat == MapConversionTargetFormat.AlphaWdt053
                    ? "Load Converted Alpha 0.5.3 WDT in Viewer"
                    : _mapConvertTargetFormat == MapConversionTargetFormat.LkAdtV18
                        ? "Load Converted LK v18 WDT in Viewer"
                        : "Load Converted Map in Viewer";
                if (ImGui.Button(loadLabel))
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

    internal void DrawWmoConverterDialog()
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
                ImGuiPathPicker.Instance.Open(
                    "Select WMO file",
                    pickFolder: false,
                    initialPath: initDir,
                    filterExtension: ".wmo",
                    picked =>
                    {
                        if (!string.IsNullOrWhiteSpace(picked))
                        {
                            _wmoConvertSourcePath = picked;
                            if (string.IsNullOrWhiteSpace(_wmoConvertOutputPath))
                                _wmoConvertOutputPath = GetDefaultWmoConverterOutputDirectory();
                        }
                    });
            }

            if (string.IsNullOrWhiteSpace(_wmoConvertOutputPath))
                _wmoConvertOutputPath = GetDefaultWmoConverterOutputDirectory();

            ImGui.Text("Output Folder:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##wmo_out_dir", ref _wmoConvertOutputPath, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##wmo_out_dir"))
            {
                ImGuiPathPicker.Instance.Open(
                    "Select output directory for converted WMO files",
                    pickFolder: true,
                    initialPath: GetDefaultWmoConverterOutputDirectory(),
                    filterExtension: null,
                    picked =>
                    {
                        if (!string.IsNullOrWhiteSpace(picked))
                            _wmoConvertOutputPath = picked;
                    });
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

    private string BuildMapConverterProjectSourceKey()
    {
        string fullSourcePath = Path.GetFullPath(_mapConvertSourcePath);
        return $"map-convert:{_mapConvertDirection}:{_mapConvertTargetFormat}:{fullSourcePath}";
    }

    internal string EnsureMapConverterProjectOutputDirectory(bool forceNew)
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
        string targetSegment = SanitizeProjectPathSegment(
            MapConversionFormats.GetCommandValue(_mapConvertTargetFormat));
        _mapConvertOutputDir = CreateTimestampedProjectOutputDirectory(
            GetProjectOutputRootDirectory(),
            $"{Path.GetFileNameWithoutExtension(_mapConvertSourcePath)}-{targetSegment}");
        return _mapConvertOutputDir;
    }

    private string DescribeMapConverterProjectOutputDirectory()
    {
        if (!string.IsNullOrWhiteSpace(_mapConvertOutputDir))
            return _mapConvertOutputDir;

        string projectName = string.IsNullOrWhiteSpace(_mapConvertSourcePath)
            ? "map-conversion"
            : Path.GetFileNameWithoutExtension(_mapConvertSourcePath);
        string targetSegment = SanitizeProjectPathSegment(
            MapConversionFormats.GetCommandValue(_mapConvertTargetFormat));
        return Path.Combine(
            GetProjectOutputRootDirectory(),
            SanitizeProjectPathSegment($"{projectName}-{targetSegment}"),
            "<timestamp>");
    }

    private static string BuildMapConverterAlphaSourceCopyPath(string projectOutputDir, string sourceWdtPath)
    {
        string mapName = Path.GetFileNameWithoutExtension(sourceWdtPath);
        return Path.Combine(projectOutputDir, "alpha-source", "World", "Maps", mapName, Path.GetFileName(sourceWdtPath));
    }

    private static string BuildMapConverterLkOutputDirectory(string projectOutputDir, string mapName)
    {
        return Path.Combine(projectOutputDir, "lk-v18", "World", "Maps", mapName);
    }

    private static string BuildMapConverterAlphaOutputPath(string projectOutputDir, string mapName)
    {
        return Path.Combine(projectOutputDir, "alpha-0.5.3", "World", "Maps", mapName, $"{mapName}.wdt");
    }

    private static string? TryInferMapConverterClientRoot(string sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
            return null;

        string fullPath;
        try
        {
            fullPath = Path.GetFullPath(sourcePath);
        }
        catch
        {
            return null;
        }

        string normalized = fullPath.Replace('/', '\\');
        int worldMapsIndex = normalized.IndexOf("\\World\\Maps\\", StringComparison.OrdinalIgnoreCase);
        if (worldMapsIndex > 0)
        {
            string candidate = normalized[..worldMapsIndex];
            if (Directory.Exists(candidate))
                return candidate;
        }

        DirectoryInfo? directory = new(
            Directory.Exists(fullPath) ? fullPath : Path.GetDirectoryName(fullPath) ?? fullPath);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "Data"))
                || Directory.EnumerateFiles(directory.FullName, "*.mpq", SearchOption.TopDirectoryOnly).Any())
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
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
