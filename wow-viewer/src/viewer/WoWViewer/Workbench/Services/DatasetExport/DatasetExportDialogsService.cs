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

namespace WoWViewer;

/// <summary>
/// VLM dataset export, MK harvest (with viewer-validation capture plan), ML finalize and terrain texture-transfer dialogs and jobs.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class DatasetExportDialogsService
{
    private readonly IViewerAppHost _host;

    internal DatasetExportDialogsService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref string _mkHarvestDatasetRoot => ref _host.MkHarvestDatasetRoot;
    private ref int _mkHarvestViewerValidationCompleted => ref _host.MkHarvestViewerValidationCompleted;
    private ref int _mkHarvestViewerValidationFailed => ref _host.MkHarvestViewerValidationFailed;
    private ref int _mkHarvestViewerValidationQueued => ref _host.MkHarvestViewerValidationQueued;
    private ref MkHarvestViewerValidationCapturePlan? _pendingMkHarvestViewerValidationCapturePlan => ref _host.PendingMkHarvestViewerValidationCapturePlan;
    private ref bool _showTerrainTextureTransferDialog => ref _host.ShowTerrainTextureTransferDialog;
    private ref bool _showVlmExportDialog => ref _host.ShowVlmExportDialog;
    private ref string _terrainTransferOutputDir => ref _host.TerrainTransferOutputDir;
    private ref string _terrainTransferSourceDir => ref _host.TerrainTransferSourceDir;
    private ref string _terrainTransferTargetDir => ref _host.TerrainTransferTargetDir;
    private ref string _vlmClientPath => ref _host.VlmClientPath;
    private ref string _vlmMapName => ref _host.VlmMapName;
    private ref string _vlmOutputDir => ref _host.VlmOutputDir;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private void LoadVlmProject(string projectRoot) => _host.LoadVlmProject(projectRoot);
    private void StitchMkHarvestViewerValidationOutputs(string mapName, string outputDirectory, string noLiquidsOutputDirectory, string noObjectsOutputDirectory, string objectsOnlyOutputDirectory, int requestedResolution) => _host.StitchMkHarvestViewerValidationOutputs(mapName, outputDirectory, noLiquidsOutputDirectory, noObjectsOutputDirectory, objectsOnlyOutputDirectory, requestedResolution);

    private int _vlmTileLimit = 0; // 0 = unlimited
    private bool _vlmExporting = false;
    private readonly List<string> _vlmExportLog = new();
    private bool _vlmExportScrollToBottom = false;
    private VlmExportResult? _vlmExportResult = null;
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

    internal void DrawVlmExportDialog()
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
                ImGuiPathPicker.Instance.Open(
                    "Select WoW Client Data Folder",
                    pickFolder: true,
                    initialPath: _vlmClientPath,
                    filterExtension: null,
                    result =>
                    {
                        if (!string.IsNullOrWhiteSpace(result))
                            _vlmClientPath = result;
                    });
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
                ImGuiPathPicker.Instance.Open(
                    "Select Output Directory",
                    pickFolder: true,
                    initialPath: _vlmOutputDir,
                    filterExtension: null,
                    result =>
                    {
                        if (!string.IsNullOrWhiteSpace(result))
                            _vlmOutputDir = result;
                    });
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

    internal void PromotePendingMlFinalizeAfterExport()
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

    internal void DrawTerrainTextureTransferDialog()
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
                ImGuiPathPicker.Instance.Open(
                    "Select source map directory",
                    pickFolder: true,
                    initialPath: _terrainTransferSourceDir,
                    filterExtension: null,
                    picked =>
                    {
                        if (!string.IsNullOrWhiteSpace(picked))
                            _terrainTransferSourceDir = picked;
                    });
            }

            ImGui.Text("Target Map Directory:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##ttt_target", ref _terrainTransferTargetDir, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##ttt_target"))
            {
                ImGuiPathPicker.Instance.Open(
                    "Select target map directory",
                    pickFolder: true,
                    initialPath: _terrainTransferTargetDir,
                    filterExtension: null,
                    picked =>
                    {
                        if (!string.IsNullOrWhiteSpace(picked))
                            _terrainTransferTargetDir = picked;
                    });
            }

            ImGui.Text("Output Directory:");
            ImGui.SetNextItemWidth(-80);
            ImGui.InputText("##ttt_output", ref _terrainTransferOutputDir, 512);
            ImGui.SameLine();
            if (ImGui.Button("Browse##ttt_output"))
            {
                ImGuiPathPicker.Instance.Open(
                    "Select output directory",
                    pickFolder: true,
                    initialPath: _terrainTransferOutputDir,
                    filterExtension: null,
                    picked =>
                    {
                        if (!string.IsNullOrWhiteSpace(picked))
                            _terrainTransferOutputDir = picked;
                    });
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
    internal static string GenerateVlmOutputPath(string clientPath, string mapName)
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

    internal void AppendMkHarvestLogLine(string message)
    {
        lock (_mkHarvestLog)
        {
            _mkHarvestLog.Add(message);
            if (_mkHarvestLog.Count > 2000)
                _mkHarvestLog.RemoveRange(0, _mkHarvestLog.Count - 1500);
        }

        _mkHarvestScrollToBottom = true;
    }

    internal MkHarvestViewerValidationCapturePlan? BuildMkHarvestViewerValidationCapturePlan(
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
}
