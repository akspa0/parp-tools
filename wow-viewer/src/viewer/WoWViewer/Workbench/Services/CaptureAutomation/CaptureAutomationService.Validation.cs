using System.Numerics;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using ImGuiNET;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Capture;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.Marketing;
using WoWViewer.Terrain.Vlm;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// CaptureAutomationService: MK-harvest viewer-validation and roof capture batches, capture timing metadata, object-visibility masks and stitching.
// CaptureAutomationService: members moved from ViewerApp_CaptureAutomation.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class CaptureAutomationService
{

    internal void PromotePendingMkHarvestViewerValidationCapturePlan()
    {
        if (_pendingMkHarvestViewerValidationCapturePlan == null || _activeMkHarvestViewerValidationBatch != null)
            return;

        MkHarvestViewerValidationCapturePlan plan = _pendingMkHarvestViewerValidationCapturePlan;
        if (_terrainManager == null)
        {
            if (!plan.RestoreWorldRequested && _dataSourceSession.HasWorldReturnTarget())
            {
                string returnMapName = Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath!);
                if (string.Equals(returnMapName, plan.MapName, StringComparison.OrdinalIgnoreCase))
                {
                    plan.RestoreWorldRequested = true;
                    _datasetExportDialogs.AppendMkHarvestLogLine($"Restoring world '{plan.MapName}' before running the WoWViewer validation capture batch.");
                    _dataSourceSession.ReturnToLastWorldScene();
                    return;
                }
            }

            _datasetExportDialogs.AppendMkHarvestLogLine($"Skipping WoWViewer validation captures because world map '{plan.MapName}' is not currently loaded in the viewer.");
            _pendingMkHarvestViewerValidationCapturePlan = null;
            return;
        }

        if (!string.Equals(_terrainManager.MapName, plan.MapName, StringComparison.OrdinalIgnoreCase))
        {
            _datasetExportDialogs.AppendMkHarvestLogLine(
                $"Skipping WoWViewer validation captures because the current world '{_terrainManager.MapName}' does not match dataset map '{plan.MapName}'.");
            _pendingMkHarvestViewerValidationCapturePlan = null;
            return;
        }

        StartMkHarvestViewerValidationBatch(plan);
        _pendingMkHarvestViewerValidationCapturePlan = null;
    }

    internal void PromotePendingRoofCaptureBatch()
    {
        if (_startupAutomation._pendingRoofCaptureBatch == null)
            return;

        if (_gl == null)
        {
            _statusMessage = "Cannot run roof capture: GL context not ready";
            _datasetExportDialogs.AppendMkHarvestLogLine(_statusMessage);
            _startupAutomation._pendingRoofCaptureBatch = null;
            return;
        }

        var batch = _startupAutomation._pendingRoofCaptureBatch;

        // Initialize on first call
        if (batch.Renderer == null)
        {
            Directory.CreateDirectory(batch.OutputDir);
            batch.Renderer = new Catalog.ScreenshotRenderer(_gl, _dataSource, _texResolver, _dbcBuild);
            _statusMessage = $"Starting roof batch capture: {batch.AssetPaths.Count} assets -> {batch.OutputDir}";
            _datasetExportDialogs.AppendMkHarvestLogLine(_statusMessage);
        }

        // Process one asset per frame
        if (batch.CurrentIndex >= batch.AssetPaths.Count)
        {
            _statusMessage = $"Roof capture complete: {batch.SuccessCount}/{batch.AssetPaths.Count} succeeded -> {batch.OutputDir}";
            _datasetExportDialogs.AppendMkHarvestLogLine(_statusMessage);

            // Write metadata
            string metaPath = Path.Combine(batch.OutputDir, "roof_capture_metadata.json");
            File.WriteAllText(metaPath,
                System.Text.Json.JsonSerializer.Serialize(new { captures = batch.Metadata },
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

            batch.Renderer.Dispose();
            _startupAutomation._pendingRoofCaptureBatch = null;

            if (batch.ExitAfterCompletion)
                _window.Close();
            return;
        }

        string assetPath = batch.AssetPaths[batch.CurrentIndex];
        string safeName = SanitizeRoofCaptureName(Path.GetFileNameWithoutExtension(assetPath.Replace('/', '\\')));
        string assetDir = Path.Combine(batch.OutputDir, safeName);

        // Resume: skip if existing successful metadata
        string existingMeta = Path.Combine(assetDir, "metadata.json");
        bool alreadyDone = File.Exists(existingMeta);
        if (alreadyDone)
        {
            try
            {
                string metaText = File.ReadAllText(existingMeta);
                var meta = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.Dictionary<string, object>>(metaText);
                alreadyDone = meta != null && meta.TryGetValue("success", out var s) && (s is bool b && b || s is string str && str == "True");
            }
            catch { alreadyDone = false; }
        }

        if (alreadyDone)
        {
            batch.SuccessCount++;
            _statusMessage = $"[RoofCapture] {batch.CurrentIndex + 1}/{batch.AssetPaths.Count} SKIP (existing) {assetPath}";
            _datasetExportDialogs.AppendMkHarvestLogLine(_statusMessage);
            batch.CurrentIndex++;
            return;
        }

        Directory.CreateDirectory(assetDir);

        _statusMessage = $"[RoofCapture] {batch.CurrentIndex + 1}/{batch.AssetPaths.Count} {assetPath}";
        _datasetExportDialogs.AppendMkHarvestLogLine(_statusMessage);

        string? result;
        if (batch.AllAngles)
            result = batch.Renderer.CaptureAllAnglesByPath(assetPath, assetDir, batch.Resolution, batch.Resolution);
        else
            result = batch.Renderer.CapturePathByExtension(assetPath, assetDir, batch.Resolution, batch.Resolution);

        if (result != null)
        {
            batch.SuccessCount++;
            var angleNames = batch.AllAngles
                ? Catalog.ScreenshotRenderer.CameraAngles.Select(a => new Dictionary<string, object>
                {
                    ["name"] = a.name,
                    ["azimuth"] = a.azimuth,
                    ["elevation"] = a.elevation,
                    ["file"] = $"{a.name}.jpg"
                }).ToList()
                : null;

            var entry = new Dictionary<string, object>
            {
                ["asset_path"] = assetPath,
                ["asset_stem"] = Path.GetFileNameWithoutExtension(assetPath.Replace('/', '\\')),
                ["success"] = true,
                ["output_dir"] = assetDir,
                ["resolution"] = batch.Resolution,
                ["build"] = _dbcBuild ?? "",
                ["capture_mode"] = batch.AllAngles ? "all_angles" : "roof_only",
                ["roof_topdown"] = "roof_topdown.jpg",
                ["angles"] = angleNames,
            };

            // Write per-asset HuggingFace-style metadata
            string assetMeta = Path.Combine(assetDir, "metadata.json");
            File.WriteAllText(assetMeta,
                System.Text.Json.JsonSerializer.Serialize(entry,
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

            batch.Metadata.Add(entry);
        }
        else
        {
            var entry = new Dictionary<string, object>
            {
                ["asset_path"] = assetPath,
                ["asset_stem"] = Path.GetFileNameWithoutExtension(assetPath.Replace('/', '\\')),
                ["success"] = false
            };
            string assetMeta = Path.Combine(assetDir, "metadata.json");
            File.WriteAllText(assetMeta,
                System.Text.Json.JsonSerializer.Serialize(entry,
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            batch.Metadata.Add(entry);
        }

        // Write dataset config on first asset
        if (batch.CurrentIndex == 0 && batch.SuccessCount > 0)
        {
            var config = new Dictionary<string, object>
            {
                ["dataset_name"] = "wmo-roof-capture",
                ["build"] = _dbcBuild ?? "",
                ["source"] = "ParpToolsWoWViewer roof capture",
                ["resolution"] = batch.Resolution,
                ["capture_mode"] = batch.AllAngles ? "all_angles" : "roof_only",
                ["total_assets"] = batch.AssetPaths.Count,
                ["camera_angles"] = batch.AllAngles
                    ? Catalog.ScreenshotRenderer.CameraAngles.Select(a => new Dictionary<string, object>
                    {
                        ["name"] = a.name,
                        ["azimuth"] = a.azimuth,
                        ["elevation"] = a.elevation
                    }).ToList()
                    : null,
                ["background"] = "black",
                ["alpha_channel"] = "background_transparent",
                ["format"] = "jpg",
                ["jpeg_quality"] = 99,
            };
            string configPath = Path.Combine(batch.OutputDir, "dataset_config.json");
            File.WriteAllText(configPath,
                System.Text.Json.JsonSerializer.Serialize(config,
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        }
        batch.CurrentIndex++;
    }

    private static string SanitizeRoofCaptureName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new System.Text.StringBuilder(name.Length);
        foreach (char c in name)
        {
            if (c == ' ' || c == '/' || c == '\\') sb.Append('_');
            else if (Array.IndexOf(invalid, c) < 0 && c != '\0') sb.Append(c);
        }
        string result = sb.ToString();
        return result.Length > 100 ? result[..100] : result;
    }

    private void StartMkHarvestViewerValidationBatch(MkHarvestViewerValidationCapturePlan plan)
    {
        if (_terrainManager == null)
            return;

        int requestedResolution = Math.Clamp(plan.RequestedResolution, 512, 4096);
        TerrainLighting terrainLighting = _terrainManager.Lighting;
        TerrainLighting? vlmLighting = _vlmTerrainManager?.Lighting;
        _activeMkHarvestViewerValidationBatch = new ActiveMkHarvestViewerValidationBatch
        {
            PreviousWindowSize = _window.Size,
            PreviousHideUiChrome = _hideUiChrome,
            PreviousDetailedTileCountOverride = _terrainManager.DetailedTileCountOverride,
            PreviousFogStart = terrainLighting.FogStart,
            PreviousFogEnd = terrainLighting.FogEnd,
            PreviousTerrainLightDirectionOverride = terrainLighting.HasExternalLightDirectionOverride,
            PreviousTerrainLightDirection = terrainLighting.ExternalLightDirection,
            PreviousVlmLightDirectionOverride = vlmLighting?.HasExternalLightDirectionOverride ?? false,
            PreviousVlmLightDirection = vlmLighting?.ExternalLightDirection ?? Vector3.Zero,
            PreviousTerrainLiquidsVisible = _terrainManager.LiquidRenderer?.ShowLiquid ?? true,
            PreviousVlmTerrainLiquidsVisible = _vlmTerrainManager?.LiquidRenderer?.ShowLiquid ?? true,
            PreviousTerrainVisible = _terrainManager.TerrainVisible,
            PreviousVlmTerrainVisible = _vlmTerrainManager?.TerrainVisible ?? true,
            PreviousObjectFogEnabled = _worldScene?.ObjectFogEnabled ?? true,
            PreviousShowWdlTerrain = _worldScene?.ShowWdlTerrain ?? true,
            PreviousShowSky = _worldScene?.ShowSky ?? true,
            PreviousObjectsVisible = _worldScene?.ObjectsVisible ?? true,
            PreviousWmosVisible = _worldScene?.WmosVisible ?? true,
            PreviousDoodadsVisible = _worldScene?.DoodadsVisible ?? true,
            PreviousWlLiquidsVisible = _worldScene?.ShowWlLiquids ?? true,
            PreviousIgnoreTerrainHolesGlobally = _terrainManager.IgnoreTerrainHolesGlobally,
            PreviousIgnoreVlmTerrainHolesGlobally = _vlmTerrainManager?.IgnoreTerrainHolesGlobally ?? false,
            PreviousObjectPathFiltersEnabled = _worldScene?.ObjectPathFiltersEnabled ?? true,
            PreviousObjectStreamingRangeMultiplier = _worldScene?.ObjectStreamingRangeMultiplier ?? 0.5f,
            PreviousMaxVisibleMdxBoundsHeight = _worldScene?.MaxVisibleMdxBoundsHeight ?? 0f,
            PreviousHideTerrainOccludedMdx = _worldScene?.HideTerrainOccludedMdx ?? false,
            PreviousEnableRuntimeWmoGroupVisibility = _worldScene?.EnableRuntimeWmoGroupVisibility ?? true,
            PreviousEnableRuntimeWmoGroupLiquids = _worldScene?.EnableRuntimeWmoGroupLiquids ?? true,
            DatasetRoot = plan.DatasetRoot,
            MapName = plan.MapName,
            OutputDirectory = plan.OutputDirectory,
            NoLiquidsOutputDirectory = plan.NoLiquidsOutputDirectory,
            NoObjectsOutputDirectory = plan.NoObjectsOutputDirectory,
            ObjectsOnlyOutputDirectory = plan.ObjectsOnlyOutputDirectory,
            RequestedResolution = requestedResolution,
            ExitAfterCompletion = plan.ExitAfterCompletion,
            RemainingCaptures = plan.Tiles.Count,
            BatchHasSettled = false,
            RequiredSettledFrames = plan.RequiredSettledFrames,
            MaxFramesBeforeCapture = plan.MaxFramesBeforeCapture,
            BatchSettledFrames = plan.BatchSettledFrames,
            FastSettleAfterBatchReady = plan.FastSettleAfterBatchReady,
        };

        _hideUiChrome = true;
        _window.Size = new Vector2D<int>(requestedResolution, requestedResolution);
        _terrainManager.DetailedTileCountOverride = Math.Min(25, TerrainManager.MaxManualDetailedTileCount);
        _terrainManager.IgnoreTerrainHolesGlobally = true;
        if (_vlmTerrainManager != null)
            _vlmTerrainManager.IgnoreTerrainHolesGlobally = true;
        terrainLighting.FogStart = MaxTerrainFogDistance * 0.75f;
        terrainLighting.FogEnd = MaxTerrainFogDistance;
        Vector3 validationLightDirection = BuildMkHarvestViewerValidationLightDirection(terrainLighting.LightDirection);
        terrainLighting.ApplyExternalLightDirection(validationLightDirection);
        vlmLighting?.ApplyExternalLightDirection(validationLightDirection);
        if (_worldScene != null)
        {
            _worldScene.ObjectFogEnabled = false;
            _worldScene.ObjectsVisible = true;
            _worldScene.WmosVisible = true;
            _worldScene.DoodadsVisible = false;
            _worldScene.ShowWlLiquids = false;
            _worldScene.ObjectPathFiltersEnabled = false;
            _worldScene.ObjectStreamingRangeMultiplier = Math.Max(_worldScene.ObjectStreamingRangeMultiplier, 1.0f);
            _worldScene.MaxVisibleMdxBoundsHeight = MkHarvestViewerValidationMaxVisibleMdxBoundsHeight;
            _worldScene.HideTerrainOccludedMdx = true;
            _worldScene.EnableRuntimeWmoGroupVisibility = false;
            _worldScene.EnableRuntimeWmoGroupLiquids = true;
        }

        foreach (MkHarvestViewerValidationCaptureTile tile in plan.Tiles)
        {
            CameraShotPoint shot = BuildMkHarvestViewerValidationShot(plan.MapName, tile);
            EnqueueShotCapture(
                shot,
                includeUi: false,
                exitAfterCapture: false,
                new CaptureQueueOptions
                {
                    OutputPathOverride = tile.OutputPath,
                    WaitForSceneReady = true,
                    TargetTileX = tile.TileX,
                    TargetTileY = tile.TileY,
                    RequiredSettledFrames = plan.RequiredSettledFrames,
                    MaxFramesBeforeCapture = plan.MaxFramesBeforeCapture,
                    CaptureLabel = tile.HideTerrain
                        ? $"{tile.TileName} (objectsonly)"
                        : (tile.HideObjects
                            ? $"{tile.TileName} (noobjects)"
                            : (tile.HideTerrainLiquids ? $"{tile.TileName} (noliquids)" : tile.TileName)),
                    IsMkHarvestViewerValidationCapture = true,
                    HideTerrainLiquids = tile.HideTerrainLiquids,
                    HideObjects = tile.HideObjects,
                    HideTerrain = tile.HideTerrain,
                });
        }

        _datasetExportDialogs.AppendMkHarvestLogLine(
            $"Started WoWViewer validation capture batch for {plan.Tiles.Count} capture(s). Settled frames: {plan.RequiredSettledFrames} (batch-fast: {plan.BatchSettledFrames}, fast-settle enabled: {plan.FastSettleAfterBatchReady}), max frames: {plan.MaxFramesBeforeCapture}. Viewer chrome is hidden, WL liquids are disabled for all variants, object path filters are disabled, MDX objects taller than {MkHarvestViewerValidationMaxVisibleMdxBoundsHeight:F0} world units are suppressed during the batch, the primary output keeps terrain liquids and visible world objects including doodads, the 'noliquids' sub-folder disables terrain liquids, the 'noobjects' sub-folder hides world objects, the 'objectsonly' sub-folder hides terrain, WDL, liquids, and sky while keeping visible world objects, object streaming is widened, the validation sun direction is forced for deterministic top-down shading, and the window was resized to {requestedResolution}x{requestedResolution} for the batch.");
    }

    private static Vector3 BuildMkHarvestViewerValidationLightDirection(Vector3 currentLightDirection)
    {
        Vector3 source = currentLightDirection.LengthSquared() > 1e-6f
            ? Vector3.Normalize(currentLightDirection)
            : Vector3.Normalize(new Vector3(0f, 0.3f, 1f));

        return Vector3.Normalize(new Vector3(source.X, -source.Y, MathF.Abs(source.Z)));
    }

    private CameraShotPoint BuildMkHarvestViewerValidationShot(string mapName, MkHarvestViewerValidationCaptureTile tile)
    {
        const float capturePitch = -89f;
        const float captureYaw = 0f;
        const float captureFovDegrees = 24f;

        float centerX = WoWConstants.MapOrigin - ((tile.TileX + 0.5f) * WoWConstants.ChunkSize);
        float centerY = WoWConstants.MapOrigin - ((tile.TileY + 0.5f) * WoWConstants.ChunkSize);
        float targetGroundHeight = 0f;
        if (_terrainManager != null
            && _terrainQuery.TrySampleTerrainHeightLoaded(_terrainManager.Renderer, centerX, centerY, out float loadedTerrainHeight, out _))
        {
            targetGroundHeight = loadedTerrainHeight;
        }
        else if (_vlmTerrainManager != null
            && _terrainQuery.TrySampleTerrainHeightLoaded(_vlmTerrainManager.Renderer, centerX, centerY, out float loadedVlmTerrainHeight, out _))
        {
            targetGroundHeight = loadedVlmTerrainHeight;
        }

        float desiredSpan = WoWConstants.ChunkSize;
        float heightAboveGround = 256f + (desiredSpan / (2f * MathF.Tan((captureFovDegrees * MathF.PI / 180f) * 0.5f)));
        float captureHeight = targetGroundHeight + heightAboveGround;

        return new CameraShotPoint
        {
            Name = $"{tile.TileName}_viewer_validation",
            MapName = mapName,
            BuildVersion = GetCurrentCaptureBuildVersion(),
            PositionX = centerX,
            PositionY = centerY,
            PositionZ = captureHeight,
            Yaw = captureYaw,
            Pitch = capturePitch,
            FovDegrees = captureFovDegrees,
        };
    }

    internal bool TryGetMkHarvestViewerValidationSceneMatrices(float aspect, out Matrix4x4 view, out Matrix4x4 proj)
    {
        view = Matrix4x4.Identity;
        proj = Matrix4x4.Identity;

        if (_activeMkHarvestViewerValidationBatch == null
            || _activeCaptureRequest?.IsMkHarvestViewerValidationCapture != true
            || _activeCaptureRequest.TargetTileX is not int tileX
            || _activeCaptureRequest.TargetTileY is not int tileY)
        {
            return false;
        }

        float centerX = WoWConstants.MapOrigin - ((tileX + 0.5f) * WoWConstants.ChunkSize);
        float centerY = WoWConstants.MapOrigin - ((tileY + 0.5f) * WoWConstants.ChunkSize);
        float targetGroundHeight = 0f;
        if (_terrainManager != null
            && _terrainQuery.TrySampleTerrainHeightLoaded(_terrainManager.Renderer, centerX, centerY, out float loadedTerrainHeight, out _))
        {
            targetGroundHeight = loadedTerrainHeight;
        }
        else if (_vlmTerrainManager != null
            && _terrainQuery.TrySampleTerrainHeightLoaded(_vlmTerrainManager.Renderer, centerX, centerY, out float loadedVlmTerrainHeight, out _))
        {
            targetGroundHeight = loadedVlmTerrainHeight;
        }

        float worldSpanX = WoWConstants.ChunkSize * Math.Max(1f, aspect);
        float worldSpanY = WoWConstants.ChunkSize / Math.Max(1f, aspect <= 0f ? 1f : Math.Min(1f, aspect));
        if (aspect > 0f && aspect < 1f)
            worldSpanX = WoWConstants.ChunkSize;
        if (aspect >= 1f)
            worldSpanY = WoWConstants.ChunkSize;

        Vector3 eye = new(centerX, centerY, targetGroundHeight + 2048f);
        Vector3 target = new(centerX, centerY, targetGroundHeight);
        view = Matrix4x4.CreateLookAt(eye, target, Vector3.UnitX);
        proj = Matrix4x4.CreateOrthographic(worldSpanX, worldSpanY, 0.1f, _terrainQuery.GetSceneFarPlane());
        return true;
    }

    private void RestoreMkHarvestViewerValidationBatch(string? statusMessage = null)
    {
        if (_activeMkHarvestViewerValidationBatch == null)
            return;

        ActiveMkHarvestViewerValidationBatch batch = _activeMkHarvestViewerValidationBatch;
        _activeMkHarvestViewerValidationBatch = null;

        _hideUiChrome = batch.PreviousHideUiChrome;
        _window.Size = batch.PreviousWindowSize;

        if (_terrainManager != null)
        {
            _terrainManager.DetailedTileCountOverride = batch.PreviousDetailedTileCountOverride;
            _terrainManager.Lighting.FogStart = batch.PreviousFogStart;
            _terrainManager.Lighting.FogEnd = batch.PreviousFogEnd;
            _terrainManager.TerrainVisible = batch.PreviousTerrainVisible;
            _terrainManager.IgnoreTerrainHolesGlobally = batch.PreviousIgnoreTerrainHolesGlobally;
            if (_terrainManager.LiquidRenderer != null)
                _terrainManager.LiquidRenderer.ShowLiquid = batch.PreviousTerrainLiquidsVisible;
            if (batch.PreviousTerrainLightDirectionOverride)
                _terrainManager.Lighting.ApplyExternalLightDirection(batch.PreviousTerrainLightDirection);
            else
                _terrainManager.Lighting.ClearExternalLightDirection();
        }

        if (_vlmTerrainManager != null)
        {
            _vlmTerrainManager.TerrainVisible = batch.PreviousVlmTerrainVisible;
            _vlmTerrainManager.IgnoreTerrainHolesGlobally = batch.PreviousIgnoreVlmTerrainHolesGlobally;
            if (_vlmTerrainManager.LiquidRenderer != null)
                _vlmTerrainManager.LiquidRenderer.ShowLiquid = batch.PreviousVlmTerrainLiquidsVisible;
            if (batch.PreviousVlmLightDirectionOverride)
                _vlmTerrainManager.Lighting.ApplyExternalLightDirection(batch.PreviousVlmLightDirection);
            else
                _vlmTerrainManager.Lighting.ClearExternalLightDirection();
        }

        if (_worldScene != null)
        {
            _worldScene.ObjectFogEnabled = batch.PreviousObjectFogEnabled;
            _worldScene.ShowWdlTerrain = batch.PreviousShowWdlTerrain;
            _worldScene.ShowSky = batch.PreviousShowSky;
            _worldScene.ObjectsVisible = batch.PreviousObjectsVisible;
            _worldScene.WmosVisible = batch.PreviousWmosVisible;
            _worldScene.DoodadsVisible = batch.PreviousDoodadsVisible;
            _worldScene.ShowWlLiquids = batch.PreviousWlLiquidsVisible;
            _worldScene.ObjectPathFiltersEnabled = batch.PreviousObjectPathFiltersEnabled;
            _worldScene.ObjectStreamingRangeMultiplier = batch.PreviousObjectStreamingRangeMultiplier;
            _worldScene.MaxVisibleMdxBoundsHeight = batch.PreviousMaxVisibleMdxBoundsHeight;
            _worldScene.HideTerrainOccludedMdx = batch.PreviousHideTerrainOccludedMdx;
            _worldScene.EnableRuntimeWmoGroupVisibility = batch.PreviousEnableRuntimeWmoGroupVisibility;
            _worldScene.EnableRuntimeWmoGroupLiquids = batch.PreviousEnableRuntimeWmoGroupLiquids;
        }

        StitchMkHarvestViewerValidationOutputs(
            batch.MapName,
            batch.OutputDirectory,
            batch.NoLiquidsOutputDirectory,
            batch.NoObjectsOutputDirectory,
            batch.ObjectsOnlyOutputDirectory,
            batch.RequestedResolution);
        GenerateMkHarvestViewerValidationObjectArtifacts(batch.DatasetRoot, batch.OutputDirectory, batch.NoObjectsOutputDirectory, batch.ObjectsOnlyOutputDirectory);

        if (!string.IsNullOrWhiteSpace(statusMessage))
        {
            _statusMessage = statusMessage;
            _datasetExportDialogs.AppendMkHarvestLogLine(statusMessage);
        }

        if (batch.ExitAfterCompletion)
            _window.Close();
    }

    private void WriteCaptureTimingMetadata(PendingCaptureRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath))
            return;

        string? outputDir = Path.GetDirectoryName(request.OutputPath);
        if (string.IsNullOrWhiteSpace(outputDir))
            return;

        Directory.CreateDirectory(outputDir);

        string baseName = Path.GetFileNameWithoutExtension(request.OutputPath);
        string metadataPath = Path.Combine(outputDir, $"{baseName}_capture_metadata.json");

        var record = new CaptureTimingRecord
        {
            TileName = request.Shot.Name,
            BuildVersion = request.Shot.BuildVersion,
            MapName = request.Shot.MapName,
            TileX = request.TargetTileX ?? 0,
            TileY = request.TargetTileY ?? 0,
            Variant = request.CaptureLabel ?? "unknown",
            SettledFrames = request.SettledFrames,
            TotalFramesSinceApplied = request.FramesSinceApplied,
            TimedOut = request.TimedOutWaitingForScene,
            RequiredSettledFrames = request.RequiredSettledFrames,
            MaxFramesBeforeCapture = request.MaxFramesBeforeCapture,
            OutputPath = request.OutputPath,
        };

        try
        {
            File.WriteAllText(metadataPath, JsonSerializer.Serialize(record, MkDatasetJsonOptions));
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Export, $"[Capture] Failed to write timing metadata for {baseName}: {ex.Message}");
        }
    }

    internal void StitchMkHarvestViewerValidationOutputs(
        string mapName,
        string outputDirectory,
        string noLiquidsOutputDirectory,
        string noObjectsOutputDirectory,
        string objectsOnlyOutputDirectory,
        int requestedResolution)
    {
        TryStitchMkHarvestViewerValidationDirectory(outputDirectory, mapName, requestedResolution, "viewer_validation_minimaps");
        TryStitchMkHarvestViewerValidationDirectory(noLiquidsOutputDirectory, mapName, requestedResolution, "viewer_validation_minimaps/noliquids");
        TryStitchMkHarvestViewerValidationDirectory(noObjectsOutputDirectory, mapName, requestedResolution, "viewer_validation_minimaps/noobjects");
        TryStitchMkHarvestViewerValidationDirectory(objectsOnlyOutputDirectory, mapName, requestedResolution, "viewer_validation_minimaps/objectsonly");
    }

    private static readonly JsonSerializerOptions MkDatasetJsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    internal void GenerateMkHarvestViewerValidationObjectArtifacts(
        string datasetRoot,
        string withObjectsOutputDirectory,
        string noObjectsOutputDirectory,
        string objectsOnlyOutputDirectory)
    {
        if (string.IsNullOrWhiteSpace(datasetRoot)
            || string.IsNullOrWhiteSpace(withObjectsOutputDirectory)
            || string.IsNullOrWhiteSpace(noObjectsOutputDirectory)
            || !Directory.Exists(withObjectsOutputDirectory)
            || !Directory.Exists(noObjectsOutputDirectory))
        {
            return;
        }

        string datasetDirectory = Path.Combine(datasetRoot, "dataset");
        if (!Directory.Exists(datasetDirectory))
            return;

        string imagesDirectory = Path.Combine(datasetRoot, "images");
        Directory.CreateDirectory(imagesDirectory);

        string buildVersion = GetCurrentCaptureBuildVersion();
        bool preferDirectObjectsOnlyMask = ShouldPreferDirectObjectsOnlyMask(buildVersion);

        int updatedTiles = 0;
        int skippedTiles = 0;

        foreach (string jsonPath in Directory.GetFiles(datasetDirectory, "*.json"))
        {
            string fileName = Path.GetFileName(jsonPath);
            if (string.Equals(fileName, "texture_database.json", StringComparison.OrdinalIgnoreCase))
                continue;

            string tileName = Path.GetFileNameWithoutExtension(jsonPath);
            string withObjectsPath = Path.Combine(withObjectsOutputDirectory, $"{tileName}_viewer_validation.png");
            string noObjectsPath = Path.Combine(noObjectsOutputDirectory, $"{tileName}_viewer_validation.png");
            if (!File.Exists(withObjectsPath) || !File.Exists(noObjectsPath))
            {
                skippedTiles++;
                continue;
            }

            try
            {
                using Image<Rgba32> withObjectsImage = SixLabors.ImageSharp.Image.Load<Rgba32>(withObjectsPath);
                using Image<Rgba32> noObjectsImage = SixLabors.ImageSharp.Image.Load<Rgba32>(noObjectsPath);
                if (noObjectsImage.Width != withObjectsImage.Width || noObjectsImage.Height != withObjectsImage.Height)
                {
                    noObjectsImage.Mutate(ctx => ctx.Resize(withObjectsImage.Width, withObjectsImage.Height));
                }

                using Image<L8> maskImage = (preferDirectObjectsOnlyMask
                    ? TryBuildDirectObjectVisibilityMask(tileName, withObjectsImage.Width, withObjectsImage.Height, objectsOnlyOutputDirectory)
                    : null)
                    ?? BuildObjectVisibilityDiffMask(withObjectsImage, noObjectsImage);

                string objectMaskFileName = $"{tileName}_object_visibility_mask.png";
                string noObjectFileName = $"{tileName}_no_objects.png";
                string objectMaskRelativePath = $"images/{objectMaskFileName}";
                string noObjectRelativePath = $"images/{noObjectFileName}";

                string objectMaskPath = Path.Combine(imagesDirectory, objectMaskFileName);
                string noObjectOutPath = Path.Combine(imagesDirectory, noObjectFileName);

                maskImage.SaveAsPng(objectMaskPath);
                noObjectsImage.SaveAsPng(noObjectOutPath);

                VlmTrainingSample? sample = JsonSerializer.Deserialize<VlmTrainingSample>(File.ReadAllText(jsonPath), MkDatasetJsonOptions);
                if (sample?.TerrainData == null)
                {
                    skippedTiles++;
                    continue;
                }

                VlmTerrainData updatedTerrain = sample.TerrainData with
                {
                    ObjectVisibilityMaskPath = objectMaskRelativePath,
                    NoObjectMinimapPath = noObjectRelativePath,
                };

                VlmTrainingSample updatedSample = sample with { TerrainData = updatedTerrain };
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(updatedSample, MkDatasetJsonOptions));
                updatedTiles++;
            }
            catch
            {
                skippedTiles++;
            }
        }

        _datasetExportDialogs.AppendMkHarvestLogLine(
            $"Object-visibility artifacts: updated {updatedTiles} tile json(s), skipped {skippedTiles} tile(s) without matching captures. {(preferDirectObjectsOnlyMask ? "This build prefers direct object-only silhouettes so early underground object bleed-through is preserved." : "This build prefers with/no-object diffs so terrain occlusion wins over terrain-hidden silhouettes." )} Build={buildVersion}.");
    }

    private static bool ShouldPreferDirectObjectsOnlyMask(string buildVersion)
    {
        if (string.IsNullOrWhiteSpace(buildVersion))
            return false;

        int separatorIndex = buildVersion.IndexOf('.');
        string majorComponent = separatorIndex >= 0
            ? buildVersion[..separatorIndex]
            : buildVersion;

        return int.TryParse(majorComponent, out int majorVersion) && majorVersion == 0;
    }

    private static Image<L8>? TryBuildDirectObjectVisibilityMask(string tileName, int width, int height, string objectsOnlyOutputDirectory)
    {
        if (string.IsNullOrWhiteSpace(objectsOnlyOutputDirectory) || !Directory.Exists(objectsOnlyOutputDirectory))
            return null;

        string objectsOnlyPath = Path.Combine(objectsOnlyOutputDirectory, $"{tileName}_viewer_validation.png");
        if (!File.Exists(objectsOnlyPath))
            return null;

        using Image<Rgba32> objectsOnlyImage = SixLabors.ImageSharp.Image.Load<Rgba32>(objectsOnlyPath);
        if (objectsOnlyImage.Width != width || objectsOnlyImage.Height != height)
            objectsOnlyImage.Mutate(ctx => ctx.Resize(width, height));

        return BuildObjectVisibilityMaskFromObjectsOnlyCapture(objectsOnlyImage);
    }

    private static Image<L8> BuildObjectVisibilityMaskFromObjectsOnlyCapture(Image<Rgba32> objectsOnly)
    {
        const int intensityThreshold = 4;
        var mask = new Image<L8>(objectsOnly.Width, objectsOnly.Height);

        for (int y = 0; y < objectsOnly.Height; y++)
        {
            for (int x = 0; x < objectsOnly.Width; x++)
            {
                Rgba32 pixel = objectsOnly[x, y];
                int intensity = Math.Max(pixel.R, Math.Max(pixel.G, pixel.B));
                mask[x, y] = new L8((byte)(intensity > intensityThreshold ? 255 : 0));
            }
        }

        return mask;
    }

    private static Image<L8> BuildObjectVisibilityDiffMask(Image<Rgba32> withObjects, Image<Rgba32> noObjects)
    {
        const int diffThreshold = 8;
        var mask = new Image<L8>(withObjects.Width, withObjects.Height);

        for (int y = 0; y < withObjects.Height; y++)
        {
            for (int x = 0; x < withObjects.Width; x++)
            {
                Rgba32 withPixel = withObjects[x, y];
                Rgba32 noPixel = noObjects[x, y];
                int diffR = Math.Abs(withPixel.R - noPixel.R);
                int diffG = Math.Abs(withPixel.G - noPixel.G);
                int diffB = Math.Abs(withPixel.B - noPixel.B);
                int diff = Math.Max(diffR, Math.Max(diffG, diffB));
                mask[x, y] = new L8((byte)(diff >= diffThreshold ? 255 : 0));
            }
        }

        return mask;
    }

    private void TryStitchMkHarvestViewerValidationDirectory(string imagesDirectory, string mapName, int requestedResolution, string variantLabel)
    {
        if (string.IsNullOrWhiteSpace(imagesDirectory) || string.IsNullOrWhiteSpace(mapName) || !Directory.Exists(imagesDirectory))
            return;

        string stitchedDirectory = Path.Combine(imagesDirectory, "stitched");
        Directory.CreateDirectory(stitchedDirectory);

        string outputPath = Path.Combine(stitchedDirectory, $"{mapName}_full_viewer_validation.png");
        var bounds = TileStitchingService.StitchFullMap(
            imagesDirectory,
            mapName,
            requestedResolution,
            outputPath,
            suffix: "_viewer_validation.png");

        if (bounds.HasValue)
        {
            _datasetExportDialogs.AppendMkHarvestLogLine(
                $"Stitched {variantLabel} into {outputPath} using tile bounds {bounds.Value.minX:D2},{bounds.Value.minY:D2} -> {bounds.Value.maxX:D2},{bounds.Value.maxY:D2}.");
        }
    }
}
