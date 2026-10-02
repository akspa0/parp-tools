using System.Diagnostics;
using System.Globalization;
using WowViewer.Core.Runtime.PromoVideo;
using static WoWViewer.ViewerApp;
using static WoWViewer.CaptureAutomationService;

namespace WoWViewer;

/// <summary>
/// Startup automation: command-line driven map loads, captures and validation batches.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class StartupAutomationService
{
    private readonly IViewerAppHost _host;

    internal StartupAutomationService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see StartupAutomationService.Host.cs.

    private sealed class StartupAutomationRequest
    {
        public string? GamePath { get; init; }
        public string? ListfilePath { get; init; }
        public string? BuildVersion { get; init; }
        public string? LooseMapOverlayPath { get; init; }
        public string? WorldPath { get; init; }
        public int? CharacterHairVariationId { get; init; }
        public int? CharacterFacialHairVariationId { get; init; }
        public string? CaptureShotName { get; init; }
        public string? CaptureOutputDir { get; init; }
        public int CaptureAfterFrames { get; init; }
        public bool CaptureIncludeUi { get; init; }
        public bool ExitAfterCapture { get; init; }
        public string? ValidationDatasetRoot { get; init; }
        public string? ValidationOutputDir { get; init; }
        public int ValidationResolution { get; init; }
        public bool ForceValidationRegeneration { get; init; }
        public bool ExitAfterValidation { get; init; }
        public int ValidationSettledFrames { get; init; }
        public int ValidationMaxFramesBeforeCapture { get; init; }
        public int ValidationBatchSettledFrames { get; init; }
        public string? RoofCaptureOutputDir { get; init; }
        public string? RoofCaptureAssetListPath { get; init; }
        public int RoofCaptureResolution { get; init; }
        public bool RoofCaptureAllAngles { get; init; }
        public int? RecordTaxiRouteId { get; init; }
        public string? RecordCameraPathName { get; init; }
        public double? RecordDurationSeconds { get; init; }
        public string? RecordOutputPath { get; init; }
        public int? RecordFps { get; init; }
        public bool? RecordIncludeUi { get; init; }
        public bool RecordFeatureTour { get; init; }
        public bool ExitAfterRecord { get; init; }
        public string? RecordTaxiPlaylist { get; init; }
        public string? RecordTaxiChain { get; init; }
        public bool ShowreelHud { get; init; }
        public bool? ShowreelZoneBanners { get; init; }
        public bool? ShowreelTelemetry { get; init; }
        public bool? ShowreelLandmarks { get; init; }
        public bool? ShowreelEngineBadges { get; init; }
        public bool? ShowreelPerf { get; init; }
        public bool? ShowreelPipeline { get; init; }
        public bool? ShowreelHitches { get; init; }
        public bool? ShowreelExpanded { get; init; }
        public float? ShowreelHitchThreshold { get; init; }
    }

    internal sealed class PendingRoofCaptureBatch
    {
        public required List<string> AssetPaths { get; init; }
        public required string OutputDir { get; init; }
        public int Resolution { get; init; } = 512;
        public bool AllAngles { get; init; }
        public bool ExitAfterCompletion { get; init; }
        public int CurrentIndex { get; set; }
        public int SuccessCount { get; set; }
        public Catalog.ScreenshotRenderer? Renderer { get; set; }
        public List<Dictionary<string, object>> Metadata { get; set; } = new();
    }

    internal PendingRoofCaptureBatch? _pendingRoofCaptureBatch;

    internal void ApplyStartupAutomation(string[]? initialArgs)
    {
        StartupAutomationRequest request = ParseStartupAutomationRequest(initialArgs, out string? legacyPath);

        if (!string.IsNullOrWhiteSpace(request.GamePath))
        {
            if (!Directory.Exists(request.GamePath))
            {
                _statusMessage = $"Startup game path does not exist: {request.GamePath}";
                return;
            }

            _dataSourceSession.LoadMpqDataSource(request.GamePath, request.ListfilePath, request.BuildVersion);
        }

        if (!string.IsNullOrWhiteSpace(request.LooseMapOverlayPath))
        {
            if (!Directory.Exists(request.LooseMapOverlayPath))
            {
                _statusMessage = $"Startup loose overlay path does not exist: {request.LooseMapOverlayPath}";
                return;
            }

            _dataSourceSession.AttachLooseMapOverlay(request.LooseMapOverlayPath);
        }

        string? startupTarget = request.WorldPath;
        if (string.IsNullOrWhiteSpace(startupTarget))
            startupTarget = legacyPath;

        _modelLoader.PrepareStandaloneCharacterCustomizationForNextLoad(request.CharacterHairVariationId, request.CharacterFacialHairVariationId);

        if (!string.IsNullOrWhiteSpace(startupTarget))
            LoadStartupTarget(startupTarget);

        if (!string.IsNullOrWhiteSpace(request.CaptureOutputDir))
            _captureOutputDir = Path.GetFullPath(request.CaptureOutputDir);

        if (!string.IsNullOrWhiteSpace(request.CaptureShotName))
            QueueNamedStartupCapture(request.CaptureShotName, request.CaptureIncludeUi, request.ExitAfterCapture, request.CaptureAfterFrames);

        if (!string.IsNullOrWhiteSpace(request.ValidationDatasetRoot))
            QueueStartupValidationCaptureBatch(request);

        if (!string.IsNullOrWhiteSpace(request.RoofCaptureOutputDir))
            QueueStartupRoofCapture(request);

        if (request.ShowreelHud)
        {
            _host.ShowreelOverlay.Config.EnableOverlay = true;
            if (request.ShowreelZoneBanners.HasValue)
                _host.ShowreelOverlay.Config.ShowZoneBanners = request.ShowreelZoneBanners.Value;
            if (request.ShowreelTelemetry.HasValue)
                _host.ShowreelOverlay.Config.ShowLiveTelemetry = request.ShowreelTelemetry.Value;
            if (request.ShowreelLandmarks.HasValue)
                _host.ShowreelOverlay.Config.ShowLandmarkCallouts = request.ShowreelLandmarks.Value;
            if (request.ShowreelEngineBadges.HasValue)
                _host.ShowreelOverlay.Config.ShowEngineBadges = request.ShowreelEngineBadges.Value;
            if (request.ShowreelPerf.HasValue)
                _host.ShowreelOverlay.Config.ShowPerformanceTelemetry = request.ShowreelPerf.Value;
            if (request.ShowreelPipeline.HasValue)
                _host.ShowreelOverlay.Config.ShowPipelineTelemetry = request.ShowreelPipeline.Value;
            if (request.ShowreelHitches.HasValue)
                _host.ShowreelOverlay.Config.ShowHitchAlerts = request.ShowreelHitches.Value;
            if (request.ShowreelExpanded.HasValue)
                _host.ShowreelOverlay.Config.ExpandedDiagnostics = request.ShowreelExpanded.Value;
            if (request.ShowreelHitchThreshold.HasValue)
                _host.ShowreelOverlay.Config.HitchThresholdMs = request.ShowreelHitchThreshold.Value;
        }

        if (request.RecordTaxiRouteId.HasValue ||
            !string.IsNullOrWhiteSpace(request.RecordCameraPathName) ||
            !string.IsNullOrWhiteSpace(request.RecordTaxiPlaylist) ||
            !string.IsNullOrWhiteSpace(request.RecordTaxiChain))
            QueueStartupVideoRecording(request);
    }

    private StartupAutomationRequest ParseStartupAutomationRequest(string[]? initialArgs, out string? legacyPath)
    {
        legacyPath = null;
        if (initialArgs == null || initialArgs.Length == 0)
            return new StartupAutomationRequest();

        string? gamePath = null;
        string? listfilePath = null;
        string? buildVersion = null;
        string? looseMapOverlayPath = null;
        string? worldPath = null;
        string? characterHairVariation = null;
        string? characterFacialHairVariation = null;
        string? captureShotName = null;
        string? captureOutputDir = null;
        string? captureAfterFrames = null;
        string? validationDatasetRoot = null;
        string? validationOutputDir = null;
        string? validationResolution = null;
        string? validationSettledFrames = null;
        string? validationMaxFrames = null;
        string? validationBatchSettledFrames = null;
        bool captureIncludeUi = false;
        bool exitAfterCapture = false;
        bool forceValidationRegeneration = false;
        bool exitAfterValidation = false;
        string? roofCaptureOutputDir = null;
        string? roofCaptureAssetListPath = null;
        string? roofCaptureResolution = null;
        bool roofCaptureAllAngles = false;
        string? recordTaxiRoute = null;
        string? recordCameraPath = null;
        string? recordDuration = null;
        string? recordOutput = null;
        string? recordFps = null;
        bool? recordIncludeUi = null;
        bool recordFeatureTour = false;
        bool exitAfterRecord = false;
        string? recordTaxiPlaylist = null;
        string? recordTaxiChain = null;
        bool showreelHud = false;
        bool? showreelZoneBanners = null;
        bool? showreelTelemetry = null;
        bool? showreelLandmarks = null;
        bool? showreelEngineBadges = null;
        bool? showreelPerf = null;
        bool? showreelPipeline = null;
        bool? showreelHitches = null;
        bool? showreelExpanded = null;
        float? showreelHitchThreshold = null;

        for (int index = 0; index < initialArgs.Length; index++)
        {
            string arg = initialArgs[index];
            switch (arg.ToLowerInvariant())
            {
                case "--game-path":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out gamePath))
                        return new StartupAutomationRequest();
                    break;

                case "--build":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out buildVersion))
                        return new StartupAutomationRequest();
                    break;

                case "--listfile":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out listfilePath))
                        return new StartupAutomationRequest();
                    break;

                case "--loose-map-overlay":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out looseMapOverlayPath))
                        return new StartupAutomationRequest();
                    break;

                case "--world":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out worldPath))
                        return new StartupAutomationRequest();
                    break;

                case "--character-hair-variation":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out characterHairVariation))
                        return new StartupAutomationRequest();
                    break;

                case "--character-facial-variation":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out characterFacialHairVariation))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-shot":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out captureShotName))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-output":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out captureOutputDir))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-after-frames":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out captureAfterFrames))
                        return new StartupAutomationRequest();
                    break;

                case "--validation-dataset-root":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out validationDatasetRoot))
                        return new StartupAutomationRequest();
                    break;

                case "--validation-output":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out validationOutputDir))
                        return new StartupAutomationRequest();
                    break;

                case "--validation-resolution":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out validationResolution))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-with-ui":
                    captureIncludeUi = true;
                    break;

                case "--capture-no-ui":
                    captureIncludeUi = false;
                    break;

                case "--exit-after-capture":
                    exitAfterCapture = true;
                    break;

                case "--force-validation-regeneration":
                    forceValidationRegeneration = true;
                    break;

                case "--exit-after-validation":
                    exitAfterValidation = true;
                    break;

                case "--validation-settled-frames":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out validationSettledFrames))
                        return new StartupAutomationRequest();
                    break;

                case "--validation-max-frames":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out validationMaxFrames))
                        return new StartupAutomationRequest();
                    break;

                case "--validation-batch-settled-frames":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out validationBatchSettledFrames))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-roof":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out roofCaptureOutputDir))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-roof-asset-list":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out roofCaptureAssetListPath))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-roof-resolution":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out roofCaptureResolution))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-roof-all-angles":
                    roofCaptureAllAngles = true;
                    break;

                case "--record-taxi-route":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out recordTaxiRoute))
                        return new StartupAutomationRequest();
                    break;

                case "--record-camera-path":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out recordCameraPath))
                        return new StartupAutomationRequest();
                    break;

                case "--record-duration":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out recordDuration))
                        return new StartupAutomationRequest();
                    break;

                case "--record-output":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out recordOutput))
                        return new StartupAutomationRequest();
                    break;

                case "--record-fps":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out recordFps))
                        return new StartupAutomationRequest();
                    break;

                case "--record-with-ui":
                    recordIncludeUi = true;
                    break;

                case "--record-no-ui":
                    recordIncludeUi = false;
                    break;

                case "--record-feature-tour":
                    recordFeatureTour = true;
                    break;

                case "--exit-after-record":
                    exitAfterRecord = true;
                    break;

                case "--record-taxi-playlist":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out recordTaxiPlaylist))
                        return new StartupAutomationRequest();
                    break;

                case "--record-taxi-chain":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out recordTaxiChain))
                        return new StartupAutomationRequest();
                    break;

                case "--showreel-hud":
                    showreelHud = true;
                    break;

                case "--showreel-zone-banners":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valBanners) && bool.TryParse(valBanners, out bool bBanners))
                        showreelZoneBanners = bBanners;
                    else
                        showreelZoneBanners = true;
                    break;

                case "--showreel-telemetry":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valTelem) && bool.TryParse(valTelem, out bool bTelem))
                        showreelTelemetry = bTelem;
                    else
                        showreelTelemetry = true;
                    break;

                case "--showreel-landmarks":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valLand) && bool.TryParse(valLand, out bool bLand))
                        showreelLandmarks = bLand;
                    else
                        showreelLandmarks = true;
                    break;

                case "--showreel-engine-badges":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valBadges) && bool.TryParse(valBadges, out bool bBadges))
                        showreelEngineBadges = bBadges;
                    else
                        showreelEngineBadges = true;
                    break;

                case "--showreel-perf":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valPerf) && bool.TryParse(valPerf, out bool bPerf))
                        showreelPerf = bPerf;
                    else
                        showreelPerf = true;
                    break;

                case "--showreel-pipeline":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valPipe) && bool.TryParse(valPipe, out bool bPipe))
                        showreelPipeline = bPipe;
                    else
                        showreelPipeline = true;
                    break;

                case "--showreel-hitches":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valHitch) && bool.TryParse(valHitch, out bool bHitch))
                        showreelHitches = bHitch;
                    else
                        showreelHitches = true;
                    break;

                case "--showreel-expanded":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valExp) && bool.TryParse(valExp, out bool bExp))
                        showreelExpanded = bExp;
                    else
                        showreelExpanded = true;
                    break;

                case "--showreel-hitch-threshold":
                    if (TryReadStartupOptionValue(initialArgs, ref index, arg, out string? valThresh) && float.TryParse(valThresh, NumberStyles.Float, CultureInfo.InvariantCulture, out float fThresh))
                        showreelHitchThreshold = fThresh;
                    break;

                default:
                    if (arg.StartsWith("--", StringComparison.Ordinal))
                    {
                        _statusMessage = $"Unknown startup option: {arg}";
                        return new StartupAutomationRequest();
                    }

                    legacyPath ??= arg;
                    break;
            }
        }

        if (!TryParseOptionalVariationId(characterHairVariation, "--character-hair-variation", out int? characterHairVariationId))
            return new StartupAutomationRequest();

        if (!TryParseOptionalVariationId(characterFacialHairVariation, "--character-facial-variation", out int? characterFacialHairVariationId))
            return new StartupAutomationRequest();

        if (!TryParseOptionalPositiveInt(captureAfterFrames, "--capture-after-frames", out int resolvedCaptureAfterFrames))
            return new StartupAutomationRequest();

        if (!TryParseOptionalPositiveInt(validationResolution, "--validation-resolution", out int resolvedValidationResolution))
            return new StartupAutomationRequest();

        if (!TryParseOptionalPositiveInt(validationSettledFrames, "--validation-settled-frames", out int resolvedValidationSettledFrames))
            return new StartupAutomationRequest();
        if (validationSettledFrames == null)
            resolvedValidationSettledFrames = DefaultRequiredSettledFrames;

        if (!TryParseOptionalPositiveInt(validationMaxFrames, "--validation-max-frames", out int resolvedValidationMaxFrames))
            return new StartupAutomationRequest();
        if (validationMaxFrames == null)
            resolvedValidationMaxFrames = DefaultMaxFramesBeforeCapture;

        if (!TryParseOptionalPositiveInt(validationBatchSettledFrames, "--validation-batch-settled-frames", out int resolvedValidationBatchSettledFrames))
            return new StartupAutomationRequest();
        if (validationBatchSettledFrames == null)
            resolvedValidationBatchSettledFrames = DefaultBatchSettledFrames;

        int? resolvedRecordTaxiRouteId = null;
        if (!string.IsNullOrWhiteSpace(recordTaxiRoute) && int.TryParse(recordTaxiRoute, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsedRouteId))
            resolvedRecordTaxiRouteId = parsedRouteId;

        double? resolvedRecordDuration = null;
        if (!string.IsNullOrWhiteSpace(recordDuration) && double.TryParse(recordDuration, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsedDuration) && parsedDuration > 0)
            resolvedRecordDuration = parsedDuration;

        int? resolvedRecordFps = null;
        if (!string.IsNullOrWhiteSpace(recordFps) && int.TryParse(recordFps, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsedFps) && parsedFps >= 12 && parsedFps <= 60)
            resolvedRecordFps = parsedFps;

        return new StartupAutomationRequest
        {
            GamePath = NormalizeOptionalPath(gamePath),
            ListfilePath = NormalizeOptionalPath(listfilePath),
            BuildVersion = NormalizeOptionalValue(buildVersion),
            LooseMapOverlayPath = NormalizeOptionalPath(looseMapOverlayPath),
            WorldPath = NormalizeOptionalValue(worldPath),
            CharacterHairVariationId = characterHairVariationId,
            CharacterFacialHairVariationId = characterFacialHairVariationId,
            CaptureShotName = NormalizeOptionalValue(captureShotName),
            CaptureOutputDir = NormalizeOptionalPath(captureOutputDir),
            CaptureAfterFrames = resolvedCaptureAfterFrames,
            CaptureIncludeUi = captureIncludeUi,
            ExitAfterCapture = exitAfterCapture,
            ValidationDatasetRoot = NormalizeOptionalPath(validationDatasetRoot),
            ValidationOutputDir = NormalizeOptionalPath(validationOutputDir),
            ValidationResolution = resolvedValidationResolution,
            ForceValidationRegeneration = forceValidationRegeneration,
            ExitAfterValidation = exitAfterValidation,
            ValidationSettledFrames = resolvedValidationSettledFrames,
            ValidationMaxFramesBeforeCapture = resolvedValidationMaxFrames,
            ValidationBatchSettledFrames = resolvedValidationBatchSettledFrames,
            RoofCaptureOutputDir = NormalizeOptionalPath(roofCaptureOutputDir),
            RoofCaptureAssetListPath = NormalizeOptionalValue(roofCaptureAssetListPath),
            RoofCaptureResolution = TryParseOptionalPositiveInt(roofCaptureResolution, "--capture-roof-resolution", out int resolvedRoofRes) ? resolvedRoofRes : 512,
            RoofCaptureAllAngles = roofCaptureAllAngles,
            RecordTaxiRouteId = resolvedRecordTaxiRouteId,
            RecordCameraPathName = NormalizeOptionalValue(recordCameraPath),
            RecordDurationSeconds = resolvedRecordDuration,
            RecordOutputPath = NormalizeOptionalPath(recordOutput),
            RecordFps = resolvedRecordFps,
            RecordIncludeUi = recordIncludeUi,
            RecordFeatureTour = recordFeatureTour,
            ExitAfterRecord = exitAfterRecord,
            RecordTaxiPlaylist = NormalizeOptionalValue(recordTaxiPlaylist),
            RecordTaxiChain = NormalizeOptionalValue(recordTaxiChain),
            ShowreelHud = showreelHud,
            ShowreelZoneBanners = showreelZoneBanners,
            ShowreelTelemetry = showreelTelemetry,
            ShowreelLandmarks = showreelLandmarks,
            ShowreelEngineBadges = showreelEngineBadges,
            ShowreelPerf = showreelPerf,
            ShowreelPipeline = showreelPipeline,
            ShowreelHitches = showreelHitches,
            ShowreelExpanded = showreelExpanded,
            ShowreelHitchThreshold = showreelHitchThreshold,
        };
    }

    private void QueueStartupValidationCaptureBatch(StartupAutomationRequest request)
    {
        MkHarvestViewerValidationCapturePlan? plan = _datasetExportDialogs.BuildMkHarvestViewerValidationCapturePlan(
            request.ValidationDatasetRoot!,
            request.ValidationOutputDir,
            request.ForceValidationRegeneration,
            request.ValidationResolution,
            out string? statusMessage,
            request.ValidationSettledFrames,
            request.ValidationMaxFramesBeforeCapture,
            request.ValidationBatchSettledFrames);

        if (!string.IsNullOrWhiteSpace(statusMessage))
            _datasetExportDialogs.AppendMkHarvestLogLine(statusMessage);

        if (plan == null)
            return;

        plan.ExitAfterCompletion = request.ExitAfterValidation;

        if (plan.Tiles.Count == 0)
        {
            StitchMkHarvestViewerValidationOutputs(
                plan.MapName,
                plan.OutputDirectory,
                plan.NoLiquidsOutputDirectory,
                plan.NoObjectsOutputDirectory,
                plan.ObjectsOnlyOutputDirectory,
                plan.RequestedResolution);
            GenerateMkHarvestViewerValidationObjectArtifacts(
                plan.DatasetRoot,
                plan.OutputDirectory,
                plan.NoObjectsOutputDirectory,
                plan.ObjectsOnlyOutputDirectory);

            if (request.ExitAfterValidation)
                _window.Close();

            return;
        }

        _pendingMkHarvestViewerValidationCapturePlan = plan;
        _mkHarvestViewerValidationQueued = plan.Tiles.Count;
        _datasetExportDialogs.AppendMkHarvestLogLine(
            $"Queued startup validation capture batch: {plan.Tiles.Count} capture(s) at {plan.RequestedResolution}px into {plan.OutputDirectory}.");
    }

    private bool TryReadStartupOptionValue(string[] args, ref int index, string optionName, out string? value)
    {
        if (index + 1 >= args.Length)
        {
            value = null;
            _statusMessage = $"Missing value for startup option {optionName}.";
            return false;
        }

        index++;
        value = args[index];
        return true;
    }

    private void LoadStartupTarget(string startupTarget)
    {
        if (File.Exists(startupTarget))
        {
            _modelLoader.LoadFileFromDisk(Path.GetFullPath(startupTarget));
            return;
        }

        if (_dataSource != null)
        {
            _modelLoader.LoadFileFromDataSource(startupTarget);
            return;
        }

        _statusMessage = $"Startup world/file path was not found on disk and no data source is loaded: {startupTarget}";
    }

    private void QueueNamedStartupCapture(string shotName, bool includeUi, bool exitAfterCapture, int captureAfterFrames = 1)
    {
        if (string.Equals(shotName, "current", StringComparison.OrdinalIgnoreCase))
        {
            QueueCurrentCameraCapture(includeUi, exitAfterCapture, captureAfterFrames, allowWindowCloseOnCapture: exitAfterCapture);
            return;
        }

        CameraShotPoint? shot = _cameraShotPoints.FirstOrDefault(candidate =>
            string.Equals(candidate.Name, shotName, StringComparison.OrdinalIgnoreCase));

        if (shot == null)
        {
            _statusMessage = $"Startup capture shot was not found: {shotName}";
            return;
        }

        EnqueueShotCapture(
            shot,
            includeUi,
            exitAfterCapture,
            new CaptureQueueOptions
            {
                WaitForSceneReady = captureAfterFrames > 1,
                RequiredSettledFrames = captureAfterFrames > 1 ? captureAfterFrames : 1,
                MaxFramesBeforeCapture = captureAfterFrames > 1 ? captureAfterFrames : 1,
                AllowWindowCloseOnCapture = exitAfterCapture,
            });
    }

private void QueueStartupRoofCapture(StartupAutomationRequest request)
    {
        if (_dataSource == null)
        {
            _statusMessage = "Cannot start roof capture: no data source loaded";
            return;
        }

        string outputDir = request.RoofCaptureOutputDir!;
        int resolution = request.RoofCaptureResolution > 0 ? request.RoofCaptureResolution : 512;

        List<string> assetPaths;
        string? assetListPath = request.RoofCaptureAssetListPath;
        _datasetExportDialogs.AppendMkHarvestLogLine($"[RoofCapture] Asset list path = '{assetListPath}'");
        if (!string.IsNullOrWhiteSpace(assetListPath) && File.Exists(assetListPath))
        {
            _datasetExportDialogs.AppendMkHarvestLogLine($"[RoofCapture] Reading asset list from {assetListPath}");
            string json = File.ReadAllText(assetListPath);
            var list = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.List<string>>(json);
            assetPaths = list ?? new List<string>();
            _datasetExportDialogs.AppendMkHarvestLogLine($"[RoofCapture] Found {assetPaths.Count} assets");
        }
        else
        {
            _statusMessage = $"Roof capture requires --capture-roof-asset-list pointing to a JSON array of asset paths. Tried: '{assetListPath}' exists={File.Exists(assetListPath ?? "")}";
            return;
        }

        _statusMessage = $"Queued roof batch capture: {assetPaths.Count} assets -> {outputDir}";
        _datasetExportDialogs.AppendMkHarvestLogLine(_statusMessage);

        _pendingRoofCaptureBatch = new PendingRoofCaptureBatch
        {
            AssetPaths = assetPaths,
            OutputDir = outputDir,
            Resolution = resolution,
            AllAngles = request.RoofCaptureAllAngles,
            ExitAfterCompletion = request.ExitAfterValidation,
        };
}

    private bool TryParseOptionalPositiveInt(string? rawValue, string optionName, out int parsedValue)
    {
        parsedValue = 1;
        string? normalized = NormalizeOptionalValue(rawValue);
        if (normalized == null)
            return true;

        if (!int.TryParse(normalized, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsedValue) || parsedValue < 1)
        {
            _statusMessage = $"Startup option {optionName} expects an integer >= 1. Received '{rawValue}'.";
            return false;
        }

        return true;
    }

    private static string? NormalizeOptionalPath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return Path.GetFullPath(value);
    }

    private static string? NormalizeOptionalValue(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return value.Trim();
    }

    private bool TryParseOptionalVariationId(string? rawValue, string optionName, out int? variationId)
    {
        variationId = null;
        if (string.IsNullOrWhiteSpace(rawValue))
            return true;

        if (int.TryParse(rawValue, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) && parsed >= 0)
        {
            variationId = parsed;
            return true;
        }

        _statusMessage = $"Invalid variation id for {optionName}: {rawValue}";
        return false;
    }

    private void QueueStartupVideoRecording(StartupAutomationRequest request)
    {
        if (!string.IsNullOrWhiteSpace(request.RecordTaxiPlaylist))
        {
            string[] parts = request.RecordTaxiPlaylist.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            foreach (string part in parts)
            {
                if (int.TryParse(part, out int rid))
                    _host.TaxiPlaylist.AddRoute(rid);
            }

            if (_host.TaxiPlaylist.Items.Count > 0)
            {
                _host.TaxiPlaylist.StartPlaylist(
                    recordVideo: true,
                    videoFps: request.RecordFps ?? 30,
                    includeUi: request.RecordIncludeUi ?? true, // Default to true so showreel HUD captures
                    customOutput: request.RecordOutputPath,
                    exitAfterRecord: request.ExitAfterRecord);
            }
            return;
        }

        if (!string.IsNullOrWhiteSpace(request.RecordTaxiChain))
        {
            string[] parts = request.RecordTaxiChain.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length > 0 && int.TryParse(parts[0], out int startNode))
            {
                int hops = parts.Length > 1 && int.TryParse(parts[1], out int h) ? h : 4;
                _host.TaxiPlaylist.BuildAutoChain(startNode, hops);

                if (_host.TaxiPlaylist.Items.Count > 0)
                {
                    _host.TaxiPlaylist.StartPlaylist(
                        recordVideo: true,
                        videoFps: request.RecordFps ?? 30,
                        includeUi: request.RecordIncludeUi ?? true,
                        customOutput: request.RecordOutputPath,
                        exitAfterRecord: request.ExitAfterRecord);
                }
            }
            return;
        }

        if (request.RecordTaxiRouteId is int routeId)
        {
            if (_worldScene == null)
            {
                _statusMessage = "Cannot record taxi route: no world scene is loaded.";
                return;
            }

            _worldScene.TaxiActors.ShowTaxi = true;
            _worldScene.TaxiActors.ShowTaxiActors = true;
            _worldScene.TaxiActors.SelectedTaxiRouteId = routeId;
            _worldScene.TaxiActors.ActiveTaxiRideRouteId = routeId;
            _host.TaxiRideCameraRouteId = routeId;
            _host.TaxiRideCameraScene = _worldScene;
            _host.TaxiRideCameraEnabled = true;
            _host.TaxiRideCameraPoseInitialized = false;
            _host.LastTaxiRideCameraTick = Stopwatch.GetTimestamp();

            _worldScene.TaxiActors.ResetTaxiRouteTravel(routeId);

            FeatureTourRecipe? tourRecipe = null;
            string routeLabel = _taxiAndAreaPoi.GetTaxiRouteDisplayLabel(routeId);
            if (request.RecordFeatureTour)
            {
                var route = _worldScene.TaxiActors.GetTaxiRoute(routeId);
                string? fromStation = route != null ? _worldScene.TaxiActors.GetTaxiNode(route.FromNodeId)?.Name : null;
                string? toStation = route != null ? _worldScene.TaxiActors.GetTaxiNode(route.ToNodeId)?.Name : null;
                tourRecipe = BuiltinFeatureTourRecipes.CreateTaxiRouteOverview(routeId, routeLabel, fromStation, toStation, request.RecordFps ?? 30);
            }

            var recRequest = new RecordingRequest
            {
                SourceKind = RecordingSourceKind.TaxiRoute,
                TaxiRouteId = routeId,
                Label = routeLabel,
                OutputPathOverride = request.RecordOutputPath,
                IncludeUi = request.RecordIncludeUi ?? false,
                Fps = request.RecordFps ?? 30,
                MaxDurationSeconds = request.RecordDurationSeconds,
                AutoStopOnRouteArrival = !request.RecordDurationSeconds.HasValue,
                ExitAfterRecording = request.ExitAfterRecord,
                TourRecipe = tourRecipe,
                IncludeShowreelOverlay = request.RecordFeatureTour || request.ShowreelHud || _host.ShowreelOverlay.Config.EnableOverlay,
            };

            _recordingCoordinator.TryStartRecording(recRequest, out string? error);
            if (!string.IsNullOrWhiteSpace(error))
            {
                _statusMessage = $"Failed to start automated taxi recording: {error}";
            }
            return;
        }

        if (!string.IsNullOrWhiteSpace(request.RecordCameraPathName))
        {
            string pathName = request.RecordCameraPathName;
            FeatureTourRecipe? tourRecipe = null;
            if (request.RecordFeatureTour)
            {
                tourRecipe = BuiltinFeatureTourRecipes.CreateCameraPathOverview(pathName, request.RecordFps ?? 30);
            }

            var recRequest = new RecordingRequest
            {
                SourceKind = RecordingSourceKind.CameraPath,
                CameraPathName = pathName,
                Label = pathName,
                OutputPathOverride = request.RecordOutputPath,
                IncludeUi = request.RecordIncludeUi ?? false,
                Fps = request.RecordFps ?? 30,
                MaxDurationSeconds = request.RecordDurationSeconds,
                ExitAfterRecording = request.ExitAfterRecord,
                TourRecipe = tourRecipe,
                IncludeShowreelOverlay = request.RecordFeatureTour || request.ShowreelHud || _host.ShowreelOverlay.Config.EnableOverlay,
            };

            _recordingCoordinator.TryStartRecording(recRequest, out string? error);
            if (!string.IsNullOrWhiteSpace(error))
            {
                _statusMessage = $"Failed to start automated camera path recording: {error}";
            }
        }
    }
}
