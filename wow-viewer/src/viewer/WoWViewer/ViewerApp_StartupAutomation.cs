using System.Globalization;

namespace WoWViewer;

public partial class ViewerApp
{
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
    }

    private sealed class PendingRoofCaptureBatch
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

    private PendingRoofCaptureBatch? _pendingRoofCaptureBatch;

    private void ApplyStartupAutomation(string[]? initialArgs)
    {
        StartupAutomationRequest request = ParseStartupAutomationRequest(initialArgs, out string? legacyPath);

        if (!string.IsNullOrWhiteSpace(request.GamePath))
        {
            if (!Directory.Exists(request.GamePath))
            {
                _statusMessage = $"Startup game path does not exist: {request.GamePath}";
                return;
            }

            LoadMpqDataSource(request.GamePath, request.ListfilePath, request.BuildVersion);
        }

        if (!string.IsNullOrWhiteSpace(request.LooseMapOverlayPath))
        {
            if (!Directory.Exists(request.LooseMapOverlayPath))
            {
                _statusMessage = $"Startup loose overlay path does not exist: {request.LooseMapOverlayPath}";
                return;
            }

            AttachLooseMapOverlay(request.LooseMapOverlayPath);
        }

        string? startupTarget = request.WorldPath;
        if (string.IsNullOrWhiteSpace(startupTarget))
            startupTarget = legacyPath;

        PrepareStandaloneCharacterCustomizationForNextLoad(request.CharacterHairVariationId, request.CharacterFacialHairVariationId);

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
        };
    }

    private void QueueStartupValidationCaptureBatch(StartupAutomationRequest request)
    {
        MkHarvestViewerValidationCapturePlan? plan = BuildMkHarvestViewerValidationCapturePlan(
            request.ValidationDatasetRoot!,
            request.ValidationOutputDir,
            request.ForceValidationRegeneration,
            request.ValidationResolution,
            out string? statusMessage,
            request.ValidationSettledFrames,
            request.ValidationMaxFramesBeforeCapture,
            request.ValidationBatchSettledFrames);

        if (!string.IsNullOrWhiteSpace(statusMessage))
            AppendMkHarvestLogLine(statusMessage);

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
        AppendMkHarvestLogLine(
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
            LoadFileFromDisk(Path.GetFullPath(startupTarget));
            return;
        }

        if (_dataSource != null)
        {
            LoadFileFromDataSource(startupTarget);
            return;
        }

        _statusMessage = $"Startup world/file path was not found on disk and no data source is loaded: {startupTarget}";
    }

    private void QueueNamedStartupCapture(string shotName, bool includeUi, bool exitAfterCapture, int captureAfterFrames = 1)
    {
        if (string.Equals(shotName, "current", StringComparison.OrdinalIgnoreCase))
        {
            QueueCurrentCameraCapture(includeUi, exitAfterCapture, captureAfterFrames);
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
            captureAfterFrames > 1
                ? new CaptureQueueOptions
                {
                    WaitForSceneReady = true,
                    RequiredSettledFrames = captureAfterFrames,
                    MaxFramesBeforeCapture = captureAfterFrames,
                }
                : null);
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
        AppendMkHarvestLogLine($"[RoofCapture] Asset list path = '{assetListPath}'");
        if (!string.IsNullOrWhiteSpace(assetListPath) && File.Exists(assetListPath))
        {
            AppendMkHarvestLogLine($"[RoofCapture] Reading asset list from {assetListPath}");
            string json = File.ReadAllText(assetListPath);
            var list = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.List<string>>(json);
            assetPaths = list ?? new List<string>();
            AppendMkHarvestLogLine($"[RoofCapture] Found {assetPaths.Count} assets");
        }
        else
        {
            _statusMessage = $"Roof capture requires --capture-roof-asset-list pointing to a JSON array of asset paths. Tried: '{assetListPath}' exists={File.Exists(assetListPath ?? "")}";
            return;
        }

        _statusMessage = $"Queued roof batch capture: {assetPaths.Count} assets -> {outputDir}";
        AppendMkHarvestLogLine(_statusMessage);

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
}
