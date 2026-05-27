using WowViewer.Core.Renderer.Validation;
using WowViewer.Core.Runtime.World.Validation;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Tools.ValidationCapture;

internal static class ValidationCaptureCommand
{
    public static int Execute(string[] args)
    {
        if (args.Length == 0 || HasFlag(args, "--help", "-h"))
        {
            ShowUsage();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();
        switch (command)
        {
            case "capture":
                return RunCapture(tail);
            case "capture-batch":
                return RunCaptureBatch(tail);
            default:
                Console.Error.WriteLine($"Unknown validation-capture command '{command}'.");
                ShowUsage();
                return 1;
        }
    }

    private static int RunCapture(string[] args)
    {
        if (args.Length == 0 || HasFlag(args, "--help", "-h"))
        {
            ShowCaptureUsage();
            return 0;
        }

        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapInput = GetOption(args, "--map-input", "-m");
        string? mapName = GetOption(args, "--map-name");
        string? datasetRoot = GetOption(args, "--dataset-root", "-d");
        string? outputRoot = GetOption(args, "--output-root", "-o");
        string? tileName = GetOption(args, "--tile-name");
        int? tileX = GetIntOption(args, "--tile-x", "-x");
        int? tileY = GetIntOption(args, "--tile-y", "-y");
        int resolution = GetIntOption(args, "--resolution", "-r") ?? 512;
        string? buildLabel = GetOption(args, "--build", "-b");
        string? looseOverlayRoot = GetOption(args, "--loose-overlay-root");
        bool dryRun = HasFlag(args, "--dry-run");
        bool gpuViewerStyle = HasFlag(args, "--gpu-viewer-style");
        bool realSceneDryRun = HasFlag(args, "--real-scene-dry-run");
        bool nativeRenderer = HasFlag(args, "--native-renderer");
        bool stubScene = HasFlag(args, "--stub-scene");

        if (string.IsNullOrWhiteSpace(clientRoot)
            || string.IsNullOrWhiteSpace(mapInput)
            || string.IsNullOrWhiteSpace(datasetRoot)
            || string.IsNullOrWhiteSpace(outputRoot)
            || string.IsNullOrWhiteSpace(tileName)
            || tileX is null
            || tileY is null)
        {
            Console.Error.WriteLine("Error: capture requires --client-root, --map-input, --dataset-root, --output-root, --tile-name, --tile-x, and --tile-y.");
            ShowCaptureUsage();
            return 1;
        }

        if (string.IsNullOrWhiteSpace(mapName))
            mapName = Path.GetFileNameWithoutExtension(mapInput);

        ValidationCaptureBatchPlan batchPlan = BuildBatchPlan(
            datasetRoot,
            mapName,
            outputRoot,
            resolution,
            buildLabel,
            [new CaptureTileInput(tileName, tileX.Value, tileY.Value)]);

        ValidationCaptureScenePolicy scenePolicy = CreateDefaultScenePolicy(resolution);
        Dictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> variantPolicies = CreateDefaultVariantPolicies();

        HeadlessValidationCaptureSession session = new(
            clientRoot,
            mapInput,
            buildLabel,
            looseOverlayRoot,
            batchPlan,
            scenePolicy,
            variantPolicies);

        if (dryRun)
        {
            Console.WriteLine("Validation capture shell dry-run succeeded.");
            Console.WriteLine($"Map: {session.BatchPlan.MapName}");
            Console.WriteLine($"Tile: {tileName} ({tileX},{tileY})");
            Console.WriteLine($"Resolution: {session.ScenePolicy.RequestedResolution}");
            Console.WriteLine($"Variant count: {session.BatchPlan.RequestCount}");
            return 0;
        }

        if (realSceneDryRun)
        {
            using ValidationWorldSceneAdapter adapter = new();
            adapter.Initialize(session);
            adapter.ApplyScenePolicy(session.ScenePolicy);

            bool allRequestsReady = true;
            foreach (ValidationCaptureTileRequest request in session.BatchPlan.TileRequests)
            {
                adapter.ApplyVariantPolicy(session.VariantPolicies[request.Variant]);
                ValidationWorldSceneSnapshot snapshot = adapter.CaptureSnapshot(request, framesObserved: 1, settledFrames: 0);
                Console.WriteLine($"Validation capture real-scene dry-run {request.Variant}: sceneContent={snapshot.HasSceneContent} framebuffer={snapshot.FramebufferWidth}x{snapshot.FramebufferHeight} tileLoaded={snapshot.TargetTileLoaded} terrainStreaming={snapshot.TerrainStreaming} pendingObjects={snapshot.PendingWorldObjectLoadCount}");
                allRequestsReady &= snapshot.HasSceneContent && snapshot.TargetTileLoaded;
            }

            return allRequestsReady ? 0 : 2;
        }

        if (gpuViewerStyle)
        {
            using ValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture gpu-viewer-style run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        if (nativeRenderer)
        {
            using NativeValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture native-renderer run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        if (stubScene)
        {
            using SyntheticValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture stub run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        Console.Error.WriteLine("Error: the validation-capture runner is not implemented yet. Re-run with --dry-run to validate arguments and shared-contract wiring.");
        return 2;
    }

    private static int RunCaptureBatch(string[] args)
    {
        if (args.Length == 0 || HasFlag(args, "--help", "-h"))
        {
            ShowCaptureBatchUsage();
            return 0;
        }

        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapInput = GetOption(args, "--map-input", "-m");
        string? mapName = GetOption(args, "--map-name");
        string? datasetRoot = GetOption(args, "--dataset-root", "-d");
        string? outputRoot = GetOption(args, "--output-root", "-o");
        string? ledgerPath = GetOption(args, "--ledger-path", "-l");
        int resolution = GetIntOption(args, "--resolution", "-r") ?? 512;
        string? buildLabel = GetOption(args, "--build", "-b");
        string? looseOverlayRoot = GetOption(args, "--loose-overlay-root");
        bool dryRun = HasFlag(args, "--dry-run");
        bool gpuViewerStyle = HasFlag(args, "--gpu-viewer-style");
        bool realSceneDryRun = HasFlag(args, "--real-scene-dry-run");
        bool nativeRenderer = HasFlag(args, "--native-renderer");
        bool stubScene = HasFlag(args, "--stub-scene");

        if (string.IsNullOrWhiteSpace(clientRoot)
            || string.IsNullOrWhiteSpace(mapInput)
            || string.IsNullOrWhiteSpace(datasetRoot)
            || string.IsNullOrWhiteSpace(outputRoot)
            || string.IsNullOrWhiteSpace(ledgerPath))
        {
            Console.Error.WriteLine("Error: capture-batch requires --client-root, --map-input, --dataset-root, --output-root, and --ledger-path.");
            ShowCaptureBatchUsage();
            return 1;
        }

        if (!File.Exists(ledgerPath))
        {
            Console.Error.WriteLine($"Error: ledger file not found at '{ledgerPath}'.");
            return 1;
        }

        if (string.IsNullOrWhiteSpace(mapName))
            mapName = Path.GetFileNameWithoutExtension(mapInput);

        CaptureLedger ledger;
        try
        {
            ledger = ReadCaptureLedger(ledgerPath);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: failed to read ledger '{ledgerPath}': {ex.Message}");
            return 1;
        }

        List<CaptureTileInput> tiles = ledger.Tiles
            .Where(static t => !string.Equals(t.Status, "captured_complete", StringComparison.OrdinalIgnoreCase))
            .Select(static t => new CaptureTileInput(t.TileName, t.TileX, t.TileY))
            .ToList();

        if (tiles.Count == 0)
        {
            Console.WriteLine("Validation capture batch: no pending tiles in ledger; nothing to do.");
            return 0;
        }

        ValidationCaptureBatchPlan batchPlan = BuildBatchPlan(datasetRoot, mapName, outputRoot, resolution, buildLabel, tiles);
        ValidationCaptureScenePolicy scenePolicy = CreateDefaultScenePolicy(resolution);
        Dictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> variantPolicies = CreateDefaultVariantPolicies();

        HeadlessValidationCaptureSession session = new(
            clientRoot,
            mapInput,
            buildLabel,
            looseOverlayRoot,
            batchPlan,
            scenePolicy,
            variantPolicies);

        if (dryRun)
        {
            Console.WriteLine("Validation capture batch dry-run succeeded.");
            Console.WriteLine($"Map: {session.BatchPlan.MapName}");
            Console.WriteLine($"Tile count: {tiles.Count}");
            Console.WriteLine($"Variant count: {session.BatchPlan.RequestCount}");
            Console.WriteLine($"Ledger: {ledgerPath}");
            return 0;
        }

        if (realSceneDryRun)
        {
            using ValidationWorldSceneAdapter adapter = new();
            adapter.Initialize(session);
            adapter.ApplyScenePolicy(session.ScenePolicy);

            bool allRequestsReady = true;
            foreach (ValidationCaptureTileRequest request in session.BatchPlan.TileRequests)
            {
                adapter.ApplyVariantPolicy(session.VariantPolicies[request.Variant]);
                ValidationWorldSceneSnapshot snapshot = adapter.CaptureSnapshot(request, framesObserved: 1, settledFrames: 0);
                Console.WriteLine($"Validation capture real-scene dry-run {request.TileName}/{request.Variant}: sceneContent={snapshot.HasSceneContent} framebuffer={snapshot.FramebufferWidth}x{snapshot.FramebufferHeight} tileLoaded={snapshot.TargetTileLoaded} terrainStreaming={snapshot.TerrainStreaming} pendingObjects={snapshot.PendingWorldObjectLoadCount}");
                allRequestsReady &= snapshot.HasSceneContent && snapshot.TargetTileLoaded;
            }

            return allRequestsReady ? 0 : 2;
        }

        if (gpuViewerStyle)
        {
            using ValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture batch gpu-viewer-style run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        if (nativeRenderer)
        {
            using NativeValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture batch native-renderer run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        if (stubScene)
        {
            using SyntheticValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture batch stub run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        Console.Error.WriteLine("Error: the validation-capture batch runner requires one of --dry-run, --real-scene-dry-run, --gpu-viewer-style, --native-renderer, or --stub-scene.");
        return 2;
    }

    private static ValidationCaptureBatchPlan BuildBatchPlan(
        string datasetRoot,
        string mapName,
        string outputRoot,
        int resolution,
        string? buildLabel,
        IReadOnlyList<CaptureTileInput> tiles)
    {
        string primaryDirectory = Path.Combine(outputRoot, "primary");
        string noLiquidsDirectory = Path.Combine(outputRoot, "noliquids");
        string noObjectsDirectory = Path.Combine(outputRoot, "noobjects");
        string objectsOnlyDirectory = Path.Combine(outputRoot, "objectsonly");

        List<ValidationCaptureTileRequest> requests = new(capacity: tiles.Count * 4);
        foreach (CaptureTileInput tile in tiles)
        {
            requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.Primary, Path.Combine(primaryDirectory, $"{tile.TileName}_viewer_validation.png")));
            requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.NoLiquids, Path.Combine(noLiquidsDirectory, $"{tile.TileName}_viewer_validation.png")));
            requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.NoObjects, Path.Combine(noObjectsDirectory, $"{tile.TileName}_viewer_validation.png")));
            requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.ObjectsOnly, Path.Combine(objectsOnlyDirectory, $"{tile.TileName}_viewer_validation.png")));
        }

        return new ValidationCaptureBatchPlan(
            datasetRoot,
            mapName,
            primaryDirectory,
            noLiquidsDirectory,
            noObjectsDirectory,
            objectsOnlyDirectory,
            resolution,
            buildLabel,
            requests);
    }

    private static ValidationCaptureScenePolicy CreateDefaultScenePolicy(int resolution)
    {
        ValidationCaptureArtifactPolicy artifactPolicy = new(
            ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette,
            ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff,
            ObjectsOnlyIntensityThreshold: 4,
            DiffMaskThreshold: 8,
            ObjectVisibilityMaskFileSuffix: "_object_visibility_mask.png",
            NoObjectMinimapFileSuffix: "_no_objects.png");

        return new ValidationCaptureScenePolicy(
            requestedResolution: resolution,
            requiredSettledFrames: 48,
            maxFramesBeforeCapture: 2400,
            detailedTileCountOverride: 25,
            fogStartFactor: 0.75f,
            fogEndDistance: 20000f,
            objectStreamingRangeMultiplierFloor: 1.0f,
            maxVisibleMdxBoundsHeight: 24f,
            disableObjectFog: true,
            disableObjectPathFilters: true,
            hideWorldLiquids: true,
            ignoreTerrainHolesGlobally: true,
            hideUiChrome: true,
            enableRuntimeWmoGroupLiquids: true,
            enableRuntimeWmoGroupVisibility: false,
            ignoreDistanceCulling: true,
            ignoreProjectedSizeCulling: true,
            ignoreVisionConeCulling: true,
            ignoreFrustumCulling: true,
            ignoreMaxViewDistanceCulling: true,
            artifactPolicy: artifactPolicy);
    }

    private static Dictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> CreateDefaultVariantPolicies()
    {
        return new Dictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy>
        {
            [ValidationCaptureVariant.Primary] = new(true, true, true, true, true, true, true, false),
            [ValidationCaptureVariant.NoLiquids] = new(true, false, true, true, true, true, true, false),
            [ValidationCaptureVariant.NoObjects] = new(true, true, false, false, false, true, true, false),
            [ValidationCaptureVariant.ObjectsOnly] = new(false, false, true, true, true, false, false, false),
        };
    }

    private static CaptureLedger ReadCaptureLedger(string ledgerPath)
    {
        string json = File.ReadAllText(ledgerPath);
        CaptureLedger? ledger = JsonSerializer.Deserialize<CaptureLedger>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        });

        if (ledger is null)
            throw new InvalidOperationException("Ledger JSON deserialized to null.");
        if (ledger.Tiles is null || ledger.Tiles.Count == 0)
            throw new InvalidOperationException("Ledger has no tiles.");

        foreach (CaptureLedgerTile tile in ledger.Tiles)
        {
            if (string.IsNullOrWhiteSpace(tile.TileName))
                throw new InvalidOperationException("Ledger tile has empty tile_name.");
        }

        return ledger;
    }

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int index = 0; index < args.Length; index++)
        {
            if (!names.Contains(args[index], StringComparer.OrdinalIgnoreCase))
                continue;

            if (index + 1 >= args.Length)
                return null;

            return args[index + 1];
        }

        return null;
    }

    private static int? GetIntOption(string[] args, params string[] names)
    {
        string? value = GetOption(args, names);
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return int.TryParse(value, out int parsed) ? parsed : null;
    }

    private static bool HasFlag(string[] args, params string[] names)
    {
        return args.Any(arg => names.Contains(arg, StringComparer.OrdinalIgnoreCase));
    }

    private static void EmitDerivedArtifacts(HeadlessValidationCaptureSession session)
    {
        ArgumentNullException.ThrowIfNull(session);

        string imagesDirectory = Path.Combine(session.BatchPlan.DatasetRoot, "images");
        Directory.CreateDirectory(imagesDirectory);

        var requestsByVariant = session.BatchPlan.TileRequests.ToDictionary(request => request.Variant);
        foreach (IGrouping<string, ValidationCaptureTileRequest> tileGroup in session.BatchPlan.TileRequests.GroupBy(static request => request.TileName, StringComparer.OrdinalIgnoreCase))
        {
            ValidationCaptureTileRequest primary = tileGroup.Single(request => request.Variant == ValidationCaptureVariant.Primary);
            ValidationCaptureTileRequest noObjects = tileGroup.Single(request => request.Variant == ValidationCaptureVariant.NoObjects);
            ValidationCaptureTileRequest? objectsOnly = tileGroup.SingleOrDefault(request => request.Variant == ValidationCaptureVariant.ObjectsOnly);

            byte[] primaryRgba = HeadlessValidationFramebufferExporter.ReadRgbaImage(primary.OutputPath, out int width, out int height);
            byte[] noObjectsRgba = HeadlessValidationFramebufferExporter.ReadRgbaImage(noObjects.OutputPath, out int noObjectsWidth, out int noObjectsHeight);
            if (noObjectsWidth != width || noObjectsHeight != height)
                throw new InvalidOperationException($"No-objects capture dimensions for '{tileGroup.Key}' do not match the primary capture.");

            byte[]? objectsOnlyRgba = null;
            if (objectsOnly is not null && File.Exists(objectsOnly.OutputPath))
            {
                objectsOnlyRgba = HeadlessValidationFramebufferExporter.ReadRgbaImage(objectsOnly.OutputPath, out int objectsOnlyWidth, out int objectsOnlyHeight);
                if (objectsOnlyWidth != width || objectsOnlyHeight != height)
                    throw new InvalidOperationException($"Objects-only capture dimensions for '{tileGroup.Key}' do not match the primary capture.");
            }

            ValidationCaptureArtifactOutputs outputs = ValidationCaptureArtifactBuilder.Build(
                new ValidationCaptureArtifactInputs(
                    tileGroup.Key,
                    session.BuildLabel,
                    width,
                    height,
                    primaryRgba,
                    noObjectsRgba,
                    objectsOnlyRgba),
                session.ScenePolicy.ArtifactPolicy);

            string objectMaskPath = Path.Combine(imagesDirectory, $"{tileGroup.Key}{session.ScenePolicy.ArtifactPolicy.ObjectVisibilityMaskFileSuffix}");
            string noObjectMinimapPath = Path.Combine(imagesDirectory, $"{tileGroup.Key}{session.ScenePolicy.ArtifactPolicy.NoObjectMinimapFileSuffix}");

            HeadlessValidationFramebufferExporter.WriteMaskImage(objectMaskPath, width, height, outputs.ObjectVisibilityMaskL8Pixels);
            HeadlessValidationFramebufferExporter.WriteImage(noObjectMinimapPath, width, height, outputs.NoObjectMinimapRgbaPixels, sourceOriginBottomLeft: false);
        }
    }

    private static void ShowUsage()
    {
        Console.WriteLine("""
            WowViewer.Tool.ValidationCapture — headless validation-capture host shell

            Usage: WowViewer.Tool.ValidationCapture <command> [options]

            Commands:
              capture        Validate one bounded validation-capture request and build a shared-runtime session
              capture-batch  Execute a manifest/ledger-driven bounded validation-capture batch

            Use 'capture --help' or 'capture-batch --help' for argument details.
            """);
    }

    private static void ShowCaptureUsage()
    {
        Console.WriteLine("""
            Usage: WowViewer.Tool.ValidationCapture capture [options]

            Required:
              --client-root <dir>
              --map-input <path>
              --dataset-root <dir>
              --output-root <dir>
              --tile-name <name>
              --tile-x <int>
              --tile-y <int>

            Optional:
              --map-name <name>
              --build <label>
              --loose-overlay-root <dir>
              --resolution <int>      Default: 512
              --dry-run               Build the shared-runtime session and print a summary
                            --gpu-viewer-style      Render bounded captures with wow-viewer GPU output using validation camera frames
                            --real-scene-dry-run    Build real runtime-frame snapshots without framebuffer rendering
              --stub-scene            Run the host loop against a clearly synthetic scene adapter
            """);
    }

    private static void ShowCaptureBatchUsage()
    {
        Console.WriteLine("""
            Usage: WowViewer.Tool.ValidationCapture capture-batch [options]

            Required:
              --client-root <dir>
              --map-input <path>
              --dataset-root <dir>
              --output-root <dir>
              --ledger-path <path>

            Optional:
              --map-name <name>
              --build <label>
              --loose-overlay-root <dir>
              --resolution <int>      Default: 512
              --dry-run               Build session + tile plan summary without rendering
              --gpu-viewer-style      Render bounded captures with wow-viewer GPU output using validation camera frames
              --real-scene-dry-run    Build real runtime-frame snapshots without framebuffer rendering
              --native-renderer       Run bounded captures through NativeValidationWorldSceneAdapter
              --stub-scene            Run the host loop against a clearly synthetic scene adapter
            """);
    }

    private sealed record CaptureTileInput(string TileName, int TileX, int TileY);

    private sealed class CaptureLedger
    {
        [JsonPropertyName("build")]
        public string? Build { get; init; }

        [JsonPropertyName("tiles")]
        public List<CaptureLedgerTile> Tiles { get; init; } = [];
    }

    private sealed class CaptureLedgerTile
    {
        [JsonPropertyName("tile_name")]
        public string TileName { get; init; } = string.Empty;

        [JsonPropertyName("tile_x")]
        public int TileX { get; init; }

        [JsonPropertyName("tile_y")]
        public int TileY { get; init; }

        [JsonPropertyName("status")]
        public string? Status { get; init; }
    }

    private sealed class SyntheticValidationWorldSceneAdapter : IValidationWorldSceneAdapter
    {
        private HeadlessValidationCaptureSession? _session;
        private ValidationCaptureVariantPolicy _variantPolicy;
        private ValidationCaptureVariant _variant;

        public void Initialize(HeadlessValidationCaptureSession session)
        {
            _session = session;
        }

        public void ApplyScenePolicy(ValidationCaptureScenePolicy scenePolicy)
        {
        }

        public void ApplyVariantPolicy(ValidationCaptureVariantPolicy variantPolicy)
        {
            _variantPolicy = variantPolicy;
        }

        public ValidationWorldSceneSnapshot CaptureSnapshot(ValidationCaptureTileRequest request, int framesObserved, int settledFrames)
        {
            _variant = request.Variant;
            int resolution = _session?.ScenePolicy.RequestedResolution ?? 512;
            return new ValidationWorldSceneSnapshot(
                HasSceneContent: true,
                FramebufferWidth: resolution,
                FramebufferHeight: resolution,
                TargetTileLoaded: true,
                TerrainStreaming: false,
                PendingWorldObjectLoadCount: 0);
        }

        public float ResolveGroundHeight(int tileX, int tileY)
        {
            return 128f;
        }

        public void RenderFrame(ValidationCaptureCameraFrame cameraFrame)
        {
        }

        public byte[] ReadFramebufferRgba()
        {
            int resolution = _session?.ScenePolicy.RequestedResolution ?? 512;
            byte[] pixels = new byte[checked(resolution * resolution * 4)];
            (byte r, byte g, byte b) = _variant switch
            {
                ValidationCaptureVariant.Primary => ((byte)64, (byte)128, (byte)96),
                ValidationCaptureVariant.NoLiquids => ((byte)96, (byte)96, (byte)64),
                ValidationCaptureVariant.NoObjects => ((byte)48, (byte)80, (byte)128),
                ValidationCaptureVariant.ObjectsOnly => ((byte)192, (byte)128, (byte)32),
                _ => ((byte)32, (byte)32, (byte)32),
            };

            for (int index = 0; index < pixels.Length; index += 4)
            {
                pixels[index + 0] = _variantPolicy.ShowTerrain ? r : (byte)(r / 2);
                pixels[index + 1] = _variantPolicy.ShowObjects ? g : (byte)(g / 2);
                pixels[index + 2] = _variantPolicy.ShowTerrainLiquids ? b : (byte)(b / 2);
                pixels[index + 3] = 255;
            }

            return pixels;
        }

        public void Dispose()
        {
        }
    }
}
