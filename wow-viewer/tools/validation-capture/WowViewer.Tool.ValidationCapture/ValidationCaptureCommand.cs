using WowViewer.Core.Renderer.Validation;
using WowViewer.Core.Runtime.World.Validation;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Tools.ValidationCapture;

internal static class ValidationCaptureCommand
{
    public static int Execute(string[] args)
    {
        if (args.Length == 0)
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
            case "profile-render":
                return RunProductionRendererProfile(tail);
            default:
                Console.Error.WriteLine($"Unknown validation-capture command '{command}'.");
                ShowUsage();
                return 1;
        }
    }

    private static int RunProductionRendererProfile(string[] args)
    {
        if (args.Length == 0 || HasFlag(args, "--help", "-h"))
        {
            ShowProductionRendererProfileUsage();
            return 0;
        }

        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? wdtPath = GetOption(args, "--map-input", "-m");
        string? outputPath = GetOption(args, "--output", "-o");
        string? buildLabel = GetOption(args, "--build", "-b");
        string? listfilePath = GetOption(args, "--listfile");
        string? looseOverlayRoot = GetOption(args, "--loose-overlay-root");
        int resolution = GetIntOption(args, "--resolution", "-r") ?? 512;
        int warmupFrames = GetIntOption(args, "--warmup-frames") ?? 8;
        int frames = GetIntOption(args, "--frames") ?? 12;
        int? tileX = GetIntOption(args, "--tile-x");
        int? tileY = GetIntOption(args, "--tile-y");
        bool loadAllTiles = HasFlag(args, "--load-all-tiles");
        bool dryRun = HasFlag(args, "--dry-run");

        if (string.IsNullOrWhiteSpace(clientRoot)
            || string.IsNullOrWhiteSpace(wdtPath)
            || string.IsNullOrWhiteSpace(outputPath))
        {
            Console.Error.WriteLine("Error: profile-render requires --client-root, --map-input, and --output.");
            ShowProductionRendererProfileUsage();
            return 1;
        }

        if (resolution <= 0 || warmupFrames < 0 || frames <= 0)
        {
            Console.Error.WriteLine("Error: --resolution and --frames must be positive; --warmup-frames cannot be negative.");
            return 1;
        }

        if (tileX.HasValue != tileY.HasValue || tileX is < 0 or > 63 || tileY is < 0 or > 63)
        {
            Console.Error.WriteLine("Error: --tile-x and --tile-y must be supplied together and each must be in [0, 63].");
            return 1;
        }

        if (dryRun)
        {
            try
            {
                string resolved = ProductionWorldSceneProfiler.ValidateWdtInput(clientRoot, wdtPath, listfilePath, looseOverlayRoot, tileX, tileY);
                Console.WriteLine("Production WorldScene profile dry-run succeeded.");
                Console.WriteLine($"Client root: {clientRoot}");
                Console.WriteLine($"Resolved: {resolved}");
                Console.WriteLine($"Output: {outputPath}");
                Console.WriteLine($"Frames: warmup={warmupFrames}, measured={frames}, tile={tileX?.ToString() ?? "map-default"}_{tileY?.ToString() ?? "map-default"}, loadAllTiles={loadAllTiles}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Production WorldScene profile dry-run failed: {ex.GetType().Name}: {ex.Message}");
                return 2;
            }
        }

        try
        {
            var report = ProductionWorldSceneProfiler.Run(
                clientRoot,
                wdtPath,
                outputPath,
                buildLabel,
                listfilePath,
                looseOverlayRoot,
                resolution,
                warmupFrames,
                frames,
                loadAllTiles,
                tileX,
                tileY);
            Console.WriteLine($"Production WorldScene profile completed: {report.Frames.Count} measured frames, {report.Findings.Count} findings.");
            Console.WriteLine($"Report: {outputPath}");
            return report.Findings.Any(static finding => finding.Severity == "error") ? 2 : 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Production WorldScene profile failed: {ex.GetType().Name}: {ex.Message}");
            return 2;
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
        int? settledFrames = GetIntOption(args, "--settled-frames");
        int? maxFrames = GetIntOption(args, "--max-frames");
        int? batchSettledFrames = GetIntOption(args, "--batch-settled-frames");
        string? variantsArg = GetOption(args, "--variants");
        bool dryRun = HasFlag(args, "--dry-run");
        bool gpuViewerStyle = HasFlag(args, "--renderer", "--gpu-viewer-style");
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
            [new CaptureTileInput(tileName, tileX.Value, tileY.Value, null, null, null, null, null, null, null, null, null)],
            ParseVariantsFlag(variantsArg));

        ValidationCaptureScenePolicy scenePolicy = CreateDefaultScenePolicy(resolution,
            settledFramesOverride: settledFrames,
            maxFramesOverride: maxFrames,
            batchSettledFramesOverride: batchSettledFrames);
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
                Console.WriteLine($"Validation capture real-scene dry-run {request.Variant}: sceneContent={snapshot.HasSceneContent} framebuffer={snapshot.FramebufferWidth}x{snapshot.FramebufferHeight} tileLoaded={snapshot.TargetTileLoaded} terrainStreaming={snapshot.TerrainStreaming} pendingObjects={snapshot.PendingWorldObjectLoadCount} terrainHeightRange={adapter.LastTerrainHeightRange:F3}");
                allRequestsReady &= snapshot.HasSceneContent && snapshot.TargetTileLoaded;
            }

            return allRequestsReady ? 0 : 2;
        }

        if (gpuViewerStyle)
        {
            using ValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture renderer run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        if (nativeRenderer)
        {
            using NativeValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            Console.WriteLine($"Validation capture renderer run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
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
        int? settledFrames = GetIntOption(args, "--settled-frames");
        int? maxFrames = GetIntOption(args, "--max-frames");
        int? batchSettledFrames = GetIntOption(args, "--batch-settled-frames");
        bool dryRun = HasFlag(args, "--dry-run");
        bool gpuViewerStyle = HasFlag(args, "--renderer", "--gpu-viewer-style");
        bool realSceneDryRun = HasFlag(args, "--real-scene-dry-run");
        bool nativeRenderer = HasFlag(args, "--native-renderer");
        bool stubScene = HasFlag(args, "--stub-scene");
        string? variantsArg = GetOption(args, "--variants");
        HashSet<ValidationCaptureVariant> enabledVariants = ParseVariantsFlag(variantsArg);

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
            .Select(static t => new CaptureTileInput(
                t.TileName,
                t.TileX,
                t.TileY,
                t.AssetPath,
                t.InstanceType,
                t.UniqueId,
                t.RotX,
                t.RotY,
                t.RotZ,
                t.Scale,
                t.ObjectInstanceCount,
                t.ObjectInstances))
            .ToList();

        if (tiles.Count == 0)
        {
            Console.WriteLine("Validation capture batch: no pending tiles in ledger; nothing to do.");
            return 0;
        }

        ValidationCaptureBatchPlan batchPlan = BuildBatchPlan(datasetRoot, mapName, outputRoot, resolution, buildLabel, tiles, enabledVariants);
        ValidationCaptureScenePolicy scenePolicy = CreateDefaultScenePolicy(resolution,
            settledFramesOverride: settledFrames,
            maxFramesOverride: maxFrames,
            batchSettledFramesOverride: batchSettledFrames);
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
            WritePoseMetadataArtifacts(session, tiles);
            Console.WriteLine($"Validation capture batch renderer run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        if (nativeRenderer)
        {
            using NativeValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            WritePoseMetadataArtifacts(session, tiles);
            Console.WriteLine($"Validation capture batch renderer run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        if (stubScene)
        {
            using SyntheticValidationWorldSceneAdapter adapter = new();
            ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, adapter);
            EmitDerivedArtifacts(session);
            WritePoseMetadataArtifacts(session, tiles);
            Console.WriteLine($"Validation capture batch stub run completed: {result.SucceededVariantCount}/{result.TotalVariantCount} succeeded, {result.TimedOutVariantCount} timed out.");
            return result.FailedVariantCount == 0 ? 0 : 2;
        }

        Console.Error.WriteLine("Error: the validation-capture batch runner requires one of --dry-run, --real-scene-dry-run, --renderer, --native-renderer, or --stub-scene.");
        return 2;
    }

    private static ValidationCaptureBatchPlan BuildBatchPlan(
        string datasetRoot,
        string mapName,
        string outputRoot,
        int resolution,
        string? buildLabel,
        IReadOnlyList<CaptureTileInput> tiles,
        HashSet<ValidationCaptureVariant> enabledVariants)
    {
        string primaryDirectory = Path.Combine(outputRoot, "primary");
        string noLiquidsDirectory = Path.Combine(outputRoot, "noliquids");
        string noObjectsDirectory = Path.Combine(outputRoot, "noobjects");
        string objectsOnlyDirectory = Path.Combine(outputRoot, "objectsonly");
        string terrainShadeDirectory = Path.Combine(outputRoot, "terrain-shade");

        List<ValidationCaptureTileRequest> requests = new(capacity: tiles.Count * enabledVariants.Count);
        foreach (CaptureTileInput tile in tiles)
        {
            if (enabledVariants.Contains(ValidationCaptureVariant.Primary))
            {
                requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.Primary, Path.Combine(primaryDirectory, $"{tile.TileName}_viewer_validation.png")));
            }
            if (enabledVariants.Contains(ValidationCaptureVariant.NoLiquids))
            {
                requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.NoLiquids, Path.Combine(noLiquidsDirectory, $"{tile.TileName}_viewer_validation.png")));
            }
            if (enabledVariants.Contains(ValidationCaptureVariant.NoObjects))
            {
                requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.NoObjects, Path.Combine(noObjectsDirectory, $"{tile.TileName}_viewer_validation.png")));
            }
            if (enabledVariants.Contains(ValidationCaptureVariant.ObjectsOnly))
            {
                requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.ObjectsOnly, Path.Combine(objectsOnlyDirectory, $"{tile.TileName}_viewer_validation.png")));
            }
            if (enabledVariants.Contains(ValidationCaptureVariant.TerrainShade))
            {
                requests.Add(new ValidationCaptureTileRequest(tile.TileName, tile.TileX, tile.TileY, ValidationCaptureVariant.TerrainShade, Path.Combine(terrainShadeDirectory, $"{tile.TileName}_terrain_shade.png")));
            }
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

    private static HashSet<ValidationCaptureVariant> ParseVariantsFlag(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return new HashSet<ValidationCaptureVariant>
            {
                ValidationCaptureVariant.Primary,
                ValidationCaptureVariant.NoLiquids,
                ValidationCaptureVariant.NoObjects,
                ValidationCaptureVariant.ObjectsOnly,
                ValidationCaptureVariant.TerrainShade,
            };
        }

        string normalized = raw.Trim().ToLowerInvariant();
        if (normalized == "all")
        {
            return new HashSet<ValidationCaptureVariant>
            {
                ValidationCaptureVariant.Primary,
                ValidationCaptureVariant.NoLiquids,
                ValidationCaptureVariant.NoObjects,
                ValidationCaptureVariant.ObjectsOnly,
                ValidationCaptureVariant.TerrainShade,
            };
        }

        HashSet<ValidationCaptureVariant> result = new();
        foreach (string token in normalized.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            switch (token)
            {
                case "primary":
                    result.Add(ValidationCaptureVariant.Primary);
                    break;
                case "no-liquids":
                case "noliquids":
                    result.Add(ValidationCaptureVariant.NoLiquids);
                    break;
                case "no-objects":
                case "noobjects":
                    result.Add(ValidationCaptureVariant.NoObjects);
                    break;
                case "objects-only":
                case "objectsonly":
                    result.Add(ValidationCaptureVariant.ObjectsOnly);
                    break;
                case "terrain-shade":
                case "terrainshade":
                    result.Add(ValidationCaptureVariant.TerrainShade);
                    break;
                default:
                    throw new ArgumentException($"Unknown --variants token '{token}'. Expected one of: primary, no-liquids, no-objects, objects-only, terrain-shade, all.");
            }
        }
        if (result.Count == 0)
        {
            throw new ArgumentException("--variants resolved to an empty set. Provide at least one of: primary, no-liquids, no-objects, objects-only, terrain-shade, all.");
        }
        return result;
    }

private static ValidationCaptureScenePolicy CreateDefaultScenePolicy(int resolution,
        int? settledFramesOverride = null,
        int? maxFramesOverride = null,
        int? batchSettledFramesOverride = null)
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
            requiredSettledFrames: settledFramesOverride ?? 12,
            maxFramesBeforeCapture: maxFramesOverride ?? 480,
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
            artifactPolicy: artifactPolicy,
            batchSettledFrames: batchSettledFramesOverride ?? 2,
            fastSettleAfterBatchReady: true);
    }

    private static Dictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> CreateDefaultVariantPolicies()
    {
        return new Dictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy>
        {
            [ValidationCaptureVariant.Primary] = new(true, true, true, true, true, true, true, false),
            [ValidationCaptureVariant.NoLiquids] = new(true, false, true, true, true, true, true, false),
            [ValidationCaptureVariant.NoObjects] = new(true, true, false, false, false, true, true, false),
            [ValidationCaptureVariant.ObjectsOnly] = new(false, false, true, true, true, false, false, false),
            [ValidationCaptureVariant.TerrainShade] = new(true, false, false, false, false, false, false, false, TerrainShadeOnly: true),
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

        ILookup<ValidationCaptureVariant, ValidationCaptureTileRequest> requestsByVariant =
            session.BatchPlan.TileRequests.ToLookup(static request => request.Variant);
        foreach (IGrouping<string, ValidationCaptureTileRequest> tileGroup in session.BatchPlan.TileRequests.GroupBy(static request => request.TileName, StringComparer.OrdinalIgnoreCase))
        {
            ValidationCaptureTileRequest? objectsOnly = tileGroup.SingleOrDefault(request => request.Variant == ValidationCaptureVariant.ObjectsOnly);
            if (objectsOnly is null || !File.Exists(objectsOnly.OutputPath))
            {
                Console.Error.WriteLine($"EmitDerivedArtifacts: skipping tile '{tileGroup.Key}' (no ObjectsOnly capture).");
                continue;
            }

            byte[] objectsOnlyRgba = HeadlessValidationFramebufferExporter.ReadRgbaImage(objectsOnly.OutputPath, out int width, out int height);

            byte[]? primaryRgba = null;
            ValidationCaptureTileRequest? primaryReq = requestsByVariant[ValidationCaptureVariant.Primary].FirstOrDefault(r => r.TileName == tileGroup.Key);
            if (primaryReq is not null && File.Exists(primaryReq.OutputPath))
            {
                primaryRgba = HeadlessValidationFramebufferExporter.ReadRgbaImage(primaryReq.OutputPath, out int primaryWidth, out int primaryHeight);
                if (primaryWidth != width || primaryHeight != height)
                    throw new InvalidOperationException($"Primary capture dimensions for '{tileGroup.Key}' do not match the ObjectsOnly capture.");
            }

            byte[]? noObjectsRgba = null;
            ValidationCaptureTileRequest? noObjectsReq = requestsByVariant[ValidationCaptureVariant.NoObjects].FirstOrDefault(r => r.TileName == tileGroup.Key);
            if (noObjectsReq is not null && File.Exists(noObjectsReq.OutputPath))
            {
                noObjectsRgba = HeadlessValidationFramebufferExporter.ReadRgbaImage(noObjectsReq.OutputPath, out int noObjectsWidth, out int noObjectsHeight);
                if (noObjectsWidth != width || noObjectsHeight != height)
                    throw new InvalidOperationException($"No-objects capture dimensions for '{tileGroup.Key}' do not match the ObjectsOnly capture.");
            }

            ValidationCaptureArtifactOutputs outputs = ValidationCaptureArtifactBuilder.Build(
                new ValidationCaptureArtifactInputs(
                    tileGroup.Key,
                    session.BuildLabel,
                    width,
                    height,
                    primaryRgba ?? objectsOnlyRgba,
                    noObjectsRgba ?? objectsOnlyRgba,
                    objectsOnlyRgba),
                session.ScenePolicy.ArtifactPolicy);

            string objectMaskPath = Path.Combine(imagesDirectory, $"{tileGroup.Key}{session.ScenePolicy.ArtifactPolicy.ObjectVisibilityMaskFileSuffix}");

            HeadlessValidationFramebufferExporter.WriteMaskImage(objectMaskPath, width, height, outputs.ObjectVisibilityMaskL8Pixels);
            if (primaryRgba is not null && noObjectsRgba is not null)
            {
                string noObjectMinimapPath = Path.Combine(imagesDirectory, $"{tileGroup.Key}{session.ScenePolicy.ArtifactPolicy.NoObjectMinimapFileSuffix}");
                HeadlessValidationFramebufferExporter.WriteImage(noObjectMinimapPath, width, height, outputs.NoObjectMinimapRgbaPixels, sourceOriginBottomLeft: false);
            }
        }
    }

    private static void WritePoseMetadataArtifacts(HeadlessValidationCaptureSession session, IReadOnlyList<CaptureTileInput> tiles)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tiles);

        if (tiles.Count == 0)
            return;

        string metadataDirectory = Path.Combine(session.BatchPlan.DatasetRoot, "pose-metadata");
        Directory.CreateDirectory(metadataDirectory);

        foreach (CaptureTileInput tile in tiles)
        {
            string path = Path.Combine(metadataDirectory, $"{tile.TileName}_pose.json");
            var payload = new
            {
                tile_name = tile.TileName,
                tile_x = tile.TileX,
                tile_y = tile.TileY,
                build = session.BuildLabel,
                map = session.BatchPlan.MapName,
                object_instance_count = tile.ObjectInstanceCount,
                object_instances = tile.ObjectInstances,
                asset_path = tile.AssetPath,
                instance_type = tile.InstanceType,
                unique_id = tile.UniqueId,
                rot_x = tile.RotX,
                rot_y = tile.RotY,
                rot_z = tile.RotZ,
                scale = tile.Scale,
            };

            File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
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
              profile-render Render the production WorldScene in a hidden OpenGL context and emit stage diagnostics

            Use 'capture --help', 'capture-batch --help', or 'profile-render --help' for argument details.
            """);
    }

    private static void ShowProductionRendererProfileUsage()
    {
        Console.WriteLine("""
            Usage: WowViewer.Tool.ValidationCapture profile-render [options]

            Required:
              --client-root <dir>    Client directory used by the production MpqDataSource
              --map-input <path>     Local WDT file or WDT virtual path inside the configured client
              --output <path>        JSON diagnostic report to write

            Optional:
              --build <label>        Format-build hint passed to WorldScene
              --listfile <path>      Supplemental MPQ listfile
              --loose-overlay-root <dir>
              --resolution <int>     Hidden render target size; default: 512
              --warmup-frames <int>  Production frames before sampling; default: 8
              --frames <int>         Production frames to measure; default: 12
              --tile-x <int>         Target ADT tile column; pair with --tile-y
              --tile-y <int>         Target ADT tile row; pair with --tile-x
              --load-all-tiles       Opt into synchronous full terrain residency before sampling
              --dry-run              Verify client/map input without opening a GPU context

            For standard-era clients, prefer a client-backed map path such as
            World\\Maps\\Azeroth\\Azeroth.wdt so terrain and assets come from the same build.
            Archive-backed Alpha WDT input is not supported; provide its local WDT file instead.

            This runs the current WorldScene.Render path, including terrain streaming, graph traversal,
            WMO/MDX visibility and submission, deferred asset loading, and all existing frame-stage timers.
            It does not claim per-stage GPU timing yet; the JSON report calls that gap out explicitly.
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
              --settled-frames <int>  Frames to wait for scene settle (default: 12)
              --max-frames <int>     Max frames before capture timeout (default: 480)
              --batch-settled-frames <int>  Fast-settle frames after first tile settles (default: 2)
              --variants <list>       primary,no-liquids,no-objects,objects-only,terrain-shade,all
              --dry-run               Build the shared-runtime session and print a summary
              --renderer             Run bounded captures through the existing WoWViewer renderer
                            --gpu-viewer-style      Back-compat alias for --renderer
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
              --settled-frames <int>  Frames to wait for scene settle (default: 12)
              --max-frames <int>     Max frames before capture timeout (default: 480)
              --batch-settled-frames <int>  Fast-settle frames after first tile settles (default: 2)
              --dry-run               Build session + tile plan summary without rendering
              --renderer             Run bounded captures through the existing WoWViewer renderer
              --gpu-viewer-style      Back-compat alias for --renderer
              --real-scene-dry-run    Build real runtime-frame snapshots without framebuffer rendering
              --native-renderer       Run bounded captures through NativeValidationWorldSceneAdapter
              --stub-scene            Run the host loop against a clearly synthetic scene adapter

            Notes:
              - capture-batch accepts optional pose metadata in ledger rows:
                asset_path, instance_type, unique_id, rot_x, rot_y, rot_z, scale
              - when rendering modes run, pose metadata artifacts are emitted to:
                <dataset-root>/pose-metadata/<tile_name>_pose.json
            """);
    }

    private sealed record CaptureTileInput(
        string TileName,
        int TileX,
        int TileY,
        string? AssetPath,
        string? InstanceType,
        int? UniqueId,
        float? RotX,
        float? RotY,
        float? RotZ,
        float? Scale,
        int? ObjectInstanceCount,
        IReadOnlyList<CaptureLedgerObjectInstance>? ObjectInstances);

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

        [JsonPropertyName("asset_path")]
        public string? AssetPath { get; init; }

        [JsonPropertyName("instance_type")]
        public string? InstanceType { get; init; }

        [JsonPropertyName("unique_id")]
        public int? UniqueId { get; init; }

        [JsonPropertyName("rot_x")]
        public float? RotX { get; init; }

        [JsonPropertyName("rot_y")]
        public float? RotY { get; init; }

        [JsonPropertyName("rot_z")]
        public float? RotZ { get; init; }

        [JsonPropertyName("scale")]
        public float? Scale { get; init; }

        [JsonPropertyName("object_instance_count")]
        public int? ObjectInstanceCount { get; init; }

        [JsonPropertyName("object_instances")]
        public List<CaptureLedgerObjectInstance>? ObjectInstances { get; init; }
    }

    private sealed class CaptureLedgerObjectInstance
    {
        [JsonPropertyName("asset_path")]
        public string? AssetPath { get; init; }

        [JsonPropertyName("instance_type")]
        public string? InstanceType { get; init; }

        [JsonPropertyName("instance_idx")]
        public int? InstanceIdx { get; init; }

        [JsonPropertyName("unique_id")]
        public int? UniqueId { get; init; }

        [JsonPropertyName("rot_x")]
        public float? RotX { get; init; }

        [JsonPropertyName("rot_y")]
        public float? RotY { get; init; }

        [JsonPropertyName("rot_z")]
        public float? RotZ { get; init; }

        [JsonPropertyName("scale")]
        public float? Scale { get; init; }

        [JsonPropertyName("pos_x")]
        public float? PosX { get; init; }

        [JsonPropertyName("pos_y")]
        public float? PosY { get; init; }

        [JsonPropertyName("pos_z")]
        public float? PosZ { get; init; }
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
