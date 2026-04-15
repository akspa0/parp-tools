using SixLabors.ImageSharp;
using System.Text.Json;
using System.Diagnostics;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.Converters;
using WoWMapConverter.Core.Formats.PM4;
using WoWMapConverter.Core.Services;
using WoWMapConverter.Core.VLM;
using WowViewer.Core.IO.Files;

namespace WoWMapConverter.Cli;

/// <summary>
/// Unified CLI for WoW Alpha → LK 3.3.5 conversion.
/// Includes WDT/ADT, WMO v14→v17, MDX→M2, and DBC crosswalks.
/// </summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        var command = args[0].ToLowerInvariant();

        return command switch
        {
            "convert" => await RunConvertAsync(args.Skip(1).ToArray()),
            "convert-lk-to-alpha" => await RunConvertLkToAlphaAsync(args.Skip(1).ToArray()),
            "convert-wmo" => RunConvertWmo(args.Skip(1).ToArray()),
            "convert-wmo-to-alpha" => RunConvertWmoToAlpha(args.Skip(1).ToArray()),
            "convert-mdx" => RunConvertMdx(args.Skip(1).ToArray()),
            "convert-m2-to-mdx" => RunConvertM2ToMdx(args.Skip(1).ToArray()),
            "pm4-export" => RunPm4Export(args.Skip(1).ToArray()),
            "pm4-validate-coords" => RunPm4ValidateCoords(args.Skip(1).ToArray()),
            "development-analyze" => RunDevelopmentAnalyze(args.Skip(1).ToArray()),
            "development-repair" => RunDevelopmentRepair(args.Skip(1).ToArray()),
            "terrain-texture-transfer" => RunTerrainTextureTransfer(args.Skip(1).ToArray()),
            "wmo-info" => RunWmoInfo(args.Skip(1).ToArray()),
            "ml-export" => await RunVlmExportAsync(args.Skip(1).ToArray()),
            "ml-list-maps" => await RunMlListMapsAsync(args.Skip(1).ToArray()),
            "ml-decode" => await RunVlmDecodeAsync(args.Skip(1).ToArray()),
            "ml-bake" => await RunVlmBakeAsync(args.Skip(1).ToArray()),
            "ml-bake-heightmap" => await RunVlmBakeHeightmapAsync(args.Skip(1).ToArray()),
            "ml-synth" => await RunVlmSynthAsync(args.Skip(1).ToArray()),
            "ml-harvest" => await RunMkHarvestAsync(args.Skip(1).ToArray()),
                "ml-corpus" => await RunMlCorpusAsync(args.Skip(1).ToArray()),
            "mk-export" => await RunVlmExportAsync(args.Skip(1).ToArray()),
            "mk-decode" => await RunVlmDecodeAsync(args.Skip(1).ToArray()),
            "mk-bake" => await RunVlmBakeAsync(args.Skip(1).ToArray()),
            "mk-bake-heightmap" => await RunVlmBakeHeightmapAsync(args.Skip(1).ToArray()),
            "mk-synth" => await RunVlmSynthAsync(args.Skip(1).ToArray()),
            "mk-harvest" => await RunMkHarvestAsync(args.Skip(1).ToArray()),
            "vlm-export" => await RunVlmExportAsync(args.Skip(1).ToArray()),
            "vlm-list-maps" => await RunMlListMapsAsync(args.Skip(1).ToArray()),
            "vlm-decode" => await RunVlmDecodeAsync(args.Skip(1).ToArray()),
            "vlm-bake" => await RunVlmBakeAsync(args.Skip(1).ToArray()),
            "vlm-bake-heightmap" => await RunVlmBakeHeightmapAsync(args.Skip(1).ToArray()),
            "vlm-synth" => await RunVlmSynthAsync(args.Skip(1).ToArray()),
            "analyze" => await RunAnalyzeAsync(args.Skip(1).ToArray()),
            "batch" or "vlm-batch" or "mk-batch" or "ml-batch" => await RunBatchAsync(args.Skip(1).ToArray()),
            _ => await RunDefaultConvertAsync(args)
        };
    }

    private static async Task<int> RunBatchAsync(string[] args)
    {
        string? configPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--config":
                case "-c":
                    if (i + 1 < args.Length) configPath = args[++i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(configPath))
        {
            Console.WriteLine("ML Dataset Batch Export");
            Console.WriteLine("Usage: wowmapconverter ml-batch --config <config.json>  (legacy aliases: mk-batch, vlm-batch)");
            return 1;
        }

        if (!File.Exists(configPath))
        {
            Console.WriteLine($"Error: Config file not found: {configPath}");
            return 1;
        }

        try
        {
            var json = await File.ReadAllTextAsync(configPath);
            var config = System.Text.Json.JsonSerializer.Deserialize<VlmBatchExportConfig>(json);
            
            if (config == null)
            {
                Console.WriteLine("Error: Invalid config JSON");
                return 1;
            }

            var exporter = new VlmDatasetExporter();
            var progress = new Progress<string>(msg => Console.WriteLine(msg));
            
            await exporter.ExportBatchAsync(config, progress);
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> RunMkHarvestAsync(string[] args)
    {
        string? datasetDir = null;
        string? outputPath = null;
        string? referenceOutputDir = null;
        bool generateReferenceMinimaps = false;
        bool force = false;
        bool applyShadows = true;
        bool invertAlpha = true;
        float shadowIntensity = 0.5f;
        bool requestedDeprecatedReferenceMinimapGeneration = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--dataset":
                case "-d":
                    if (i + 1 < args.Length) datasetDir = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--reference-output":
                case "-r":
                    if (i + 1 < args.Length) referenceOutputDir = args[++i];
                    break;
                case "--generate-reference-minimaps":
                    generateReferenceMinimaps = true;
                    requestedDeprecatedReferenceMinimapGeneration = true;
                    break;
                case "--force":
                    force = true;
                    break;
                case "--no-shadows":
                    applyShadows = false;
                    break;
                case "--invert-alpha":
                    invertAlpha = true;
                    break;
                case "--no-invert-alpha":
                    invertAlpha = false;
                    break;
                case "--shadow-intensity":
                    if (i + 1 < args.Length && float.TryParse(args[++i], out float parsedIntensity))
                        shadowIntensity = Math.Clamp(parsedIntensity, 0f, 1f);
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(datasetDir))
        {
            Console.WriteLine("ML Dataset Harvest - Audit per-tile coverage and write the dataset manifest");
            Console.WriteLine();
            Console.WriteLine("Usage: ml-harvest --dataset <dir> [options]  (legacy alias: mk-harvest)");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --dataset, -d <dir>               ML dataset root directory (must contain dataset/*.json)");
            Console.WriteLine("  --output, -o <json>               Output manifest path (default: <dataset>/ml_dataset_manifest.json)");
            Console.WriteLine("  Reference minimap generation is disabled on the ML dataset surface. Use MdxViewer validation capture for rendered minimaps.");
            return 1;
        }

        if (generateReferenceMinimaps || requestedDeprecatedReferenceMinimapGeneration || force || !string.IsNullOrWhiteSpace(referenceOutputDir))
            Console.WriteLine("Warning: baked 4k reference minimap generation is disabled on the ML dataset surface; those options are ignored.");

        var harvester = new MkDatasetHarvester();
        var options = new MkDatasetHarvestOptions(
            DatasetRoot: datasetDir,
            ManifestOutputPath: outputPath,
            GenerateReferenceMinimaps: false,
            ForceRegenerateReferenceMinimaps: false,
            ApplyShadows: applyShadows,
            ShadowIntensity: shadowIntensity,
            InvertAlpha: invertAlpha,
            ReferenceMinimapDirectory: null);

        try
        {
            var progress = new Progress<string>(msg => Console.WriteLine(msg));
            MkDatasetHarvestResult result = await harvester.HarvestAsync(options, progress);
            Console.WriteLine();
            Console.WriteLine("ML dataset harvest complete:");
            Console.WriteLine($"  Tiles processed: {result.TilesProcessed}");
            Console.WriteLine($"  Source minimaps found: {result.SourceMinimapsFound}");
            Console.WriteLine($"  Local heightmaps found: {result.LocalHeightmapsFound}");
            Console.WriteLine($"  Global heightmaps found: {result.GlobalHeightmapsFound}");
            Console.WriteLine($"  Tiles with alpha masks: {result.TilesWithAlphaMasks}");
            Console.WriteLine($"  Manifest: {result.ManifestPath}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }
    private static async Task<int> RunMlCorpusAsync(string[] args)
    {
        string? configPath = null;
        string? archiveRootOverride = null;
        string? mountRootOverride = null;
        string? mountScriptOverride = null;
        string? stagingRootOverride = null;
        string? outOverride = null;
        bool dryRun = false;
        bool harvestOnly = false;
        bool resume = false;
        bool pruneStagedClients = false;
        bool forceRestage = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--config":
                case "-c":
                    if (i + 1 < args.Length) configPath = args[++i];
                    break;
                case "--archive-root":
                case "-a":
                    if (i + 1 < args.Length) archiveRootOverride = args[++i];
                    break;
                case "--mount-root":
                    if (i + 1 < args.Length) mountRootOverride = args[++i];
                    break;
                case "--mount-script":
                    if (i + 1 < args.Length) mountScriptOverride = args[++i];
                    break;
                case "--staging-root":
                    if (i + 1 < args.Length) stagingRootOverride = args[++i];
                    break;
                case "--out":
                case "-o":
                    if (i + 1 < args.Length) outOverride = args[++i];
                    break;
                case "--dry-run":
                    dryRun = true;
                    break;
                case "--harvest-only":
                    harvestOnly = true;
                    break;
                case "--resume":
                    resume = true;
                    break;
                case "--prune-staged-clients":
                    pruneStagedClients = true;
                    break;
                case "--force-restage":
                    forceRestage = true;
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(configPath))
        {
            Console.WriteLine("ml-corpus — run the full ML dataset pipeline from a config file.");
            Console.WriteLine();
            Console.WriteLine("Usage: ml-corpus --config <corpus.json> [--archive-root <dir>] [--mount-root <dir>] [--staging-root <dir>] [--out <dir>] [--dry-run] [--harvest-only] [--resume]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --config, -c <path>          JSON config file (VlmBatchExportConfig format)");
            Console.WriteLine("  --archive-root, -a <dir>     Override or set legacy archive_root for resolving relative local client paths");
            Console.WriteLine("  --mount-root <dir>           Override or set mount_root for archive-backed client staging");
            Console.WriteLine("  --mount-script <path>        Override or set mount_script for bringing WoWArchive online");
            Console.WriteLine("  --staging-root <dir>         Override or set the local stage root for archive-backed clients");
            Console.WriteLine("  --out, -o <dir>              Override default_output_root from config");
            Console.WriteLine("  --dry-run                    Print what would run without executing");
            Console.WriteLine("  --harvest-only               Skip export, only run ml-harvest on existing datasets");
            Console.WriteLine("  --resume                     Skip fully completed map roots and only rerun incomplete export or harvest work");
            Console.WriteLine("  --prune-staged-clients       Remove stale staged archive-backed client copies after the run");
            Console.WriteLine("  --force-restage              Recopy staged archive-backed clients even when a matching stage exists");
            Console.WriteLine();
            Console.WriteLine("Config search order: <arg>, ./ml_corpus.json, <exe_dir>/ml_corpus.json");
            return 1;
        }

        if (!File.Exists(configPath))
        {
            Console.Error.WriteLine($"Config not found: {configPath}");
            return 1;
        }

        string resolvedConfigPath = Path.GetFullPath(configPath);
        string configDirectory = Path.GetDirectoryName(resolvedConfigPath) ?? Directory.GetCurrentDirectory();

        VlmBatchExportConfig config;
        try
        {
            string json = await File.ReadAllTextAsync(resolvedConfigPath);
            config = JsonSerializer.Deserialize<VlmBatchExportConfig>(json,
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })
                ?? throw new InvalidDataException("Config deserialized to null.");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to read config: {ex.Message}");
            return 1;
        }

        string? archiveRoot = ResolveMlCorpusPath(archiveRootOverride ?? config.ArchiveRoot, null, configDirectory);
        string? mountRoot = ResolveMlCorpusPath(mountRootOverride ?? config.MountRoot, null, configDirectory);
        string? mountScript = ResolveMlCorpusPath(mountScriptOverride ?? config.MountScript, null, configDirectory);
        string stagingRoot = ResolveMlCorpusPath(stagingRootOverride ?? config.StagingRoot, null, configDirectory)
            ?? Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "output", "tmp", "wowarchive-clients"));
        string? defaultOutputRoot = ResolveMlCorpusPath(outOverride ?? config.DefaultOutputRoot, null, configDirectory);
        bool shouldPruneStagedClients = pruneStagedClients || config.PruneStagedClients;
        string? legacyClientBaseRoot = archiveRoot ?? mountRoot;
        string? archiveClientBaseRoot = mountRoot ?? archiveRoot;

        int totalJobs = 0;
        int failedJobs = 0;
        var exporter = new VlmDatasetExporter();
        var harvester = new MkDatasetHarvester();
        var progress = new Progress<string>(msg => Console.WriteLine($"  {msg}"));
        var stagedLabelsToKeep = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var client in config.Clients)
        {
            string clientFolderName = !string.IsNullOrWhiteSpace(client.Label)
                ? client.Label
                : client.ClientVersion.Replace('.', '_');

            MlCorpusWorkingRootResolution clientRoot = ResolveMlCorpusWorkingRoot(
                label: clientFolderName,
                directPath: client.ClientPath,
                directBaseRoot: legacyClientBaseRoot,
                localPath: client.LocalClientPath,
                archivePath: client.ArchiveClientPath,
                archiveBaseRoot: archiveClientBaseRoot,
                mountRoot: mountRoot,
                mountScript: mountScript,
                stagingRoot: stagingRoot,
                forceRestage: forceRestage,
                dryRun: dryRun,
                configDirectory: configDirectory);

            string clientPath = clientRoot.WorkingPath;
            string discoveryClientPath = dryRun && clientRoot.Staged && !Directory.Exists(clientPath)
                ? clientRoot.SourcePath
                : clientPath;

            MlCorpusWorkingRootResolution? minimapRootResolution = null;
            string? minimapRoot = null;
            if (!string.IsNullOrWhiteSpace(client.MinimapRoot)
                || !string.IsNullOrWhiteSpace(client.LocalMinimapRoot)
                || !string.IsNullOrWhiteSpace(client.ArchiveMinimapRoot))
            {
                minimapRootResolution = ResolveMlCorpusWorkingRoot(
                    label: $"{clientFolderName}-minimap",
                    directPath: client.MinimapRoot,
                    directBaseRoot: legacyClientBaseRoot,
                    localPath: client.LocalMinimapRoot,
                    archivePath: client.ArchiveMinimapRoot,
                    archiveBaseRoot: archiveClientBaseRoot,
                    mountRoot: mountRoot,
                    mountScript: mountScript,
                    stagingRoot: stagingRoot,
                    forceRestage: forceRestage,
                    dryRun: dryRun,
                    configDirectory: configDirectory);
                minimapRoot = minimapRootResolution.WorkingPath;
            }

            if (clientRoot.Staged)
                stagedLabelsToKeep.Add(clientFolderName);

            if (minimapRootResolution is not null && minimapRootResolution.Staged)
                stagedLabelsToKeep.Add($"{clientFolderName}-minimap");

            string clientOutputRoot = !string.IsNullOrWhiteSpace(client.OutputRoot)
                ? ResolveMlCorpusPath(client.OutputRoot, null, configDirectory)!
                : !string.IsNullOrWhiteSpace(defaultOutputRoot)
                    ? Path.Combine(defaultOutputRoot, clientFolderName)
                    : Path.Combine("datasets", clientFolderName);

            List<string> mapsToProcess;
            if (client.AllMaps)
            {
                mapsToProcess = DiscoverClientMapDirectories(discoveryClientPath, listfilePath: null);
                if (mapsToProcess.Count == 0)
                {
                    if (dryRun)
                    {
                        Console.WriteLine($"[dry-run] No maps discovered for {clientFolderName} at {discoveryClientPath}");
                    }
                    else
                    {
                        Console.Error.WriteLine($"FAILED: No maps discovered for {clientFolderName} at {discoveryClientPath}");
                        failedJobs++;
                    }

                    continue;
                }

                Console.WriteLine($"Discovered {mapsToProcess.Count} maps for {clientFolderName}");
            }
            else
            {
                mapsToProcess = client.Maps;
            }

            foreach (string map in mapsToProcess)
            {
                totalJobs++;
                string mapOutput = Path.GetFullPath(Path.Combine(clientOutputRoot, map));
                bool skipExport = false;

                if (resume && !harvestOnly)
                {
                    MlCorpusResumeDecision resumeDecision = EvaluateMlCorpusResume(
                        mapOutput,
                        clientFolderName,
                        client.ClientVersion,
                        map,
                        harvestRequested: true,
                        client.GenerateDepth,
                        client.TileLimit,
                        client.InterestingOnly,
                        client.InterestingMinScore,
                        client.SkipDerivedAssets,
                        minimapRoot);

                    if (resumeDecision.Kind == MlCorpusResumeDecisionKind.SkipAll)
                    {
                        Console.WriteLine();
                        Console.WriteLine($"[{totalJobs}] {client.ClientVersion} / {map}");
                        Console.WriteLine($"    output : {mapOutput}");
                        Console.WriteLine($"    => resume skip ({resumeDecision.Reason})");
                        continue;
                    }

                    if (resumeDecision.Kind == MlCorpusResumeDecisionKind.RunHarvestOnly)
                    {
                        skipExport = true;
                    }
                }

                Console.WriteLine();
                Console.WriteLine($"[{totalJobs}] {client.ClientVersion} / {map}");
                Console.WriteLine($"    client : {clientPath} [{clientRoot.SourceType}]");
                if (clientRoot.Staged)
                    Console.WriteLine($"    source : {clientRoot.SourcePath}");
                if (!string.IsNullOrWhiteSpace(minimapRoot))
                {
                    string minimapMode = minimapRootResolution?.SourceType ?? "direct";
                    Console.WriteLine($"    minimap: {minimapRoot} [{minimapMode}]");
                }
                Console.WriteLine($"    output : {mapOutput}");

                if (dryRun)
                    continue;

                try
                {
                    if (!harvestOnly && !skipExport)
                    {
                        Console.WriteLine("    => ml-export");
                        int tileLimit = client.TileLimit ?? int.MaxValue;
                        var exportResult = await exporter.ExportMapAsync(
                            clientPath,
                            map,
                            mapOutput,
                            progress,
                            tileLimit,
                            listfilePath: null,
                            generateDepth: client.GenerateDepth,
                            minimapRoot: minimapRoot,
                            tileFilter: null,
                            skipDerivedAssets: client.SkipDerivedAssets,
                            interestingOnly: client.InterestingOnly,
                            interestingMinScore: client.InterestingMinScore);
                        Console.WriteLine($"    exported {exportResult.TilesExported} tiles, skipped {exportResult.TilesSkipped}");

                        WriteMlCorpusResumeState(
                            mapOutput,
                            new MlCorpusResumeState
                            {
                                ClientLabel = clientFolderName,
                                ClientVersion = client.ClientVersion,
                                MapName = map,
                                HarvestRequested = true,
                                ExportCompleted = true,
                                HarvestCompleted = false,
                                GenerateDepth = client.GenerateDepth,
                                TileLimit = client.TileLimit,
                                InterestingOnly = client.InterestingOnly,
                                InterestingMinScore = client.InterestingMinScore,
                                SkipDerivedAssets = client.SkipDerivedAssets,
                                MinimapRoot = NormalizeMlCorpusResumePath(minimapRoot),
                                TileJsonCount = CountMlCorpusDatasetJsonFiles(mapOutput),
                                UpdatedAtUtc = DateTime.UtcNow
                            });
                    }
                    else if (skipExport)
                    {
                        Console.WriteLine("    => ml-export skipped (resume state marked export complete)");
                    }

                    string datasetJsonDir = Path.Combine(mapOutput, "dataset");
                    int datasetJsonCount = CountMlCorpusDatasetJsonFiles(mapOutput);
                    bool hasDatasetJson = Directory.Exists(datasetJsonDir) && datasetJsonCount > 0;
                    if (!hasDatasetJson)
                    {
                        Console.WriteLine("    => ml-harvest skipped (no tile JSON files found)");
                        continue;
                    }

                    Console.WriteLine("    => ml-harvest");
                    var harvestResult = await harvester.HarvestAsync(
                        new MkDatasetHarvestOptions(DatasetRoot: mapOutput), progress);
                    Console.WriteLine($"    harvested {harvestResult.TilesProcessed} tiles");

                    WriteMlCorpusResumeState(
                        mapOutput,
                        new MlCorpusResumeState
                        {
                            ClientLabel = clientFolderName,
                            ClientVersion = client.ClientVersion,
                            MapName = map,
                            HarvestRequested = true,
                            ExportCompleted = true,
                            HarvestCompleted = true,
                            GenerateDepth = client.GenerateDepth,
                            TileLimit = client.TileLimit,
                            InterestingOnly = client.InterestingOnly,
                            InterestingMinScore = client.InterestingMinScore,
                            SkipDerivedAssets = client.SkipDerivedAssets,
                            MinimapRoot = NormalizeMlCorpusResumePath(minimapRoot),
                            TileJsonCount = datasetJsonCount,
                            UpdatedAtUtc = DateTime.UtcNow
                        });
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"    FAILED: {ex.Message}");
                    failedJobs++;
                }
            }
        }

        if (shouldPruneStagedClients)
        {
            List<string> removedStages = RemoveStaleMlCorpusStages(stagingRoot, stagedLabelsToKeep, dryRun);
            if (removedStages.Count > 0)
                Console.WriteLine($"Pruned {removedStages.Count} stale staged client(s).");
        }

        Console.WriteLine();
        Console.WriteLine($"ml-corpus done. {totalJobs - failedJobs}/{totalJobs} jobs succeeded.");
        return failedJobs == 0 ? 0 : 1;
    }

    private const string MlCorpusStageMetadataFileName = ".wowarchive-stage.json";
    private const string MlCorpusResumeStateFileName = ".ml-corpus-resume-state.json";

    private sealed record MlCorpusStageMetadata(string Label, string SourcePath, string SourceType, string UpdatedAtUtc);

    private sealed record MlCorpusWorkingRootResolution(string WorkingPath, string SourcePath, string SourceType, bool Staged, string? StagePath);

    private enum MlCorpusResumeDecisionKind
    {
        RunExport,
        RunHarvestOnly,
        SkipAll,
    }

    private sealed record MlCorpusResumeDecision(MlCorpusResumeDecisionKind Kind, string Reason);

    private sealed class MlCorpusResumeState
    {
        public string SchemaVersion { get; set; } = "ml-corpus-resume-state.v1";
        public string ClientLabel { get; set; } = string.Empty;
        public string ClientVersion { get; set; } = string.Empty;
        public string MapName { get; set; } = string.Empty;
        public bool HarvestRequested { get; set; }
        public bool ExportCompleted { get; set; }
        public bool HarvestCompleted { get; set; }
        public bool GenerateDepth { get; set; }
        public int? TileLimit { get; set; }
        public bool InterestingOnly { get; set; }
        public int InterestingMinScore { get; set; }
        public bool SkipDerivedAssets { get; set; }
        public string? MinimapRoot { get; set; }
        public int TileJsonCount { get; set; }
        public DateTime UpdatedAtUtc { get; set; }
    }

    private sealed class MlCorpusManifestCoverage
    {
        public int TilesProcessed { get; set; }
    }

    private sealed class MlCorpusManifestEnvelope
    {
        public MlCorpusManifestCoverage? Coverage { get; set; }
    }

    private static readonly JsonSerializerOptions MlCorpusResumeJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
    };

    private static string? ResolveMlCorpusPath(string? value, string? baseRoot, string configDirectory)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        if (Path.IsPathRooted(value))
            return Path.GetFullPath(value);

        if (!string.IsNullOrWhiteSpace(baseRoot))
            return Path.GetFullPath(Path.Combine(baseRoot, value));

        return Path.GetFullPath(Path.Combine(configDirectory, value));
    }

    private static string? NormalizeMlCorpusResumePath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return Path.GetFullPath(value).Replace('\\', '/');
    }

    private static IEnumerable<string> EnumerateMlCorpusDatasetJsonFiles(string mapOutput)
    {
        string datasetJsonDir = Path.Combine(mapOutput, "dataset");
        if (!Directory.Exists(datasetJsonDir))
            yield break;

        foreach (string path in Directory.EnumerateFiles(datasetJsonDir, "*.json", SearchOption.TopDirectoryOnly))
        {
            if (string.Equals(Path.GetFileName(path), "texture_database.json", StringComparison.OrdinalIgnoreCase))
                continue;

            yield return path;
        }
    }

    private static int CountMlCorpusDatasetJsonFiles(string mapOutput)
        => EnumerateMlCorpusDatasetJsonFiles(mapOutput).Count();

    private static bool IsMlCorpusManifestCurrent(string mapOutput, int datasetJsonCount)
    {
        if (datasetJsonCount == 0)
            return false;

        string manifestPath = Path.Combine(mapOutput, "ml_dataset_manifest.json");
        if (!File.Exists(manifestPath))
            return false;

        try
        {
            string json = File.ReadAllText(manifestPath);
            MlCorpusManifestEnvelope? manifest = JsonSerializer.Deserialize<MlCorpusManifestEnvelope>(json, MlCorpusResumeJsonOptions);
            if (manifest?.Coverage is null || manifest.Coverage.TilesProcessed != datasetJsonCount)
                return false;

            DateTime manifestWriteTimeUtc = File.GetLastWriteTimeUtc(manifestPath);
            DateTime latestDatasetWriteTimeUtc = EnumerateMlCorpusDatasetJsonFiles(mapOutput)
                .Select(File.GetLastWriteTimeUtc)
                .DefaultIfEmpty(DateTime.MinValue)
                .Max();
            return manifestWriteTimeUtc >= latestDatasetWriteTimeUtc;
        }
        catch
        {
            return false;
        }
    }

    private static MlCorpusResumeState? TryReadMlCorpusResumeState(string mapOutput)
    {
        string statePath = Path.Combine(mapOutput, MlCorpusResumeStateFileName);
        if (!File.Exists(statePath))
            return null;

        try
        {
            string json = File.ReadAllText(statePath);
            return JsonSerializer.Deserialize<MlCorpusResumeState>(json, MlCorpusResumeJsonOptions);
        }
        catch
        {
            return null;
        }
    }

    private static void WriteMlCorpusResumeState(string mapOutput, MlCorpusResumeState state)
    {
        Directory.CreateDirectory(mapOutput);
        string statePath = Path.Combine(mapOutput, MlCorpusResumeStateFileName);
        string json = JsonSerializer.Serialize(state, MlCorpusResumeJsonOptions);
        File.WriteAllText(statePath, json);
    }

    private static bool MlCorpusResumeStateMatches(
        MlCorpusResumeState? state,
        string clientLabel,
        string clientVersion,
        string mapName,
        bool generateDepth,
        int? tileLimit,
        bool interestingOnly,
        int interestingMinScore,
        bool skipDerivedAssets,
        string? minimapRoot)
    {
        if (state is null)
            return false;

        return string.Equals(state.SchemaVersion, "ml-corpus-resume-state.v1", StringComparison.Ordinal)
            && string.Equals(state.ClientLabel, clientLabel, StringComparison.OrdinalIgnoreCase)
            && string.Equals(state.ClientVersion, clientVersion, StringComparison.OrdinalIgnoreCase)
            && string.Equals(state.MapName, mapName, StringComparison.OrdinalIgnoreCase)
            && state.GenerateDepth == generateDepth
            && state.TileLimit == tileLimit
            && state.InterestingOnly == interestingOnly
            && state.InterestingMinScore == interestingMinScore
            && state.SkipDerivedAssets == skipDerivedAssets
            && string.Equals(state.MinimapRoot, NormalizeMlCorpusResumePath(minimapRoot), StringComparison.OrdinalIgnoreCase);
    }

    private static MlCorpusResumeDecision EvaluateMlCorpusResume(
        string mapOutput,
        string clientLabel,
        string clientVersion,
        string mapName,
        bool harvestRequested,
        bool generateDepth,
        int? tileLimit,
        bool interestingOnly,
        int interestingMinScore,
        bool skipDerivedAssets,
        string? minimapRoot)
    {
        int datasetJsonCount = CountMlCorpusDatasetJsonFiles(mapOutput);
        bool manifestCurrent = IsMlCorpusManifestCurrent(mapOutput, datasetJsonCount);
        MlCorpusResumeState? state = TryReadMlCorpusResumeState(mapOutput);
        bool stateMatches = MlCorpusResumeStateMatches(
            state,
            clientLabel,
            clientVersion,
            mapName,
            generateDepth,
            tileLimit,
            interestingOnly,
            interestingMinScore,
            skipDerivedAssets,
            minimapRoot);

        if (stateMatches && state!.ExportCompleted)
        {
            if (!harvestRequested)
                return new MlCorpusResumeDecision(MlCorpusResumeDecisionKind.SkipAll, "resume state already marks export complete for the same job settings");

            if (state.HarvestCompleted && manifestCurrent)
                return new MlCorpusResumeDecision(MlCorpusResumeDecisionKind.SkipAll, "resume state and manifest already mark this map complete");

            return new MlCorpusResumeDecision(MlCorpusResumeDecisionKind.RunHarvestOnly, "resume state marks export complete but harvest metadata is missing or stale");
        }

        if (manifestCurrent)
            return new MlCorpusResumeDecision(MlCorpusResumeDecisionKind.SkipAll, "existing manifest is current for this dataset root");

        return new MlCorpusResumeDecision(MlCorpusResumeDecisionKind.RunExport, "no matching completion state was found");
    }

    private static bool IsPathWithinRoot(string? path, string? root)
    {
        if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(root))
            return false;

        string fullPath = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        if (fullPath.Equals(fullRoot, StringComparison.OrdinalIgnoreCase))
            return true;

        if (fullPath.Length <= fullRoot.Length)
            return false;

        string prefix = fullRoot + Path.DirectorySeparatorChar;
        return fullPath.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
    }

    private static void EnsureMlCorpusArchiveMounted(string? mountRoot, string? mountScript, bool dryRun)
    {
        if (string.IsNullOrWhiteSpace(mountRoot) || Directory.Exists(mountRoot))
            return;

        if (string.IsNullOrWhiteSpace(mountScript))
            throw new InvalidOperationException($"Archive mount root '{mountRoot}' is unavailable and no mount script was provided.");

        if (!File.Exists(mountScript))
            throw new FileNotFoundException($"Archive mount script not found: {mountScript}");

        Console.WriteLine($"Mounting WoWArchive via {mountScript}");
        if (dryRun)
            return;

        string workingDirectory = Path.GetDirectoryName(mountScript) ?? Directory.GetCurrentDirectory();
        var processStartInfo = new ProcessStartInfo("cmd.exe")
        {
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        processStartInfo.ArgumentList.Add("/c");
        processStartInfo.ArgumentList.Add(mountScript);

        using Process process = Process.Start(processStartInfo)
            ?? throw new InvalidOperationException($"Failed to start mount script: {mountScript}");

        string stdOut = process.StandardOutput.ReadToEnd();
        string stdErr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (!string.IsNullOrWhiteSpace(stdOut))
            Console.Write(stdOut);
        if (!string.IsNullOrWhiteSpace(stdErr))
            Console.Error.Write(stdErr);

        if (process.ExitCode != 0)
            throw new InvalidOperationException($"Mount script failed with exit code {process.ExitCode}: {mountScript}");

        if (!Directory.Exists(mountRoot))
            throw new DirectoryNotFoundException($"Archive mount root still not available after mount script completed: {mountRoot}");
    }

    private static MlCorpusWorkingRootResolution ResolveMlCorpusWorkingRoot(
        string label,
        string? directPath,
        string? directBaseRoot,
        string? localPath,
        string? archivePath,
        string? archiveBaseRoot,
        string? mountRoot,
        string? mountScript,
        string? stagingRoot,
        bool forceRestage,
        bool dryRun,
        string configDirectory)
    {
        string? resolvedDirectPath = ResolveMlCorpusPath(directPath, directBaseRoot, configDirectory);
        string? resolvedLocalPath = ResolveMlCorpusPath(localPath, null, configDirectory);
        string? resolvedArchivePath = ResolveMlCorpusPath(archivePath, archiveBaseRoot, configDirectory);

        if (!string.IsNullOrWhiteSpace(resolvedLocalPath) && Directory.Exists(resolvedLocalPath))
            return new MlCorpusWorkingRootResolution(resolvedLocalPath, resolvedLocalPath, "local-fixed", false, null);

        if (!string.IsNullOrWhiteSpace(resolvedArchivePath))
            return StageMlCorpusArchiveRoot(label, resolvedArchivePath, mountRoot, mountScript, stagingRoot, forceRestage, dryRun);

        if (!string.IsNullOrWhiteSpace(resolvedDirectPath) && IsPathWithinRoot(resolvedDirectPath, mountRoot))
            return StageMlCorpusArchiveRoot(label, resolvedDirectPath, mountRoot, mountScript, stagingRoot, forceRestage, dryRun);

        if (!string.IsNullOrWhiteSpace(resolvedDirectPath) && (dryRun || Directory.Exists(resolvedDirectPath)))
            return new MlCorpusWorkingRootResolution(resolvedDirectPath, resolvedDirectPath, "direct", false, null);

        var attemptedPaths = new[] { resolvedLocalPath, resolvedArchivePath, resolvedDirectPath }
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Cast<string>()
            .ToArray();

        throw new DirectoryNotFoundException(
            attemptedPaths.Length > 0
                ? $"Could not resolve a usable client root for '{label}'. Checked: {string.Join("; ", attemptedPaths)}"
                : $"Could not resolve a usable client root for '{label}'. No client path values were provided.");
    }

    private static MlCorpusWorkingRootResolution StageMlCorpusArchiveRoot(
        string label,
        string sourcePath,
        string? mountRoot,
        string? mountScript,
        string? stagingRoot,
        bool forceRestage,
        bool dryRun)
    {
        string resolvedSourcePath = Path.GetFullPath(sourcePath);
        if (IsPathWithinRoot(resolvedSourcePath, mountRoot) && !Directory.Exists(resolvedSourcePath))
            EnsureMlCorpusArchiveMounted(mountRoot, mountScript, dryRun);

        if (!Directory.Exists(resolvedSourcePath) && !dryRun)
            throw new DirectoryNotFoundException($"Archive-backed client root not found: {resolvedSourcePath}");

        string resolvedStagingRoot = !string.IsNullOrWhiteSpace(stagingRoot)
            ? Path.GetFullPath(stagingRoot)
            : Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "output", "tmp", "wowarchive-clients"));

        string stagePath = Path.Combine(resolvedStagingRoot, label);
        MlCorpusStageMetadata? existingMetadata = Directory.Exists(stagePath)
            ? TryReadMlCorpusStageMetadata(stagePath)
            : null;

        bool sourceMatchesMetadata = existingMetadata is not null
            && Path.GetFullPath(existingMetadata.SourcePath).Equals(resolvedSourcePath, StringComparison.OrdinalIgnoreCase);

        bool shouldRestage = forceRestage || !Directory.Exists(stagePath) || !sourceMatchesMetadata;
        if (shouldRestage)
        {
            Console.WriteLine($"Staging archive-backed client: {resolvedSourcePath} -> {stagePath}");
            if (!dryRun)
            {
                if (Directory.Exists(stagePath))
                    DeleteMlCorpusStageDirectory(stagePath, resolvedStagingRoot);

                CopyMlCorpusDirectory(resolvedSourcePath, stagePath);
                WriteMlCorpusStageMetadata(stagePath, label, resolvedSourcePath, "archive-staged");
            }
        }
        else
        {
            Console.WriteLine($"Reusing staged client: {stagePath}");
        }

        if (!dryRun && !shouldRestage)
            WriteMlCorpusStageMetadata(stagePath, label, resolvedSourcePath, "archive-staged");

        return new MlCorpusWorkingRootResolution(stagePath, resolvedSourcePath, "archive-staged", true, stagePath);
    }

    private static List<string> RemoveStaleMlCorpusStages(string stagingRoot, IEnumerable<string> keepLabels, bool dryRun)
    {
        string resolvedStagingRoot = Path.GetFullPath(stagingRoot);
        if (!Directory.Exists(resolvedStagingRoot))
            return new List<string>();

        var keepSet = new HashSet<string>(keepLabels.Where(label => !string.IsNullOrWhiteSpace(label)), StringComparer.OrdinalIgnoreCase);
        List<string> removed = new();

        foreach (string stageDirectory in Directory.EnumerateDirectories(resolvedStagingRoot))
        {
            string label = Path.GetFileName(stageDirectory);
            if (keepSet.Contains(label))
                continue;

            if (!File.Exists(GetMlCorpusStageMetadataPath(stageDirectory)))
                continue;

            Console.WriteLine($"Removing staged client: {stageDirectory}");
            if (!dryRun)
                DeleteMlCorpusStageDirectory(stageDirectory, resolvedStagingRoot);

            removed.Add(stageDirectory);
        }

        return removed;
    }

    private static void CopyMlCorpusDirectory(string sourcePath, string destinationPath)
    {
        Directory.CreateDirectory(destinationPath);

        foreach (string directory in Directory.EnumerateDirectories(sourcePath, "*", SearchOption.AllDirectories))
        {
            string relativePath = Path.GetRelativePath(sourcePath, directory);
            Directory.CreateDirectory(Path.Combine(destinationPath, relativePath));
        }

        foreach (string file in Directory.EnumerateFiles(sourcePath, "*", SearchOption.AllDirectories))
        {
            string relativePath = Path.GetRelativePath(sourcePath, file);
            string destinationFile = Path.Combine(destinationPath, relativePath);
            string? destinationDirectory = Path.GetDirectoryName(destinationFile);
            if (!string.IsNullOrWhiteSpace(destinationDirectory))
                Directory.CreateDirectory(destinationDirectory);

            File.Copy(file, destinationFile, overwrite: true);
        }
    }

    private static void DeleteMlCorpusStageDirectory(string stagePath, string stagingRoot)
    {
        if (!IsPathWithinRoot(stagePath, stagingRoot))
            throw new InvalidOperationException($"Refusing to remove stage path outside staging root: {stagePath}");

        if (Directory.Exists(stagePath))
            Directory.Delete(stagePath, recursive: true);
    }

    private static string GetMlCorpusStageMetadataPath(string stagePath)
        => Path.Combine(stagePath, MlCorpusStageMetadataFileName);

    private static MlCorpusStageMetadata? TryReadMlCorpusStageMetadata(string stagePath)
    {
        string metadataPath = GetMlCorpusStageMetadataPath(stagePath);
        if (!File.Exists(metadataPath))
            return null;

        string rawJson = File.ReadAllText(metadataPath);
        if (string.IsNullOrWhiteSpace(rawJson))
            return null;

        try
        {
            return JsonSerializer.Deserialize<MlCorpusStageMetadata>(rawJson);
        }
        catch
        {
            return null;
        }
    }

    private static void WriteMlCorpusStageMetadata(string stagePath, string label, string sourcePath, string sourceType)
    {
        Directory.CreateDirectory(stagePath);
        MlCorpusStageMetadata metadata = new(
            label,
            Path.GetFullPath(sourcePath),
            sourceType,
            DateTime.UtcNow.ToString("o"));

        string metadataPath = GetMlCorpusStageMetadataPath(stagePath);
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static async Task<int> RunConvertAsync(string[] args)
    {
        string? inputPath = null;
        string? outputDir = null;
        string? alphaClientPath = null;
        string? lkClientPath = null;
        string? crosswalkDir = null;
        string? communityListfile = null;
        string? lkListfile = null;
        string? wmoDir = null;
        bool fuzzy = false;
        bool verbose = false;
        bool convertWmos = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputDir = args[++i];
                    break;
                case "--crosswalk":
                    if (i + 1 < args.Length) crosswalkDir = args[++i];
                    break;
                case "--alpha-client":
                    if (i + 1 < args.Length) alphaClientPath = args[++i];
                    break;
                case "--lk-client":
                    if (i + 1 < args.Length) lkClientPath = args[++i];
                    break;
                case "--listfile":
                    if (i + 1 < args.Length) communityListfile = args[++i];
                    break;
                case "--lk-listfile":
                    if (i + 1 < args.Length) lkListfile = args[++i];
                    break;
                case "--fuzzy":
                    fuzzy = true;
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                case "--convert-wmos":
                    convertWmos = true;
                    break;
                case "--wmo-dir":
                    if (i + 1 < args.Length) wmoDir = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input path required");
            return 1;
        }

        outputDir ??= Path.Combine(Directory.GetCurrentDirectory(), "output");

        var options = new ConversionOptions
        {
            AlphaClientPath = alphaClientPath,
            LkClientPath = lkClientPath,
            CrosswalkDirectory = crosswalkDir,
            CommunityListfile = communityListfile,
            LkListfile = lkListfile,
            FuzzyAssetMatching = fuzzy,
            Verbose = verbose,
            ConvertWmos = convertWmos,
            AlphaWmoDirectory = wmoDir
        };

        var converter = new AlphaToLkConverter(options);

        Console.WriteLine("WoW Map Converter v3");
        Console.WriteLine("====================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputDir)}");
        Console.WriteLine();

        var result = await converter.ConvertWdtAsync(inputPath, outputDir);

        if (result.Success)
        {
            Console.WriteLine($"✓ Conversion completed in {result.ElapsedMs}ms");
            Console.WriteLine($"  Map: {result.MapName}");
            Console.WriteLine($"  Tiles: {result.TilesConverted}/{result.TotalTiles}");
            return 0;
        }
        else
        {
            Console.Error.WriteLine($"✗ Conversion failed: {result.Error}");
            return 1;
        }
    }

    private static int RunConvertWmo(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--mode":
                case "--extended":
                    Console.Error.WriteLine("Error: convert-wmo no longer supports alternate modes. The maintained converter path is always used.");
                    return 1;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input WMO path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".v17.wmo");

        Console.WriteLine("WMO v14 → v17 Converter");
        Console.WriteLine("=======================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new WmoV14ToV17Converter();
            List<string> textures = converter.Convert(inputPath, outputPath);
            
            // Auto-copy textures
            CopyTextures(inputPath, outputPath, textures);
            
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static void CopyTextures(string inputWmoPath, string outputWmoPath, List<string> textures)
    {
        if (textures == null || textures.Count == 0) return;

        string inputDir = Path.GetDirectoryName(Path.GetFullPath(inputWmoPath))!;
        string outputDir = Path.GetDirectoryName(Path.GetFullPath(outputWmoPath))!;

        // Attempt to find the "Root" data directory by looking for the first part of the texture path
        // e.g. if tex is "World/wmos/...", we look for a "World" folder in inputDir or its parents.
        
        Console.WriteLine($"[INFO] wmo references {textures.Count} textures. Copying...");
        foreach (var t in textures) Console.WriteLine($"  - {t}");

        // Simple heuristic: Try to find the file relative to inputDir, then check parents
        foreach (var tex in textures)
        {
            var cleanTex = tex.Replace('/', '\\');
            string srcPath = null;
            
            // 1. Try relative to wmo file itself (unlikely but possible)
            var p1 = Path.Combine(inputDir, cleanTex);
            if (File.Exists(p1)) srcPath = p1;
            else
            {
                // 2. Walk up 5 levels to find the root
                var curr = new DirectoryInfo(inputDir);
                DirectoryInfo rootDir = null;
                for (int i = 0; i < 5 && curr != null; i++)
                {
                   var p2 = Path.Combine(curr.FullName, cleanTex);
                   if (File.Exists(p2))
                   {
                       srcPath = p2;
                       break;
                   }
                   if (Directory.Exists(Path.Combine(curr.FullName, "DUNGEONS")) || 
                       Directory.Exists(Path.Combine(curr.FullName, "World")) ||
                       Directory.Exists(Path.Combine(curr.FullName, "Textures")))
                   {
                       rootDir = curr;
                   }
                   curr = curr.Parent;
                }

                // 3. Fallback: Recursive search in Root if identified, or InputDir parents
                if (srcPath == null)
                {
                     var searchRoot = rootDir ?? new DirectoryInfo(inputDir).Parent?.Parent;
                     if (searchRoot != null && searchRoot.Exists)
                     {
                         var filename = Path.GetFileName(cleanTex);
                         var found = Directory.EnumerateFiles(searchRoot.FullName, filename, SearchOption.AllDirectories).FirstOrDefault();
                         if (found != null)
                         {
                             srcPath = found;
                             Console.WriteLine($"    Found via search: {srcPath}");
                         }
                     }
                }
            } // End else

            if (srcPath != null)
            {
                // Copy to output
                // Preserve directory structure matches WMO path
                string targetRelPath = cleanTex;
                var destPath = Path.Combine(outputDir, targetRelPath);
                
                Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);
                if (!File.Exists(destPath)) 
                {
                    File.Copy(srcPath, destPath, true);
                }
            }
            else
            {
                Console.WriteLine($"  [WARN] Missing texture: {cleanTex}");
            }
        }
    }

    private static int RunConvertMdx(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input MDX path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".m2");

        Console.WriteLine("MDX → M2 Converter");
        Console.WriteLine("==================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new MdxToM2Converter();
            converter.Convert(inputPath, outputPath);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> RunConvertLkToAlphaAsync(string[] args)
    {
        string? wdtPath = null;
        string? mapDir = null;
        string? outputPath = null;
        bool verbose = false;
        bool skipM2 = false;
        bool skipWmo = false;
        bool convertLiquids = true;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--wdt":
                    if (i + 1 < args.Length) wdtPath = args[++i];
                    break;
                case "--map-dir":
                    if (i + 1 < args.Length) mapDir = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                case "--skip-m2":
                    skipM2 = true;
                    break;
                case "--skip-wmo":
                    skipWmo = true;
                    break;
                case "--no-liquids":
                    convertLiquids = false;
                    break;
                default:
                    if (!args[i].StartsWith("-") && wdtPath == null)
                        wdtPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(wdtPath))
        {
            Console.Error.WriteLine("Error: LK WDT path required (--wdt or first positional arg)");
            return 1;
        }

        mapDir ??= Path.GetDirectoryName(wdtPath) ?? ".";
        outputPath ??= Path.Combine(Directory.GetCurrentDirectory(), "alpha_output", 
            Path.GetFileNameWithoutExtension(wdtPath) + ".wdt");

        var options = new LkToAlphaOptions
        {
            Verbose = verbose,
            SkipM2 = skipM2,
            SkipWmo = skipWmo,
            ConvertLiquids = convertLiquids
        };

        var converter = new LkToAlphaConverter(options);

        Console.WriteLine("LK → Alpha Converter");
        Console.WriteLine("====================");
        Console.WriteLine($"WDT:     {Path.GetFullPath(wdtPath)}");
        Console.WriteLine($"Map Dir: {Path.GetFullPath(mapDir)}");
        Console.WriteLine($"Output:  {Path.GetFullPath(outputPath)}");
        Console.WriteLine();

        var result = await converter.ConvertAsync(wdtPath, mapDir, outputPath);

        if (result.Success)
        {
            Console.WriteLine($"✓ Conversion completed in {result.ElapsedMs}ms");
            Console.WriteLine($"  Map: {result.MapName}");
            Console.WriteLine($"  Tiles: {result.TilesConverted}/{result.TotalTiles}");
            if (result.Warnings.Count > 0)
            {
                Console.WriteLine($"  Warnings: {result.Warnings.Count}");
                foreach (var w in result.Warnings.Take(5))
                    Console.WriteLine($"    - {w}");
                if (result.Warnings.Count > 5)
                    Console.WriteLine($"    ... and {result.Warnings.Count - 5} more");
            }
            return 0;
        }
        else
        {
            Console.Error.WriteLine($"✗ Conversion failed: {result.Error}");
            return 1;
        }
    }

    private static int RunConvertWmoToAlpha(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input WMO v17 path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".v14.wmo");

        Console.WriteLine("WMO v17 → v14 Converter");
        Console.WriteLine("=======================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new WmoV17ToV14Converter();
            converter.Convert(inputPath, outputPath);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunConvertM2ToMdx(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input M2 path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".mdx");

        Console.WriteLine("M2 → MDX Converter");
        Console.WriteLine("==================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new M2ToMdxConverter();
            converter.Convert(inputPath, outputPath);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunPm4Export(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input PM4 path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".obj");

        Console.WriteLine("PM4 → OBJ Exporter");
        Console.WriteLine("==================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var pm4 = WoWMapConverter.Core.Formats.PM4.Pm4File.FromFile(inputPath);
            Console.WriteLine($"  Version: {pm4.Version}");
            Console.WriteLine($"  Mesh Vertices: {pm4.MeshVertices.Count}");
            Console.WriteLine($"  Mesh Indices: {pm4.MeshIndices.Count}");
            Console.WriteLine($"  Surfaces: {pm4.Surfaces.Count}");
            Console.WriteLine($"  Links: {pm4.LinkEntries.Count}");
            Console.WriteLine($"  Position Refs: {pm4.PositionRefs.Count}");
            Console.WriteLine($"  Path Vertices: {pm4.PathVertices.Count}");
            Console.WriteLine($"  Exterior Vertices: {pm4.ExteriorVertices.Count}");
            Console.WriteLine($"  Chunks: {string.Join(", ", pm4.ChunkSizes.Select(kv => $"{kv.Key}:{kv.Value}"))}");

            if (pm4.MeshVertices.Count == 0)
            {
                Console.WriteLine("[WARN] No mesh vertices found - PM4 may be empty or use different chunk names");
                File.WriteAllText(outputPath, "# Empty PM4 - no mesh data\n");
                return 0;
            }

            var obj = pm4.ExportToObj();
            File.WriteAllText(outputPath, obj);
            Console.WriteLine($"[SUCCESS] Exported to: {outputPath}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            Console.Error.WriteLine($"Stack: {ex.StackTrace}");
            return 1;
        }
    }

    private static int RunPm4ValidateCoords(string[] args)
    {
        string inputDir = Pm4CoordinateService.DefaultDevelopmentMapDirectory;
        string? jsonOutputPath = null;
        int? tileLimit = null;
        float threshold = 32f;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input-dir":
                case "-i":
                    if (i + 1 < args.Length) inputDir = args[++i];
                    break;
                case "--json":
                    if (i + 1 < args.Length) jsonOutputPath = args[++i];
                    break;
                case "--tile-limit":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int parsedTileLimit))
                        tileLimit = parsedTileLimit;
                    break;
                case "--threshold":
                    if (i + 1 < args.Length && float.TryParse(args[++i], out float parsedThreshold))
                        threshold = parsedThreshold;
                    break;
            }
        }

        Console.WriteLine("PM4 Coordinate Validation");
        Console.WriteLine("=========================");
        Console.WriteLine($"Input:      {Pm4CoordinateService.ResolveMapDirectory(inputDir)}");
        Console.WriteLine($"Threshold:  {threshold:F1} units");
        if (tileLimit.HasValue)
            Console.WriteLine($"Tile limit: {tileLimit.Value}");

        try
        {
            var report = Pm4CoordinateValidator.ValidateDirectory(new Pm4CoordinateValidationOptions(
                MapDirectory: inputDir,
                TileLimit: tileLimit,
                MatchThreshold: threshold,
                TileBoundsTolerance: 2f,
                SampleCount: 3));

            Console.WriteLine();
            Console.WriteLine($"Tiles scanned:              {report.TilesScanned}");
            Console.WriteLine($"Tiles validated:            {report.TilesValidated}");
            Console.WriteLine($"Skipped without _obj0.adt:  {report.TilesSkippedMissingObj0}");
            Console.WriteLine($"Skipped without placements: {report.TilesSkippedMissingPlacements}");
            Console.WriteLine($"ADT placements:             {report.TotalPlacements}");
            Console.WriteLine($"PM4 MPRL refs:              {report.TotalPositionRefs}");

            float inTileRatio = report.TotalPositionRefs > 0
                ? (float)report.TotalInTileBounds / report.TotalPositionRefs * 100f
                : 0f;
            float matchRatio = report.TotalPositionRefs > 0
                ? (float)report.TotalMatchedWithinThreshold / report.TotalPositionRefs * 100f
                : 0f;

            Console.WriteLine($"Refs in expected tile:      {report.TotalInTileBounds} ({inTileRatio:F1}%)");
            Console.WriteLine($"Refs within threshold:      {report.TotalMatchedWithinThreshold} ({matchRatio:F1}%)");
            Console.WriteLine($"Avg nearest placement dist: {(report.AverageNearestDistance?.ToString("F2") ?? "n/a")}");

            foreach (var tile in report.Tiles
                .OrderBy(tile => tile.AverageNearestDistance ?? float.MaxValue)
                .Take(5))
            {
                Console.WriteLine();
                Console.WriteLine($"Tile {tile.TileName} ({tile.TileX},{tile.TileY})");
                Console.WriteLine($"  placements={tile.PlacementCount}, refs={tile.PositionRefCount}, inTile={tile.InTileBoundsCount}, matched={tile.MatchedWithinThresholdCount}");
                Console.WriteLine($"  avgNearest={(tile.AverageNearestDistance?.ToString("F2") ?? "n/a")}, avgMatched={(tile.AverageMatchedDistance?.ToString("F2") ?? "n/a")}");

                foreach (var sample in tile.BestMatches)
                {
                    Console.WriteLine(
                        $"  sample {sample.PlacementKind}:{sample.PlacementLabel} dist={sample.HorizontalDistance:F2} heightΔ={sample.HeightDelta:F2} ref=({sample.RefPlacementPosition.X:F2}, {sample.RefPlacementPosition.Y:F2}, {sample.RefPlacementPosition.Z:F2})");
                }
            }

            if (!string.IsNullOrEmpty(jsonOutputPath))
            {
                string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(jsonOutputPath, json);
                Console.WriteLine();
                Console.WriteLine($"JSON report: {Path.GetFullPath(jsonOutputPath)}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunWmoInfo(string[] args)
    {
        string? inputPath = null;
        bool verbose = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input WMO path required");
            return 1;
        }

        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Error: File not found: {inputPath}");
            return 1;
        }

        Console.WriteLine();
        Console.WriteLine($"WMO Analysis: {Path.GetFileName(inputPath)}");
        Console.WriteLine(new string('=', 70));

        try
        {
            // Use WmoV14ToV17Converter's ParseWmoV14 to get data
            var converter = new WmoV14ToV17Converter();
            var wmoData = converter.ParseWmoV14(inputPath);

            Console.WriteLine($"Version: v{wmoData.Version}");
            Console.WriteLine($"Groups:  {wmoData.Groups.Count}");
            Console.WriteLine($"Materials: {wmoData.Materials.Count}");
            Console.WriteLine($"Textures: {wmoData.Textures.Count}");
            Console.WriteLine($"Doodad Sets: {wmoData.DoodadSets.Count}");
            Console.WriteLine($"Portals: {wmoData.Portals.Count}");
            Console.WriteLine();

            // List groups
            Console.WriteLine("Groups:");
            Console.WriteLine(new string('-', 70));
            Console.WriteLine($"{"#",-4} {"Name",-35} {"Verts",-8} {"Faces",-8} {"Flags",-14}");
            Console.WriteLine(new string('-', 70));

            int totalVerts = 0, totalFaces = 0, emptyGroups = 0;

            for (int i = 0; i < wmoData.Groups.Count; i++)
            {
                var g = wmoData.Groups[i];
                int faceCount = g.Indices?.Count / 3 ?? 0;
                int vertCount = g.Vertices?.Count ?? 0;
                string flagsHex = $"0x{g.Flags:X8}";

                // Decode common flags
                var flagNotes = new List<string>();
                if ((g.Flags & 0x01) != 0) flagNotes.Add("BSP");
                if ((g.Flags & 0x02) != 0) flagNotes.Add("Light");
                if ((g.Flags & 0x04) != 0) flagNotes.Add("Doodads");
                if ((g.Flags & 0x08) != 0) flagNotes.Add("Liquid");
                if ((g.Flags & 0x40) != 0) flagNotes.Add("Exterior");
                if ((g.Flags & 0x2000) != 0) flagNotes.Add("ExtLit");
                if ((g.Flags & 0x80000) != 0) flagNotes.Add("Indoor");

                string displayName = string.IsNullOrEmpty(g.Name) ? $"(group_{i})" : g.Name;
                if (displayName.Length > 35) displayName = displayName.Substring(0, 32) + "...";

                Console.WriteLine($"{i,-4} {displayName,-35} {vertCount,-8} {faceCount,-8} {flagsHex}");

                if (verbose && flagNotes.Count > 0)
                {
                    Console.WriteLine($"     Flags: {string.Join(", ", flagNotes)}");
                }

                if (vertCount == 0)
                {
                    emptyGroups++;
                    if (verbose)
                        Console.WriteLine($"     ⚠️  EMPTY GROUP");
                }

                totalVerts += vertCount;
                totalFaces += faceCount;
            }

            Console.WriteLine(new string('-', 70));
            Console.WriteLine($"Totals: {totalVerts} vertices, {totalFaces} faces");

            if (emptyGroups > 0)
            {
                Console.WriteLine();
                Console.WriteLine($"⚠️  Warning: {emptyGroups} empty groups (no vertices)");
            }

            // Show group name details in verbose mode
            if (verbose && wmoData.GroupInfos.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("MOGI Group Info (parsed from root file):");
                Console.WriteLine(new string('-', 70));
                for (int i = 0; i < wmoData.GroupInfos.Count; i++)
                {
                    var info = wmoData.GroupInfos[i];
                    Console.WriteLine($"  [{i}] Flags: 0x{info.Flags:X8}, NameOfs: 0x{info.NameOffset:X4}");
                }
            }

            // Texture list
            if (verbose && wmoData.Textures.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("Textures:");
                Console.WriteLine(new string('-', 70));
                for (int i = 0; i < wmoData.Textures.Count; i++)
                {
                    Console.WriteLine($"  [{i}] {wmoData.Textures[i]}");
                }
            }

            Console.WriteLine();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error parsing WMO: {ex.Message}");
            if (verbose)
                Console.Error.WriteLine(ex.StackTrace);
            return 1;
        }
    }

    private static int RunDevelopmentAnalyze(string[] args)
    {
        string inputDir = DevelopmentMapAnalyzer.DefaultDevelopmentMapDirectory;
        string? jsonOutputPath = null;
        int? tileLimit = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input-dir":
                case "--input":
                case "-i":
                    if (i + 1 < args.Length)
                        inputDir = args[++i];
                    break;
                case "--json":
                    if (i + 1 < args.Length)
                        jsonOutputPath = args[++i];
                    break;
                case "--tile-limit":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int parsedTileLimit))
                        tileLimit = parsedTileLimit;
                    break;
                case "--help":
                case "-h":
                    Console.WriteLine("Development Dataset Analysis");
                    Console.WriteLine();
                    Console.WriteLine("Usage: wowmapconverter development-analyze [options]");
                    Console.WriteLine();
                    Console.WriteLine("Options:");
                    Console.WriteLine("  --input-dir, -i <dir>  Development source directory (default: fixed development dataset)");
                    Console.WriteLine("  --json <path>          Write the full report to JSON");
                    Console.WriteLine("  --tile-limit <n>       Analyze only the first N tiles after sorting");
                    return 0;
            }
        }

        Console.WriteLine("Development Dataset Analysis");
        Console.WriteLine("============================");
        Console.WriteLine($"Input: {Pm4CoordinateService.ResolveMapDirectory(inputDir)}");
        if (tileLimit.HasValue)
            Console.WriteLine($"Tile limit: {tileLimit.Value}");

        try
        {
            var report = DevelopmentMapAnalyzer.Analyze(inputDir, tileLimit);

            Console.WriteLine();
            Console.WriteLine($"Map files: WDT={(report.WdtExists ? "yes" : "no")}, WDL={(report.WdlExists ? "yes" : "no")}");
            Console.WriteLine($"Tiles analyzed:              {report.TilesAnalyzed}");
            Console.WriteLine($"Root ADTs:                   {report.RootAdtCount}");
            Console.WriteLine($"_obj0 ADTs:                  {report.Obj0Count}");
            Console.WriteLine($"_tex0 ADTs:                  {report.Tex0Count}");
            Console.WriteLine($"PM4 files:                   {report.Pm4Count}");
            Console.WriteLine($"WLW/WLM/WLQ/WLL tiles:       {report.WlwCount}/{report.WlmCount}/{report.WlqCount}/{report.WllCount}");
            Console.WriteLine($"Zero-byte roots:             {report.ZeroByteRootCount}");
            Console.WriteLine($"Roots missing MCIN:          {report.MissingMcinRootCount}");
            Console.WriteLine($"Roots with partial chunks:   {report.PartialMcnkRootCount}");
            Console.WriteLine($"Tiles without usable ground: {report.TilesWithoutUsableTerrain}");
            Console.WriteLine($"Tiles with bad chunk index:  {report.TilesWithHeaderIndexMismatches}");
            Console.WriteLine($"Tiles with repeated 0,0 idx: {report.TilesWithRepeatedZeroIndices}");
            Console.WriteLine($"Class healthy-split:         {report.HealthySplitCount}");
            Console.WriteLine($"Class index-corrupt:         {report.IndexCorruptCount}");
            Console.WriteLine($"Class scan-only-root:        {report.ScanOnlyRootCount}");
            Console.WriteLine($"Class wdl-rebuild:           {report.WdlRebuildCount}");
            Console.WriteLine($"Class manual-review:         {report.ManualReviewCount}");

            foreach (var tile in report.Tiles
                .Where(tile => tile.TileClass != "healthy-split" || tile.HeaderIndexMismatchCount > 0 || tile.ZeroIndexHeaderCount > 1)
                .OrderByDescending(tile => tile.HeaderIndexMismatchCount)
                .ThenByDescending(tile => tile.ZeroIndexHeaderCount)
                .ThenBy(tile => tile.TileY)
                .ThenBy(tile => tile.TileX)
                .Take(20))
            {
                Console.WriteLine();
                Console.WriteLine($"Tile {tile.TileName}");
                Console.WriteLine($"  class={tile.TileClass}");
                Console.WriteLine($"  root={tile.RootStatus}, validChunks={tile.ValidRootChunkCount}, mcin={tile.HasMcin}, topLevelMcnk={tile.TopLevelMcnkCount}");
                Console.WriteLine($"  idxMismatch={tile.HeaderIndexMismatchCount}, zeroIdx={tile.ZeroIndexHeaderCount}, dupIdx={tile.DuplicateHeaderIndexCount}");
                Console.WriteLine($"  obj0={tile.HasObj0Adt}, tex0={tile.HasTex0Adt}, pm4={tile.HasPm4}, wl={(tile.HasWlw || tile.HasWlm || tile.HasWlq || tile.HasWll)}");
                Console.WriteLine($"  action={tile.RecommendedAction}");
            }

            if (!string.IsNullOrEmpty(jsonOutputPath))
            {
                string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(jsonOutputPath, json);
                Console.WriteLine();
                Console.WriteLine($"JSON report: {Path.GetFullPath(jsonOutputPath)}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunDevelopmentRepair(string[] args)
    {
        DevelopmentRepairOptions options = DevelopmentRepairOptions.CreateDefault();
        string inputDir = options.InputDirectory;
        string outputDir = options.OutputDirectory;
        string mode = options.Mode;
        int? tileLimit = options.TileLimit;
        string? tile = options.Tile;
        bool skipWl = options.SkipWl;
        bool skipWdlGenerate = options.SkipWdlGenerate;
        bool skipPm4 = options.SkipPm4;
        string? manifestPath = options.ManifestPath;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input-dir":
                case "--input":
                case "-i":
                    if (i + 1 < args.Length)
                        inputDir = args[++i];
                    break;
                case "--output-dir":
                case "--output":
                case "-o":
                    if (i + 1 < args.Length)
                        outputDir = args[++i];
                    break;
                case "--mode":
                    if (i + 1 < args.Length)
                        mode = args[++i];
                    break;
                case "--tile":
                    if (i + 1 < args.Length)
                        tile = args[++i];
                    break;
                case "--tile-limit":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int parsedTileLimit))
                        tileLimit = parsedTileLimit;
                    break;
                case "--skip-wl":
                    skipWl = true;
                    break;
                case "--skip-wdl-generate":
                    skipWdlGenerate = true;
                    break;
                case "--skip-pm4":
                    skipPm4 = true;
                    break;
                case "--manifest":
                    if (i + 1 < args.Length)
                        manifestPath = args[++i];
                    break;
                case "--help":
                case "-h":
                    Console.WriteLine("Development Dataset Repair");
                    Console.WriteLine();
                    Console.WriteLine("Usage: wowmapconverter development-repair [options]");
                    Console.WriteLine();
                    Console.WriteLine("Options:");
                    Console.WriteLine("  --input-dir, -i <dir>    Development source directory (default: fixed development dataset)");
                    Console.WriteLine("  --output-dir, -o <dir>   Repair output root (default: output/development-repair)");
                    Console.WriteLine("  --mode <audit|repair>    audit = classify + manifests, repair = write repaired ADTs");
                    Console.WriteLine("  --tile <x_y>             Process only one tile coordinate");
                    Console.WriteLine("  --tile-limit <n>         Process only first N discovered tiles");
                    Console.WriteLine("  --skip-wl                Skip WL* to MH2O conversion");
                    Console.WriteLine("  --skip-wdl-generate      Skip WDL terrain generation fallback");
                    Console.WriteLine("  --skip-pm4               Keep PM4 refinement disabled for this first slice");
                    Console.WriteLine("  --manifest <path>        Summary manifest output path");
                    return 0;
            }
        }

        var runOptions = new DevelopmentRepairOptions(
            InputDirectory: inputDir,
            OutputDirectory: outputDir,
            Mode: mode,
            TileLimit: tileLimit,
            Tile: tile,
            SkipWl: skipWl,
            SkipWdlGenerate: skipWdlGenerate,
            SkipPm4: skipPm4,
            ManifestPath: manifestPath);

        Console.WriteLine("Development Dataset Repair");
        Console.WriteLine("==========================");
        Console.WriteLine($"Input:  {Pm4CoordinateService.ResolveMapDirectory(runOptions.InputDirectory)}");
        Console.WriteLine($"Output: {Path.GetFullPath(runOptions.OutputDirectory)}");
        Console.WriteLine($"Mode:   {runOptions.Mode}");
        if (!string.IsNullOrWhiteSpace(runOptions.Tile))
            Console.WriteLine($"Tile:   {runOptions.Tile}");
        if (runOptions.TileLimit.HasValue)
            Console.WriteLine($"Limit:  {runOptions.TileLimit.Value}");
        Console.WriteLine($"Skip WL conversion:      {(runOptions.SkipWl ? "yes" : "no")}");
        Console.WriteLine($"Skip WDL generation:     {(runOptions.SkipWdlGenerate ? "yes" : "no")}");
        Console.WriteLine($"Skip PM4 refinement:     {(runOptions.SkipPm4 ? "yes" : "no")}");

        try
        {
            DevelopmentRepairExecutionReport report = DevelopmentRepairService.Execute(runOptions);

            Console.WriteLine();
            Console.WriteLine($"Map: {report.MapName}");
            Console.WriteLine($"Tiles processed: {report.TilesProcessed}");
            Console.WriteLine($"Tiles written:   {report.TilesWritten}");
            Console.WriteLine($"WDT written:     {(report.WdtWritten ? "yes" : "no")}");
            Console.WriteLine($"WDL copied:      {(report.WdlCopied ? "yes" : "no")}");

            Console.WriteLine();
            Console.WriteLine($"Class healthy-split:  {report.Tiles.Count(t => t.TileClass == "healthy-split")}");
            Console.WriteLine($"Class index-corrupt:  {report.Tiles.Count(t => t.TileClass == "index-corrupt")}");
            Console.WriteLine($"Class scan-only-root: {report.Tiles.Count(t => t.TileClass == "scan-only-root")}");
            Console.WriteLine($"Class wdl-rebuild:    {report.Tiles.Count(t => t.TileClass == "wdl-rebuild")}");
            Console.WriteLine($"Class manual-review:  {report.Tiles.Count(t => t.TileClass == "manual-review" || t.NeedsManualReview)}");

            Console.WriteLine();
            Console.WriteLine($"Summary manifest: {report.SummaryManifestPath}");

            foreach (DevelopmentTileRepairManifest tileManifest in report.Tiles
                .Where(tileManifest => tileManifest.NeedsManualReview || tileManifest.Warnings.Count > 0)
                .Take(15))
            {
                Console.WriteLine();
                Console.WriteLine($"Tile {tileManifest.TileName}");
                Console.WriteLine($"  class={tileManifest.TileClass}, route={tileManifest.RepairRoute}, output={tileManifest.OutputWritten}");
                Console.WriteLine($"  splitMerge={tileManifest.SplitMergeRan}, indexRepair={tileManifest.ChunkIndicesRepairRan} (mismatch={tileManifest.ChunkIndexMismatchCount})");
                Console.WriteLine($"  wdlGenerate={tileManifest.WdlGenerationRan}, wlToMh2o={tileManifest.WlLiquidsConverted} (chunks={tileManifest.WlLiquidChunkCount})");
                if (tileManifest.Warnings.Count > 0)
                    Console.WriteLine($"  warning={tileManifest.Warnings[0]}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunTerrainTextureTransfer(string[] args)
    {
        TerrainTextureTransferOptions defaults = TerrainTextureTransferOptions.CreateDefault();

        string sourceDir = defaults.SourceDirectory;
        string targetDir = defaults.TargetDirectory;
        string outputDir = defaults.OutputDirectory;
        string mode = defaults.Mode;
        int? tileLimit = defaults.TileLimit;
        int? globalDeltaX = defaults.GlobalDeltaX;
        int? globalDeltaY = defaults.GlobalDeltaY;
        int chunkOffsetX = defaults.ChunkOffsetX;
        int chunkOffsetY = defaults.ChunkOffsetY;
        bool copyMtex = defaults.CopyMtex;
        bool copyMcly = defaults.CopyMcly;
        bool copyMcal = defaults.CopyMcal;
        bool copyMcsh = defaults.CopyMcsh;
        bool copyHoles = defaults.CopyHoles;
        string? manifestPath = defaults.ManifestPath;
        var pairs = new List<TerrainTilePair>();

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--source-dir":
                case "--source":
                case "-s":
                    if (i + 1 < args.Length)
                        sourceDir = args[++i];
                    break;
                case "--target-dir":
                case "--target":
                case "-t":
                    if (i + 1 < args.Length)
                        targetDir = args[++i];
                    break;
                case "--output-dir":
                case "--output":
                case "-o":
                    if (i + 1 < args.Length)
                        outputDir = args[++i];
                    break;
                case "--mode":
                    if (i + 1 < args.Length)
                        mode = args[++i];
                    break;
                case "--pair":
                    if (i + 1 < args.Length)
                    {
                        if (!TryParseTilePair(args[++i], out TerrainTilePair pair))
                        {
                            Console.Error.WriteLine($"Invalid --pair value '{args[i]}'. Expected srcX_srcY:dstX_dstY");
                            return 1;
                        }

                        pairs.Add(pair);
                    }
                    break;
                case "--global-delta":
                    if (i + 1 < args.Length)
                    {
                        if (!TryParseIntPair(args[++i], out int dx, out int dy))
                        {
                            Console.Error.WriteLine($"Invalid --global-delta value '{args[i]}'. Expected dx,dy");
                            return 1;
                        }

                        globalDeltaX = dx;
                        globalDeltaY = dy;
                    }
                    break;
                case "--chunk-offset":
                    if (i + 1 < args.Length)
                    {
                        if (!TryParseIntPair(args[++i], out int dx, out int dy))
                        {
                            Console.Error.WriteLine($"Invalid --chunk-offset value '{args[i]}'. Expected dx,dy");
                            return 1;
                        }

                        chunkOffsetX = dx;
                        chunkOffsetY = dy;
                    }
                    break;
                case "--tile-limit":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int parsedTileLimit))
                        tileLimit = parsedTileLimit;
                    break;
                case "--no-mtex":
                    copyMtex = false;
                    break;
                case "--no-mcly":
                    copyMcly = false;
                    break;
                case "--no-mcal":
                    copyMcal = false;
                    break;
                case "--no-mcsh":
                    copyMcsh = false;
                    break;
                case "--no-holes":
                    copyHoles = false;
                    break;
                case "--manifest":
                    if (i + 1 < args.Length)
                        manifestPath = args[++i];
                    break;
                case "--help":
                case "-h":
                    Console.WriteLine("Terrain Texture Transfer");
                    Console.WriteLine();
                    Console.WriteLine("Usage: wowmapconverter terrain-texture-transfer [options]");
                    Console.WriteLine();
                    Console.WriteLine("Mapping options (required):");
                    Console.WriteLine("  --pair <sx_sy:tx_ty>     Add explicit source->target tile pair (repeatable)");
                    Console.WriteLine("  --global-delta <dx,dy>   Auto-build pairs from source roots with target=(source+delta)");
                    Console.WriteLine();
                    Console.WriteLine("Core options:");
                    Console.WriteLine("  --source-dir, -s <dir>   Source map directory (default: fixed development dataset)");
                    Console.WriteLine("  --target-dir, -t <dir>   Target map directory (default: fixed development dataset)");
                    Console.WriteLine("  --output-dir, -o <dir>   Output root (default: output/terrain-texture-transfer)");
                    Console.WriteLine("  --mode <dry-run|apply>   dry-run writes manifests only; apply also writes ADTs");
                    Console.WriteLine("  --chunk-offset <dx,dy>   Source chunk remap offset per target chunk");
                    Console.WriteLine("  --tile-limit <n>         Limit number of planned pairs");
                    Console.WriteLine("  --manifest <path>        Summary manifest output path");
                    Console.WriteLine();
                    Console.WriteLine("Payload toggles:");
                    Console.WriteLine("  --no-mtex                Do not copy MTEX");
                    Console.WriteLine("  --no-mcly                Do not copy MCLY layers");
                    Console.WriteLine("  --no-mcal                Do not copy MCAL alpha maps");
                    Console.WriteLine("  --no-mcsh                Do not copy MCSH shadow maps");
                    Console.WriteLine("  --no-holes               Do not copy MCNK holes");
                    return 0;
            }
        }

        var runOptions = new TerrainTextureTransferOptions(
            SourceDirectory: sourceDir,
            TargetDirectory: targetDir,
            OutputDirectory: outputDir,
            Mode: mode,
            Pairs: pairs,
            TileLimit: tileLimit,
            GlobalDeltaX: globalDeltaX,
            GlobalDeltaY: globalDeltaY,
            ChunkOffsetX: chunkOffsetX,
            ChunkOffsetY: chunkOffsetY,
            CopyMtex: copyMtex,
            CopyMcly: copyMcly,
            CopyMcal: copyMcal,
            CopyMcsh: copyMcsh,
            CopyHoles: copyHoles,
            ManifestPath: manifestPath);

        Console.WriteLine("Terrain Texture Transfer");
        Console.WriteLine("========================");
        Console.WriteLine($"Source: {Pm4CoordinateService.ResolveMapDirectory(runOptions.SourceDirectory)}");
        Console.WriteLine($"Target: {Pm4CoordinateService.ResolveMapDirectory(runOptions.TargetDirectory)}");
        Console.WriteLine($"Output: {Path.GetFullPath(runOptions.OutputDirectory)}");
        Console.WriteLine($"Mode:   {runOptions.Mode}");
        Console.WriteLine($"Pairs:  {(runOptions.Pairs.Count > 0 ? runOptions.Pairs.Count : 0)} explicit");
        if (runOptions.GlobalDeltaX.HasValue && runOptions.GlobalDeltaY.HasValue)
            Console.WriteLine($"Delta:  ({runOptions.GlobalDeltaX.Value}, {runOptions.GlobalDeltaY.Value})");
        Console.WriteLine($"Chunk offset: ({runOptions.ChunkOffsetX}, {runOptions.ChunkOffsetY})");
        Console.WriteLine($"Copy: MTEX={runOptions.CopyMtex}, MCLY={runOptions.CopyMcly}, MCAL={runOptions.CopyMcal}, MCSH={runOptions.CopyMcsh}, Holes={runOptions.CopyHoles}");

        try
        {
            TerrainTextureTransferExecutionReport report = TerrainTextureTransferService.Execute(runOptions);

            Console.WriteLine();
            Console.WriteLine($"Source map: {report.SourceMapName}");
            Console.WriteLine($"Target map: {report.TargetMapName}");
            Console.WriteLine($"Tiles planned:   {report.TilesPlanned}");
            Console.WriteLine($"Tiles processed: {report.TilesProcessed}");
            Console.WriteLine($"Tiles written:   {report.TilesWritten}");
            Console.WriteLine($"Manual review:   {report.TilesNeedingManualReview}");
            Console.WriteLine($"Chunk pairs:     {report.ChunkPairsApplied}");
            Console.WriteLine();
            Console.WriteLine($"Summary manifest: {report.SummaryManifestPath}");

            foreach (TerrainTextureTransferTileManifest tile in report.Tiles
                .Where(tile => tile.NeedsManualReview || tile.Warnings.Count > 0)
                .Take(15))
            {
                Console.WriteLine();
                Console.WriteLine($"Pair {tile.SourceTileName} -> {tile.TargetTileName}");
                Console.WriteLine($"  touched={tile.TargetChunksTouched}, sourceChunks={tile.SourceChunksUsed}, missingSource={tile.MissingSourceChunkCount}, outOfRange={tile.OutOfRangeChunkRemapCount}");
                Console.WriteLine($"  copied: mtex={tile.MtexCopied}, mcly={tile.MclyCopied}, mcal={tile.McalCopied}, mcsh={tile.McshCopied}, holes={tile.HolesCopied}");
                if (tile.Warnings.Count > 0)
                    Console.WriteLine($"  warning={tile.Warnings[0]}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static bool TryParseTilePair(string raw, out TerrainTilePair pair)
    {
        pair = new TerrainTilePair(0, 0, 0, 0);
        string[] split = raw.Split(':', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (split.Length != 2)
            return false;

        if (!TryParseIntPair(split[0], out int sx, out int sy)
            || !TryParseIntPair(split[1], out int tx, out int ty))
        {
            return false;
        }

        pair = new TerrainTilePair(sx, sy, tx, ty);
        return true;
    }

    private static bool TryParseIntPair(string raw, out int x, out int y)
    {
        x = 0;
        y = 0;
        string normalized = raw.Replace('_', ',').Replace('x', ',').Replace('X', ',');
        string[] parts = normalized.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
            return false;

        return int.TryParse(parts[0], out x) && int.TryParse(parts[1], out y);
    }

    private static async Task<int> RunAnalyzeAsync(string[] args)
    {
        Console.WriteLine("Analyze command - not yet implemented");
        await Task.CompletedTask;
        return 0;
    }

    private static async Task<int> RunMlListMapsAsync(string[] args)
    {
        string? clientPath = null;
        string? listfilePath = null;
        string? outputJsonPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--client":
                case "-c":
                    if (i + 1 < args.Length) clientPath = args[++i];
                    break;
                case "--listfile":
                case "-l":
                    if (i + 1 < args.Length) listfilePath = args[++i];
                    break;
                case "--output-json":
                case "-o":
                    if (i + 1 < args.Length) outputJsonPath = args[++i];
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(clientPath))
        {
            Console.WriteLine("ML Map Discovery - list all map directories visible for a client root");
            Console.WriteLine();
            Console.WriteLine("Usage: ml-list-maps --client <path> [--listfile <csv>] [--output-json <file>]  (legacy alias: vlm-list-maps)");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --client, -c <path>      Client root (or root containing Data)");
            Console.WriteLine("  --listfile, -l <csv>     Optional explicit listfile path");
            Console.WriteLine("  --output-json, -o <file> Optional JSON output path for scripting");
            return 1;
        }

        try
        {
            var maps = DiscoverClientMapDirectories(clientPath!, listfilePath);

            if (!string.IsNullOrWhiteSpace(outputJsonPath))
            {
                string fullOutput = Path.GetFullPath(outputJsonPath);
                Directory.CreateDirectory(Path.GetDirectoryName(fullOutput) ?? Directory.GetCurrentDirectory());
                await File.WriteAllTextAsync(fullOutput, JsonSerializer.Serialize(maps, new JsonSerializerOptions { WriteIndented = true }));
            }

            foreach (string map in maps)
                Console.WriteLine(map);

            return maps.Count > 0 ? 0 : 1;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static List<string> DiscoverClientMapDirectories(string clientPath, string? listfilePath)
    {
        string dataPath = clientPath;
        if (!Directory.Exists(Path.Combine(clientPath, "World")) &&
            Directory.Exists(Path.Combine(clientPath, "Data", "World")))
        {
            dataPath = Path.Combine(clientPath, "Data");
        }

        var searchPaths = new List<string> { dataPath };
        if (!string.Equals(clientPath, dataPath, StringComparison.OrdinalIgnoreCase))
            searchPaths.Add(clientPath);

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();

        string[] listfileSearchPaths =
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "community-listfile-withcapitals.csv"),
            "community-listfile-withcapitals.csv",
            "listfile.csv",
        };

        string? resolvedListfile = !string.IsNullOrWhiteSpace(listfilePath) && File.Exists(listfilePath)
            ? listfilePath
            : listfileSearchPaths.FirstOrDefault(File.Exists);

        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, searchPaths, resolvedListfile);

        var maps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Disk-backed discovery for extracted clients.
        foreach (string basePath in searchPaths)
        {
            string mapsDir = Path.Combine(basePath, "World", "Maps");
            if (!Directory.Exists(mapsDir))
                continue;

            IEnumerable<string> diskWdtCandidates = Directory
                .EnumerateFiles(mapsDir, "*.wdt", SearchOption.AllDirectories)
                .Concat(Directory.EnumerateFiles(mapsDir, "*.wdt.mpq", SearchOption.AllDirectories));

            foreach (string wdtPath in diskWdtCandidates)
            {
                if (TryExtractDiskMapDirectory(wdtPath, out string? mapDirectory))
                    maps.Add(mapDirectory!);
            }
        }

        // Archive-backed discovery for MPQ clients.
        var archiveWdtCandidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string knownFile in archiveCatalog.GetAllKnownFiles())
        {
            if (!TryExtractMapDirectoryFromWdtPath(knownFile, out _, out string? normalizedPath))
                continue;

            archiveWdtCandidates.Add(normalizedPath!);
        }

        foreach (string candidatePath in archiveWdtCandidates)
        {
            if (!archiveCatalog.FileExists(candidatePath))
                continue;

            if (TryExtractMapDirectoryFromWdtPath(candidatePath, out string? mapDirectory, out _))
                maps.Add(mapDirectory!);
        }

        return maps.OrderBy(static name => name, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static bool TryExtractDiskMapDirectory(string wdtPath, out string? mapDirectory)
    {
        mapDirectory = null;
        if (string.IsNullOrWhiteSpace(wdtPath))
            return false;

        string fileName = Path.GetFileName(wdtPath);
        if (!TryGetWdtBaseName(fileName, out string fileStem))
            return false;

        string directoryName = new DirectoryInfo(Path.GetDirectoryName(wdtPath) ?? string.Empty).Name;
        if (string.IsNullOrWhiteSpace(directoryName))
            return false;

        if (!fileStem.Equals(directoryName, StringComparison.OrdinalIgnoreCase))
            return false;

        mapDirectory = directoryName;
        return true;
    }

    private static bool TryExtractMapDirectoryFromWdtPath(string virtualPath, out string? mapDirectory, out string? normalizedPath)
    {
        mapDirectory = null;
        normalizedPath = null;

        if (string.IsNullOrWhiteSpace(virtualPath))
            return false;

        string normalized = virtualPath.Replace('\\', '/').TrimStart('/');

        string[] segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length < 4)
            return false;

        int worldIndex = Array.FindIndex(segments, static segment => segment.Equals("World", StringComparison.OrdinalIgnoreCase));
        if (worldIndex < 0 || worldIndex + 3 >= segments.Length)
            return false;

        if (!segments[worldIndex + 1].Equals("Maps", StringComparison.OrdinalIgnoreCase))
            return false;

        string candidateMapDirectory = segments[worldIndex + 2];
        string fileName = segments[worldIndex + 3];
        if (!TryGetWdtBaseName(fileName, out string fileStem))
            return false;

        if (!fileStem.Equals(candidateMapDirectory, StringComparison.OrdinalIgnoreCase))
            return false;

        mapDirectory = candidateMapDirectory;
        normalizedPath = normalized;
        return true;
    }

    private static bool TryGetWdtBaseName(string fileName, out string baseName)
    {
        baseName = string.Empty;
        if (string.IsNullOrWhiteSpace(fileName))
            return false;

        if (fileName.EndsWith(".wdt.mpq", StringComparison.OrdinalIgnoreCase))
        {
            baseName = fileName[..^8];
        }
        else if (fileName.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
        {
            baseName = fileName[..^4];
        }

        return !string.IsNullOrWhiteSpace(baseName);
    }


    private static async Task<int> RunDefaultConvertAsync(string[] args)
    {
        // Treat first arg as input path for backward compatibility
        return await RunConvertAsync(args);
    }

    private static async Task<int> RunVlmExportAsync(string[] args)
    {
        string? clientPath = null;
        string? mapName = null;
        string? outputDir = null;
        string? listfilePath = null;
        string? minimapRoot = null;
        string? tileFilter = null;
        int limit = int.MaxValue;
        bool generateDepth = false;
        bool batchAll = false;
        bool skipDerivedAssets = false;
        bool interestingOnly = false;
        int interestingMinScore = 1;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--client":
                case "-c":
                    if (i + 1 < args.Length) clientPath = args[++i];
                    break;
                case "--map":
                case "-m":
                    if (i + 1 < args.Length) mapName = args[++i];
                    break;
                case "--out":
                case "-o":
                    if (i + 1 < args.Length) outputDir = args[++i];
                    break;
                case "--listfile":
                case "-l":
                    if (i + 1 < args.Length) listfilePath = args[++i];
                    break;
                case "--minimap-root":
                    if (i + 1 < args.Length) minimapRoot = args[++i];
                    break;
                case "--tile":
                    if (i + 1 < args.Length) tileFilter = args[++i];
                    break;
                case "--limit":
                case "-n":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int n))
                        limit = n;
                    break;
                case "--depth":
                case "-d":
                    generateDepth = true;
                    break;
                case "--batch-all":
                    batchAll = true;
                    break;
                case "--skip-derived-assets":
                    skipDerivedAssets = true;
                    break;
                case "--interesting-only":
                    interestingOnly = true;
                    break;
                case "--interesting-min-score":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int parsedInterestingMinScore))
                        interestingMinScore = parsedInterestingMinScore;
                    break;
            }
        }

        if (string.IsNullOrEmpty(clientPath) || string.IsNullOrEmpty(outputDir) || (!batchAll && string.IsNullOrEmpty(mapName)))
        {
            Console.WriteLine("ML Dataset Export - Generate terrain supervision dataset from WoW client files");
            Console.WriteLine();
            Console.WriteLine("Usage: ml-export --client <path> --map <name> --out <dir> [options]  (legacy aliases: mk-export, vlm-export)");
            Console.WriteLine("Batch: ml-export --client <path> --batch-all --out <root_dir>");
            Console.WriteLine();
            Console.WriteLine("Required:");
            Console.WriteLine("  --client, -c <path>   Path to Alpha 0.5.3 client Data folder");
            Console.WriteLine("  --map, -m <name>      Map name (e.g., 'development') OR use --batch-all");
            Console.WriteLine("  --out, -o <dir>       Output directory for dataset (Root dir for batch)");
            Console.WriteLine();
            Console.WriteLine("Optional:");
            Console.WriteLine("  --batch-all           Automatically export 8 standard maps (Azeroth, Kalimdor, etc)");
            Console.WriteLine("  --listfile, -l <csv>  Path to listfile for name resolution");
            Console.WriteLine("  --minimap-root <dir>  Optional explicit root for minimap lookup; keeps terrain input and minimap source separate");
            Console.WriteLine("  --tile <x_y>          Export only one specific tile coordinate");
            Console.WriteLine("  --limit, -n <N>       Export only first N tiles");
            Console.WriteLine("  --depth, -d           Generate depth maps (requires DepthAnything3)");
            Console.WriteLine("  --skip-derived-assets Skip tilesets, stitched outputs, and semantic postprocess assets for faster core export coverage");
            Console.WriteLine("  --interesting-only    Only export scored interesting tiles, with a one-tile fallback for otherwise empty maps");
            Console.WriteLine("  --interesting-min-score <N> Minimum tile-interest score when --interesting-only is enabled");
            return 1;
        }

        if (!string.IsNullOrWhiteSpace(minimapRoot) && !Directory.Exists(minimapRoot))
        {
            Console.WriteLine($"Error: minimap root not found: {minimapRoot}");
            return 1;
        }

        var exporter = new VlmDatasetExporter();
        var progress = new Progress<string>(msg => Console.WriteLine(msg));

        try
        {
            if (batchAll)
            {
                var maps = new[] 
                { 
                    "Azeroth", "Kalimdor", "Kalidar", 
                    "DeadminesInstance", "RazorfenKraulInstance", "Shadowfang",
                    "PVPZone01", "PVPZone02" 
                };

                Console.WriteLine("ML DATASET BATCH EXPORT");
                Console.WriteLine($"Exporting {maps.Length} maps to {outputDir}...");
                Console.WriteLine(new string('=', 60));

                foreach (var map in maps)
                {
                    var mapOutputDir = Path.Combine(outputDir, $"053_{map}_v30");
                    Console.WriteLine($"\n[BATCH] Processing {map} -> {mapOutputDir}");
                    
                    try 
                    {
                        var res = await exporter.ExportMapAsync(clientPath, map, mapOutputDir, progress, limit, listfilePath, generateDepth, minimapRoot, tileFilter, skipDerivedAssets, interestingOnly, interestingMinScore);
                        Console.WriteLine($"[BATCH] {map} Complete: {res.TilesExported} tiles.");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[BATCH] {map} FAILED: {ex.Message}");
                    }
                }
                
                Console.WriteLine("\nBatch Export Complete.");
                return 0;
            }
            else
            {
                // Single Map Mode
                Console.WriteLine($"ML Dataset Export: {mapName}");
                Console.WriteLine($"  Client: {clientPath}");
                Console.WriteLine($"  Output: {outputDir}");
                if (!string.IsNullOrWhiteSpace(minimapRoot))
                    Console.WriteLine($"  Minimap root: {minimapRoot}");
                if (skipDerivedAssets)
                    Console.WriteLine("  Derived assets: skipped");
                if (interestingOnly)
                    Console.WriteLine($"  Interesting tile curation: enabled (min score {interestingMinScore})");
                
                var result = await exporter.ExportMapAsync(clientPath, mapName!, outputDir, progress, limit, listfilePath, generateDepth, minimapRoot, tileFilter, skipDerivedAssets, interestingOnly, interestingMinScore);
                
                Console.WriteLine();
                Console.WriteLine($"Export complete:");
                Console.WriteLine($"  Tiles exported: {result.TilesExported}");
                Console.WriteLine($"  Tiles skipped: {result.TilesSkipped}");
                Console.WriteLine($"  Unique textures: {result.UniqueTextures}");
                Console.WriteLine($"  Output: {result.OutputDirectory}");
                
                return 0;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> RunVlmDecodeAsync(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("ML Dataset Decode - Reconstruct ADT from ML dataset JSON output");
            Console.WriteLine();
            Console.WriteLine("Usage: ml-decode --input <json> --output <adt>  (legacy aliases: mk-decode, vlm-decode)");
            Console.WriteLine();
            Console.WriteLine("Required:");
            Console.WriteLine("  --input, -i <json>    ML dataset JSON file");
            Console.WriteLine("  --output, -o <adt>    Output ADT file path");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".adt");

        Console.WriteLine($"ML Dataset Decode: {Path.GetFileName(inputPath)} → {Path.GetFileName(outputPath)}");

        var decoder = new VlmAdtDecoder();

        try
        {
            var success = await decoder.DecodeAsync(inputPath, outputPath);
            
            if (success)
            {
                Console.WriteLine($"Decoded successfully: {outputPath}");
                return 0;
            }
            else
            {
                Console.WriteLine("Decode failed - invalid input data");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }


    private static async Task<int> RunVlmBakeAsync(string[] args)
    {
        string? datasetDir = null;
        string? inputPath = null;
        string? outputPath = null;
        string? minimapPath = null;
        bool withShadows = true;
        bool debakeShadows = false;
        bool exportLayers = false;
        bool invertAlpha = true;  // Default true for correct layer blending
        float shadowIntensity = 0.5f;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--dataset":
                case "-d":
                    if (i + 1 < args.Length) datasetDir = args[++i];
                    break;
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--minimap":
                case "-m":
                    if (i + 1 < args.Length) minimapPath = args[++i];
                    break;
                case "--shadows":
                    withShadows = true;
                    break;
                case "--no-shadows":
                    withShadows = false;
                    break;
                case "--debake":
                    debakeShadows = true;
                    break;
                case "--export-layers":
                case "-l":
                    exportLayers = true;
                    break;
                case "--invert-alpha":
                    invertAlpha = true;
                    break;
                case "--no-invert-alpha":
                    invertAlpha = false;
                    break;
                case "--shadow-intensity":
                    if (i + 1 < args.Length && float.TryParse(args[++i], out var intensity))
                        shadowIntensity = Math.Clamp(intensity, 0f, 1f);
                    break;
            }
        }

        if (string.IsNullOrEmpty(datasetDir) && string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("ML Dataset Bake - Reconstruct high-resolution reference minimaps");
            Console.WriteLine();
            Console.WriteLine("Usage: ml-bake --dataset <dir> [--input <json>] [--output <png>]  (legacy aliases: mk-bake, vlm-bake)");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --dataset, -d <dir>       Path to the ML dataset root (containing tilesets and masks)");
            Console.WriteLine("  --input, -i <json>        Specific ML dataset JSON file (default: all in dataset/*.json)");
            Console.WriteLine("  --output, -o <png>        Output PNG path (for single input) or output directory");
            Console.WriteLine();
            Console.WriteLine("Shadow Options:");
            Console.WriteLine("  --shadows                 Apply shadow maps to output (default)");
            Console.WriteLine("  --no-shadows              Disable shadow map application");
            Console.WriteLine("  --shadow-intensity <0-1>  Shadow darkness (default: 0.5)");
            Console.WriteLine("  --debake                  Remove shadows from existing minimap (requires --minimap)");
            Console.WriteLine("  --minimap, -m <png>       Source minimap for debaking shadows");
            Console.WriteLine();
            Console.WriteLine("Layer Options:");
            Console.WriteLine("  --export-layers, -l       Export individual texture layers as separate PNGs");
            Console.WriteLine("  --no-invert-alpha         Disable alpha inversion (default: inverted for correct blending)");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  ml-bake -d ./ml_output -i dataset/Azeroth_0_0.json --shadows");
            Console.WriteLine("  ml-bake -d ./ml_output -i dataset/Azeroth_0_0.json --export-layers");
            Console.WriteLine("  ml-bake -d ./ml_output --debake -m minimap.png -i tile.json -o clean.png");
            return 1;
        }

        // If datasetDir is null, infer it from inputPath
        if (string.IsNullOrEmpty(datasetDir) && !string.IsNullOrEmpty(inputPath))
        {
            datasetDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(inputPath))) ?? ".";
        }

        Console.WriteLine($"ML Dataset Bake: High-Resolution Reconstruction");
        Console.WriteLine($"  Dataset: {datasetDir}");
        Console.WriteLine($"  Shadows: {(withShadows ? "enabled" : "disabled")} (intensity: {shadowIntensity:F2})");
        if (invertAlpha) Console.WriteLine($"  Alpha: INVERTED");
        if (debakeShadows) Console.WriteLine($"  Mode: De-bake (remove shadows)");
        if (exportLayers) Console.WriteLine($"  Mode: Export individual layers");

        var baker = new MinimapBakeService(datasetDir!) { ShadowIntensity = shadowIntensity, InvertAlpha = invertAlpha };
        var filesToProcess = new List<string>();

        if (!string.IsNullOrEmpty(inputPath))
        {
            filesToProcess.Add(inputPath);
        }
        else
        {
            var datasetFolder = Path.Combine(datasetDir!, "dataset");
            if (Directory.Exists(datasetFolder))
            {
                filesToProcess.AddRange(Directory.EnumerateFiles(datasetFolder, "*.json"));
            }
        }

        if (filesToProcess.Count == 0)
        {
            Console.WriteLine("Error: No JSON files found to process.");
            return 1;
        }

        var outputBase = outputPath ?? Path.Combine(datasetDir!, "reconstructed_minimaps");
        Directory.CreateDirectory(outputBase);

        foreach (var file in filesToProcess)
        {
            try
            {
                Console.Write($"  Processing {Path.GetFileName(file)}... ");
                var timer = System.Diagnostics.Stopwatch.StartNew();
                
                Image<Rgba32> image;
                string extraInfo = "";
                
                if (exportLayers)
                {
                    // Export individual layers + composite
                    var (composite, layerCount, stats) = await baker.BakeTileWithLayersAsync(file, outputBase);
                    image = composite;
                    extraInfo = $" [{stats}]";
                }
                else if (debakeShadows && !string.IsNullOrEmpty(minimapPath))
                {
                    // De-bake: remove shadows from existing minimap
                    image = await baker.DebakeShadowsFromMinimapAsync(minimapPath, file);
                }
                else if (withShadows)
                {
                    // Bake with shadows
                    image = await baker.BakeTileWithShadowsAsync(file, applyShadows: true);
                }
                else
                {
                    // Bake without shadows
                    image = await baker.BakeTileAsync(file);
                }
                
                var outName = Path.GetFileNameWithoutExtension(file) + "_highres.png";
                var outPath = Path.IsPathRooted(outputPath) && filesToProcess.Count == 1 
                    ? outputPath 
                    : Path.Combine(outputBase, outName);

                await image.SaveAsPngAsync(outPath);
                image.Dispose();
                
                timer.Stop();
                Console.WriteLine($"done ({timer.ElapsedMilliseconds}ms){extraInfo}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"failed: {ex.Message}");
            }
        }

        return 0;
    }

    private static async Task<int> RunVlmBakeHeightmapAsync(string[] args)
    {
        string? datasetDir = null;
        string? inputPath = null;
        string? outputPath = null;
        bool fullRes = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--dataset":
                case "-d":
                    datasetDir = args[++i];
                    break;
                case "--input":
                case "-i":
                    inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    outputPath = args[++i];
                    break;
                case "--full-res":
                    fullRes = true;
                    break;
                case "--help":
                case "-h":
                    Console.WriteLine("ML Dataset Bake Heightmap - Generate heightmaps from ML dataset JSON data");
                    Console.WriteLine();
                    Console.WriteLine("Usage: ml-bake-heightmap --dataset <dir> [--input <json>] [--output <dir>]  (legacy aliases: mk-bake-heightmap, vlm-bake-heightmap)");
                    Console.WriteLine();
                    Console.WriteLine("Options:");
                    Console.WriteLine("  --dataset, -d <dir>   ML dataset root directory");
                    Console.WriteLine("  --input, -i <json>    Specific JSON file (or process all if omitted)");
                    Console.WriteLine("  --output, -o <dir>    Output directory (default: dataset/heightmaps)");
                    Console.WriteLine("  --full-res            Generate 4096x4096 instead of 256x256");
                    return 0;
            }
        }

        if (string.IsNullOrEmpty(datasetDir) && string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("Error: Specify --dataset or --input");
            return 1;
        }

        if (string.IsNullOrEmpty(datasetDir) && !string.IsNullOrEmpty(inputPath))
        {
            datasetDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(inputPath))) ?? ".";
        }

        Console.WriteLine($"ML Dataset Bake Heightmap");
        Console.WriteLine($"  Dataset: {datasetDir}");
        Console.WriteLine($"  Resolution: {(fullRes ? "4096x4096" : "256x256")}");

        var baker = new HeightmapBakeService(datasetDir!);
        var outputBase = outputPath ?? Path.Combine(datasetDir!, "heightmaps");

        // If no specific input, use map-wide export (scans for global bounds first)
        if (string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("  Mode: MAP-WIDE (global height bounds)");
            var progress = new Progress<string>(msg => Console.WriteLine($"  {msg}"));
            await baker.ExportMapHeightmapsAsync(datasetDir!, outputBase, progress);
            return 0;
        }

        // Single tile mode (per-tile bounds - not recommended)
        Console.WriteLine("  Mode: SINGLE TILE (per-tile bounds)");
        Directory.CreateDirectory(outputBase);

        try
        {
            var tileName = Path.GetFileNameWithoutExtension(inputPath);
            Console.Write($"  {tileName}... ");

            if (fullRes)
            {
                var (heightmap, min, max) = await baker.BakeHeightmap4096Async(inputPath);
                var outPath = Path.Combine(outputBase, $"{tileName}_heightmap_4096.png");
                await heightmap.SaveAsPngAsync(outPath);
                heightmap.Dispose();
                Console.WriteLine($"OK [{min:F1} to {max:F1}]");
            }
            else
            {
                await baker.ExportWithMetadataAsync(inputPath, outputBase);
                Console.WriteLine("OK");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAILED: {ex.Message}");
            return 1;
        }

        return 0;
    }

    private static async Task<int> RunVlmSynthAsync(string[] args)
    {
        string? datasetDir = null;
        string? inputPath = null;
        string? outputPath = null;
        int resolution = 256;
        bool withVariations = false;
        float hillshade = 0.4f;
        float ao = 0.2f;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--dataset":
                case "-d":
                    datasetDir = args[++i];
                    break;
                case "--input":
                case "-i":
                    inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    outputPath = args[++i];
                    break;
                case "--resolution":
                case "-r":
                    resolution = int.Parse(args[++i]);
                    break;
                case "--variations":
                    withVariations = true;
                    break;
                case "--hillshade":
                    hillshade = float.Parse(args[++i]);
                    break;
                case "--ao":
                    ao = float.Parse(args[++i]);
                    break;
                case "--help":
                case "-h":
                    Console.WriteLine("ML Dataset Synth - Generate synthesized training pairs");
                    Console.WriteLine();
                    Console.WriteLine("Creates perfectly matched minimap/heightmap pairs where the minimap");
                    Console.WriteLine("is deformed based on the heightmap (hillshading, ambient occlusion).");
                    Console.WriteLine();
                    Console.WriteLine("Usage: ml-synth --dataset <dir> [--input <json>] [--output <dir>]  (legacy aliases: mk-synth, vlm-synth)");
                    Console.WriteLine();
                    Console.WriteLine("Options:");
                    Console.WriteLine("  --dataset, -d <dir>     ML dataset root directory");
                    Console.WriteLine("  --input, -i <json>      Specific JSON file (or process all)");
                    Console.WriteLine("  --output, -o <dir>      Output directory (default: synthesized/)");
                    Console.WriteLine("  --resolution, -r <n>    Output resolution (default: 256)");
                    Console.WriteLine("  --variations            Generate 4 lighting variations per tile");
                    Console.WriteLine("  --hillshade <0-1>       Hillshade strength (default: 0.4)");
                    Console.WriteLine("  --ao <0-1>              Ambient occlusion strength (default: 0.2)");
                    return 0;
            }
        }

        if (string.IsNullOrEmpty(datasetDir) && string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("Error: Specify --dataset or --input");
            return 1;
        }

        if (string.IsNullOrEmpty(datasetDir) && !string.IsNullOrEmpty(inputPath))
        {
            datasetDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(inputPath))) ?? ".";
        }

        Console.WriteLine($"ML Dataset Synthesized Training Pair Generator");
        Console.WriteLine($"  Dataset: {datasetDir}");
        Console.WriteLine($"  Resolution: {resolution}x{resolution}");
        Console.WriteLine($"  Hillshade: {hillshade:F2}, AO: {ao:F2}");
        if (withVariations) Console.WriteLine($"  Mode: 4 lighting variations per tile");

        var synth = new SynthesizedTrainingService(datasetDir!)
        {
            HillshadeStrength = hillshade,
            AmbientOcclusion = ao
        };

        var filesToProcess = new List<string>();

        if (!string.IsNullOrEmpty(inputPath))
        {
            filesToProcess.Add(inputPath);
        }
        else
        {
            var datasetFolder = Path.Combine(datasetDir!, "dataset");
            if (Directory.Exists(datasetFolder))
            {
                filesToProcess.AddRange(Directory.EnumerateFiles(datasetFolder, "*.json"));
            }
        }

        if (filesToProcess.Count == 0)
        {
            Console.WriteLine("Error: No JSON files found.");
            return 1;
        }

        var outputBase = outputPath ?? Path.Combine(datasetDir!, "synthesized");
        Directory.CreateDirectory(outputBase);

        int processed = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        foreach (var file in filesToProcess)
        {
            try
            {
                var tileName = Path.GetFileNameWithoutExtension(file);
                Console.Write($"  {tileName}... ");

                if (withVariations)
                {
                    await synth.ExportWithVariationsAsync(file, outputBase, resolution);
                    Console.WriteLine("OK (4 variations)");
                }
                else
                {
                    await synth.ExportPairAsync(file, outputBase, resolution);
                    Console.WriteLine("OK");
                }
                processed++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FAILED: {ex.Message}");
            }
        }

        sw.Stop();
        Console.WriteLine($"Processed {processed}/{filesToProcess.Count} tiles in {sw.Elapsed.TotalSeconds:F1}s");
        return 0;
    }

    private static void ShowUsage()
    {
        Console.WriteLine("WoW Map Converter v3 - Bidirectional Alpha ↔ LK 3.3.5 Conversion");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  wowmapconverter convert <input.wdt> [options]           Convert Alpha WDT → LK ADT");
        Console.WriteLine("  wowmapconverter convert-lk-to-alpha <wdt> [options]     Convert LK ADT → Alpha WDT");
        Console.WriteLine("  wowmapconverter convert-wmo <input.wmo> [options]       Convert WMO v14 → v17");
        Console.WriteLine("  wowmapconverter convert-wmo-to-alpha <wmo> [options]    Convert WMO v17 → v14");
        Console.WriteLine("  wowmapconverter convert-mdx <input.mdx> [options]       Convert MDX → M2");
        Console.WriteLine("  wowmapconverter convert-m2-to-mdx <m2> [options]        Convert M2 → MDX");
        Console.WriteLine("  wowmapconverter pm4-export <pm4> [options]              Export PM4 to OBJ");
        Console.WriteLine("  wowmapconverter pm4-validate-coords [options]           Validate PM4 MPRL refs against _obj0 placements");
        Console.WriteLine("  wowmapconverter development-analyze [options]           Audit the original development ADT/PM4/WL dataset");
        Console.WriteLine("  wowmapconverter development-repair [options]            Run the active development repair slice + manifests");
        Console.WriteLine("  wowmapconverter terrain-texture-transfer [options]      Transfer MCAL/MCLY/MCSH/holes across mapped tiles");
        Console.WriteLine("  wowmapconverter wmo-info <wmo> [options]                List WMO groups and structure info");
        Console.WriteLine("  wowmapconverter ml-export [options]                     Export ML dataset (legacy aliases: mk-export, vlm-export)");
        Console.WriteLine("  wowmapconverter ml-list-maps [options]                  Discover all map directories for a client root");
        Console.WriteLine("  wowmapconverter ml-harvest [options]                    Harvest ML dataset coverage and references");
            Console.WriteLine("  wowmapconverter ml-corpus --config <file> [options]     Run full export+harvest pipeline from a config file");
        Console.WriteLine("  wowmapconverter ml-decode [options]                     Decode ML dataset JSON to ADT");
        Console.WriteLine("  wowmapconverter ml-bake [options]                       Bake high-resolution reference minimaps");
        Console.WriteLine("  wowmapconverter ml-bake-heightmap [options]             Bake ML dataset heightmaps");
        Console.WriteLine("  wowmapconverter ml-synth [options]                      Generate synthesized ML training pairs");
        Console.WriteLine("  wowmapconverter batch --input-dir <dir> [options]       Batch convert directory");
        Console.WriteLine();
        Console.WriteLine("Alpha → LK Conversion Options:");
        Console.WriteLine("  --input, -i <path>      Input Alpha WDT file path");
        Console.WriteLine("  --output, -o <dir>      Output directory (default: ./output)");
        Console.WriteLine("  --alpha-client <dir>    Alpha client/archive root for direct MPQ DBC reads");
        Console.WriteLine("  --lk-client <dir>       LK client/archive root for direct MPQ DBC reads");
        Console.WriteLine("  --crosswalk <dir>       AreaID crosswalk CSV directory");
        Console.WriteLine("  --listfile <csv>        Community listfile CSV for asset fixups");
        Console.WriteLine("  --lk-listfile <txt>     LK listfile for archive discovery");
        Console.WriteLine("  --convert-wmos          Convert WMO v14 files to v17 with _alpha suffix");
        Console.WriteLine("  --wmo-dir <dir>         Alpha WMO source directory (e.g., test_data/0.5.3/tree)");
        Console.WriteLine("  --verbose, -v           Verbose output");
        Console.WriteLine();
        Console.WriteLine("LK → Alpha Conversion Options:");
        Console.WriteLine("  --wdt <path>            Input LK WDT file path");
        Console.WriteLine("  --map-dir <dir>         Directory containing LK ADT files");
        Console.WriteLine("  --output, -o <path>     Output Alpha WDT path");
        Console.WriteLine("  --skip-m2               Skip M2 doodad placements");
        Console.WriteLine("  --skip-wmo              Skip WMO placements");
        Console.WriteLine("  --no-liquids            Disable MH2O → MCLQ conversion");
        Console.WriteLine("  --verbose, -v           Verbose output");
        Console.WriteLine();
        Console.WriteLine("WMO Conversion Options:");
        Console.WriteLine("  --input, -i <path>      Input WMO v14 file");
        Console.WriteLine("  --output, -o <path>     Output WMO v17 path (creates root + _XXX.wmo groups)");
        Console.WriteLine("                          Uses the maintained converter path only");
        Console.WriteLine();
        Console.WriteLine("MDX Conversion Options:");
        Console.WriteLine("  --input, -i <path>      Input MDX file");
        Console.WriteLine("  --output, -o <path>     Output M2 path (also creates .skin file)");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  wowmapconverter convert World/Maps/Azeroth/Azeroth.wdt -o ./out");
        Console.WriteLine("  wowmapconverter convert-lk-to-alpha development.wdt --map-dir ./maps -o alpha.wdt");
        Console.WriteLine("  wowmapconverter convert-wmo castle01.wmo -o castle01_v17.wmo");
        Console.WriteLine("  wowmapconverter wmo-info ironforge.wmo -v");
        Console.WriteLine("  wowmapconverter convert-mdx Human/HumanMale.mdx -o HumanMale.m2");
    }
}
