using System.Diagnostics;
using System.Text.Json;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.Converter;

internal static class SplitAdtToLkCommand
{
    public static void Run(string[] args)
    {
        try
        {
            SplitAdtToLkOptions options = ParseOptions(args);
            if (string.IsNullOrWhiteSpace(options.ClientRoot) || string.IsNullOrWhiteSpace(options.MapName) || string.IsNullOrWhiteSpace(options.OutputDir))
            {
                Console.Error.WriteLine("Error: convert-split-adt-to-lk requires --client-root <dir>, --map <name>, and --output-dir <dir>.");
                Environment.ExitCode = 1;
                return;
            }

            string clientRoot = Path.GetFullPath(options.ClientRoot);
            if (!Directory.Exists(clientRoot))
            {
                Console.Error.WriteLine($"Error: Client root not found: {clientRoot}");
                Environment.ExitCode = 1;
                return;
            }

            string outputDir = Path.GetFullPath(options.OutputDir);
            string? overlayRoot = string.IsNullOrWhiteSpace(options.OverlayRoot)
                ? null
                : Path.GetFullPath(options.OverlayRoot);
            if (!string.IsNullOrWhiteSpace(overlayRoot) && !Directory.Exists(overlayRoot))
            {
                Console.Error.WriteLine($"Error: Overlay root not found: {overlayRoot}");
                Environment.ExitCode = 1;
                return;
            }

            LkDonorContext? lkDonor = TryCreateLkDonorContext(options, out string? lkDonorError);
            if (!string.IsNullOrWhiteSpace(lkDonorError))
            {
                Console.Error.WriteLine($"Error: {lkDonorError}");
                Environment.ExitCode = 1;
                return;
            }

            AlphaDonorContext? alphaDonor = TryCreateAlphaDonorContext(options, out string? donorError);
            if (!string.IsNullOrWhiteSpace(donorError))
            {
                Console.Error.WriteLine($"Error: {donorError}");
                Environment.ExitCode = 1;
                return;
            }

            string mapName = options.MapName.Trim();
            string wdtVirtualPath = $"World\\Maps\\{mapName}\\{mapName}.wdt";

            Console.WriteLine("WowViewer.Tool.Converter convert-split-adt-to-lk report");
            Console.WriteLine($"  Client:   {clientRoot}");
            Console.WriteLine($"  Map:      {mapName}");
            Console.WriteLine($"  Overlay:  {overlayRoot ?? "<none>"}");
            if (lkDonor is not null)
                Console.WriteLine($"  LK Donor: {lkDonor.Root}");
            if (alphaDonor is not null)
                Console.WriteLine($"  Donor:    {alphaDonor.ClientRoot} :: {alphaDonor.MapName} [{FormatTileSet(alphaDonor.AllowedTiles)}]");
            Console.WriteLine($"  Output:   {outputDir}");
            if (!string.IsNullOrWhiteSpace(options.ReportPath))
                Console.WriteLine($"  Report:   {Path.GetFullPath(options.ReportPath)}");
            Console.WriteLine($"  Verbose:  {options.Verbose}");

            Directory.CreateDirectory(outputDir);

            using var catalog = new NativeMpqService();
            catalog.LoadArchives([clientRoot]);

            if (!TryReadVirtualOrLooseFile(wdtVirtualPath, overlayRoot, catalog, out byte[]? wdtBytes, out string wdtSourcePath) || wdtBytes is null)
            {
                Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtualPath}' from overlay or client archives.");
                Environment.ExitCode = 1;
                return;
            }

            IReadOnlyList<WdtTileCoordinate> occupiedTiles;
            using (var wdtStream = new MemoryStream(wdtBytes, writable: false))
            {
                MapFileSummary wdtSummary = MapFileSummaryReader.Read(wdtStream, wdtSourcePath);
                occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(wdtStream, wdtSummary);
            }

            if (occupiedTiles.Count == 0)
            {
                Console.Error.WriteLine($"Error: No occupied tiles were found in '{wdtSourcePath}'.");
                Environment.ExitCode = 1;
                return;
            }

            List<WdtTileCoordinate> tilesToProcess = BuildTileQueue(occupiedTiles, alphaDonor);

            int? limit = GetIntOption(args, "--limit", "-n");
            var sw = Stopwatch.StartNew();
            var warnings = new List<string>();
            var reportEntries = new List<SplitAdtToLkReportEntry>(tilesToProcess.Count);
            var emittedTiles = new HashSet<(int tileX, int tileY)>();
            int converted = 0;
            int failed = 0;

            foreach (WdtTileCoordinate tile in tilesToProcess)
            {
                string adtVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tile.TileX}_{tile.TileY}.adt";
                string texVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tile.TileX}_{tile.TileY}_tex0.adt";
                string objVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tile.TileX}_{tile.TileY}_obj0.adt";

                if (!TryReadVirtualOrLooseFile(adtVirtualPath, overlayRoot, catalog, out byte[]? adtBytes, out string adtSourcePath) || adtBytes is null)
                {
                    if (TryBuildLkDonorTile(lkDonor, mapName, tile.TileX, tile.TileY, overlayRoot, out LkAdtData? lkDonorTile, out SplitAdtTileSourceDetails lkDonorDetails))
                    {
                        byte[] donorBytes = LkAdtWriter.Build(lkDonorTile);
                        string donorOutputPath = Path.Combine(outputDir, $"{mapName}_{tile.TileX}_{tile.TileY}.adt");
                        File.WriteAllBytes(donorOutputPath, donorBytes);
                        emittedTiles.Add((tile.TileX, tile.TileY));
                        converted++;
                        reportEntries.Add(CreateSuccessReportEntry(tile.TileX, tile.TileY, donorOutputPath, donorBytes.Length, lkDonorDetails));

                        if (options.Verbose)
                            Console.WriteLine($"  Borrowed:  {mapName}_{tile.TileX}_{tile.TileY}.adt ({donorBytes.Length:N0} bytes) <- {lkDonorDetails.DisplaySource}");

                        if (limit.HasValue && converted >= limit.Value)
                            break;

                        continue;
                    }

                    if (TryBuildAlphaDonorTile(alphaDonor, tile.TileX, tile.TileY, out LkAdtData? donorTile, out string donorSource))
                    {
                        byte[] donorBytes = LkAdtWriter.Build(donorTile);
                        string donorOutputPath = Path.Combine(outputDir, $"{mapName}_{tile.TileX}_{tile.TileY}.adt");
                        File.WriteAllBytes(donorOutputPath, donorBytes);
                        emittedTiles.Add((tile.TileX, tile.TileY));
                        converted++;
                        reportEntries.Add(CreateSuccessReportEntry(
                            tile.TileX,
                            tile.TileY,
                            donorOutputPath,
                            donorBytes.Length,
                            new SplitAdtTileSourceDetails(
                                Outcome: "borrowed-alpha-donor",
                                RootSourceKind: "alpha-donor",
                                RootSourcePath: donorSource,
                                TextureSourceKind: null,
                                TextureSourcePath: null,
                                ObjectSourceKind: null,
                                ObjectSourcePath: null,
                                OriginalTexturePreserved: false,
                                OriginalObjectPlacementsPreserved: false,
                                DisplaySource: donorSource)));

                        if (options.Verbose)
                            Console.WriteLine($"  Borrowed:  {mapName}_{tile.TileX}_{tile.TileY}.adt ({donorBytes.Length:N0} bytes) <- {donorSource}");

                        if (limit.HasValue && converted >= limit.Value)
                            break;

                        continue;
                    }

                    failed++;
                    string warning = BuildMissingTileWarning(lkDonor, alphaDonor, adtVirtualPath, tile.TileX, tile.TileY);
                    warnings.Add(warning);
                    reportEntries.Add(new SplitAdtToLkReportEntry(
                        TileX: tile.TileX,
                        TileY: tile.TileY,
                        Outcome: "missing-everything",
                        RootSourceKind: null,
                        RootSourcePath: null,
                        TextureSourceKind: null,
                        TextureSourcePath: null,
                        ObjectSourceKind: null,
                        ObjectSourcePath: null,
                        OriginalTexturePreserved: false,
                        OriginalObjectPlacementsPreserved: false,
                        OutputPath: null,
                        OutputByteLength: null,
                        Message: warning));
                    continue;
                }

                TryReadVirtualOrLooseFile(texVirtualPath, overlayRoot, catalog, out byte[]? tex0Bytes, out string texSourcePath);
                TryReadVirtualOrLooseFile(objVirtualPath, overlayRoot, catalog, out byte[]? obj0Bytes, out string objSourcePath);

                try
                {
                    LkAdtData adtData = LkAdtReader.Read(adtBytes, tex0Bytes, obj0Bytes, tile.TileX, tile.TileY);
                    byte[] monolithicBytes = LkAdtWriter.Build(adtData);
                    string outputPath = Path.Combine(outputDir, $"{mapName}_{tile.TileX}_{tile.TileY}.adt");
                    File.WriteAllBytes(outputPath, monolithicBytes);
                    emittedTiles.Add((tile.TileX, tile.TileY));
                    converted++;
                    reportEntries.Add(CreateSuccessReportEntry(
                        tile.TileX,
                        tile.TileY,
                        outputPath,
                        monolithicBytes.Length,
                        new SplitAdtTileSourceDetails(
                            Outcome: "converted-original",
                            RootSourceKind: ClassifySourceKind(adtSourcePath, overlayRoot, null, allowArchive: true),
                            RootSourcePath: adtSourcePath,
                            TextureSourceKind: ClassifyOptionalSidecarSource(texSourcePath, overlayRoot, null),
                            TextureSourcePath: string.IsNullOrWhiteSpace(texSourcePath) ? null : texSourcePath,
                            ObjectSourceKind: ClassifyOptionalSidecarSource(objSourcePath, overlayRoot, null),
                            ObjectSourcePath: string.IsNullOrWhiteSpace(objSourcePath) ? null : objSourcePath,
                            OriginalTexturePreserved: IsOriginalSidecarSource(texSourcePath, overlayRoot),
                            OriginalObjectPlacementsPreserved: IsOriginalSidecarSource(objSourcePath, overlayRoot),
                            DisplaySource: adtSourcePath)));

                    if (options.Verbose)
                        Console.WriteLine($"  Converted: {mapName}_{tile.TileX}_{tile.TileY}.adt ({monolithicBytes.Length:N0} bytes) <- {adtSourcePath}");

                    if (limit.HasValue && converted >= limit.Value)
                        break;
                }
                catch (Exception ex)
                {
                    failed++;
                    warnings.Add($"Tile ({tile.TileX},{tile.TileY}): {ex.Message}");
                    reportEntries.Add(new SplitAdtToLkReportEntry(
                        TileX: tile.TileX,
                        TileY: tile.TileY,
                        Outcome: "error",
                        RootSourceKind: ClassifySourceKind(adtSourcePath, overlayRoot, null, allowArchive: true),
                        RootSourcePath: adtSourcePath,
                        TextureSourceKind: ClassifyOptionalSidecarSource(texSourcePath, overlayRoot, null),
                        TextureSourcePath: string.IsNullOrWhiteSpace(texSourcePath) ? null : texSourcePath,
                        ObjectSourceKind: ClassifyOptionalSidecarSource(objSourcePath, overlayRoot, null),
                        ObjectSourcePath: string.IsNullOrWhiteSpace(objSourcePath) ? null : objSourcePath,
                        OriginalTexturePreserved: IsOriginalSidecarSource(texSourcePath, overlayRoot),
                        OriginalObjectPlacementsPreserved: IsOriginalSidecarSource(objSourcePath, overlayRoot),
                        OutputPath: null,
                        OutputByteLength: null,
                        Message: ex.Message));
                    if (options.Verbose)
                        Console.Error.WriteLine($"  Error converting ({tile.TileX},{tile.TileY}): {ex}");
                }
            }

            string wdtOutputPath = Path.Combine(outputDir, $"{mapName}.wdt");
            File.WriteAllBytes(wdtOutputPath, LkWdtWriter.Build(emittedTiles));

            sw.Stop();
            Console.WriteLine($"  WDT:      {wdtSourcePath}");
            Console.WriteLine($"  Tiles:    {tilesToProcess.Count}");
            Console.WriteLine($"  Converted:{converted}");
            Console.WriteLine($"  Failed:   {failed}");
            Console.WriteLine($"  Output:   {outputDir}");
            Console.WriteLine($"  Wrote:    {wdtOutputPath}");
            Console.WriteLine($"  Elapsed:  {sw.ElapsedMilliseconds}ms");

            if (!string.IsNullOrWhiteSpace(options.ReportPath))
            {
                string reportPath = Path.GetFullPath(options.ReportPath);
                string? reportDirectory = Path.GetDirectoryName(reportPath);
                if (!string.IsNullOrWhiteSpace(reportDirectory))
                    Directory.CreateDirectory(reportDirectory);

                SplitAdtToLkReport report = BuildReport(
                    mapName,
                    clientRoot,
                    overlayRoot,
                    lkDonor?.Root,
                    alphaDonor,
                    outputDir,
                    reportEntries,
                    tilesToProcess.Count,
                    converted,
                    failed,
                    sw.ElapsedMilliseconds,
                    wdtSourcePath,
                    wdtOutputPath,
                    warnings.Count);
                File.WriteAllText(reportPath, JsonSerializer.Serialize(report, CreateJsonOptions()));
                Console.WriteLine($"  Report:   {reportPath}");
            }

            if (warnings.Count > 0)
            {
                Console.WriteLine($"  Warnings: {warnings.Count}");
                foreach (string warning in warnings.Take(10))
                    Console.WriteLine($"    {warning}");
                if (warnings.Count > 10)
                    Console.WriteLine($"    ... and {warnings.Count - 10} more");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            if (args.Contains("--verbose") || args.Contains("-v"))
                Console.Error.WriteLine(ex.StackTrace);
            Environment.ExitCode = 1;
        }
    }

    private static SplitAdtToLkOptions ParseOptions(string[] args)
    {
        return new SplitAdtToLkOptions(
            ClientRoot: GetOption(args, "--client-root", "-c"),
            MapName: GetOption(args, "--map", "-m"),
            OverlayRoot: GetOption(args, "--overlay-root", "-or"),
            LkDonorRoot: GetOption(args, "--lk-donor-root", "-ldr"),
            AlphaDonorClientRoot: GetOption(args, "--alpha-donor-client-root", "-adcr"),
            AlphaDonorMap: GetOption(args, "--alpha-donor-map", "-adm"),
            AlphaDonorTiles: GetOption(args, "--alpha-donor-tiles", "-adt"),
                ReportPath: GetOption(args, "--report", "-r"),
            OutputDir: GetOption(args, "--output-dir", "-o"),
            Verbose: HasFlag(args, "--verbose") || HasFlag(args, "-v"));
    }

    private static LkDonorContext? TryCreateLkDonorContext(SplitAdtToLkOptions options, out string? error)
    {
        error = null;

        if (string.IsNullOrWhiteSpace(options.LkDonorRoot))
            return null;

        string root = Path.GetFullPath(options.LkDonorRoot);
        if (!Directory.Exists(root))
        {
            error = $"LK donor root not found: {root}";
            return null;
        }

        return new LkDonorContext(root);
    }

    private static AlphaDonorContext? TryCreateAlphaDonorContext(SplitAdtToLkOptions options, out string? error)
    {
        error = null;

        bool hasAnyDonorSetting = !string.IsNullOrWhiteSpace(options.AlphaDonorClientRoot)
            || !string.IsNullOrWhiteSpace(options.AlphaDonorMap)
            || !string.IsNullOrWhiteSpace(options.AlphaDonorTiles);
        if (!hasAnyDonorSetting)
            return null;

        if (string.IsNullOrWhiteSpace(options.AlphaDonorClientRoot) || string.IsNullOrWhiteSpace(options.AlphaDonorMap))
        {
            error = "Alpha donor fallback requires both --alpha-donor-client-root <dir> and --alpha-donor-map <name>.";
            return null;
        }

        string clientRoot = Path.GetFullPath(options.AlphaDonorClientRoot);
        if (!Directory.Exists(clientRoot))
        {
            error = $"Alpha donor client root not found: {clientRoot}";
            return null;
        }

        HashSet<(int tileX, int tileY)>? allowedTiles = null;
        if (!string.IsNullOrWhiteSpace(options.AlphaDonorTiles))
        {
            if (!TryParseTileSet(options.AlphaDonorTiles, out allowedTiles))
            {
                error = "Invalid --alpha-donor-tiles value. Use forms like '63,1-3' or '63,1;63,2;63,3'.";
                return null;
            }
        }

        string donorMapName = options.AlphaDonorMap.Trim();
        string donorWdtVirtualPath = $"World\\Maps\\{donorMapName}\\{donorMapName}.wdt";
        byte[] donorWdtBytes;
        try
        {
            donorWdtBytes = ArchiveVirtualFileReader.ReadVirtualFile(donorWdtVirtualPath, [clientRoot]);
        }
        catch (Exception ex)
        {
            error = $"Could not read donor Alpha WDT '{donorWdtVirtualPath}' from '{clientRoot}': {ex.Message}";
            return null;
        }

        if (!AlphaWdtReader.IsAlphaWdt(donorWdtBytes))
        {
            error = $"Donor WDT '{donorWdtVirtualPath}' is not recognized as an Alpha WDT.";
            return null;
        }

        return new AlphaDonorContext(clientRoot, donorMapName, donorWdtVirtualPath, donorWdtBytes, allowedTiles);
    }

    private static bool TryBuildAlphaDonorTile(AlphaDonorContext? donor, int tileX, int tileY, out LkAdtData adtData, out string donorSource)
    {
        adtData = null!;
        donorSource = string.Empty;

        if (donor is null)
            return false;

        if (!ShouldAttemptAlphaDonor(donor, tileX, tileY))
            return false;

        if (!AlphaWdtReader.TryReadTile(donor.WdtBytes, tileX, tileY, out AlphaTileData? tileData) || tileData is null)
            return false;

        adtData = AlphaToLkConverter.ConvertTile(tileData, tileX, tileY);
        donorSource = $"{donor.DonorWdtVirtualPath}#{tileX},{tileY}";
        return true;
    }

    private static bool TryBuildLkDonorTile(LkDonorContext? donor, string mapName, int tileX, int tileY, string? overlayRoot, out LkAdtData adtData, out SplitAdtTileSourceDetails sourceDetails)
    {
        adtData = null!;
        sourceDetails = default!;

        if (donor is null)
            return false;

        string adtVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
        string texVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_tex0.adt";
        string objVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_obj0.adt";

        if (!TryReadLooseVirtualFile(adtVirtualPath, donor.Root, out byte[]? adtBytes, out string adtSourcePath) || adtBytes is null)
            return false;

        byte[]? tex0Bytes = null;
        byte[]? obj0Bytes = null;
        string? texSourcePath = null;
        string? objSourcePath = null;

        if (TryReadLooseVirtualFile(texVirtualPath, overlayRoot, out tex0Bytes, out string originalTexSourcePath))
            texSourcePath = originalTexSourcePath;
        else if (TryReadLooseVirtualFile(texVirtualPath, donor.Root, out tex0Bytes, out string donorTexSourcePath))
            texSourcePath = donorTexSourcePath;

        if (TryReadLooseVirtualFile(objVirtualPath, overlayRoot, out obj0Bytes, out string originalObjSourcePath))
            objSourcePath = originalObjSourcePath;
        else if (TryReadLooseVirtualFile(objVirtualPath, donor.Root, out obj0Bytes, out string donorObjSourcePath))
            objSourcePath = donorObjSourcePath;

        adtData = LkAdtReader.Read(adtBytes, tex0Bytes, obj0Bytes, tileX, tileY);
        if (obj0Bytes is { Length: > 0 } && !string.IsNullOrWhiteSpace(objSourcePath) && !string.IsNullOrWhiteSpace(overlayRoot) && IsPathUnderRoot(objSourcePath, overlayRoot))
        {
            AdtPlacementCatalog originalPlacements = ReadPlacementCatalog(obj0Bytes, objSourcePath);
            adtData = ApplyOriginalPlacementPreference(adtData, originalPlacements);
        }

        sourceDetails = new SplitAdtTileSourceDetails(
            Outcome: "borrowed-lk-donor",
            RootSourceKind: "lk-donor",
            RootSourcePath: adtSourcePath,
            TextureSourceKind: ClassifyOptionalSidecarSource(texSourcePath, overlayRoot, donor.Root),
            TextureSourcePath: texSourcePath,
            ObjectSourceKind: ClassifyOptionalSidecarSource(objSourcePath, overlayRoot, donor.Root),
            ObjectSourcePath: objSourcePath,
            OriginalTexturePreserved: IsOriginalSidecarSource(texSourcePath, overlayRoot),
            OriginalObjectPlacementsPreserved: IsOriginalSidecarSource(objSourcePath, overlayRoot),
            DisplaySource: BuildLkDonorSource(adtSourcePath, texSourcePath, objSourcePath, overlayRoot, donor.Root));
        return true;
    }

    private static AdtPlacementCatalog ReadPlacementCatalog(byte[] bytes, string sourcePath)
    {
        using MemoryStream stream = new(bytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        return AdtPlacementReader.Read(stream, fileSummary);
    }

    private static SplitAdtToLkReportEntry CreateSuccessReportEntry(int tileX, int tileY, string outputPath, int outputByteLength, SplitAdtTileSourceDetails sourceDetails)
    {
        return new SplitAdtToLkReportEntry(
            TileX: tileX,
            TileY: tileY,
            Outcome: sourceDetails.Outcome,
            RootSourceKind: sourceDetails.RootSourceKind,
            RootSourcePath: sourceDetails.RootSourcePath,
            TextureSourceKind: sourceDetails.TextureSourceKind,
            TextureSourcePath: sourceDetails.TextureSourcePath,
            ObjectSourceKind: sourceDetails.ObjectSourceKind,
            ObjectSourcePath: sourceDetails.ObjectSourcePath,
            OriginalTexturePreserved: sourceDetails.OriginalTexturePreserved,
            OriginalObjectPlacementsPreserved: sourceDetails.OriginalObjectPlacementsPreserved,
            OutputPath: outputPath,
            OutputByteLength: outputByteLength,
            Message: null);
    }

    private static SplitAdtToLkReport BuildReport(
        string mapName,
        string clientRoot,
        string? overlayRoot,
        string? lkDonorRoot,
        AlphaDonorContext? alphaDonor,
        string outputDir,
        IReadOnlyList<SplitAdtToLkReportEntry> entries,
        int tileCount,
        int converted,
        int failed,
        long elapsedMilliseconds,
        string wdtSourcePath,
        string wdtOutputPath,
        int warningCount)
    {
        return new SplitAdtToLkReport(
            MapName: mapName,
            ClientRoot: clientRoot,
            OverlayRoot: overlayRoot,
            LkDonorRoot: lkDonorRoot,
            AlphaDonorClientRoot: alphaDonor?.ClientRoot,
            AlphaDonorMap: alphaDonor?.MapName,
            OutputDir: outputDir,
            WdtSourcePath: wdtSourcePath,
            WdtOutputPath: wdtOutputPath,
            TileCount: tileCount,
            Converted: converted,
            Failed: failed,
            WarningCount: warningCount,
            ElapsedMilliseconds: elapsedMilliseconds,
            Entries: entries,
            OutcomeCounts: entries
                .GroupBy(static entry => entry.Outcome)
                .ToDictionary(static group => group.Key, static group => group.Count()));
    }

    private static JsonSerializerOptions CreateJsonOptions()
    {
        return new JsonSerializerOptions
        {
            WriteIndented = true,
        };
    }

    private static LkAdtData ApplyOriginalPlacementPreference(LkAdtData adtData, AdtPlacementCatalog originalPlacements)
    {
        if (originalPlacements.ModelPlacements.Count == 0 && originalPlacements.WorldModelPlacements.Count == 0)
            return adtData;

        Dictionary<int, LkMddfEntry> preferredModelsByUniqueId = BuildPreferredModelPlacementsByUniqueId(adtData.ModelPlacements);
        Dictionary<int, LkModfEntry> preferredWorldModelsByUniqueId = BuildPreferredWorldModelPlacementsByUniqueId(adtData.WorldModelPlacements);

        List<LkMddfEntry> modelPlacements = [];
        Dictionary<int, int> modelIndexByUniqueId = [];
        foreach (AdtModelPlacement placement in originalPlacements.ModelPlacements)
        {
            if (modelIndexByUniqueId.ContainsKey(placement.UniqueId))
                continue;

            if (!preferredModelsByUniqueId.TryGetValue(placement.UniqueId, out LkMddfEntry entry))
                continue;

            modelIndexByUniqueId.Add(placement.UniqueId, modelPlacements.Count);
            modelPlacements.Add(entry);
        }

        List<LkModfEntry> worldModelPlacements = [];
        Dictionary<int, int> worldModelIndexByUniqueId = [];
        foreach (AdtWorldModelPlacement placement in originalPlacements.WorldModelPlacements)
        {
            if (worldModelIndexByUniqueId.ContainsKey(placement.UniqueId))
                continue;

            if (!preferredWorldModelsByUniqueId.TryGetValue(placement.UniqueId, out LkModfEntry entry))
                continue;

            worldModelIndexByUniqueId.Add(placement.UniqueId, worldModelPlacements.Count);
            worldModelPlacements.Add(entry);
        }

        List<LkMcnkData> remappedChunks = new(adtData.Chunks.Count);
        foreach (LkMcnkData chunk in adtData.Chunks)
            remappedChunks.Add(RemapChunkPlacementRefs(chunk, adtData.ModelPlacements, adtData.WorldModelPlacements, modelIndexByUniqueId, worldModelIndexByUniqueId));

        return new LkAdtData
        {
            MapName = adtData.MapName,
            TileX = adtData.TileX,
            TileY = adtData.TileY,
            TextureNames = adtData.TextureNames,
            ModelNames = adtData.ModelNames,
            WorldModelNames = adtData.WorldModelNames,
            ModelPlacements = modelPlacements,
            WorldModelPlacements = worldModelPlacements,
            Chunks = remappedChunks,
            MhdrFlags = adtData.MhdrFlags,
            MfboFlightBounds = adtData.MfboFlightBounds
        };
    }

    private static Dictionary<int, LkMddfEntry> BuildPreferredModelPlacementsByUniqueId(IReadOnlyList<LkMddfEntry> placements)
    {
        Dictionary<int, LkMddfEntry> byUniqueId = [];
        foreach (LkMddfEntry placement in placements)
            byUniqueId[placement.UniqueId] = placement;

        return byUniqueId;
    }

    private static Dictionary<int, LkModfEntry> BuildPreferredWorldModelPlacementsByUniqueId(IReadOnlyList<LkModfEntry> placements)
    {
        Dictionary<int, LkModfEntry> byUniqueId = [];
        foreach (LkModfEntry placement in placements)
            byUniqueId[placement.UniqueId] = placement;

        return byUniqueId;
    }

    private static LkMcnkData RemapChunkPlacementRefs(
        LkMcnkData chunk,
        IReadOnlyList<LkMddfEntry> originalModelPlacements,
        IReadOnlyList<LkModfEntry> originalWorldModelPlacements,
        IReadOnlyDictionary<int, int> modelIndexByUniqueId,
        IReadOnlyDictionary<int, int> worldModelIndexByUniqueId)
    {
        List<int> doodadRefs = RemapPlacementRefs(chunk.DoodadRefs, originalModelPlacements, modelIndexByUniqueId, static placement => placement.UniqueId);
        List<int> worldModelRefs = RemapPlacementRefs(chunk.WorldModelRefs, originalWorldModelPlacements, worldModelIndexByUniqueId, static placement => placement.UniqueId);

        return new LkMcnkData
        {
            IndexX = chunk.IndexX,
            IndexY = chunk.IndexY,
            Flags = chunk.Flags,
            AreaId = chunk.AreaId,
            NLayers = chunk.NLayers,
            HoleMask = chunk.HoleMask,
            BaseHeight = chunk.BaseHeight,
            Heights = chunk.Heights,
            Normals = chunk.Normals,
            ShadowMap = chunk.ShadowMap,
            AlphaMapData = chunk.AlphaMapData,
            AlphaMapSize = chunk.AlphaMapSize,
            Layers = chunk.Layers,
            DoodadRefs = doodadRefs,
            WorldModelRefs = worldModelRefs,
            LiquidData = chunk.LiquidData,
            MccvColors = chunk.MccvColors,
            MclvLighting = chunk.MclvLighting,
            PosX = chunk.PosX,
            PosY = chunk.PosY,
            PosZ = chunk.PosZ
        };
    }

    private static List<int> RemapPlacementRefs<TPlacement>(
        IReadOnlyList<int> refs,
        IReadOnlyList<TPlacement> placements,
        IReadOnlyDictionary<int, int> newIndexByUniqueId,
        Func<TPlacement, int> getUniqueId)
    {
        List<int> remapped = [];
        HashSet<int> seen = [];

        foreach (int refIndex in refs)
        {
            if ((uint)refIndex >= placements.Count)
                continue;

            int uniqueId = getUniqueId(placements[refIndex]);
            if (!newIndexByUniqueId.TryGetValue(uniqueId, out int newIndex) || !seen.Add(newIndex))
                continue;

            remapped.Add(newIndex);
        }

        return remapped;
    }

    private static string BuildLkDonorSource(string adtSourcePath, string? texSourcePath, string? objSourcePath, string? overlayRoot, string donorRoot)
    {
        List<string> details = [];
        if (!string.IsNullOrWhiteSpace(texSourcePath))
            details.Add($"tex0={DescribeSidecarSource(texSourcePath, overlayRoot, donorRoot)}");

        if (!string.IsNullOrWhiteSpace(objSourcePath))
            details.Add($"obj0={DescribeSidecarSource(objSourcePath, overlayRoot, donorRoot)}");

        return details.Count == 0
            ? adtSourcePath
            : $"{adtSourcePath} [{string.Join(", ", details)}]";
    }

    private static string DescribeSidecarSource(string sourcePath, string? overlayRoot, string donorRoot)
    {
        if (!string.IsNullOrWhiteSpace(overlayRoot) && IsPathUnderRoot(sourcePath, overlayRoot))
            return $"original:{Path.GetFileName(sourcePath)}";

        if (IsPathUnderRoot(sourcePath, donorRoot))
            return $"lk-donor:{Path.GetFileName(sourcePath)}";

        return Path.GetFileName(sourcePath);
    }

    private static string ClassifySourceKind(string sourcePath, string? overlayRoot, string? donorRoot, bool allowArchive)
    {
        if (!string.IsNullOrWhiteSpace(overlayRoot) && !string.IsNullOrWhiteSpace(sourcePath) && IsPathUnderRoot(sourcePath, overlayRoot))
            return "original";

        if (!string.IsNullOrWhiteSpace(donorRoot) && !string.IsNullOrWhiteSpace(sourcePath) && IsPathUnderRoot(sourcePath, donorRoot))
            return "lk-donor";

        return allowArchive ? "client-archive" : "unknown";
    }

    private static string? ClassifyOptionalSidecarSource(string? sourcePath, string? overlayRoot, string? donorRoot)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
            return null;

        return ClassifySourceKind(sourcePath, overlayRoot, donorRoot, allowArchive: true);
    }

    private static bool IsOriginalSidecarSource(string? sourcePath, string? overlayRoot)
    {
        return !string.IsNullOrWhiteSpace(sourcePath)
            && !string.IsNullOrWhiteSpace(overlayRoot)
            && IsPathUnderRoot(sourcePath, overlayRoot);
    }

    private static bool IsPathUnderRoot(string path, string root)
    {
        string fullPath = Path.GetFullPath(path)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string fullRoot = Path.GetFullPath(root)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        return fullPath.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase);
    }

    private static bool ShouldAttemptAlphaDonor(AlphaDonorContext? donor, int tileX, int tileY)
    {
        if (donor is null)
            return false;

        return donor.AllowedTiles is not { Count: > 0 } || donor.AllowedTiles.Contains((tileX, tileY));
    }

    private static List<WdtTileCoordinate> BuildTileQueue(IReadOnlyList<WdtTileCoordinate> occupiedTiles, AlphaDonorContext? donor)
    {
        HashSet<(int tileX, int tileY)> seen = [];
        List<WdtTileCoordinate> tiles = new(occupiedTiles.Count + (donor?.AllowedTiles?.Count ?? 0));

        foreach (WdtTileCoordinate tile in occupiedTiles)
        {
            if (seen.Add((tile.TileX, tile.TileY)))
                tiles.Add(tile);
        }

        if (donor?.AllowedTiles is { Count: > 0 })
        {
            foreach ((int tileX, int tileY) in donor.AllowedTiles.OrderBy(static tile => tile.tileY).ThenBy(static tile => tile.tileX))
            {
                if (seen.Add((tileX, tileY)))
                    tiles.Add(new WdtTileCoordinate(tileX, tileY));
            }
        }

        tiles.Sort(static (left, right) =>
        {
            int byY = left.TileY.CompareTo(right.TileY);
            return byY != 0 ? byY : left.TileX.CompareTo(right.TileX);
        });

        return tiles;
    }

    private static bool TryParseTileSet(string value, out HashSet<(int tileX, int tileY)> tiles)
    {
        tiles = [];
        foreach (string token in value.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string[] parts = token.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length != 2)
                return false;

            if (!TryParseRange(parts[0], out IReadOnlyList<int> xs) || !TryParseRange(parts[1], out IReadOnlyList<int> ys))
                return false;

            foreach (int x in xs)
            foreach (int y in ys)
                tiles.Add((x, y));
        }

        return tiles.Count > 0;
    }

    private static bool TryParseRange(string text, out IReadOnlyList<int> values)
    {
        values = [];
        string[] bounds = text.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (bounds.Length == 0 || bounds.Length > 2)
            return false;

        if (!int.TryParse(bounds[0], out int start))
            return false;

        int end = start;
        if (bounds.Length == 2 && !int.TryParse(bounds[1], out end))
            return false;

        if (end < start)
            return false;

        List<int> expanded = new(end - start + 1);
        for (int value = start; value <= end; value++)
            expanded.Add(value);

        values = expanded;
        return true;
    }

    private static string FormatTileSet(HashSet<(int tileX, int tileY)>? tiles)
    {
        if (tiles is not { Count: > 0 })
            return "all-missing";

        return string.Join(';', tiles.OrderBy(static tile => tile.tileX).ThenBy(static tile => tile.tileY).Select(static tile => $"{tile.tileX},{tile.tileY}"));
    }

    private static string BuildMissingTileWarning(LkDonorContext? lkDonor, AlphaDonorContext? alphaDonor, string adtVirtualPath, int tileX, int tileY)
    {
        List<string> donorMisses = [];
        if (lkDonor is not null)
            donorMisses.Add($"LK donor '{lkDonor.Root}'");

        if (ShouldAttemptAlphaDonor(alphaDonor, tileX, tileY))
            donorMisses.Add($"Alpha donor '{alphaDonor!.DonorWdtVirtualPath}#{tileX},{tileY}'");

        return donorMisses.Count == 0
            ? $"Tile ({tileX},{tileY}): missing root ADT '{adtVirtualPath}'."
            : $"Tile ({tileX},{tileY}): missing root ADT '{adtVirtualPath}' and {string.Join(" plus ", donorMisses)} was unavailable.";
    }

    private static bool TryReadVirtualOrLooseFile(string virtualPath, string? overlayRoot, NativeMpqService catalog, out byte[]? bytes, out string sourcePath)
    {
        bytes = null;
        sourcePath = string.Empty;

        if (TryReadLooseVirtualFile(virtualPath, overlayRoot, out bytes, out sourcePath))
            return true;

        bytes = catalog.ReadFile(virtualPath);
        if (bytes is null || bytes.Length == 0)
            return false;

        sourcePath = virtualPath;
        return true;
    }

    private static bool TryReadLooseVirtualFile(string virtualPath, string? overlayRoot, out byte[]? bytes, out string sourcePath)
    {
        bytes = null;
        sourcePath = string.Empty;

        if (string.IsNullOrWhiteSpace(overlayRoot))
            return false;

        string root = Path.GetFullPath(overlayRoot);
        if (!Directory.Exists(root))
            return false;

        foreach (string candidate in BuildOverlayCandidates(root, virtualPath))
        {
            if (!File.Exists(candidate))
                continue;

            bytes = File.ReadAllBytes(candidate);
            sourcePath = candidate;
            return bytes.Length > 0;
        }

        return false;
    }

    private static IEnumerable<string> BuildOverlayCandidates(string overlayRoot, string virtualPath)
    {
        string normalizedVirtualPath = virtualPath.Replace('\\', '/').TrimStart('/');
        yield return Path.Combine(overlayRoot, normalizedVirtualPath.Replace('/', Path.DirectorySeparatorChar));

        const string worldMapsPrefix = "World/Maps/";
        if (!normalizedVirtualPath.StartsWith(worldMapsPrefix, StringComparison.OrdinalIgnoreCase))
            yield break;

        string relativeMapPath = normalizedVirtualPath[worldMapsPrefix.Length..];
        yield return Path.Combine(overlayRoot, relativeMapPath.Replace('/', Path.DirectorySeparatorChar));

        int separatorIndex = relativeMapPath.IndexOf('/');
        if (separatorIndex < 0 || separatorIndex == relativeMapPath.Length - 1)
            yield break;

        string mapName = relativeMapPath[..separatorIndex];
        string mapRelativePath = relativeMapPath[(separatorIndex + 1)..];
        string overlayLeaf = Path.GetFileName(overlayRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (string.Equals(overlayLeaf, mapName, StringComparison.OrdinalIgnoreCase))
            yield return Path.Combine(overlayRoot, mapRelativePath.Replace('/', Path.DirectorySeparatorChar));
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], longName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[i], shortName, StringComparison.OrdinalIgnoreCase))
            {
                return args[i + 1];
            }
        }

        return null;
    }

    private static int? GetIntOption(string[] args, string longName, string shortName)
    {
        string? value = GetOption(args, longName, shortName);
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return int.TryParse(value, out int parsed) ? parsed : null;
    }

    private static bool HasFlag(string[] args, string name)
    {
        foreach (string arg in args)
        {
            if (string.Equals(arg, name, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    private readonly record struct SplitAdtToLkOptions(
        string? ClientRoot,
        string? MapName,
        string? OverlayRoot,
        string? LkDonorRoot,
        string? AlphaDonorClientRoot,
        string? AlphaDonorMap,
        string? AlphaDonorTiles,
        string? ReportPath,
        string? OutputDir,
        bool Verbose);

    private sealed record LkDonorContext(string Root);

    private sealed record AlphaDonorContext(
        string ClientRoot,
        string MapName,
        string DonorWdtVirtualPath,
        byte[] WdtBytes,
        HashSet<(int tileX, int tileY)>? AllowedTiles);

    private sealed record SplitAdtTileSourceDetails(
        string Outcome,
        string RootSourceKind,
        string RootSourcePath,
        string? TextureSourceKind,
        string? TextureSourcePath,
        string? ObjectSourceKind,
        string? ObjectSourcePath,
        bool OriginalTexturePreserved,
        bool OriginalObjectPlacementsPreserved,
        string DisplaySource);

    private sealed record SplitAdtToLkReport(
        string MapName,
        string ClientRoot,
        string? OverlayRoot,
        string? LkDonorRoot,
        string? AlphaDonorClientRoot,
        string? AlphaDonorMap,
        string OutputDir,
        string WdtSourcePath,
        string WdtOutputPath,
        int TileCount,
        int Converted,
        int Failed,
        int WarningCount,
        long ElapsedMilliseconds,
        IReadOnlyList<SplitAdtToLkReportEntry> Entries,
        IReadOnlyDictionary<string, int> OutcomeCounts);

    private sealed record SplitAdtToLkReportEntry(
        int TileX,
        int TileY,
        string Outcome,
        string? RootSourceKind,
        string? RootSourcePath,
        string? TextureSourceKind,
        string? TextureSourcePath,
        string? ObjectSourceKind,
        string? ObjectSourcePath,
        bool OriginalTexturePreserved,
        bool OriginalObjectPlacementsPreserved,
        string? OutputPath,
        int? OutputByteLength,
        string? Message);
}