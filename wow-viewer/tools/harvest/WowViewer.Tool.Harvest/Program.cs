using System.Collections.Concurrent;
using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using StreamProfile = WowViewer.Core.IO.Maps.RawArraySerializer.StreamProfile;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tools.Harvest;

static class Program
{
    private static Dictionary<string, string>? _md5Lookup;
    private static readonly ConcurrentDictionary<string, Lazy<WlLooseFileEntry[]>> _wlLooseFileCache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly int DefaultHarvestTileWorkers = Math.Max(1, Math.Min(8, Environment.ProcessorCount));
    private sealed record HarvestMapDiscoveryResult(
        string Map,
        string DisplayName,
        bool Include,
        string Reason,
        bool IsAlpha,
        bool IsWmoBased,
        bool HasWorldModelAsset,
        int WorldModelNameCount,
        int TilesWithData,
        bool HasReadableTile,
        bool HasUsableTile,
        int? ProbeTileX,
        int? ProbeTileY);

    private sealed class WlLooseFileEntry
    {
        public required string Path { get; init; }
        public required WlFile File { get; init; }
    }

    private readonly record struct HarvestTileJob(int Order, int TileX, int TileY);
    private readonly record struct HarvestTileResult(int Order, byte[]? Blob, bool HadError, string? ErrorMessage);

    static int Main(string[] args)
    {
        Environment.ExitCode = 0;
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();

        switch (command)
        {
            case "harvest-tile":
                RunHarvestTile(tail);
                break;
            case "harvest-map":
                RunHarvestMap(tail);
                break;
            case "synthetic-minimap":
                RunSyntheticMinimap(tail);
                break;
            case "extract-unified":
                RunExtractUnified(tail);
                break;
            case "harvest-map-mpq":
                RunHarvestMapMpq(tail);
                break;
            case "harvest-stream":
                RunHarvestStream(tail);
                break;
            case "discover-maps":
                RunDiscoverMaps(tail);
                break;
            case "extract-holes":
                RunExtractHoles(tail);
                break;
            case "extract-tilesets":
                RunExtractTilesets(tail);
                break;
            default:
                Console.Error.WriteLine($"Unknown command '{command}'.");
                ShowUsage();
                return 1;
        }

        return Environment.ExitCode;
    }

    static void ShowUsage()
    {
        Console.WriteLine("""
            WowViewer.Tool.Harvest — V14 dataset generation tool

            Usage: WowViewer.Tool.Harvest <command> [options]

            Commands:
              harvest-tile      Extract NPZ shard from a single ADT tile (disk path)
              harvest-map       Batch-extract all tiles from a map directory (disk path)
              extract-unified   Extract NPZ shard from a tile inside MPQ archives
                                (reads tileset BLP + ADT + WDL from MPQ, outputs NPZ shard)
              harvest-stream    Stream raw tile blobs from a map to stdout
              discover-maps     List terrain-trainable maps from a staged client using
                                WDT summary + tile probe checks
              extract-holes     Dump raw per-chunk MCNK hole bitmasks (uint16, 4x4
                                hole groups) for every terrain tile of the given maps
                                to JSON (era-aware: alpha WDT + LK/split ADT)
              extract-tilesets  Decode the listed tileset BLPs from this client's
                                MPQs to PNGs + a manifest JSON (era-specific pixels)
              synthetic-minimap Composite tilesets + alpha → synthetic minimap

            Global options:
              --build, -b       Client build version (e.g. "4.3.4.15595") for
                               version-aware ADT profile selection. Auto-detected
                               from input path if not specified.
              --client-root     WoW client root directory (for extract-unified)
              --map, -m         Map name (e.g. "Azeroth") for extract-unified

            See --help on each command for options.
            """);
    }

    static void RunDiscoverMaps(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        LoadMd5Translate(clientRoot, catalog);

        MapDirectoryLookup lookup = new();
        lookup.Load(BuildClientSearchRoots(clientRoot), catalog);
        if (!lookup.IsLoaded)
        {
            Console.Error.WriteLine("Error: Map.dbc could not be loaded from the staged client.");
            Environment.ExitCode = 1;
            return;
        }

        List<HarvestMapDiscoveryResult> results = [];
        foreach (MapDirectoryEntry entry in lookup.Entries.OrderBy(static entry => entry.Directory, StringComparer.OrdinalIgnoreCase))
        {
            results.Add(DiscoverMap(catalog, entry));
        }

        string json = System.Text.Json.JsonSerializer.Serialize(
            results,
            new System.Text.Json.JsonSerializerOptions
            {
                PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
                WriteIndented = true
            });
        Console.WriteLine(json);
    }

    static string? TryFindDefaultListfilePath()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = Path.Combine(current.FullName, "libs", "wowdev", "wow-listfile", "listfile.txt");
            if (File.Exists(candidate))
                return candidate;

            current = current.Parent;
        }

        return null;
    }

    static void TryLoadSupplementalListfile(NativeMpqService catalog)
    {
        string? listfilePath = TryFindDefaultListfilePath();
        if (string.IsNullOrWhiteSpace(listfilePath) || !File.Exists(listfilePath))
            return;

        try
        {
            catalog.LoadListfile(listfilePath);
            Console.Error.WriteLine($"  Loaded supplemental listfile: {listfilePath}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: failed to load supplemental listfile '{listfilePath}': {ex.Message}");
        }
    }

    static string[] ReadSupplementalListfileEntriesForMap(string mapName)
    {
        string? listfilePath = TryFindDefaultListfilePath();
        if (string.IsNullOrWhiteSpace(listfilePath) || !File.Exists(listfilePath))
            return [];

        string mapPrefix = $"world\\maps\\{mapName}\\".ToLowerInvariant();
        return File.ReadLines(listfilePath)
            .Select(line => line.Trim())
            .Where(line => line.Length > 0)
            .Select(line => line.Replace('/', '\\'))
            .Where(path =>
                path.StartsWith(mapPrefix, StringComparison.OrdinalIgnoreCase) &&
                (path.EndsWith(".wlw", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlq", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wll", StringComparison.OrdinalIgnoreCase)))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    static void RunHarvestTile(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        string? textureSource = GetOption(args, "--texture-source", "-t");
        string? minimapRoot = GetOption(args, "--minimap-root", "-m");
        string? buildVersion = GetOption(args, "--build", "-b");
        int? tileXOpt = GetIntOption(args, "--tile-x", "-x");
        int? tileYOpt = GetIntOption(args, "--tile-y", "-y");

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: --input <root.adt or alpha.wdt> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (string.IsNullOrWhiteSpace(output))
        {
            string dir = Path.GetDirectoryName(input) ?? ".";
            string stem = Path.GetFileNameWithoutExtension(input);
            output = Path.Combine(dir, $"{stem}_harvest.npz");
        }

        try
        {
            TerrainTileTensorPack pack;

            if (Path.GetExtension(input).Equals(".wdt", StringComparison.OrdinalIgnoreCase)
                && TryDetectAlphaWdt(input, out bool isAlpha, out bool isWmoBased))
            {
                if (isWmoBased)
                {
                    Console.Error.WriteLine($"Error: {input} is a WMO-based Alpha map (no terrain tiles). "
                        + "WMO-based Alpha maps are not yet supported for tensor pack export.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!isAlpha)
                {
                    Console.Error.WriteLine($"Error: {input} is a Retail WDT file. "
                        + "Use a root ADT file (e.g., mapname_XX_YY.adt) instead.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!tileXOpt.HasValue || !tileYOpt.HasValue)
                {
                    Console.Error.WriteLine("Error: Alpha WDT tiles require --tile-x <0-63> and --tile-y <0-63>.");
                    Environment.ExitCode = 1;
                    return;
                }

                int tileX = tileXOpt.Value;
                int tileY = tileYOpt.Value;
                if ((uint)tileX >= 64 || (uint)tileY >= 64)
                {
                    Console.Error.WriteLine("Error: tile coordinates must be 0-63.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!AlphaWdtReader.TryReadTile(input, tileX, tileY, out AlphaTileData? tileData))
                {
                    Console.Error.WriteLine($"Error: tile ({tileX},{tileY}) is not present in {input} "
                        + "(or the tile data could not be parsed).");
                    Environment.ExitCode = 1;
                    return;
                }

                pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
            }
            else
            {
                pack = AdtTensorPackBuilder.Build(input, textureSource, buildVersion);
            }

            if (!string.IsNullOrWhiteSpace(minimapRoot))
            {
                if (TryLoadMinimap(input, minimapRoot, out byte[,,]? minimap))
                {
                    pack.MinimapRgb256 = minimap;
                    pack.MinimapSourceTag = "raw";
                    pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
                    {
                        "minimap_rgb_256"
                    };
                }
            }

            NpzTileSerializer.Serialize(pack, output);
            Console.WriteLine($"Harvested: {output}");
            Console.WriteLine($"Signals: {string.Join(", ", pack.AvailableSignals)}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error harvesting {input}: {ex.Message}");
            Environment.ExitCode = 1;
        }
    }

    private static bool TryDetectAlphaWdt(string wdtPath, out bool isAlpha, out bool isWmoBased)
    {
        isAlpha = false;
        isWmoBased = false;
        try
        {
            byte[] bytes = File.ReadAllBytes(wdtPath);
            isAlpha = AlphaWdtReader.IsAlphaWdt(bytes);

            if (isAlpha && bytes.Length >= 16)
            {
                int mphdDataOffset = 8 + 4;
                if (mphdDataOffset + 8 <= bytes.Length)
                {
                    int mdnmOffset = BitConverter.ToInt32(bytes, mphdDataOffset + 4);
                    int monmOffset = BitConverter.ToInt32(bytes, mphdDataOffset + 12);
                    if (mdnmOffset == 2 || monmOffset == 2)
                        isWmoBased = true;
                }
            }
        }
        catch
        {
        }
        return true;
    }

    static void RunHarvestMap(string[] args)
    {
        string? inputDir = GetOption(args, "--input-dir", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? outputDir = GetOption(args, "--output-dir", "-o");
        string? minimapRoot = GetOption(args, "--minimap-root", "-m");
        string? buildVersion = GetOption(args, "--build", "-b");
        int limit = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;
        bool overwrite = HasFlag(args, "--overwrite");

        if (string.IsNullOrWhiteSpace(inputDir) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --input-dir <dir> and --output-dir <dir> are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(inputDir))
        {
            Console.Error.WriteLine($"Error: input directory not found: {inputDir}");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputDir);

        int processed = 0;
        int skipped = 0;

        foreach (string adtPath in Directory.EnumerateFiles(inputDir, "*.adt")
            .Where(p => !p.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
                && !p.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
                && !p.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
            .Take(limit))
        {
            string stem = Path.GetFileNameWithoutExtension(adtPath);
            string outputPath = Path.Combine(outputDir, $"{stem}_harvest.npz");

            if (!overwrite && File.Exists(outputPath))
            {
                skipped++;
                continue;
            }

            string candidateTextureSource = Path.Combine(inputDir, $"{stem}_tex0.adt");
            string? textureSource = File.Exists(candidateTextureSource) ? candidateTextureSource : null;

            try
            {
                TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(adtPath, textureSource, buildVersion);

                if (!string.IsNullOrWhiteSpace(minimapRoot))
                {
                    if (TryLoadMinimap(adtPath, minimapRoot, out byte[,,]? minimap))
                    {
                        pack.MinimapRgb256 = minimap;
                        pack.MinimapSourceTag = "raw";
                        pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
                        {
                            "minimap_rgb_256"
                        };
                    }
                }

                NpzTileSerializer.Serialize(pack, outputPath);
                processed++;
                Console.WriteLine($"Harvested: {stem}");
            }
            catch (Exception ex)
            {
                skipped++;
                Console.Error.WriteLine($"Skipped {stem}: {ex.Message}");
            }
        }

        Console.WriteLine($"Done. Processed={processed} Skipped={skipped}");
    }

    static void RunSyntheticMinimap(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: --input <npz> is required.");
            Environment.ExitCode = 1;
            return;
        }

        Console.WriteLine("Synthetic minimap compositor ready — texture pixel lookup not yet wired.");
    }

    static void RunHarvestMapMpq(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        string? outputDir = GetOption(args, "--output-dir", "-o");
        int? limit = GetIntOption(args, "--limit", "-n");
        bool force = HasFlag(args, "--force");
        int maxTiles = limit ?? int.MaxValue;

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --client-root, --map, and --output-dir are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);
        Console.WriteLine($"Build version: {buildVersion ?? "(unknown)"}");

        Directory.CreateDirectory(outputDir);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        int extracted = 0, skipped = 0, errors = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int tx = 0; tx < 64; tx++)
        {
            for (int ty = 0; ty < 64; ty++)
            {
                if (extracted >= maxTiles) break;

                string outputPath = Path.Combine(outputDir, $"{mapName}_{tx}_{ty}_harvest.npz");
                if (!force && File.Exists(outputPath)) { skipped++; continue; }

                try
                {
                    var oldErr = Console.Error;
                    Console.SetError(TextWriter.Null);
                    try
                    {
                        if (RunExtractTileFromMpq(catalog, clientRoot, mapName, wdtBytes, tx, ty, outputPath, exportPlacements: false, buildVersion: buildVersion))
                            extracted++;
                    }
                    finally { Console.SetError(oldErr); }
                }
                catch { }
            }
        }

        sw.Stop();
        Console.WriteLine($"Done. Extracted={extracted} Skipped={skipped} Errors={errors} in {sw.Elapsed.TotalSeconds:F0}s ({sw.Elapsed.TotalSeconds / Math.Max(1, extracted):F1}s/tile)");
    }

    static void RunHarvestStream(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        int? limit = GetIntOption(args, "--limit", "-n");
        int maxTiles = limit ?? int.MaxValue;
        int tileWorkers = Math.Max(1, GetIntOption(args, "--tile-workers", "--tile-workers") ?? DefaultHarvestTileWorkers);
        string streamProfileRaw = (GetOption(args, "--stream-profile", "") ?? "v16").Trim();
        StreamProfile streamProfile = ResolveStreamProfile(streamProfileRaw);

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("Error: --client-root and --map are required for harvest-stream.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        // Redirect Console.Out to stderr so stdout is pure binary
        var originalOut = Console.Out;
        Console.SetOut(Console.Error);

        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        int extracted = 0;
        int errors = 0;
        string? firstError = null;
        var stdout = Console.OpenStandardOutput();
        bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);

        if (tileWorkers <= 1)
        {
            for (int tx = 0; tx < 64; tx++)
            {
                for (int ty = 0; ty < 64; ty++)
                {
                    if (extracted >= maxTiles) break;

                    HarvestTileResult result = HarvestStreamTileWorker(
                        catalog,
                        clientRoot,
                        mapName,
                        wdtBytes,
                        isAlpha,
                        buildVersion,
                        streamProfile,
                        new HarvestTileJob(extracted + errors, tx, ty));
                    if (result.HadError)
                    {
                        errors++;
                        firstError ??= result.ErrorMessage;
                        continue;
                    }
                    if (result.Blob is null)
                        continue;

                    WriteHarvestStreamBlob(stdout, result.Blob);
                    extracted++;
                }
                if (extracted >= maxTiles) break;
            }
        }
        else
        {
            Console.Error.WriteLine($"  harvest-stream tile_workers={tileWorkers} profile={streamProfileRaw}");
            List<HarvestTileJob> jobs = new(64 * 64);
            int order = 0;
            for (int tx = 0; tx < 64; tx++)
                for (int ty = 0; ty < 64; ty++)
                    jobs.Add(new HarvestTileJob(order++, tx, ty));

            int prefetch = Math.Max(tileWorkers * 2, tileWorkers);
            Dictionary<int, Task<HarvestTileResult>> inflight = [];
            int nextLaunch = 0;

            void LaunchUntilWindowFull()
            {
                while (nextLaunch < jobs.Count && inflight.Count < prefetch)
                {
                    HarvestTileJob job = jobs[nextLaunch++];
                    inflight[job.Order] = Task.Run(() =>
                        HarvestStreamTileWorker(catalog, clientRoot, mapName, wdtBytes, isAlpha, buildVersion, streamProfile, job));
                }
            }

            LaunchUntilWindowFull();
            for (int nextOrder = 0; nextOrder < jobs.Count; nextOrder++)
            {
                HarvestTileResult result = inflight[nextOrder].GetAwaiter().GetResult();
                inflight.Remove(nextOrder);
                LaunchUntilWindowFull();

                if (result.HadError)
                {
                    errors++;
                    firstError ??= result.ErrorMessage;
                    continue;
                }
                if (result.Blob is null)
                    continue;

                WriteHarvestStreamBlob(stdout, result.Blob);
                extracted++;
                if (extracted >= maxTiles)
                    break;
            }

            if (inflight.Count > 0)
                Task.WaitAll(inflight.Values.ToArray());
        }

        // Write end marker: 4 bytes "ENDS" + 4 zero bytes
        byte[] endMarker = new byte[8];
        System.Text.Encoding.ASCII.GetBytes("ENDS").CopyTo(endMarker, 0);
        stdout.Write(endMarker, 0, 8);
        stdout.Flush();

        Console.Error.WriteLine($"Streamed {extracted} tiles, {errors} errors");
        if (firstError is not null)
            Console.Error.WriteLine($"First harvest-stream tile error: {firstError}");
        if (extracted == 0)
            Environment.ExitCode = 1;
    }

    private static HarvestTileResult HarvestStreamTileWorker(
        NativeMpqService catalog,
        string clientRoot,
        string mapName,
        byte[] wdtBytes,
        bool isAlpha,
        string? buildVersion,
        StreamProfile streamProfile,
        HarvestTileJob job)
    {
        try
        {
            return new HarvestTileResult(
                job.Order,
                TryBuildHarvestStreamBlob(catalog, clientRoot, mapName, wdtBytes, isAlpha, buildVersion, streamProfile, job.TileX, job.TileY),
                false,
                null);
        }
        catch (Exception ex)
        {
            return new HarvestTileResult(job.Order, null, true, $"tile ({job.TileX},{job.TileY}): {ex.Message}");
        }
    }

    private static byte[]? TryBuildHarvestStreamBlob(
        NativeMpqService catalog,
        string clientRoot,
        string mapName,
        byte[] wdtBytes,
        bool isAlpha,
        string? buildVersion,
        StreamProfile streamProfile,
        int tileX,
        int tileY)
    {
        TerrainTileTensorPack? pack = null;
        if (isAlpha)
        {
            string alphaWdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, alphaWdtVirtual, out AlphaTileData? tileData) || tileData is null)
                return null;
            pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
        }
        else
        {
            pack = BuildPackFromArchiveAdt(catalog, mapName, tileX, tileY, buildVersion);
        }

        if (pack is null)
            return null;

        if (pack.UnifiedLiquidMask is null)
            TryAddWlLiquidFromArchiveFiles(catalog, clientRoot, mapName, tileX, tileY, pack);

        if (pack.MinimapRgb256 is null)
        {
            byte[,,]? minimapRgb = TryLoadMinimapFromMpq(catalog, mapName, tileX, tileY);
            if (minimapRgb is not null)
            {
                pack.MinimapRgb256 = minimapRgb;
                pack.MinimapSourceTag = "mpq_blp";
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "minimap_rgb" };
            }
        }

        if (streamProfile == StreamProfile.Full && pack.MclyTextureNames.Count > 0)
        {
            var texPixels = new List<byte[,,]>();
            foreach (string texName in pack.MclyTextureNames)
            {
                byte[,,]? pixels = LoadTextureFromMpq(catalog, texName);
                if (pixels is not null)
                    texPixels.Add(pixels);
            }
            if (texPixels.Count > 0)
            {
                pack.MclyTexturePixels = texPixels;
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "mcly_texture_pixels" };
            }
        }

        using var ms = new MemoryStream();
        RawArraySerializer.Serialize(pack, ms, streamProfile);
        return ms.ToArray();
    }

    private static StreamProfile ResolveStreamProfile(string value)
    {
        if (value.Equals("full", StringComparison.OrdinalIgnoreCase))
            return StreamProfile.Full;
        if (value.Equals("v22", StringComparison.OrdinalIgnoreCase))
            return StreamProfile.V22;
        return StreamProfile.V16;
    }

    private static void WriteHarvestStreamBlob(Stream stdout, byte[] blob)
    {
        byte[] header = new byte[8];
        System.Text.Encoding.ASCII.GetBytes("ARRY").CopyTo(header, 0);
        BitConverter.TryWriteBytes(header.AsSpan(4, 4), blob.Length);
        stdout.Write(header, 0, 8);
        stdout.Write(blob, 0, blob.Length);
    }

    static void RunExtractUnified(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        string? output = GetOption(args, "--output", "-o");
        int? tileX = GetIntOption(args, "--tile-x", "-x");
        int? tileY = GetIntOption(args, "--tile-y", "-y");
        bool exportPlacements = HasFlag(args, "--export-placements");
        string? syntheticMinimap = GetOption(args, "--synthetic-minimap", "-s");
        bool dumpHex = HasFlag(args, "--dump-hex");

        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);

        if (string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("Error: --map <name> is required.");
            Environment.ExitCode = 1;
            return;
        }

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        if (tileX.HasValue && tileY.HasValue)
        {
            if (!RunExtractTileFromMpq(catalog, clientRoot, mapName, wdtBytes, tileX.Value, tileY.Value, output, exportPlacements, syntheticMinimap, buildVersion: buildVersion))
                Environment.ExitCode = 1;
        }
        else
        {
            using var ms = new MemoryStream(wdtBytes);
            MapFileSummary fileSummary = MapFileSummaryReader.Read(ms, wdtVirtual);
            var wdt = WdtSummaryReader.Read(ms, fileSummary);
            Console.WriteLine($"WDT: {mapName}");
            Console.WriteLine($"  IsWmoBased: {wdt.IsWmoBased}");
            Console.WriteLine($"  Tiles with data: {wdt.TilesWithData}/{wdt.TotalTiles}");
            Console.WriteLine();
            Console.WriteLine("Use --tile-x <0-63> --tile-y <0-63> to harvest a specific tile.");
        }
    }

    static void RunExtractHoles(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapsOption = GetOption(args, "--maps", "-m");
        string? output = GetOption(args, "--output", "-o");

        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required and must exist.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(mapsOption))
        {
            Console.Error.WriteLine("Error: --maps <comma-separated map names> is required.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(output))
        {
            Console.Error.WriteLine("Error: --output <json path> is required.");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string[] maps = mapsOption.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var mapsOut = new Dictionary<string, List<Dictionary<string, object>>>();
        int totalTiles = 0;
        int totalHoled = 0;

        foreach (string mapName in maps)
        {
            string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
            if (wdtBytes is null)
            {
                Console.Error.WriteLine($"Warning: WDT not readable for map '{mapName}', skipping.");
                continue;
            }

            var tiles = new List<Dictionary<string, object>>();
            bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);

            for (int tileY = 0; tileY < 64; tileY++)
            {
                for (int tileX = 0; tileX < 64; tileX++)
                {
                    ushort[,]? holesYx = null;

                    if (isAlpha)
                    {
                        if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, wdtVirtual, out AlphaTileData? tileData) || tileData?.HoleFullMasks is null)
                            continue;
                        // Alpha reader stores [cx, cy]; normalize to [y, x].
                        holesYx = new ushort[16, 16];
                        for (int cy = 0; cy < 16; cy++)
                            for (int cx = 0; cx < 16; cx++)
                                holesYx[cy, cx] = tileData.HoleFullMasks[cx, cy];
                    }
                    else
                    {
                        string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
                        byte[]? adtBytes = catalog.ReadFile(adtVirtual);
                        if (adtBytes is null)
                            continue;
                        using var ms = new MemoryStream(adtBytes);
                        holesYx = AdtTensorPackBuilder.ReadHoleBitmasks(ms, adtVirtual);
                        if (holesYx is null)
                            continue;
                    }

                    int[] flat = new int[256];
                    bool anyHole = false;
                    for (int cy = 0; cy < 16; cy++)
                    {
                        for (int cx = 0; cx < 16; cx++)
                        {
                            flat[cy * 16 + cx] = holesYx[cy, cx];
                            anyHole |= holesYx[cy, cx] != 0;
                        }
                    }

                    tiles.Add(new Dictionary<string, object>
                    {
                        ["x"] = tileX,
                        ["y"] = tileY,
                        ["holes"] = flat,
                    });
                    totalTiles++;
                    if (anyHole)
                        totalHoled++;
                }
            }

            mapsOut[mapName] = tiles;
            Console.WriteLine($"extract-holes: {mapName}: {tiles.Count} tiles");
        }

        var payload = new Dictionary<string, object>
        {
            ["build_version"] = buildVersion ?? "",
            ["client_root"] = clientRoot,
            ["hole_field"] = "mcnk_holes_uint16_row_major_yx",
            ["maps"] = mapsOut,
        };

        string json = System.Text.Json.JsonSerializer.Serialize(payload);
        File.WriteAllText(output, json);
        Console.WriteLine($"extract-holes: wrote {totalTiles} tiles ({totalHoled} with holes) -> {output}");
    }

    static void RunExtractTilesets(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? pathsFile = GetOption(args, "--paths-file", "-p");
        string? outputDir = GetOption(args, "--output-dir", "-o");

        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required and must exist.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(pathsFile) || !File.Exists(pathsFile))
        {
            Console.Error.WriteLine("Error: --paths-file <txt, one BLP virtual path per line> is required.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --output-dir <dir> is required.");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        Directory.CreateDirectory(outputDir);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string[] paths = File.ReadAllLines(pathsFile)
            .Select(l => l.Trim())
            .Where(l => l.Length > 0)
            .ToArray();

        var entries = new List<Dictionary<string, object>>();
        int failed = 0;
        int index = 0;
        foreach (string virtualPath in paths)
        {
            // Dataset stores carry normalized names (forward slashes, .png);
            // MPQ lookup wants the raw BLP virtual path. Try as-given first,
            // then the BLP-converted form. The manifest keeps the original
            // string so downstream joins stay exact.
            string converted = virtualPath.Replace('/', '\\');
            if (converted.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                converted = converted[..^4] + ".blp";

            byte[,,]? rgb = LoadTextureFromMpq(catalog, virtualPath)
                ?? (converted != virtualPath ? LoadTextureFromMpq(catalog, converted) : null);
            if (rgb is null)
            {
                failed++;
                continue;
            }

            int h = rgb.GetLength(0);
            int w = rgb.GetLength(1);
            string fileName = $"t{index:D4}.png";
            using (var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(w, h))
            {
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2], 255);
                using var fs = File.Create(Path.Combine(outputDir, fileName));
                image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            }

            entries.Add(new Dictionary<string, object>
            {
                ["path"] = virtualPath,
                ["file"] = fileName,
                ["width"] = w,
                ["height"] = h,
            });
            index++;
        }

        var payload = new Dictionary<string, object>
        {
            ["build_version"] = DetectBuildVersionFromClientRoot(clientRoot) ?? "",
            ["client_root"] = clientRoot,
            ["tilesets"] = entries,
        };
        string manifestPath = Path.Combine(outputDir, "manifest.json");
        File.WriteAllText(manifestPath, System.Text.Json.JsonSerializer.Serialize(payload));
        Console.WriteLine($"extract-tilesets: {entries.Count} decoded, {failed} failed -> {manifestPath}");
    }

    static bool RunExtractTileFromMpq(NativeMpqService catalog, string clientRoot, string mapName, byte[] wdtBytes, int tileX, int tileY, string? outputPath, bool exportPlacements, string? syntheticMinimapPath = null, string? buildVersion = null)
    {
        TerrainTileTensorPack pack;
        AdtPlacementCatalog? placementCatalog = null;

if (AlphaWdtReader.IsAlphaWdt(wdtBytes))
            {
                string alphaWdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
                if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, alphaWdtVirtual, out AlphaTileData? tileData) || tileData is null)
                {
                    Console.Error.WriteLine($"Error: Alpha tile ({tileX},{tileY}) not present in WDT.");
                    return false;
                }
                pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
                if (exportPlacements)
                    placementCatalog = tileData.ToPlacementCatalog();
            }
            else
            {
                pack = BuildPackFromArchiveAdt(catalog, mapName, tileX, tileY, buildVersion);
                if (pack is null)
                {
                    string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
                    Console.Error.WriteLine($"Error: Could not read ADT '{adtVirtual}' from client.");
                    return false;
                }
            }

            // Try loose WL* liquid fallback for tiles with no MH2O/MCLQ coverage.
            if (pack.UnifiedLiquidMask is null)
                TryAddWlLiquidFromArchiveFiles(catalog, clientRoot, mapName, tileX, tileY, pack);

        if (pack.MinimapRgb256 is null)
        {
            byte[,,]? minimapRgb = TryLoadMinimapFromMpq(catalog, mapName, tileX, tileY);
            if (minimapRgb is not null)
            {
                pack.MinimapRgb256 = minimapRgb;
                pack.MinimapSourceTag = "mpq_blp";
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "minimap_rgb" };
            }
        }

        // Load texture swatches from MPQ for tileset identification training
        if (pack.MclyTextureNames.Count > 0)
        {
            var texPixels = new List<byte[,,]>();
            foreach (string texName in pack.MclyTextureNames)
            {
                byte[,,]? pixels = LoadTextureFromMpq(catalog, texName);
                if (pixels is not null)
                {
                    texPixels.Add(pixels);
                }
            }
            if (texPixels.Count > 0)
            {
                pack.MclyTexturePixels = texPixels;
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "mcly_texture_pixels" };
            }
        }

        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), $"{mapName}_{tileX}_{tileY}_v14.npz");

        if (outputPath == "-")
        {
            // Write raw binary to stdout as length-prefixed blob
            using var ms = new MemoryStream();
            RawArraySerializer.Serialize(pack, ms);
            byte[] blob = ms.ToArray();
            var stdout = Console.OpenStandardOutput();
            // 8-byte header: magic "ARRY" + 4-byte little-endian length
            stdout.Write(System.Text.Encoding.ASCII.GetBytes("ARRY"), 0, 4);
            byte[] lenBytes = BitConverter.GetBytes(blob.Length);
            stdout.Write(lenBytes, 0, 4);
            stdout.Write(blob, 0, blob.Length);
            stdout.Flush();
        }
        else
        {
            NpzTileSerializer.Serialize(pack, outputPath);
            Console.WriteLine($"Harvested: {outputPath}");
            Console.WriteLine($"Signals: {string.Join(", ", pack.AvailableSignals)}");

            if (pack.MinimapRgb256 != null)
                Console.WriteLine($"  minimap: 256x256 RGB from MPQ");
        }

        if (exportPlacements && placementCatalog is not null)
        {
            string placementPath = Path.ChangeExtension(outputPath, ".placement.json");
            string json = System.Text.Json.JsonSerializer.Serialize(placementCatalog, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(placementPath, json);
            Console.WriteLine($"Placements: {placementPath}");
        }

        if (!string.IsNullOrWhiteSpace(syntheticMinimapPath))
            GenerateSyntheticMinimap(catalog, pack, tileX, tileY, syntheticMinimapPath);

        return true;
    }

    private static HarvestMapDiscoveryResult DiscoverMap(NativeMpqService catalog, MapDirectoryEntry entry)
    {
        string mapName = entry.Directory;
        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null || wdtBytes.Length == 0)
        {
            return new HarvestMapDiscoveryResult(
                Map: mapName,
                DisplayName: entry.Name,
                Include: false,
                Reason: "wdt_missing",
                IsAlpha: false,
                IsWmoBased: false,
                HasWorldModelAsset: false,
                WorldModelNameCount: 0,
                TilesWithData: 0,
                HasReadableTile: false,
                HasUsableTile: false,
                ProbeTileX: null,
                ProbeTileY: null);
        }

        using MemoryStream ms = new(wdtBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(ms, wdtVirtual);
        WdtSummary summary = WdtSummaryReader.Read(ms, fileSummary);
        IReadOnlyList<WdtTileCoordinate> occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(ms, fileSummary);
        bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);
        bool hasWorldModelAsset = summary.WorldModelNameCount > 0;

        int? probeTileX = null;
        int? probeTileY = null;
        bool hasReadableTile = false;
        bool hasUsableTile = false;

        if (summary.TilesWithData > 0)
        {
            foreach (WdtTileCoordinate tile in occupiedTiles)
            {
                ProbeTileState state = ProbeMapTile(catalog, mapName, tile, wdtBytes, isAlpha);
                if (!state.HasReadableTile)
                    continue;

                hasReadableTile = true;
                if (probeTileX is null || probeTileY is null)
                {
                    probeTileX = tile.TileX;
                    probeTileY = tile.TileY;
                }

                if (!state.HasUsableTile)
                    continue;

                hasUsableTile = true;
                probeTileX = tile.TileX;
                probeTileY = tile.TileY;
                break;
            }
        }

        string reason;
        bool include;
        if (summary.TilesWithData <= 0 && hasWorldModelAsset)
        {
            include = false;
            reason = "wmo_only";
        }
        else if (summary.TilesWithData <= 0)
        {
            include = false;
            reason = "no_tiles";
        }
        else if (!hasReadableTile)
        {
            include = false;
            reason = "no_readable_tile";
        }
        else if (!hasUsableTile)
        {
            include = false;
            reason = "no_v16_usable_tile";
        }
        else if (hasWorldModelAsset)
        {
            include = true;
            reason = "terrain_plus_wmo";
        }
        else
        {
            include = true;
            reason = "terrain";
        }

        return new HarvestMapDiscoveryResult(
            Map: mapName,
            DisplayName: entry.Name,
            Include: include,
            Reason: reason,
            IsAlpha: isAlpha,
            IsWmoBased: summary.IsWmoBased,
            HasWorldModelAsset: hasWorldModelAsset,
            WorldModelNameCount: summary.WorldModelNameCount,
            TilesWithData: summary.TilesWithData,
            HasReadableTile: hasReadableTile,
            HasUsableTile: hasUsableTile,
            ProbeTileX: probeTileX,
            ProbeTileY: probeTileY);
    }

    private readonly record struct ProbeTileState(bool HasReadableTile, bool HasUsableTile);

    private static ProbeTileState ProbeMapTile(
        NativeMpqService catalog,
        string mapName,
        WdtTileCoordinate tile,
        byte[] wdtBytes,
        bool isAlpha)
    {
        if (isAlpha)
        {
            string alphaWdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            if (!AlphaWdtReader.TryReadTile(wdtBytes, tile.TileX, tile.TileY, alphaWdtVirtual, out AlphaTileData? tileData) || tileData is null)
                return new ProbeTileState(false, false);

            TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tileData, tile.TileX, tile.TileY);
            if (pack.MinimapRgb256 is null)
                pack.MinimapRgb256 = TryLoadMinimapFromMpq(catalog, mapName, tile.TileX, tile.TileY);

            bool hasUsableTile = pack.Height257 is not null && pack.MinimapRgb256 is not null;
            return new ProbeTileState(true, hasUsableTile);
        }

        TerrainTileTensorPack? archivePack = BuildPackFromArchiveAdt(catalog, mapName, tile.TileX, tile.TileY, buildVersion: null);
        if (archivePack is null)
            return new ProbeTileState(false, false);

        if (archivePack.MinimapRgb256 is null)
            archivePack.MinimapRgb256 = TryLoadMinimapFromMpq(catalog, mapName, tile.TileX, tile.TileY);

        bool usable = archivePack.Height257 is not null && archivePack.MinimapRgb256 is not null;
        return new ProbeTileState(true, usable);
    }

    private static IReadOnlyList<string> BuildClientSearchRoots(string clientRoot)
    {
        List<string> roots = [];
        string dataRoot = Path.Combine(clientRoot, "Data");
        if (Directory.Exists(dataRoot))
            roots.Add(dataRoot);

        if (!string.Equals(clientRoot, dataRoot, StringComparison.OrdinalIgnoreCase))
            roots.Add(clientRoot);

        return roots.Count > 0 ? roots : [clientRoot];
    }

    private static TerrainTileTensorPack? BuildPackFromArchiveAdt(
        NativeMpqService catalog,
        string mapName,
        int tileX,
        int tileY,
        string? buildVersion)
    {
        string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
        byte[]? adtBytes = catalog.ReadFile(adtVirtual);
        if (adtBytes is null)
            return null;

        string tex0Virtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_tex0.adt";
        string obj0Virtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_obj0.adt";
        byte[]? tex0Bytes = catalog.ReadFile(tex0Virtual);
        byte[]? obj0Bytes = catalog.ReadFile(obj0Virtual);

        return AdtTensorPackBuilder.BuildFromBytes(
            adtVirtual,
            adtBytes,
            tex0Bytes,
            obj0Bytes,
            buildVersion,
            tex0Bytes is not null ? tex0Virtual : null,
            obj0Bytes is not null ? obj0Virtual : null,
            catalog.ReadFile);
    }

    static void GenerateSyntheticMinimap(NativeMpqService catalog, TerrainTileTensorPack pack, int tileX, int tileY, string outputPath)
    {
        if (pack.McalAlphaPack256 is null || pack.MclyTextureIds is null)
        {
            Console.Error.WriteLine("Error: tile has no MCAL/MCLY data, cannot generate synthetic minimap.");
            return;
        }

        var textureNames = pack.MclyTextureNames;
        if (textureNames.Count == 0)
        {
            Console.Error.WriteLine("Error: no texture names in tile metadata.");
            return;
        }

        var usedIds = new HashSet<int>();
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                for (int l = 0; l < 4; l++)
                    if (pack.MclyTextureIds[cy, cx, l] >= 0)
                        usedIds.Add(pack.MclyTextureIds[cy, cx, l]);

        Console.WriteLine($"  Loading {usedIds.Count} unique textures for synthetic minimap...");

        var textures = new Dictionary<int, byte[,,]>();
        int maxId = usedIds.Max();
        for (int id = 0; id <= maxId && id < textureNames.Count; id++)
        {
            if (!usedIds.Contains(id)) continue;
            string texPath = textureNames[id];
            byte[,,]? rgb = LoadTextureFromMpq(catalog, texPath);
            if (rgb is not null)
                textures[id] = rgb;
            else
                Console.Error.WriteLine($"    Warning: Could not load texture [{id}] {texPath}");
        }

        if (textures.Count == 0)
        {
            Console.Error.WriteLine("Error: no textures could be loaded.");
            return;
        }

        Console.WriteLine($"  Loaded {textures.Count}/{usedIds.Count} textures, compositing...");

        const int size = 256;
        const float tileSize = 533.33333f;
        const float mapOrigin = 17066.666f;
        const float textureScale = 20f;

        using var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(size, size);

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int chunkY = y / 16;
                int chunkX = x / 16;

                // World position for proper texture UV tiling
                float worldX = mapOrigin - tileX * tileSize + (x / (float)(size - 1)) * tileSize;
                float worldY = mapOrigin - tileY * tileSize + (y / (float)(size - 1)) * tileSize;

                float r = 0f, g = 0f, b = 0f;
                float implicitAlpha = 1f;

                for (int l = 1; l < 4; l++)
                {
                    float a = pack.McalAlphaPack256[y, x, l];
                    implicitAlpha -= a;
                    if (a <= 0.001f) continue;

                    int texId = pack.MclyTextureIds[chunkY, chunkX, l];
                    if (texId < 0 || !textures.TryGetValue(texId, out var tex)) continue;

                    int texH = tex.GetLength(0);
                    int texW = tex.GetLength(1);
                    float tu = (worldX / textureScale) % 1f;
                    float tv = (worldY / textureScale) % 1f;
                    if (tu < 0) tu += 1f;
                    if (tv < 0) tv += 1f;
                    int tx = (int)(tu * texW) % texW;
                    int ty = (int)(tv * texH) % texH;
                    r += tex[ty, tx, 0] * a;
                    g += tex[ty, tx, 1] * a;
                    b += tex[ty, tx, 2] * a;
                }

                if (implicitAlpha > 0.001f)
                {
                    int baseTexId = pack.MclyTextureIds[chunkY, chunkX, 0];
                    if (baseTexId >= 0 && textures.TryGetValue(baseTexId, out var baseTex))
                    {
                        int texH = baseTex.GetLength(0);
                        int texW = baseTex.GetLength(1);
                        float tu = (worldX / textureScale) % 1f;
                        float tv = (worldY / textureScale) % 1f;
                        if (tu < 0) tu += 1f;
                        if (tv < 0) tv += 1f;
                        int tx = (int)(tu * texW) % texW;
                        int ty = (int)(tv * texH) % texH;
                        r += baseTex[ty, tx, 0] * implicitAlpha;
                        g += baseTex[ty, tx, 1] * implicitAlpha;
                        b += baseTex[ty, tx, 2] * implicitAlpha;
                    }
                }

                image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(
                    (byte)Math.Clamp((int)r, 0, 255),
                    (byte)Math.Clamp((int)g, 0, 255),
                    (byte)Math.Clamp((int)b, 0, 255),
                    255);
            }
        }

        string? dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        using var fs = File.Create(outputPath);
        image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
        Console.WriteLine($"Synthetic minimap: {outputPath}");
    }

    static WlLooseFileEntry[] GetArchiveWlFiles(NativeMpqService catalog, string clientRoot, string mapName)
    {
        string cacheKey = $"{Path.GetFullPath(clientRoot)}|{mapName}";
        Lazy<WlLooseFileEntry[]> lazy = _wlLooseFileCache.GetOrAdd(
            cacheKey,
            _ => new Lazy<WlLooseFileEntry[]>(
                () => LoadArchiveWlFiles(catalog, clientRoot, mapName),
                LazyThreadSafetyMode.ExecutionAndPublication));
        return lazy.Value;
    }

    static WlLooseFileEntry[] LoadArchiveWlFiles(NativeMpqService catalog, string clientRoot, string mapName)
    {
        string mapPrefix = $"World\\Maps\\{mapName}\\";
        string[] paths = catalog.GetAllKnownFiles()
            .Select(path => path.Replace('/', '\\'))
            .Where(path =>
                path.StartsWith(mapPrefix, StringComparison.OrdinalIgnoreCase) &&
                (path.EndsWith(".wlw", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlq", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wll", StringComparison.OrdinalIgnoreCase)))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (paths.Length == 0)
            paths = ReadSupplementalListfileEntriesForMap(mapName);

        if (paths.Length == 0)
        {
            Console.WriteLine($"  [WL] no WL* files found in loaded archives for {mapName}");
            return [];
        }

        var loaded = new List<WlLooseFileEntry>(paths.Length);
        foreach (string path in paths)
        {
            try
            {
                byte[]? data = catalog.ReadFile(path);
                if (data is null || data.Length == 0)
                    continue;

                using var ms = new MemoryStream(data, writable: false);
                loaded.Add(new WlLooseFileEntry
                {
                    Path = path,
                    File = WlFileReader.Read(ms, path)
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [WL] failed to read {Path.GetFileName(path)}: {ex.Message}");
            }
        }

        Console.WriteLine($"  [WL] discovered {loaded.Count}/{paths.Length} WL* files in loaded archives for {mapName}");
        return loaded.ToArray();
    }

    static void TryAddWlLiquidFromArchiveFiles(NativeMpqService catalog, string clientRoot, string mapName, int tileX, int tileY, TerrainTileTensorPack pack)
    {
        WlLooseFileEntry[] wlFiles = GetArchiveWlFiles(catalog, clientRoot, mapName);
        if (wlFiles.Length == 0)
            return;

        bool any = false;
        float[,]? mask = null;
        float[,]? heights = null;

        const int size = 257;
        const float tileWorldSize = 533.33333f;
        const float mapOrigin = 17066.666f;

        foreach (WlLooseFileEntry entry in wlFiles)
        {
            foreach (var block in entry.File.Blocks)
            {
                Vector3 pos = block.WorldPosition;
                int blockTileX = Math.Clamp((int)Math.Floor((mapOrigin - pos.Y) / tileWorldSize), 0, 63);
                int blockTileY = Math.Clamp((int)Math.Floor((mapOrigin - pos.X) / tileWorldSize), 0, 63);
                if (blockTileX != tileX || blockTileY != tileY)
                    continue;

                float avgH = block.Vertices.Average(v => v.Z);
                float localX = (mapOrigin - pos.Y) - (tileX * tileWorldSize);
                float localY = (mapOrigin - pos.X) - (tileY * tileWorldSize);
                int cx = Math.Clamp((int)(localX / tileWorldSize * (size - 1)), 0, size - 1);
                int cy = Math.Clamp((int)(localY / tileWorldSize * (size - 1)), 0, size - 1);

                mask ??= new float[size, size];
                heights ??= new float[size, size];

                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int px = Math.Clamp(cx + dx, 0, size - 1);
                        int py = Math.Clamp(cy + dy, 0, size - 1);
                        mask[py, px] = 1.0f;
                        heights[py, px] = avgH;
                    }
                }
                any = true;
            }
        }

        if (any)
        {
            pack.WlLiquidMask = mask;
            pack.WlLiquidHeight = heights;
            pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "wl_liquid_mask", "wl_liquid_height" };

            // Rebuild unified liquid: MH2O > MCLQ > WL*
            if (pack.UnifiedLiquidMask is null && pack.Mh2oSurfaceHeight is null && pack.MclqSurfaceHeight is null)
            {
                pack.UnifiedLiquidMask = mask;
                pack.UnifiedLiquidHeight = heights;
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "unified_liquid_mask", "unified_liquid_height" };
            }
        }
    }

    static bool TryLoadMinimap(string adtPath, string minimapRoot, out byte[,,]? minimap)
    {
        minimap = null;
        string stem = Path.GetFileNameWithoutExtension(adtPath);

        string[] candidates =
        [
            Path.Combine(minimapRoot, $"{stem}.png"),
            Path.Combine(minimapRoot, "images", $"{stem}.png"),
            Path.Combine(minimapRoot, "reference_minimaps", $"{stem}_reference_minimap.png"),
        ];

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                try
                {
                    using var img = SixLabors.ImageSharp.Image.Load<Rgba32>(candidate);
                    if (img.Width == 256 && img.Height == 256)
                    {
                        minimap = new byte[256, 256, 3];
                        for (int y = 0; y < 256; y++)
                        {
                            for (int x = 0; x < 256; x++)
                            {
                                var px = img[x, y];
                                minimap[y, x, 0] = px.R;
                                minimap[y, x, 1] = px.G;
                                minimap[y, x, 2] = px.B;
                            }
                        }
                        return true;
                    }
                }
                catch { }
            }
        }

        return false;
    }

    static void LoadMd5Translate(string clientRoot, NativeMpqService catalog)
    {
        if (Md5TranslateResolver.TryLoad(
            new[] { clientRoot },
            path => catalog.FileExists(path),
            path => catalog.ReadFile(path),
            out var md5Index))
        {
            _md5Lookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in md5Index.PlainToHash)
                _md5Lookup[kv.Key] = kv.Value;
            Console.Error.WriteLine($"  Loaded {_md5Lookup.Count} md5translate entries");
        }
    }

    static byte[,,]? TryLoadMinimapFromMpq(NativeMpqService catalog, string mapName, int tileX, int tileY)
    {
        string mapLower = mapName.ToLowerInvariant();
        string x2 = tileX.ToString("00");
        string y2 = tileY.ToString("00");
        string canonicalPath = $"textures/minimap/{mapLower}/map{x2}_{y2}.blp";

        // MD5 translate lookup
        if (_md5Lookup is not null && _md5Lookup.TryGetValue(canonicalPath, out string? md5Name))
        {
            byte[]? blpBytes = catalog.ReadFile(md5Name);
            if (blpBytes is not null && blpBytes.Length >= 8)
            {
                try
                {
                    byte[,,]? rgb = DecodeBlpToRgb(blpBytes);
                    if (rgb is not null) return rgb;
                }
                catch { }
            }
        }

        // Fallback candidates matching WoWViewer EnumerateTileCandidates
        string[] candidates =
        [
            canonicalPath.Replace('/', '\\'),                                        // textures\minimap\azeroth\map32_32.blp
            $"textures\\minimap\\{mapLower}\\map{x2}_{y2}.blp",                    // textures\minimap\azeroth\map32_32.blp
            $"textures\\Minimap\\{mapLower}\\map{x2}_{y2}.blp",                    // textures\Minimap\azeroth\map32_32.blp
            $"world\\minimaps\\{mapLower}\\map{x2}_{y2}.blp",                      // world\minimaps\azeroth\map32_32.blp
            $"world\\Minimaps\\{mapName}\\map{x2}_{y2}.blp",                       // world\Minimaps\Azeroth\map32_32.blp
            $"{mapLower}\\map{x2}_{y2}.blp",                                        // azeroth\map32_32.blp
        ];

        foreach (string candidate in candidates)
        {
            byte[]? blpBytes = catalog.ReadFile(candidate);
            if (blpBytes is null || blpBytes.Length < 8) continue;

            try
            {
                byte[,,]? rgb = DecodeBlpToRgb(blpBytes);
                if (rgb is not null) return rgb;
            }
            catch { }
        }

        return null;
    }

    static byte[,,]? LoadTextureFromMpq(NativeMpqService catalog, string virtualPath)
    {
        byte[]? blpBytes = catalog.ReadFile(virtualPath);
        if (blpBytes is null || blpBytes.Length < 8)
            return null;
        try { return DecodeBlpToRgbNative(blpBytes); }
        catch { return null; }
    }

    static byte[,,]? DecodeBlpToRgbNative(byte[] blpBytes)
    {
        using var ms = new MemoryStream(blpBytes, writable: false);
        using var blp = new SereniaBLPLib.BlpFile(ms);
        using Image<Rgba32>? image = blp.GetImage(0);
        if (image == null) return null;

        int w = image.Width;
        int h = image.Height;
        if (w < 1 || h < 1) return null;

        var rgb = new byte[h, w, 3];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Rgba32 px = image[x, y];
                rgb[y, x, 0] = px.R;
                rgb[y, x, 1] = px.G;
                rgb[y, x, 2] = px.B;
            }
        }
        return rgb;
    }

    static byte[,,]? DecodeBlpToRgb(byte[] blpBytes)
    {
        using var ms = new MemoryStream(blpBytes, writable: false);
        using var blp = new SereniaBLPLib.BlpFile(ms);
        using Image<Rgba32>? image = blp.GetImage(0);
        if (image == null) return null;

        int w = image.Width;
        int h = image.Height;
        if (w < 1 || h < 1) return null;

        var rgb = new byte[256, 256, 3];
        float scaleX = (float)(w - 1) / 255f;
        float scaleY = (float)(h - 1) / 255f;

        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                int sx = Math.Clamp((int)(x * scaleX + 0.5f), 0, w - 1);
                int sy = Math.Clamp((int)(y * scaleY + 0.5f), 0, h - 1);
                Rgba32 px = image[sx, sy];
                rgb[y, x, 0] = px.R;
                rgb[y, x, 1] = px.G;
                rgb[y, x, 2] = px.B;
            }
        }

        return rgb;
    }

    static string? DetectBuildVersionFromClientRoot(string clientRoot)
    {
        string dirName = Path.GetFileName(clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(dirName))
            return null;

        // If the last directory is a generic game root (not a build key), walk up.
        if (dirName.Equals("World of Warcraft", StringComparison.OrdinalIgnoreCase)
            || dirName.Equals("Data", StringComparison.OrdinalIgnoreCase))
        {
            string? parent = Path.GetDirectoryName(clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (string.IsNullOrWhiteSpace(parent))
                return null;
            dirName = Path.GetFileName(parent);
        }

        string versionString = dirName.Replace('_', '.');
        return ClientBuildKey.TryParse(versionString, out _) ? versionString : null;
    }

    static string ResolveGameClientRoot(string clientRoot)
    {
        string normalizedRoot = clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string dataDir = Path.Combine(normalizedRoot, "Data");
        if (Directory.Exists(dataDir))
            return normalizedRoot;

        string nestedGameRoot = Path.Combine(normalizedRoot, "World of Warcraft");
        if (Directory.Exists(Path.Combine(nestedGameRoot, "Data")))
            return nestedGameRoot;

        return normalizedRoot;
    }

    static string? GetOption(string[] args, string name, string shortName)
    {
        int idx = Array.IndexOf(args, name);
        if (idx >= 0 && idx + 1 < args.Length) return args[idx + 1];
        if (!string.IsNullOrEmpty(shortName))
        {
            idx = Array.IndexOf(args, shortName);
            if (idx >= 0 && idx + 1 < args.Length) return args[idx + 1];
        }
        return null;
    }

    static int? GetIntOption(string[] args, string name, string shortName)
    {
        string? val = GetOption(args, name, shortName);
        return int.TryParse(val, out int v) ? v : null;
    }

    static bool HasFlag(string[] args, string flag)
    {
        return args.Contains(flag, StringComparer.OrdinalIgnoreCase);
    }
}
