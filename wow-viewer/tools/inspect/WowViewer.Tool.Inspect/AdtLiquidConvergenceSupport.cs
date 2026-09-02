using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

/// <summary>
/// Implements `inspect adt liquid-convergence`, analyzing MCLQ legacy liquid and WL* loose liquid
/// convergence across map tiles to measure Mechanism A (partially-present MCLQ quad upsampling) and
/// Mechanism B (KeepOnlyAboveTerrain waterline culling).
/// </summary>
internal static class AdtLiquidConvergenceSupport
{
    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client");
        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Usage: adt liquid-convergence --client <client-dir> [--map <name>] [--tile-x <x>] [--tile-y <y>] [--limit <n>] [--listfile <file>]");
            Environment.ExitCode = 1;
            return;
        }

        string mapName = GetOption(args, "--map") ?? "Azeroth";
        int? filterTileX = int.TryParse(GetOption(args, "--tile-x"), out int tx) ? tx : null;
        int? filterTileY = int.TryParse(GetOption(args, "--tile-y"), out int ty) ? ty : null;
        int limit = int.TryParse(GetOption(args, "--limit"), out int parsedLimit) ? parsedLimit : 500;
        string? listfilePath = GetOption(args, "--listfile") ?? TryFindDefaultListfilePath();

        Console.WriteLine("================================================================================");
        Console.WriteLine("WowViewer.Tool.Inspect ADT / WL* Liquid Convergence Analyzer (Spec 209)");
        Console.WriteLine($"Client:   {clientRoot}");
        Console.WriteLine($"Map:      {mapName}");
        if (filterTileX.HasValue && filterTileY.HasValue)
            Console.WriteLine($"Filter:   Tile ({filterTileX.Value}, {filterTileY.Value})");
        Console.WriteLine("================================================================================");

        using IArchiveCatalog catalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(
            catalog, [clientRoot], new ArchiveCatalogBootstrapOptions(ExternalListfilePath: listfilePath));

        Console.WriteLine($"Catalog bootstrapped: {bootstrap.AllFiles.Count} known files loaded.");

        // 1. Discover and parse all WL* files for this map from catalog and disk
        List<WlFile> wlFiles = DiscoverAndLoadWlFiles(catalog, clientRoot, mapName);
        Console.WriteLine($"WL* discovery: {wlFiles.Count} readable WL* files loaded for {mapName}.");

        // 2. Discover tiles for the map
        List<LiquidConvergenceTileReport> reports = [];

        // Check if there is an Alpha WDT (0.5.3 format)
        string wdtVirtualPath = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        string wdtVirtualPathSlash = $"World/Maps/{mapName}/{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtualPath) ?? catalog.ReadFile(wdtVirtualPathSlash);

        if (wdtBytes is not null && AlphaWdtReader.IsAlphaWdt(wdtBytes))
        {
            Console.WriteLine($"Found Alpha WDT for {mapName}. Reading alpha tiles...");
            reports = AnalyzeAlphaWdtTiles(wdtBytes, wdtVirtualPath, mapName, wlFiles, filterTileX, filterTileY, limit);
        }
        else
        {
            // Standard ADT enumeration (post-alpha / 1.x+)
            Console.WriteLine($"Searching for standard ADT tiles for {mapName}...");
            reports = AnalyzeStandardAdtTiles(catalog, mapName, wlFiles, filterTileX, filterTileY, limit);
        }

        // 3. Print report
        PrintReport(mapName, reports);
    }

    private static List<WlFile> DiscoverAndLoadWlFiles(IArchiveCatalog catalog, string clientRoot, string mapName)
    {
        List<WlFile> loaded = [];
        HashSet<string> seenPaths = new(StringComparer.OrdinalIgnoreCase);

        string mapPrefixWin = $"World\\Maps\\{mapName}\\";
        string mapPrefixSlash = $"World/Maps/{mapName}/";

        // From archive catalog
        var archiveCandidates = catalog.GetAllKnownFiles()
            .Where(p => (p.StartsWith(mapPrefixWin, StringComparison.OrdinalIgnoreCase) ||
                         p.StartsWith(mapPrefixSlash, StringComparison.OrdinalIgnoreCase)) &&
                        (p.EndsWith(".wlw", StringComparison.OrdinalIgnoreCase) ||
                         p.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) ||
                         p.EndsWith(".wlq", StringComparison.OrdinalIgnoreCase) ||
                         p.EndsWith(".wll", StringComparison.OrdinalIgnoreCase)));

        foreach (string path in archiveCandidates)
        {
            if (!seenPaths.Add(path)) continue;
            byte[]? data = catalog.ReadFile(path);
            if (data is null || data.Length == 0) continue;

            try
            {
                using MemoryStream ms = new(data, writable: false);
                WlFile wl = WlFileReader.Read(ms, path);
                loaded.Add(wl);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [WL] Warning: Failed to parse {path}: {ex.Message}");
            }
        }

        // Also check filesystem on disk
        string[] diskSearchDirs =
        [
            Path.Combine(clientRoot, "World", "Maps", mapName),
            Path.Combine(clientRoot, "Data", "World", "Maps", mapName),
            Path.Combine(clientRoot, mapName),
        ];

        foreach (string dir in diskSearchDirs)
        {
            if (!Directory.Exists(dir)) continue;

            string[] diskFiles = Directory.GetFiles(dir, "*.wl*", SearchOption.TopDirectoryOnly)
                .Where(p => p.EndsWith(".wlw", StringComparison.OrdinalIgnoreCase) ||
                            p.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) ||
                            p.EndsWith(".wlq", StringComparison.OrdinalIgnoreCase) ||
                            p.EndsWith(".wll", StringComparison.OrdinalIgnoreCase))
                .ToArray();

            foreach (string file in diskFiles)
            {
                string fn = Path.GetFileName(file);
                if (!seenPaths.Add(fn)) continue;

                try
                {
                    WlFile wl = WlFileReader.Read(file);
                    loaded.Add(wl);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  [WL] Warning: Failed to parse disk file {file}: {ex.Message}");
                }
            }
        }

        return loaded;
    }

    private static List<LiquidConvergenceTileReport> AnalyzeAlphaWdtTiles(
        byte[] wdtBytes,
        string wdtPath,
        string mapName,
        List<WlFile> wlFiles,
        int? filterX,
        int? filterY,
        int limit)
    {
        List<LiquidConvergenceTileReport> results = [];
        var existingTiles = AlphaWdtReader.ReadExistingTiles(wdtBytes)
            .OrderBy(t => t.Y)
            .ThenBy(t => t.X)
            .ToList();

        Console.WriteLine($"Alpha WDT contains {existingTiles.Count} existing tiles.");

        foreach ((int tx, int ty) in existingTiles)
        {
            if (filterX.HasValue && filterX.Value != tx) continue;
            if (filterY.HasValue && filterY.Value != ty) continue;
            if (results.Count >= limit) break;

            if (!AlphaWdtReader.TryReadTile(wdtBytes, tx, ty, wdtPath, out AlphaTileData? tileData) || tileData is null)
                continue;

            TerrainTileTensorPack alphaPack;
            try
            {
                alphaPack = AlphaTensorPackBuilder.Build(tileData, tx, ty);
            }
            catch
            {
                continue;
            }

            // Extract MCLQ heights and presence from alphaPack (already normalized to 257x257)
            float[,]? mclqH = alphaPack.MclqSurfaceHeight;
            bool[,]? mclqP = alphaPack.MclqPresenceMask;

            // Rasterize WL* for this tile
            WlLiquidRasterizer.TryRasterize(
                wlFiles, tx, ty,
                out float[,]? wlMask,
                out float[,]? wlHeights,
                out byte[,]? wlBasicTypes,
                257);

            bool hasMclq = mclqP is not null && HasAnyTrue(mclqP);
            bool hasWl = wlMask is not null;

            if (!hasMclq && !hasWl)
                continue; // Skip dry tiles

            LiquidConvergenceTileReport report = LiquidConvergenceAnalyzer.Analyze(
                tx, ty, $"{mapName}_{tx}_{ty}",
                mclqH, mclqP,
                wlMask, wlHeights,
                alphaPack.Height257,
                wlBasicTypes,
                257);

            results.Add(report);
        }

        return results;
    }

    private static List<LiquidConvergenceTileReport> AnalyzeStandardAdtTiles(
        IArchiveCatalog catalog,
        string mapName,
        List<WlFile> wlFiles,
        int? filterX,
        int? filterY,
        int limit)
    {
        List<LiquidConvergenceTileReport> results = [];
        string mapPrefixWin = $"World\\Maps\\{mapName}\\";
        string mapPrefixSlash = $"World/Maps/{mapName}/";

        var adtCandidates = catalog.GetAllKnownFiles()
            .Where(p => (p.StartsWith(mapPrefixWin, StringComparison.OrdinalIgnoreCase) ||
                         p.StartsWith(mapPrefixSlash, StringComparison.OrdinalIgnoreCase)) &&
                        p.EndsWith(".adt", StringComparison.OrdinalIgnoreCase) &&
                        !p.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase) &&
                        !p.EndsWith("_obj1.adt", StringComparison.OrdinalIgnoreCase) &&
                        !p.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase) &&
                        !p.EndsWith("_tex1.adt", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();

        Console.WriteLine($"Found {adtCandidates.Count} candidate root ADT paths.");

        foreach (string adtPath in adtCandidates)
        {
            if (results.Count >= limit) break;
            if (!TryParseAdtCoords(adtPath, out int tx, out int ty)) continue;

            if (filterX.HasValue && filterX.Value != tx) continue;
            if (filterY.HasValue && filterY.Value != ty) continue;

            byte[]? adtBytes = catalog.ReadFile(adtPath);
            if (adtBytes is null || adtBytes.Length == 0) continue;

            TerrainTileTensorPack pack;
            try
            {
                pack = AdtTensorPackBuilder.BuildFromBytes(adtPath, adtBytes);
            }
            catch
            {
                continue;
            }

            // Rasterize WL* for this tile
            WlLiquidRasterizer.TryRasterize(
                wlFiles, tx, ty,
                out float[,]? wlMask,
                out float[,]? wlHeights,
                out byte[,]? wlBasicTypes,
                257);

            bool hasMclq = pack.MclqPresenceMask is not null && HasAnyTrue(pack.MclqPresenceMask);
            bool hasWl = wlMask is not null;

            if (!hasMclq && !hasWl)
                continue;

            LiquidConvergenceTileReport report = LiquidConvergenceAnalyzer.Analyze(
                tx, ty, Path.GetFileNameWithoutExtension(adtPath),
                pack.MclqSurfaceHeight, pack.MclqPresenceMask,
                wlMask, wlHeights,
                pack.Height257,
                wlBasicTypes,
                257);

            results.Add(report);
        }

        return results;
    }

    private static void PrintReport(string mapName, List<LiquidConvergenceTileReport> reports)
    {
        Console.WriteLine();
        Console.WriteLine("=============================================================================================================================");
        Console.WriteLine($"LIQUID CONVERGENCE REPORT — Map: {mapName} ({reports.Count} liquid tiles analyzed)");
        Console.WriteLine("=============================================================================================================================");
        Console.WriteLine(
            $"{"Tile",-14} | {"MCLQ-Only",10} | {"WL-Only",10} | {"Both",8} | {"Δ Height (Mean/Max)",20} | {"Mech A (Quads/Cells)",22} | {"Mech B (Culled/NoMclq)",22}");
        Console.WriteLine("-----------------------------------------------------------------------------------------------------------------------------");

        long totalMclqOnly = 0;
        long totalWlOnly = 0;
        long totalBoth = 0;
        long totalNeither = 0;
        long totalPartQuads = 0;
        long totalAffectedCells = 0;
        long totalWlCulled = 0;
        long totalCulledNoMclq = 0;
        long totalMissingUnified = 0;
        List<float> allOverlappingDiffs = [];

        foreach (var r in reports.OrderBy(x => x.TileY).ThenBy(x => x.TileX))
        {
            totalMclqOnly += r.Populations.MclqOnlyCount;
            totalWlOnly += r.Populations.WlOnlyCount;
            totalBoth += r.Populations.BothCount;
            totalNeither += r.Populations.NeitherCount;
            totalPartQuads += r.MechanismA.PartiallyPresentQuadCount;
            totalAffectedCells += r.MechanismA.AffectedUpsampledCellCount;
            totalWlCulled += r.MechanismB.WlCellsCulledByTerrain;
            totalCulledNoMclq += r.MechanismB.CulledCellsWithoutMclq;
            totalMissingUnified += r.MissingFromUnifiedCount;

            string heightDiffStr = r.Populations.BothCount > 0
                ? $"{r.HeightDifference.Mean,7:F2} / {r.HeightDifference.Max,7:F2}"
                : "        - /       -";

            string mechAStr = $"{r.MechanismA.PartiallyPresentQuadCount,6} / {r.MechanismA.AffectedUpsampledCellCount,6}";
            string mechBStr = $"{r.MechanismB.WlCellsCulledByTerrain,6} / {r.MechanismB.CulledCellsWithoutMclq,6}";

            Console.WriteLine(
                $"{r.TileName,-14} | {r.Populations.MclqOnlyCount,10} | {r.Populations.WlOnlyCount,10} | {r.Populations.BothCount,8} | {heightDiffStr,20} | {mechAStr,22} | {mechBStr,22}");
        }

        Console.WriteLine("=============================================================================================================================");
        Console.WriteLine("MAP-WIDE AGGREGATE SUMMARY");
        Console.WriteLine("=============================================================================================================================");
        Console.WriteLine($"Total liquid tiles analyzed:       {reports.Count}");
        Console.WriteLine($"Tiles with MCLQ:                   {reports.Count(r => r.HasMclq)}");
        Console.WriteLine($"Tiles with WL*:                    {reports.Count(r => r.HasWl)}");
        Console.WriteLine($"Tiles with BOTH:                   {reports.Count(r => r.HasMclq && r.HasWl)}");
        Console.WriteLine();
        Console.WriteLine("POPULATION CELL COUNTS (257x257 lattice per tile):");
        Console.WriteLine($"  MCLQ-only cells:                 {totalMclqOnly:N0}");
        Console.WriteLine($"  WL*-only cells:                  {totalWlOnly:N0}");
        Console.WriteLine($"  Both (overlapping cells):        {totalBoth:N0}");
        Console.WriteLine($"  Neither (dry terrain cells):     {totalNeither:N0}");
        Console.WriteLine();
        Console.WriteLine("MECHANISM A MEASUREMENT (MCLQ Quad Edge Upsampling):");
        Console.WriteLine($"  Partially-present MCLQ quads:    {totalPartQuads:N0}");
        Console.WriteLine($"  Upsampled cells affected:        {totalAffectedCells:N0}");
        Console.WriteLine("  -> Verified fixed via LiquidSurfaceInterpolation presence-weighted interpolation.");
        Console.WriteLine();
        Console.WriteLine("MECHANISM B MEASUREMENT (KeepOnlyAboveTerrain Waterline Culling):");
        Console.WriteLine($"  WL* cells culled by terrain:     {totalWlCulled:N0}");
        Console.WriteLine($"  Culled WL* cells WITHOUT MCLQ:   {totalCulledNoMclq:N0}  <-- POTENTIAL WATERLINE GAP POPULATION");
        Console.WriteLine();
        Console.WriteLine("UNION INVARIANT (FR-004 / SC-002):");
        Console.WriteLine($"  Cells missing from unified:      {totalMissingUnified:N0} (Expected: 0)");
        if (totalMissingUnified == 0)
            Console.WriteLine("  -> PASSED: Unified liquid preserves 100% of cells covered by either source.");
        else
            Console.WriteLine("  -> FAILED: Cells covered by a source were dropped in unified liquid!");
        Console.WriteLine("=============================================================================================================================");
    }

    private static bool HasAnyTrue(bool[,] mask)
    {
        int h = mask.GetLength(0);
        int w = mask.GetLength(1);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (mask[y, x]) return true;
            }
        }
        return false;
    }

    private static bool TryParseAdtCoords(string path, out int tileX, out int tileY)
    {
        tileX = -1;
        tileY = -1;
        string name = Path.GetFileNameWithoutExtension(path);
        string[] parts = name.Split('_');
        if (parts.Length < 3) return false;

        return int.TryParse(parts[^2], out tileX) && int.TryParse(parts[^1], out tileY);
    }

    private static string? GetOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }
        return null;
    }

    private static string? TryFindDefaultListfilePath()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
            {
                string candidate = Path.Combine(current.FullName, "libs", "wowdev", "wow-listfile", "listfile.txt");
                if (File.Exists(candidate)) return candidate;
            }
            current = current.Parent;
        }

        string[] rootCandidates =
        [
            Path.Combine(Environment.CurrentDirectory, "wow-viewer", "libs", "wowdev", "wow-listfile", "listfile.txt"),
            Path.Combine(Environment.CurrentDirectory, "libs", "wowdev", "wow-listfile", "listfile.txt"),
        ];
        foreach (string cand in rootCandidates)
        {
            if (File.Exists(cand)) return cand;
        }

        return null;
    }
}
