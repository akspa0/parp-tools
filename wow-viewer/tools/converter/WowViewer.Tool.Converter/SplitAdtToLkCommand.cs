using System.Diagnostics;
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

            string mapName = options.MapName.Trim();
            string wdtVirtualPath = $"World\\Maps\\{mapName}\\{mapName}.wdt";

            Console.WriteLine("WowViewer.Tool.Converter convert-split-adt-to-lk report");
            Console.WriteLine($"  Client:   {clientRoot}");
            Console.WriteLine($"  Map:      {mapName}");
            Console.WriteLine($"  Overlay:  {overlayRoot ?? "<none>"}");
            Console.WriteLine($"  Output:   {outputDir}");
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

            int? limit = GetIntOption(args, "--limit", "-n");
            var sw = Stopwatch.StartNew();
            var warnings = new List<string>();
            var emittedTiles = new HashSet<(int tileX, int tileY)>();
            int converted = 0;
            int failed = 0;

            foreach (WdtTileCoordinate tile in occupiedTiles.OrderBy(static tile => tile.TileY * 64 + tile.TileX))
            {
                string adtVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tile.TileX}_{tile.TileY}.adt";
                string texVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tile.TileX}_{tile.TileY}_tex0.adt";
                string objVirtualPath = $"World\\Maps\\{mapName}\\{mapName}_{tile.TileX}_{tile.TileY}_obj0.adt";

                if (!TryReadVirtualOrLooseFile(adtVirtualPath, overlayRoot, catalog, out byte[]? adtBytes, out string adtSourcePath) || adtBytes is null)
                {
                    failed++;
                    warnings.Add($"Tile ({tile.TileX},{tile.TileY}): missing root ADT '{adtVirtualPath}'.");
                    continue;
                }

                TryReadVirtualOrLooseFile(texVirtualPath, overlayRoot, catalog, out byte[]? tex0Bytes, out _);
                TryReadVirtualOrLooseFile(objVirtualPath, overlayRoot, catalog, out byte[]? obj0Bytes, out _);

                try
                {
                    LkAdtData adtData = LkAdtReader.Read(adtBytes, tex0Bytes, obj0Bytes, tile.TileX, tile.TileY);
                    byte[] monolithicBytes = LkAdtWriter.Build(adtData);
                    string outputPath = Path.Combine(outputDir, $"{mapName}_{tile.TileX}_{tile.TileY}.adt");
                    File.WriteAllBytes(outputPath, monolithicBytes);
                    emittedTiles.Add((tile.TileX, tile.TileY));
                    converted++;

                    if (options.Verbose)
                        Console.WriteLine($"  Converted: {mapName}_{tile.TileX}_{tile.TileY}.adt ({monolithicBytes.Length:N0} bytes) <- {adtSourcePath}");

                    if (limit.HasValue && converted >= limit.Value)
                        break;
                }
                catch (Exception ex)
                {
                    failed++;
                    warnings.Add($"Tile ({tile.TileX},{tile.TileY}): {ex.Message}");
                    if (options.Verbose)
                        Console.Error.WriteLine($"  Error converting ({tile.TileX},{tile.TileY}): {ex}");
                }
            }

            string wdtOutputPath = Path.Combine(outputDir, $"{mapName}.wdt");
            File.WriteAllBytes(wdtOutputPath, LkWdtWriter.Build(emittedTiles));

            sw.Stop();
            Console.WriteLine($"  WDT:      {wdtSourcePath}");
            Console.WriteLine($"  Tiles:    {occupiedTiles.Count}");
            Console.WriteLine($"  Converted:{converted}");
            Console.WriteLine($"  Failed:   {failed}");
            Console.WriteLine($"  Output:   {outputDir}");
            Console.WriteLine($"  Wrote:    {wdtOutputPath}");
            Console.WriteLine($"  Elapsed:  {sw.ElapsedMilliseconds}ms");

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
            OutputDir: GetOption(args, "--output-dir", "-o"),
            Verbose: HasFlag(args, "--verbose") || HasFlag(args, "-v"));
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
        string? OutputDir,
        bool Verbose);
}