using System.Diagnostics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.Converter;

internal static class AlphaToLkCommand
{
    public static void Run(string[] args)
    {
        try
        {
            AlphaToLkOptions options = ParseOptions(args);

            if (string.IsNullOrEmpty(options.InputPath))
            {
                Console.Error.WriteLine("Error: --input <wdt-path> is required.");
                Environment.ExitCode = 1;
                return;
            }

            if (string.IsNullOrEmpty(options.OutputDir))
            {
                Console.Error.WriteLine("Error: --output <dir> is required.");
                Environment.ExitCode = 1;
                return;
            }

            string wdtPath = Path.GetFullPath(options.InputPath);
            string outputDir = Path.GetFullPath(options.OutputDir);

            if (!File.Exists(wdtPath))
            {
                Console.Error.WriteLine($"Error: WDT file not found: {wdtPath}");
                Environment.ExitCode = 1;
                return;
            }

            Console.WriteLine("WowViewer.Tool.Converter convert-alpha-to-lk report");
            Console.WriteLine($"  Input:    {wdtPath}");
            Console.WriteLine($"  Output:   {outputDir}");
            Console.WriteLine($"  Verbose:  {options.Verbose}");

            var sw = Stopwatch.StartNew();

            byte[] wdtData = File.ReadAllBytes(wdtPath);

            if (!AlphaWdtReader.IsAlphaWdt(wdtData))
            {
                Console.Error.WriteLine($"Error: File is not an Alpha WDT: {wdtPath}");
                Console.Error.WriteLine("  The MAIN chunk size does not match Alpha format (expected 65536 bytes).");
                Environment.ExitCode = 1;
                return;
            }

            string mapName = Path.GetFileNameWithoutExtension(wdtPath);

            var existingTiles = AlphaWdtReader.ReadExistingTiles(wdtData);

            Console.WriteLine($"  Map:      {mapName}");
            Console.WriteLine($"  Tiles:    {existingTiles.Count}");

            Directory.CreateDirectory(outputDir);

            byte[] wdtLk = LkWdtWriter.Build(existingTiles);
            string wdtOutPath = Path.Combine(outputDir, $"{mapName}.wdt");
            File.WriteAllBytes(wdtOutPath, wdtLk);
            if (options.Verbose)
                Console.WriteLine($"  Wrote WDT: {wdtOutPath}");

            var wdlTiles = new List<WdlHeightTile>();
            int converted = 0;
            int failed = 0;
            var warnings = new List<string>();

            foreach (var (tileX, tileY) in existingTiles.OrderBy(t => t.Item2 * 64 + t.Item1))
            {
                if (!AlphaWdtReader.TryReadTile(wdtData, tileX, tileY, out AlphaTileData? tileData) || tileData == null)
                {
                    failed++;
                    warnings.Add($"Tile ({tileX},{tileY}): failed to read");
                    continue;
                }

                try
                {
                    LkAdtData adtData = AlphaToLkConverter.ConvertTile(tileData, tileX, tileY);
                    byte[] adtBytes = LkAdtWriter.Build(adtData);
                    string adtOutPath = Path.Combine(outputDir, $"{mapName}_{tileX}_{tileY}.adt");
                    File.WriteAllBytes(adtOutPath, adtBytes);

                    wdlTiles.Add(WdlWriter.ExtractTileHeightsFromAlpha(tileData.Heightmap, tileX, tileY));
                    converted++;

                    if (options.Verbose)
                        Console.WriteLine($"  Converted: {mapName}_{tileX}_{tileY}.adt ({adtBytes.Length:N0} bytes)");
                }
                catch (Exception ex)
                {
                    failed++;
                    warnings.Add($"Tile ({tileX},{tileY}): {ex.Message}");
                    if (options.Verbose)
                        Console.Error.WriteLine($"  Error converting ({tileX},{tileY}): {ex}");
                }
            }

            string wdlOutPath = Path.Combine(outputDir, $"{mapName}.wdl");
            byte[] wdlBytes = WdlWriter.Build(wdlTiles);
            File.WriteAllBytes(wdlOutPath, wdlBytes);
            if (options.Verbose)
                Console.WriteLine($"  Wrote WDL: {wdlOutPath} ({wdlBytes.Length:N0} bytes, {wdlTiles.Count} tiles)");

            long totalBytes = 0;
            foreach (var f in Directory.GetFiles(outputDir, $"{mapName}_*"))
                totalBytes += new FileInfo(f).Length;

            sw.Stop();
            Console.WriteLine($"  Converted: {converted}/{existingTiles.Count} tiles");
            Console.WriteLine($"  Failed:    {failed} tiles");
            Console.WriteLine($"  WDT:       {wdtOutPath}");
            Console.WriteLine($"  WDL:       {wdlOutPath}");
            Console.WriteLine($"  ADT bytes:  {totalBytes:N0}");
            Console.WriteLine($"  Elapsed:   {sw.ElapsedMilliseconds}ms");

            if (warnings.Count > 0)
            {
                Console.WriteLine($"  Warnings:  {warnings.Count}");
                foreach (var w in warnings.Take(10))
                    Console.WriteLine($"    {w}");
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

    private static AlphaToLkOptions ParseOptions(string[] args)
    {
        return new AlphaToLkOptions(
            InputPath: GetOption(args, "--input", "-i") ?? GetOption(args, "--wdt", "-w"),
            OutputDir: GetOption(args, "--output", "-o"),
            Verbose: HasFlag(args, "--verbose") || HasFlag(args, "-v"));
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

    private static bool HasFlag(string[] args, string name)
    {
        foreach (var arg in args)
        {
            if (string.Equals(arg, name, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    private readonly record struct AlphaToLkOptions(
        string? InputPath,
        string? OutputDir,
        bool Verbose);
}