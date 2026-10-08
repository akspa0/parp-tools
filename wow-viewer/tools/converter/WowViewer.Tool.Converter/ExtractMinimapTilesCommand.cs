using System;
using System.IO;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.Converter;

internal static class ExtractMinimapTilesCommand
{
    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string map = GetOption(args, "--map", "-m") ?? "Northrend";
        string? outputDir = GetOption(args, "--output-dir", "-o");

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: extract-minimap-tiles requires --client-root <dir> and --output-dir <dir>.");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputDir);
        using NativeMpqService catalog = new();
        catalog.LoadArchives([clientRoot]);

        byte[]? wdtBytes = catalog.ReadFile($"World\\Maps\\{map}\\{map}.wdt");
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read World\\Maps\\{map}\\{map}.wdt");
            return;
        }

        using MemoryStream wdtStream = new(wdtBytes, writable: false);
        MapFileSummary wdtSummary = MapFileSummaryReader.Read(wdtStream, $"World\\Maps\\{map}\\{map}.wdt");
        var occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(wdtStream, wdtSummary);
        Console.WriteLine($"Northrend.wdt has {occupiedTiles.Count} occupied tiles.");

        Md5TranslateResolver.TryLoad(
            [clientRoot],
            catalog.FileExists,
            catalog.ReadFile,
            out Md5TranslateIndex? md5Index);

        Console.WriteLine($"md5translate loaded: {md5Index != null} ({md5Index?.PlainToHash.Count ?? 0} entries)");
        if (md5Index is not null)
        {
            var sampleNorthrend = md5Index.PlainToHash
                .Where(kvp => kvp.Key.Contains("northrend", StringComparison.OrdinalIgnoreCase))
                .Take(5)
                .ToList();
            Console.WriteLine($"Sample Northrend md5 entries ({sampleNorthrend.Count}):");
            foreach (var kvp in sampleNorthrend)
            {
                Console.WriteLine($"  Plain: '{kvp.Key}' -> Hash: '{kvp.Value}'");
            }
        }

        int extracted = 0;

        foreach (var tile in occupiedTiles)
        {
            int x = tile.TileX;
            int y = tile.TileY;

            string[] plainCandidates =
            [
                $"Textures/Minimap/{map}/map{y:D2}_{x:D2}.blp",
                $"Textures/Minimap/{map}/map{x:D2}_{y:D2}.blp",
                $"Textures/Minimap/{map.ToLowerInvariant()}/map{y:D2}_{x:D2}.blp",
                $"Textures/Minimap/{map.ToLowerInvariant()}/map{x:D2}_{y:D2}.blp",
                $"Textures/Minimap/{map}/map{y}_{x}.blp",
                $"Textures/Minimap/{map}/map{x}_{y}.blp",
                $"{map}/map{y:D2}_{x:D2}.blp",
                $"{map.ToLowerInvariant()}/map{y:D2}_{x:D2}.blp",
            ];

            byte[]? blpBytes = null;
            string? resolvedPath = null;

            if (md5Index != null)
            {
                foreach (string plain in plainCandidates)
                {
                    foreach (string hash in md5Index.GetHashCandidates(plain))
                    {
                        string[] hashPaths =
                        [
                            hash,
                            $"Textures\\Minimap\\{hash}",
                            $"Textures\\Minimap\\{hash}.blp",
                            $"{hash}.blp",
                        ];

                        foreach (string hp in hashPaths)
                        {
                            if (catalog.FileExists(hp))
                            {
                                blpBytes = catalog.ReadFile(hp);
                                if (blpBytes is { Length: > 0 })
                                {
                                    resolvedPath = hp;
                                    break;
                                }
                            }
                        }
                        if (blpBytes != null) break;
                    }
                    if (blpBytes != null) break;
                }
            }

            if (blpBytes == null)
            {
                foreach (string cand in plainCandidates)
                {
                    string diskCand = cand.Replace('/', '\\');
                    if (catalog.FileExists(diskCand))
                    {
                        blpBytes = catalog.ReadFile(diskCand);
                        if (blpBytes is { Length: > 0 })
                        {
                            resolvedPath = diskCand;
                            break;
                        }
                    }
                }
            }

            if (blpBytes is null)
                continue;

            try
            {
                using MemoryStream stream = new(blpBytes, writable: false);
                using BlpFile blp = new(stream);
                using Image<Rgba32> image = blp.GetImage(0);

                if (image.Width != 256 || image.Height != 256)
                {
                    image.Mutate(ctx => ctx.Resize(256, 256));
                }

                string outPath = Path.Combine(outputDir, $"{map}_{x}_{y}.png");
                image.SaveAsPng(outPath);
                extracted++;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[WARN] Failed decoding tile ({x}, {y}): {ex.Message}");
            }
        }

        Console.WriteLine($"Extracted {extracted} minimap tile(s) for map '{map}'.");
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (string.Equals(args[i], longName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[i], shortName, StringComparison.OrdinalIgnoreCase))
            {
                if (i + 1 < args.Length && !args[i + 1].StartsWith('-'))
                    return args[i + 1];
            }
        }
        return null;
    }
}
