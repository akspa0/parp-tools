using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tools.Harvest;

static class Program
{
    static int Main(string[] args)
    {
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
            default:
                Console.Error.WriteLine($"Unknown command '{command}'.");
                ShowUsage();
                return 1;
        }

        return 0;
    }

    static void ShowUsage()
    {
        Console.WriteLine("""
            WowViewer.Tool.Harvest — V14 dataset generation tool

            Usage: WowViewer.Tool.Harvest <command> [options]

            Commands:
              harvest-tile      Extract NPZ shard from a single ADT tile
              harvest-map       Batch-extract all tiles from a map directory
              synthetic-minimap Composite tilesets + alpha → synthetic minimap

            Global options:
              --build, -b       Client build version (e.g. "4.3.4.15595") for
                               version-aware ADT profile selection. Auto-detected
                               from input path if not specified.

            See --help on each command for options.
            """);
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

                pack = AlphaTensorPackBuilder.Build(tileData);
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

            string? textureSource = File.Exists($"{inputDir}\\{stem}_tex0.adt")
                ? $"{inputDir}\\{stem}_tex0.adt"
                : null;

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