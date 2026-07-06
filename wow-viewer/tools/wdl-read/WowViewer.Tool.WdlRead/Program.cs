using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tools.WdlRead;

/// <summary>
/// CLI shim that exposes the existing C# WDL surfaces to Python (Spec 094).
/// `read` wraps <see cref="WdlSummaryReader"/> (real staged-client WDLs, resolved
/// through <see cref="NativeMpqService"/>); `synth` wraps
/// <see cref="WdlWriter.ExtractTileHeightsFromAlpha"/> (terrain -> WDL lattice).
/// The core libraries are the source of truth and are not modified.
/// </summary>
static class Program
{
    private const string Version = "094.1";
    private const int OuterDim = 17;
    private const int InnerDim = 16;

    static int Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        if (args.Contains("--version"))
        {
            Console.WriteLine(Version);
            return 0;
        }

        try
        {
            return args[0].ToLowerInvariant() switch
            {
                "read" => RunRead(args.Skip(1).ToArray()),
                "synth" => RunSynth(args.Skip(1).ToArray()),
                _ => Fail($"Unknown command '{args[0]}'. Use --help."),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    static void ShowUsage()
    {
        Console.WriteLine("""
            WowViewer.Tool.WdlRead - Spec 094 WDL shim (wraps WowViewer.Core.IO; core is not modified)

            Usage:
              WowViewer.Tool.WdlRead read --client-root <staged-root> --map <name> [--tile-x N --tile-y N] --output <npz>
              WowViewer.Tool.WdlRead read --wdl <loose .wdl path> [--tile-x N --tile-y N] --output <npz>
              WowViewer.Tool.WdlRead synth --height <npz> [--liquid <npz>] --output <npz>
              WowViewer.Tool.WdlRead --help | --version

            read   Reads real WDL MARE tiles via WdlSummaryReader. Emits every present tile
                   (or one tile when --tile-x/--tile-y are given) as NPZ:
                     tile_xy (N,2) int32, outer (N,17,17) float32, inner (N,16,16) float32,
                     version (1,) int32 (-1 when the WDL has no MVER).
                   Exit codes: 0 ok, 2 no WDL found for the map, 3 requested tile not present.

            synth  Builds synthetic WDL grids from height data via WdlWriter.ExtractTileHeightsFromAlpha.
                   --height NPZ key 'height_257' (or 'height'): (257,257) or (N,257,257) float32.
                   --liquid NPZ key 'liquid_mask' (or 'liquid'): (256,256) or (N,256,256), >0.5 = liquid.
                   Lattice sample points that sit on liquid are re-sampled from the nearest
                   non-liquid pixel before extraction. Emits outer (N,17,17) / inner (N,16,16) float32.
            """);
    }

    static int Fail(string message)
    {
        Console.Error.WriteLine($"Error: {message}");
        return 1;
    }

    // ---------------------------------------------------------------- read

    static int RunRead(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root");
        string? map = GetOption(args, "--map");
        string? loosePath = GetOption(args, "--wdl");
        string? output = GetOption(args, "--output");
        int? tileX = GetIntOption(args, "--tile-x");
        int? tileY = GetIntOption(args, "--tile-y");

        if (string.IsNullOrWhiteSpace(output))
            return Fail("read requires --output <npz>.");
        if (tileX.HasValue != tileY.HasValue)
            return Fail("--tile-x and --tile-y must be given together.");

        WdlSummary summary;
        if (!string.IsNullOrWhiteSpace(loosePath))
        {
            if (!File.Exists(loosePath))
                return Fail($"WDL file not found: {loosePath}");
            summary = WdlSummaryReader.Read(loosePath);
        }
        else
        {
            if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(map))
                return Fail("read requires either --wdl <path> or --client-root <dir> plus --map <name>.");
            if (!Directory.Exists(clientRoot))
                return Fail($"Client root not found: {clientRoot}");

            byte[]? bytes = ReadWdlBytesFromClient(ResolveGameClientRoot(clientRoot), map, out string? resolvedPath);
            if (bytes is null || bytes.Length == 0)
            {
                Console.Error.WriteLine($"No WDL found for map '{map}' under '{clientRoot}'.");
                return 2;
            }

            Console.Error.WriteLine($"  Resolved WDL: {resolvedPath} ({bytes.Length} bytes)");
            using MemoryStream stream = new(bytes);
            summary = WdlSummaryReader.Read(stream, resolvedPath ?? map);
        }

        List<WdlTileSummary> tiles = summary.Tiles.ToList();
        if (tileX.HasValue && tileY.HasValue)
        {
            tiles = tiles.Where(tile => tile.TileX == tileX.Value && tile.TileY == tileY.Value).ToList();
            if (tiles.Count == 0)
            {
                Console.Error.WriteLine($"WDL has no MARE entry for tile ({tileX}, {tileY}).");
                return 3;
            }
        }

        int count = tiles.Count;
        int[] tileXy = new int[count * 2];
        float[] outer = new float[count * OuterDim * OuterDim];
        float[] inner = new float[count * InnerDim * InnerDim];
        for (int i = 0; i < count; i++)
        {
            WdlTileSummary tile = tiles[i];
            tileXy[i * 2] = tile.TileX;
            tileXy[(i * 2) + 1] = tile.TileY;
            for (int j = 0; j < OuterDim * OuterDim; j++)
                outer[(i * OuterDim * OuterDim) + j] = tile.OuterHeights[j];
            for (int j = 0; j < InnerDim * InnerDim; j++)
                inner[(i * InnerDim * InnerDim) + j] = tile.InnerHeights[j];
        }

        Npy.WriteNpz(output, [
            ("tile_xy", tileXy, [count, 2]),
            ("outer", outer, [count, OuterDim, OuterDim]),
            ("inner", inner, [count, InnerDim, InnerDim]),
            ("version", new int[] { unchecked((int)(summary.Version ?? uint.MaxValue)) }, [1]),
        ]);

        Console.WriteLine($"read: wrote {count} MARE tile(s) to {output}");
        return 0;
    }

    static byte[]? ReadWdlBytesFromClient(string gameRoot, string map, out string? resolvedPath)
    {
        resolvedPath = null;
        using NativeMpqService catalog = new();
        catalog.LoadArchives([gameRoot]);

        string[] candidates =
        [
            $"World\\Maps\\{map}\\{map}.wdl",
            $"World\\Maps\\{map}\\{map}.wdl.mpq",
        ];

        foreach (string candidate in candidates)
        {
            byte[]? bytes = catalog.ReadFile(candidate);
            if (bytes is { Length: > 0 })
            {
                resolvedPath = candidate;
                return bytes;
            }
        }

        // Era fallback: some alpha-format archives only expose the file by name.
        foreach (string known in catalog.GetAllKnownFiles())
        {
            if (!known.EndsWith(".wdl", StringComparison.OrdinalIgnoreCase))
                continue;
            if (!Path.GetFileNameWithoutExtension(known).Equals(map, StringComparison.OrdinalIgnoreCase))
                continue;

            byte[]? bytes = catalog.ReadFile(known);
            if (bytes is { Length: > 0 })
            {
                resolvedPath = known;
                return bytes;
            }
        }

        return null;
    }

    static string ResolveGameClientRoot(string clientRoot)
    {
        string normalized = clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (Directory.Exists(Path.Combine(normalized, "Data")))
            return normalized;

        string nested = Path.Combine(normalized, "World of Warcraft");
        if (Directory.Exists(Path.Combine(nested, "Data")))
            return nested;

        return normalized;
    }

    // ---------------------------------------------------------------- synth

    static int RunSynth(string[] args)
    {
        string? heightPath = GetOption(args, "--height");
        string? liquidPath = GetOption(args, "--liquid");
        string? output = GetOption(args, "--output");

        if (string.IsNullOrWhiteSpace(heightPath) || string.IsNullOrWhiteSpace(output))
            return Fail("synth requires --height <npz> and --output <npz>.");
        if (!File.Exists(heightPath))
            return Fail($"Height NPZ not found: {heightPath}");

        Npy.NpyArray heights = SelectArray(Npy.ReadNpz(heightPath), ["height_257", "height"], heightPath);
        Npy.NpyArray? liquid = null;
        if (!string.IsNullOrWhiteSpace(liquidPath))
        {
            if (!File.Exists(liquidPath))
                return Fail($"Liquid NPZ not found: {liquidPath}");
            liquid = SelectArray(Npy.ReadNpz(liquidPath), ["liquid_mask", "liquid"], liquidPath);
        }

        (int tileCount, int heightStride) = heights.Shape switch
        {
            [257, 257] => (1, 257 * 257),
            [var n, 257, 257] => (n, 257 * 257),
            _ => throw new InvalidDataException($"height array must be (257,257) or (N,257,257); got ({string.Join(",", heights.Shape)})."),
        };

        int liquidStride = 256 * 256;
        if (liquid is not null)
        {
            bool liquidOk = liquid.Shape switch
            {
                [256, 256] => tileCount == 1,
                [var n, 256, 256] => n == tileCount,
                _ => false,
            };
            if (!liquidOk)
                throw new InvalidDataException($"liquid array must be (256,256) or (N,256,256) matching the height stack; got ({string.Join(",", liquid.Shape)}).");
        }

        float[] outer = new float[tileCount * OuterDim * OuterDim];
        float[] inner = new float[tileCount * InnerDim * InnerDim];

        for (int t = 0; t < tileCount; t++)
        {
            float[,] tileHeights = new float[257, 257];
            for (int y = 0; y < 257; y++)
                for (int x = 0; x < 257; x++)
                    tileHeights[y, x] = heights.Data[(t * heightStride) + (y * 257) + x];

            if (liquid is not null)
                ReplaceLiquidSamplePoints(tileHeights, liquid.Data.AsSpan(t * liquidStride, liquidStride));

            WdlHeightTile tile = WdlWriter.ExtractTileHeightsFromAlpha(tileHeights, 0, 0);
            for (int j = 0; j < OuterDim * OuterDim; j++)
                outer[(t * OuterDim * OuterDim) + j] = tile.OuterHeights[j];
            for (int j = 0; j < InnerDim * InnerDim; j++)
                inner[(t * InnerDim * InnerDim) + j] = tile.InnerHeights[j];
        }

        Npy.WriteNpz(output, [
            ("outer", outer, [tileCount, OuterDim, OuterDim]),
            ("inner", inner, [tileCount, InnerDim, InnerDim]),
        ]);

        Console.WriteLine($"synth: wrote {tileCount} tile(s) to {output}");
        return 0;
    }

    /// <summary>
    /// Rewrites the height value at each WDL lattice sample point that sits on liquid
    /// with the height of the nearest non-liquid pixel, so the untouched C# extraction
    /// path (WdlWriter.ExtractTileHeightsFromAlpha) samples dry terrain. A 257-grid
    /// pixel is "liquid" when every adjacent 256-grid cell has mask > 0.5.
    /// </summary>
    static void ReplaceLiquidSamplePoints(float[,] heights, ReadOnlySpan<float> liquidMask)
    {
        bool[,] pixelLiquid = new bool[257, 257];
        for (int y = 0; y < 257; y++)
        {
            for (int x = 0; x < 257; x++)
            {
                bool allLiquid = true;
                for (int cy = Math.Max(0, y - 1); cy <= Math.Min(255, y) && allLiquid; cy++)
                    for (int cx = Math.Max(0, x - 1); cx <= Math.Min(255, x) && allLiquid; cx++)
                        allLiquid &= liquidMask[(cy * 256) + cx] > 0.5f;
                pixelLiquid[y, x] = allLiquid;
            }
        }

        List<(int Y, int X)> samplePoints = [];
        for (int r = 0; r < OuterDim; r++)
            for (int c = 0; c < OuterDim; c++)
                samplePoints.Add((Math.Min(r * 16, 256), Math.Min(c * 16, 256)));
        for (int r = 0; r < InnerDim; r++)
            for (int c = 0; c < InnerDim; c++)
                samplePoints.Add((Math.Min((r * 16) + 8, 256), Math.Min((c * 16) + 8, 256)));

        foreach ((int y, int x) in samplePoints)
        {
            if (!pixelLiquid[y, x])
                continue;

            if (TryFindNearestDryHeight(heights, pixelLiquid, y, x, out float dryHeight))
                heights[y, x] = dryHeight;
        }
    }

    static bool TryFindNearestDryHeight(float[,] heights, bool[,] pixelLiquid, int y, int x, out float dryHeight)
    {
        for (int radius = 1; radius <= 256; radius++)
        {
            int yMin = Math.Max(0, y - radius);
            int yMax = Math.Min(256, y + radius);
            int xMin = Math.Max(0, x - radius);
            int xMax = Math.Min(256, x + radius);
            for (int cy = yMin; cy <= yMax; cy++)
            {
                for (int cx = xMin; cx <= xMax; cx++)
                {
                    if (Math.Max(Math.Abs(cy - y), Math.Abs(cx - x)) != radius)
                        continue;
                    if (pixelLiquid[cy, cx])
                        continue;

                    dryHeight = heights[cy, cx];
                    return true;
                }
            }
        }

        dryHeight = 0f;
        return false;
    }

    // ---------------------------------------------------------------- shared

    static Npy.NpyArray SelectArray(Dictionary<string, Npy.NpyArray> arrays, string[] keys, string source)
    {
        foreach (string key in keys)
        {
            if (arrays.TryGetValue(key, out Npy.NpyArray? array))
                return array;
        }

        throw new InvalidDataException($"'{source}' has none of the expected keys: {string.Join(", ", keys)} (found: {string.Join(", ", arrays.Keys)}).");
    }

    static string? GetOption(string[] args, string name)
    {
        int index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    static int? GetIntOption(string[] args, string name)
    {
        string? raw = GetOption(args, name);
        return raw is not null && int.TryParse(raw, out int value) ? value : null;
    }
}
