using AlphaWDTReader.Readers;

namespace AlphaWDTReader.Tool;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintHelp();
            return 0;
        }

        var cmd = args[0];
        switch (cmd)
        {
            case "scan":
                return RunScan(args.Skip(1).ToArray());
            case "dump-tiles":
                return RunDumpTiles(args.Skip(1).ToArray());
            case "dump-chunks":
                return RunDumpChunks(args.Skip(1).ToArray());
            case "dump-terrain":
                return RunDumpTerrain(args.Skip(1).ToArray());
            default:
                Console.Error.WriteLine($"Unknown command: {cmd}\n");
                PrintHelp();
                return 2;
        }
    }

    private static int RunScan(string[] args)
    {
        string? input = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--input" && i + 1 < args.Length)
            {
                input = args[++i];
            }
        }

        if (string.IsNullOrWhiteSpace(input) || !File.Exists(input))
        {
            Console.Error.WriteLine("Missing or invalid --input <path to Alpha combined file>");
            return 2;
        }

        var result = AlphaScanner.Scan(input);

        Console.WriteLine($"File: {result.FilePath}");
        Console.WriteLine($"Chunks scanned: {result.ChunkCount}");
        Console.WriteLine($"Has MVER: {result.HasMver}");
        Console.WriteLine($"Has MPHD: {result.HasMphd}");
        Console.WriteLine($"Has MAIN: {result.HasMain}");
        Console.WriteLine($"MAIN declared tiles (non-zero offsets): {result.MainDeclaredTiles?.ToString() ?? "n/a"}");
        Console.WriteLine($"MDNM names: {result.DoodadNameCount}");
        if (result.FirstDoodadNames.Count > 0)
            Console.WriteLine("  First MDNM: " + string.Join(", ", result.FirstDoodadNames));
        Console.WriteLine($"MONM names: {result.WmoNameCount}");
        if (result.FirstWmoNames.Count > 0)
            Console.WriteLine("  First MONM: " + string.Join(", ", result.FirstWmoNames));

        return 0;
    }

    private static int RunDumpTiles(string[] args)
    {
        string? input = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--input" && i + 1 < args.Length)
            {
                input = args[++i];
            }
        }
        if (string.IsNullOrWhiteSpace(input) || !File.Exists(input))
        {
            Console.Error.WriteLine("Missing or invalid --input <path to Alpha combined file>");
            return 2;
        }

        var entries = AlphaMainReader.ReadMainTable(input);
        if (entries.Count == 0)
        {
            Console.WriteLine("No MAIN table found or empty.");
            return 0;
        }

        Console.WriteLine("tile_x,tile_y,present,mcnk_count");
        foreach (var e in entries)
        {
            bool present = e.Offset != 0 && e.Size != 0;
            int mcnk = 0;
            if (present)
            {
                var viaMcin = AlphaMcinReader.CountPresentChunks(input, e);
                mcnk = viaMcin ?? AlphaTileScanner.CountMcnkBlocks(input, e);
            }
            Console.WriteLine($"{e.TileX},{e.TileY},{(present ? 1 : 0)},{mcnk}");
        }
        return 0;
    }

    private static int RunDumpChunks(string[] args)
    {
        string? input = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--input" && i + 1 < args.Length)
            {
                input = args[++i];
            }
        }
        if (string.IsNullOrWhiteSpace(input) || !File.Exists(input))
        {
            Console.Error.WriteLine("Missing or invalid --input <path to Alpha combined file>");
            return 2;
        }

        var entries = AlphaMainReader.ReadMainTable(input);
        if (entries.Count == 0)
        {
            Console.WriteLine("No MAIN table found or empty.");
            return 0;
        }

        Console.WriteLine("tile_x,tile_y,chunks,mcvt,mcnr,mclq");
        foreach (var e in entries)
        {
            if (e.Offset == 0 || e.Size == 0)
            {
                Console.WriteLine($"{e.TileX},{e.TileY},0,0,0,0");
                continue;
            }
            var idx = AlphaChunkIndexer.BuildForTile(input, e);
            int chunks = idx.Count;
            int mcvt = idx.Count(c => c.OfsMCVT != 0);
            int mcnr = idx.Count(c => c.OfsMCNR != 0);
            int mclq = idx.Count(c => c.OfsMCLQ != 0);
            Console.WriteLine($"{e.TileX},{e.TileY},{chunks},{mcvt},{mcnr},{mclq}");
        }
        return 0;
    }

    private static int RunDumpTerrain(string[] args)
    {
        string? input = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--input" && i + 1 < args.Length)
            {
                input = args[++i];
            }
        }
        if (string.IsNullOrWhiteSpace(input) || !File.Exists(input))
        {
            Console.Error.WriteLine("Missing or invalid --input <path to Alpha combined file>");
            return 2;
        }

        var entries = AlphaMainReader.ReadMainTable(input);
        if (entries.Count == 0)
        {
            Console.WriteLine("No MAIN table found or empty.");
            return 0;
        }

        Console.WriteLine("tile_x,tile_y,chunks_with_mcvt,chunks_with_mcnr,min_height,max_height");
        foreach (var e in entries)
        {
            if (e.Offset == 0 || e.Size == 0)
            {
                Console.WriteLine($"{e.TileX},{e.TileY},0,0,0,0");
                continue;
            }

            var idx = AlphaChunkIndexer.BuildForTile(input, e);
            int okMcvt = 0, okMcnr = 0;
            float minH = float.PositiveInfinity, maxH = float.NegativeInfinity;

            foreach (var ci in idx)
            {
                var heights = AlphaTerrainDecoder.ReadHeights(input, ci);
                if (heights != null)
                {
                    okMcvt++;
                    foreach (var h in heights)
                    {
                        if (h < minH) minH = h;
                        if (h > maxH) maxH = h;
                    }
                }
                var normals = AlphaTerrainDecoder.ReadNormals(input, ci);
                if (normals != null) okMcnr++;
            }

            if (float.IsPositiveInfinity(minH)) { minH = 0; maxH = 0; }
            Console.WriteLine($"{e.TileX},{e.TileY},{okMcvt},{okMcnr},{minH},{maxH}");
        }
        return 0;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("AlphaWDTReader.Tool");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  scan --input <path>         Scan Alpha combined file and print summary");
        Console.WriteLine("  dump-tiles --input <path>   Print per-tile presence and MCNK counts");
        Console.WriteLine("  dump-chunks --input <path>  Print per-tile subchunk presence summary");
        Console.WriteLine("  dump-terrain --input <path> Validate MCVT/MCNR and report height range per tile");
        Console.WriteLine();
    }
}
