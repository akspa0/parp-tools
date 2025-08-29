using AlphaWDTReader.Readers;
using System.Text;

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

        var outDir = EnsureOutputDir(input);
        var outFile = Path.Combine(outDir, "scan.txt");
        var sb = new StringBuilder();
        sb.AppendLine($"File: {result.FilePath}");
        sb.AppendLine($"Chunks scanned: {result.ChunkCount}");
        sb.AppendLine($"Has MVER: {result.HasMver}");
        sb.AppendLine($"Has MPHD: {result.HasMphd}");
        sb.AppendLine($"Has MAIN: {result.HasMain}");
        sb.AppendLine($"MAIN declared tiles (non-zero offsets): {result.MainDeclaredTiles?.ToString() ?? "n/a"}");
        sb.AppendLine($"MDNM names: {result.DoodadNameCount}");
        if (result.FirstDoodadNames.Count > 0)
            sb.AppendLine("  First MDNM: " + string.Join(", ", result.FirstDoodadNames));
        sb.AppendLine($"MONM names: {result.WmoNameCount}");
        if (result.FirstWmoNames.Count > 0)
            sb.AppendLine("  First MONM: " + string.Join(", ", result.FirstWmoNames));
        File.WriteAllText(outFile, sb.ToString());

        Console.WriteLine($"Saved scan to: {outFile}");
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

        var outDir = EnsureOutputDir(input);
        var outFile = Path.Combine(outDir, "dump-tiles.csv");
        var sb = new StringBuilder();
        sb.AppendLine("tile_x,tile_y,present,mcnk_count");
        foreach (var e in entries)
        {
            bool present = e.Offset != 0; // Alpha: Size may be 0 even when present
            int mcnk = 0;
            if (present)
            {
                var viaMcin = AlphaMcinReader.CountPresentChunks(input, e);
                mcnk = viaMcin ?? AlphaTileScanner.CountMcnkBlocks(input, e);
            }
            sb.AppendLine($"{e.TileX},{e.TileY},{(present ? 1 : 0)},{mcnk}");
        }
        File.WriteAllText(outFile, sb.ToString());
        Console.WriteLine($"Saved tiles CSV to: {outFile}");
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

        var outDir = EnsureOutputDir(input);
        var outFile = Path.Combine(outDir, "dump-chunks.csv");
        var sb = new StringBuilder();
        sb.AppendLine("tile_x,tile_y,chunks,mcvt,mcnr,mclq");
        foreach (var e in entries)
        {
            if (e.Offset == 0)
            {
                sb.AppendLine($"{e.TileX},{e.TileY},0,0,0,0");
                continue;
            }
            var idx = AlphaChunkIndexer.BuildForTile(input, e);
            int chunks = idx.Count;
            int mcvt = idx.Count(c => c.OfsMCVT != 0);
            int mcnr = idx.Count(c => c.OfsMCNR != 0);
            int mclq = idx.Count(c => c.OfsMCLQ != 0);
            sb.AppendLine($"{e.TileX},{e.TileY},{chunks},{mcvt},{mcnr},{mclq}");
        }
        File.WriteAllText(outFile, sb.ToString());
        Console.WriteLine($"Saved chunk CSV to: {outFile}");
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

        var outDir = EnsureOutputDir(input);
        var outFile = Path.Combine(outDir, "dump-terrain.csv");
        var sb = new StringBuilder();
        sb.AppendLine("tile_x,tile_y,chunks_with_mcvt,chunks_with_mcnr,min_height,max_height");
        foreach (var e in entries)
        {
            if (e.Offset == 0)
            {
                sb.AppendLine($"{e.TileX},{e.TileY},0,0,0,0");
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
            sb.AppendLine($"{e.TileX},{e.TileY},{okMcvt},{okMcnr},{minH},{maxH}");
        }
        File.WriteAllText(outFile, sb.ToString());
        Console.WriteLine($"Saved terrain CSV to: {outFile}");
        return 0;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("AlphaWDTReader.Tool");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  scan --input <path>         Scan Alpha combined file and write scan.txt");
        Console.WriteLine("  dump-tiles --input <path>   Write per-tile presence and MCNK counts to dump-tiles.csv");
        Console.WriteLine("  dump-chunks --input <path>  Write per-tile subchunk presence to dump-chunks.csv");
        Console.WriteLine("  dump-terrain --input <path> Write MCVT/MCNR validation and height range to dump-terrain.csv");
        Console.WriteLine();
    }

    private static string EnsureOutputDir(string inputPath)
    {
        var wdtName = Path.GetFileNameWithoutExtension(inputPath);
        var ts = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        var baseDir = Path.Combine(Environment.CurrentDirectory, "wdt_outputs");
        var outDir = Path.Combine(baseDir, $"{wdtName}-{ts}");
        Directory.CreateDirectory(outDir);
        return outDir;
    }
}
