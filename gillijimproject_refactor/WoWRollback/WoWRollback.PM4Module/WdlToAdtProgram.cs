// WDL to ADT Generator - Generates ADT terrain from WDL heights for tiles without existing ADTs
// Run with: dotnet run --project WoWRollback/WoWRollback.PM4Module -- wdl-to-adt

namespace WoWRollback.PM4Module;

public static class WdlToAdtProgram
{
    public static int Run(string[] args)
    {
        Console.WriteLine("=== WDL to ADT Generator ===\n");

        // Parse arguments
        string? wdlPath = null;
        string? outDir = null;
        string? existingDir = null;
        string? mapName = null;
        string? minimapDir = null;
        bool fillGapsOnly = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--in": wdlPath = args[++i]; break;
                case "--out": outDir = args[++i]; break;
                case "--existing": existingDir = args[++i]; break;
                case "--map": mapName = args[++i]; break;
                case "--minimap": minimapDir = args[++i]; break;
                case "--fill-gaps": fillGapsOnly = true; break;
                case "--help":
                case "-h":
                    PrintHelp();
                    return 0;
            }
        }

        if (string.IsNullOrEmpty(wdlPath) || !File.Exists(wdlPath))
        {
            Console.Error.WriteLine("Error: --in <wdl-file> is required and must exist");
            PrintHelp();
            return 1;
        }

        outDir ??= Path.Combine(Path.GetDirectoryName(wdlPath) ?? ".", "adt_from_wdl");
        mapName ??= Path.GetFileNameWithoutExtension(wdlPath);

        Console.WriteLine($"Input WDL: {wdlPath}");
        Console.WriteLine($"Output: {outDir}");
        Console.WriteLine($"Map name: {mapName}");
        Console.WriteLine($"Fill gaps only: {fillGapsOnly}");
        if (!string.IsNullOrEmpty(existingDir))
            Console.WriteLine($"Existing ADTs: {existingDir}");
        if (!string.IsNullOrEmpty(minimapDir))
            Console.WriteLine($"Minimap dir: {minimapDir} (MCCV painting enabled)");
        Console.WriteLine();

        try
        {
            // Parse WDL file
            var wdlTiles = ParseWdl(wdlPath);
            int tileCount = 0;
for (int ty = 0; ty < 64; ty++)
    for (int tx = 0; tx < 64; tx++)
        if (wdlTiles[ty, tx] != null) tileCount++;
Console.WriteLine($"Parsed WDL: {tileCount} tiles with data");

            Directory.CreateDirectory(outDir);

            int generated = 0;
            int skipped = 0;
            int existingSkipped = 0;

            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    var tile = wdlTiles[y, x];
                    if (tile == null)
                    {
                        skipped++;
                        continue;
                    }

                    var adtPath = Path.Combine(outDir, $"{mapName}_{x}_{y}.adt");

                    // Skip if --fill-gaps and ADT already exists
                    if (fillGapsOnly)
                    {
                        bool exists = File.Exists(adtPath);
                        if (!exists && !string.IsNullOrEmpty(existingDir))
                        {
                            var existingPath = Path.Combine(existingDir, $"{mapName}_{x}_{y}.adt");
                            exists = File.Exists(existingPath);
                        }
                        if (exists)
                        {
                            existingSkipped++;
                            continue;
                        }
                    }

                    // Load minimap MCCV data if available
                    byte[][]? mccvData = null;
                    if (!string.IsNullOrEmpty(minimapDir))
                    {
                        mccvData = TryLoadMinimapMccv(minimapDir, mapName, x, y);
                    }

                    var adtData = WdlToAdtGenerator.GenerateAdt(tile, x, y, mccvData);
                    File.WriteAllBytes(adtPath, adtData);
                    generated++;

                    if (generated % 50 == 0 || generated == 1)
                    {
                        Console.WriteLine($"Generated {generated} ADTs...");
                    }
                }
            }

            Console.WriteLine($"\nComplete: {generated} ADTs generated");
            Console.WriteLine($"  Skipped (no WDL data): {skipped}");
            Console.WriteLine($"  Skipped (existing ADT): {existingSkipped}");

            // Generate WDT based on actual ADT files in output directory
            var wdtPath = Path.Combine(outDir, $"{mapName}.wdt");
            GenerateWdtFromAdtFiles(outDir, mapName, wdtPath);
            Console.WriteLine($"WDT written: {wdtPath}");

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 2;
        }
    }

    private static WdlToAdtGenerator.WdlTileData?[,] ParseWdl(string path)
    {
        var tiles = new WdlToAdtGenerator.WdlTileData?[64, 64];
        var data = File.ReadAllBytes(path);
        int pos = 0;

        // Find MAOF chunk (offsets to MARE chunks)
        uint[]? maofOffsets = null;
        
        while (pos < data.Length - 8)
        {
            string sig = System.Text.Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);
            
            if (sig == "FOAM") // MAOF reversed
            {
                maofOffsets = new uint[64 * 64];
                for (int i = 0; i < 64 * 64; i++)
                {
                    maofOffsets[i] = BitConverter.ToUInt32(data, pos + 8 + i * 4);
                }
            }
            
            pos += 8 + size;
        }

        if (maofOffsets == null)
        {
            Console.WriteLine("Warning: No MAOF chunk found in WDL");
            return tiles;
        }

        // Parse each tile
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                uint offset = maofOffsets[y * 64 + x];
                if (offset == 0) continue;

                var tile = new WdlToAdtGenerator.WdlTileData();
                int tilePos = (int)offset;

                // Read MARE chunk
                if (tilePos + 8 < data.Length)
                {
                    string sig = System.Text.Encoding.ASCII.GetString(data, tilePos, 4);
                    int size = BitConverter.ToInt32(data, tilePos + 4);
                    
                    if (sig == "ERAM") // MARE reversed
                    {
                        int dataPos = tilePos + 8;
                        
                        // 17x17 outer heights
                        for (int j = 0; j < 17; j++)
                        {
                            for (int i = 0; i < 17; i++)
                            {
                                tile.Height17[j, i] = BitConverter.ToInt16(data, dataPos);
                                dataPos += 2;
                            }
                        }
                        
                        // 16x16 inner heights
                        for (int j = 0; j < 16; j++)
                        {
                            for (int i = 0; i < 16; i++)
                            {
                                tile.Height16[j, i] = BitConverter.ToInt16(data, dataPos);
                                dataPos += 2;
                            }
                        }
                    }
                }

                tiles[y, x] = tile;
            }
        }

        return tiles;
    }

    private static void GenerateWdt(WdlToAdtGenerator.WdlTileData?[,] tiles, string wdtPath)
    {
        using var fs = new FileStream(wdtPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // MVER
        bw.Write(System.Text.Encoding.ASCII.GetBytes("REVM"));
        bw.Write(4);
        bw.Write(18);

        // MPHD (32 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("DHPM"));
        bw.Write(32);
        bw.Write(0x0E); // flags: MCCV | BigAlpha | DoodadRefsSorted
        for (int i = 0; i < 28; i++) bw.Write((byte)0);

        // MAIN (64*64*8 = 32768 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("NIAM"));
        bw.Write(64 * 64 * 8);
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                uint flags = tiles[y, x] != null ? 1u : 0u;
                bw.Write(flags);
                bw.Write(0u);
            }
        }

        // MWMO (empty)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("OMWM"));
        bw.Write(0);

        // MODF (empty)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("FDOM"));
        bw.Write(0);
    }

    private static void GenerateWdtFromAdtFiles(string adtDir, string mapName, string wdtPath)
    {
        // Scan for actual ADT files
        var tileFlags = new bool[64, 64];
        int count = 0;
        
        foreach (var file in Directory.GetFiles(adtDir, $"{mapName}_*.adt"))
        {
            var name = Path.GetFileNameWithoutExtension(file);
            var parts = name.Split('_');
            if (parts.Length >= 3 &&
                int.TryParse(parts[^2], out int x) &&
                int.TryParse(parts[^1], out int y) &&
                x >= 0 && x < 64 && y >= 0 && y < 64)
            {
                tileFlags[y, x] = true;
                count++;
            }
        }

        using var fs = new FileStream(wdtPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // MVER
        bw.Write(System.Text.Encoding.ASCII.GetBytes("REVM"));
        bw.Write(4);
        bw.Write(18);

        // MPHD (32 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("DHPM"));
        bw.Write(32);
        bw.Write(0x0E); // flags: MCCV | BigAlpha | DoodadRefsSorted
        for (int i = 0; i < 28; i++) bw.Write((byte)0);

        // MAIN (64*64*8 = 32768 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("NIAM"));
        bw.Write(64 * 64 * 8);
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                uint flags = tileFlags[y, x] ? 1u : 0u;
                bw.Write(flags);
                bw.Write(0u);
            }
        }

        // MWMO (empty)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("OMWM"));
        bw.Write(0);

        // MODF (empty)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("FDOM"));
        bw.Write(0);

        Console.WriteLine($"  Tiles marked in WDT: {count}");
    }

    private static byte[][]? TryLoadMinimapMccv(string minimapDir, string mapName, int tileX, int tileY)
    {
        // Try common minimap naming patterns (development_X_Y.png is the primary format)
        string[] patterns = 
        {
            $"{mapName}_{tileX}_{tileY}.png",
            $"map{tileX:D2}_{tileY:D2}.png",
            $"map{tileX}_{tileY}.png",
            $"{mapName}{tileX:D2}_{tileY:D2}.png",
            $"{mapName}_{tileX}_{tileY}.blp",
            $"map{tileX:D2}_{tileY:D2}.blp",
        };

        foreach (var pattern in patterns)
        {
            var path = Path.Combine(minimapDir, pattern);
            if (File.Exists(path))
            {
                try
                {
                    return MccvPainter.GenerateAllMccvFromImage(path);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  [WARN] Failed to load minimap {path}: {ex.Message}");
                }
            }
        }

        return null;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("WDL to ADT Generator - Generate 3.3.5 ADT terrain from WDL heights");
        Console.WriteLine();
        Console.WriteLine("Usage: dotnet run -- wdl-to-adt --in <wdl-file> [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --in <path>       Input WDL file (required)");
        Console.WriteLine("  --out <dir>       Output directory (default: adt_from_wdl)");
        Console.WriteLine("  --map <name>      Map name for output files (default: WDL filename)");
        Console.WriteLine("  --minimap <dir>   Directory with minimap PNG/BLP files for MCCV painting");
        Console.WriteLine("  --fill-gaps       Only generate ADTs for tiles without existing ADTs");
        Console.WriteLine("  --existing <dir>  Directory with existing ADTs to check (with --fill-gaps)");
    }
}
