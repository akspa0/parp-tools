using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Creates WDT files for 3.3.5 format.
/// </summary>
public sealed class Wdt335Writer
{
    // WDT chunk signatures - REVERSED on disk (e.g., "MVER" stored as "REVM")
    private static readonly byte[] MVER = Encoding.ASCII.GetBytes("REVM");
    private static readonly byte[] MPHD = Encoding.ASCII.GetBytes("DHPM");
    private static readonly byte[] MAIN = Encoding.ASCII.GetBytes("NIAM");
    private static readonly byte[] MWMO = Encoding.ASCII.GetBytes("OMWM");
    private static readonly byte[] MODF = Encoding.ASCII.GetBytes("FDOM");

    /// <summary>
    /// MAIN entry flags for WDT.
    /// </summary>
    [Flags]
    public enum MainFlags : uint
    {
        None = 0,
        HasAdt = 0x0001,      // ADT file exists for this tile
        AllWater = 0x0002,    // Tile is all water (no ADT needed)
    }

    /// <summary>
    /// MPHD flags for WDT.
    /// </summary>
    [Flags]
    public enum MphdFlags : uint
    {
        None = 0,
        WdtUsesGlobalMapObj = 0x0001,    // Use global WMO
        AdtHasMccv = 0x0002,             // ADTs have MCCV chunk
        AdtHasBigAlpha = 0x0004,         // ADTs use big alpha
        AdtHasDoodadRefsSortedBySizeCat = 0x0008,
        LightingVertices = 0x0010,       // ADTs have MCLV
        UpsideDownGround = 0x0020,
        Unk0x0040 = 0x0040,
        AdtHasHeightTexturing = 0x0080,  // ADTs have MTXF
        Unk0x0100 = 0x0100,
        WdtHasMaid = 0x0200,             // WDT has MAID chunk
    }

    /// <summary>
    /// Create a WDT file for the given tiles.
    /// </summary>
    public byte[] CreateWdt(HashSet<(int x, int y)> tiles, MphdFlags flags = MphdFlags.None)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // MVER - Version (always 18 for 3.3.5)
        bw.Write(MVER);
        bw.Write(4);
        bw.Write(18);

        // MPHD - Map header (32 bytes)
        bw.Write(MPHD);
        bw.Write(32);
        bw.Write((uint)flags);  // flags
        bw.Write(0u);           // something
        bw.Write(0u);           // unused[0]
        bw.Write(0u);           // unused[1]
        bw.Write(0u);           // unused[2]
        bw.Write(0u);           // unused[3]
        bw.Write(0u);           // unused[4]
        bw.Write(0u);           // unused[5]

        // MAIN - 64x64 tile flags (8 bytes per entry = 32768 bytes)
        bw.Write(MAIN);
        bw.Write(64 * 64 * 8);
        
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                uint entryFlags = tiles.Contains((x, y)) ? (uint)MainFlags.HasAdt : 0;
                bw.Write(entryFlags);  // flags
                bw.Write(0u);          // asyncId (unused)
            }
        }

        // MWMO - Empty (no global WMO)
        bw.Write(MWMO);
        bw.Write(0);

        // MODF - Empty (no global WMO placement)
        bw.Write(MODF);
        bw.Write(0);

        return ms.ToArray();
    }

    /// <summary>
    /// Create a WDT file from a list of existing ADT files.
    /// </summary>
    public byte[] CreateWdtFromAdtFiles(string adtDirectory, string mapName)
    {
        var tiles = new HashSet<(int x, int y)>();

        // Scan for ADT files matching the pattern: mapname_X_Y.adt
        var pattern = $"{mapName}_*.adt";
        foreach (var file in Directory.GetFiles(adtDirectory, pattern))
        {
            var fileName = Path.GetFileNameWithoutExtension(file);
            var parts = fileName.Split('_');
            
            if (parts.Length >= 3)
            {
                // Try to parse the last two parts as coordinates
                if (int.TryParse(parts[^2], out int x) && int.TryParse(parts[^1], out int y))
                {
                    if (x >= 0 && x < 64 && y >= 0 && y < 64)
                    {
                        tiles.Add((x, y));
                    }
                }
            }
        }

        Console.WriteLine($"[INFO] Found {tiles.Count} tiles for map '{mapName}'");
        return CreateWdt(tiles);
    }

    /// <summary>
    /// Write WDT file to disk.
    /// </summary>
    public void WriteWdt(string outputPath, HashSet<(int x, int y)> tiles, MphdFlags flags = MphdFlags.None)
    {
        var data = CreateWdt(tiles, flags);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, data);
        Console.WriteLine($"[INFO] Wrote WDT: {outputPath} ({data.Length} bytes, {tiles.Count} tiles)");
    }

    /// <summary>
    /// Write WDT file based on existing ADT files.
    /// </summary>
    public void WriteWdtFromAdtFiles(string adtDirectory, string mapName, string? outputPath = null)
    {
        outputPath ??= Path.Combine(adtDirectory, $"{mapName}.wdt");
        
        var tiles = new HashSet<(int x, int y)>();
        var pattern = $"{mapName}_*.adt";
        
        foreach (var file in Directory.GetFiles(adtDirectory, pattern))
        {
            var fileName = Path.GetFileNameWithoutExtension(file);
            var parts = fileName.Split('_');
            
            if (parts.Length >= 3)
            {
                if (int.TryParse(parts[^2], out int x) && int.TryParse(parts[^1], out int y))
                {
                    if (x >= 0 && x < 64 && y >= 0 && y < 64)
                    {
                        tiles.Add((x, y));
                    }
                }
            }
        }

        if (tiles.Count == 0)
        {
            Console.WriteLine($"[WARN] No ADT files found matching '{pattern}' in {adtDirectory}");
            return;
        }

        WriteWdt(outputPath, tiles);
    }
}
