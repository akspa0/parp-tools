using System;
using System.IO;
using System.Text;
using System.Collections.Generic;

namespace WoWRollback.PM4Module.Services;

/// <summary>
/// Service to parse WDL files and extract tile height data for ADT generation.
/// </summary>
public class WdlService
{
    private class WdlChunk
    {
        public string Header { get; set; } = "";
        public uint Size { get; set; }
        public long DataOffset { get; set; }
    }

    /// <summary>
    /// Reads a WDL file and extracts data for a specific tile.
    /// If the tile has no data (offset is 0), returns null.
    /// </summary>
    public WdlToAdtGenerator.WdlTileData? GetTileData(string wdlPath, int tileX, int tileY)
    {
        if (!File.Exists(wdlPath)) return null;

        using var fs = File.OpenRead(wdlPath);
        using var br = new BinaryReader(fs);

        // Read chunks until we find MAOF (offsets)
        uint[]? maofOffsets = null;
        
        while (fs.Position < fs.Length)
        {
            if (fs.Position + 4 > fs.Length) break;
            
            var headBytes = br.ReadBytes(4);
            Array.Reverse(headBytes);
            string head = Encoding.ASCII.GetString(headBytes);
            uint size = br.ReadUInt32();
            long nextChunk = fs.Position + size;

            if (head == "MAOF")
            {
                // Area Offsets - 64x64 array of uints
                // Indices are [y * 64 + x]
                if (size != 64 * 64 * 4) 
                {
                    // Invalid MAOF size?
                }
                maofOffsets = new uint[64 * 64];
                for (int i = 0; i < 64 * 64; i++)
                {
                    maofOffsets[i] = br.ReadUInt32();
                }
            }
            // MVER, MHDR, etc can be skipped for now as we just need height data
            
            fs.Position = nextChunk;
        }

        if (maofOffsets == null) return null;

        int index = tileY * 64 + tileX;
        uint tileOffset = maofOffsets[index];

        if (tileOffset == 0) return null;

        // Read MARE chunk data at tileOffset
        fs.Position = tileOffset;
        
        // MARE chunk header isn't standard?
        // Actually MAOF points directly to the MARE data for that tile (HeightMap).
        // Format:
        // 17x17 heights (int16) = 289 * 2 = 578 bytes
        // 16x16 heights (int16) = 256 * 2 = 512 bytes
        // Total 1090 bytes (plus potential MAHO but usually separate or following)
        
        var result = new WdlToAdtGenerator.WdlTileData();

        // 17x17 Outer
        for (int y = 0; y < 17; y++)
        {
            for (int x = 0; x < 17; x++)
            {
                result.Height17[y, x] = br.ReadInt16();
            }
        }

        // 16x16 Inner
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                result.Height16[y, x] = br.ReadInt16();
            }
        }
        
        // TODO: Handle MAHO (holes) if present in WDL? 
        // Standard WDL v1.1 usually just has height data here.

        return result;
    }
}
