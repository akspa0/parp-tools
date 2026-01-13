using System;
using System.IO;
using System.Text;
using WoWRollback.Core.Models.ADT; // Reusing AdtData/AdtChunk concepts where applicable or defining new ones?
// Let's keep WDL models simple.

namespace WoWRollback.Core.Services.Parsers;

public class WdlParser
{
    public class WdlData
    {
        // 64x64 grid of tiles
        public WdlTile?[] Tiles { get; } = new WdlTile?[64 * 64];
    }

    public class WdlTile
    {
        // MARE data
        public short[,] Height17 { get; } = new short[17, 17];
        public short[,] Height16 { get; } = new short[16, 16];
        public bool HasData { get; set; }
    }

    /// <summary>
    /// Parses a WDL file entirely.
    /// </summary>
    public static WdlData Parse(byte[] data)
    {
        var result = new WdlData();
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);

        uint[]? maofOffsets = null;

        while (ms.Position < ms.Length)
        {
            if (ms.Position + 4 > ms.Length) break;
            
            // Read standard 4-char header inverted
            var headBytes = br.ReadBytes(4);
            Array.Reverse(headBytes);
            string head = Encoding.ASCII.GetString(headBytes);
            uint size = br.ReadUInt32();
            
            long chunkStart = ms.Position;
            long nextChunk = chunkStart + size;

            if (head == "MAOF")
            {
                if (size != 64 * 64 * 4)
                {
                    // Warning?
                }
                maofOffsets = new uint[64 * 64];
                for (int i = 0; i < 64 * 64; i++)
                {
                    maofOffsets[i] = br.ReadUInt32();
                }
            }
            // MVER, MHDR? default skip
            
            ms.Position = nextChunk;
        }

        if (maofOffsets != null)
        {
            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    int index = y * 64 + x;
                    uint offset = maofOffsets[index];
                    
                    if (offset > 0 && offset < data.Length)
                    {
                        ms.Position = offset; // MARE is at offset
                        // Validation: Check if there's enough space for 17x17 + 16x16 shorts
                        // 17*17*2 = 578
                        // 16*16*2 = 512
                        // Total 1090 bytes
                        if (ms.Position + 1090 <= ms.Length)
                        {
                            var tile = new WdlTile { HasData = true };
                            
                            // Read 17x17 (Outer)
                            for (int r = 0; r < 17; r++)
                            {
                                for (int c = 0; c < 17; c++)
                                {
                                    tile.Height17[r, c] = br.ReadInt16();
                                }
                            }

                            // Read 16x16 (Inner)
                            for (int r = 0; r < 16; r++)
                            {
                                for (int c = 0; c < 16; c++)
                                {
                                    tile.Height16[r, c] = br.ReadInt16();
                                }
                            }
                            result.Tiles[index] = tile;
                        }
                    }
                }
            }
        }

        return result;
    }
}
