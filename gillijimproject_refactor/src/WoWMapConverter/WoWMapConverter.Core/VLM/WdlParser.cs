using System;
using System.IO;
using System.Text;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Parses WDL (World Detail Level) files for WoW client map previews.
/// Alpha-era files use version 0x12, but later clients still retain the same MAOF/MARE core layout.
/// 
/// File layout:
///   MVER chunk → version uint32
///   MAOF chunk → uint32[4096] absolute file offsets (64×64 grid)
///   Per nonzero offset → MARE chunk header + int16[545] heights (17×17 outer + 16×16 inner)
/// </summary>
public class WdlParser
{
    // FourCC values as uint32 (little-endian on disk: "REVM", "FOAM", "ERAM")
    private const uint FOURCC_MVER = 0x4D564552; // 'M','V','E','R'
    private const uint FOURCC_MAOF = 0x4D414F46; // 'M','A','O','F'
    private const uint FOURCC_MARE = 0x4D415245; // 'M','A','R','E'

    private const int VERSION_EXPECTED = 0x12;
    private const int GRID_SIZE = 64;
    private const int CELL_COUNT = GRID_SIZE * GRID_SIZE; // 4096
    private const int MARE_HEIGHT_COUNT = 545; // 17*17 + 16*16
    private const int MARE_BYTES = MARE_HEIGHT_COUNT * 2; // 1090 bytes of int16
    private const int MARE_CHUNK_HEADER = 8; // FourCC(4) + size(4)

    public class WdlData
    {
        // 64x64 grid of tiles
        public WdlTile?[] Tiles { get; } = new WdlTile?[CELL_COUNT];
        public int Version { get; set; }
    }

    public class WdlTile
    {
        // MARE data: 545 heights as flat array matching client layout
        // [0..288] = 17×17 outer lattice, [289..544] = 16×16 inner lattice
        public float[] Heights { get; } = new float[MARE_HEIGHT_COUNT];

        // Also expose as 2D arrays for backward compatibility
        public short[,] Height17 { get; } = new short[17, 17];
        public short[,] Height16 { get; } = new short[16, 16];
        public bool HasData { get; set; }

        // Precomputed bounds
        public float MinZ { get; set; }
        public float MaxZ { get; set; }
    }

    /// <summary>
    /// Parses a WDL file matching client behavior (CMap::LoadWdl).
    /// </summary>
    public static WdlData? Parse(byte[] data)
    {
        if (data == null || data.Length < 20) return null;

        uint mverFourCC = ReadUInt32(data, 0);
        uint mverSize = ReadUInt32(data, 4);
        if (mverFourCC != FOURCC_MVER)
        {
            Console.WriteLine($"[WdlParser] Expected MVER (0x{FOURCC_MVER:X8}), got 0x{mverFourCC:X8}");
            return null;
        }

        if (mverSize < sizeof(int) || 8 + mverSize > data.Length)
        {
            Console.WriteLine($"[WdlParser] Invalid MVER size {mverSize} for WDL length {data.Length}");
            return null;
        }

        int version = BitConverter.ToInt32(data, 8);
        if (version != VERSION_EXPECTED)
            Console.WriteLine($"[WdlParser] Non-Alpha WDL version {version} (0x{version:X}); attempting MAOF/MARE parse");

        int scanOffset = checked((int)(8 + mverSize));
        if (!TryFindChunk(data, scanOffset, FOURCC_MAOF, out int maofDataOffset, out uint maofSize))
        {
            Console.WriteLine("[WdlParser] Failed to locate MAOF chunk");
            return null;
        }

        if (maofSize < CELL_COUNT * sizeof(uint) || maofDataOffset + (CELL_COUNT * sizeof(uint)) > data.Length)
        {
            Console.WriteLine($"[WdlParser] Invalid MAOF size {maofSize} for WDL length {data.Length}");
            return null;
        }

        var maofOffsets = new uint[CELL_COUNT];
        for (int i = 0; i < CELL_COUNT; i++)
            maofOffsets[i] = ReadUInt32(data, maofDataOffset + (i * sizeof(uint)));

        var result = new WdlData { Version = version };
        int tilesLoaded = 0;

        for (int i = 0; i < CELL_COUNT; i++)
        {
            uint offset = maofOffsets[i];
            if (offset == 0) continue;

            int tileDataOffset = (int)offset;
            if (tileDataOffset < 0 || tileDataOffset >= data.Length)
                continue;

            if (tileDataOffset + MARE_CHUNK_HEADER <= data.Length)
            {
                uint mareFourCC = ReadUInt32(data, tileDataOffset);
                uint mareSize = ReadUInt32(data, tileDataOffset + sizeof(uint));
                if (mareFourCC == FOURCC_MARE)
                {
                    if (mareSize < MARE_BYTES || tileDataOffset + MARE_CHUNK_HEADER + MARE_BYTES > data.Length)
                        continue;

                    tileDataOffset += MARE_CHUNK_HEADER;
                }
                else if (tileDataOffset + MARE_BYTES > data.Length)
                {
                    continue;
                }
            }
            else
            {
                continue;
            }

            var tile = new WdlTile { HasData = true };
            float minZ = float.PositiveInfinity;
            float maxZ = float.NegativeInfinity;
            int cursor = tileDataOffset;

            int hIdx = 0;
            for (int r = 0; r < 17; r++)
            {
                for (int c = 0; c < 17; c++)
                {
                    short raw = BitConverter.ToInt16(data, cursor);
                    cursor += sizeof(short);
                    float z = raw;
                    tile.Heights[hIdx++] = z;
                    tile.Height17[r, c] = raw;
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                }
            }

            for (int r = 0; r < 16; r++)
            {
                for (int c = 0; c < 16; c++)
                {
                    short raw = BitConverter.ToInt16(data, cursor);
                    cursor += sizeof(short);
                    float z = raw;
                    tile.Heights[hIdx++] = z;
                    tile.Height16[r, c] = raw;
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                }
            }

            tile.MinZ = minZ;
            tile.MaxZ = maxZ;
            result.Tiles[i] = tile;
            tilesLoaded++;
        }

        Console.WriteLine($"[WdlParser] Parsed WDL v{version}: {tilesLoaded}/{CELL_COUNT} tiles with MARE data");
        return result;
    }

    private static bool TryFindChunk(byte[] data, int startOffset, uint fourCC, out int chunkDataOffset, out uint chunkSize)
    {
        int offset = Math.Max(0, startOffset);
        while (offset + 8 <= data.Length)
        {
            uint chunkFourCC = ReadUInt32(data, offset);
            uint size = ReadUInt32(data, offset + sizeof(uint));
            if (chunkFourCC == fourCC && offset + 8 + size <= data.Length)
            {
                chunkDataOffset = offset + 8;
                chunkSize = size;
                return true;
            }

            if (size == 0)
            {
                offset += 8;
                continue;
            }

            long nextOffset = offset + 8L + size;
            if (nextOffset > data.Length)
                break;

            offset = (int)nextOffset;
        }

        for (offset = Math.Max(0, startOffset); offset + 8 <= data.Length; offset++)
        {
            if (ReadUInt32(data, offset) != fourCC)
                continue;

            uint size = ReadUInt32(data, offset + sizeof(uint));
            if (offset + 8 + size > data.Length)
                continue;

            chunkDataOffset = offset + 8;
            chunkSize = size;
            return true;
        }

        chunkDataOffset = 0;
        chunkSize = 0;
        return false;
    }

    private static uint ReadUInt32(byte[] data, int offset)
    {
        return BitConverter.ToUInt32(data, offset);
    }
}
