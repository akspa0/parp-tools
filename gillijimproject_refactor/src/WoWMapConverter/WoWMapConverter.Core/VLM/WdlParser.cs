using System;
using System.IO;
using System.Text;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Parses WDL (World Detail Level) files for WoW Alpha 0.5.3.
/// Based on Ghidra analysis of CMap::LoadWdl (0x0067FA20).
/// 
/// File layout:
///   MVER chunk → version uint32 (must be 0x12)
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

        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);

        // 1) Read MVER chunk
        uint mverFourCC = br.ReadUInt32();
        uint mverSize = br.ReadUInt32();
        if (mverFourCC != FOURCC_MVER)
        {
            Console.WriteLine($"[WdlParser] Expected MVER (0x{FOURCC_MVER:X8}), got 0x{mverFourCC:X8}");
            return null;
        }

        int version = br.ReadInt32();
        if (version != VERSION_EXPECTED)
        {
            Console.WriteLine($"[WdlParser] Unsupported WDL version {version} (0x{version:X}), expected {VERSION_EXPECTED} (0x{VERSION_EXPECTED:X})");
            return null;
        }

        // Skip any remaining MVER data
        ms.Position = 8 + mverSize;

        // 2) Read MAOF chunk
        uint maofFourCC = br.ReadUInt32();
        uint maofSize = br.ReadUInt32();
        if (maofFourCC != FOURCC_MAOF)
        {
            Console.WriteLine($"[WdlParser] Expected MAOF (0x{FOURCC_MAOF:X8}), got 0x{maofFourCC:X8}");
            return null;
        }

        // Client reads fixed 0x4000 bytes (4096 × uint32)
        var maofOffsets = new uint[CELL_COUNT];
        for (int i = 0; i < CELL_COUNT; i++)
            maofOffsets[i] = br.ReadUInt32();

        var result = new WdlData { Version = version };
        int tilesLoaded = 0;

        // 3) For each nonzero MAOF offset, seek and read MARE chunk
        for (int i = 0; i < CELL_COUNT; i++)
        {
            uint offset = maofOffsets[i];
            if (offset == 0) continue;
            if (offset + MARE_CHUNK_HEADER + MARE_BYTES > data.Length) continue;

            ms.Position = offset;

            // Read MARE chunk header
            uint mareFourCC = br.ReadUInt32();
            uint mareSize = br.ReadUInt32();

            if (mareFourCC != FOURCC_MARE)
                continue; // Robust fallback: skip bad entries

            // Read 545 int16 heights, convert to float, compute bounds
            var tile = new WdlTile { HasData = true };
            float minZ = float.PositiveInfinity;
            float maxZ = float.NegativeInfinity;

            // First 289 values: 17×17 outer lattice
            int hIdx = 0;
            for (int r = 0; r < 17; r++)
            {
                for (int c = 0; c < 17; c++)
                {
                    short raw = br.ReadInt16();
                    float z = raw;
                    tile.Heights[hIdx++] = z;
                    tile.Height17[r, c] = raw;
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                }
            }

            // Next 256 values: 16×16 inner lattice (cell centers)
            for (int r = 0; r < 16; r++)
            {
                for (int c = 0; c < 16; c++)
                {
                    short raw = br.ReadInt16();
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
}
