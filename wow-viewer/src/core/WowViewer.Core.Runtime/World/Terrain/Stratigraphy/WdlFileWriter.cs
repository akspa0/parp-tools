using System.Buffers.Binary;

namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// Container holding a single tile's WDL low-resolution height data.
/// </summary>
public sealed class WdlTileData
{
    public short[,] Height17 { get; } = new short[17, 17];
    public short[,] Height16 { get; } = new short[16, 16];
    public ushort[] HoleMask16 { get; } = new ushort[16];
    public bool HasData { get; set; }
}

/// <summary>
/// Serializes Blizzard-standard .wdl (World Distance Low-res) binary files containing MVER, MAOF, and MARE/MAHO chunks.
/// </summary>
public static class WdlFileWriter
{
    private const int TotalTiles = 64 * 64;
    private const int Height17Count = 17 * 17; // 289
    private const int Height16Count = 16 * 16; // 256
    private const int MarePayloadSize = (Height17Count + Height16Count) * sizeof(short); // 1090 bytes
    private const int MahoPayloadSize = 16 * sizeof(ushort); // 32 bytes

    /// <summary>
    /// Writes a complete .wdl binary file containing the specified 64x64 tile height data.
    /// </summary>
    public static byte[] Write(IReadOnlyDictionary<(int tileX, int tileY), WdlTileData> tiles)
    {
        ArgumentNullException.ThrowIfNull(tiles);

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // 1. MVER chunk
        WriteChunkHeader(writer, "MVER", 4);
        writer.Write((uint)18);

        // 2. MWMO chunk (empty)
        WriteChunkHeader(writer, "MWMO", 0);

        // 3. MWID chunk (empty)
        WriteChunkHeader(writer, "MWID", 0);

        // 4. MODF chunk (empty)
        WriteChunkHeader(writer, "MODF", 0);

        // 5. MAOF chunk (16,384 bytes = 4096 uint offsets)
        long maofHeaderPos = ms.Position;
        WriteChunkHeader(writer, "MAOF", TotalTiles * sizeof(uint));
        long maofDataPos = ms.Position;

        // Reserve space for 4096 offsets
        for (int i = 0; i < TotalTiles; i++)
            writer.Write((uint)0);

        var tileOffsets = new uint[TotalTiles];

        // 6. Write MARE and MAHO for each present tile
        foreach (var (coord, tile) in tiles.OrderBy(kv => kv.Key.tileX * 64 + kv.Key.tileY))
        {
            if (!tile.HasData)
                continue;

            int tileIndex = coord.tileX * 64 + coord.tileY;
            if (tileIndex < 0 || tileIndex >= TotalTiles)
                continue;

            long tileStartOffset = ms.Position;
            tileOffsets[tileIndex] = (uint)tileStartOffset;

            // Write MARE (1090 bytes)
            WriteChunkHeader(writer, "MARE", MarePayloadSize);
            for (int y = 0; y < 17; y++)
                for (int x = 0; x < 17; x++)
                    writer.Write(tile.Height17[y, x]);

            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                    writer.Write(tile.Height16[y, x]);

            // Write MAHO (32 bytes)
            WriteChunkHeader(writer, "MAHO", MahoPayloadSize);
            for (int i = 0; i < 16; i++)
                writer.Write(tile.HoleMask16[i]);
        }

        // 7. Rewind and fill MAOF table with actual file offsets
        long endPos = ms.Position;
        ms.Position = maofDataPos;
        for (int i = 0; i < TotalTiles; i++)
            writer.Write(tileOffsets[i]);

        ms.Position = endPos;
        return ms.ToArray();
    }

    /// <summary>
    /// Builds a WdlTileData object by downsampling 256 chunk heights or 257x257 lattice.
    /// </summary>
    public static WdlTileData FromLattice257(float[,] height257)
    {
        ArgumentNullException.ThrowIfNull(height257);
        var tile = new WdlTileData { HasData = true };

        int dim = height257.GetLength(0);
        float stepX = (dim - 1) / 16f;
        float stepY = (dim - 1) / 16f;

        // 17x17 grid
        for (int y = 0; y < 17; y++)
        {
            int sy = Math.Clamp((int)MathF.Round(y * stepY), 0, dim - 1);
            for (int x = 0; x < 17; x++)
            {
                int sx = Math.Clamp((int)MathF.Round(x * stepX), 0, dim - 1);
                tile.Height17[y, x] = (short)Math.Clamp(MathF.Round(height257[sy, sx]), short.MinValue, short.MaxValue);
            }
        }

        // 16x16 chunk centers
        for (int y = 0; y < 16; y++)
        {
            int sy = Math.Clamp((int)MathF.Round((y + 0.5f) * stepY), 0, dim - 1);
            for (int x = 0; x < 16; x++)
            {
                int sx = Math.Clamp((int)MathF.Round((x + 0.5f) * stepX), 0, dim - 1);
                tile.Height16[y, x] = (short)Math.Clamp(MathF.Round(height257[sy, sx]), short.MinValue, short.MaxValue);
            }
        }

        return tile;
    }

    private static void WriteChunkHeader(BinaryWriter writer, string fourcc, int payloadSize)
    {
        Span<byte> tag = stackalloc byte[4];
        for (int i = 0; i < 4; i++)
            tag[i] = (byte)(i < fourcc.Length ? fourcc[3 - i] : ' '); // Little-endian FourCC

        writer.Write(tag);
        writer.Write((uint)payloadSize);
    }
}
