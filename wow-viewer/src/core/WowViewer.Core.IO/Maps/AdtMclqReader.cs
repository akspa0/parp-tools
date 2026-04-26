using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Reads legacy MCLQ liquid subchunks from within an MCNK payload.
/// Based on wowdev.wiki ADT_v18 and Noggit3 implementation.
/// Each MCLQ instance covers one chunk with a 9x9 vertex grid and 8x8 tile flags.
/// </summary>
public static class AdtMclqReader
{
    private const int VertexGridSize = 9;  // 9x9 vertices
    private const int TileGridSize = 8;    // 8x8 tiles
    private const int VertexCount = 81;    // 9*9
    private const int TileCount = 64;      // 8*8

    /// <summary>
    /// Parses MCLQ data from an MCNK subchunk payload (without FourCC header).
    /// Returns null if the data is too short or unrecognizable.
    /// </summary>
    public static AdtMclqData? Read(byte[] payload)
    {
        if (payload.Length < 8)
            return null;

        int offset = 0;

        float minHeight = BitConverter.ToSingle(payload, offset); offset += 4;
        float maxHeight = BitConverter.ToSingle(payload, offset); offset += 4;

        // Need at least 81 vertices + 64 tiles
        int minimumSize = 8 + (VertexCount * 8) + TileCount + 4 + (2 * 40);
        if (payload.Length < minimumSize && payload.Length < 8 + (VertexCount * 8) + TileCount)
        {
            // Some alpha formats have shorter MCLQ (no flow vectors)
            // Allow shorter if we have at least vertices + tiles
            if (payload.Length < 8 + (VertexCount * 8) + TileCount)
                return null;
        }

        float[] heights = new float[VertexCount];
        for (int i = 0; i < VertexCount && offset + 8 <= payload.Length; i++)
        {
            // Vertex layout: depth(u8), flow0(u8), flow1(u8), filler(u8), height(f32)
            // For magma: s(u16), t(u16), height(f32) — we read generically and take height from last 4 bytes
            offset += 4; // skip depth/flow or UV
            heights[i] = BitConverter.ToSingle(payload, offset); offset += 4;
        }

        // Read 8x8 tile flags
        byte[] tileFlags = new byte[TileCount];
        for (int i = 0; i < TileCount && offset < payload.Length; i++)
        {
            tileFlags[i] = payload[offset++];
        }

        return new AdtMclqData
        {
            MinHeight = minHeight,
            MaxHeight = maxHeight,
            Heights = heights,
            TileFlags = tileFlags
        };
    }

    /// <summary>
    /// Locates MCLQ subchunk data offset within an MCNK payload.
    /// Returns the offset to the MCLQ payload (after FourCC+size), or -1 if not found.
    /// </summary>
    public static int LocateMclqOffset(ReadOnlySpan<byte> payload)
    {
        const int subchunkOffset = 0x80;
        int position = subchunkOffset;

        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = checked((int)header.Size);
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + declaredSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mclq)
            {
                if (header.Size < 8)
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = checked((int)nextOffset);
        }

        return -1;
    }
}

/// <summary>
/// Parsed MCLQ liquid data for a single terrain chunk.
/// </summary>
public sealed class AdtMclqData
{
    public float MinHeight { get; init; }
    public float MaxHeight { get; init; }
    public float[] Heights { get; init; } = Array.Empty<float>(); // 81 entries (9x9)
    public byte[] TileFlags { get; init; } = Array.Empty<byte>(); // 64 entries (8x8)

    /// <summary>Liquid type from lower 4 bits of the first tile flag (best guess).</summary>
    public int LiquidType => TileFlags.Length > 0 ? TileFlags[0] & 0x0F : 0;
}
