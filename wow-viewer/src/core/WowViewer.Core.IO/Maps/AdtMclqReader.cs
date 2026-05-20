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
    private const int RootMcnkHeaderSize = 128;
    private const int MclqOffsetField = 0x60;
    private const int MclqSizeField = 0x64;
    private const int VertexGridSize = 9;  // 9x9 vertices
    private const int TileGridSize = 8;    // 8x8 tiles
    private const int VertexCount = 81;    // 9*9
    private const int TileCount = 64;      // 8*8
    private const uint LiquidFlagMask = 0x3Cu;

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
    /// Parses legacy MCLQ liquid from a root-ADT MCNK payload using the same
    /// subchunk scan + header-offset fallback path as the working legacy reader.
    /// </summary>
    public static AdtMclqData? Read(byte[] mcnkPayload, AdtFormatProfile profile)
    {
        ArgumentNullException.ThrowIfNull(mcnkPayload);
        ArgumentNullException.ThrowIfNull(profile);

        if (mcnkPayload.Length < RootMcnkHeaderSize)
            return null;

        uint mcnkFlags = BinaryPrimitives.ReadUInt32LittleEndian(mcnkPayload.AsSpan(0x00, sizeof(uint)));
        if ((mcnkFlags & LiquidFlagMask) == 0)
            return null;

        byte[]? payload = TryExtractInlineMclqPayload(mcnkPayload)
            ?? TryExtractHeaderRelativeMclqPayload(mcnkPayload, mcnkFlags, profile);

        if (payload is null || payload.Length < 8)
            return null;

        return ParseLegacyPayload(payload, mcnkFlags, profile.MclqLayerStride, profile.MclqTileFlagsOffset);
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

            int declaredSize = unchecked((int)header.Size);
            if (declaredSize < 0 || declaredSize > payload.Length)
                break;
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + declaredSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mclq)
            {
                if (header.Size < 8)
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = unchecked((int)nextOffset);
            if (position < 0) break;
        }

        return -1;
    }

    private static byte[]? TryExtractInlineMclqPayload(byte[] mcnkPayload)
    {
        const int subchunkOffset = 0x80;
        int position = subchunkOffset;

        while (position <= mcnkPayload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(mcnkPayload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = unchecked((int)header.Size);
            if (declaredSize < 0)
                break;

            int dataOffset = position + ChunkHeader.SizeInBytes;
            long nextOffset = (long)dataOffset + declaredSize;
            if (nextOffset > mcnkPayload.Length)
                break;

            if (header.Id == AdtChunkIds.Mclq)
            {
                byte[] payload = new byte[declaredSize];
                Buffer.BlockCopy(mcnkPayload, dataOffset, payload, 0, declaredSize);
                return payload;
            }

            position = checked((int)nextOffset);
        }

        return null;
    }

    private static byte[]? TryExtractHeaderRelativeMclqPayload(byte[] mcnkPayload, uint mcnkFlags, AdtFormatProfile profile)
    {
        if (mcnkPayload.Length < MclqSizeField + sizeof(uint))
            return null;

        int mclqRelativeOffset = BinaryPrimitives.ReadInt32LittleEndian(mcnkPayload.AsSpan(MclqOffsetField, sizeof(int)));
        if (mclqRelativeOffset <= 8)
            return null;

        int mclqHeaderOffset = mclqRelativeOffset - ChunkHeader.SizeInBytes;
        if (mclqHeaderOffset < 0 || mclqHeaderOffset >= mcnkPayload.Length)
            return null;

        int headerDeclaredPayloadSize = BinaryPrimitives.ReadInt32LittleEndian(mcnkPayload.AsSpan(MclqSizeField, sizeof(int)));

        if (mclqHeaderOffset <= mcnkPayload.Length - ChunkHeader.SizeInBytes
            && ChunkHeaderReader.TryRead(mcnkPayload.AsSpan(mclqHeaderOffset, ChunkHeader.SizeInBytes), out ChunkHeader header)
            && header.Id == AdtChunkIds.Mclq)
        {
            int dataOffset = mclqHeaderOffset + ChunkHeader.SizeInBytes;
            int payloadSize = ResolveLegacyPayloadSize(mcnkPayload.Length, dataOffset, checked((int)header.Size), headerDeclaredPayloadSize, mcnkFlags, profile);
            if (payloadSize <= 0 || dataOffset + payloadSize > mcnkPayload.Length)
                return null;

            byte[] payload = new byte[payloadSize];
            Buffer.BlockCopy(mcnkPayload, dataOffset, payload, 0, payloadSize);
            return payload;
        }

        int headerlessPayloadSize = ResolveLegacyPayloadSize(mcnkPayload.Length, mclqHeaderOffset, 0, headerDeclaredPayloadSize, mcnkFlags, profile);
        if (headerlessPayloadSize <= 0 || mclqHeaderOffset + headerlessPayloadSize > mcnkPayload.Length)
            return null;

        byte[] headerlessPayload = new byte[headerlessPayloadSize];
        Buffer.BlockCopy(mcnkPayload, mclqHeaderOffset, headerlessPayload, 0, headerlessPayloadSize);
        return StripMclqChunkHeaderIfPresent(headerlessPayload);
    }

    private static int ResolveLegacyPayloadSize(int bufferLength, int dataOffset, int chunkDeclaredSize, int headerDeclaredPayloadSize, uint mcnkFlags, AdtFormatProfile profile)
    {
        if (chunkDeclaredSize > 0 && dataOffset + chunkDeclaredSize <= bufferLength)
            return chunkDeclaredSize;

        if (headerDeclaredPayloadSize > 0 && dataOffset + headerDeclaredPayloadSize <= bufferLength)
            return headerDeclaredPayloadSize;

        int packedPayloadSize = CountLiquidInstances(mcnkFlags) * profile.MclqLayerStride;
        if (packedPayloadSize > 0 && dataOffset + packedPayloadSize <= bufferLength)
            return packedPayloadSize;

        int remainder = bufferLength - dataOffset;
        return remainder >= 8 ? remainder : 0;
    }

    private static AdtMclqData? ParseLegacyPayload(byte[] payload, uint mcnkFlags, int layerStride, int tileFlagsOffset)
    {
        byte[] mclqPayload = StripMclqChunkHeaderIfPresent(payload);
        if (mclqPayload.Length < 8)
            return null;

        uint[] liquidBits = [0x04, 0x08, 0x10, 0x20];
        int[] liquidTypes = [1, 2, 3, 4];

        int instanceCount = CountLiquidInstances(mcnkFlags);
        if (instanceCount == 0)
            return null;

        int packedPayloadSize = instanceCount * layerStride;
        bool usePacked = mclqPayload.Length >= packedPayloadSize && packedPayloadSize > 0;
        if (!usePacked && mclqPayload.Length < 720)
            return null;

        int offset = 0;
        AdtMclqData? result = null;

        for (int bitIndex = 0; bitIndex < liquidBits.Length; bitIndex++)
        {
            if ((mcnkFlags & liquidBits[bitIndex]) == 0)
                continue;

            if (offset + 8 > mclqPayload.Length)
                break;

            float minHeight = BitConverter.ToSingle(mclqPayload, offset + 0);
            float maxHeight = BitConverter.ToSingle(mclqPayload, offset + 4);
            if (float.IsNaN(minHeight) || float.IsNaN(maxHeight))
            {
                if (usePacked)
                {
                    offset += layerStride;
                    continue;
                }

                break;
            }

            float[] heights = new float[VertexCount];
            if (offset + 8 + (VertexCount * 8) <= mclqPayload.Length)
            {
                for (int vertexIndex = 0; vertexIndex < VertexCount; vertexIndex++)
                {
                    int heightOffset = offset + 8 + (vertexIndex * 8) + 4;
                    float height = BitConverter.ToSingle(mclqPayload, heightOffset);
                    heights[vertexIndex] = float.IsNaN(height) || MathF.Abs(height) > 50000f
                        ? maxHeight
                        : height;
                }
            }
            else
            {
                Array.Fill(heights, maxHeight);
            }

            int tileFlagsDataOffset = usePacked ? offset + tileFlagsOffset : offset + 8 + (VertexCount * 8);
            byte[]? tileFlags = null;
            if (tileFlagsDataOffset + TileCount <= mclqPayload.Length)
            {
                tileFlags = new byte[TileCount];
                Buffer.BlockCopy(mclqPayload, tileFlagsDataOffset, tileFlags, 0, TileCount);
            }

            bool anyVisible = tileFlags is null;
            if (tileFlags is not null)
            {
                for (int tileIndex = 0; tileIndex < TileCount; tileIndex++)
                {
                    if ((tileFlags[tileIndex] & 0x0F) != 0x0F)
                    {
                        anyVisible = true;
                        break;
                    }
                }
            }

            if (anyVisible && result is null)
            {
                result = new AdtMclqData
                {
                    MinHeight = minHeight,
                    MaxHeight = maxHeight,
                    Heights = heights,
                    TileFlags = tileFlags ?? Array.Empty<byte>(),
                    LiquidType = liquidTypes[bitIndex]
                };
            }

            if (usePacked)
                offset += layerStride;
            else
                break;
        }

        return result;
    }

    private static int CountLiquidInstances(uint mcnkFlags)
    {
        int count = 0;
        if ((mcnkFlags & 0x04) != 0) count++;
        if ((mcnkFlags & 0x08) != 0) count++;
        if ((mcnkFlags & 0x10) != 0) count++;
        if ((mcnkFlags & 0x20) != 0) count++;
        return count;
    }

    private static byte[] StripMclqChunkHeaderIfPresent(byte[] mclqData)
    {
        if (mclqData.Length < ChunkHeader.SizeInBytes)
            return mclqData;

        if (!ChunkHeaderReader.TryRead(mclqData.AsSpan(0, ChunkHeader.SizeInBytes), out ChunkHeader header)
            || header.Id != AdtChunkIds.Mclq)
            return mclqData;

        int declaredSize = checked((int)header.Size);
        if (declaredSize <= 0 || ChunkHeader.SizeInBytes + declaredSize > mclqData.Length)
            return mclqData;

        byte[] payload = new byte[declaredSize];
        Buffer.BlockCopy(mclqData, ChunkHeader.SizeInBytes, payload, 0, declaredSize);
        return payload;
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
    public int LiquidType { get; init; }
}
