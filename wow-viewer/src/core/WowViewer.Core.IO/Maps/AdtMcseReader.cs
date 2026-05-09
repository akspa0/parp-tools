using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Reads MCSE (sound emitter) subchunks from within MCNK payloads.
/// The common root-ADT emitter payload uses 0x1C-byte entries; older Alpha-era
/// emitters can use larger records. This reader preserves exact entry bytes for
/// any supported fixed stride and only decodes fields that are unambiguous.
/// </summary>
public static class AdtMcseReader
{
    public const int StandardEntrySize = 0x1C;
    public const int Alpha053EntrySize = 76;

    public static AdtMcseData Read(byte[] payload, int declaredEmitterCount)
    {
        if (payload.Length < StandardEntrySize)
            return new AdtMcseData();

        if (!TryResolveEntryLayout(payload.Length, declaredEmitterCount, out int entryCount, out int entrySize))
            return new AdtMcseData();

        byte[,] entryBytes = new byte[entryCount, entrySize];
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            for (int byteIndex = 0; byteIndex < entrySize; byteIndex++)
                entryBytes[entryIndex, byteIndex] = payload[(entryIndex * entrySize) + byteIndex];
        }

        if (entrySize != StandardEntrySize)
        {
            return new AdtMcseData
            {
                EntryCount = entryCount,
                EntrySize = entrySize,
                EntryBytes = entryBytes,
            };
        }

        int[] entryIds = new int[entryCount];
        float[,] positionXyz = new float[entryCount, 3];
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            ReadOnlySpan<byte> entry = payload.AsSpan(entryIndex * entrySize, entrySize);
            entryIds[entryIndex] = BinaryPrimitives.ReadInt32LittleEndian(entry.Slice(0x00, 4));
            positionXyz[entryIndex, 0] = BitConverter.ToSingle(entry.Slice(0x04, 4));
            positionXyz[entryIndex, 1] = BitConverter.ToSingle(entry.Slice(0x08, 4));
            positionXyz[entryIndex, 2] = BitConverter.ToSingle(entry.Slice(0x0C, 4));
        }

        return new AdtMcseData
        {
            EntryCount = entryCount,
            EntrySize = entrySize,
            EntryBytes = entryBytes,
            EntryIds = entryIds,
            PositionXyz = positionXyz,
        };
    }

    public static bool TryLocateMcsePayload(ReadOnlySpan<byte> payload, out int payloadOffset, out int payloadSize)
    {
        payloadOffset = -1;
        payloadSize = 0;

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

            if (header.Id == AdtChunkIds.Mcse)
            {
                payloadOffset = position + ChunkHeader.SizeInBytes;
                payloadSize = declaredSize;
                return true;
            }

            position = unchecked((int)nextOffset);
            if (position < 0)
                break;
        }

        return false;
    }

    private static bool TryResolveEntryLayout(int payloadLength, int declaredEmitterCount, out int entryCount, out int entrySize)
    {
        entryCount = 0;
        entrySize = 0;

        if (declaredEmitterCount > 0 && payloadLength % declaredEmitterCount == 0)
        {
            int declaredEntrySize = payloadLength / declaredEmitterCount;
            if (declaredEntrySize is StandardEntrySize or Alpha053EntrySize)
            {
                entryCount = declaredEmitterCount;
                entrySize = declaredEntrySize;
                return true;
            }
        }

        if (payloadLength % StandardEntrySize == 0)
        {
            entryCount = payloadLength / StandardEntrySize;
            entrySize = StandardEntrySize;
            return entryCount > 0;
        }

        if (payloadLength % Alpha053EntrySize == 0)
        {
            entryCount = payloadLength / Alpha053EntrySize;
            entrySize = Alpha053EntrySize;
            return entryCount > 0;
        }

        return false;
    }
}

public sealed class AdtMcseData
{
    public int EntryCount { get; init; }

    public int EntrySize { get; init; }

    public byte[,]? EntryBytes { get; init; }

    public int[]? EntryIds { get; init; }

    public float[,]? PositionXyz { get; init; }
}