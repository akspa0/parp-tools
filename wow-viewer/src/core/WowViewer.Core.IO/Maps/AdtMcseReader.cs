using System.Buffers.Binary;
using System.Numerics;
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
    // CWSoundEmitter in the 0.5.3 client is 0x4C bytes in memory, but the
    // on-disk MCSE record copied by CMapChunk::Create is 0x34 bytes.
    public const int Alpha053EntrySize = 0x34;

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
                Emitters = ReadAlphaEmitters(payload, entryCount, entrySize),
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

        List<AdtMcseEmitter> emitters = new(entryCount);
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            ReadOnlySpan<byte> entry = payload.AsSpan(entryIndex * entrySize, entrySize);
            emitters.Add(new AdtMcseEmitter(
                SoundPointId: 0,
                SoundNameId: unchecked((uint)entryIds[entryIndex]),
                Position: new Vector3(
                    positionXyz[entryIndex, 0],
                    positionXyz[entryIndex, 1],
                    positionXyz[entryIndex, 2]),
                MinDistance: 0f,
                MaxDistance: 0f,
                CutoffDistance: 0f,
                StartTime: 0,
                EndTime: 0,
                Mode: 0,
                RawEntry: entry.ToArray()));
        }

        return new AdtMcseData
        {
            EntryCount = entryCount,
            EntrySize = entrySize,
            EntryBytes = entryBytes,
            EntryIds = entryIds,
            PositionXyz = positionXyz,
            Emitters = emitters,
        };
    }

    /// <summary>
    /// Reads the Alpha 0.5.3 MCNK representation. Alpha stores MCSE as raw
    /// entries at an offset relative to the 128-byte MCNK header; there is no
    /// MCSE FourCC/size wrapper. CMapChunk::Create reads the offset/count from
    /// the same header fields and copies 0x34-byte records into CWSoundEmitter.
    /// </summary>
    public static AdtMcseData ReadAlpha053Mcnk(ReadOnlySpan<byte> mcnkPayload)
    {
        const int headerSize = 0x80;
        const int offsetField = 0x5C;
        const int countField = 0x60;

        if (mcnkPayload.Length < headerSize)
            return new AdtMcseData();

        int relativeOffset = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mcnkPayload.Slice(offsetField, 4)));
        int declaredCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mcnkPayload.Slice(countField, 4)));
        if (relativeOffset <= 0 || declaredCount <= 0)
            return new AdtMcseData();

        long byteCount = (long)declaredCount * Alpha053EntrySize;
        long start = (long)headerSize + relativeOffset;
        if (start < 0 || byteCount > int.MaxValue || start + byteCount > mcnkPayload.Length)
            return new AdtMcseData();

        return Read(mcnkPayload.Slice((int)start, (int)byteCount).ToArray(), declaredCount);
    }

    private static List<AdtMcseEmitter> ReadAlphaEmitters(byte[] payload, int entryCount, int entrySize)
    {
        if (entrySize != Alpha053EntrySize)
            return [];

        List<AdtMcseEmitter> emitters = new(entryCount);
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            ReadOnlySpan<byte> entry = payload.AsSpan(entryIndex * entrySize, entrySize);
            emitters.Add(new AdtMcseEmitter(
                SoundPointId: BinaryPrimitives.ReadUInt32LittleEndian(entry.Slice(0x00, 4)),
                SoundNameId: BinaryPrimitives.ReadUInt32LittleEndian(entry.Slice(0x04, 4)),
                Position: new Vector3(
                    BitConverter.ToSingle(entry.Slice(0x08, 4)),
                    BitConverter.ToSingle(entry.Slice(0x0C, 4)),
                    BitConverter.ToSingle(entry.Slice(0x10, 4))),
                MinDistance: BitConverter.ToSingle(entry.Slice(0x14, 4)),
                MaxDistance: BitConverter.ToSingle(entry.Slice(0x18, 4)),
                CutoffDistance: BitConverter.ToSingle(entry.Slice(0x1C, 4)),
                StartTime: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x20, 2)),
                EndTime: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x22, 2)),
                Mode: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x24, 2)),
                RawEntry: entry.ToArray(),
                LoopCountMin: entry[0x26],
                LoopCountMax: entry[0x27],
                GroupSilenceMin: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x28, 2)),
                GroupSilenceMax: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x2A, 2)),
                PlayInstancesMin: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x2C, 2)),
                PlayInstancesMax: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x2E, 2)),
                InterSoundGapMin: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x30, 2)),
                InterSoundGapMax: BinaryPrimitives.ReadUInt16LittleEndian(entry.Slice(0x32, 2))));
        }

        return emitters;
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

    public IReadOnlyList<AdtMcseEmitter> Emitters { get; init; } = Array.Empty<AdtMcseEmitter>();
}

public sealed record AdtMcseEmitter(
    uint SoundPointId,
    uint SoundNameId,
    Vector3 Position,
    float MinDistance,
    float MaxDistance,
    float CutoffDistance,
    uint StartTime,
    uint EndTime,
    uint Mode,
    byte[] RawEntry,
    byte LoopCountMin = 0,
    byte LoopCountMax = 0,
    ushort GroupSilenceMin = 0,
    ushort GroupSilenceMax = 0,
    ushort PlayInstancesMin = 0,
    ushort PlayInstancesMax = 0,
    ushort InterSoundGapMin = 0,
    ushort InterSoundGapMax = 0);
