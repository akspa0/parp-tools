using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

internal static class WmoGroupReaderCommon
{
    public const int MinimumGroupHeaderSize = 0x38;
    public const int PreferredHeaderSize = 0x80;
    public const int AlternateHeaderSize = 0x44;

    private static readonly HashSet<FourCC> KnownSubchunkIds =
    [
        WmoChunkIds.Mopy,
        WmoChunkIds.Movi,
        WmoChunkIds.Moin,
        WmoChunkIds.Movt,
        WmoChunkIds.Monr,
        WmoChunkIds.Motv,
        WmoChunkIds.Moba,
        WmoChunkIds.Molr,
        WmoChunkIds.Mobn,
        WmoChunkIds.Mobr,
        WmoChunkIds.Mocv,
        WmoChunkIds.Mliq,
        WmoChunkIds.Modr,
    ];

    public static (uint? Version, byte[] Mogp) ReadGroupPayload(Stream stream, string sourcePath)
    {
        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.WmoGroup)
            throw new InvalidDataException($"WMO group summary requires a WMO group file, but found {detection.Kind}.");

        ChunkSpan? mogpChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Mogp);
        if (mogpChunk is null)
            throw new InvalidDataException("WMO group summary requires a MOGP chunk.");

        byte[] mogp = ReadChunkPayload(stream, mogpChunk.Value);
        if (mogp.Length < MinimumGroupHeaderSize)
            throw new InvalidDataException($"MOGP payload is too short ({mogp.Length} bytes). Expected at least {MinimumGroupHeaderSize} bytes.");

        return (version, mogp);
    }

    public static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != WmoChunkIds.Mver)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }

    public static byte[] ReadChunkPayload(Stream stream, ChunkSpan chunk)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Header.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    public static int FindHeaderSize(byte[] mogp)
    {
        foreach (int candidate in new[] { AlternateHeaderSize, PreferredHeaderSize })
        {
            if (HasKnownSubchunkAt(mogp, candidate))
                return candidate;
        }

        for (int candidate = MinimumGroupHeaderSize; candidate <= mogp.Length - ChunkHeader.SizeInBytes; candidate += 4)
        {
            if (HasKnownSubchunkAt(mogp, candidate))
                return candidate;
        }

        return Math.Min(PreferredHeaderSize, mogp.Length);
    }

    public static IEnumerable<(ChunkHeader Header, int DataOffset)> EnumerateSubchunks(byte[] mogp, int headerSizeBytes)
    {
        int position = headerSizeBytes;
        while (position <= mogp.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(mogp.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                yield break;

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + header.Size;
            if (nextOffset > mogp.Length)
                yield break;

            yield return (header, position + ChunkHeader.SizeInBytes);
            position = checked((int)nextOffset);
        }
    }

    public static byte[]? TryReadFirstSubchunkPayload(byte[] mogp, int headerSizeBytes, FourCC chunkId)
    {
        foreach ((ChunkHeader header, int dataOffset) in EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != chunkId)
                continue;

            return mogp.AsSpan(dataOffset, checked((int)header.Size)).ToArray();
        }

        return null;
    }

    public static int CountMopyEntries(uint byteCount, uint? version)
    {
        if (byteCount == 0)
            return 0;

        int stride = InferMopyEntrySize(checked((int)byteCount), version);

        return checked((int)byteCount / stride);
    }

    public static int InferMopyEntrySize(int byteCount, uint? version)
    {
        if (byteCount <= 0)
            return 0;

        if (version >= 17)
            return 2;

        if (version is not null && version <= 16)
            return 4;

        if (byteCount % 4 == 0)
            return 4;

        return 2;
    }

    public static Vector3 ReadVector3(ReadOnlySpan<byte> bytes)
    {
        return new Vector3(
            BitConverter.ToSingle(bytes[0..4]),
            BitConverter.ToSingle(bytes[4..8]),
            BitConverter.ToSingle(bytes[8..12]));
    }

    private static bool HasKnownSubchunkAt(byte[] mogp, int offset)
    {
        if (offset > mogp.Length - ChunkHeader.SizeInBytes)
            return false;

        if (!ChunkHeaderReader.TryRead(mogp.AsSpan(offset, ChunkHeader.SizeInBytes), out ChunkHeader header))
            return false;

        return KnownSubchunkIds.Contains(header.Id)
            && (long)offset + ChunkHeader.SizeInBytes + header.Size <= mogp.Length;
    }
}
