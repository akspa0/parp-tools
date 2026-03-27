using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupSummaryReader
{
    private const int MinimumGroupHeaderSize = 0x38;
    private const int PreferredHeaderSize = 0x80;
    private const int AlternateHeaderSize = 0x44;
    private const int VertexStride = 12;
    private const int IndexStride = 2;
    private const int NormalStride = 12;
    private const int UvStride = 8;
    private const int BatchStride = 24;
    private const int VertexColorStride = 4;
    private const int DoodadRefStride = 2;

    private static readonly HashSet<FourCC> KnownSubchunkIds =
    [
        WmoChunkIds.Mopy,
        WmoChunkIds.Movi,
        WmoChunkIds.Moin,
        WmoChunkIds.Movt,
        WmoChunkIds.Monr,
        WmoChunkIds.Motv,
        WmoChunkIds.Moba,
        WmoChunkIds.Mocv,
        WmoChunkIds.Mliq,
        WmoChunkIds.Modr,
    ];

    public static WmoGroupSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

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

        int headerSizeBytes = FindHeaderSize(mogp);
        int faceMaterialCount = 0;
        int vertexCount = 0;
        int indexCount = 0;
        int normalCount = 0;
        int primaryUvCount = 0;
        int additionalUvSetCount = 0;
        int batchCount = 0;
        int vertexColorCount = 0;
        int doodadRefCount = 0;
        bool hasLiquid = false;

        int position = headerSizeBytes;
        while (position <= mogp.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(mogp.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + header.Size;
            if (nextOffset > mogp.Length)
                break;

            if (header.Id == WmoChunkIds.Mopy)
                faceMaterialCount += CountMopyEntries(header.Size, version);
            else if (header.Id == WmoChunkIds.Movi || header.Id == WmoChunkIds.Moin)
                indexCount += checked((int)header.Size / IndexStride);
            else if (header.Id == WmoChunkIds.Movt)
                vertexCount += checked((int)header.Size / VertexStride);
            else if (header.Id == WmoChunkIds.Monr)
                normalCount += checked((int)header.Size / NormalStride);
            else if (header.Id == WmoChunkIds.Motv)
            {
                int uvCount = checked((int)header.Size / UvStride);
                if (primaryUvCount == 0)
                    primaryUvCount = uvCount;
                else
                    additionalUvSetCount++;
            }
            else if (header.Id == WmoChunkIds.Moba)
                batchCount += checked((int)header.Size / BatchStride);
            else if (header.Id == WmoChunkIds.Mocv)
                vertexColorCount += checked((int)header.Size / VertexColorStride);
            else if (header.Id == WmoChunkIds.Modr)
                doodadRefCount += checked((int)header.Size / DoodadRefStride);
            else if (header.Id == WmoChunkIds.Mliq)
                hasLiquid = true;

            position = checked((int)nextOffset);
        }

        return new WmoGroupSummary(
            sourcePath,
            version,
            headerSizeBytes,
            nameOffset: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x00, 4)),
            descriptiveNameOffset: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x04, 4)),
            flags: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x08, 4)),
            boundsMin: ReadVector3(mogp.AsSpan(0x0C, 12)),
            boundsMax: ReadVector3(mogp.AsSpan(0x18, 12)),
            portalStart: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x24, 2)),
            portalCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x26, 2)),
            transparentBatchCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x28, 2)),
            interiorBatchCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x2A, 2)),
            exteriorBatchCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x2C, 2)),
            groupLiquid: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x34, 4)),
            faceMaterialCount,
            vertexCount,
            indexCount,
            normalCount,
            primaryUvCount,
            additionalUvSetCount,
            batchCount,
            vertexColorCount,
            doodadRefCount,
            hasLiquid);
    }

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != WmoChunkIds.Mver)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }

    private static byte[] ReadChunkPayload(Stream stream, ChunkSpan chunk)
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

    private static int FindHeaderSize(byte[] mogp)
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

    private static bool HasKnownSubchunkAt(byte[] mogp, int offset)
    {
        if (offset > mogp.Length - ChunkHeader.SizeInBytes)
            return false;

        if (!ChunkHeaderReader.TryRead(mogp.AsSpan(offset, ChunkHeader.SizeInBytes), out ChunkHeader header))
            return false;

        return KnownSubchunkIds.Contains(header.Id)
            && (long)offset + ChunkHeader.SizeInBytes + header.Size <= mogp.Length;
    }

    private static int CountMopyEntries(uint byteCount, uint? version)
    {
        if (byteCount == 0)
            return 0;

        int stride;
        if (version >= 17)
            stride = 2;
        else if (version is not null && version <= 16)
            stride = 4;
        else if (byteCount % 4 == 0)
            stride = 4;
        else
            stride = 2;

        return checked((int)byteCount / stride);
    }

    private static Vector3 ReadVector3(ReadOnlySpan<byte> bytes)
    {
        return new Vector3(
            BitConverter.ToSingle(bytes[0..4]),
            BitConverter.ToSingle(bytes[4..8]),
            BitConverter.ToSingle(bytes[8..12]));
    }
}
