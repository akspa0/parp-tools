using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoDoodadSetSummaryReader
{
    private const int ModsEntrySize = 32;
    private const int NameBytes = 20;

    public static WmoDoodadSetSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoDoodadSetSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO doodad-set summary requires a WMO root file, but found {detection.Kind}.");

        ChunkSpan? modsChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Mods);
        if (modsChunk is null)
            throw new InvalidDataException("WMO doodad-set summary requires a MODS chunk.");

        byte[] payload = ReadChunkPayload(stream, modsChunk.Value);
        if (payload.Length % ModsEntrySize != 0)
            throw new InvalidDataException($"MODS payload size {payload.Length} is not divisible by {ModsEntrySize}.");

        int entryCount = payload.Length / ModsEntrySize;
        int nonEmptySetCount = 0;
        int longestNameLength = 0;
        int totalDoodadRefs = 0;
        int maxStartIndex = 0;
        int maxRangeEnd = 0;

        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * ModsEntrySize;
            int length = 0;
            while (length < NameBytes && payload[offset + length] != 0)
                length++;

            uint startIndex = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 20, 4));
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 24, 4));

            longestNameLength = Math.Max(longestNameLength, length);
            if (count > 0)
                nonEmptySetCount++;

            totalDoodadRefs += checked((int)count);
            maxStartIndex = Math.Max(maxStartIndex, checked((int)startIndex));
            maxRangeEnd = Math.Max(maxRangeEnd, checked((int)(startIndex + count)));
        }

        return new WmoDoodadSetSummary(
            sourcePath,
            version,
            payload.Length,
            entryCount,
            nonEmptySetCount,
            longestNameLength,
            totalDoodadRefs,
            maxStartIndex,
            maxRangeEnd);
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
}
