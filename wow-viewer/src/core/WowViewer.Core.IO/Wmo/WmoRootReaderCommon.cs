using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

internal static class WmoRootReaderCommon
{
    public static (uint? Version, IReadOnlyList<ChunkSpan> Chunks) ReadRootChunks(Stream stream, string sourcePath)
    {
        IReadOnlyList<ChunkSpan> topLevelChunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, topLevelChunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, topLevelChunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO root summary requires a WMO root file, but found {detection.Kind}.");

        IReadOnlyList<ChunkSpan> chunks = topLevelChunks.Count > 1 && topLevelChunks[1].Header.Id == WmoChunkIds.Momo
            ? ExpandRootChunks(stream, topLevelChunks)
            : topLevelChunks;

        return (version, chunks);
    }

    public static byte[] ReadRequiredChunkPayload(Stream stream, IReadOnlyList<ChunkSpan> chunks, FourCC chunkId)
    {
        ChunkSpan? chunk = chunks.FirstOrDefault(c => c.Header.Id == chunkId);
        if (chunk is null || chunk.Value.Header.Id != chunkId)
            throw new InvalidDataException($"WMO root summary requires a {chunkId} chunk.");

        return ReadChunkPayload(stream, chunk.Value);
    }

    public static byte[]? TryReadChunkPayload(Stream stream, IReadOnlyList<ChunkSpan> chunks, FourCC chunkId)
    {
        ChunkSpan? chunk = chunks.FirstOrDefault(c => c.Header.Id == chunkId);
        if (chunk is null || chunk.Value.Header.Id != chunkId)
            return null;

        return ReadChunkPayload(stream, chunk.Value);
    }

    public static string ResolveStringAtOffset(byte[] rawBlob, uint offset)
    {
        if (rawBlob.Length == 0 || offset >= rawBlob.Length)
            return string.Empty;

        int end = Array.IndexOf(rawBlob, (byte)0, (int)offset);
        if (end < 0)
            end = rawBlob.Length;

        return System.Text.Encoding.UTF8.GetString(rawBlob, (int)offset, end - (int)offset);
    }

    public static int InferMogiEntrySize(byte[] payload, int reportedGroupCount)
    {
        if (payload.Length == 0)
            return 0;

        if (reportedGroupCount > 0)
        {
            if (payload.Length == reportedGroupCount * 32)
                return 32;
            if (payload.Length == reportedGroupCount * 40)
                return 40;
        }

        if (payload.Length % 32 == 0)
            return 32;
        if (payload.Length % 40 == 0)
            return 40;

        return 0;
    }

    public static int ReadReportedGroupCount(byte[] mohd)
    {
        return mohd.Length >= 8 ? checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(4, 4))) : 0;
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

    private static IReadOnlyList<ChunkSpan> ExpandRootChunks(Stream stream, IReadOnlyList<ChunkSpan> topLevelChunks)
    {
        List<ChunkSpan> expanded = new(topLevelChunks.Count + 8);
        foreach (ChunkSpan chunk in topLevelChunks)
        {
            if (chunk.Header.Id == WmoChunkIds.Momo)
            {
                AddMomoSubchunks(stream, chunk, expanded);
                continue;
            }

            expanded.Add(chunk);
        }

        return expanded;
    }

    private static void AddMomoSubchunks(Stream stream, ChunkSpan momoChunk, List<ChunkSpan> output)
    {
        long previousPosition = stream.Position;
        try
        {
            long offset = momoChunk.DataOffset;
            long end = momoChunk.EndOffset;
            while (offset + ChunkHeader.SizeInBytes <= end)
            {
                stream.Position = offset;
                byte[] headerBytes = new byte[ChunkHeader.SizeInBytes];
                stream.ReadExactly(headerBytes);
                FourCC id = FourCC.FromFileBytes(headerBytes.AsSpan(0, 4));
                uint size = BinaryPrimitives.ReadUInt32LittleEndian(headerBytes.AsSpan(4, 4));
                long dataOffset = offset + ChunkHeader.SizeInBytes;
                long chunkEnd = dataOffset + size;
                if (chunkEnd > end)
                    break;

                output.Add(new ChunkSpan(new ChunkHeader(id, size), offset, dataOffset));
                offset = chunkEnd;
            }
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != WmoChunkIds.Mver)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }
}
