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
        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO root summary requires a WMO root file, but found {detection.Kind}.");

        return (version, chunks);
    }

    public static byte[] ReadRequiredChunkPayload(Stream stream, IReadOnlyList<ChunkSpan> chunks, FourCC chunkId)
    {
        ChunkSpan? chunk = chunks.FirstOrDefault(c => c.Header.Id == chunkId);
        if (chunk is null)
            throw new InvalidDataException($"WMO root summary requires a {chunkId} chunk.");

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

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != WmoChunkIds.Mver)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }
}
