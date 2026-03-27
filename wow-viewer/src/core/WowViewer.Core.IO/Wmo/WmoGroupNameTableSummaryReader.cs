using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupNameTableSummaryReader
{
    public static WmoGroupNameTableSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupNameTableSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO group-name summary requires a WMO root file, but found {detection.Kind}.");

        ChunkSpan? mognChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == FourCC.FromString("MOGN"));
        if (mognChunk is null || mognChunk.Value.Header.Id != WmoChunkIds.Mogn)
            throw new InvalidDataException("WMO group-name summary requires a MOGN chunk.");

        byte[] payload = ReadChunkPayload(stream, mognChunk.Value);
        int count = 0;
        int longest = 0;
        int maxOffset = 0;
        int currentStart = -1;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] == 0)
            {
                if (currentStart >= 0)
                {
                    int length = index - currentStart;
                    count++;
                    longest = Math.Max(longest, length);
                    maxOffset = Math.Max(maxOffset, currentStart);
                    currentStart = -1;
                }

                continue;
            }

            if (currentStart < 0)
                currentStart = index;
        }

        if (currentStart >= 0)
        {
            count++;
            longest = Math.Max(longest, payload.Length - currentStart);
            maxOffset = Math.Max(maxOffset, currentStart);
        }

        return new WmoGroupNameTableSummary(sourcePath, version, payload.Length, count, longest, maxOffset);
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
