using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoTextureTableSummaryReader
{
    public static WmoTextureTableSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoTextureTableSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO texture-table summary requires a WMO root file, but found {detection.Kind}.");

        ChunkSpan? motxChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Motx);
        if (motxChunk is null)
            throw new InvalidDataException("WMO texture-table summary requires a MOTX chunk.");

        byte[] payload = ReadChunkPayload(stream, motxChunk.Value);
        int textureCount = 0;
        int longestEntryLength = 0;
        int maxOffset = 0;
        HashSet<string> extensions = [];
        int blpEntryCount = 0;

        int currentStart = -1;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] == 0)
            {
                if (currentStart >= 0)
                {
                    int length = index - currentStart;
                    string entry = System.Text.Encoding.UTF8.GetString(payload, currentStart, length);
                    textureCount++;
                    longestEntryLength = Math.Max(longestEntryLength, length);
                    maxOffset = Math.Max(maxOffset, currentStart);

                    string extension = Path.GetExtension(entry).ToLowerInvariant();
                    if (!string.IsNullOrEmpty(extension))
                        extensions.Add(extension);

                    if (string.Equals(extension, ".blp", StringComparison.OrdinalIgnoreCase))
                        blpEntryCount++;

                    currentStart = -1;
                }

                continue;
            }

            if (currentStart < 0)
                currentStart = index;
        }

        if (currentStart >= 0)
        {
            int length = payload.Length - currentStart;
            string entry = System.Text.Encoding.UTF8.GetString(payload, currentStart, length);
            textureCount++;
            longestEntryLength = Math.Max(longestEntryLength, length);
            maxOffset = Math.Max(maxOffset, currentStart);

            string extension = Path.GetExtension(entry).ToLowerInvariant();
            if (!string.IsNullOrEmpty(extension))
                extensions.Add(extension);

            if (string.Equals(extension, ".blp", StringComparison.OrdinalIgnoreCase))
                blpEntryCount++;
        }

        return new WmoTextureTableSummary(
            sourcePath,
            version,
            payload.Length,
            textureCount,
            longestEntryLength,
            maxOffset,
            distinctExtensionCount: extensions.Count,
            blpEntryCount);
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
