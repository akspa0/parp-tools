using WowViewer.Core.Files;
using WowViewer.Core.Maps;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Maps;

public static class MapFileSummaryReader
{
    public static MapFileSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MapFileSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunkSpans = ChunkedFileReader.ReadTopLevelChunks(stream);
        MapChunkLocation[] chunks = chunkSpans
            .Select(static chunk => new MapChunkLocation(chunk.Header.Id, chunk.Header.Size, chunk.HeaderOffset, chunk.DataOffset))
            .ToArray();

        uint? version = TryReadVersion(stream, chunkSpans);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunkSpans, version);
        MapFileKind kind = ToMapFileKind(detection.Kind);
        return new MapFileSummary(sourcePath, kind, version, chunks);
    }

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != MapChunkIds.Mver)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }

    private static MapFileKind ToMapFileKind(WowFileKind kind)
    {
        return kind switch
        {
            WowFileKind.Wdt => MapFileKind.Wdt,
            WowFileKind.Adt => MapFileKind.Adt,
            WowFileKind.AdtTex => MapFileKind.AdtTex,
            WowFileKind.AdtObj => MapFileKind.AdtObj,
            WowFileKind.AdtLod => MapFileKind.AdtLod,
            _ => MapFileKind.Unknown,
        };
    }
}