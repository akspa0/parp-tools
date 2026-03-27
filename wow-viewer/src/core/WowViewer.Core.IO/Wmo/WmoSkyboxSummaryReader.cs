using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoSkyboxSummaryReader
{
    public static WmoSkyboxSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoSkyboxSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO skybox summary requires a WMO root file, but found {detection.Kind}.");

        ChunkSpan? mosbChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == FourCC.FromString("MOSB"));
        if (mosbChunk is null || mosbChunk.Value.Header.Id != WmoChunkIds.Mosb)
            throw new InvalidDataException("WMO skybox summary requires a MOSB chunk.");

        byte[] payload = ReadChunkPayload(stream, mosbChunk.Value);
        int length = Array.IndexOf(payload, (byte)0);
        if (length < 0)
            length = payload.Length;

        string skyboxName = System.Text.Encoding.UTF8.GetString(payload, 0, length);
        return new WmoSkyboxSummary(sourcePath, version, payload.Length, skyboxName);
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
