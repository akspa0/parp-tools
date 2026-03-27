using WowViewer.Core.Chunks;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoOpaqueChunkSummaryReader
{
    public static WmoOpaqueChunkSummary Read(string path, FourCC chunkId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path), chunkId);
    }

    public static WmoOpaqueChunkSummary Read(Stream stream, string sourcePath, FourCC chunkId)
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, chunkId, out uint? version);
        return new WmoOpaqueChunkSummary(sourcePath, version, chunkId.ToString(), payload.Length);
    }
}
