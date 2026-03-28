using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoLightDetailReader
{
    public static IReadOnlyList<WmoLightDetail> Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoLightDetail> Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Molt, out uint? version);
        return WmoLightReaderCommon.ReadDetails(payload, sourcePath, version);
    }
}